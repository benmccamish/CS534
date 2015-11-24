require 'dp'
require 'rnn'
require 'unsup'
require 'image'
require 'optim'
require 'autoencoder-data'
require 'fooballdataLSTM'

version = 1

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Language Model on PennTreeBank dataset using RNN or LSTM')
cmd:text('Example:')
cmd:text("recurrent-language-model.lua --cuda --useDevice 2 --progress --zeroFirst --cutoffNorm 4 --rho 10")
cmd:text('Options:')
cmd:option('--learningRate', 0.05, 'learning rate at t=0')
cmd:option('--minLR', 0.00001, 'minimum learning rate')
cmd:option('--saturateEpoch', 400, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--maxOutNorm', -1, 'max l2-norm of each layer\'s output neuron weights')
cmd:option('--cutoffNorm', -1, 'max l2-norm of concatenation of all gradParam tensors')
cmd:option('--batchSize', 100, 'number of examples per batch')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 1000, 'maximum number of epochs to run')
cmd:option('--maxTries', 50, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'don\'t print anything to stdout')
cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')

-- recurrent layer 
cmd:option('--lstm', true, 'use Long Short Term Memory (nn.LSTM instead of nn.Recurrent)')
cmd:option('--rho', 5, 'back-propagate through time (BPTT) for rho time-steps')
cmd:option('--hiddenSize', '{200}', 'number of hidden units used at output of each recurrent layer. When more than one is specified, RNN/LSTMs are stacked')
cmd:option('--zeroFirst', false, 'first step will forward zero through recurrence (i.e. add bias of recurrence). As opposed to learning bias specifically for first step.')
cmd:option('--dropout', false, 'apply dropout after each recurrent layer')
cmd:option('--dropoutProb', 0.5, 'probability of zeroing a neuron (dropout probability)')

-- data
cmd:option('--trainEpochSize', -1, 'number of train examples seen between each epoch')
cmd:option('--validEpochSize', -1, 'number of valid examples used for early stopping and cross-validation') 

cmd:text()
opt = cmd:parse(arg or {})
opt.hiddenSize = dp.returnString(opt.hiddenSize)
if not opt.silent then
   table.print(opt)
end

--[[Data]]--
ds = footballplaydata('/Users/benmccamish/LSTMVideos/Training/', 0.9)
--ds:validSet():contextSize(opt.batchSize)
--ds:testSet():contextSize(opt.batchSize)
print("Got data!")

--ds = dp.PennTreeBank{
--   context_size=opt.rho, 
--   recurrent=true
--}
--ds:validSet():contextSize(opt.batchSize)
--ds:testSet():contextSize(opt.batchSize)

--[[Model]]--

-- language model
lm = nn.Sequential()

local inputSize = opt.hiddenSize[1]
for i,hiddenSize in ipairs(opt.hiddenSize) do 

   if i~= 1 and not opt.lstm then
      lm:add(nn.Sequencer(nn.Linear(inputSize, hiddenSize)))
   end
   
   -- recurrent layer
   local rnn
   if opt.lstm then
      -- Long Short Term Memory
      rnn = nn.Sequencer(nn.FastLSTM(inputSize, hiddenSize))
   else
      -- simple recurrent neural network
      rnn = nn.Recurrent(
         hiddenSize, -- first step will use nn.Add
         nn.Identity(), -- for efficiency (see above input layer) 
         nn.Linear(hiddenSize, hiddenSize), -- feedback layer (recurrence)
         nn.Sigmoid(), -- transfer function 
         99999 -- maximum number of time-steps per sequence
      )
      if opt.zeroFirst then
         -- this is equivalent to forwarding a zero vector through the feedback layer
         rnn.startModule:share(rnn.feedbackModule, 'bias')
      end
      rnn = nn.Sequencer(rnn)
   end

   lm:add(rnn)
   
   if opt.dropout then -- dropout it applied between recurrent layers
      lm:add(nn.Sequencer(nn.Dropout(opt.dropoutProb)))
   end
   
   inputSize = hiddenSize
end

-- input layer (i.e. word embedding space)
lm:insert(nn.SplitTable(1,2), 1) -- tensor to table of tensors

if opt.dropout then
   lm:insert(nn.Dropout(opt.dropoutProb), 1)
end

lookup = nn.LookupTable(ds:featureSize(), opt.hiddenSize[1])
lookup.maxOutNorm = -1 -- disable maxParamNorm on the lookup table
lm:insert(lookup, 1)

-- output layer
lm:add(nn.Sequencer(nn.Linear(inputSize, ds:featureSize())))
--lm:add(nn.Sequencer(nn.LogSoftMax()))

if opt.uniform > 0 then
   for k,param in ipairs(lm:parameters()) do
      param:uniform(-opt.uniform, opt.uniform)
   end
end

-- will recurse a single continuous sequence
lm:remember(opt.lstm and 'both' or 'eval')   

--[[Propagators]]--

-- linear decay
opt.decayFactor = (opt.minLR - opt.learningRate)/opt.saturateEpoch

train = dp.Optimizer{
   loss = nn.MSECriterion(),
   epoch_callback = function(model, report) -- called every epoch
      if report.epoch > 0 then
         opt.learningRate = opt.learningRate + opt.decayFactor
         opt.learningRate = math.max(opt.minLR, opt.learningRate)
         if not opt.silent then
            print("learningRate", opt.learningRate)
            if opt.meanNorm then
               print("mean gradParam norm", opt.meanNorm)
            end
         end
      end
   end,
   callback = function(model, report) -- called every batch
      if opt.cutoffNorm > 0 then
         local norm = model:gradParamClip(opt.cutoffNorm) -- affects gradParams
         opt.meanNorm = opt.meanNorm and (opt.meanNorm*0.9 + norm*0.1) or norm
      end
      model:updateGradParameters(opt.momentum) -- affects gradParams
      model:updateParameters(opt.learningRate) -- affects params
      model:maxParamNorm(opt.maxOutNorm) -- affects params
      model:zeroGradParameters() -- affects gradParams 
   end,
   feedback = dp.Perplexity(),  
   sampler = dp.ShuffleSampler{epoch_size = opt.trainEpochSize, batch_size = opt.batchSize}, 
   progress = opt.progress
}

valid = dp.Evaluator{
   feedback = dp.Perplexity(),  
   sampler = dp.Sampler{epoch_size = opt.validEpochSize, batch_size = 1},
   progress = opt.progress
}

tester = dp.Evaluator{
   feedback = dp.Perplexity(),  
   sampler = dp.Sampler{batch_size = 1} 
}

--[[Experiment]]--

xp = dp.Experiment{
   model = lm,
   optimizer = train,
   validator = valid,
   tester = tester,
   observer = {
      dp.FileLogger(),
      dp.EarlyStopper{
         max_epochs = opt.maxTries, 
         error_report={'validator','feedback','perplexity','ppl'}
      }
   },
   random_seed = os.time(),
   max_epoch = opt.maxEpoch,
   target_module = nn.SplitTable(1,1):type('torch.IntTensor')
}

--[[GPU or CPU]]--

if opt.cuda then
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.useDevice)
   xp:cuda()
end

xp:verbose(not opt.silent)
if not opt.silent then
   print"Language Model :"
   print(lm)
end

xp:run(ds)