require 'dp'
require 'torchx' -- for paths.indexdir

function footballplaydata(dataPath, validRatio)
   validRatio = validRatio or 0.15
   videos = {}

   local reducedImageWidth = 120
   local reducedImageHeight = 68


   -- 1. load images into input and target Tensors

   local video1 = paths.indexdir(paths.concat(dataPath, 'video_frames')) -- 1
   print('Got Images')
   local targetVid = paths.indexdir(paths.concat(dataPath, 'video_frames')) 
   print('Got Target')
   --local size = video1:size() + targetVid:size()
   local size = 100
   local shuffle = torch.randperm(size) -- shuffle the data
   local input = torch.FloatTensor(size, 1, reducedImageHeight, reducedImageWidth)
   local target = torch.FloatTensor(size, 1, reducedImageHeight, reducedImageWidth)
   
   local my_index = 1
   
   local video_filenames = {}
   local target_filenames = {}
   
   for i = 1,10 do
   	for j = 1,10 do
   		table.insert(video_filenames, dataPath .. 'video_frames/' .. i .. '_' .. j .. '.png')
   		table.insert(target_filenames, dataPath .. 'video_frames/' .. i .. '_' .. j + 1 .. '.png')
   	end
   end
   		

   for i=1,size do
      print('Iteration: '..i)
      print(video_filenames[i])
      print(target_filenames[i])
      local img = image.load(video_filenames[i])
      --local img = image.load(video1:filename(i))
      local imgray = image.rgb2y(img)      
      imgray = image.scale(imgray, reducedImageWidth, reducedImageHeight)

      local target_img = image.load(target_filenames[i])
      local target_imgray = image.rgb2y(target_img)      
      target_imgray = image.scale(imgray, reducedImageWidth, reducedImageHeight)

      --print(imgray[1]:size())
      --local idx = shuffle[i]
      input[i]:copy(imgray)
      target[i]:copy(target_imgray)
      collectgarbage()
   end

   -- 2. divide into train and valid set and wrap into dp.Views

   local nValid = math.floor(size*validRatio)
   local nTrain = size - nValid

input:narrow(1, 1, nTrain)
print(input:size())

print(input:narrow(1, 1, nTrain):size())
   
   local trainInput = dp.ImageView('bchw', input:narrow(1, 1, nTrain))
   local trainTarget = dp.ImageView('bchw', target:narrow(1, 1, nTrain))
   local validInput = dp.ImageView('bchw', input:narrow(1, nTrain+1, nValid))
   local validTarget = dp.ImageView('bchw', target:narrow(1, nTrain+1, nValid))

--[[
   local trainInput = dp.SequenceView('bwc', input:narrow(1, 1, nTrain))
   local trainTarget = dp.SequenceView('bwc', target:narrow(1, 1, nTrain))
   local validInput = dp.SequenceView('bwc', input:narrow(1, nTrain+1, nValid))
   local validTarget = dp.SequenceView('bwc', target:narrow(1, nTrain+1, nValid))
]]--   
   -- 3. wrap views into datasets

   local train = dp.DataSet{inputs=trainInput,targets=trainTarget,which_set='train'}
   local valid = dp.DataSet{inputs=validInput,targets=validTarget,which_set='valid'}

   -- 4. wrap datasets into datasource

   local ds = dp.DataSource{train_set=train,valid_set=valid}
   --ds:classes{'image', 'image2'}
   return ds
end

function FootballImageData (dataPath)

validRatio = validRatio or 0.15
   videos = {}

   local reducedImageWidth = 120
   local reducedImageHeight = 68


   -- 1. load images into input and target Tensors

   local video1 = paths.indexdir(paths.concat(dataPath, 'video_frames')) -- 1
   print('Got Images')
   local targetVid = paths.indexdir(paths.concat(dataPath, 'video_frames')) 
   print('Got Target')
   --local size = video1:size() + targetVid:size()
   local size = 100
   local shuffle = torch.randperm(size) -- shuffle the data
   local input = torch.FloatTensor(size, 1, reducedImageHeight, reducedImageWidth - 1)
   local target = torch.FloatTensor(size, 1, reducedImageHeight, reducedImageWidth - 1)
   
   local my_index = 1
   
   local video_filenames = {}
   local target_filenames = {}
   
   for i = 1,10 do
   	for j = 1,10 do
   		table.insert(video_filenames, dataPath .. 'video_frames/' .. i .. '_' .. j .. '.png')
   		table.insert(target_filenames, dataPath .. 'video_frames/' .. i .. '_' .. j + 1 .. '.png')
   	end
   end
   		

   for i=1,size do
      print('Iteration: '..i)
      print(video_filenames[i])
      print(target_filenames[i])
      local img = image.load(video_filenames[i])
      --local img = image.load(video1:filename(i))
      local imgray = image.rgb2y(img)      
      imgray = image.scale(imgray, reducedImageWidth, reducedImageHeight)

      local target_img = image.load(target_filenames[i])
      local target_imgray = image.rgb2y(target_img)      
      target_imgray = image.scale(imgray, reducedImageWidth, reducedImageHeight)

      --print(imgray[1]:size())
      --local idx = shuffle[i]
      input[i]:copy(imgray:narrow(3, 1, reducedImageWidth - 1))
      target[i]:copy(target_imgray:narrow(3, 2, reducedImageWidth - 1))
      collectgarbage()
   end

   -- 2. divide into train and valid set and wrap into dp.Views

   local nValid = math.floor(size*validRatio)
   local nTrain = size - nValid

input:narrow(1, 1, nTrain)
print(input:size())

print(input:narrow(1, 1, nTrain):size())
   
   local trainInput = dp.ImageView('bchw', input:narrow(1, 1, nTrain))
   local trainTarget = dp.ImageView('bchw', target:narrow(1, 1, nTrain))
   local validInput = dp.ImageView('bchw', input:narrow(1, nTrain+1, nValid))
   local validTarget = dp.ImageView('bchw', target:narrow(1, nTrain+1, nValid))

--[[
   local trainInput = dp.SequenceView('bwc', input:narrow(1, 1, nTrain))
   local trainTarget = dp.SequenceView('bwc', target:narrow(1, 1, nTrain))
   local validInput = dp.SequenceView('bwc', input:narrow(1, nTrain+1, nValid))
   local validTarget = dp.SequenceView('bwc', target:narrow(1, nTrain+1, nValid))
]]--   
   -- 3. wrap views into datasets

   local train = dp.DataSet{inputs=trainInput,targets=trainTarget,which_set='train'}
   local valid = dp.DataSet{inputs=validInput,targets=validTarget,which_set='valid'}

   -- 4. wrap datasets into datasource

   local ds = dp.DataSource{train_set=train,valid_set=valid}
   --ds:classes{'image', 'image2'}
   return ds
end
