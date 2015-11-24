require 'dp'
require 'torchx' -- for paths.indexdir

function footballplaydata(dataPath, validRatio)
   validRatio = validRatio or 0.15
   videos = {}

   local reducedImageWidth = 120
   local reducedImageHeight = 68


   -- 1. load images into input and target Tensors
   local video1 = paths.indexdir(paths.concat(dataPath, 'videos')) -- 1
   print('Got Images')
   local targetVid = paths.indexdir(paths.concat(dataPath, 'target')) 
   print('Got Target')
   local size = video1:size() + targetVid:size()
   local shuffle = torch.randperm(size) -- shuffle the data
   local input = torch.FloatTensor(size, 1, reducedImageWidth, reducedImageHeight)
   local target = torch.FloatTensor(size, 1, reducedImageWidth, reducedImageHeight)

   for i=1,video1:size() do
      print('Iteration: '..i)
      local img = image.load(video1:filename(i))
      local imgray = image.rgb2y(img)      
      imgray = image.scale(imgray, reducedImageWidth, reducedImageHeight)

      local target_img = image.load(targetVid:filename(1))
      local target_imgray = image.rgb2y(target_img)      
      target_imgray = image.scale(imgray, reducedImageWidth, reducedImageHeight)

      print(imgray:size())
      local idx = shuffle[i]
      input[idx]:copy(imgray)
      target[idx]:copy(target_imgray)
      collectgarbage()
   end

   -- 2. divide into train and valid set and wrap into dp.Views

   local nValid = math.floor(size*validRatio)
   local nTrain = size - nValid

   local trainInput = dp.ImageView('bchw', input:narrow(1, 1, nTrain))
   local trainTarget = dp.ImageView('bchw', target:narrow(1, 1, nTrain))
   local validInput = dp.ImageView('bchw', input:narrow(1, nTrain+1, nValid))
   local validTarget = dp.ImageView('bchw', target:narrow(1, nTrain+1, nValid))

   -- 3. wrap views into datasets

   local train = dp.DataSet{inputs=trainInput,targets=trainTarget,which_set='train'}
   local valid = dp.DataSet{inputs=validInput,targets=validTarget,which_set='valid'}

   -- 4. wrap datasets into datasource

   local ds = dp.DataSource{train_set=train,valid_set=valid}
   --ds:classes{'image', 'image2'}
   return ds
end