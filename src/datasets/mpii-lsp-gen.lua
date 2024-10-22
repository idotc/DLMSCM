--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Copyright (c) 2016, YANG Wei
--

local cjson = require 'cjson'
local paths = require 'paths'

local M = {}
-- parse Joints in table to joint [tensor] and isvisible [tensor]
local function TableToTensor(sample)
  -- parse joints
  local numJoints = #sample.joint_self
  local numCoords = 2
  jointsTensor = torch.Tensor(numJoints, numCoords)
  isVisible = torch.IntTensor(numJoints)

  for i = 1, numJoints do
    for j = 1, numCoords do
      jointsTensor[i][j] = sample.joint_self[i][j]
    end
    isVisible[i] = sample.joint_self[i][3]
  end
  sample.joint = jointsTensor
  sample.isVisible = isVisible

  -- parse objpos
  local objpos = torch.Tensor(2)
  objpos[1], objpos[2] = sample.objpos[1], sample.objpos[2]
  sample.objpos = objpos

  return sample
end

local function Serialize(dataset)
  local n = #dataset
  local numJoints, numCoords = dataset[1]['joint']:size(1), dataset[1]['joint']:size(2)
  local labels = {}
  labels['nsamples']=n
  labels['part']=torch.Tensor(n, numJoints, numCoords)
  labels['center']=torch.Tensor(n, numCoords)
  labels['visible']=torch.Tensor(n, numJoints)
  labels['scale']=torch.Tensor(n)

  local data, toIdxs = {}, {}
  data['images'] = {}
  data['dataset'] = {}

  for i = 1,n do
    labels['part'][i] = dataset[i]['joint']
    labels['center'][i] = dataset[i]['objpos']
    labels['visible'][i] = dataset[i]['isVisible']
    labels['scale'][i] = dataset[i]['scale_provided']
    local img_paths = dataset[i]['img_paths']
    data['images'][i] = img_paths
    data['dataset'][i] = dataset[i]['dataset']
    if not toIdxs[img_paths] then toIdxs[img_paths] = {} end
    table.insert(toIdxs[img_paths], i)
  end

  -- This allows us to reference multiple people who are in the same image
  data['imageToIdxs'] = toIdxs
  return data, labels
end

-- Split dataset into train/val from JSON
local function readJson(filename)
  local file = io.open(filename, 'r')
  local text = file:read()
  file:close()

  local annot = cjson.decode(text)

  -- split train val
  local dataTimer = torch.Timer()
  local trainSet, testSet = {}, {}
  for _, sample in ipairs(annot) do
    -- parse joints
    sample = TableToTensor(sample)

    -- remove empty field
    if sample.numOtherPeople == 0 then
      sample.objpos_other = nil
      sample.joint_others = nil
      sample.scale_provided_other = nil
    end

    -- split train/val
    if sample['isValidation'] == 0 then
      table.insert(trainSet, sample)
    else
      table.insert(testSet, sample)
    end
  end


  print(('    Training data: %d | Validation data: %d | Processing time: %.2fs')
  :format(#trainSet, #testSet, dataTimer:time().real))
  return trainSet, testSet
end

local function computeMean( train )
  print('==> Compute image mean and std')
  local meanstd
  local data, labels = train.data, train.labels
  if not paths.dirp('gen/mpii-lsp') then
    paths.mkdir('gen/mpii-lsp')
  end
  if not paths.filep('gen/mpii-lsp/meanstd.t7') then
    local size_ = labels['nsamples']
    local rgbs = torch.Tensor(3, size_)

    for idx = 1, size_ do
      xlua.progress(idx, size_)
      local dataset = data['dataset'][idx]
      local imgpath = data['images'][idx]
      if dataset == 'LEEDS' then
        imgpath = paths.concat('data/lsp', imgpath)
      else
        imgpath = paths.concat('data/mpii/images', imgpath)
      end
      local imdata = image.load(imgpath)
      rgbs:select(2, idx):copy(imdata:view(3, -1):mean(2))
    end

    local mean = rgbs:mean(2):squeeze()
    local std = rgbs:std(2):squeeze()

    meanstd = {
      mean = mean,
      std = std
    }

    torch.save('gen/mpii-lsp/meanstd.t7', meanstd)
  else
    meanstd = torch.load('gen/mpii-lsp/meanstd.t7')
  end

  print(('    mean: %.4f %.4f %.4f'):format(meanstd.mean[1], meanstd.mean[2], 
      meanstd.mean[3]))
  print(('     std: %.4f %.4f %.4f'):format(meanstd.std[1], meanstd.std[2], 
      meanstd.std[3]))
  return meanstd
end

local function  concatTable(table1, table2)
  local combTable = {}
  for _, v in pairs(table1) do
    table.insert(combTable, v)
  end
  for _, v in pairs(table2) do
    table.insert(combTable, v)
  end
  return combTable
end

function M.exec(opt, cacheFile)
  print('==> Create mpii + LSP fusion dataset')
  local mpiiTrain, mpiiTest = readJson('data/mpii/mpii_annotations.json')
  local lspTrain, lspTest = readJson('data/lsp/LEEDS_annotations_corrected.json')

  local trainSet = concatTable(lspTrain, mpiiTrain)
  local valSet = lspTest
  -- Simplified for serialization
  trainData, trainLabels = Serialize(trainSet)
  valData, valLabels = Serialize(valSet)

  local train, val = {data=trainData, labels=trainLabels}, {data=valData, labels=valLabels}

  -- LSP test
  testData, testLabels = Serialize(lspTest)
  local test = {data=testData, labels=testLabels}

  print(" | saving MPII + LSP dataset to " .. cacheFile)
  computeMean(train)
  torch.save(cacheFile, {
    train = train,
    val = test,
    test = test,
  })
end

return M
