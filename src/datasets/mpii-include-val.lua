--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--

torch.setdefaulttensortype('torch.FloatTensor')

local image = require 'image'
local paths = require 'paths'
local t = require 'src.datasets.posetransforms'

-------------------------------------------------------------------------------
-- Helper Functions
-------------------------------------------------------------------------------
local drawGaussian = t.drawGaussian
local shuffleLR = t.shuffleLR
local flip = t.flip
local colorNormalize = t.colorNormalize

local drawLimbMap = t.drawLimbMap
local composeMaps = t.composeMaps
local shuffleLR = t.shuffleLR

local affineTransform = t.affineTransform
local affineCrop = t.affineCrop

-------------------------------------------------------------------------------
-- Create dataset Class
-------------------------------------------------------------------------------
local M = {}
local MpiiDataset = torch.class('resnet.MpiiDataset', M)

function MpiiDataset:__init(imageInfo, opt, split)
  assert(imageInfo[split], split)
  self.imageInfo = imageInfo[split]
  self.split = split
  -- Some arguments
  self.inputRes = opt.inputRes
  self.outputRes = opt.outputRes
  -- Options for augmentation
  self.scaleFactor = opt.scaleFactor
  self.rotFactor = opt.rotFactor
  self.shearFactor = opt.shearFactor
  self.dataset = opt.dataset
  self.nStack = opt.nStack
  self.meanstd = torch.load('gen/mpii-include-val/meanstd.t7')
  self.nGPU = opt.nGPU
  self.batchSize = opt.batchSize
  self.minusMean = opt.minusMean
  self.gsize = opt.gsize
  self.rotProbab = opt.rotProbab
  self.shearProbab = opt.shearProbab
  self.nSemanticLevels = opt.nSemanticLevels
  self.struct = opt.struct
  self.catPartEnds = opt.catPartEnds
end

function MpiiDataset:get(i, scaleFactor)
  local scaleFactor = scaleFactor or 1
  local img = image.load(paths.concat('data/mpii/images', 
      self.imageInfo.data['images'][i]))

  -- Generate samples
  local points = self.imageInfo.labels['part'][i]:float()
  local center = self.imageInfo.labels['center'][i]:float()
  local scale = self.imageInfo.labels['scale'][i]*scaleFactor
  -- (x,y)
  scale = torch.Tensor(2):fill(scale)
  local rot = 0
  -- (x,y)
  local shear = torch.zeros(2)

  -- Augmentation stage 1
  -- Determine the affine transform before preparing image and points
  if self.split == 'train' then
    -- (x,y)
    local scaleAug = torch.randn(2):mul(self.scaleFactor):add(1):
        clamp(1-self.scaleFactor,1+self.scaleFactor)
    -- 1D
    local rotAug = torch.randn(1):mul(self.rotFactor):
        clamp(-2*self.rotFactor,2*self.rotFactor)[1]
    -- (x,y)
    local shearAug = torch.randn(2):mul(self.shearFactor):
        clamp(-self.shearFactor,self.shearFactor)

    -- Prob. of scale augmentation is 1
    scale:cmul(scaleAug)
    -- rotation
    if torch.uniform() <= self.rotProbab then rot = rot + rotAug end
    -- shearing
    if torch.uniform() <= self.shearProbab then shear:add(shearAug) end
  end

  -- Crop the input image
  local inp = affineCrop(img, center, scale, rot, shear, self.inputRes)

  -- Transform the points
  local newPoints = torch.zeros(points:size()) -- (16, 2)
  for i = 1,newPoints:size(1) do
    if points[i][1] > 0 then
      newPoints[i] = affineTransform(points[i]+1, center, scale, rot, shear,
          self.outputRes)
    end
  end

  -- Generate the gt score maps for each level
  local level2gt = {}
  -- Whether a score map is empty
  local level2isEmpty = {}
  local nParts = self.struct.nParts
  local children = self.struct.children
  local nSemanticLevels = self.nSemanticLevels
  -- level 1 parts
  level2gt[1] = torch.zeros(nParts[1], self.outputRes, self.outputRes)
  level2isEmpty[1] = {}
  for i = 1,nParts[1] do
    local j = newPoints[i]
    if j[1] > 0 and j[1] <= self.outputRes and j[2] > 0 and 
      j[2] <= self.outputRes then
      drawGaussian(level2gt[1][i], j, self.gsize)
      table.insert(level2isEmpty[1], false)
    else -- If there is no annotation
      table.insert(level2isEmpty[1], true)
    end
  end
  -- level 2 parts
  if nSemanticLevels >= 2 then
    level2gt[2] = torch.zeros(nParts[2], self.outputRes, self.outputRes)
    level2isEmpty[2] = {}
    for i = 1,nParts[2] do 
      local idx1, idx2 = unpack(children[2][i])
      if not (level2isEmpty[1][idx1] or level2isEmpty[1][idx2]) then
        local j1, j2 = newPoints[idx1], newPoints[idx2]
        drawLimbMap(level2gt[2][i], j1, j2, self.gsize)
        table.insert(level2isEmpty[2], false)
      else
        table.insert(level2isEmpty[2], true)
      end
    end
  end
  -- level >=3 parts
  for l = 3,nSemanticLevels do
    level2gt[l] = torch.zeros(nParts[l], self.outputRes, self.outputRes)
    level2isEmpty[l] = {}
    for i = 1,nParts[l] do
      local idx_set = children[l][i]
      local isempty = true
      for _, v in ipairs(idx_set) do
        if not level2isEmpty[l-1][v] then
          isempty = false
          break
        end
      end
      if not isempty then
        composeMaps(level2gt[l][i], level2gt[l-1], idx_set)
        table.insert(level2isEmpty[l], false)
      else
        table.insert(level2isEmpty[l], true)
      end
    end
  end

  -- Augmentation stage 2
  -- Since we have done affine transform, the remaining augmentation is 
  -- filpping and color jittering
  inp, level2gt, level2isEmpty = self.augmentation(self, inp, level2gt, 
      level2isEmpty)

  local level2empty = {}
  for l = 1,#level2isEmpty do
    level2empty[l] = {}
    for i = 1,#level2isEmpty[l] do
      if level2isEmpty[l][i] then
        table.insert(level2empty[l], i)
      end
    end
  end

  if self.split ~= 'train' then
    scale = scale[1]
  end

  -- Cat the unary heatmaps to higher-level heatmaps
  if self.catPartEnds and nSemanticLevels > 1 then
    for l = 2,nSemanticLevels do
      level2gt[l] = torch.cat(level2gt[l], level2gt[1], 1)
      for _,v in ipairs(level2empty[1]) do
        table.insert(level2empty[l], v+nParts[l])
      end
    end
  end

  collectgarbage()
  return {
    input = inp,
    target = level2gt,
    center = center,
    scale = scale,
    width = img:size(3),
    height = img:size(2),
    imgPath = paths.concat('data/mpii/images', 
        self.imageInfo.data['images'][i]),
    level2empty = level2empty
  }
end

function MpiiDataset:size()
  if self.split == 'test' then
    return self.imageInfo.labels.nsamples
  end

  local nSamples = self.imageInfo.labels.nsamples - 
      (self.imageInfo.labels.nsamples%self.nGPU)
  nSamples = nSamples - nSamples%self.batchSize

  return nSamples
end

function MpiiDataset:preprocess()
  return function(img)
    if img:max() > 2 then
      img:div(255)
    end
    return self.minusMean == 'true' and colorNormalize(img, self.meanstd) or img
  end
end

function MpiiDataset:augmentation(input, level2gt, level2isEmpty)
  -- Augment data (during training only)
  if self.split == 'train' then
    -- Color
    input[{1, {}, {}}]:mul(torch.uniform(0.8, 1.2)):clamp(0, 1)
    input[{2, {}, {}}]:mul(torch.uniform(0.8, 1.2)):clamp(0, 1)
    input[{3, {}, {}}]:mul(torch.uniform(0.8, 1.2)):clamp(0, 1)

    -- Flip
    if torch.uniform() <= .5 then
      input = flip(input)
      level2gt = shuffleLR(level2gt, self.struct.partners, 
      torch.range(1,#level2gt):totable())
      for l=1,#level2gt do
        level2gt[l] = flip(level2gt[l])
        -- shuffle empty status
        for i=1,#self.struct.partners[l] do
          local idx1, idx2 = unpack(self.struct.partners[l][i])
          level2isEmpty[l][idx1], level2isEmpty[l][idx2] = 
              level2isEmpty[l][idx2], level2isEmpty[l][idx1] 
        end
      end
    end
  end

  return input, level2gt, level2isEmpty
end

return M.MpiiDataset
