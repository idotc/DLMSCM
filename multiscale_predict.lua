--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--

require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'xlua'
local DataLoader = require 'src.dataloader-multiscale'
local models = require 'src.models.init'
local Trainer = require 'src.train'
local opts = require 'src.opts'
local checkpoints = require 'src.checkpoints'
local Logger = require 'src.utils.Logger'

local scales = torch.range(0.8, 1.3, 0.1):totable() 

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

-- Data loading
local trainLoader, valLoader, testLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)

if opt.testRelease then
  print('=> Test Release')
  local testAcc, testLoss = trainer:multiScaleTest(opt.epochNumber, 
      testLoader, scales)
  print(string.format(' * Results acc: %6.3f, loss: %6.3f', testAcc, testLoss))
  return
end

if opt.testOnly then
  print('=> Test Only')
  local testAcc, testLoss = trainer:multiScaleTest(opt.epochNumber, valLoader, 
      scales)
  print(string.format(' * Results acc: %6.3f, loss: %6.3f', testAcc, testLoss))
  return
end
