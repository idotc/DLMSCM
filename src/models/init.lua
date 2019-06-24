--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--

require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'

local M = {}

function M.setup(opt, checkpoint)
  local model
  if checkpoint then
    local modelPath = paths.concat(opt.resume, checkpoint.modelFile)
    assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
    print('=> Resuming model from ' .. modelPath)
    model = torch.load(modelPath):cuda()
  elseif opt.retrain ~= 'none' then
    assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
    print('Loading model from file: ' .. opt.retrain)
    model = torch.load(opt.retrain):cuda()
    model.__memoryOptimized = nil
  elseif opt.loadModel ~= 'none' then
    assert(paths.filep(opt.loadModel), 'File not found: ' .. opt.loadModel)
    print('Loading model from file: ' .. opt.loadModel)
    model = torch.load(opt.loadModel):cuda()
  else
    print('=> Creating model from file: src/models/' .. opt.netType .. '.lua')
    model = require('src.models.' .. opt.netType)(opt)
  end

  -- First remove any DataParallelTable
  if torch.type(model) == 'nn.DataParallelTable' then
    model = model:get(1)
  end


  -- Set the CUDNN flags
  if opt.cudnn == 'fastest' then
    -- cudnn.fastest = true
    cudnn.benchmark = true
  elseif opt.cudnn == 'deterministic' then
    -- Use a deterministic convolution implementation
    model:apply(function(m)
      if m.setMode then m:setMode(1, 1, 1) end
    end)
  end

  -- Wrap the model with DataParallelTable, if using more than one GPU
  if opt.nGPU > 1 then
    local gpus = torch.range(1, opt.nGPU):totable()
    local fastest, benchmark = cudnn.fastest, cudnn.benchmark

    local dpt = nn.DataParallelTable(1, true, true)
        :add(model, gpus)
        :threads(function()
            local cudnn = require 'cudnn'
            local nngraph = require 'nngraph'  
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
        end)
    dpt.gradInput = nil

    model = dpt:cuda()
  end

  local criterion
  if opt.nStack == 1 then
    criterion = nn[opt.crit .. 'Criterion']():cuda()
  else
    criterion = nn.ParallelCriterion()
    for i = 1,opt.nStack do 
      criterion:add(nn[opt.crit .. 'Criterion']():cuda()) 
    end
    criterion:cuda()
  end

  return model, criterion
end

return M
