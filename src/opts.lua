--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--

local M = {}

function M.parse(arg)
  local cmd = torch.CmdLine()
  cmd:text('Options:')
  ------------ General options --------------------
  cmd:option('-data',         '',         'Path to dataset')
  cmd:option('-dataset',      'mpii-lsp',     'Options: mpii | mpii-lsp')
  cmd:option('-manualSeed',   0,          'Manually set RNG seed')
  cmd:option('-nGPU',         1,          'Number of GPUs to use by default')
  cmd:option('-backend',      'cudnn',    'Options: cudnn | cunn')
  cmd:option('-cudnn',        'fastest',  'Options: fastest | default | ' ..
      'deterministic')
  cmd:option('-gen',          'gen',      'Path to save generated files')
  ------------- Data options ------------------------
  cmd:option('-nThreads',     2,          'number of data loading threads')
  cmd:option('-inputRes',     256,        'Input image resolution')
  cmd:option('-outputRes',    64,         'Output heatmap resolution')
  cmd:option('-scaleFactor',  .25,        'Degree of scale augmentation')
  cmd:option('-rotFactor',    30,         'Degree of rotation augmentation')
  cmd:option('-rotProbab',    .4,         'Degree of rotation augmentation')
  cmd:option('-shearFactor',  .5,         'Degree of shearing augmentation')
  cmd:option('-shearProbab',  .5,         'Degree of shearing augmentation')
  cmd:option('-flipFactor',   .5,         'Degree of flip augmentation')
  cmd:option('-minusMean',    'true',     'Minus image mean')
  cmd:option('-gsize',        1,          'Kernel size to generate the ' ..
      'Gassian-like labelmap')
  cmd:option('-ignoreEmptyLabel','false',    'Whether we should not penalize' ..
      ' the empty labelmap')
  ------------- Training options --------------------
  cmd:option('-nEpochs',      0,          'Number of total epochs to run')
  cmd:option('-epochNumber',  1,          'Manual epoch number (useful on ' ..
      'restarts)')
  cmd:option('-batchSize',    4,          'mini-batch size (1 = pure '..
      'stochastic)')
  cmd:option('-testOnly',     'false',    'Run on validation set only')
  cmd:option('-testRelease',  'false',    'Run on testing set only')
  cmd:option('-crit',         'MSE',      'Criterion type: MSE | CrossEntropy')
  cmd:option('-optMethod',    'rmsprop',  'Optimization method: rmsprop | ' ..
      'sgd | nag | adadelta | adam')
  cmd:option('-snapshot',     1,          'How often to take a snapshot of ' ..
      'the model (0 = never)')
  ------------- Checkpointing options ---------------
  cmd:option('-save',         'checkpoints','Directory in which to save ' ..
      'checkpoints')
  cmd:option('-expID',        'default',  'Experiment ID')
  cmd:option('-resume',       'none',     'Resume from the latest ' ..
      'checkpoint in this directory')
  cmd:option('-loadModel',    'none',     'Load model')
  ---------- Optimization options ----------------------
  cmd:option('-LR',           2.5e-4,     'initial learning rate')
  cmd:option('-momentum',     0.0,        'momentum')
  cmd:option('-weightDecay',  0.0,        'weight decay')
  cmd:option('-alpha',        0.99,       'Alpha')
  cmd:option('-epsilon',      1e-8,       'Epsilon')
  cmd:option('-dropout',      0,          'Dropout ratio')
  cmd:option('-init',         'none',     'Weight initialization method: ' ..
      'none | xavier | kaiming')
  cmd:option('-schedule',     '150 180 200', 'schedule to decay learning rate')
  cmd:option('-gamma',        0.1,        'LR is multiplied by gamma on ' ..
      'schedule.')
  ---------- Model options ----------------------------------
  cmd:option('-netType',      'dlmscm',     'Options: dlcm')
  cmd:option('-retrain',      'none',     'Path to model to retrain with')
  cmd:option('-optimState',   'none',     'Path to an optimState to reload from')
  cmd:option('-nFeats',       256,        'Number of features in the ' ..
      'hourglass (for hg-generic)')
  cmd:option('-nResidual',    1,          'Number of residual module in the ' ..
      'hourglass (for hg-generic)')
  cmd:option('-baseWidth',    6,          'PRM: base width', 'number')
  cmd:option('-cardinality',  30,         'PRM: cardinality', 'number')
  ---------- Model options ----------------------------------
  cmd:option('-struct',       '3levels_16joints', 'Compositional structure')
  cmd:option('-nSemanticLevels', 3,       'Number of semantic levels')
  cmd:option('-catPartEnds',   'false',    'Cat joint heatmaps with ' ..
      'higher-level maps')
  cmd:text()

  local opt = cmd:parse(arg or {})

  opt.testOnly = opt.testOnly ~= 'false'
  opt.testRelease = opt.testRelease ~= 'false'
  opt.ignoreEmptyLabel = opt.ignoreEmptyLabel ~= 'false'
  opt.catPartEnds = opt.catPartEnds ~= 'false'

  if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
    cmd:error('error: unable to create checkpoint directory: ' .. opt.save ..
        '\n')
  end

  opt.nEpochs = opt.nEpochs == 0 and 200 or opt.nEpochs

  -- Parse schedule
  local schedule = {}
  for x in string.gmatch(opt.schedule, "%S+") do
    table.insert(schedule, tonumber(x))
  end
  opt.schedule = schedule

  -- Load body structure
  opt.struct = require('src.structures.' .. opt.struct)
  assert(opt.nSemanticLevels <= #opt.struct.children)

  -- Infer the semantic level of each hg module output
  local outLevels = {}
  for l=1,opt.nSemanticLevels do
    table.insert(outLevels, l)
  end
  for l=opt.nSemanticLevels-1,1,-1 do
    table.insert(outLevels, l)
  end
  opt.outLevels = outLevels
  opt.nStack = #opt.outLevels

  return opt
end

return M
