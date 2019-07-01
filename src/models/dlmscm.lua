local nn = require 'nn'
local cunn = require 'cunn'
local cudnn = require 'cudnn'
local nngraph = require 'nngraph'
local nnlib = cudnn
local Residual = require('src.models.layers.Residual')
local MSResidua = require('src.models.layers.MSRM')

local function hourglass(n, f, nModules,inp)
    -- Upper branch
    local up1 = inp
    for i = 1,nModules do up1 = Residual(f,f)(up1) end

    -- Lower branch
    local low1 = nnlib.SpatialMaxPooling(2,2,2,2)(inp)
    for i = 1,nModules do low1 = Residual(f,f)(low1) end
    local low2

    if n > 1 then low2 = hourglass(n-1,f,nModules,low1)
    else
        low2 = low1
        for i = 1,nModules do low2 = Residual(f,f)(low2) end
    end

    local low3 = low2
    for i = 1,nModules do low3 = Residual(f,f)(low3) end
    local up2 = nn.SpatialUpSamplingNearest(2)(low3)

    -- Bring two branches together
    return nn.CAddTable()({up1,up2})
end

local function hourglass_ms(n, f, nModules, inp, inputRes, type, B, C)
    local ResidualUp = n >= 2 and MSResidua or Residual
    local ResidualDown = n >= 3 and MSResidua or Residual

    -- Upper branch
    local up1 = inp
    for i = 1,nModules do up1 = ResidualUp(f,f,1,type,false,inputRes,B,C)(up1) end

    -- Lower branch
    local low1 = nnlib.SpatialMaxPooling(2,2,2,2)(inp)
    for i = 1,nModules do low1 = ResidualDown(f,f,1,type,false, inputRes/2,B,C)(low1) end
    local low2

    if n > 1 then low2 = hourglass_ms(n-1,f,nModules,low1,inputRes/2,type,B,C)
    else
        low2 = low1
        for i = 1,nModules do low2 = ResidualDown(f,f,1,type,false,inputRes/2,B,C)(low2) end
    end

    local low3 = low2
    for i = 1,nModules do low3 = ResidualDown(f,f,1,type,true,inputRes/2,B,C)(low3) end
    local up2 = nn.SpatialUpSamplingNearest(2)(low3)

    -- Bring two branches together
    return nn.CAddTable()({up1,up2})
end

local function lin(numIn,numOut,inp)
  -- Apply 1x1 convolution, stride 1, no padding
  local l = nnlib.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)
  return nnlib.ReLU(true)(nn.SpatialBatchNormalization(numOut)(l))
end

function createModel(opt)
  -- load compositional structure of human bodies

  local B, C = opt.baseWidth, opt.cardinality
  local inputRes = opt.inputRes/4

  local struct = opt.struct
  local nSemanticLevels = opt.nSemanticLevels
  local nParts = struct.nParts
  assert(nSemanticLevels <= #nParts)

  local nStack = opt.nStack

  local nResidual = opt.nResidual -- =1
  local inp = nn.Identity()()

  -- Initial processing of the image
  local cnv1_ = nnlib.SpatialConvolution(3,64,3,3,1,1,1,1)(inp)           -- 256
  local cnv1 = nnlib.ReLU(true)(nn.SpatialBatchNormalization(64)(cnv1_))
  local pool1 = nnlib.SpatialMaxPooling(2,2,2,2)(cnv1)                    -- 128
  local cnv2_ = nnlib.SpatialConvolution(64,64,3,3,1,1,1,1)(pool1)
  local cnv2 = nnlib.ReLU(true)(nn.SpatialBatchNormalization(64)(cnv2_))
  local cnv3_ = nnlib.SpatialConvolution(64,64,3,3,1,1,1,1)(cnv2)
  local cnv3 = nnlib.ReLU(true)(nn.SpatialBatchNormalization(64)(cnv3_))

  local r1 = MSResidua(64,128,1,'no_preact',false,opt.inputRes/2, B,C)(cnv3)
  local pool = nnlib.SpatialMaxPooling(2,2,2,2)(r1)                       -- 64
  local r4 = MSResidua(128,128,1,'preact',false,inputRes,B,C)(pool)
  local r5 = MSResidua(128,opt.nFeats,1,'preact',false,inputRes,B,C)(r4)

  local out = {}
  local inter = r5

  -- Bottom-up inference
  local outIdxSet = {}
  for levelIdx = 1,nSemanticLevels do
    local hg = hourglass_ms(4,opt.nFeats, nResidual, inter, inputRes,'preact',B,C)

    -- Residual layers at output resolution
    local ll = hg
    -- for j = 1,nResidual do ll = Residual(opt.nFeats,opt.nFeats)(ll) end
    -- Linear layer to produce first set of predictions
    ll = lin(opt.nFeats,opt.nFeats,ll)

    -- Predicted heatmaps
    local nOutChannels = nParts[levelIdx]
    if opt.catPartEnds and levelIdx ~= 1 then
      nOutChannels = nOutChannels + nParts[1]
      print (nOutChannels)
    end

    local tmpOut = nnlib.SpatialConvolution(opt.nFeats,nOutChannels,
        1,1,1,1,0,0)(ll)
    table.insert(out,tmpOut)

    -- Add predictions back if this is not the last hg module
    if #out < nStack then
      local ll_ = nnlib.SpatialConvolution(opt.nFeats,opt.nFeats,
          1,1,1,1,0,0)(ll)
      local tmpOut_ = nnlib.SpatialConvolution(nOutChannels,opt.nFeats,
          1,1,1,1,0,0)(tmpOut)
      inter = nn.CAddTable()({inter, ll_, tmpOut_})
    end

    table.insert(outIdxSet, 1, #out)
  end

  table.remove(outIdxSet, 1)
  -- Top-down inference
  for levelIdx = nSemanticLevels-1,1,-1 do
    -- Top-down connections
    local nRemainChannels = opt.nFeats - nParts[levelIdx]
    if opt.catPartEnds and levelIdx ~= 1 then
      nRemainChannels = nRemainChannels - nParts[1]
    end

    local inter_ = MSResidua(opt.nFeats, nRemainChannels,1,'preact',false,inputRes,B,C)(inter)
    local srcOutIdx = table.remove(outIdxSet, 1)
    inter = nn.JoinTable(2)({out[srcOutIdx], inter_})

    local hg = hourglass(4,opt.nFeats, nResidual, inter)

    -- Residual layers at output resolution
    local ll = hg
    -- for j = 1,nResidual do ll = Residual(opt.nFeats,opt.nFeats)(ll) end
    -- Linear layer to produce first set of predictions
    ll = lin(opt.nFeats,opt.nFeats,ll)

    -- Predicted heatmaps
    nOutChannels = nParts[levelIdx]
    if opt.catPartEnds and levelIdx ~= 1 then
      nOutChannels = nOutChannels + nParts[1]
    end

    local tmpOut = nnlib.SpatialConvolution(opt.nFeats,nOutChannels,
        1,1,1,1,0,0)(ll)
    table.insert(out,tmpOut)

    -- Add predictions back if this is not the last hg module
    if #out < nStack then
      local ll_ = nnlib.SpatialConvolution(opt.nFeats,opt.nFeats,
          1,1,1,1,0,0)(ll)
      local tmpOut_ = nnlib.SpatialConvolution(nOutChannels,opt.nFeats,
          1,1,1,1,0,0)(tmpOut)
      inter = nn.CAddTable()({inter, ll_, tmpOut_})
    end
  end

  -- Final model
  local model = nn.gModule({inp}, out)

  -- return model
  return model:cuda()

end

return createModel
