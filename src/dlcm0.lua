local nn = require 'nn'
local cunn = require 'cunn'
local cudnn = require 'cudnn'
local nngraph = require 'nngraph'
local nnlib = cudnn
local opt = require 'opts'

local nSemanticLevels = opt.nSemanticLevels
for levelIdx = 2,1,-1 do
  print (levelIdx)
end

a = torch.ones(5,5,2)*2
a [1] = a[1]*3
print (a)
b = torch.ones(5,5,2)*3
print(nn.CAddTable(false):forward({a,b}))

a = torch.ones(5,5,2)*2
a [1] = a[1]*3
print (a)
b = torch.ones(5,5,2)*3
print(nn.CAddTable(false):forward({a,b}))
