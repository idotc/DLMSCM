local nn = require 'nn'
local cunn = require 'cunn'
local cudnn = require 'cudnn'
local nngraph = require 'nngraph'
local nnlib = cudnn
local opt = require 'opts'

local struct = opt.struct
local nParts = struct.nParts
local nOutChannels = nParts[levelIdx]
print (nOutChannels)
