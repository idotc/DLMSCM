local image = require 'image'

local M = {}

-------------------------------------------------------------------------------
-- Affine transformation
-------------------------------------------------------------------------------
function M.getAffineTransform(center, scale, rot, shear, res)
  -- Original person size
  local w = 200 * scale[1]
  local h = 200 * scale[2]

  -- The transform matrix
  local affineMat = torch.eye(3)

  -- Scaling
  -- target size / original size
  affineMat[1][1] = res / w
  affineMat[2][2] = res / h

  -- Translation
  -- Translation is after scaling, so in the target space
  -- res/w*(0.5*w-center[1])
  affineMat[1][3] = res * (-center[1] / w + .5)
  affineMat[2][3] = res * (-center[2] / h + .5)

  -- Shearing and Rotation 
  if rot ~= 0 or shear[1] ~= 0 or shear[2] ~=0 then
    local rotMat = torch.eye(3)
    local ang = rot * math.pi / 180
    local sinVal = math.sin(ang)
    local cosVal = math.cos(ang)
    rotMat[1][1] = cosVal
    rotMat[1][2] = -sinVal
    rotMat[2][1] = sinVal
    rotMat[2][2] = cosVal

    local shearMat = torch.eye(3)
    shearMat[1][2] = shear[1]
    shearMat[2][1] = shear[2]

    -- Need to make sure rotation is around center
    -- Fisrt move the center point to the origin then move back
    local shiftMat = torch.eye(3)
    shiftMat[1][3] = -res/2
    shiftMat[2][3] = -res/2
    local shiftMatInv = torch.eye(3)
    shiftMatInv[1][3] = res/2
    shiftMatInv[2][3] = res/2

    -- First shear, then rotate
    affineMat = shiftMatInv * rotMat * shearMat * shiftMat * affineMat
  end

  return affineMat
end

function M.affineTransform(point, center, scale, rot, shear, res, invert)
  local point_ = torch.ones(3)
  point_[1],point_[2] = point[1]-1,point[2]-1

  local affineMat = M.getAffineTransform(center, scale, rot, shear, res)
  if invert then
    affineMat = torch.inverse(affineMat)
  end
  local newPoint = (affineMat*point_):sub(1,2)

  return newPoint:int():add(1)
end

function M.affineCrop(img, center, scale, rot, shear, res)
  -- scale should be a tensor with 2 elements (x_scale, y_scale)
  assert(scale:nElement()==2) 
  -- shear should be a tensor with 2 elements (x_shear, y_shear)
  assert(shear:nElement()==2)

  local ndim = img:nDimension()
  if ndim == 2 then img = img:view(1,img:size(1),img:size(2)) end
  local ht, wd = img:size(2), img:size(3)
  local tmpImg, newImg = img, torch.zeros(img:size(1), res, res)

  -- scale first to save computation
  local scaleFactor = (200 * scale) / res
  local newWd = math.floor(wd / scaleFactor[1])
  local newHt = math.floor(ht / scaleFactor[2])
  tmpImg = image.scale(img, newWd, newHt)
  ht, wd = tmpImg:size(2), tmpImg:size(3)

  -- calculate upper left and bottom right coordinates defining crop region
  local center = torch.cdiv(center, scaleFactor) 
  local scale = torch.cdiv(scale, scaleFactor)
  local ul = M.affineTransform({1,1}, center, scale, 0, torch.zeros(2), res, 
      true)
  br = ul + res

  -- If the image is to be rotated or sheared, pad the cropped area
  local pad = 0
  if rot ~= 0 or shear[1] ~= 0 or shear[2] ~= 0 then
    -- account for max rotation
    pad = math.ceil(torch.norm((ul - br):float())/2 - (br[1]-ul[1])/2)
    pad = pad*2
    ul:add(-pad); br:add(pad) 
  end

  -- Define the range of pixels to take from the old image
  local old_ = {1,-1,math.max(1, ul[2]), math.min(br[2], ht+1) - 1,
      math.max(1, ul[1]), math.min(br[1], wd+1) - 1}
  -- And where to put them in the new image
  local new_ = {1,-1,math.max(1, -ul[2] + 2), math.min(br[2], ht+1) - ul[2],
      math.max(1, -ul[1] + 2), math.min(br[1], wd+1) - ul[1]}

  -- Initialize new image and copy pixels over
  local newImg = torch.zeros(img:size(1), br[2] - ul[2], br[1] - ul[1])
  if not pcall(function() 
    newImg:sub(unpack(new_)):copy(tmpImg:sub(unpack(old_))) end) then
    print("Error occurred during crop!")
  end

  if rot ~= 0 or shear[1] ~= 0 or shear[2] ~= 0 then
    -- Rotate and shear the image
    local rotMat = torch.eye(2)
    local ang = rot*math.pi/180
    local sinVal, cosVal = math.sin(ang), math.cos(ang)
    rotMat[1][1] = cosVal
    rotMat[1][2] = -sinVal
    rotMat[2][1] = sinVal
    rotMat[2][2] = cosVal

    local shearMat = torch.eye(2)
    shearMat[1][2], shearMat[2][1] = shear[1], shear[2]

    local affineMat = rotMat*shearMat
    -- Change to (y, x)
    affineMat[1][1], affineMat[2][2] = affineMat[2][2], affineMat[1][1]
    affineMat[1][2], affineMat[2][1] = affineMat[2][1], affineMat[1][2]
    newImg = image.affinetransform(newImg, torch.inverse(affineMat), 'bilinear',
        torch.Tensor({0,0}), 'pad')

    -- Remove padded area
    newImg = newImg:sub(1,-1,pad+1,newImg:size(2)-pad,pad+1,newImg:size(3)-pad)
        :clone()
  end

  if ndim == 2 then newImg = newImg:view(newImg:size(2),newImg:size(3)) end
  return newImg
end

-------------------------------------------------------------------------------
-- Coordinate transformation
-------------------------------------------------------------------------------
function M.getTransform(center, scale, rot, res)
  local h = 200 * scale
  local t = torch.eye(3)

  -- Scaling
  t[1][1] = res / h
  t[2][2] = res / h

  -- Translation
  t[1][3] = res * (-center[1] / h + .5)
  t[2][3] = res * (-center[2] / h + .5)

  -- Rotation
  if rot ~= 0 then
    rot = -rot
    local r = torch.eye(3)
    local ang = rot * math.pi / 180
    local s = math.sin(ang)
    local c = math.cos(ang)
    r[1][1] = c
    r[1][2] = -s
    r[2][1] = s
    r[2][2] = c
    -- Need to make sure rotation is around center
    local t_ = torch.eye(3)
    t_[1][3] = -res/2
    t_[2][3] = -res/2
    local t_inv = torch.eye(3)
    t_inv[1][3] = res/2
    t_inv[2][3] = res/2
    t = t_inv * r * t_ * t
  end

  return t
end

function M.transform(pt, center, scale, rot, res, invert)
  local pt_ = torch.ones(3)
  pt_[1],pt_[2] = pt[1]-1,pt[2]-1

  local t = M.getTransform(center, scale, rot, res)
  if invert then
    t = torch.inverse(t)
  end
  local new_point = (t*pt_):sub(1,2)

  return new_point:int():add(1)
end

-------------------------------------------------------------------------------
-- Cropping
-------------------------------------------------------------------------------
function M.crop(img, center, scale, rot, res)
  local ul = M.transform({1,1}, center, scale, 0, res, true)
  local br = M.transform({res+1,res+1}, center, scale, 0, res, true)


  local pad = math.floor(torch.norm((ul - br):float())/2 - (br[1]-ul[1])/2)
  if rot ~= 0 then
    ul = ul - pad
    br = br + pad
  end

  local newDim,newImg,ht,wd

  if img:size():size() > 2 then
    newDim = torch.IntTensor({img:size(1), br[2] - ul[2], br[1] - ul[1]})
    newImg = torch.zeros(newDim[1],newDim[2],newDim[3])
    ht = img:size(2)
    wd = img:size(3)
  else
    newDim = torch.IntTensor({br[2] - ul[2], br[1] - ul[1]})
    newImg = torch.zeros(newDim[1],newDim[2])
    ht = img:size(1)
    wd = img:size(2)
  end

  local newX = torch.Tensor({math.max(1, -ul[1] + 2), math.min(br[1], wd+1) - ul[1]})
  local newY = torch.Tensor({math.max(1, -ul[2] + 2), math.min(br[2], ht+1) - ul[2]})
  local oldX = torch.Tensor({math.max(1, ul[1]), math.min(br[1], wd+1) - 1})
  local oldY = torch.Tensor({math.max(1, ul[2]), math.min(br[2], ht+1) - 1})

  if newDim:size(1) > 2 then
    newImg:sub(1,newDim[1],newY[1],newY[2],newX[1],newX[2]):
        copy(img:sub(1,newDim[1],oldY[1],oldY[2],oldX[1],oldX[2]))
  else
    newImg:sub(newY[1],newY[2],newX[1],newX[2]):
        copy(img:sub(oldY[1],oldY[2],oldX[1],oldX[2]))
  end

  if rot ~= 0 then
    newImg = image.rotate(newImg, rot * math.pi / 180, 'bilinear')
    if newDim:size(1) > 2 then
      newImg = newImg:sub(1,newDim[1],pad,newDim[2]-pad,pad,newDim[3]-pad)
    else
      newImg = newImg:sub(pad,newDim[1]-pad,pad,newDim[2]-pad)
    end
  end

  newImg = image.scale(newImg,res,res)
  return newImg
end

function M.crop2(img, center, scale, rot, res)
  local ndim = img:nDimension()
  if ndim == 2 then img = img:view(1,img:size(1),img:size(2)) end
  local ht,wd = img:size(2), img:size(3)
  local tmpImg,newImg = img, torch.zeros(img:size(1), res, res)

  -- Modify crop approach depending on whether we zoom in/out
  -- This is for efficiency in extreme scaling cases
  local scaleFactor = (200 * scale) / res
  if scaleFactor < 2 then scaleFactor = 1
  else
    local newSize = math.floor(math.max(ht,wd) / scaleFactor)
    if newSize < 2 then
      -- Zoomed out so much that the image is now a single pixel or less
      if ndim == 2 then newImg = newImg:view(newImg:size(2),newImg:size(3)) end
      return newImg
    else
      tmpImg = image.scale(img,newSize)
      ht,wd = tmpImg:size(2),tmpImg:size(3)
    end
  end

  -- Calculate upper left and bottom right coordinates defining crop region
  local c,s = center:float()/scaleFactor, scale/scaleFactor
  local ul = M.transform({1,1}, c, s, 0, res, true)
  local br = M.transform({res+1,res+1}, c, s, 0, res, true)
  if scaleFactor >= 2 then br:add(-(br - ul - res)) end

  -- If the image is to be rotated, pad the cropped area
  local pad = math.ceil(torch.norm((ul - br):float())/2 - (br[1]-ul[1])/2)
  if rot ~= 0 then ul:add(-pad); br:add(pad) end

  -- Define the range of pixels to take from the old image
  local old_ = {1,-1,math.max(1, ul[2]), math.min(br[2], ht+1) - 1,
  math.max(1, ul[1]), math.min(br[1], wd+1) - 1}
  -- And where to put them in the new image
  local new_ = {1,-1,math.max(1, -ul[2] + 2), math.min(br[2], ht+1) - ul[2],
  math.max(1, -ul[1] + 2), math.min(br[1], wd+1) - ul[1]}

  -- Initialize new image and copy pixels over
  local newImg = torch.zeros(img:size(1), br[2] - ul[2], br[1] - ul[1])
  if not pcall(function() newImg:sub(unpack(new_)):copy(tmpImg:sub(unpack(old_))) end) then
  print("Error occurred during crop!")
end

if rot ~= 0 then
  -- Rotate the image and remove padded area
  newImg = image.rotate(newImg, rot * math.pi / 180, 'bilinear')
  newImg = newImg:sub(1,-1,pad+1,newImg:size(2)-pad,pad+1,newImg:size(3)-pad):clone()
end

if scaleFactor < 2 then newImg = image.scale(newImg,res,res) end
if ndim == 2 then newImg = newImg:view(newImg:size(2),newImg:size(3)) end
return newImg
end

-------------------------------------------------------------------------------
-- Draw gaussian
-------------------------------------------------------------------------------
function M.drawGaussian(img, pt, sigma)
  -- Draw a 2D gaussian
  -- Check that any part of the gaussian is in-bounds
  local tmpSize = math.ceil(3*sigma)
  local ul = {math.floor(pt[1] - tmpSize), math.floor(pt[2] - tmpSize)}
  local br = {math.floor(pt[1] + tmpSize), math.floor(pt[2] + tmpSize)}
  -- If not, return the image as is
  if (ul[1] > img:size(2) or ul[2] > img:size(1) or br[1] < 1 or br[2] < 1) then return img end
  -- Generate gaussian
  local size = 2*tmpSize + 1
  local g = image.gaussian(size)
  -- Usable gaussian range
  local g_x = {math.max(1, -ul[1]), math.min(br[1], img:size(2)) - math.max(1, ul[1]) + math.max(1, -ul[1])}
  local g_y = {math.max(1, -ul[2]), math.min(br[2], img:size(1)) - math.max(1, ul[2]) + math.max(1, -ul[2])}
  -- Image range
  local img_x = {math.max(1, ul[1]), math.min(br[1], img:size(2))}
  local img_y = {math.max(1, ul[2]), math.min(br[2], img:size(1))}
  assert(g_x[1] > 0 and g_y[1] > 0)
  img:sub(img_y[1], img_y[2], img_x[1], img_x[2]):cmax(g:sub(g_y[1], g_y[2], g_x[1], g_x[2]))
  return img
end

local function getSegmentPoints(x1, y1, x2, y2) 
  -- Bresenham's line algorithm
  -- Return all the points located on line segment between (x1,y1) and (x2,y2)
  local steep = math.abs(y2-y1) > math.abs(x2-x1)

  if steep then
    x1, y1 = y1, x1
    x2, y2 = y2, x2
  end

  if x1 > x2 then
    x1, x2 = x2, x1
    y1, y2 = y2, y1
  end

  local dx = x2 - x1
  local dy = math.abs(y2 - y1)

  local error = dx/2.0
  local ystep = y1 < y2 and 1 or -1
  local y = math.floor(y1)
  local maxX = math.floor(x2)
  local x = x1
  res = torch.zeros(maxX-x1, 2)

  for i=1,res:size(1) do
    if steep then
      res[i][1] = y
      res[i][2] = x
    else
      res[i][1] = x
      res[i][2] = y
    end
    error = error - dy
    if error < 0 then
      y = y + ystep
      error = error + dx
    end
    x = x + 1
  end
  return res
end

function M.drawLimbMap(img, pt1, pt2, sigma)
  -- Created by Wei
  -- Draw gaussian heat maps along the limb between pt1 and pt2
  -- pt1 and pt2 are 1-based locations {x, y}
  assert(img:dim() == 2)

  if pt1[1] == pt2[1] and pt1[2] == pt2[2] then
    M.drawGaussian(img, pt1, sigma)
    return
  end

  local h, w = img:size(1), img:size(2)
  local tmpSize = math.ceil(3*sigma)
  -- Make sure both joint heat maps are in-bounds
  if (math.floor(math.max(pt1[1], pt2[1]) - tmpSize) > w or 
      math.floor(math.max(pt1[2], pt2[2]) - tmpSize) > h or
      math.floor(math.min(pt1[1], pt2[1]) + tmpSize) < 1 or 
      math.floor(math.min(pt1[2], pt2[2]) + tmpSize) < 1 ) then
    return img
  end

  local segment = getSegmentPoints(pt1[1], pt1[2], pt2[1], pt2[2])
  for i = 1,segment:size(1) do
    M.drawGaussian(img, segment[i], sigma)
  end
end

function M.composeMaps(out, inp, idx_set)
  -- Created by Wei
  -- Compose heat maps of children to get heat maps of parents
  assert(inp:dim() == 3 and out:dim() == 2)
  assert(inp:size(2) == out:size(1) and inp:size(3) == out:size(2))
  for _,v in ipairs(idx_set) do
    out:cmax(inp[v])
  end
end

-------------------------------------------------------------------------------
-- Draw Offset Field
-------------------------------------------------------------------------------
function M.drawOffset(img, pt)
  -- img: 2xHxW offset field
  local h, w = img:size(2), img:size(3)
  assert(h == w)
  for i = 1, h do
    local dy = i - pt[2]
    img[{{2}, {i}, {}}] = dy
  end

  for j = 1, w do
    local dx = j - pt[1]
    img[{{1}, {}, {j}}] = dx
  end
  return img
end

-------------------------------------------------------------------------------
-- Flipping functions
-------------------------------------------------------------------------------
function M.shuffleLR(x, partners, outLevels)
  -- x = #out*{Tensor(#batch, nPart, h, w)}
  -- outLevels = {1, 2, 3, 2, 1}
  assert(#x == #outLevels)
  local dim
  if x[1]:nDimension() == 4 then
    dim = 2
  else
    assert(x[1]:nDimension() == 3)
    dim = 1
  end

  for k = 1,#x do
    local l = outLevels[k] 
    for i = 1,#partners[l] do
      local idx1, idx2 = unpack(partners[l][i])
      local tmp = x[k]:narrow(dim, idx1, 1):clone()
      x[k]:narrow(dim, idx1, 1):copy(x[k]:narrow(dim, idx2, 1))
      x[k]:narrow(dim, idx2, 1):copy(tmp)
    end
  end

  return x
end

function M.flip(x)
  require 'image'
  local y = torch.FloatTensor(x:size())
  for i = 1, x:size(1) do
    image.hflip(y[i], x[i]:float())
  end
  return y:typeAs(x)
end

function M.colorNormalize(img, meanstd)
  assert(img:size(1) == 3, ('images should be 3 channel (%d channel now)'):
      format(img:dim()))
  for i=1,3 do
    img[i]:add(-meanstd.mean[i])
  end
  return img
end


function M.colorNormalizeMeanImg(img, meanImg)
  assert(img:size(1) == 3, ('images should be 3 channel (%d channel now)'):
      format(img:dim()))
  assert(meanImg:size(1) == 3, ('meanImg should be 3 channel (%d channel now)'):
      format(meanImg:dim()))
  img:csub(meanImg)
  return img
end


local function blend(img1, img2, alpha)
  return img1:mul(alpha):add(1 - alpha, img2)
end

local function grayscale(dst, img)
  dst:resizeAs(img)
  dst[1]:zero()
  dst[1]:add(0.299, img[1]):add(0.587, img[2]):add(0.114, img[3])
  dst[2]:copy(dst[1])
  dst[3]:copy(dst[1])
  return dst
end

function M.Saturation(input, var)
  if var == 0 then return input end
  local gs = gs or input.new()
  grayscale(gs, input)

  local alpha = 1.0 + torch.uniform(-var, var)
  blend(input, gs, alpha)
  return input
end

function M.Brightness(input, var)
  if var == 0 then return input end
  local gs
  gs = gs or input.new()
  gs:resizeAs(input):zero()

  local alpha = 1.0 + torch.uniform(-var, var)
  blend(input, gs, alpha)
  return input
end

function M.Contrast(input, var)
  if var == 0 then return input end
  local gs
  gs = gs or input.new()
  grayscale(gs, input)
  gs:fill(gs[1]:mean())

  local alpha = 1.0 + torch.uniform(-var, var)
  blend(input, gs, alpha)
  return input
end

function M.colorJitter(input, brightness, contrast, saturation)
  local brightness = brightness or 0
  local contrast = contrast or 0
  local saturation = saturation or 0

  local ts = {'Brightness', 'Contrast', 'Saturation'}
  local var = {brightness, contrast, saturation}
  local order = torch.randperm(#ts)
  for i=1,#ts do
    input = M[ts[order[i]]](input, var[order[i]])
  end

  return input
end

function M.colorNoise(input, var)

  if var == 0 then return input end
  local h, w = input:size(2), input:size(3)
  local gs = torch.Tensor(1, h, w):normal(0, 0.2)
  local mask = torch.Tensor(1, h, w):uniform(0, 1)

  gs[mask:gt(var)] = 0

  input = input + gs:expandAs(input)
  return input:clamp(0, 1)
end

function M.gaussianBlur(input, var)
  local kw = math.floor(torch.uniform(0, var))
  if torch.uniform() <= .6 then kw = 0 end

  if kw ~= 0 then
    local k = image.gaussian{width=3, normalize=true}:typeAs(input)
    return image.convolve(input, k, 'same'):contiguous()
  end
  return input
end

return M
