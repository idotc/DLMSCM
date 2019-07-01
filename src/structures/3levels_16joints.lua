-- Define a compositional structure of human body
-- It has three semantic levels, which respectively contain 16, 12 and 6 parts

-- struct = {children, partners, nParts}

local children = {}

-- level 1 (16 parts): 1 - r ankle, 2 - r knee, 3 - r hip, 4 - l hip,
--   5 - l knee, 6 - l ankle, 7 - pelvis, 8 - thorax, 9 - upper neck,
--   10 - head top, 11 - r wrist, 12 - r elbow, 13 - r shoulder,
--   14 - l shoulder, 15 - l elbow, 16 - l wrist
table.insert(children, torch.range(1,16):totable())

-- level 2 (12 parts):
table.insert(children, {
  {1,2},  --1 rl leg
  {2,3},  --2 ru leg
  {4,5},  --3 lu leg
  {5,6},  --4 ll leg
  {3,7},  --5 r hip
  {4,7},  --6 l hip
  {8,9},  --7 neck
  {9,10}, --8 head
  {11,12},--9 rl arm
  {12,13},--10 ru arm
  {14,15},--11 lu arm
  {15,16} --12 ll arm
})

-- level 3 (6 parts)
table.insert(children, {
  {1,2},  --1 r leg
  {3,4},  --2 l leg
  {5,6},  --3 waist
  {7,8},  --4 head-neck
  {9,10}, --5 r arm
  {11,12} --6 l arm
})

local partners = {}

-- level 1
table.insert(partners, {
  {1,6},   {2,5},   {3,4},
  {11,16}, {12,15}, {13,14}
})

-- level 2
table.insert(partners, {
  {1,4}, {2,3}, {5,6},
  {9,12}, {10,11}
})

-- level 3
table.insert(partners, {
  {1,2}, {5,6}
})

local nParts = {}
for k,v in pairs(children) do
  table.insert(nParts, #v)
end

local struct = {children=children, partners=partners, nParts=nParts}

return struct
