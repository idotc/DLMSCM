-- Define a compositional structure of human body
-- Remove parts 7 and 8 because they are not annotated in the LSP data
-- It has three semantic levels, which respectively contain 14, 9 and 5 parts

-- struct = {children, partners, nParts}

local children = {}

-- level 1 (14 parts): 1 - r ankle, 2 - r knee, 3 - r hip, 4 - l hip, 
--   5 - l knee, 6 - l ankle, 9 - upper neck, 
--   10 - head top, 11 - r wrist, 12 - r elbow, 13 - r shoulder, 
--   14 - l shoulder, 15 - l elbow, 16 - l wrist
-- table.insert(children, torch.range(1,16):totable())
table.insert(children, 
{1, --1
2,  --2
3,  --3
4,  --4
5,  --5
6,  --6
9,  --7
10, --8
11, --9
12, --10
13, --11
14, --12
15, --13
16  --14
})

-- level 2 (9 parts):
table.insert(children, {
  {1,2},  --1 rl leg
  {2,3},  --2 ru leg
  {4,5},  --3 lu leg
  {5,6},  --4 ll leg
  {7,8},  --5 head
  {9,10}, --6 rl arm
  {10,11},--7 ru arm
  {12,13},--8 lu arm
  {13,14} --9 ll arm
})

-- level 3 (5 parts)
table.insert(children, {
  {1,2},  --1 r leg
  {3,4},  --2 l leg
  {5},    --3 head-neck
  {6,7},  --4 r arm
  {8,9}   --5 l arm
})

local partners = {}

-- level 1
table.insert(partners, {
  {1,6},   {2,5},   {3,4},
  {9,14}, {10,13}, {11,12}
})
    
-- level 2
table.insert(partners, {
  {1,4}, {2,3},
  {6,9}, {7,8}
})

-- level 3
table.insert(partners, {
  {1,2}, {4,5}
})

local nParts = {}
for k,v in pairs(children) do
  table.insert(nParts, #v)
end

local struct = {children=children, partners=partners, nParts=nParts}

return struct
