-- Define a compositional structure of human body
-- Design for FLIC dataset

-- struct = {children, partners, nParts}

local children = {}

-- level 1 (6 parts)
-- 1 l sho, 2 l elb, 3 l wrt
-- 4 r sho, 5 r elb, 6 r wrt
table.insert(children, 
{1, --1
2,  --2
3,  --3
4,  --4
5,  --5
6  --6
})

-- level 2 (4 parts):
table.insert(children, {
  {1,2},  --1 lu arm
  {2,3},  --2 ll arm
  {4,5},  --3 ru arm
  {5,6}   --4 rl arm
})

-- level 3 (2 parts)
table.insert(children, {
  {1,2},  --1 l arm
  {3,4}   --2 r arm
})

local partners = {}

-- level 1
table.insert(partners, {
  {1,4},   {2,5},   {3,6}
})
    
-- level 2
table.insert(partners, {
  {1,3}, {2,4}
})

-- level 3
table.insert(partners, {
  {1,2}
})

local nParts = {}
for k,v in pairs(children) do
  table.insert(nParts, #v)
end

local struct = {children=children, partners=partners, nParts=nParts}

return struct
