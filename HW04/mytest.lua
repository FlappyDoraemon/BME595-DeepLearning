local logicGatesLib = require 'logicGates'
local mnist = require 'mnist'
local img2numLib = require 'img2num'
require 'nn'



----------------------------------------------------------------------
-- PART 1 : MNIST TEST
----------------------------------------------------------------------



--  test with minist dataset
--timer = torch.Timer()
--timer:reset()
img2numLib.train()
--print(timer:time().real)
--  local trainset = mnist.traindataset()
local testset = mnist.testdataset()
--  print(trainset.size) -- to retrieve the size
print(testset.size) -- to retrieve the size
local i
local mark = 0
for i = 1 , 100 do
    local ex = testset[i]
    local img = ex.x -- the input (a 28x28 ByteTensor)
    local y = ex.y -- the label (0--9)
    local printy = img2numLib.forward(img)
    print(y,printy)
    if printy == y then
        mark = mark + 1
    end
end
print('accuracy rate after training the whole train-set:',mark,'%')




----------------------------------------------------------------------
-- PART 2 : LOGIC GATES TEST
----------------------------------------------------------------------



logicGatesLib.AND.train()
print('AND result:')
print('false and false should be 0: ',logicGatesLib.AND.forward(false, false))
print('false and true should be 0: ',logicGatesLib.AND.forward(false, true))
print('true and false should be 0: ',logicGatesLib.AND.forward(true, false))
print('true and true should be 1: ',logicGatesLib.AND.forward(true, true))


logicGatesLib.OR.train()
print('OR result:')
print('false and false should be 0: ',logicGatesLib.OR.forward(false, false))
print('false and true should be 1: ',logicGatesLib.OR.forward(false, true))
print('true and false should be 1: ',logicGatesLib.OR.forward(true, false))
print('true and true should be 1: ',logicGatesLib.OR.forward(true, true))



logicGatesLib.NOT.train()
print('NOT result:')
print('true should be 0: ',logicGatesLib.NOT.forward(true))
print('false should be 1: ',logicGatesLib.NOT.forward(false))

logicGatesLib.XOR.train()
print('XOR result:')
print('false and false should be 0: ',logicGatesLib.XOR.forward(false, false))
print('false and true should be 1: ',logicGatesLib.XOR.forward(false, true))
print('true and false should be 1: ',logicGatesLib.XOR.forward(true, false))
print('true and true should be 0: ',logicGatesLib.XOR.forward(true, true))


