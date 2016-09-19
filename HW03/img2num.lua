require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'paths'
local mnist = require 'mnist'
local NeuralNetworkLib = require 'NeuralNetwork'
local img2num={}

--  local ex = testset[1]
--  local x = ex.x -- the input (a 28x28 ByteTensor)
--  local y = ex.y -- the label (0--9)
--  x = nn.Reshape(784):forward(x:double())

function img2num.train()
    NeuralNetworkLib.build({784,500,10})
    local trainset = mnist.traindataset()
    --  local testset = mnist.testdataset()
    local i
    local j
    local batch_size = 20
    print('batch_size = ',batch_size)
    for i = 1 , 30 do --trainset.size do
        --if i % 100 == 0 then
        print('training the sample of ',i,'of all','500 training batches')
        local ex = trainset[(i-1)*batch_size+1]
        local x = nn.Reshape(784):forward(ex.x:double()) -- the input (a 28x28 ByteTensor)
        local y = ex.y -- the label (0--9)
        local target = torch.zeros(10)
        target[{y+1}] = 1       --  0=>1,1=>2,2=>3,3=>4,4=>5
        for j = 1 , (batch_size-1) do
            ex = trainset[(i-1)*batch_size+1+j]
            x = torch.cat(x , nn.Reshape(784):forward(ex.x:double()) , 2)
            y = ex.y -- the label (0--9)
            local target_2 = torch.zeros(10)
            target_2[{y+1}] = 1       --  0=>1,1=>2,2=>3,3=>4,4=>5
            target = torch.cat(target , target_2 , 2)
        end
        NeuralNetworkLib.forward(torch.cat(torch.ones(1,batch_size) , x , 1))
        NeuralNetworkLib.backward(target)
        NeuralNetworkLib.updateParams(0.05)
    end
end

function img2num.forward(img)
    local x = nn.Reshape(784):forward(img:double())
    local output = NeuralNetworkLib.forward(torch.cat(torch.ones(1) , x , 1))
    local currentmax = output[{1}]
    local i
    local outputnum = 0
    for i = 1 , 9 do
        if output[{i+1}] > currentmax then
            outputnum = i
            currentmax = output[{i+1}]
        end
    end
    return outputnum
end









return img2num
