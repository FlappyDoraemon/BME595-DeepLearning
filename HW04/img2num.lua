require 'nn'
local mnist = require 'mnist'
local img2num={}
local net

--  local ex = testset[1]
--  local x = ex.x -- the input (a 28x28 ByteTensor)
--  local y = ex.y -- the label (0--9)
--  x = nn.Reshape(784):forward(x:double())

function img2num.train()
    net = nn.Sequential()
    net:add(nn.Linear(784,500))
    net:add(nn.Sigmoid())
    net:add(nn.Linear(500,10))
    net:add(nn.Sigmoid())
    local trainset = mnist.traindataset()
    local i
    local j
    net:zeroGradParameters()
    local criterion = nn.MSECriterion()
    for i = 1,5000 do
        local ex = trainset[i]
        local input = nn.Reshape(784):forward(ex.x:double()) -- the input (a 28x28 ByteTensor)
        local y = ex.y -- the label (0--9)
        local output = torch.zeros(10)
        output[{y+1}] = 1       --  0=>1,1=>2,2=>3,3=>4,4=>5
        criterion:forward(net:forward(input), output)
        net:zeroGradParameters()
        net:backward(input, criterion:backward(net.output, output))
        net:updateParameters(0.03)
    end
end

function img2num.forward(img)
    local x = nn.Reshape(784):forward(img:double())
    local output = net:forward(x)
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
