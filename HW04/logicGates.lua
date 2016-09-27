require 'nn';
local logicGates = {}
logicGates.AND = {} 
logicGates.OR = {} 
logicGates.NOT = {}
logicGates.XOR = {} 
local net

function logicGates.AND.train()
    net = nn.Sequential()
    net:add(nn.Linear(2,5))
    net:add(nn.Sigmoid())
    net:add(nn.Linear(5,1))
    net:add(nn.Sigmoid())
    local i
    net:zeroGradParameters()
    local criterion = nn.MSECriterion()
    for i = 1,2500 do
        local input= torch.Tensor({1,1});     -- normally distributed example in 2d
        local output= torch.Tensor({1});
        criterion:forward(net:forward(input), output)
        net:zeroGradParameters()
        net:backward(input, criterion:backward(net.output, output))
        net:updateParameters(0.1)
        
        input= torch.Tensor({1,0});     -- normally distributed example in 2d
        output= torch.Tensor({0});
        criterion:forward(net:forward(input), output)
        net:zeroGradParameters()
        net:backward(input, criterion:backward(net.output, output))
        net:updateParameters(0.1)
        
        input= torch.Tensor({0,1});     -- normally distributed example in 2d
        output= torch.Tensor({0});
        criterion:forward(net:forward(input), output)
        net:zeroGradParameters()
        net:backward(input, criterion:backward(net.output, output))
        net:updateParameters(0.1)
        
        input= torch.Tensor({0,0});     -- normally distributed example in 2d
        output= torch.Tensor({0});
        criterion:forward(net:forward(input), output)
        net:zeroGradParameters()
        net:backward(input, criterion:backward(net.output, output))
        net:updateParameters(0.1)
    end
end

function logicGates.AND.forward(x_in, y_in)
    local x = x_in and 1 or 0
    local y = y_in and 1 or 0
    local output = net:forward(torch.Tensor({x , y}))
    return output[1] > 0.5
end

function logicGates.OR.train()
    net = nn.Sequential()
    net:add(nn.Linear(2,5))
    net:add(nn.Sigmoid())
    net:add(nn.Linear(5,1))
    net:add(nn.Sigmoid())
    local i
    net:zeroGradParameters()
    local criterion = nn.MSECriterion()
    for i = 1,2500 do
        local input= torch.Tensor({1,1});     -- normally distributed example in 2d
        local output= torch.Tensor({1});
        criterion:forward(net:forward(input), output)
        net:zeroGradParameters()
        net:backward(input, criterion:backward(net.output, output))
        net:updateParameters(0.1)
        
        input= torch.Tensor({1,0});     -- normally distributed example in 2d
        output= torch.Tensor({1});
        criterion:forward(net:forward(input), output)
        net:zeroGradParameters()
        net:backward(input, criterion:backward(net.output, output))
        net:updateParameters(0.1)
        
        input= torch.Tensor({0,1});     -- normally distributed example in 2d
        output= torch.Tensor({1});
        criterion:forward(net:forward(input), output)
        net:zeroGradParameters()
        net:backward(input, criterion:backward(net.output, output))
        net:updateParameters(0.1)
        
        input= torch.Tensor({0,0});     -- normally distributed example in 2d
        output= torch.Tensor({0});
        criterion:forward(net:forward(input), output)
        net:zeroGradParameters()
        net:backward(input, criterion:backward(net.output, output))
        net:updateParameters(0.1)
    end
end

function logicGates.OR.forward(x_in, y_in)
    local x = x_in and 1 or 0
    local y = y_in and 1 or 0
    local output = net:forward(torch.Tensor({x , y}))
    return output[1] > 0.5
end

function logicGates.NOT.train()
    net = nn.Sequential()
    net:add(nn.Linear(1,5))
    net:add(nn.Sigmoid())
    net:add(nn.Linear(5,1))
    net:add(nn.Sigmoid())
    local i
    net:zeroGradParameters()
    local criterion = nn.MSECriterion()
    for i = 1,2500 do
        local input= torch.Tensor({1});     -- normally distributed example in 2d
        local output= torch.Tensor({0});
        criterion:forward(net:forward(input), output)
        net:zeroGradParameters()
        net:backward(input, criterion:backward(net.output, output))
        net:updateParameters(0.1)
        
        input= torch.Tensor({0});     -- normally distributed example in 2d
        output= torch.Tensor({1});
        criterion:forward(net:forward(input), output)
        net:zeroGradParameters()
        net:backward(input, criterion:backward(net.output, output))
        net:updateParameters(0.1)
    end
end

function logicGates.NOT.forward(x_in)
    local x = x_in and 1 or 0
    local output = net:forward(torch.Tensor({x}))
    return output[1] > 0.5
end


function logicGates.XOR.train()
    net = nn.Sequential()
    net:add(nn.Linear(2,5))
    net:add(nn.Sigmoid())
    net:add(nn.Linear(5,1))
    net:add(nn.Sigmoid())
    local i
    net:zeroGradParameters()
    local criterion = nn.MSECriterion()
    for i = 1,2500 do
        local input= torch.Tensor({1,1});     -- normally distributed example in 2d
        local output= torch.Tensor({0});
        criterion:forward(net:forward(input), output)
        net:zeroGradParameters()
        net:backward(input, criterion:backward(net.output, output))
        net:updateParameters(0.1)
        
        input= torch.Tensor({1,0});     -- normally distributed example in 2d
        output= torch.Tensor({1});
        criterion:forward(net:forward(input), output)
        net:zeroGradParameters()
        net:backward(input, criterion:backward(net.output, output))
        net:updateParameters(0.1)
        
        input= torch.Tensor({0,1});     -- normally distributed example in 2d
        output= torch.Tensor({1});
        criterion:forward(net:forward(input), output)
        net:zeroGradParameters()
        net:backward(input, criterion:backward(net.output, output))
        net:updateParameters(0.1)
        
        input= torch.Tensor({0,0});     -- normally distributed example in 2d
        output= torch.Tensor({0});
        criterion:forward(net:forward(input), output)
        net:zeroGradParameters()
        net:backward(input, criterion:backward(net.output, output))
        net:updateParameters(0.1)
    end
end

function logicGates.XOR.forward(x_in, y_in)
    local x = x_in and 1 or 0
    local y = y_in and 1 or 0
    local output = net:forward(torch.Tensor({x , y}))
    return output[1] > 0.5
end

return logicGates
