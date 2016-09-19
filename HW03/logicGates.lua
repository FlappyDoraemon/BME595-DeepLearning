local NeuralNetworkLib = require 'NeuralNetwork'
local logicGates = {}
logicGates.AND = {} 
logicGates.OR = {} 
logicGates.NOT = {}
logicGates.XOR = {} 

function logicGates.AND.train()
    NeuralNetworkLib.build({2,1})
    --  NeuralNetworkLib.build({2,2,1})
    local i
    for i = 1 , 250 do
        -- 1,1 => 1
        NeuralNetworkLib.forward(torch.Tensor({1,1,1}))
        NeuralNetworkLib.backward(torch.Tensor({1}))
        NeuralNetworkLib.updateParams(0.5)
        -- 0,0 => 0
        NeuralNetworkLib.forward(torch.Tensor({1,0,0}))
        NeuralNetworkLib.backward(torch.Tensor({0}))
        NeuralNetworkLib.updateParams(0.5)
        -- 0,1 => 0
        NeuralNetworkLib.forward(torch.Tensor({1,0,1}))
        NeuralNetworkLib.backward(torch.Tensor({0}))
        NeuralNetworkLib.updateParams(0.5)
        -- 1,0 => 0
        NeuralNetworkLib.forward(torch.Tensor({1,1,0}))
        NeuralNetworkLib.backward(torch.Tensor({0}))
        NeuralNetworkLib.updateParams(0.5)
    end
end

function logicGates.AND.forward(x_in, y_in)
    local x = x_in and 1 or 0
    local y = y_in and 1 or 0
    local output = NeuralNetworkLib.forward(torch.Tensor({1, x , y}))
    return output[1] > 0.5
end

function logicGates.AND.set()
    NeuralNetworkLib.build({2,1})
    local temp = NeuralNetworkLib.getLayer(1)  
    -- torch.Tensor({{-1.5 , 1 , 1}})
    temp:sub(1,1,1,1):fill(-1.5)
    temp:sub(1,1,2,2):fill(1)
    temp:sub(1,1,3,3):fill(1)
    -- local output = NeuralNetworkLib.forward(torch.Tensor({1, x , y}))
    -- return output[1] > 0.5
end

function logicGates.OR.train()
    NeuralNetworkLib.build({2,1})
    -- NeuralNetworkLib.build({2,2,1})
    local i
    for i = 1 , 250 do
        -- 1,1 => 1
        NeuralNetworkLib.forward(torch.Tensor({1,1,1}))
        NeuralNetworkLib.backward(torch.Tensor({1}))
        NeuralNetworkLib.updateParams(0.5)
        -- 0,0 => 0
        NeuralNetworkLib.forward(torch.Tensor({1,0,0}))
        NeuralNetworkLib.backward(torch.Tensor({0}))
        NeuralNetworkLib.updateParams(0.5)
        -- 0,1 => 1
        NeuralNetworkLib.forward(torch.Tensor({1,0,1}))
        NeuralNetworkLib.backward(torch.Tensor({1}))
        NeuralNetworkLib.updateParams(0.5)
        -- 1,0 => 1
        NeuralNetworkLib.forward(torch.Tensor({1,1,0}))
        NeuralNetworkLib.backward(torch.Tensor({1}))
        NeuralNetworkLib.updateParams(0.5)
    end
end

function logicGates.OR.forward(x_in, y_in)
    local x = x_in and 1 or 0
    local y = y_in and 1 or 0
    local output = NeuralNetworkLib.forward(torch.Tensor({1, x , y}))
    return output[1] > 0.5
end

function logicGates.OR.set(x_in , y_in)
    local x = x_in and 1 or 0
    local y = y_in and 1 or 0
    NeuralNetworkLib.build({2,1})
    local layer_temp = NeuralNetworkLib.getLayer(1)  
    -- torch.Tensor({{-0.5 , 1 , 1}})
    layer_temp:sub(1,1,1,1):fill(-0.5)
    layer_temp:sub(1,1,2,2):fill(1)
    layer_temp:sub(1,1,3,3):fill(1)
    local output = NeuralNetworkLib.forward(torch.Tensor({1, x , y}))
    return output[1] > 0.5
end

function logicGates.NOT.train()
    NeuralNetworkLib.build({1,1})
    local i
    for i = 1 , 250 do
        -- 1 => 0
        NeuralNetworkLib.forward(torch.Tensor({1,1}))
        NeuralNetworkLib.backward(torch.Tensor({0}))
        NeuralNetworkLib.updateParams(0.5)
        -- 0 => 1
        NeuralNetworkLib.forward(torch.Tensor({1,0}))
        NeuralNetworkLib.backward(torch.Tensor({1}))
        NeuralNetworkLib.updateParams(0.5)
    end
end

function logicGates.NOT.forward(x_in)
    local x = x_in and 1 or 0
    local output = NeuralNetworkLib.forward(torch.Tensor({1, x}))
    return output[1] > 0.5
end

function logicGates.NOT.set(x_in)
    local x = x_in and 1 or 0
    NeuralNetworkLib.build({1,1})
    local layer_temp = NeuralNetworkLib.getLayer(1)
    -- torch.Tensor({{0.5 , -1}})
    layer_temp:sub(1,1,1,1):fill(0.5)
    layer_temp:sub(1,1,2,2):fill(-1)
    local output = NeuralNetworkLib.forward(torch.Tensor({1, x}))
    return output[1] > 0.5
end

function logicGates.XOR.train()
    NeuralNetworkLib.build({2,5,1})
    local i
    for i = 1 , 1000 do
        -- 1,1 => 0
        NeuralNetworkLib.forward(torch.Tensor({1,1,1}))
        NeuralNetworkLib.backward(torch.Tensor({0}))
        NeuralNetworkLib.updateParams(0.5)
        -- 0,0 => 0
        NeuralNetworkLib.forward(torch.Tensor({1,0,0}))
        NeuralNetworkLib.backward(torch.Tensor({0}))
        NeuralNetworkLib.updateParams(0.5)
        -- 0,1 => 1
        NeuralNetworkLib.forward(torch.Tensor({1,0,1}))
        NeuralNetworkLib.backward(torch.Tensor({1}))
        NeuralNetworkLib.updateParams(0.5)
        -- 1,0 => 1
        NeuralNetworkLib.forward(torch.Tensor({1,1,0}))
        NeuralNetworkLib.backward(torch.Tensor({1}))
        NeuralNetworkLib.updateParams(0.5)
    end
end

function logicGates.XOR.forward(x_in, y_in)
    local x = x_in and 1 or 0
    local y = y_in and 1 or 0
    local output = NeuralNetworkLib.forward(torch.Tensor({1, x , y}))
    return output[1] > 0.5
end

function logicGates.XOR.set(x_in , y_in)
    logicGates.XOR.train()
end

return logicGates
