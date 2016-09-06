local NeuralNetworkLib = require 'NeuralNetwork'
local logicGates = {}

function logicGates.AND(x_in , y_in)
    local x = x_in and 1 or 0
    local y = y_in and 1 or 0
    NeuralNetworkLib.build({2,1})
    local temp = NeuralNetworkLib.getLayer(1)  
    -- torch.Tensor({{-1.5 , 1 , 1}})
    temp:sub(1,1,1,1):fill(-1.5)
    temp:sub(1,1,2,2):fill(1)
    temp:sub(1,1,3,3):fill(1)
    local output = NeuralNetworkLib.forward(torch.Tensor({1, x , y}))
    return output[1] > 0.5
end

function logicGates.OR(x_in , y_in)
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

function logicGates.NOT(x_in)
    local x = x_in and 1 or 0
    NeuralNetworkLib.build({1,1})
    local layer_temp = NeuralNetworkLib.getLayer(1)
    -- torch.Tensor({{0.5 , -1}})
    layer_temp:sub(1,1,1,1):fill(0.5)
    layer_temp:sub(1,1,2,2):fill(-1)
    local output = NeuralNetworkLib.forward(torch.Tensor({1, x , y}))
    return output[1] > 0.5
end

function logicGates.XOR(x , y)
    local item1 = logicGates.AND(logicGates.NOT(x) , y)
    local item2 = logicGates.AND(x , logicGates.NOT(y))
    local output = logicGates.OR(item1 , item2)
    return output
end

return logicGates
