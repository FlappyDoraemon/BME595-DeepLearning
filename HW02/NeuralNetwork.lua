require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'paths'
local NeuralNetwork={}
NeuralNetwork.theta = {}

--[[
local function destroy()
    theta = {}
end
]]

function NeuralNetwork.build(x)
    local i
    local idx = #x - 1
    NeuralNetwork.theta = {}
    for i = 1 , idx do
        table.insert(NeuralNetwork.theta , torch.randn(x[i+1] , x[i]+1) / torch.sqrt(x[i]))
    end
end

function NeuralNetwork.getLayer(layer)
    return NeuralNetwork.theta[layer]
end

function NeuralNetwork.forward(input)
    local times = #(NeuralNetwork.theta)
    local dim = #(input:size())
    local for_out = input
    local i
    if dim == 1 then
        for i = 1 , times do
            if i > 1 then
                for_out = torch.cat(torch.ones(1) , for_out)
            end
            for_out = NeuralNetwork.theta[i] * for_out
            for_out = torch.cdiv(  torch.ones(NeuralNetwork.theta[i]:size(1))  ,  torch.exp(-for_out)+torch.ones(NeuralNetwork.theta[i]:size(1))  )
        end
    else
        for i = 1 , times do
            if i > 1 then
                for_out = torch.cat(torch.ones(1,for_out:size(2)) , for_out , 1)
            end
            for_out = NeuralNetwork.theta[i] * for_out
            for_out = torch.cdiv(  torch.ones(for_out:size(1),for_out:size(2))  ,  torch.exp(-for_out)+torch.ones(for_out:size(1),for_out:size(2))  )
        end
    end
    return for_out
end

return NeuralNetwork
