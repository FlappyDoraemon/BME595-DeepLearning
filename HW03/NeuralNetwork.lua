require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'paths'
local NeuralNetwork={}
NeuralNetwork.theta = {}
NeuralNetwork.de_dtheta = {}
--  decided every forward
--  local input_local_mul           -- remain the iuput to calculate the gradient of the first layer
local output_local_mul          -- remain the output to calculate gradient
local process_record = {}       -- input + hidden and for each layer, use format [1 h1 h2 ...], 1 to calculate w_0; as for output, use the output_local_mul; also former
--  decided every  build
local network_depth             -- (layers - 1) for loop 
local former_layer_neu_num      -- neuron number for every layer (except bias)


--[[
local function destroy()
    theta = {}
end
]]

function NeuralNetwork.build(x)
    local i
    network_depth = #x - 1
    NeuralNetwork.theta = {}
    former_layer_neu_num = torch.zeros(network_depth+1)
    for i = 1 , network_depth do
        table.insert(NeuralNetwork.theta , torch.randn(x[i+1] , x[i]+1) / torch.sqrt(x[i]))
        former_layer_neu_num[{i}] = x[i]+1
    end
    former_layer_neu_num[{network_depth+1}] = x[network_depth+1]+1
--  print(NeuralNetwork.theta[1])
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
        local for_out2 = torch.ones(for_out:size(1),2)
        for_out2[{{},{1}}] = for_out
        for_out2[{{},{2}}] = for_out
        for i = 1 , times do
            if i > 1 then
                for_out2 = torch.cat(torch.ones(1,for_out2:size(2)) , for_out2 , 1)
                -- satisfaction
                for_out = torch.cat(torch.ones(1) , for_out)
            end
            process_record[i] = for_out2   -- for the first layer there already added a "1" in front of other values
            for_out2 = NeuralNetwork.theta[i] * for_out2
            for_out2 = torch.cdiv(  torch.ones(for_out2:size(1),for_out2:size(2))  ,  torch.exp(-for_out2)+torch.ones(for_out2:size(1),for_out2:size(2))  )
            -- satisfaction
            for_out = NeuralNetwork.theta[i] * for_out
            for_out = torch.cdiv(  torch.ones(NeuralNetwork.theta[i]:size(1))  ,  torch.exp(-for_out)+torch.ones(NeuralNetwork.theta[i]:size(1))  )
        end
        output_local_mul = for_out  -- could possible has 2 dims or 1 dims, here 1
--print('in FORWARD, output_local_mul = \n',output_local_mul)
--print('in FORWARD, process_record[1] = \n',process_record[1])
    else
        for i = 1 , times do
            if i > 1 then
                for_out = torch.cat(torch.ones(1,for_out:size(2)) , for_out , 1)
            end
            process_record[i] = for_out   -- for the first layer there already added a "1" in front of other values
            for_out = NeuralNetwork.theta[i] * for_out
            for_out = torch.cdiv(  torch.ones(for_out:size(1),for_out:size(2))  ,  torch.exp(-for_out)+torch.ones(for_out:size(1),for_out:size(2))  )
        end
        output_local_mul = for_out  -- could possible has 2 dims or 1 dims,here 2
    end
    return for_out
end

local function backward_init(target, output, num , totalnum)
--print('in BP_INIT Begin, output_local_mul = \n',output_local_mul)
--print('in BP_INIT Begin, process_record[1] = \n',process_record[1])
    local dneuout_dneuin = {}                 -- network_depth , layer_neu_num[{i}] , calculated results
    local de_dtheta_local = {}                -- network_depth , NeuralNetwork.theta[i] , layer_neu_num[{idx1+1}] * (layer_neu_num[{idx1} + 1]) result
    local idx1 , idx2 , idx3 , idx4
    -- initialization
    for idx1 = 1 , network_depth do
	dneuout_dneuin[idx1] = torch.zeros(NeuralNetwork.theta[idx1]:size(1))
        de_dtheta_local[idx1] = NeuralNetwork.theta[idx1]:clone()  -- do not care value inside
    end
    for idx1 = 1 , network_depth do
        idx2 = network_depth + 1 - idx1
	if idx2 == network_depth then   -- (network_depth + 1 - idx1) == network_depth, which means the output layer
        -- idx1, idx2: layers
        -- idx3      : layer_neu_num
        -- idx4      : former_layer_num 
            for idx3 = 1 , target:size(1) do
                dneuout_dneuin[idx2][{idx3}] = (output[{idx3}] - target[{idx3}]) * output[{idx3}] * (1 - output[{idx3}]) * 2 / target:size(1)             
            end
--  print('dneuout_dneuin at layer',idx2,'/',network_depth,'has value :\n',dneuout_dneuin[idx2])
            for idx3 = 1 , target:size(1) do          
                for idx4 = 1 , NeuralNetwork.theta[idx2]:size(2) do
--  print(num, process_record[idx2][{idx4,num}])
                    de_dtheta_local[idx2][{idx3,idx4}] = dneuout_dneuin[idx2][{idx3}] * process_record[idx2][{idx4,num}]
                end
            end
--  print('de_dtheta_local[',idx2,']\n',de_dtheta_local[idx2])
        else 
        -- idx1, idx2: layers
        -- idx3      : last_layer_neu_num
        -- idx4      : last_last_layer_num 
            for idx3 = 1 , NeuralNetwork.theta[idx2]:size(1) do  -- former_layer_neu_num contains bias
            -- former_layer_neu_num[{idx2+1}]-1 means there is no bias, and also not "for 2 , former_layer_neu_num[{idx2+1}]"
                -- calculate dneuout_dneuin[idx2][{idx3}]
                dneuout_dneuin[idx2][{idx3}] = 0
                -- add all the relations to next layer neuron's derivatives
                for idx4 = 1 , NeuralNetwork.theta[idx2 + 1]:size(1) do
                    dneuout_dneuin[idx2][{idx3}] = dneuout_dneuin[idx2][{idx3}] + NeuralNetwork.theta[idx2+1][{idx4,1+idx3}] * dneuout_dneuin[idx2+1][{idx4}]
                end
                dneuout_dneuin[idx2][{idx3}] = dneuout_dneuin[idx2][{idx3}] * process_record[idx2+1][{idx3+1,num}] * (1 - process_record[idx2+1][{idx3+1,num}])   
            end
--  print('dneuout_dneuin at layer',idx2,'/',network_depth,'has value :\n',dneuout_dneuin[idx2])
            for idx3 = 1 , NeuralNetwork.theta[idx2]:size(1) do
                for idx4 = 1 , NeuralNetwork.theta[idx2]:size(2) do
                    de_dtheta_local[idx2][{idx3,idx4}] = dneuout_dneuin[idx2][{idx3}] * process_record[idx2][{idx4,num}]
                end  
            end
--  print('de_dtheta_local[',idx2,']\n',de_dtheta_local[idx2])
        end
    end
    local de_dtheta_output = {}
    for idx1 = 1 , network_depth do
        de_dtheta_output[idx1] = de_dtheta_local[idx1] / totalnum
    end
    return de_dtheta_output
end

function NeuralNetwork.backward(target)
    local dim = #(target:size())
    local idx
    for idx = 1 , network_depth do
        NeuralNetwork.de_dtheta[idx] = torch.zeros(NeuralNetwork.theta[idx]:size(1) , NeuralNetwork.theta[idx]:size(2))
    end
--  print('before bp, theta0 is: \n', NeuralNetwork.theta[1])
--  print('before bp, de-dtheta0 is: \n', NeuralNetwork.de_dtheta[1])
    if dim == 1 then
        local output_here = output_local_mul
--  print(target)
--  print(output_here)
        local idx
        local de_dtheta_output = backward_init(target, output_here, 1 , 2)
        for idx = 1 , network_depth do
            NeuralNetwork.de_dtheta[idx] = NeuralNetwork.de_dtheta[idx] + de_dtheta_output[idx]
        end 
--  print('after bp, de-de_dtheta_output1 is: \n', de_dtheta_output[1]) 
        de_dtheta_output = backward_init(target, output_here, 2 , 2)
        for idx = 1 , network_depth do
            NeuralNetwork.de_dtheta[idx] = NeuralNetwork.de_dtheta[idx] + de_dtheta_output[idx]
        end 
--  print('after bp, de-de_dtheta_output2 is: \n', de_dtheta_output[1]) 
--  print('after bp, theta is: \n', NeuralNetwork.theta[1])
--  print('after bp, de-dtheta is: \n', NeuralNetwork.de_dtheta[1])        
    else  -- dim of output  >= 2
        local num = target:size(2) 
        local index
        local j 
        for index = 1 , num do
            local output_here = output_local_mul[{{},index}]
            local target_here = target[{{},index}]
            local de_dtheta_output = backward_init(target_here, output_here, index , num)
            for j = 1 , network_depth do
                NeuralNetwork.de_dtheta[j] = NeuralNetwork.de_dtheta[j] + de_dtheta_output[j]
            end
        end
    end
end

function NeuralNetwork.updateParams(etha)
    local idx
--  print('before update')
--  print('network_depth=',network_depth)
--  print('theta: \n',NeuralNetwork.theta[1])
--  print('NeuralNetwork.de_dtheta: \n', NeuralNetwork.de_dtheta[1])
    for idx = 1 , network_depth do
        NeuralNetwork.theta[idx] = NeuralNetwork.theta[idx] - NeuralNetwork.de_dtheta[idx] * etha
    end 
--  print('after update')
--print('de_dtheta[1]: \n',NeuralNetwork.de_dtheta[1])
--print('de_dtheta[2]: \n',NeuralNetwork.de_dtheta[2])
--print('theta[1]: \n',NeuralNetwork.theta[1])
--print('theta[2]: \n',NeuralNetwork.theta[2])
end

return NeuralNetwork
