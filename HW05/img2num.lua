require 'nn'
local mnist = require 'mnist'
local img2num={}
local net

--  local ex = testset[1]
--  local x = ex.x -- the input (a 28x28 ByteTensor)
--  local y = ex.y -- the label (0--9)
--  x = nn.Reshape(784):forward(x:double())

function img2num.train()
    if(not paths.filep("lenet_model.t7")) then
        net = nn.Sequential()
        net:add(nn.Reshape(1, 28, 28))  
        net:add(nn.SpatialConvolution(1, 6, 5, 5, 1, 1, 0, 0))   --  #kernel = 6
        net:add(nn.Tanh())
        net:add(nn.SpatialMaxPooling(2, 2 , 2, 2, 0, 0))  
        net:add(nn.SpatialConvolution(6, 16, 5, 5, 1, 1, 0, 0))  -- #channel = 6  #kernel = 16
        net:add(nn.Tanh())
        net:add(nn.SpatialMaxPooling(2, 2 , 2, 2, 0, 0))  
        net:add(nn.Reshape(4*4*16))  
        net:add(nn.Linear(4*4*16, 120)) 
        net:add(nn.Tanh()) 
        net:add(nn.Linear(120, 84))
        net:add(nn.Tanh()) 
        net:add(nn.Linear(84, 10)) 
        -- net:add(nn.Tanh())
        net:add(nn.LogSoftMax()) 
--[[
        net:add(nn.Reshape(784))
        net:add(nn.Linear(784,500))
        net:add(nn.Sigmoid())
        net:add(nn.Linear(500,10))
        net:add(nn.Sigmoid())
--]]

    else
	local  lenet_checkpoint=torch.load('lenet_model.t7')
	net=lenet_checkpoint
    end 
    local criterion = nn.MSECriterion()
    local trainset = mnist.traindataset()
    print('train_size=',trainset.size)
    local i
    local j
    local s 
    for s = 1 , 1 do 
        j = 0
        local seq = torch.randperm(trainset.size)
        for i = 1,trainset.size do --trainset.size do
            local ex = trainset[seq[i]]
            local input = ex.x:double() -- the input (a 28x28 ByteTensor)
            input = input[{{},{}}] / 255.0
            -- local input = ex.x:double()
            local y = ex.y -- the label (0--9)
            local output = torch.zeros(10)
            output[{y+1}] = 1       --  0=>1,1=>2,2=>3,3=>4,4=>5
            -- local output = torch.ones(1)
            -- output[1] = y + 1
            criterion:forward(net:forward(input), output)
            net:zeroGradParameters()
            net:backward(input, criterion:backward(net.output, output)) -- net.output
            net:updateParameters(0.03)
            if i*100 >= j*trainset.size then
                print('the', s, 'th shuffled turn, completed',j,'% already')
                j=j+1
            end
        end
    end
    torch.save('lenet_model.t7',net)
end

function img2num.forward(img)
    -- print(img:size())
    local img1 = img:double()
    local img2 = img1[{{},{}}]/255.0
    -- local x = nn.Reshape(784):forward(img2)
    local output = net:forward(img2)
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
