require 'nn'
require 'cunn'
require 'image'
local img2obj={}
require 'optim'
local net

--  local ex = testset[1]
--  local x = ex.x -- the input (a 28x28 ByteTensor)
--  local y = ex.y -- the label (0--9)
--  x = nn.Reshape(784):forward(x:double())

local mean = {} -- store the mean, test set use this value too
local stdv  = {} -- store the standard-deviation, test set use this value too


local function convertCifar100BinToTorchTensor(inputFname, outputFname)
    local m=torch.DiskFile(inputFname, 'r'):binary()
    m:seekEnd()
    local length = m:position() - 1
    local nSamples = length / 3074 -- 1 coarse-label byte, 1 fine-label byte, 3072 pixel bytes

    assert(nSamples == math.floor(nSamples), 'expecting numSamples to be an exact integer')
    m:seek(1)

    local coarse = torch.ByteTensor(nSamples)
    local fine = torch.ByteTensor(nSamples)
    local data = torch.ByteTensor(nSamples, 3, 32, 32)
    for i=1,nSamples do
        coarse[i] = m:readByte()
        fine[i]   = m:readByte()
        local store = m:readByte(3072)
        data[i]:copy(torch.ByteTensor(store))
    end

    local out = {}
    out.data = data
    out.label = fine
    out.labelCoarse = coarse
    print(out)
    torch.save(outputFname, out)
end

function img2obj.train()
    
    -- loading the dataset file

    if(not paths.filep("cifar100-train.t7")) then
        os.execute('wget -c http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz')
        os.execute('tar -xvf cifar-100-binary.tar.gz')
        convertCifar100BinToTorchTensor('cifar-100-binary/train.bin', 'cifar100-train.t7')
        convertCifar100BinToTorchTensor('cifar-100-binary/test.bin', 'cifar100-test.t7')
    end
    c100 = torch.load('cifar100-train.t7')
    -- image.display{image=c100.data[{{1},{},{},{}}]}
    -- train the date set
    trainset = torch.load('cifar100-train.t7')
    print(trainset.data:size(3))
    setmetatable(trainset, 
        {__index = function(t, i) 
                       return {t.data[i], t.label[i]} 
                   end}
    );
    trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.

    -- getting the standard input 

    
    -- define the network and the training settings
    if(not paths.filep("c100_model.t7")) then
        net = nn.Sequential()
        net:add(nn.Reshape(3, 32, 32))  
                                                                    ------------------------------------------------------
        net:add(nn.SpatialConvolutionMM(3, 100, 5, 5, 1, 1, 0, 0))   --  #kernel = 78  edge = 32 - 5 + 1 = 28
        net:add(nn.ReLU())
        net:add(nn.SpatialMaxPooling(2, 2 , 2, 2, 0, 0))            -- edge = 28 / 2 = 14
                                                                    ------------------------------------------------------
        net:add(nn.SpatialConvolutionMM(100, 200, 4, 4, 1, 1, 0, 0))  -- #channel = 78  #kernel = 178 edge = 14 - 5 + 1 = 10
        net:add(nn.ReLU())
                                                                    ------------------------------------------------------
        net:add(nn.SpatialConvolutionMM(200, 90, 3, 3, 1, 1, 0, 0))  -- #channel = 178  #kernel = 90 edge = 10 - 3 + 1 = 8
        net:add(nn.ReLU())
        net:add(nn.SpatialMaxPooling(2, 2 , 2, 2, 0, 0))            -- edge = 8 / 2 = 4
                                                                    ------------------------------------------------------
        net:add(nn.Reshape(4*4*90))                                 -- reshape to fully connected
                                                                    ------------------------------------------------------
        net:add(nn.Linear(4*4*90, 700))                             -- fully connected
        net:add(nn.ReLU()) 
                                                                    ------------------------------------------------------
        net:add(nn.Linear(700, 450))                                -- fully connected
        net:add(nn.ReLU())
                                                                    ------------------------------------------------------ 
        net:add(nn.Linear(450, 100))                                -- fully connected 
        net:add(nn.LogSoftMax()) 
    else 	
        local lenet_checkpoint=torch.load('c100_model.t7')
	net=lenet_checkpoint
    end 

    -- send the data to cuda
    criterion = nn.MSECriterion()
    net = net:cuda()
    criterion = criterion:cuda()
    local trainset_cuda_pre = trainset.data:float()/255.0
    local trainset_inputcuda = trainset_cuda_pre:cuda()
    local i
    local index_lenth = trainset.data:size(1)
    local output = torch.zeros(index_lenth , 100)
    for i = 1 , index_lenth do
        local y = trainset.label[i] -- the label (0--99?)
        output[{{i},{y+1}}] = 1.0
    end
    trainset_labelcuda = output:cuda()
        
    -- trainse_labelcuda = trainset.label:cuda()
    --[[
    trainer = nn.StochasticGradient(net, criterion)
    trainer.learningRate = 0.001
    trainer.maxIteration = 5 -- just do 5 epochs of training.
    trainer:train(tra bad argument #1 to 'set' (expecting number or Tensor or Storage)
stack traceback:
inset)
    testset = torch.load('cifar10-test.t7')
    testset.data = testset.data:double()   -- convert from Byte tensor to Double tensor
    for i=1,3 do -- over each image channel
        testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
        testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
    end
    --]]
    print(index_lenth , trainset.data:size(2),trainset.data:size(3),trainset.data:size(4),trainset.label[2] , trainset.labelCoarse[2]) 
    print('--------------------------------')
    for s = 1 , 10 do 
        print(s)
        local j = 0
        local seq = torch.randperm(index_lenth)            -- shuffle
        for i = 1,index_lenth do
            local input = trainset_inputcuda[seq[i]]
            --  local input = x:float() -- the input (a 32x32 ByteTensor)
            --  input = input[{{},{}}] / 255.0
            --  local input = ex.x:double()
            --  local y = trainse_labelcuda[seq[i]] -- the label (0--99?)
            --  local output = torch.zeros(100)
            --  output[{y+1}] = 1       --  0=>1,1=>2,2=>3,3=>4,4=>5
            criterion:forward(net:forward(input), trainset_labelcuda[{{seq[i]},{}}])
            net:zeroGradParameters()
            net:backward(input, criterion:backward(net.output, trainset_labelcuda[{{seq[i]},{}}])) -- net.output
            net:updateParameters(0.03)
            --if i*100 >= j*index_lenth then
            --    print('the', s, 'th shuffled turn, completed',j,'% already')
            --    j=j+1
            --end
        end
    end
    torch.save('c100_model.t7',net)
end

function img2obj.forwardnum(img)
    -- local x = nn.Reshape(3*32*32):forward(img:double())
    local input = img:float()/255.0
    local output = net:forward(input:cuda())
    local currentmax = output[{1}]
    local i
    local outputnum = 0
    for i = 1 , 99 do
        if output[{i+1}] > currentmax then
            outputnum = i
            currentmax = output[{i+1}]
        end
    end
    return outputnum
end

function img2obj.forward(img)
    -- local x = nn.Reshape(3*32*32):forward(img:double())    
    local input = img:float()/255.0
    local output = net:forward(input:cuda())
    local currentmax = output[{1}]
    local i
    local outputnum = 0
    for i = 1 , 99 do
        if output[{i+1}] > currentmax then
            outputnum = i
            currentmax = output[{i+1}]
        end
    end
    local names = torch.load('c100_names.t7')
    return names.fine_label_names[outputnum]
end

return img2obj


