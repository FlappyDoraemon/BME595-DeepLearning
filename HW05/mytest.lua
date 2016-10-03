local mnist = require 'mnist'
local img2numLib = require 'img2num'
local img2objLib = require 'img2obj'
require 'nn'
require 'torch' 

--[[
require 'camera'
camera.testme()   -- a simple grabber+display
cam = image.Camera()  -- create the camera grabber
frame = cam:forward()  -- return the next frame available
cam:stop() -- release the camera
image.display(frame)  -- display frame
--]]

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

-------------------------------------------------------------------
-- PART 0 : CIFAR TEST
-------------------------------------------------------------------



for i = 1 , 1 do
    print('##',i,'##')
    timer = torch.Timer()
    timer:reset()
    img2objLib.train()
    print(timer:time().real)
    if(not paths.filep("cifar100-train.t7")) then
        os.execute('wget -c http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz')
        os.execute('tar -xvf cifar-100-binary.tar.gz')
        convertCifar100BinToTorchTensor('cifar-100-binary/train.bin', 'cifar100-train.t7')
        convertCifar100BinToTorchTensor('cifar-100-binary/test.bin', 'cifar100-test.t7')
    end
    testset = torch.load('cifar100-test.t7')
    trainset = torch.load('cifar100-train.t7')
    local i
    local mark = 0
    local lenth = testset.data:size(1)
    -- image.display{image = testset.data[1],zoom = 5}

    img2objLib.view(trainset.data[2])
    img2objLib.cam(0)

end



----------------------------------------------------------------------
-- PART 1 : MNIST TEST
----------------------------------------------------------------------

--[[
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

--]]
