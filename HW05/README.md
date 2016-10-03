# **HW5- CNN on MNIST and C100 datasets**
###Student: Kuan Han


1. Train LeNet5 on MNIST with the standard LeNet Archetecture and RelU activation train on the whole set on my laptop wth i7 6400 CPU takes around 75 seconds while the nn-package in the virtual machine takes around 410 seconds.
2. Train a architecture similar to Alexnet on C100 dataset. Train on the whole dataset takes 66 seconds on my laptop with GTX 1060 GPU
3. The architexture of the network or C100

        net = nn.Sequential()
        net:add(nn.Reshape(3, 32, 32))  
                                                                    -------------------------------------
        net:add(nn.SpatialConvolutionMM(3, 100, 5, 5, 1, 1, 0, 0))  --  #kernel = 100  edge = 32 - 5 + 1 = 28
        
        net:add(nn.ReLU())
        net:add(nn.SpatialMaxPooling(2, 2 , 2, 2, 0, 0))            -- edge = 28 / 2 = 14
                                                                    --------------------------------------
        net:add(nn.SpatialConvolutionMM(100, 200, 4, 4, 1, 1, 0, 0))  -- #channel = 100  #kernel = 200
        net:add(nn.ReLU())
                                                                    ---------------------------------------
        net:add(nn.SpatialConvolutionMM(200, 90, 3, 3, 1, 1, 0, 0)) -- #channel =200  #kernel = 90 edge =8
        net:add(nn.ReLU())
        net:add(nn.SpatialMaxPooling(2, 2 , 2, 2, 0, 0))            -- edge = 8 / 2 = 4
                                                                    -----------------------------------
        net:add(nn.Reshape(4*4*90))                                 -- reshape to fully connected
                                                                    -----------------------------------
        net:add(nn.Linear(4*4*90, 700))                             -- fully connected
        net:add(nn.ReLU()) 
                                                                    --------------------------
        net:add(nn.Linear(700, 450))                                -- fully connected
        net:add(nn.ReLU())
                                                                    --------------------------------- 
        net:add(nn.Linear(450, 100))                                -- fully connected 
        net:add(nn.LogSoftMax()) 
        
4. Train the c-100 net over the whole c-100 training set for 500 times with shuffle, we can get the training set accuracy rate of 82.312% and test set accuracy rate of 44.2%, both of which are top-1 accuracy rate.

5. c100_names.t7 is obtained from the Matlab version of mat file of the original dataset.
 
----------



