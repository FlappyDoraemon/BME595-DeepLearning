require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'paths'

function lua_conv(x , k)
    local height , width = x:size(1) , x:size(2)
    local kernel_height , kernel_width = k:size(1) , k:size(2)
    local th_result = torch.zeros(height - kernel_height + 1 , width - kernel_width + 1)
    --  inverse to get the kernel
    local k1 = torch.zeros(kernel_height , kernel_width)
    local k2 = torch.zeros(kernel_height , kernel_width)
    local i , j
    for i = 1 , kernel_height do
        k1[{{i},{}}] = k[{{kernel_height+1-i},{}}]
    end
    for j = 1 , kernel_width do
        k2[{{},{j}}] = k1[{{},{kernel_width+1-j}}]
    end
    --  get the convolution result
    for i = 1 , height - kernel_height + 1 do
        for j = 1 , width - kernel_width + 1 do
            th_result[{{i},{j}}] = torch.sum(torch.cmul(x[{{i,i+kernel_height-1},{j,j+kernel_width-1}}],k2))
        end
    end
    return th_result
end

function C_conv(x , k)
    local height , width = x:size(1) , x:size(2)
    local kernel_height , kernel_width = k:size(1) , k:size(2)
    local c_result = torch.zeros(height - kernel_height + 1 , width - kernel_width + 1)
    local i , j
    local k1 = torch.zeros(kernel_height , kernel_width)
    --  get the kernel by inversion
    local kernel = torch.zeros(kernel_height , kernel_width)
    for i = 1 , kernel_height do
        k1[{{i},{}}] = k[{{kernel_height+1-i},{}}]
    end
    for j = 1 , kernel_width do
        kernel[{{},{j}}] = k1[{{},{kernel_width+1-j}}]
    end
    --  ffi
    ffi = require("ffi")
    mylib = ffi.load(paths.cwd() .. '/libc_convolution.so')
    ffi.cdef[[
        void c_convo(double *mylena , double * kernel , double * output , int height , int width , int kernel_height , int kernel_width)
    ]]
    mylib.c_convo(torch.data(x) , torch.data(kernel) , torch.data(c_result) , height , width , kernel_height , kernel_width)
    return c_result
end
