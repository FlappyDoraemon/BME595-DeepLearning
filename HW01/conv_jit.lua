require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'paths'
local conv_jit={}

function conv_jit.jit_conv(x , k)
    local height , width = x:size(1) , x:size(2)
    local kernel_height , kernel_width = k:size(1) , k:size(2)
    local th_result = torch.zeros(height - kernel_height + 1 , width - kernel_width + 1)
    local k2 = torch.zeros(kernel_height , kernel_width)
    -- accessing raw data
    x_data = torch.data(x)
    k_data = torch.data(k)
    k2_data = torch.data(k2)
    th_result_data = torch.data(th_result)
    local i , j , inneri , innerj
    --[[ 
        calculate inversion to get the kernel    
        k2(i , j) = k(kernel_height + 1 - i , kernel_width + 1 - j) 
    ]]
    for i = 0 , kernel_height-1 do
        for j = 0 , kernel_width-1 do
            k2_data[i*kernel_width+j] = k_data[(kernel_height - 1 -i) * kernel_width + (kernel_width - 1 - j)]
        end
    end
    local output_width = width - kernel_width + 1
    -- convolution 
    for i = 0 , height - kernel_height do
        for j = 0 , width - kernel_width do
            --[[
                reference from this
                th_result[{{i},{j}}] = torch.sum(torch.cmul(x[{{i,i+kernel_height-1},{j,j+kernel_width-1}}],k2))
            ]]
            th_result_data[i*output_width + j] = 0
            for inneri = 0 , kernel_height-1 do
                for innerj = 0 , kernel_width-1 do
                    --[[
                        in C: out(i,j) += x(i+inneri , j+innerj) * k(inneri , innerj)
                        kernel is already inversed
                    ]]
                    th_result_data[i*output_width + j] = th_result_data[i*output_width + j] + x_data[(i+inneri) * width + (j+innerj)] * k2_data[inneri*kernel_width + innerj]
                end
            end 
        end
    end
    return th_result
end

return conv_jit
