require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'gnuplot'
require 'paths'
require 'conv'
require 'conv_jit'

local looptimes
local maxlooptime = 15
local mylena = image.lena()
local mylena_r = mylena[{1,{},{}}]
local kernel = torch.rand(9,9)
local standard_result , lua_result , jit_result , c_result
local total_record = torch.zeros(4 , maxlooptime)
local ave_record = torch.zeros(4 , maxlooptime)
image.display(mylena_r)

standard_result = torch.conv2(mylena_r, kernel)
--[[
--  warm up
local warmup_iter
for warmup_iter = 1 , 3 do
    standard_result = torch.conv2(mylena_r, kernel)
    lua_result = lua_conv(mylena_r, kernel)
    jit_result = jit_conv(mylena_r, kernel)
    c_result = C_conv(mylena_r, kernel)
end
]]

--  testing now
local standard_result , lua_result , c_result , jit_result
local standard_time , lua_time , c_time , jit_time
timer = torch.Timer()
timer:reset()
for looptimes = 1, maxlooptime do
    print('--------       begin iteration',looptimes,'--------')
    --  original function test
    timer:reset()
    for i = 1 , looptimes do
        standard_result = torch.conv2(mylena_r, kernel)
    end
    standard_time = timer:time().real 
    total_record[{1,looptimes}] = standard_time     
    ave_record[{1,looptimes}] = standard_time / looptimes
    --  lua test
    timer:reset()
    for i = 1 , looptimes do
        lua_result = lua_conv(mylena_r, kernel)
    end
    lua_time = timer:time().real
    total_record[{2,looptimes}] = lua_time     
    ave_record[{2,looptimes}] = lua_time / looptimes
    --  jit test
    timer:reset()
    for i = 1 , looptimes do
        jit_result = jit_conv(mylena_r, kernel)
    end
    jit_time = timer:time().real
    total_record[{3,looptimes}] = jit_time     
    ave_record[{3,looptimes}] = jit_time / looptimes 
    --  C test
    timer:reset()
    for i = 1 , looptimes do
        c_result = C_conv(mylena_r, kernel)
    end    
    c_time = timer:time().real
    total_record[{4,looptimes}] = c_time     
    ave_record[{4,looptimes}] = c_time / looptimes 
end

--  plot
--    plot total time 
gnuplot.pngfigure('total_4.png')
gnuplot.plot({'torch.conv2',total_record[{1,{}}]},{'lua',total_record[{2,{}}]},{'jit',total_record[{3,{}}]},{'C-ffi',total_record[{4,{}}]})
gnuplot.xlabel('seconds')
gnuplot.ylabel('total time')
gnuplot.plotflush()
--    plot total time of hand-made ones
gnuplot.pngfigure('total_3.png')
gnuplot.plot({'torch.conv2',total_record[{1,{}}]},{'jit',total_record[{3,{}}]},{'C-ffi',total_record[{4,{}}]})
gnuplot.xlabel('seconds')
gnuplot.ylabel('total time')
gnuplot.plotflush()
--    plot average time 
gnuplot.pngfigure('ave_4.png')
gnuplot.plot({'torch.conv2',ave_record[{1,{}}]},{'lua',ave_record[{2,{}}]},{'jit',ave_record[{3,{}}]},{'C-ffi',ave_record[{4,{}}]})
gnuplot.xlabel('seconds')
gnuplot.ylabel('average time')
gnuplot.plotflush()
--    plot average time of hand-made ones
gnuplot.pngfigure('ave_3.png')
gnuplot.plot({'torch.conv2',ave_record[{1,{}}]},{'jit',ave_record[{3,{}}]},{'C-ffi',ave_record[{4,{}}]})
gnuplot.xlabel('seconds')
gnuplot.ylabel('average time')
gnuplot.plotflush()
--    plot average acceleration ratio
local acc_ratio_con2 = torch.cdiv(ave_record[{2,{}}] , ave_record[{1,{}}])
local acc_ratio_jit = torch.cdiv(ave_record[{2,{}}] , ave_record[{3,{}}])
local acc_ratio_Cffi = torch.cdiv(ave_record[{2,{}}] , ave_record[{4,{}}])
gnuplot.pngfigure('ave_acc_ratio.png')
gnuplot.plot({'torch.conv2',acc_ratio_con2},{'jit',acc_ratio_jit},{'C-ffi',acc_ratio_Cffi})
gnuplot.xlabel('ratio')
gnuplot.ylabel('acceleration ratio against lua-implementation')
gnuplot.plotflush()
