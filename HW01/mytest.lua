require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'gnuplot'
require 'paths'
local convjit=require 'conv_jit'
local conv=require 'conv'

local looptimes
local maxlooptime = 15
local scale_num = 6
local i
local mylena = image.lena()
local mylena_r = mylena[{1,{},{}}]
local kernel = torch.rand(9,9)
local standard_result , lua_result , jit_result , c_result
local total_record = torch.zeros(4 , maxlooptime)
local ave_record = torch.zeros(4 , maxlooptime)
local height_record = torch.zeros(4 , scale_num)
local width_record = torch.zeros(4 , scale_num)
image.display(mylena_r)


--  warm up
local warmup_iter
for warmup_iter = 1 , 2 do
    standard_result = torch.conv2(mylena_r, kernel)
    lua_result = conv.Lua_conv(mylena_r, kernel)
    jit_result = convjit.jit_conv(mylena_r, kernel)
    c_result = conv.C_conv(mylena_r, kernel)
end



timer = torch.Timer()
--  test with height scale variance
local this_height = mylena_r:size(1)
for i = 1 , scale_num do 
    local img_temp = mylena_r[{{1,this_height},{}}]
    --  lua
    timer:reset()
    lua_result = conv.Lua_conv(img_temp, kernel)
    height_record[{1,i}] = timer:time().real
    --  jit
    timer:reset()
    jit_result = convjit.jit_conv(img_temp, kernel)
    height_record[{2,i}] = timer:time().real
    --  C
    timer:reset()
    c_result = conv.C_conv(img_temp, kernel)
    height_record[{3,i}] = timer:time().real
    --  torch.conv2
    timer:reset()
    standard_result = torch.conv2(img_temp, kernel)
    height_record[{4,i}] = timer:time().real
    --  change height for next iteration
    this_height = torch.round(this_height/2)
end


--  test with width scale variance
local this_width = mylena_r:size(2)
img_temp = mylena_r
for i = 1 , scale_num do 
    local img_temp = mylena_r[{{},{1,this_width}}]
    --  lua
    timer:reset()
    lua_result = conv.Lua_conv(img_temp, kernel)
    width_record[{1,i}] = timer:time().real
    --  jit
    timer:reset()
    jit_result = convjit.jit_conv(img_temp, kernel)
    width_record[{2,i}] = timer:time().real
    --  C
    timer:reset()
    c_result = conv.C_conv(img_temp, kernel)
    width_record[{3,i}] = timer:time().real
    --  torch.conv2
    timer:reset()
    standard_result = torch.conv2(img_temp, kernel)
    width_record[{4,i}] = timer:time().real
    --  change height for next iteration
    this_width = torch.round(this_width/2)
end

--  Scale Variance plot
gnuplot.grid('on')
    --  plot height variance
gnuplot.pngfigure('heightv.png')
gnuplot.plot({'lua',height_record[{1,{}}]},{'jit',height_record[{2,{}}]},{'C-ffi',height_record[{3,{}}]},{'torch.conv2',height_record[{4,{}}]})
gnuplot.xlabel('scale down ratio in height')
gnuplot.ylabel('time consumption')
gnuplot.plotflush()
gnuplot.pngfigure('heightv2.png')
gnuplot.plot({'jit',height_record[{2,{}}]},{'C-ffi',height_record[{3,{}}]},{'torch.conv2',height_record[{4,{}}]})
gnuplot.xlabel('scale down ratio in height')
gnuplot.ylabel('time consumption')
gnuplot.plotflush()
    --  plot width variance
gnuplot.pngfigure('widthv.png')
gnuplot.plot({'lua',width_record[{1,{}}]},{'jit',width_record[{2,{}}]},{'C-ffi',width_record[{3,{}}]},{'torch.conv2',width_record[{4,{}}]})
gnuplot.xlabel('scale down ratio in width')
gnuplot.ylabel('time consumption')
gnuplot.plotflush()
gnuplot.pngfigure('widthv2.png')
gnuplot.plot({'jit',width_record[{2,{}}]},{'C-ffi',width_record[{3,{}}]},{'torch.conv2',width_record[{4,{}}]})
gnuplot.xlabel('scale down ratio in width')
gnuplot.ylabel('time consumption')
gnuplot.plotflush()

--  testing time average
local standard_result , lua_result , c_result , jit_result
local standard_time , lua_time , c_time , jit_time
for looptimes = 1, maxlooptime do
    print('--------       begin iteration',looptimes,'--------')
    --  original function test
    timer:reset()
    for i = 1 , looptimes do
        standard_result = torch.conv2(mylena_r, kernel)
    end
    standard_time = timer:time().real 
    width_record[{3,i}] = timer:time().real
    total_record[{1,looptimes}] = standard_time     
    ave_record[{1,looptimes}] = standard_time / looptimes
    --  lua test
    timer:reset()
    for i = 1 , looptimes do
        lua_result = conv.Lua_conv(mylena_r, kernel)
    end
    lua_time = timer:time().real
    total_record[{2,looptimes}] = lua_time     
    ave_record[{2,looptimes}] = lua_time / looptimes
    --  jit test
    timer:reset()
    for i = 1 , looptimes do
        jit_result = convjit.jit_conv(mylena_r, kernel)
    end
    jit_time = timer:time().real
    total_record[{3,looptimes}] = jit_time     
    ave_record[{3,looptimes}] = jit_time / looptimes 
    --  C test
    timer:reset()
    for i = 1 , looptimes do
        c_result = conv.C_conv(mylena_r, kernel)
    end    
    c_time = timer:time().real
    total_record[{4,looptimes}] = c_time     
    ave_record[{4,looptimes}] = c_time / looptimes 
end



--  Iteration Variance plot
--    plot total time 
gnuplot.pngfigure('total_4.png')
gnuplot.plot({'torch.conv2',total_record[{1,{}}]},{'lua',total_record[{2,{}}]},{'jit',total_record[{3,{}}]},{'C-ffi',total_record[{4,{}}]})
gnuplot.xlabel('iterations')
gnuplot.ylabel('total time')
gnuplot.plotflush()
--    plot total time of hand-made ones
gnuplot.pngfigure('total_3.png')
gnuplot.plot({'torch.conv2',total_record[{1,{}}]},{'jit',total_record[{3,{}}]},{'C-ffi',total_record[{4,{}}]})
gnuplot.xlabel('iterations')
gnuplot.ylabel('total time')
gnuplot.plotflush()
--    plot average time 
gnuplot.pngfigure('ave_4.png')
gnuplot.plot({'torch.conv2',ave_record[{1,{}}]},{'lua',ave_record[{2,{}}]},{'jit',ave_record[{3,{}}]},{'C-ffi',ave_record[{4,{}}]})
gnuplot.xlabel('iterations')
gnuplot.ylabel('average time')
gnuplot.plotflush()
--    plot average time of hand-made ones
gnuplot.pngfigure('ave_3.png')
gnuplot.plot({'torch.conv2',ave_record[{1,{}}]},{'jit',ave_record[{3,{}}]},{'C-ffi',ave_record[{4,{}}]})
gnuplot.xlabel('iterations')
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

print('finished')
