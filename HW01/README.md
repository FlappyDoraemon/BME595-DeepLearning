# **HW1- Accelerate a lua Function in C**
###Student: Kuan Han



A convolution operation is accelerated in C and JIT, rather than a lua handwritten version. For these 4 versions: **[1] lua**(baseline),   **[2] C**(with FFI), **[3]JIT**, and **[4]torch.conv2**(original function), comparison result shows that the *torch.conv2()* is the fastest, which is almost 90 times faster rather than the hand written version.  Besides, **C** and **JIT** are almost the same, both with ~25 acceleration ratio rather than baseline.
 
----------


>###**Comparison of Time Consumption by Scale Variance**
>
The scale of the "lena" image is reduced in single side (either height or width) while the number of pixels along another edge is fixed. The comparison result is shown below.
#### <i class="icon-file"></i> **Reduce height scale of 4 approaches and compare time consumption**
![](https://purdue0-my.sharepoint.com/personal/han424_purdue_edu/_layouts/15/guestaccess.aspx?guestaccesstoken=5sAM0uI9BoDftG7zM3U3XuoFuZMxqa%2byuO8bfWbdreg%3d&docid=008600439a14c4044ac9c3efbbc864a18&rev=1)

To get the comparison more clearly, we remove the line of lua computation because it costs too much time.
#### <i class="icon-file"></i> **Remove lua computation line and compare**
![](https://purdue0-my.sharepoint.com/personal/han424_purdue_edu/_layouts/15/guestaccess.aspx?guestaccesstoken=lhQxywsQgx9vuNy4PcJ60TfN5n2V%2bvWz7CGwr7a4Xu8%3d&docid=01f48e15a361a47c0ad734a4b02f4563a&rev=1)

We can see that for each approach, time consumption is almost linear to the pixel number along height when the number of pixels along width is fixed.

#### <i class="icon-file"></i> **Reduce width scale of 4 approaches and compare time consumption**
![](https://purdue0-my.sharepoint.com/personal/han424_purdue_edu/_layouts/15/guestaccess.aspx?guestaccesstoken=CYaQuvs194ouEdgons2HqBxdRBK2CfYwmwH6zRAGz3U%3d&docid=0ab16a1852c43462b9bbf842c50e5007f&rev=1)
#### <i class="icon-file"></i> **Remove lua computation line and compare**
![](https://purdue0-my.sharepoint.com/personal/han424_purdue_edu/_layouts/15/guestaccess.aspx?guestaccesstoken=oiLzznnTw8f8DeDFc7BA8AC%2fwAYM4SDBgFj58W6XtNI%3d&docid=04dd046fcd2cf4f89ac7f3287c00d9556&rev=1)
Also the computation efficiency of jit method and C method are almost the same, while jit is a little better. Let us make further comparison against the baseline in the next section.


>###**Comparison by Iteration Variance**
>
The program computes the convolution from 1 to 15 times respectively to reduce random and record both the total time consumption and the average.  
#### <i class="icon-file"></i> **Total time consumption for 4 different ways**
![](https://purdue0-my.sharepoint.com/personal/han424_purdue_edu/Documents/Semester_1_1/BME595/week_0/HW1/total_4.png)
Time consumed is linear to iteration times for each approach. Since Lua manually described approach consumes too much time and we can not view detail of others in a same view, let's remove lua line and compare.
#### <i class="icon-folder-open"></i> Total time consumption of s different ways except the baseline
![](https://purdue0-my.sharepoint.com/personal/han424_purdue_edu/_layouts/15/guestaccess.aspx?guestaccesstoken=tBwseDvHYvW%2fMwl4rZgYoZ53GgyaDQbswLDcKybAXv0%3d&docid=05ac9e073d5ce48eba7441acfcec92afe&rev=1)
Apart from very little disturbance, time consumption is linear  to iterations, which means for each time the convolution takes the same time.
#### <i class="icon-folder-open"></i> Acceleration ratio of 3 advanced approaches against baseline
We might be interested in the acceleration ratio of 1) torch.conv2, 2) C-ffi and 3)jit against the baseline (manually described lua). So we list the comparison result here. The result is averaged by iteration times to reduce random.
![](https://purdue0-my.sharepoint.com/personal/han424_purdue_edu/_layouts/15/guestaccess.aspx?guestaccesstoken=tEpmpb2USupUJIkAPSIMLQEj%2fGaMCqMutWUQiHbCo7M%3d&docid=009c6e5d148834d2996d8f62a89894b5b&rev=1)
The result shous that the original function torch.conv2 is very optimized. jit and C both achieve a acceleration ratio of ~20 against baseline method (lua description), while jit is a little better than C-ffi.


----------
