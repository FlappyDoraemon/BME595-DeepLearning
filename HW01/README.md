# **HW1- Accelerate a lua Function in C**
###Student: Kuan Han



A convolution operation is accelerated in C and JIT, rather than a lua handwritten version. For these 4 versions: **[1] lua**(baseline),   **[2] C**(with FFI), **[3]JIT**, and **[4]torch.conv2**(original function), comparison result shows that the *torch.conv2()* is the fastest, which is almost 90 times faster rather than the hand written version.  Besides, **C** and **JIT** are almost the same, both with ~25 acceleration ratio rather than baseline.
 
----------


>### **Comparison of Time Consumption by Scale Variance**
>
The scale of the "lena" image is reduced in single side (either height or width) while the number of pixels along another edge is fixed. The comparison result is shown below.

**Time Consumption List by the (scale down ratio) in hight**


|scale down ratio|image scale|lua|jit|C|torch.conv2|

|-----|-----|-----|-----|-----|-----|

|1|512*512|2.693 s|0.226 s|0.264 s|0.0442 s|

|2|256*512|1.332 s|0.108 s|0.130 s|0.0103 s|

|3|128*512|0.742 s|0.0556 s|0.0603 s|0.00367 s|

|4|64*512|0.510 s|0.0187 s|0.0374 s|0.00175 s| 

|5|32*512|0.2534 s|0.0138 s|0.00548 s|8.029e-4 s|

|6|16*512|0.0716 s|0.00170 s|0.00884 s|2.591e-4|

#### => **Reduce height scale of 4 approaches and compare time consumption**
![](https://purdue0-my.sharepoint.com/personal/han424_purdue_edu/_layouts/15/guestaccess.aspx?guestaccesstoken=msRJKvz5kXxt57iM%2bLob%2bfxkTQKeItvpohGtMRK%2bl%2fE%3d&docid=06c814fdc223f434e844612305aaa7339&rev=1)

To get the comparison more clearly, we remove the line of lua computation because it costs too much time.
#### => **Remove lua computation line and compare**
![](https://purdue0-my.sharepoint.com/personal/han424_purdue_edu/_layouts/15/guestaccess.aspx?guestaccesstoken=9WElFjioLVLk5bjVSkIS6a8%2fok9HjkBP3YwlsLP1qAI%3d&docid=0240765a342e5400f81260a0c4e9ec91d&rev=1)

We can see that for each approach, time consumption is almost linear to the pixel number along height when the number of pixels along width is fixed.

**Time Consumption List by the (scale down ratio) in width**
|scale down ratio|image scale|lua|jit|C|torch.conv2|
|-----|-----|-----|-----|-----|-----|
|1|512*512|2.598 s|0.236 s|0.257 s|0.0374 s|
|2|512*256|1.315 s|0.109 s|0.127 s|0.0192 s|
|3|512*128|0.751 s|0.0573 s|0.0624 s|0.00425 s|
|4|512*64|0.535 s|0.0189 s|0.0375 s|0.00210 s| 
|5|512*32|0.2492 s|0.0125 s|0.0198 s|0.00120 s|
|6|512*16|0.109 s|0.00169 s|0.00208 s|4.53e-4|

#### => **Reduce width scale of 4 approaches and compare time consumption**
![](https://purdue0-my.sharepoint.com/personal/han424_purdue_edu/_layouts/15/guestaccess.aspx?guestaccesstoken=rQvfCLxEp8qyX8VRjyOfmbt7BUy%2bGWLx3LHEOJgwRv0%3d&docid=02b716c189ad947f08a075ad6ea53466b&rev=1)
#### => **Remove lua computation line and compare**
![](https://purdue0-my.sharepoint.com/personal/han424_purdue_edu/_layouts/15/guestaccess.aspx?guestaccesstoken=fHp6PQbDZIqH5V7TMu9FOc8YcQqJ47btU91XFKxMeBM%3d&docid=0c179879ac3e542e38932fde026a8189b&rev=1)

Also the computation efficiency of jit method and C method are almost the same, while jit is a little better. Let us make further comparison against the baseline in the next section.


>### **Comparison by Iteration Variance**
>
The program computes the convolution from 1 to 15 times respectively to reduce random and record both the total time consumption and the average.  
#### => **Total time consumption for 4 different ways**
![](https://purdue0-my.sharepoint.com/personal/han424_purdue_edu/_layouts/15/guestaccess.aspx?guestaccesstoken=IQre3H%2fzWTpnSGjC56KsAfD3B0hKNV%2fTQA%2b4VXheViA%3d&docid=0e8ba33e2cf8142448e5788381c488c11&rev=1)

Time consumed is linear to iteration times for each approach. Since Lua manually described approach consumes too much time and we can not view detail of others in a same view, let's remove lua line and compare.
#### => Total time consumption of 3 different ways except the baseline
![](https://purdue0-my.sharepoint.com/personal/han424_purdue_edu/_layouts/15/guestaccess.aspx?guestaccesstoken=tBwseDvHYvW%2fMwl4rZgYoZ53GgyaDQbswLDcKybAXv0%3d&docid=05ac9e073d5ce48eba7441acfcec92afe&rev=1)

Apart from very little disturbance, time consumption is linear  to iterations, which means for each time the convolution takes the same time.
#### => Acceleration ratio of 3 advanced approaches against baseline
We might be interested in the acceleration ratio of 1) torch.conv2, 2) C-ffi and 3)jit against the baseline (manually described lua). So we list the comparison result here. The result is averaged by iteration times to reduce random.
![](https://purdue0-my.sharepoint.com/personal/han424_purdue_edu/_layouts/15/guestaccess.aspx?guestaccesstoken=tEpmpb2USupUJIkAPSIMLQEj%2fGaMCqMutWUQiHbCo7M%3d&docid=009c6e5d148834d2996d8f62a89894b5b&rev=1)

The result shows that the original function torch.conv2 is the most optimized one. Jit and C both achieve a acceleration ratio of ~20 against baseline method (lua description), while jit is a little better than C-ffi.


----------
