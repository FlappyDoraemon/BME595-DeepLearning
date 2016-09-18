# **HW3- Neural Networks Backward Propagation Pass**
###Student: Kuan Han



A backward propagation pass was designed to update the parameters in the neural netwwork. The scheme was evaluated to be correct by following ways: 1) check the result with the 2-2-2 online demo case (https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/) and every parameter was checked for the first update iteration; 2) compare the trained parameters with the setted parameters for logic functions, and also classification result are excatly the same; 3) test the result on MNIST dataset
 
----------


>### **Scheme of the BP Implementation**
>
Every iteration the program calculate the gradient for all parameters from back to front of the network. Suppose that the network has n+1 layers, we calculate the neuron and weight's gradient with following sequence:

(n+1)th layer neuron => weights between nth and (n+1)th layer => nth layer neuron => 

weights between (n-1)th and nth layer => (n-1)th neuron => ... => weights between 2nd and 1st layer



>### **Comparison between han-crafted Θ and trained Θ**
AND Logic:

Trained Network has 3 layers, structure is: 2 - 2 - 1

|theta 1|bias|a1|a2|
|-----|-----|-----|-----|
|h1|512*512|2.693 s|0.226 s|
|h2|256*512|1.332 s|0.108 s|

--

|theta 2|bias|h1|h2|
|-----|-----|-----|-----|
|o|512*512|2.693 s|0.226 s|
--
Set logic has two layers: 2 - 1

|theta |bias|h1|h2|
|-----|-----|-----|-----|
|o|512*512|2.693 s|0.226 s|


>### **MNIST**

Train 100 examples in train set and get the test accuracy rate of 42%.

Train all of the examples in train set and get the test accuracy rate of .
