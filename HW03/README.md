# **HW3- Neural Networks Backward Propagation Pass**
###Student: Kuan Han



A backward propagation pass was designed to update the parameters in the neural netwwork. The scheme was evaluated to be correct by following ways: 1) check the result with the 2-2-2 online demo case (https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/) and every parameter was checked for the first update iteration; 2) compare the trained parameters with the setted parameters for logic functions, and also classification result are excatly the same; 3) test the result on MNIST dataset
 
----------


>### **Scheme of the BP Implementation**

(Train 250 times and LR = 0.5)

Every iteration the program calculate the gradient for all parameters from top to down of the network. Suppose that the network has n+1 layers of neurons & n layers of weights matrix, we calculate the neuron and weight's gradient with following sequence:

(n+1)th layer neuron => weights between nth and (n+1)th layer => nth layer neuron => 

weights between (n-1)th and nth layer => (n-1)th neuron => ... => weights between 2nd and 1st layer



>### **Comparison between hand-crafted Θ and trained Θ**
#### AND Logic:

##### Trained Network has 3 layers, structure is: 2 - 2 - 1

##### Trained W1 of size 2*3

|theta 1|bias|a1|a2|
|-----|-----|-----|-----|
|h1|-0.6047|0.6412|1.7730|
|h2|3.0369|-2.6292|-2.0625|

##### Trained W2 of size 1*3

|theta 2|bias|h1|h2|
|-----|-----|-----|-----|
|o|-0.4170|-2.9038|5.0600|


##### Set logic has two layers: 2 - 1

|theta |bias|h1|h2|
|-----|-----|-----|-----|
|o|-1.5|1|1|

#### OR Logic:

##### Trained Network has 3 layers, structure is: 2 - 2 - 1

##### Trained W1 of size 2*3

|theta 1|bias|a1|a2|
|-----|-----|-----|-----|
|h1|-1.5270|2.6331|2.8328|
|h2|-1.6748|3.0894|2.9540|

##### Trained W2 of size 1*3

|theta 2|bias|h1|h2|
|-----|-----|-----|-----|
|o|-3.4006|3.6446|4.2209|


##### Set logic has two layers: 2 - 1

|theta |bias|h1|h2|
|-----|-----|-----|-----|
|o|-0.5|1|1|

#### OR Logic:

##### Trained Network has 3 layers, structure is: 2 - 2 - 1

##### Trained W1 of size 2*3

|theta 1|bias|a1|a2|
|-----|-----|-----|-----|
|h1|-1.5270|2.6331|2.8328|
|h2|-1.6748|3.0894|2.9540|

##### Trained W2 of size 1*3

|theta 2|bias|h1|h2|
|-----|-----|-----|-----|
|o|-3.4006|3.6446|4.2209|


##### Set logic has two layers: 2 - 1

|theta |bias|h1|h2|
|-----|-----|-----|-----|
|o|-0.5|1|1|


>### **MNIST**

Train 100 examples in train set and get the test accuracy rate of 42%, but this takes too much time becaus it doesn't use batch train technique.

After using  BATCH-TRAIN mode (with batch_size = 10):
1) Training speed is improved 6-8 times approximately
2) If ramdom test without training, accuracy rate is 10%. If train 1 batch (=10 examples) and test, the accuracy rate is 22% in test set. If train 100 batches (=100 examples), the accuracy rate is 49% in test set.
