# **HW3- Neural Networks Backward Propagation Pass**
###Student: Kuan Han



A backward propagation pass was designed to update the parameters in the neural netwwork. The scheme was evaluated to be correct by following ways: 1) check the result with the 2-2-2 online demo case (https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/) and every parameter was checked for the first update iteration; 2) compare the trained parameters with the setted parameters for logic functions, and also classification result are excatly the same; 3) test the result on MNIST dataset
 
----------


>### **Scheme of the BP Implementation**

(Train 250 times and LR = 0.5)

Every iteration the program calculate the gradient for all parameters from top to down of the network. Suppose that the network has n+1 layers of neurons & n layers of weights matrix, we calculate the neuron and weight's gradient with following sequence:

(n+1)th layer neuron => weights between nth and (n+1)th layer => nth layer neuron => 

weights between (n-1)th and nth layer => (n-1)th neuron => ... => weights between 2nd and 1st layer

---------------------------


>### **Comparison between hand-crafted Θ and trained Θ**

#### Show conclusion first: for each logic function, if the trained theta and the set theta share the same structure.their value are different, but the relative scale of the parameter in a certain position are similar compared with other parameters in the network.

#### AND Logic:

##### Trained AND Network has 2 layers, structure is: 2 - 1
|theta 1|bias|a1|a2|
|-----|-----|-----|-----|
|o1|-5.8924  |3.8564  |3.8411|

##### Another Version: Trained AND Network has 3 layers, structure is: 2 - 2 - 1

##### Trained AND W1 of size 2*3

|theta 1|bias|a1|a2|
|-----|-----|-----|-----|
|h1|-0.6047|0.6412|1.7730|
|h2|3.0369|-2.6292|-2.0625|

##### Trained AND W2 of size 1*3

|theta 2|bias|h1|h2|
|-----|-----|-----|-----|
|o|-0.4170|-2.9038|5.0600|


##### Set AND logic has two layers: 2 - 1

|theta |bias|h1|h2|
|-----|-----|-----|-----|
|o|-1.5|1|1|
-----

#### OR Logic:

##### Trained OR Network has two layers: 2 - 1

|theta |bias|x1|x2|
|-----|-----|-----|-----|
|o|-1.9890  |4.5251|  4.5331|


##### Another Version: Trained OR Network has 3 layers, structure is: 2 - 2 - 1

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
-----

#### NOT Logic:

##### Trained Network has 2 layers, structure is: 1 - 1

##### Trained W of size 1*2

|theta|bias|x1|
|-----|-----|-----|
|o|2.3415|-4.9105|


##### Set logic has two layers: 1 - 1

|theta |bias|x1|
|-----|-----|-----|
|o|0.5|-1|
-----

#### XOR Logic:

##### Trained Network has 3 layers, structure is: 2 - 5 - 1

##### Trained W1 of size 2*3

|theta 1|bias|a1|a2|
|-----|-----|-----|-----|
|h1|-0.4409|-2.1576 |-2.5149
|h2| 0.8348 |-4.0153 |-4.2974
|h3|-1.6814 | 3.7030 |-5.5483
|h4|-0.9777 |-5.6127 | 3.0639
|h5|-0.4078 | 0.3130| -2.0854


##### Trained W2 of size 1*3

|theta 2|bias|h1|h2|h3|h4|h5|
|-----|-----|-----|-----|-----|-----|-----|
|o|-2.9617 |-1.5278| -4.5006 | 6.3336  |6.6567  |1.2095|


##### Set logic has two layers: 2 - 1
According to sequential logic, a⊕b = (¬a ∧ b) ∨ (a ∧¬b), so the XOR set logic is consisting of 2 NOT gates, 2 AND gates and an OR gate.

-------------------

>### **MNIST Test Result**

Train 100 examples in train set and get the test accuracy rate of 42%, but this takes too much time becaus it doesn't use batch train technique.

After using  BATCH-TRAIN mode (with batch_size = 10):

1) Training speed is improved 6-8 times approximately

2) If ramdom test without training, accuracy rate is 10%. If train 1 batch (=10 examples) and test, the accuracy rate is 22% in test set. If train 100 batches (=100 examples), the accuracy rate is 49% in test set.

3) Update: with batch_size = 20 and batch-train iteration = 500, the program gets the accuracy rate of 76% in the test set.
