# **HW2- Neural Networks - Feedforward pass**
Student: Kuan Han

---
In the file of "NeuralNetwork" there are 3 functions:

1) NeuralNetwork.build: nuild the table of weight matrix based on the layer information in the input variable. Between layer i (m components) and layer i+1 (n components), theta(i) is a matrix of size n * (m+1).

2) NeuralNetwork.getLayer: get a certain layer according the input index

3) NeuralNetwork.forward: calculate the forward result based on both input tensor and theta.


Based on NeuralNetwork.lua, there is another table logicGates which gives 4 logic gate function

1) logicGates.AND/OR/NOT are simply based on single layer neural network because they are all linearly seperatable for the input logic space.

2) logicGates.XOR cannot directly based on single layer neural network because it is not linearly seperatable. So there should be multi-layer network for representing XOR. There are 2 layers: the first layer consists of two AND networks and the second layer is an OR network previous defined. 

---
