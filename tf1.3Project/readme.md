# shadow_domain_adversarial_NN
1. test tensorflow NN via NNclassifer.py
1. copy NNclassifer into DANN class
1. share variables between methods by self.
1. copy .gitignore and apply
## error
1. ValueError: Cannot feed value of shape (4, 2) for Tensor u'x:0', which has shape '(1, 2)'.<br>
--train data and placeholder,  ANS:@ \[None, 2\] not \[1,2\] to shape placeholder
1. Dimensions must be equal, but are 2 and 15 for 'mul_2' (op: 'Mul' * ) with input shapes: \[1,2\], \[2,15\].<br>
Error: hidden_layer = tf.sigmoid(x*W+b,name="hidden_layer"). <br>
\* is Mul  here we need Matmul @@ tf.matmul(x,W)+b

# 0.2TrainableParameters.py
using loss function to measures how far apart the current model is from the provided data.<br>
least square is the common way to train linear model<br>
 square of (model output - expected value provided data)  let the sum minimum<br>

reduce_sum, can it train the variables? no! because you didn't use the optimizers<br>

tensorflow provides optimizers that slowly change each variable in order to minimize the loss funcion.<br>
The simplest optimizer is gradient descent. It modifies each variable according to the magnitude of the derivative of loss
 with respect to that variable.   [magnitude:The great size or extent of something.]<br>
optimizer = tf.train.GradientDescentOptimizer(0.025)#bigger, and the train will be quicker !! but if bigger than 0.025 it can not converge
train = optimizer.minimize(loss)<br>
print(optimizer,train)#(<tensorflow.python.training.gradient_descent.GradientDescentOptimizer object at 0x2b9cb35cc9d0>, <tf.Operation 'GradientDescent' type=NoOp>)




# 1 getting_start.py
# 0.1letTensorFlow.py
the core unit of tensorflow is tensor. A tensor consist of a set of primitive values shaped into an array of any number of dimensions.

we can regard the neunal network as a computational graph.<br>
it is the series of tersorflow operations arranged into graph of nodes.<br>
[arrange: Put (things) in a neat, attractive, or required order.]<br>
Each node takes zero or more tensors as inputs and produces a tensor as an output.<br>

type of node: depent on what type the input and output tensors are.<br>
constant: takes no inputs and outputs a value it stores in internally.<br>
operation: such add several nodes, implemented by + or tf.add()<br>
placeholers: accept external input<br>

### (train)
variables: allow us to add trainable parameters to a graph. They are constructed with a type and initial value

init: <tf.Operation 'init' type=NoOp>

Square: <tf.Tensor 'Square:0' <br>
Reduce_sum: f.Tensor 'Sum:0' <br>
Optimizer(not train) tensorflow.python.training.gradient_descent.GradientDescentOptimizer <br>
Train(optimizer,minizize(loss)) tf.Operation 'GradientDescent' type=NoOp> <br>

tip: you can print the node, to see the type, shape, and datatype

let the tensor flow: session <br>
To actually evaluate the nodes, we must run the computational graph within a session. <br>
A session encapsulates the control and state the Tensorflow runtime. <br>


