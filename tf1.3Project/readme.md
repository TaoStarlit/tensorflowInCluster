# 0.2TrainableParameters.py
using loss function to measures how far apart the current model is from the provided data.
least square is the common way to train linear model
 square of (model output - expected value provided data)  let the sum minimum

reduce_sum, can it train the variables? no! because you didn't use the optimizers

tensorflow provides optimizers that slowly change each variable in order to minimize the loss funcion.
The simplest optimizer is gradient descent. It modifies each variable according to the magnitude of the derivative of loss
 with respect to that variable.   [magnitude:The great size or extent of something.]
optimizer = tf.train.GradientDescentOptimizer(0.025)#bigger, and the train will be quicker !! but if bigger than 0.025 it can not converge
train = optimizer.minimize(loss)
print(optimizer,train)#(<tensorflow.python.training.gradient_descent.GradientDescentOptimizer object at 0x2b9cb35cc9d0>, <tf.Operation 'GradientDescent' type=NoOp>)




# 1 getting_start.py
# 0.1letTensorFlow.py
the core unit of tensorflow is tensor. A tensor consist of a set of primitive values shaped into an array of any number of dimensions.

we can regard the neunal network as a computational graph.
it is the series of tersorflow operations arranged into graph of nodes.
[arrange: Put (things) in a neat, attractive, or required order.]
Each node takes zero or more tensors as inputs and produces a tensor as an output.

type of node: depent on what type the input and output tensors are.
constant: takes no inputs and outputs a value it stores in internally.
operation: such add several nodes, implemented by + or tf.add()
placeholers: accept external input

(train)
variables: allow us to add trainable parameters to a graph. They are constructed with a type and initial value

init: <tf.Operation 'init' type=NoOp>

Square: <tf.Tensor 'Square:0'
Reduce_sum: f.Tensor 'Sum:0'
Optimizer(not train) tensorflow.python.training.gradient_descent.GradientDescentOptimizer
Train(optimizer,minizize(loss)) tf.Operation 'GradientDescent' type=NoOp>

tip: you can print the node, to see the type, shape, and datatype

let the tensor flow: session
To actually evaluate the nodes, we must run the computational graph within a session.
A session encapsulates the control and state the Tensorflow runtime.


