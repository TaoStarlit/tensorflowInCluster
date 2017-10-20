"""
# ValueError: Cannot feed value of shape (4, 2) for Tensor u'x:0', which has shape '(1, 2)'
but (4) and for 1

ValueError: Cannot feed value of shape (64, 64, 3) for Tensor u'Placeholder:0', which has shape '(?, 64, 64, 3)'
reshape 64,64,3 --> 1, 64, 64 ,3
I should set x as ?,2
@ [None, 2]

x_train = [[1, 2], [3, 4],[-1, -2],[-3, -4]]
y_train = [[0, 1], [0, 1],[1, 0],[1, 0]]

x = tf.placeholder(tf.float32,[1,2],"x")

#Dimensions must be equal, but are 2 and 15 for 'mul_2' (op: 'Mul' * ) with input shapes: [1,2], [2,15].
Error: hidden_layer = tf.sigmoid(x*W+b,name="hidden_layer")
8:Mul
you should use matmul
@ tf.matmul(x,W)+b
"""


import tensorflow as tf

x = tf.placeholder(tf.float32,[None,2],"x")

W = tf.Variable((6.0/(2+15))**2*(2*tf.random_uniform([2,15])-1.0),name="W")
b = tf.Variable(tf.constant(0.0, shape=[1,15]),name="b")

hidden_layer = tf.sigmoid(tf.matmul(x,W)+b,name="hidden_layer")

V = tf.Variable((6.0/(15+2))**2*(2*tf.random_uniform([15,2])-1.0),name="V")
c = tf.Variable(tf.constant(0.0, shape=[1,2]),name="b")

output_layer = tf.nn.softmax(tf.matmul(hidden_layer,V)+c,name="output_layer")


y = tf.placeholder(tf.float32,[None,2],"y")

# loss function
loss =tf.reduce_sum(tf.square(output_layer-y))
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)


# training data
x_train = [[1, 2], [3, 4],[-1, -2],[-3, -4]]
y_train = [[0, 1], [0, 1],[1, 0],[1, 0]]
#initial the variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
# training loop
print("W,b before train:",sess.run([W,b]))
for i in range(200):
    sess.run(train, {x: x_train, y: y_train})
    if(i % 20 == 0):
        print("W,b:",sess.run([W,b]))
        print('loss:',sess.run(loss, {x:x_train,y:y_train}))

print("W,b after train:", sess.run([W, b]))
print('loss:', sess.run(loss, {x: x_train, y: y_train}))








