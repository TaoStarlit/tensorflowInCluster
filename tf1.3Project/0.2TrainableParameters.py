import tensorflow as tf

sess=tf.Session()

W = tf.Variable([3], dtype=tf.float32)
b = tf.Variable([-3], dtype=tf.float32)
init = tf.global_variables_initializer()

print(W,b,init) #<tf.Variable 'Variable:0' shape=(1,) dtype=float32_ref>  ... <tf.Operation 'init' type=NoOp>

print(sess.run(init))

x = tf.placeholder(tf.float32)
linear_model = W * x + b

print(W,b,x)
print("linear model node:",linear_model)


#here we just use the variable, but don't train them
print(sess.run(linear_model,{x:[1,2,3,4]}))#placeholder take list input, 3*1-3=0 ... 3*4-3=9



#using loss function to measures how far apart the current model is from the provided data.
#least square is the common way to train linear model
# square of (model output - expected value provided data)  let the sum minimum
y = tf.placeholder(tf.float32)
squared_delta = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_delta) # can it train the variables, no! because you didn't use the optimizers
print(squared_delta,loss) #(<tf.Tensor 'Square:0' shape=<unknown> dtype=float32>, <tf.Tensor 'Sum:0' shape=<unknown> dtype=float32>)
print(sess.run(loss,{x:[1,2,3,4],y: [0, 2, 7, 9]}))#0 3 6 9, so the sum will be around 2


'''
train
'''
#tensorflow provides optimizers that slowly change each variable in order to minimize the loss funcion.
#The simplest optimizer is gradient descent. It modifies each variable according to the magnitude of the derivative of loss
# with respect to that variable.   [magnitude:The great size or extent of something.]
optimizer = tf.train.GradientDescentOptimizer(0.025)#bigger, and the train will be quicker !! if bigger than 0.025 it can not converge
train = optimizer.minimize(loss)
print(optimizer,train)#(<tensorflow.python.training.gradient_descent.GradientDescentOptimizer object at 0x2b9cb35cc9d0>, <tf.Operation 'GradientDescent' type=NoOp>)

sess.run(init)
fixW = tf.assign(W, [.3])
fixb = tf.assign(b, [-.3])
sess.run([fixW, fixb])
print("W,b before train:",sess.run([W,b]))
for i in range(200):
    sess.run(train, {x:[1,2,3,4],y:[0,-1,-2,-3]})
    if(i % 20 == 0):
        print("W,b:",sess.run([W,b]))
        print('loss:',sess.run(loss, {x:[1,2,3,4],y:[0,-1,-2,-3]}))
print("W,b after train:", sess.run([W, b]))
print('loss:', sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

'''
how to set the right parameter of GradientDesecentOptimizer and the repetition of train.
'''




