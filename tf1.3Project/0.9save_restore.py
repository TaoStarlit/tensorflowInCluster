import tensorflow as tf
import os

sess = tf.Session()
print(sess)  # <tensorflow.python.client.session.Session object at 0x0000020EA43B5F60>

W = tf.Variable([3], dtype=tf.float32)
b = tf.Variable([-3], dtype=tf.float32)
init = tf.global_variables_initializer()

# <tf.Variable 'Variable:0' shape=(1,) dtype=float32_ref>  ... <tf.Operation 'init' type=NoOp>
print(W, b, init)

print(sess.run(init))

x = tf.placeholder(tf.float32)
linear_model = W * x + b

print(W, b, x)  # <tf.Variable 'Variable:0' shape=(1,) dtype=float32_ref> <tf.Variable 'Variable_1:0' shape=(1,) dtype=float32_ref> Tensor("Placeholder:0", dtype=float32)
# linear model node: Tensor("add:0", dtype=float32)
print("linear model node:", linear_model)


# here we just use the variable, but don't train them
# placeholder take list input, 3*1-3=0 ... 3*4-3=9
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))


# using loss function to measures how far apart the current model is from the provided data.
# least square is the common way to train linear model
# square of (model output - expected value provided data)  let the sum minimum
y = tf.placeholder(tf.float32)
squared_delta = tf.square(linear_model - y)
# can it train the variables, no! because you didn't use the optimizers
loss = tf.reduce_sum(squared_delta)
# (<tf.Tensor 'Square:0' shape=<unknown> dtype=float32>, <tf.Tensor 'Sum:0' shape=<unknown> dtype=float32>)
print("squared_delta,loss: ", squared_delta, loss)
# 0 3 6 9, so the sum will be around 2
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, 2, 7, 9]}))


'''
train
'''
# tensorflow provides optimizers that slowly change each variable in order to minimize the loss funcion.
# The simplest optimizer is gradient descent. It modifies each variable according to the magnitude of the derivative of loss
# with respect to that variable.   [magnitude:The great size or extent of something.]
# bigger, and the train will be quicker !! if bigger than 0.025 it can not converge
optimizer = tf.train.GradientDescentOptimizer(0.025)
train = optimizer.minimize(loss)
# (<tensorflow.python.training.gradient_descent.GradientDescentOptimizer object at 0x2b9cb35cc9d0>,
print(optimizer)
print(train)  # <tf.Operation 'GradientDescent' type=NoOp>)

sess.run(init)
fixW = tf.assign(W, [.3])
fixb = tf.assign(b, [-.3])
sess.run([fixW, fixb])
print("W,b,x : ", W, b, x)
print("W,b value before train: ", sess.run([W, b]))


savepara_file = 'model.ckpt'  # NameError: name 'FLAGS' is not defined
savepara_dir = './save_parameter/'
start_checkpoint = savepara_dir + savepara_file
# start_checkpoint='' # if start_checkpoint  false

# saver = tf.train.Saver({"v2": v2}) # I did not give the variable name here, so tf.global_variables()
saver = tf.train.Saver(tf.global_variables())
if not os.path.exists(savepara_dir):  # python 3 not 2
    os.makedirs(savepara_dir)
if start_checkpoint:  # NameError: name 'start_checkpoint' is not defined
    saver.restore(sess, savepara_dir + savepara_file)
    print("W,b value after restore: ", sess.run([W, b]))
    #start_step = global_step.eval(session=sess)
    ##global_step = tf.contrib.framework.get_or_create_global_step()
else:
    print("no save_parameter")

if not start_checkpoint:
    for i in range(200):
        sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
        if(i % 20 == 0):
            print("W,b:", sess.run([W, b]))
            print('loss:', sess.run(
                loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
    print("save the parameter!!")
    saver.save(sess, savepara_dir + savepara_file)
print("W,b,x : ", W, b, x)
print("W,b after train or restore:", sess.run([W, b]))
print('loss:', sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

'''
how to set the right parameter of GradientDesecentOptimizer and the repetition of train.
'''
