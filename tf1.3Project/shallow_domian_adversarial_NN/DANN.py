"""
NameError: global name 'W' is not defined
W is define in __init__
but if you invoke in another member function, it can not find it
"""

import tensorflow as tf

class DANN(object): #derived class DANN inherit the base class object


    def __init__(self, learning_rate=0.05, input_layer_size=2, output_layer_size=2, hidden_layer_size=25, lambda_adapt=1., maxiter=200,
                 epsilon_init=None, adversarial_representation=True, seed=12342, verbose=False):
        """
        Domain Adversarial Neural Network for classification

        option "learning_rate" is the learning rate of the neural network.
        option "hidden_layer_size" is the hidden layer size.
        option "lambda_adapt" weights the domain adaptation regularization term.
                if 0 or None or False, then no domain adaptation regularization is performed
        option "maxiter" number of training iterations.
        option "epsilon_init" is a term used for initialization.
                if None the weight matrices are weighted by 6/(sqrt(r+c))
                (where r and c are the dimensions of the weight matrix)
        option "adversarial_representation": if False, the adversarial classifier is trained
                but has no impact on the hidden layer representation. The label predictor is
                then the same as a standard neural-network one (see experiments_moon.py figures).
        option "seed" is the seed of the random number generator.
        """

        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.maxiter = maxiter
        self.lambda_adapt = lambda_adapt if lambda_adapt not in (None, False) else 0.
        self.epsilon_init = epsilon_init
        self.learning_rate = learning_rate
        self.adversarial_representation = adversarial_representation
        self.seed = seed
        self.verbose = verbose

        '''
        input_layer:2
        hidden_layer:15
            so W shape(2,15)
            b shape(1,15)
            sigmoid
        labels:2 classis
            so V shape(15,2)
            c shape (1,2)
            softmax
        '''
        self.x = tf.placeholder(tf.float32, [None, input_layer_size], "x")

        valInit=(6.0 / (input_layer_size + hidden_layer_size)) ** 2 * (2 * tf.random_uniform([input_layer_size, hidden_layer_size]) - 1.0)
        self.W = tf.Variable(valInit, name="W")
        self.b = tf.Variable(tf.constant(0.0, shape=[1, hidden_layer_size]), name="b")

        hidden_layer = tf.sigmoid(tf.matmul(self.x, self.W) + self.b, name="hidden_layer")

        valInit = (6.0 / (output_layer_size + hidden_layer_size)) ** 2 * (2 * tf.random_uniform([hidden_layer_size, output_layer_size]) - 1.0)
        V = tf.Variable(valInit, name="V")
        c = tf.Variable(tf.constant(0.0, shape=[1, output_layer_size]), name="b")

        output_layer = tf.nn.softmax(tf.matmul(hidden_layer, V) + c, name="output_layer")

        self.y = tf.placeholder(tf.float32, [None, output_layer_size], "y")

        # loss function
        self.loss = tf.reduce_sum(tf.square(output_layer - self.y))
        # optimizer
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        self.train = optimizer.minimize(self.loss)

    def fit(self, x_train, y_train, X_adapt=None, X_valid=None, Y_valid=None, do_random_init=True):
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)  # reset values to wrong
        # training loop
        print("W,b before train:", self.sess.run([self.W, self.b]))
        Observation_interval=self.maxiter/10
        for i in range(self.maxiter):
            self.sess.run(self.train, {self.x: x_train, self.y: y_train})
            if (i % Observation_interval == 0):
                print("W,b:", self.sess.run([self.W, self.b]))
                print('loss:', self.sess.run(self.loss, {self.x: x_train, self.y: y_train}))



    def evaluate(self,x_test,y_test):
        print("W,b after train:", self.sess.run([self.W, self.b]))
        print('loss:', self.sess.run(self.loss, {self.x: x_test, self.y: y_test}))