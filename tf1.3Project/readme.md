# 20171215
core model function is here, and the paper: 

def create_conv_model(fingerprint_input, model_settings, is_training):
  This is roughly the network labeled as 'cnn-trad-fpool3' in the
  'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
  http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

  This produces fairly good quality results, but can involve a large number of
  weight parameters and computations. For a cheaper alternative from the same
  paper with slightly less accuracy, see 'low_latency_conv' below.

review how convolution:  multiply, tranverse, add.  (f*g)(n)=sigma g(n-tau)f(tau)
eg1. smooth the image. multiply:let the pixel be the average of the neast average, so the kernel is a 3x3 matrix, each element is 1/9; tranverse:this kernel slide to all imamge; add:construct a new image
eg2. calculate the shapenee.  kernel=[-1 2 -2 2 -1], if is the near pixels is similar then it will tend to 0.
we define the kernel to get the feature.
here the kernel is the parameter to be train, it is use to "find" or "explore" the feature.


# 20171213
ImportError: No module named 'inputdata'
inputdata is a key module to pre-process the speech dataset, to extract their features. Maybe is only a files in the same folder of train.py

Ans: inputdata.py is the file in the same folder, it only contains 13 function, is not a class.

so as model.py  the model is created by the function
  logits, dropout_prob = models.create_model(     ---- the model is going to be used for training
      fingerprint_input,
      model_settings,
      FLAGS.model_architecture,
      is_training=True)
and the setting and architecture are set by the arguments, the argument is set by
      model_settings = models.prepare_model_settings(
      len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
      FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
      FLAGS.window_stride_ms, FLAGS.dct_coefficient_count)
look at the default arguments:
    sample_rate: 16000
    wanted_words:'yes,no,up,down,left,right,on,off,stop,go'
    clip_duration_ms: 1000   'Expected duration in milliseconds of the wavs'
    window_size_ms:  30.0  'How long each spectrogram timeslice is'
    window_stride_ms: 10.0  'How long each spectrogram timeslice is'
    dct_coefficient_count: 40      'How many bins to use for the MFCC fingerprint'

    model_architecture: conv



# 20171211
another how does the CNNs work for Small-footprint Keyword Spotting
1\tradition Keyword Spotting Task
more detail: G.Chen  Small-footprint Keyword Spotting using Deep Neural Networks.
3 Modules: 1)Feature extraction, 2)DNN, 3)Posterior Handling
    1)performs voice-activity detection and generates a vector of features every frame (10ms, voice sample rate 8K at least, so no more the 1.25ms) -- tip: think deeply and then know how to dropout the data(one subword is at least 100ms, so feature frame 10ms is fine)  -- look at the code
    2)These features are stacked using the left to right context to create a larger vecter, the is fed to DNN
    3)predict posterior probabilities



speech_command: 
save the dataset      /tmp/speech_dataset/
Directory to write event logs and checkpoint '/tmp/speech_commands_train'
'--summaries_dir', '/tmp/retrain_logs'  'Where to save summary logs for TensorBoard.'
Problem:
cd /tmp/speech_commands_train
bash: cd: /tmp/speech_commands_train
because it is tmp, so I can't find it again.   besides, you can not use ~/   but /home/tonzheng

?? how to copy the relog file to my computer , because the firefox can see the graph

Read, Extract and feed:
  audio_processor = input_data.AudioProcessor(
      FLAGS.data_url, FLAGS.data_dir, FLAGS.silence_percentage,
      FLAGS.unknown_percentage,
      FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
      FLAGS.testing_percentage, model_settings)

    # Pull the audio samples we'll use for training.
    train_fingerprints, train_ground_truth = audio_processor.get_data(
        FLAGS.batch_size, 0, model_settings, FLAGS.background_frequency,
        FLAGS.background_volume, time_shift_samples, 'training', sess)
    # Run the graph with this batch of training data.
    train_summary, train_accuracy, cross_entropy_value, _, _ = sess.run(
        [
            merged_summaries, evaluation_step, cross_entropy_mean, train_step,
            increment_global_step
        ],
        feed_dict={
            fingerprint_input: train_fingerprints,
            ground_truth_input: train_ground_truth,
            learning_rate_input: learning_rate_value,
            dropout_prob: 0.5
        })





# 20171207
# 3.1tensorFlow_Mechanics/fully_connect_feed.py
The goal of this tutorial is to show how to use TensorFlow to train and evaluate a simple feed-forward neural network for handwritten digit classification using the (classic) MNIST data set. 
The intended audience for this tutorial is experienced machine learning users interested in using TensorFlow.

99\1 parser and add the argument in python 
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='Initial learning rate.'
  )
  parser.add_argument(...)

20171208 continue

99\2 gfile, save or delete but how to reuse?
def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  run_training()
ANS:  
1) module: tf.gfile   Import router for file_io.
2) parser: Command line option and argument parsing was added to python 2.7
becase when:
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/logs/fully_connected_feed'),
      help='Directory to put the log data.'
  )
so when i run (py35) python fully_connect_feed.py --log_dir ./herelog             
then:
it will make a new dir ./herelog
#here FLAGS is a special 'global' variable  -- learn again，saving the auguments

  data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)
if the files have existed ,then just extract,
otherwise, must download first, and then extract
experiment:
python fully_connect_feed.py --log_dir ./herelog --input_data_dir ./hereinput
./herelog Exists
make a new dir ./herelog
Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.

#save the parameter,  look at the train.py in tensorflow/example/speech_command





# 20171206
# 1.2 tf.estimator.py
tf.estimator is a high-level TensorFlow library that simplifies the mechanics of machine learning, including the following:
running training loops
running evaluation loops
managing data sets
tf.estimator defines many common models.

predefined types of training / evaluation like linear regression / linear classification / NN classisfer and regressors
and predefined the type parameters,  like a kind of linear regression feature_column

we build a regresion model by estimator:
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns) # the feature_column shape is [1]
print("train metrics: %r"% train_metrics) #train metrics: {'loss': 1.4188086e-07, 'global_step': 1000, 'average_loss': 3.5470215e-08}
print("eval metrics: %r"% eval_metrics) #eval metrics: {'loss': 0.010155011, 'global_step': 1000, 'average_loss': 0.0025387527}
so the trained parameter, may be one W and one b

# 1.3 customer_model.py
redefine the estimator,  overwrite the function member, by the low-level tensorflow API, just like 1.1
estimator = tf.estimator.Estimator(model_fn=model_fn)    mode_fn is redefine by us



# 2.1 MnistBeginner.py
contrast it with the previous linear regression:
1 shape of data, and the operation   ??*1 vs ??*784 ;  linear一元一次 vs 降维的线性映射  
    x = tf.placeholder(tf.float32) # default is None
    linear_model = W * x + b
VS
    x = tf.placeholder(tf.float32, [None, 784]) # also there is a none, so the second number is the nunber of columns the 784 is the number of pixels
    W = tf.Variable(tf.zeros([784, 10])) # matrix multiple and add, here map ?? * 784 to ?? * 10, so the matrix shape is 784*10
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b
2 optimization and  loss function ==> train: least squares vs softmax_cross_entroy_with_logits   optimizer is the same: GradientDescentOpimizer
    loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)
VS
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

3 session and initializer :    Seession vs InteractiveSession;    global_variables_initilalizer is the same
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
VS
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

4 test the trained model (evaluate the accuracy)      original loss function of original train_set  vs  mean cast equal(argmax output, argmax label)
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
VS
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))

doc:  API    
    argmax: Returns the index with the largest value across axes of a tensor.  because y is last layer, we should find the index(class result)  
        tensor = y , axis = 1 Describes which axis of the input Tensor to reduce across. For vectors, use axis = 0. Here matrix, the 2nd dim
    equal: Returns the truth value of (x == y) element-wise.  vector with element 1 and 0;
    cast: Casts a tensor to a new type.  from bool to float32
    reduce_mean: Computes the mean of elements across dimensions of a tensor. Reduces input_tensor along the dimensions given in axis.
        If axis has no entries, all dimensions are reduced, and a tensor with a single element is returned.
            x = tf.constant([[1., 1.], [2., 2.]])  这个轴可能是从高维算起，如 这里现有列数，再有行数，所以 行算高的，然后降低行的个数，这个就是竖的啦
            tf.reduce_mean(x)  # 1.5
            tf.reduce_mean(x, 0)  # [1.5, 1.5]
            tf.reduce_mean(x, 1)  # [1.,  2.]


# 2.2 MnistExpert
in 2.1 we use softmax to classify the digits, now we implement A deep MNIST classifier using convolutional layers.
review: 
    input: (N_examples, 784)    each example is a column, and the number of rows is the number of examples
python keyword
    with:   it seems like the duration/scope of an operation 

loss function: the same with MnistBeginner
    tensorflow function:
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv)
    softmax_cross_entropy_with_logits:  Computes softmax cross entropy between logits and labels.
        labels: Each row labels[i] must be a valid probability distribution.  ??
            Ans: y_ = tf.placeholder(tf.float32, [None, 10])  这里每一行都是1个1，九个0，所以 P(1)=0.1 P(0)=0.9
        logits: Unscaled log probabilities.
optimizer:   adamOptimizer
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

graph_location:
  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

Train and Evaluate the Model
How well does this model do? To train and evaluate it we will use code that is nearly identical to that for the simple one layer SoftMax network above.
The differences are that:
    We will replace the steepest gradient descent optimizer with the more sophisticated ADAM optimizer.
    We will include the additional parameter keep_prob in feed_dict to control the dropout rate.
    We will add logging to every 100th iteration in the training process.
    We will also use tf.Session rather than tf.InteractiveSession. This better separates the process of creating the graph (model specification) and the process of evaluating the graph (model fitting). It generally makes for cleaner code. 






#before 201712

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


