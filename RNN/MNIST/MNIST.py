'''
Some explanations refer to this article：
    https://zhuanlan.zhihu.com/p/23018343?refer=xishi
Code reference：
    https://github.com/FanGhost/RNNStudy/blob/master/simpleRNN.ipynb
The version information as follows:
    Python-3.5.3
    tensorflow-1.4.0
'''
#!/usr/bin/env python
import time
import tensorflow as tf
from tensorflow.contrib import rnn
start = time.clock()    # start the timer

# inport data
from tensorflow.examples.tutorials.mnist import input_data       # the data will be download from https://storage.googleapis.com/cvdf-datasets/mnist/,
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)   # and save in "./MNIST_data/" folder


# Defina the hyperparameters which related to the model and the model training process
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 100

n_input = 28      # MNIST data input (img shape: 28*28)
n_steps = 28      # timesteps
n_hidden = 240    # hidden layer num of features
n_output = 10     # MNIST total classes (0-9 digits)

# Define the input and output of model
x = tf.placeholder("float32", [None, n_steps*n_input])    # tf Graph input, the data structure of MNIST is a 28*28
                                                           # vector represent the pixel value of handwritten digit picture
y = tf.placeholder("float32", [None, n_output])          # Tensorflow LSTM cell requires 2x n_hidden length (state & cell)

# Define weights and biases
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])),  # the input layer to hidden layer weights matrix
    'out': tf.Variable(tf.random_normal([n_hidden, n_output]))}    # the hidden layer to output layer weights matrix
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),   # the input layer to hidden layer biases vector
    'out': tf.Variable(tf.random_normal([n_output]))}      # the hidden layer to output layer biasess vector

# Create a basic LSTM cell and initialize its state
lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)  # the basic LSTM cell is define in: https://arxiv.org/abs/1409.2329 
init_state = lstm_cell.zero_state(batch_size,tf.float32)                       # initialize the state 0

    # the size of the input of tf.nn.dynamic_rnn is [batch_size, n_steps, n_hidden]
input_x = tf.reshape(x, [-1, n_input])
input_x = tf.matmul(input_x, weights['hidden']) + biases['hidden']
input_x = tf.reshape(input_x, [-1, n_steps, n_hidden])

outputs, states = tf.nn.dynamic_rnn(lstm_cell, input_x, initial_state=init_state)   # the size of outputs is [batch_size, n_steps, n_hidden]
pred = tf.matmul(outputs[:, -1, :], weights['out']) + biases['out']

# Define the loss function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)    # Adam Optimizer
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)    # Gradient Descent Optimizer
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Training the model
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

step = 1
while step * batch_size < training_iters:                         # Keep training until reach max iterations
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)       # Reshape data to get 28 seq of 28 elements
    #batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))   # Fit training using batch data
    sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
    if step % display_step == 0:
        acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,})     # Calculate batch accuracy
        loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})         # Calculate batch loss
        print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) +  ", Training Accuracy= " + "{:.5f}".format(acc))
    step += 1
print("Optimization Finished!")

# Calculate accuracy for 256 mnist test images
test_len = batch_size
test_data = mnist.test.images[:test_len]
test_label = mnist.test.labels[:test_len]
print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))    # Evaluate model

end = time.clock()    # end the timer
print(end - start)
