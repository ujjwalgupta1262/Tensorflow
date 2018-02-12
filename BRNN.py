import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#parameters
#time steps through which the RNN is unrolled for
time_steps = 28
#Number of inputs at each time steps
batch_size = 128
#Number of hidden layers
num_layers = 1
#number of LSTM units in each forward LSTM cell
num_forward_units = 128
#number of LSTM units in each backward LSTM cell
num_backward_units = 28
#size of input vector
input_size = 28
#output size
output_size = 10
#learning rate
learning_rate = 0.001
#no. of iterations
iterations = 800

#Building the graph
X = tf.placeholder("float",[None,time_steps,input_size])    #None represents the batch size
Y = tf.placeholder("float",[None,output_size])

#output weights and bias
output_weights = tf.Variable(tf.random_normal([num_forward_units + num_backward_units,output_size]))
output_bias = tf.Variable(tf.random_normal([output_size]))

#preparing input to be used as input to rnn
input = tf.unstack(X ,time_steps,1)

#Building the bidirectional RNN
hidden_forward_layer = tf.contrib.rnn.BasicLSTMCell(num_forward_units, forget_bias = 1)
hidden_backward_layer = tf.contrib.rnn.BasicLSTMCell(num_backward_units, forget_bias = 1)
output,state1,state2 = tf.contrib.rnn.static_bidirectional_rnn(hidden_forward_layer,hidden_backward_layer,input,dtype = "float32")

#prediction
pred = tf.matmul(output[-1],output_weights) + output_bias

#cost functiono
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=Y))

#optimizer
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

#evaluating the model
correct_predictions = tf.equal(tf.argmax(pred,1),tf.argmax(Y,1))
in_sample_accuracy = tf.reduce_mean(tf.cast(correct_predictions,tf.float32))

#initialising the model
model = tf.global_variables_initializer()

#Running the graph
with tf.Session() as sess:
	sess.run(model)
	#training
	while(iterations > 0):
		train_x,train_y = mnist.train.next_batch(batch_size = batch_size)
		batch_x = train_x.reshape((batch_size,time_steps,input_size))
		sess.run(optimizer, feed_dict = {X : batch_x, Y:train_y})
		c = sess.run(cost, feed_dict = {X : batch_x, Y:train_y})
		if(iterations%100 == 0):
			print(c)
		iterations -= 1

	#testing
	test_x,test_y = mnist.train.next_batch(batch_size = batch_size)
	batch_x = test_x.reshape((batch_size,time_steps,input_size))
	out_sample_accuracy = sess.run(in_sample_accuracy, feed_dict = {X : batch_x, Y : test_y})
	print("Accuracy: ", out_sample_accuracy)

