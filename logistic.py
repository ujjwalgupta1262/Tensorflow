import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#parameters
learning_rate = 0.01
epochs = 1
batch_size = 100

#Building the graph
X = tf.placeholder(tf.float32,[None,784])       #No_of_examples * no_of_features
Y = tf.placeholder(tf.float32,[None,10])		 #No_of_examples * no_of_classes

#Weights andd biases
W = tf.Variable(tf.zeros([784,10]), name = "weight")
b = tf.Variable(tf.zeros([10]), name = "bias")

#prediction (no of samples * 1)
pred = tf.nn.softmax(tf.matmul(X,W)+b)

#cost_function
cost_function = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(pred), reduction_indices=1))

#minimizing cost function
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

model = tf.global_variables_initializer()

#Running the computational graph
with tf.Session() as sess:
	sess.run(model)
	error = 0
	#Stochastic gradient descent
	for epoch in range(epochs):
		no_of_batches = int(mnist.train.num_examples/batch_size)
		#gradient descent for each batch
		for i in range(no_of_batches):
			train_x , train_y = mnist.train.next_batch(batch_size)
			cost = sess.run(cost_function, feed_dict = {X: train_x, Y: train_y})
			sess.run(optimizer, feed_dict = {X: train_x, Y: train_y})
		error += cost/no_of_batches

print(error)


