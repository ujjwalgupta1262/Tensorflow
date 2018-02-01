import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#parameters
learning_rate = 0.1
steps = 500
batch_size = 100

#network weight parameters
neurons_layer1 = 300
neurons_layer2 = 300

#Building the graph
X = tf.placeholder("float",[None,784])
Y = tf.placeholder("float",[None,10])

#weights parameters
weights_layer01 = tf.Variable(tf.random_normal([784,300]))
weights_layer12 = tf.Variable(tf.random_normal([300,300]))
weights_layer23 = tf.Variable(tf.random_normal([300,10]))

#bias parameters
bias_layer1 = tf.Variable(tf.random_normal([300]))
bias_layer2 = tf.Variable(tf.random_normal([300]))
bias_layer3 = tf.Variable(tf.random_normal([10]))

#find predictions of neural network
layer1_output = tf.matmul(X,weights_layer01) + bias_layer1
layer2_output = tf.matmul(layer1_output,weights_layer12) + bias_layer2
final_output = tf.matmul(layer2_output,weights_layer23) + bias_layer3
pred = tf.nn.softmax(final_output)

#cost_function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final_output, labels=Y))

#minimizing cost function
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#error
correct = tf.equal(tf.argmax(pred,1),tf.argmax(Y,1))
right = tf.reduce_mean(tf.cast(correct,tf.float32))

model = tf.global_variables_initializer()

#Running the graph
with tf.Session() as sess:
	sess.run(model)
	error = 0
	for i in range(steps):
		train_x , train_y = mnist.train.next_batch(batch_size)
		sess.run(optimizer, feed_dict = {X: train_x, Y: train_y})
		a = sess.run(right, feed_dict = {X: train_x, Y: train_y})
		print(a)
	print("out of sample accuracy", sess.run(right, feed_dict = {X: mnist.test.images, Y: mnist.test.labels}))
