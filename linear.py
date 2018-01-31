import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#training data
train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]

#parameters
learning_rate = 0.01
epochs = 1000

#Building the graph
X = tf.placeholder("float",None)
Y = tf.placeholder("float",None)

#Weights andd biases
W = tf.Variable(np.random.rand(), name = "weight")
b = tf.Variable(np.random.rand(), name = "bias")

#prediction (no of samples * 1)
pred = tf.add(tf.multiply(X,W),b)

#error
cost_function = tf.reduce_sum(tf.pow((pred - train_Y),2))/n_samples

#minimize error
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

model = tf.global_variables_initializer()

#Running the graph in session
with tf.Session() as sess:
	sess.run(model)
	for i in range(epochs):
		sess.run(optimizer, feed_dict = {X: np.transpose(train_X), Y: np.transpose(train_Y)})
	#final error in sample
	err = sess.run(cost_function, feed_dict = {X: np.transpose(train_X), Y: np.transpose(train_Y)})
	print("Final in sample error = " + str(err))

	test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
	test_Y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

	testing_cost = sess.run(tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),feed_dict={X: test_X, Y: test_Y}) 
	print("Final out of sample error = " + str(testing_cost))






