import numpy as np
import tensorflow as tf
import random
from random import shuffle

seq_length = 10
test_set_size = 1000
#generating training data
def generate_data(seq_length,training_set_size):
	temp = []
	#generating training sequences
	while(training_set_size > 0):
		s = ""
		for i in range(seq_length):
			s += str(int(round(random.random())))
		temp.append(s)
		training_set_size -= 1
	input_list = []
	label_list = []
	#converting input data in (batch_size,seq_length(time_steps),input_size)
	for i in temp:
		temp_list = []
		count = 0
		all_zero = [0]*(seq_length+1)
		for j in str(i):
			temp_list.append([int(j)])
			count += int(j)
		input_list.append(temp_list)
		all_zero[count] = 1
		label_list.append(all_zero)
	return (input_list,label_list)


#parameters
#time steps through which the RNN is unrolled for
time_steps = seq_length
#Number of inputs at each time steps
batch_size = 1000
#Number of hidden layers
num_layers = 1
#number of LSTM units in each LSTM cell
num_units = 128
#size of input vector
input_size = 1
#output size
output_size = seq_length+1
#learning rate
learning_rate = 0.01
#no. of iterations
iterations = 500

#Building the graph
X = tf.placeholder("float",[None,time_steps,input_size])    #None represents the batch size
Y = tf.placeholder("float",[None,output_size])

#output weights and bias
output_weights = tf.Variable(tf.random_normal([num_units,output_size]))
output_bias = tf.Variable(tf.random_normal([output_size]))

input = tf.unstack(X ,time_steps,1)

#Building the RNN
hidden_layer = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias = 1)
output,state = tf.contrib.rnn.static_rnn(hidden_layer,input,dtype = "float32")

#prediction
pred = tf.matmul(output[-1],output_weights) + output_bias

#cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=Y))

#optimizer
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

#evaluating the model
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(Y,1))
in_sample_accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

#initialising the model
model = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(model)
	#training
	while(iterations > 0):
		train_x,train_y = generate_data(10,500)
		sess.run(optimizer, feed_dict = {X : train_x, Y:train_y})
		c = sess.run(cost, feed_dict = {X : train_x, Y:train_y})
		if(iterations%100 == 0):
			print(c)
		iterations -= 1

	#testing
	test_x,test_y = generate_data(seq_length,test_set_size)
	out_sample_accuracy = sess.run(in_sample_accuracy, feed_dict = {X : test_x, Y : test_y})
	print("Accuracy: ", out_sample_accuracy)






