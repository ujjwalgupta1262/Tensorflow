import numpy as np
import tensorflow as tf
import random
from random import shuffle

#generating training data
def generate_data(min_seq_length,max_seq_length,training_set_size):
	input_list = []
	label_list = []
	seq_length = []
	#generating training sequences
	while(training_set_size > 0):
		temp_list = []
		len = random.randint(min_seq_length,max_seq_length)
		seq_length.append(len)
		if(random.random() > 0.5):
			l = range(len)*int(round(random.randint(1,9)))
			for j in range(max_seq_length):
				if(j < l):
					temp_list.append([j])
				else:
					temp_list.append([0])
			label_list.append([0,1])
		else:
			l = [random.randint(2,50) for i in range(len)]
			for j in range(max_seq_length):
				if(j < l):
					temp_list.append([j])
				else:
					temp_list.append([0])
			label_list.append([1,0])
		input_list.append(temp_list)
		training_set_size -= 1
	return (input_list,label_list,seq_length)


#parameters
#maximum and minimum sequence length
max_seq_length = 10
min_seq_length = 2
#time steps through which the RNN is unrolled for
time_steps = max_seq_length
#Number of inputs at each time steps
batch_size = 20
#Number of hidden layers
num_layers = 1
#number of LSTM units in each LSTM cell
num_units = 128
#size of input vector
input_size = 1
#output size
output_size = 2
#learning rate
learning_rate = 0.01
#no. of iterations
iterations = 500

#Building the graph
X = tf.placeholder("float",[None,time_steps,input_size])    #None represents the batch size
Y = tf.placeholder("float",[None,output_size])
sequence_length = tf.placeholder(tf.int32,[None])

#output weights and bias
output_weights = tf.Variable(tf.random_normal([num_units,output_size]))
output_bias = tf.Variable(tf.random_normal([output_size]))

#input = tf.unstack(X ,time_steps,1)

#Building the RNN
hidden_layer = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias = 1)
output,state = tf.nn.dynamic_rnn(hidden_layer,X,dtype = "float32",sequence_length = sequence_length)

#prediction
output = tf.transpose(output,[1,0,2])
#list of end indices for each sequence in batch
indices = tf.range(0, batch_size) * max_seq_length + (sequence_length - 1)
#list of outputs
outputs = tf.gather(tf.reshape(output, [-1, num_units]), indices)
pred = tf.matmul(outputs,output_weights) + output_bias
real_pred = tf.nn.softmax(pred);

#cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))

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
		train_x,train_y,train_seq_length = generate_data(2,10,20)
		sess.run(optimizer, feed_dict = {X : train_x, Y:train_y, sequence_length: train_seq_length})
		c = sess.run(cost, feed_dict = {X : train_x, Y:train_y, sequence_length: train_seq_length})
		if(iterations%100 == 0):
			print(c)
		iterations -= 1

	#testing
	test_x,test_y,test_seq_length = generate_data(2,10,20)
	prediction = sess.run(real_pred,feed_dict = {X : test_x, Y : test_y, sequence_length: test_seq_length})
	out_sample_accuracy = sess.run(in_sample_accuracy, feed_dict = {X : test_x, Y : test_y,sequence_length : test_seq_length})
	#print("prediction: ",prediction)
	print("Accuracy: ", out_sample_accuracy)






