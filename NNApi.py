import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

#Parameters
learning_rate = 0.1
steps = 500
batch_size = 100

#no of neurons
neurons_layer1 = 400
neurons_layer2 = 400

#define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"input": mnist.train.images}, y=mnist.train.labels,
    batch_size=batch_size, num_epochs=None, shuffle=True)

#define the model(neural network)
def neural_network(features_dict):
	input_layer = features_dict["input"]					#layer containing input
	layer1 = tf.layers.dense(input_layer, neurons_layer1)
	layer2 = tf.layers.dense(layer1, neurons_layer2)
	output_layer = tf.layers.dense(layer2, 10)			#layer giving output
	return output_layer

def model_fn(features, labels, mode):

    # build the neural network

    logits = neural_network(features)

    # prediction

    pred_probability = tf.nn.softmax(logits)
    pred_class = tf.argmax(logits, 1)

    # if predict mode

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_class)

        # cost function

    cost = \
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                       labels=tf.transpose(tf.cast(labels,tf.int32))))

       # minimizing cost function

    optimizer = \
        tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=tf.train.get_global_step())

       # finding correct predictions

    correct = tf.metrics.accuracy(labels=labels, predictions=pred_class)

    return tf.estimator.EstimatorSpec(mode=mode,
            predictions=pred_class, loss=cost, train_op=optimizer,
            eval_metric_ops={'accuracy': correct})

model = tf.estimator.Estimator(model_fn)

model.train(input_fn, steps=steps)

#define the input function for testing
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"input": mnist.test.images}, y=mnist.test.labels,
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])








