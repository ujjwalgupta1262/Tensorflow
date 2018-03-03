import tensorflow as tf

START_ID=0
PAD_ID=1
END_ID=2
global_sess = tf.Session()
class PointerWrapper(tf.contrib.seq2seq.AttentionWrapper):
  """Customized AttentionWrapper for PointerNet."""

  def __init__(self,cell,attention_size,memory,initial_cell_state=None,name=None):
    # In the paper, Bahdanau Attention Mechanism is used
    # We want the scores rather than the probabilities of alignments
    # Hence, we customize the probability_fn to return scores directly
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(attention_size, memory, probability_fn=lambda x: x )
    # According to the paper, no need to concatenate the input and attention
    # Therefore, we make cell_input_fn to return input only
    cell_input_fn=lambda input, attention: input
    # Call super __init__
    super(PointerWrapper, self).__init__(cell,
                                         attention_mechanism=attention_mechanism,
                                         attention_layer_size=None,
                                         alignment_history=False,
                                         cell_input_fn=cell_input_fn,
                                         output_attention=True,
                                         initial_cell_state=initial_cell_state,
                                         name=name)
  @property
  def output_size(self):
    return self.state_size.alignments

  def call(self, inputs, state):
    _, next_state = super(PointerWrapper, self).call(inputs, state)
    return next_state.alignments, next_state
 

class PointerNet(object):
  """ Pointer Net Model
  
  This class implements a multi-layer Pointer Network 
  aimed to solve the Convex Hull problem. It is almost 
  the same as the model described in this paper: 
  https://arxiv.org/abs/1506.03134.
  """

  def __init__(self, batch_size=128, num_features = 2, max_input_seqlen=5, max_output_seqlen=7, 
              LSTM_units=128, attention_size=128,learning_rate=0.001, max_gradient_norm=5):
    self.batch_size = batch_size
    self.max_input_sequence_len = max_input_seqlen
    self.max_output_sequence_len = max_output_seqlen
    self.learning_rate = learning_rate

    self.vocab_size = max_input_seqlen+3
    cell = tf.contrib.rnn.LSTMCell
    # parameters
    self.inputs = tf.placeholder(tf.float32, shape=[self.batch_size,self.max_input_sequence_len,num_features], name="inputs")
    self.outputs = tf.placeholder(tf.int32, shape=[self.batch_size,self.max_output_sequence_len+1], name="outputs")
    self.enc_input_weights = tf.placeholder(tf.int32,shape=[self.batch_size,self.max_input_sequence_len], name="enc_input_weights")
    self.dec_input_weights = tf.placeholder(tf.int32,shape=[self.batch_size,self.max_output_sequence_len], name="dec_input_weights")
    # Calculate the lengths
    enc_input_lens=tf.reduce_sum(self.enc_input_weights,axis=1)
    dec_input_lens=tf.reduce_sum(self.dec_input_weights,axis=1)
    # Special token embedding
    special_token_embedding = tf.get_variable("special_token_embedding", [3,num_features], tf.float32, tf.contrib.layers.xavier_initializer())
    # Embedding_table
    # Shape: [batch_size,vocab_size,features_size]
    embedding_table = tf.concat([tf.tile(tf.expand_dims(special_token_embedding,0),[self.batch_size,1,1]), self.inputs],axis=1)   
    # Unstack embedding_table
    # Shape: batch_size*[vocab_size,features_size]
    embedding_table_list = tf.unstack(embedding_table, axis=0)
    # Unstack outputs
    # Shape: (max_output_sequence_len+1)*[batch_size]
    outputs_list = tf.unstack(self.outputs, axis=1)
    # targets
    # Shape: [batch_size,max_output_sequence_len]
    self.targets = tf.stack(outputs_list[1:],axis=1)
    # decoder input ids 
    # Shape: batch_size*[max_output_sequence_len,1]
    dec_input_ids = tf.unstack(tf.expand_dims(tf.stack(outputs_list[:-1],axis=1),2),axis=0)
    # encoder input ids 
    # Shape: batch_size*[max_input_sequence_len+1,1]
    enc_input_ids = [tf.expand_dims(tf.range(2,self.vocab_size),1)]*self.batch_size
    # Look up encoder and decoder inputs
    encoder_inputs = []
    decoder_inputs = []
    for i in range(self.batch_size):
      encoder_inputs.append(tf.gather_nd(embedding_table_list[i], enc_input_ids[i]))
      decoder_inputs.append(tf.gather_nd(embedding_table_list[i], dec_input_ids[i]))
    # Shape: [batch_size,max_input_sequence_len+1,2]
    encoder_inputs = tf.stack(encoder_inputs,axis=0)
    # Shape: [batch_size,max_output_sequence_len,2]
    decoder_inputs = tf.stack(decoder_inputs,axis=0)
    #equal number of LSTM units are used for both encoder and decoder
    forward_cell = cell(LSTM_units)
    backward_cell = cell(LSTM_units)    
     #get outputs from encoder to feed into the attention mechanism
    memory,_ = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, encoder_inputs, dtype=tf.float32)
    memory = tf.concat(memory, 2) 
   	#create a pointer cell using a LSTM cell 
	#memory is inputted from the outputs of encoder
    pointer_cell = PointerWrapper(cell(LSTM_units), attention_size, memory)
    #decoder cell
    decoder_cell = pointer_cell
    cur_batch_max_len = tf.reduce_max(dec_input_lens)
    # Training Helper
    helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs, dec_input_lens)    
    # Basic Decoder
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_cell.zero_state(self.batch_size,tf.float32)) 
    # Decode
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,impute_finished=True)
    # logits
    logits = outputs.rnn_output
    # predicted_ids_with_logits
    self.predicted_ids_with_logits=tf.nn.top_k(logits)
    # Pad logits to the same shape as targets
    logits = tf.concat([logits,tf.ones([self.batch_size,self.max_output_sequence_len-cur_batch_max_len,self.max_input_sequence_len+1])],axis=1)
    self.shifted_targets = (self.targets - 2)*self.dec_input_weights
      # cost
    cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.shifted_targets, logits=logits)
      # Total loss
    self.loss = tf.reduce_sum(cost*tf.cast(self.dec_input_weights,tf.float32))/self.batch_size
      # Get all trainable variables
    parameters = tf.trainable_variables()
      # Calculate gradients
    gradients = tf.gradients(self.loss, parameters)
      # Clip gradients
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
      # Optimization
    optimizer = tf.train.AdamOptimizer(self.learning_rate)


  def step(self, session, inputs, enc_input_weights, outputs=None, dec_input_weights=None):
	feed_dict = {self.inputs : inputs , self.enc_input_weights : enc_input_weights, self.dec_input_weights : dec_input_weights, self.outputs : outputs}
	pred = session.run(self.predicted_ids_with_logits, feed_dict = feed_dict)
	loss = session.run(self.loss, feed_dict = feed_dict)
	return (pred,loss)


model = PointerNet()
global_sess.run(tf.global_variables_initializer())
inputs = [[0,1],[0,1],[0,1],[0,1],[0,1]]
enc_input_weights = [1,2,3,4,5]
dec_input_weights = [1,2,3,4,5,6,7]
outputs = [1,2,3,4,5,6,7,8]
final_enc_input_weights = []
final_inputs = []
final_dec_input_weights = []
final_outputs = []
for i in xrange(128):
	final_inputs.append(inputs)
	final_enc_input_weights.append(enc_input_weights)
	final_dec_input_weights.append(dec_input_weights)
	final_outputs.append(outputs)
print(model.step(session = global_sess, inputs = final_inputs, enc_input_weights = final_enc_input_weights, dec_input_weights = final_dec_input_weights, outputs = final_outputs))