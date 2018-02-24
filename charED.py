from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
from keras.utils.vis_utils import plot_model

#training batch size
train_batch_size = 50
#No of epochs
epochs = 20
#Number of units in each LSTM layer in encoder
LSTM_encoder_units = 128
#Number of units in each LSTM layer in decoder
LSTM_decoder_units = 128
#training_set_size
training_set_size = 10000
#start sequence character
start_char = "\t"
#end sequence character
end_char = "\n"

filename = "/Users/ujjawal/desktop/fra-eng/fra.txt"

input_words = []
target_words = []
input_chars = []
target_chars = []

with open(filename,"r") as fopen:
	total_lines = fopen.read().split("\n")
	for line in total_lines[0:training_set_size]:
		[input_word,target_word] = line.split("\t")
		#attach start andd end tokens to target word
		target_word = start_char + target_word + end_char
		print(target_word)
		#increment list of input and target words
		input_words.append(input_word)
		target_words.append(target_word)
		#increment list of input and target characters
		for char in input_word:
			input_chars.append(char)
		for char in target_word:
			target_chars.append(char)

input_chars = set(input_chars)
target_chars = set(target_chars)

#number of unique encoder input tokens
num_encoder_tokens = len(input_chars)
#number of unique decoder output tokens
num_decoder_tokens = len(target_chars)
#maximum encoder input sequence length
max_input_seqlen = max((len(word) for word in input_words))
#maximum decoder output sequence length
max_output_seqlen = max((len(word) for word in target_words))

#mapping input and output tokens to an index
encoder_tokens_ind = dict(((char,i) for (i,char) in enumerate(input_chars)))
decoder_tokens_ind = dict(((char,i) for (i,char) in enumerate(target_chars)))

#encoder inputs 
encoder_inputs = np.zeros((training_set_size,max_input_seqlen,num_encoder_tokens))
#decoder inputs
decoder_inputs = np.zeros((training_set_size,max_output_seqlen,num_decoder_tokens))
#decoder outputs
decoder_outputs = np.zeros((training_set_size,max_output_seqlen,num_decoder_tokens))

for i in xrange(len(input_words)):
	for j in xrange(len(input_words[i])):
		#make the element corresponding to the jth character of the ith word equal to 1
		encoder_inputs[i,j,encoder_tokens_ind[input_words[i][j]]] = 1.0 
	for j in xrange(len(target_words[i])):
		decoder_inputs[i,j,decoder_tokens_ind[target_words[i][j]]] = 1.0
		if(j > 0):
			decoder_outputs[i,j-1,decoder_tokens_ind[target_words[i][j]]] = 1.0

#making the encoder decoder model
#None represents the length of the phrase, each character is represented by num_encoder tokens
#build encoder
encoder_input_seq = Input(shape = (None,num_encoder_tokens)) 
encoder_LSTM = LSTM(LSTM_encoder_units, return_state = True)
outputs,state1,state2 = encoder_LSTM(encoder_input_seq)
#find the encoder states
encoder_states = [state1,state2]

#build decoder
decoder_input_seq = Input(shape = (None,num_decoder_tokens))
decoder_LSTM = LSTM(LSTM_decoder_units, return_state = True, return_sequences = True)
#return states of decoder are not required during training
decoder_LSTM_outputs,_,_ = decoder_LSTM(decoder_input_seq,initial_state = encoder_states)
decoder_output_layer = Dense(num_decoder_tokens,activation = "softmax")
decoder_LSTM_outputs = decoder_output_layer(decoder_LSTM_outputs)

#build the model
model = Model([encoder_input_seq,decoder_input_seq],decoder_LSTM_outputs)

#plot the model
#plot the built model
plot_model(model, to_file='model_train.png', show_shapes = True)

#training the model
model.compile(optimizer = "rmsprop", loss = "categorical_crossentropy")
model.fit([encoder_inputs,decoder_inputs],decoder_outputs, batch_size = train_batch_size, epochs = epochs, validation_split = 0.2)

#define a new model for testing
#define the state inputs to the decoder
#define the encoder model to get the states of the encoder model to feed into decoder
encoder_model = Model(encoder_input_seq,encoder_states)

decoder_input_s1 = Input(shape = (LSTM_encoder_units,))
decoder_inputs_s2 = Input(shape = (LSTM_encoder_units,))
decoder_input_states = [decoder_input_s1,decoder_inputs_s2]
#we need both the decoder outputs and the decoder states now
decoder_LSTM_outputs,decoder_state1,decoder_state2 = decoder_LSTM(decoder_input_seq,initial_state = decoder_input_states)
decoder_output_states = [decoder_state1,decoder_state2]
decoder_LSTM_outputs = decoder_output_layer(decoder_LSTM_outputs)
#define the decoder model
#decoder takes as input both the sequence and decoder states and gives as output sequences and states
decoder_model = Model([decoder_input_seq] + decoder_input_states,[decoder_LSTM_outputs] + decoder_output_states)

#plot the models
plot_model(encoder_model, to_file='model_infer_encoder.png', show_shapes = True)
plot_model(decoder_model, to_file='model_infer_decoder.png', show_shapes = True)


#define the reverse mapping from chararter indices to characters
rev_encoder_ind = dict((i,char) for (char,i) in encoder_tokens_ind.items())
rev_decoder_ind = dict((i,char) for (char,i) in decoder_tokens_ind.items())




#define the starting target sequence character
def get_sentence(encoder_input_seq):
	#get the states from the encoder model
	encoder_output_states = encoder_model.predict(encoder_input_seq)

	output_seq = ""
	first = True
	while True:
		#define the starting target sequence character
		if(first):
			target_start_seq = np.zeros((1,1,num_decoder_tokens))
			target_start_seq[0,0,decoder_tokens_ind["\t"]] = 1
		decoder_output_tokens,output_state1,output_state2 = decoder_model.predict([target_start_seq] + encoder_output_states)
		#get the output character
		output_token = np.argmax(decoder_output_tokens[0,-1,:])

		output_char = rev_decoder_ind[output_token]
		#if output character is end character or length reaches max value
		if(output_char == "\n" or len(output_seq) > max_output_seqlen):
			break
		#add the character to the output sequence
		output_seq += output_char
		first = False
		#update the starting target sequence for next step (teacher forcing)
		target_start_seq = np.zeros((1,1,num_decoder_tokens))
		target_start_seq[0,0,output_token] = 1	
		#update the states too 
		encoder_output_states = [output_state1,output_state2]
	return output_seq

ind = 20
encoder_input_seq = encoder_inputs[ind:ind+1]
output_sentence = get_sentence(encoder_input_seq)
print("Input Sentence :" ,input_words[ind])
print("Output Sentence :" ,output_sentence)










