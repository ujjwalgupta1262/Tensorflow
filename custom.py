from keras import initializers
from keras.activations import tanh, softmax
from keras.layers import LSTM, TimeDistributed, Dense,Input
from keras.engine import InputSpec
import keras.backend as K
from keras.models import Model

class PointerLSTM(LSTM):
	def __init__(self, num_units, *args, **kwargs):
		self.num_units = num_units
		self.state_size = num_units
		#self.kernel_initializer = initializers.get("orthogonal")
		super(PointerLSTM, self).__init__(num_units,*args, **kwargs)

	def build(self, input_shape):
		self.input_spec = [InputSpec(shape=input_shape)]
		self.W1 = self.add_weight(shape = (self.state_size , 1), name = "W1", initializer = self.kernel_initializer)
		self.W2 = self.add_weight(shape = (self.state_size , 1), name = "W2", initializer = self.kernel_initializer)
		self.vt = self.add_weight(shape = (input_shape[1] , 1), name = "vt", initializer = self.kernel_initializer)

	def call(self, inputs, states, mask = None, constants = None):
		input_shape = self.input_spec[0].shape
		en_seq = inputs
		timesteps = input_shape[1]
		inputs = inputs[:, input_shape[1] - 1, :]
		inputs = K.repeat(inputs, input_shape[1])
		constants = states[-self._num_constants:]
		constants.append(inputs)
		initial_states = self.get_initial_states(x_input)
		last_output, outputs, states = K.rnn(self.step, inputs,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             constants=constants,
                                             input_length=input_shape[1])


	def step(self,inputs,states):
		input_shape = self.input_spec[0].shape
		states = states[:-self._num_constants]
		en_seq = states[-1]
		_, [h, c] = super(PointerLSTM, self).call(x_input, states[:-1])
		dec_seq = K.repeat(h, input_shape[1])
		Eij = K.dot(self.W1, en_seq)
		Dij = K.dot(self.W2, dec_seq)
		U = self.vt * tanh(Eij + Dij)
		U = K.squeeze(U, 2)
		pointer = softmax(U)
		return pointer, [h, c]

	def get_output_shape_for(self, input_shape):
		return (input_shape[0], input_shape[1], input_shape[1])

	
hidden_size = 128
seq_len = 10
nb_epochs = 10
learning_rate = 0.1

print("building model...")
main_input = Input(shape=(seq_len, 2), name='main_input')

encoder = LSTM(units = hidden_size, return_sequences = True, name="encoder")(main_input)
decoder = PointerLSTM(hidden_size, name = "decoder")(encoder)

model = Model(input=main_input, output=decoder)
model.compile(optimizer='adadelta',
              loss='categorical_crossentropy',
              metrics=['accuracy'])			



