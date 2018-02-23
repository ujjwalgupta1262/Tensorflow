import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

training_set_size = 5000
((x_train, y_train), (x_test, y_test)) = imdb.load_data(num_words = training_set_size)

max_input_length = 500
x_train = sequence.pad_sequences(x_train, maxlen = max_input_length)
x_test = sequence.pad_sequences(x_test, maxlen = max_input_length)

#building the model
embedding_vector_length = 50
model = Sequential()
model.add(Embedding(training_set_size, embedding_vector_length, input_length = max_input_length))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#running the model
model.fit(x_train, y_train, epochs = 3, batch_size=64)

#testing
test = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: " , (test[1]*100))