import numpy as np

import tensorflow_datasets

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras import Sequential
from keras.layers import Embedding, AveragePooling2D, Flatten, Dense
from keras.optimizers import Adam

from deep_learning.plot import embedding_projector
from deep_learning.Utils import preprocesing_tensors

VOCAB_SIZE = 10000
EMBENDING_DIM = 16
MAX_LENGTH = 120
TRUNC_TYPE = 'post'
OVV_TOKEN = '<OOV>'

imbd, info = tensorflow_datasets.load('imdb_reviews', with_info=True, as_supervised=True)
train, test = imbd['train'], imbd['test']

train_seqs, train_labels = preprocesing_tensors(train)
test_seqs, test_labels = preprocesing_tensors(test)

tokenizer = Tokenizer(num_words=VOCAB_SIZE,
                      oov_token=OVV_TOKEN)

tokenizer.fit_on_texts(train_seqs)

word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(train_seqs)
padded = pad_sequences(sequences=sequences,
                       maxlen=MAX_LENGTH,
                       truncating=TRUNC_TYPE)

test_seq = tokenizer.texts_to_sequences(test_seqs)
test_padded = pad_sequences(sequences=test_seq,
                            maxlen=MAX_LENGTH)

model = Sequential()
model.add(Embedding(VOCAB_SIZE, EMBENDING_DIM, input_length=MAX_LENGTH))
model.add(Flatten())
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['acc'])

model.summary()

hist = model.fit(padded,
          train_labels,
          epochs=10,
          validation_data=(test_padded, test_labels))

embeding_layer = model.layers[0]

embeding_weights = embeding_layer.get_weights()[0]
print(embeding_weights.shape)

reverse_word_idx = tokenizer.index_word

embedding_projector(VOCAB_SIZE, tokenizer, embeding_weights)