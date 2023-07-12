from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding


def model_glove(max_features, dim,  maxlen, embedding_matrix):
    model = Sequential()
    model.add(Embedding(max_features, dim, input_length=maxlen))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False

    model.summary()

    return model


def model_simple_rnn():
    from keras.layers import SimpleRNN

    model = Sequential()
    model.add(Embedding(10000, 32))
    model.add(SimpleRNN(32, return_sequences=True))
    model.add(SimpleRNN(32, return_sequences=True))
    model.add(SimpleRNN(32, return_sequences=True))
    model.add(SimpleRNN(32))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    
    return model

def model_simple_lstm():
    from keras.layers import LSTM

    model = Sequential()
    model.add(Embedding(10000, 32))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    
    return model

def model_bidirectional_lstm():
    from keras.layers import LSTM, Bidirectional

    model = Sequential()
    model.add(Embedding(10000, 32))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    
    return model

def model_mult_bidirectional_lstm():
    from keras.layers import LSTM, Bidirectional

    model = Sequential()
    model.add(Embedding(10000, 32))
    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

    return model

def model_conv1_gru(input_data):
    from keras.optimizers import RMSprop
    from keras.layers import Conv1D, MaxPooling1D, GRU
    

    model = Sequential()
    model.add(Conv1D(32, 5, activation='relu',
                        input_shape=(None, input_data.shape[-1])))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(32, 5, activation='relu'))
    model.add(GRU(32, dropout=0.1, recurrent_dropout=0.5))
    model.add(Dense(1))
    model.summary()

    model.compile(optimizer=RMSprop(), loss='mae')

    return model

def model_conv1(vocab_size, embeding_dim, max_length):
    from keras.optimizers import RMSprop
    from keras.layers import Conv1D, GlobalMaxPooling1D, LSTM
    

    model = Sequential()
    model.add(Embedding(vocab_size, embeding_dim, input_length=max_length))
    model.add(Conv1D(128, kernel_size=5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64))
    model.add(Dense(1))
    model.summary()
    
    model.compile(optimizer=RMSprop(), loss='mae')

    return model