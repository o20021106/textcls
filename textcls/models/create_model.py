from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Sequential


def create_model(embedding_dim, maxlen, num_labels, word_index, dropout=0.2, filters=250, kernel_size=3, strides=1, 
                 pool_size=3, matrics='acc'):
    model = Sequential()
    model.add(Embedding(len(word_index) + 1, embedding_dim, input_length=maxlen))
    model.add(Dropout(dropout))
    model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=strides))
    model.add(MaxPooling1D(pool_size))
    model.add(Flatten())
    model.add(Dense(embedding_dim, activation='relu'))
    model.add(Dense(num_labels, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=[matrics])
    return model

