from __future__ import absolute_import, division, print_function, unicode_literals
import collections
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, GRU, Bidirectional
from sklearn.metrics import mean_squared_error
import math


# fix random seed for reproducibility
np.random.seed(7)


def build_LSTM(X_train, y_train):
    # Get your input dimensions
    # Input length is the length for one input sequence (i.e. the number of rows for your sample)
    # Input dim is the number of dimensions in one input vector (i.e. number of input columns)
    input_length = X_train.shape[1]
    input_dim = X_train.shape[2]
    # Output dimensions is the shape of a single output vector
    # In this case it's just 1, but it could be more
    output_dim = 0

    classifier = Sequential()

    classifier.add(LSTM(units=50,
                        return_sequences=True,
                        input_dim=input_dim,
                        input_length=input_length))
    classifier.add(Dropout(0.2))

    # Second LSTM layer
    classifier.add(LSTM(units=50, return_sequences=True))
    classifier.add(Dropout(0.2))

    # Third LSTM layer
    classifier.add(LSTM(units=50, return_sequences=True))
    classifier.add(Dropout(0.2))

    # Fourth LSTM layer
    classifier.add(LSTM(units=50))
    classifier.add(Dropout(0.2))

    # The output layer
    classifier.add(Dense(units=5, activation='softmax'))

    # Compile the model
    classifier.compile(optimizer=optimizers.Adam(),
                       loss='mean_squared_error',
                       metrics=['accuracy'])

    # Summary
    print(classifier.summary())


    # Fit
    classifier.fit(X_train, y_train, epochs=50, batch_size=32)


def build_model():
    model = keras.Sequential()
    model.add(layers.Dense(32, activation = tf.nn.relu, input_shape = (6,)))
    model.add(layers.Dense(32, activation = tf.nn.relu))
    model.add(layers.Dense(1))
    optimizer_function = keras.optimizers.Adam()
    model.compile(loss = 'mean_absolute_error',\
        optimizer = optimizer_function,\
        metrics = ['mean_absolute_error', 'mean_squared_error'])
    return model

def build_linear():
    model = keras.Sequential([layers.Dense(1, activation = 'linear', input_shape = (6,))])
    optimizer_function = keras.optimizers.SGD()
    model.compile(loss = 'mean_absolute_error',\
        optimizer = optimizer_function,\
        metrics = ['mean_absolute_error', 'mean_squared_error'])
    return model


# Train and save model
def train(model, X_train, y_train):
    csv_logger = keras.callbacks.CSVLogger('training.log', separator = ',', append = False)
    history = model.fit(X_train, y_train, epochs = 1000, callbacks=[csv_logger])
    model.save('model.h5')
    return history



