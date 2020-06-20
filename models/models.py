from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow import test
import math


# Config Tensorflow to run on CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Run Tensorflow on CPU only

# Fix random seed for reproducibility
np.random.seed(7)


# Test if GPU is present
if test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")


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
    classifier.add(Dense(5, activation='softmax'))

    # Compile the model
    classifier.compile(optimizer=optimizers.Adam(),
                       loss='sparse_categorical_crossentropy',
                       metrics=['sparse_categorical_accuracy'])

    # Summary
    print(classifier.summary())


    # Fit
    classifier.fit(X_train, y_train, epochs=50, batch_size=32)