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
# np.random.seed(7)


def build_dnn():
    model = keras.Sequential()

    model.add(layers.Dense(128, activation = 'relu', input_shape = (20,)))
    model.add(Dropout(0.2))

    model.add(layers.Dense(128, activation = 'relu'))
    model.add(Dropout(0.2))

    model.add(layers.Dense(64, activation = 'relu'))
    model.add(Dropout(0.2))

    model.add(layers.Dense(5, activation = 'softmax'))
    model.compile(loss = 'sparse_categorical_crossentropy',
                    optimizer = keras.optimizers.Adam(),
                    metrics = ['sparse_categorical_accuracy'])
    return model