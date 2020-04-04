from __future__ import absolute_import, division, print_function, unicode_literals
import collections
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from sklearn.metrics import mean_squared_error
import math







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



