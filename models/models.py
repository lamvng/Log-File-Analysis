from __future__ import absolute_import, division, print_function, unicode_literals
import os
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout


# Config Tensorflow to run on CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Run Tensorflow on CPU only

# Fix random seed for reproducibility
# np.random.seed(7)


def build_dnn():
    model = keras.Sequential()

    model.add(Dense(128, activation = 'relu', input_shape = (20,)))
    model.add(Dropout(0.2))

    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.2))

    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.2))

    model.add(Dense(5, activation = 'softmax'))
    model.compile(loss = 'sparse_categorical_crossentropy',
                  optimizer = keras.optimizers.Adam(),
                  metrics = ['sparse_categorical_accuracy'])
    return model


def train(model, X_train, y_train):
    csv_logger = keras.callbacks.CSVLogger('training.log', separator = ',', append = False)
    history = model.fit(X_train, y_train, validation_split = 0.2, epochs = 150, callbacks=[csv_logger])
    model.save('model.h5')
    return history


