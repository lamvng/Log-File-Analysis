import settings
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

settings.init()


path = "{}/saved_model/model.h5".format(settings.root)
loaded_model = load_model(path)
loaded_model.summary()




# y_predict = loaded_model.predict(X_test)