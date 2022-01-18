import os
from PIL import Image
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.utils import to_categorical

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with open('C:\\Users\\Nina\\PycharmProjects\\test\\venv\\X_Train.data', 'rb') as filehandle:
    # read the data as binary data stream
    X_Train = pickle.load(filehandle)

with open('C:\\Users\\Nina\\PycharmProjects\\test\\venv\\Y_Train.data', 'rb') as filehandle:
    # read the data as binary data stream
    Y_Train = pickle.load(filehandle)

with open('C:\\Users\\Nina\\PycharmProjects\\test\\venv\\Meta.data', 'rb') as filehandle:
    # read the data as binary data stream
    Meta = pickle.load(filehandle)

X_Train = X_Train/255
X_Train = X_Train.reshape(X_Train.shape[0], X_Train.shape[1], X_Train.shape[2], 1)


Y_Train = to_categorical(Y_Train)

model = keras.models.load_model('C:\\Users\\Nina\\PycharmProjects\\test\\venv\\best_model.h5')
model.summary()

# score = model.evaluate(X_Train, Y_Train, verbose=0)

# loss, acc = model.evaluate(X_Train, Y_Train)
print("Trained model, accuracy: {:5.2f}%".format(100*acc))

Y_Predict = model.predict(X_Train)

jpgfile = Image.open('C:\\Deep Learn\\Images\\7.jpg').convert('L')
aspect = jpgfile.size[0] / jpgfile.size[1]
image_height = 256
out = jpgfile.resize((int(aspect * image_height), image_height), Image.ANTIALIAS)
pixels = np.asarray(out)
pixels=pixels/255
pixels=pixels.reshape(1, pixels.shape[0], pixels.shape[1], 1)
Y_Predict = model.predict(pixels)
