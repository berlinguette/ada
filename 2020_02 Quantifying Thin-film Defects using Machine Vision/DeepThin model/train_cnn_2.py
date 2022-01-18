import os
from PIL import Image
import pickle
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import optimizers, regularizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with open('C:\\Users\\Nina\\PycharmProjects\\test\\venv\\X_Train2.data', 'rb') as filehandle:
    # read the data as binary data stream
    X_Train = pickle.load(filehandle)

with open('C:\\Users\\Nina\\PycharmProjects\\test\\venv\\Y_Train2.data', 'rb') as filehandle:
    # read the data as binary data stream
    Y_Train = pickle.load(filehandle)

with open('C:\\Users\\Nina\\PycharmProjects\\test\\venv\\Meta2.data', 'rb') as filehandle:
    # read the data as binary data stream
    Meta = pickle.load(filehandle)

X_Train = X_Train.reshape(X_Train.shape[0], X_Train.shape[1], X_Train.shape[2], 1)

X_Train = X_Train/255
Y_Train = to_categorical(Y_Train)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                 kernel_regularizer=regularizers.l2(0.001),
                 input_shape=(X_Train.shape[1],
                 X_Train.shape[2], 1),
                 padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.001),
                 padding="same"))
model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.001),
                 padding="same"))
# model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.001),
#                 padding="same"))

#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

#model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.001),
#                 padding="same"))
#model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.001),
#                 padding="same"))

# model.add(Conv2D(16, kernel_size=3, activation='relu'))

model.add(Flatten())

model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

sgd = optimizers.SGD(lr=0.01, momentum=0.0001, nesterov=True)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])

# model.fit(X_Train, Y_Train, validation_data=(X_Train, Y_Train), epochs=3)
X_test = X_Train[0:291]
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
y_test = Y_Train[0:291]
model.fit(X_Train, Y_Train,validation_data=(X_Train, Y_Train), epochs=30, batch_size=10, shuffle=True)

model.save('my_model.h5')
