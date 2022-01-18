import os
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# The last number is 1, which signifies that the images are greyscale.
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

# ‘one-hot-encode’: sixth number in our array will have a 1 and the rest of the array will be filled with 0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()

# 64 in the first layer and 32 in the second layer are the number of nodes in each layer
# Kernel size is the size oth the filted matrix for our convolution

model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))

# Flatten serves as a connection between the convolution and dense layers
model.add(Flatten())

# We will have 10 nodes in our output layer, one for each possible outcome
model.add(Dense(10, activation='softmax'))

# The optimizer controls the learning rate
#  ‘categorical_crossentropy’ for our loss function. This is the most common choice for classification
# ‘accuracy’ metric to see the accuracy score on the validation set when we train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(X_train, y_train, batch_size=50, validation_data=(X_test, y_test), epochs=3)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

# predict first 4 images in the test set
model.predict(X_test[:4])

# actual results for first 4 images in test set
y_test[:4]


