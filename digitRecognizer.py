import tensorflow as tf
import numpy as np
from tensorflow import keras

data = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = data.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

for train in range(len(x_train)):
    for row in range(28):
        for x in range(28):
            if x_train[train][row][x] != 0:
                x_train[train][row][x] = 1


# Making the model 
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.fit(x_train, y_train, epochs=10)
model.save("m.model")

