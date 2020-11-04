import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
# FOR TESTING

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_test = tf.keras.utils.normalize(x_test, axis=1)
for test in range(len(x_test)):
    for row in range(28):
        for x in range(28):
            if x_test[test][row][x] != 0:
                x_test[test][row][x] = 1


for i in range(10):
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.show()