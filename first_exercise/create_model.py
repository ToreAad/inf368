# Train a convolutional neural network
# Used this as guide: https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5
# Another good background: https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/

import numpy as np
import random as rn
import os.path
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.utils import to_categorical
from split_data import X_train, y_train

rn.seed(42)
np.random.seed(42)

def conv_reshaper(values):
    return values.reshape(len( values),28,28,1)

def create_model():    
    cnn = Sequential()
    cnn.add(Conv2D(100, kernel_size=3, activation="relu", input_shape=(28,28,1)))
    cnn.add(Conv2D(50, kernel_size=3, activation='relu'))
    cnn.add(Conv2D(100, kernel_size=3, activation='relu'))
    cnn.add(Flatten())
    cnn.add(Dense(10, activation='softmax'))
    cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    cnn.fit(conv_reshaper(X_train), to_categorical(y_train), epochs=10, batch_size=int(len(X_train)*0.01))

    cnn.save('./output/first_exercise.h5')
    with open("./output/first_exercise.json", 'w+') as f:
        f.write(cnn.to_json())

if __name__ == "__main__":
    create_model()
