# Tasks: Experiment with different hyperparameter settings (e.g. learning rate, early stopping, batch sizes, etc)
# and observe their effects on training time, overfitting, and final accuracy.
# See if L2 regularisation has effect on network using relu activation function.
# Compare contrast with relu + dropout.

# Train a convolutional neural network
# Used this as guide: https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5
# Another good background: https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/
# For tensorboard: http://fizzylogic.nl/2017/05/08/monitor-progress-of-your-keras-based-neural-network-using-tensorboard/


import numpy as np
import random as rn
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.utils import to_categorical
from split_data import X_train, y_train, X_validation, y_validation
from keras.callbacks import TensorBoard
import datetime
import os

rn.seed(42)
np.random.seed(42)

def fit_and_save_model(name, model, epochs = 5, batch_size=64, validation_split = 1.0/12, eval_batch_size=128):
    # type: tensorboard --logdir=path-to-output to monitor training progress.
    required_folders = './output/{}/logs/'.format(name)
    if not os.path.exists(required_folders):
        os.makedirs(required_folders)

    tensorboard = TensorBoard('./output/{}/logs/{}'.format(name, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")))
    history = model.fit(X_train, to_categorical(y_train), 
                        validation_split=validation_split, 
                        epochs=epochs,
                        batch_size=batch_size, 
                        verbose=1, 
                        callbacks=[tensorboard])
    
    plt.plot(history.history['loss'], label='Training data loss')
    plt.plot(history.history['val_loss'], label='Validation data loss')
    plt.legend()
    plt.title("Model {} loss".format(name))
    plt.savefig('./output/{}/model_loss.png'.format(name))

    score = model.evaluate(X_validation, y_validation, batch_size = eval_batch_size)
    
    model.save('./output/{}/{}.h5'.format(name, name))
    with open("./output/{}/{}.json".format(name, name), 'w+') as f:
        f.write(model.to_json())
    with open("./output/{}/{}.score".format(name, name), 'w+') as f:
        f.write("Validation loss: {}".format(score['loss']))
        f.write("Validation accuracy: {}".format(score['acc']))


def create_convolutional_model():
    model = Sequential()
    model.add(Conv2D(50, 
                     kernel_size=3, 
                     activation="relu", 
                     input_shape=(28, 28, 1)))
    model.add(Conv2D(100, 
                     kernel_size=3, 
                     activation='relu'))
    model.add(Conv2D(250, 
                     kernel_size=3, 
                     activation='relu'))
    model.add(Conv2D(500, 
                     kernel_size=3, 
                     activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def create_simple_classifier():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer="SGD", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    fit_and_save_model("mnist_simple", create_simple_classifier())
    # fit_and_save_model("mnist_convolutional", create_convolutional_model(), 
    #                    batch_size=int(len(X_train)*0.001),
    #                    epochs=20)
