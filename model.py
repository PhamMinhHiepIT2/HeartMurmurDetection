from pyexpat import model
from keras import optimizers, losses, activations, models
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D
import tensorflow as tf


# fit model with 2D features and labels from challenge data
def fit_model(features, labels, verbose):
    # define model
    model = models.Sequential()
    model.add(Convolution1D(32, 3, activation=activations.relu,
              input_shape=(features.shape[1], 1)))
    model.add(MaxPool1D(2))
    model.add(Convolution1D(32, 3, activation=activations.relu))
    model.add(MaxPool1D(2))
    model.add(Convolution1D(32, 3, activation=activations.relu))
    model.add(GlobalMaxPool1D())
    model.add(Dense(2, activation=activations.softmax))

    # compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                    loss=losses.categorical_crossentropy,
                  metrics=['accuracy'])

    # fit model
    model.fit(features, labels, epochs=10, verbose=verbose)

    return model
