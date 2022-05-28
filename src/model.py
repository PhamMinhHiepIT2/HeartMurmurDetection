from pyexpat import model
from keras import optimizers, losses, activations, models
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D
import tensorflow as tf
from sklearn import svm
from tensorflow.keras.applications.resnet50 import ResNet50


# convolution classification models with features shape (942, 26), labels shape (942, 2) and (942, 3)
def get_model(num_classes, features_shape, activation='relu', dropout=0.5, kernel_size=3, pool_size=2,
              filters=32, dense_size=64, dense_activation='relu', dense_dropout=0.5, verbose=0):
    if verbose >= 1:
        print('Building model...')

    # Input layer.
    input_layer = Input(shape=features_shape)

    # Convolutional layers.
    conv_layer = Convolution1D(filters=filters, kernel_size=kernel_size, activation=activation,
                               padding='same')(input_layer)
    conv_layer = MaxPool1D(pool_size=pool_size)(conv_layer)
    conv_layer = Dropout(dropout)(conv_layer)

    # Dense layers.
    dense_layer = Dense(dense_size, activation=dense_activation)(conv_layer)
    dense_layer = Dropout(dense_dropout)(dense_layer)

    # Output layer.
    output_layer = Dense(num_classes, activation='softmax')(dense_layer)

    # Build model.
    model = models.Model(inputs=input_layer, outputs=output_layer)

    if verbose >= 1:
        print('Done.')

    return model


# VGG16 model
def get_vgg16_model(num_classes, features_shape, activation='relu', dropout=0.5, kernel_size=3, pool_size=2,
                    filters=32, dense_size=64, dense_activation='relu', dense_dropout=0.5, verbose=0):
    if verbose >= 1:
        print('Building model...')

    # Input layer.
    input_layer = Input(shape=features_shape)

    # Convolutional layers.
    conv_layer = Convolution1D(filters=filters, kernel_size=kernel_size, activation=activation,
                               padding='same')(input_layer)
    conv_layer = MaxPool1D(pool_size=pool_size)(conv_layer)
    conv_layer = Dropout(dropout)(conv_layer)

    conv_layer = Convolution1D(filters=filters, kernel_size=kernel_size, activation=activation,
                               padding='same')(conv_layer)
    conv_layer = MaxPool1D(pool_size=pool_size)(conv_layer)
    conv_layer = Dropout(dropout)(conv_layer)

    conv_layer = Convolution1D(filters=filters, kernel_size=kernel_size, activation=activation,
                               padding='same')(conv_layer)
    conv_layer = MaxPool1D(pool_size=pool_size)(conv_layer)
    conv_layer = Dropout(dropout)(conv_layer)

    # Dense layers.
    dense_layer = Dense(dense_size, activation=dense_activation)(conv_layer)
    dense_layer = Dropout(dense_dropout)(dense_layer)

    dense_layer = Dense(dense_size, activation=dense_activation)(dense_layer)
    dense_layer = Dropout(dense_dropout)(dense_layer)

    # Output layer.
    output_layer = Dense(num_classes, activation='softmax')(dense_layer)

    # Build model.
    model = models.Model(input, output_layer)

    if verbose >= 1:
        print('Done.')
    return model
