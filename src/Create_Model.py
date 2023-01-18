import numpy as np
from tensorflow import keras
from keras import layers

from src.Decode_Layer import decoded_layer
from src.Gene_Pool import conv_block


def create_model(model_array=np.random.randint(0, 2, (9, 18)),
                 num_classes=2, input_shape=(128, 128, 3)):

    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = conv_block(x, kernel_size=2, filters=16, strides=2)

    for i in range(9):
        x = decoded_layer(x, model_array[i])

    x = conv_block(x, filters=320, kernel_size=1, strides=1)
    x = layers.GlobalAvgPool2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes)(x)

    return keras.Model(inputs, outputs)


def model_summary(model):
    model.summary()
    print('Number of trainable weights = {}'.format(len(model.trainable_weights)))
