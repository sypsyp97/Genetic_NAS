from keras import layers
from tensorflow import keras

from src.Decode_Block import decoded_block
from src.Gene_Pool import conv_block

'''This function takes in 3 inputs, model_array, num_classes and input_shape. The function creates a keras model by defining the layers in it.

It starts by creating an input layer with the shape specified by the input_shape variable. Then it applies a 
rescaling layer with a scale factor of 1/255 to the input. It then applies a convolutional block with a kernel size 
of 2, 16 filters and a stride of 2 to the input.

It then enters a for loop that iterates 9 times. On each iteration, it applies a decoded_block function to the 
current output, passing in the current element of the model_array.

After the for loop, it applies another convolutional block with 320 filters, a kernel size of 1 and a stride of 1 to 
the output. It then applies a global average pooling layer and a dropout layer with a rate of 0.5. Finally, 
it adds a dense layer with num_classes number of units and returns the model.'''


def create_model(model_array, num_classes=2, input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = conv_block(x, kernel_size=2, filters=16, strides=2)

    for i in range(9):
        x = decoded_block(x, model_array[i])

    x = conv_block(x, filters=320, kernel_size=1, strides=1)
    x = layers.GlobalAvgPool2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)


def model_summary(model):
    model.summary()
    print('Number of trainable weights = {}'.format(len(model.trainable_weights)))
