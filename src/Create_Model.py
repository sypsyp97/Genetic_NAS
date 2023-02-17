from keras import layers
import tensorflow_addons as tfa
from tensorflow import keras
import tensorflow as tf

from src.Decode_Block import decoded_block
from src.Gene_Pool import conv_block

# if tf.config.list_physical_devices('GPU'):
#     strategy = tf.distribute.MirroredStrategy()
# else:  # Use the Default Strategy
#     strategy = tf.distribute.get_strategy()

'''This function takes in 3 inputs, model_array, num_classes and input_shape. The function creates a keras model by defining the layers in it.

It starts by creating an input layer with the shape specified by the input_shape variable. Then it applies a 
rescaling layer with a scale factor of 1/255 to the input. It then applies a convolutional block with a kernel size 
of 2, 16 filters and a stride of 2 to the input.

It then enters a for loop that iterates 9 times. On each iteration, it applies a decoded_block function to the 
current output, passing in the current element of the model_array.

After the for loop, it applies another convolutional block with 320 filters, a kernel size of 1 and a stride of 1 to 
the output. It then applies a global average pooling layer and a dropout layer with a rate of 0.5. Finally, 
it adds a dense layer with num_classes number of units and returns the model.'''


def create_model(model_array, num_classes=5, input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = conv_block(x, kernel_size=2, filters=16, strides=2)

    for i in range(9):
        x = decoded_block(x, model_array[i])

    x = conv_block(x, filters=320, kernel_size=1, strides=1)
    x = layers.GlobalAvgPool2D()(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    return model


def model_summary(model):
    model.summary()
    print('Number of trainable weights = {}'.format(len(model.trainable_weights)))


# def train_model(train_ds, val_ds,
#                 model, epochs=20,
#                 checkpoint_filepath="checkpoints/checkpoint"):
#     checkpoint_callback = keras.callbacks.ModelCheckpoint(checkpoint_filepath,
#                                                           monitor="val_accuracy",
#                                                           save_best_only=True,
#                                                           save_weights_only=True)
#
#     loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
#
#     opt = tfa.optimizers.LazyAdam(learning_rate=0.002)
#     opt = tfa.optimizers.MovingAverage(opt)
#     opt = tfa.optimizers.Lookahead(opt)
#
#     model.compile(optimizer=opt,
#                   loss=loss_fn,
#                   metrics=['accuracy'])
#
#     try:
#         history = model.fit(train_ds,
#                             epochs=epochs,
#                             validation_data=val_ds,
#                             callbacks=[checkpoint_callback])
#
#         model.load_weights(checkpoint_filepath)
#     except Exception as e:
#         history = None
#         print(e)
#     return model, history

def train_model(train_ds, val_ds,
                model, epochs=20,
                checkpoint_filepath="checkpoints/checkpoint",
                early_stopping_patience=15):
    checkpoint_callback = keras.callbacks.ModelCheckpoint(checkpoint_filepath,
                                                          monitor="val_accuracy",
                                                          save_best_only=True,
                                                          save_weights_only=True)

    loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    opt = tfa.optimizers.LazyAdam(learning_rate=0.002)
    opt = tfa.optimizers.MovingAverage(opt)
    opt = tfa.optimizers.Lookahead(opt)

    model.compile(optimizer=opt,
                  loss=loss_fn,
                  metrics=['accuracy'])

    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                            patience=early_stopping_patience,
                                                            restore_best_weights=True)

    try:
        history = model.fit(train_ds,
                            epochs=epochs,
                            validation_data=val_ds,
                            callbacks=[checkpoint_callback, early_stopping_callback])

        model.load_weights(checkpoint_filepath)
    except Exception as e:
        history = None
        print(e)
    return model, history
