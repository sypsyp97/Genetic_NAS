from keras import layers
import tensorflow_addons as tfa
from tensorflow import keras

from src.Decode_Block import decoded_block
from src.Gene_Pool import conv_block


# This function constructs the model architecture.
def create_model(model_array, num_classes=5, input_shape=(256, 256, 3)):
    """
    Parameters
    ----------
    model_array : list
        List containing the parameters for each layer in the network.
    num_classes : int, default=5
        The number of output classes in the classification problem.
    input_shape : tuple, default=(256, 256, 3)
        The shape of the input data (image size and channels).

    Returns
    -------
    model : keras.Model
        The constructed Keras model.
    """
    # Define input layer with the specified input shape
    inputs = layers.Input(shape=input_shape)
    # Normalize the pixel values to [0, 1] range
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    # Add a convolution block
    x = conv_block(x, kernel_size=2, filters=16, strides=2)

    # Create the specified number of MobileViT blocks
    for i in range(9):
        x = decoded_block(x, model_array[i])

    # Add a final convolution block
    x = conv_block(x, filters=320, kernel_size=1, strides=1)
    # Apply global average pooling to the feature map
    x = layers.GlobalAvgPool2D()(x)

    # Add the output layer with softmax activation for multi-class classification
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    # Construct the model
    model = keras.Model(inputs, outputs)

    return model


def model_summary(model):
    """
    Prints the model summary and the number of trainable weights.
    """
    model.summary()
    print('Number of trainable weights = {}'.format(len(model.trainable_weights)))


def train_model(train_ds, val_ds, model, epochs=30,
                checkpoint_filepath="checkpoints/checkpoint",
                early_stopping_patience=10):
    """
    Trains the given model with the specified training and validation datasets.

    Parameters
    ----------
    train_ds : tf.data.Dataset
        The training dataset.
    val_ds : tf.data.Dataset
        The validation dataset.
    model : keras.Model
        The model to train.
    epochs : int, default=30
        The number of training epochs.
    checkpoint_filepath : str, default="checkpoints/checkpoint"
        The file path to save the model weights with the best validation accuracy.
    early_stopping_patience : int, default=10
        The number of epochs with no improvement after which training will be stopped.

    Returns
    -------
    model : keras.Model
        The trained model.
    history : History
        A record of training loss values and metrics values at successive epochs.
    """
    # Define callback to save the model weights with the best validation accuracy
    checkpoint_callback = keras.callbacks.ModelCheckpoint(checkpoint_filepath,
                                                          monitor="val_accuracy",
                                                          save_best_only=True,
                                                          save_weights_only=True)

    # Define the loss function
    loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    # Define the optimizer
    opt = tfa.optimizers.LazyAdam(learning_rate=0.002)
    opt = tfa.optimizers.MovingAverage(opt)
    opt = tfa.optimizers.Lookahead(opt)

    # Compile the model with the specified loss function and optimizer
    model.compile(optimizer=opt,
                  loss=loss_fn,
                  metrics=['accuracy'])

    # Define early stopping callback to stop training when validation loss stops improving
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                            patience=early_stopping_patience,
                                                            restore_best_weights=True)

    # Begin training the model
    try:
        history = model.fit(train_ds,
                            epochs=epochs,
                            validation_data=val_ds,
                            callbacks=[checkpoint_callback, early_stopping_callback])
        # Load the best weights from the model checkpoint
        model.load_weights(checkpoint_filepath)
    except Exception as e:
        # In case of any exceptions during training, print the exception and return None for history
        history = None
        print(e)

    # Return the trained model and its history
    return model, history


