import tensorflow_addons as tfa
from tensorflow import keras

'''This function is used for training a model using the TensorFlow Keras API. It takes in a training dataset (
train_ds), a validation dataset (val_ds), the model to be trained, the number of training epochs, and a filepath to 
save the model checkpoint after each epoch. The function also takes in an optimizer which is used to update the 
model's parameters during training. By default, it uses the AdaBelief optimizer. The function also specifies the loss 
function as BinaryFocalCrossentropy and the metrics to be monitored during training is accuracy.

The function also includes a try-except block to handle the case when the model's memory usage exceeds the available 
memory. If this happens, the function will print a message and return None for the history. The function then returns 
the trained model and the history of the training process.'''


def train_model(train_ds, val_ds,
                model, epochs=20,
                checkpoint_filepath="C:/Users/yd57yjac/PycharmProjects/Genetic_NAS/checkpoints/checkpoint",
                optimizer=tfa.optimizers.AdaBelief(learning_rate=1e-3,
                                                   total_steps=10000,
                                                   warmup_proportion=0.1,
                                                   min_lr=2e-6,
                                                   rectify=True)):
    checkpoint_callback = keras.callbacks.ModelCheckpoint(checkpoint_filepath,
                                                          monitor="val_accuracy",
                                                          save_best_only=True,
                                                          save_weights_only=True)

    optimizer = tfa.optimizers.LazyAdam(learning_rate=0.002)
    optimizer = tfa.optimizers.MovingAverage(optimizer)
    optimizer = tfa.optimizers.Lookahead(optimizer)

    model.compile(optimizer=optimizer,
                  loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # TODO: Find a solution to the problem of large transformer sequence.
    #       Which can lead to insufficient Graph Memory.

    try:
        # Fit the model on the batches generated by datagen.flow().
        history = model.fit(train_ds,
                            epochs=epochs,
                            validation_data=val_ds,
                            callbacks=[checkpoint_callback])

        model.load_weights(checkpoint_filepath)

    except:
        history = None
        print("Do not have enough memory.")

    return model, history
