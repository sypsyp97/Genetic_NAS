from get_datasets.Get_Datasets import get_data_array, get_datasets
from src.Create_Model import train_model
from src.Create_Model import create_model
from src.Evolutionary_Algorithm import create_next_population, create_first_population, select_best_2_model, \
    start_evolution

import os
import random
import tensorflow as tf
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import mixed_precision

# tf.config.optimizer.set_jit(True)


# if tf.config.list_physical_devices('GPU'):
#     strategy = tf.distribute.MirroredStrategy()
# else:  # Use the Default Strategy
#     strategy = tf.distribute.get_strategy()


policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


tfds.core.utils.gcs_utils._is_gcs_disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['NO_GCE_CHECK'] = 'true'

image_size = 256
batch_size = 64
auto = tf.data.AUTOTUNE
resize_bigger = 280
num_classes = 5


def preprocess_dataset(is_training=True):
    def _pp(image, label):
        if is_training:
            # Resize to a bigger spatial resolution and take the random
            # crops.
            image = tf.image.resize(image, (resize_bigger, resize_bigger))
            image = tf.image.random_crop(image, (image_size, image_size, 3))
            image = tf.image.random_flip_left_right(image)
        else:
            image = tf.image.resize(image, (image_size, image_size))
        label = tf.one_hot(label, depth=num_classes)
        return image, label

    return _pp


def prepare_dataset(dataset, is_training=True):
    if is_training:
        dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.map(preprocess_dataset(is_training), num_parallel_calls=auto)
    return dataset.cache().batch(batch_size).prefetch(auto)


if __name__ == '__main__':
    # physical_devices = tf.config.list_physical_devices('GPU')
    # try:
    #     tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # except:
    #     pass

    train_dataset, val_dataset = tfds.load("tf_flowers", split=["train[:90%]", "train[90%:]"],
                                           download=False, as_supervised=True)

    num_train = train_dataset.cardinality()
    num_val = val_dataset.cardinality()
    train_dataset = prepare_dataset(train_dataset, is_training=True)
    val_dataset = prepare_dataset(val_dataset, is_training=False)

    print(f"Number of training examples: {num_train}")
    print(f"Number of validation examples: {num_val}")

    population_array, max_fitness_history, average_fitness_history, a, b = start_evolution(train_ds=train_dataset,
                                                                                           val_ds=val_dataset,
                                                                                           test_ds=val_dataset,
                                                                                           generations=16,
                                                                                           population=14,
                                                                                           num_classes=5,
                                                                                           epochs=30)
