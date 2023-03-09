from src.Evolutionary_Algorithm import start_evolution

import tensorflow as tf
import tensorflow_datasets as tfds

from datetime import datetime
tf.random.set_seed(123)

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

    now = datetime.now()
    formatted_date = now.strftime("%d/%m/%Y %H:%M:%S")

    train_dataset, val_dataset, test_dataset = tfds.load("tf_flowers", shuffle_files=True,
                                                         split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
                                                         download=True, as_supervised=True)

    num_train = train_dataset.cardinality()
    num_val = val_dataset.cardinality()
    num_test = test_dataset.cardinality()
    train_dataset = prepare_dataset(train_dataset, is_training=True)
    val_dataset = prepare_dataset(val_dataset, is_training=False)
    test_dataset = prepare_dataset(test_dataset, is_training=False)

    population_array, max_fitness_history, average_fitness_history, best_models_arrays = start_evolution(
        train_ds=train_dataset,
        val_ds=val_dataset,
        test_ds=test_dataset,
        generations=2,
        population=5,
        num_classes=5,
        epochs=1,
        time=formatted_date)
