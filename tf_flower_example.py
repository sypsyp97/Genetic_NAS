import gc
import pickle
from datetime import datetime

import tensorflow as tf
import tensorflow_datasets as tfds

from src.Evolutionary_Algorithm import create_next_population, start_evolution

# Set the random seed for reproducibility.
tf.random.set_seed(123)

# Set the image size, batch size, and other constants.
image_size = 256
batch_size = 64
auto = tf.data.AUTOTUNE
resize_bigger = 280
num_classes = 5

# Load the data from the pickle file.
with open("results_28032023195907/generation_2/best_model_arrays.pkl", "rb") as f:
    data = pickle.load(f)
    f.close()

# Create the next population using the parent arrays.
next = create_next_population(parent_arrays=data, population=20, num_classes=5)


# Define the preprocessing function for the dataset.
def preprocess_dataset(is_training=True):
    def _pp(image, label):
        if is_training:
            # Resize to a bigger spatial resolution and take random crops.
            image = tf.image.resize(image, (resize_bigger, resize_bigger))
            image = tf.image.random_crop(image, (image_size, image_size, 3))
            image = tf.image.random_flip_left_right(image)
        else:
            image = tf.image.resize(image, (image_size, image_size))
        label = tf.one_hot(label, depth=num_classes)
        return image, label

    return _pp


# Prepare the dataset for training, validation, and testing.
def prepare_dataset(dataset, is_training=True):
    if is_training:
        dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.map(preprocess_dataset(is_training), num_parallel_calls=auto)
    return dataset.cache().batch(batch_size).prefetch(auto)


if __name__ == "__main__":
    # Enable garbage collection.
    gc.enable()

    # Get the current date and time for formatting purposes.
    now = datetime.now()
    formatted_date = now.strftime("%d%m%Y%H%M%S")

    # Load the train, validation, and test datasets using TensorFlow Datasets.
    train_dataset, val_dataset, test_dataset = tfds.load(
        "tf_flowers",
        shuffle_files=False,
        split=["train[:85%]", "train[85%:95%]", "train[95%:]"],
        download=True,
        as_supervised=True,
    )

    # Prepare the datasets for training, validation, and testing.
    train_dataset = prepare_dataset(train_dataset, is_training=True)
    val_dataset = prepare_dataset(val_dataset, is_training=False)
    test_dataset = prepare_dataset(test_dataset, is_training=False)

    # Start the evolution process.
    (
        population_array,
        max_fitness_history,
        average_fitness_history,
        best_models_arrays,
    ) = start_evolution(
        train_ds=train_dataset,
        val_ds=val_dataset,
        test_ds=test_dataset,
        generations=4,
        population=20,
        num_classes=5,
        epochs=30,
        population_array=next,
        time=formatted_date,
    )
