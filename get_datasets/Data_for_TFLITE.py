import tensorflow as tf  # Import TensorFlow library
import tensorflow_datasets as tfds  # Import TensorFlow Datasets library
import numpy as np  # Import NumPy library for array operations
import random  # Import random module for random number generation

# Set the image size, batch size, and other constants.
image_size = 256
batch_size = 1
auto = tf.data.AUTOTUNE
resize_bigger = 280
num_classes = 5

# Set the seed value for random number generation to ensure reproducibility.
seed_value = 666  # You can choose any number as your fixed seed value

# Set seed for Python's built-in random module.
random.seed(seed_value)

# Set seed for NumPy.
np.random.seed(seed_value)

# Set seed for TensorFlow.
tf.random.set_seed(seed_value)

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

def prepare_dataset(dataset, is_training=True):
    if is_training:
        dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.map(preprocess_dataset(is_training), num_parallel_calls=auto)
    return dataset.cache().batch(batch_size).prefetch(auto)

# Load the train, validation, and test datasets using TensorFlow Datasets.
train_dataset, val_dataset, test_dataset = tfds.load(
    "tf_flowers", shuffle_files=False,
    split=["train[:85%]", "train[85%:95%]", "train[95%:]"],
    download=True, as_supervised=True
)

# Prepare the test dataset.
test_dataset = prepare_dataset(test_dataset, is_training=False)

# Separate the test images and labels.
test_images = test_dataset.map(lambda x, y: x)
test_labels = test_dataset.map(lambda x, y: y)

# Convert the test images and labels into NumPy arrays.
x_test = np.array(list(test_images.as_numpy_iterator())).squeeze()
y_test = np.array(list(test_labels.as_numpy_iterator())).squeeze()




