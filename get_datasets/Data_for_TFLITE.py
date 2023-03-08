import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

image_size = 256
batch_size = 1
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


train_dataset, val_dataset, test_dataset = tfds.load("tf_flowers", shuffle_files=True,
                                                     split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
                                                     download=True, as_supervised=True)

test_dataset = prepare_dataset(test_dataset, is_training=False)

test_images = test_dataset.map(lambda x, y: x)
test_labels = test_dataset.map(lambda x, y: y)

x_test = np.array(list(test_images.as_numpy_iterator())).squeeze()
y_test = np.array(list(test_labels.as_numpy_iterator())).squeeze()



