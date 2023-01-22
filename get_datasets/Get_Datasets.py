from os import listdir
from os.path import join

import cv2
import numpy as np
import tensorflow as tf
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm


def get_data(data_dir, img_size=128, labels=None):
    if labels is None:
        labels = ['PNEUMONIA', 'NORMAL']
    data = []
    for label in labels:
        path = join(data_dir, label)
        class_num = labels.index(label)
        for img in tqdm(listdir(path)):
            try:
                img_arr = cv2.imread(join(path, img))
                resized_arr = tf.image.resize(img_arr, (img_size, img_size))  # Reshaping images to preferred size
                data.append((resized_arr, class_num))
            except Exception as e:
                print(e)
    return data


def get_data_array(train_dir, test_dir, img_size=128, num_classes=2):
    train_data = get_data(train_dir)
    test_data = get_data(test_dir)
    x_train, y_train = zip(*train_data)
    x_test, y_test = zip(*test_data)

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    x_train = x_train.reshape(-1, img_size, img_size, 3)
    x_test = x_test.reshape(-1, img_size, img_size, 3)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

    y_train = tf.one_hot(y_train, num_classes)
    y_val = tf.one_hot(y_val, num_classes)
    y_test = tf.one_hot(y_test, num_classes)

    return x_train, y_train, x_val, y_val, x_test, y_test


def augment(images):
    rand_aug = iaa.RandAugment(n=2, m=7)
    images = tf.cast(images, tf.uint8)
    return rand_aug(images=images.numpy())


def get_datasets(x_train, y_train, x_val, y_val, x_test, y_test,
                 auto=tf.data.AUTOTUNE, batch_size=16):
    train_ds_rand = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(batch_size * 100)
        .batch(batch_size)
        .map(lambda x, y: (tf.py_function(augment, [x], [tf.float32])[0], y), num_parallel_calls=auto)
        .cache()
        .prefetch(auto)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .shuffle(batch_size * 100)
        .batch(batch_size)
        .cache()
        .prefetch(auto)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .shuffle(batch_size * 100)
        .batch(batch_size)
        .cache()
        .prefetch(auto)
    )

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(batch_size * 100)
        .batch(batch_size)
        .cache()
        .prefetch(auto)
    )
    #
    # return train_ds, val_ds, test_ds

    return train_ds_rand, val_ds, test_ds, train_ds
