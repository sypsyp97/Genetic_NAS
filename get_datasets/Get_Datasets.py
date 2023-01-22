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
                resized_arr = tf.image.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append((resized_arr, class_num))
            except Exception as e:
                print(e)
    return data


def get_data_array(train_dir='../content/chest_xray_new/train',
                   test_dir='../content/chest_xray_new/test',
                   img_size=128, num_classes=2):

    train = get_data(train_dir)
    test = get_data(test_dir)

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    for feature, label in train:
        x_train.append(feature)
        y_train.append(label)

    for feature, label in test:
        x_test.append(feature)
        y_test.append(label)

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    x_train = x_train.reshape(-1, img_size, img_size, 3)
    y_train = np.array(y_train)

    x_test = x_test.reshape(-1, img_size, img_size, 3)
    y_test = np.array(y_test)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=33)

    y_train = tf.one_hot(y_train, num_classes)
    y_val = tf.one_hot(y_val, num_classes)
    y_test = tf.one_hot(y_test, num_classes)

    return x_train, y_train, x_val, y_val, x_test, y_test

# TODO: Fix the RangAugment

# def augment(images):
#     rand_aug = iaa.RandAugment(n=2, m=7)
#     images = tf.cast(images, tf.uint8)
#     return rand_aug(images=images.numpy())


def get_datasets(x_train, y_train, x_test, y_test, x_val, y_val,
                 auto=tf.data.AUTOTUNE, batch_size=2):
    # train_ds_rand = (
    #     tf.data.Dataset.from_tensor_slices((x_train, y_train))
    #     .shuffle(batch_size * 100)
    #     .batch(batch_size)
    #     .map(lambda x, y: (tf.py_function(augment, [x], [tf.float32])[0], y), num_parallel_calls=auto)
    #     .prefetch(auto)
    # )

    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .shuffle(batch_size * 100)
        .batch(batch_size)
        .prefetch(auto)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .shuffle(batch_size * 100)
        .batch(batch_size)
        .prefetch(auto)
    )

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(batch_size * 100)
        .batch(batch_size)
        .prefetch(auto)
    )

    return train_ds, val_ds, test_ds

    # return train_ds_rand, val_ds, test_ds, train_ds
