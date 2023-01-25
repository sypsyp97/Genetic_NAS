from os import listdir
from os.path import join

from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import cv2
import tensorflow as tf

'''This function take in an image file path (img_path), and then uses the OpenCV library to read in the image and 
resize it to a specified size (img_size), returning the resized image as a numpy array and the number of classes 
(class_num). If there is an exception when reading or resizing the image, it will be caught and printed.'''


def process_image(img_path, img_size=128, class_num=2):
    try:
        img_arr = cv2.imread(img_path)
        resized_arr = cv2.resize(img_arr, (img_size, img_size), interpolation=cv2.INTER_AREA)
        return resized_arr, class_num
    except Exception as e:
        print(e)


'''This function take in a directory of images (data_dir) and an optional list of labels (labels). The function sets 
the default labels to 'PNEUMONIA' and 'NORMAL'. It creates an empty list called data which will be used to store the 
processed image data. Then it uses the ProcessPoolExecutor from the concurrent.futures module to concurrently process 
the images. It iterates through each label and finds the path for the label's images, it then assigns the class 
number as the index of the label in the label list. It creates a list of image paths for each label by joining the 
label path with each image file in the label's directory. It uses the executor.map method to apply the process_image 
function to each image path with the specified image size and class number, and then adds the returned data to the 
data list. Finally, it returns the data list.'''


def get_data(data_dir, img_size=128, labels=None):
    if labels is None:
        labels = ['PNEUMONIA', 'NORMAL']
    data = []
    with ProcessPoolExecutor() as executor:
        for label in labels:
            path = join(data_dir, label)
            class_num = labels.index(label)
            img_paths = [join(path, img) for img in listdir(path)]
            data += executor.map(partial(process_image, img_size=img_size, class_num=class_num), img_paths)
    return data


'''This function take in the directories for the training and testing images (train_dir and test_dir), 
the desired image size (img_size), and the number of classes (num_classes). It calls the get_data function with the 
training and testing directories to get the image data and labels.

The function then uses the zip function to extract the images and labels from the data returned by the get_data 
function, and assigns them to x_train and y_train for the training data, and x_test and y_test for the test data.

It then converts the lists of images to numpy arrays and reshapes them to have the correct dimensions for the image 
size and number of channels.

It uses the train_test_split function from the sklearn.model_selection module to split the training data into 
training and validation sets, and assigns them to x_train, y_train, x_val, and y_val.

It then uses the tf.one_hot function to convert the labels to one-hot encodings and returns the train, validation, 
and test sets with their corresponding labels.'''


def get_data_array(train_dir, test_dir, img_size=128, num_classes=2):
    x_train, y_train = zip(*get_data(train_dir, img_size))
    x_test, y_test = zip(*get_data(test_dir, img_size))

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    x_train = x_train.reshape(-1, img_size, img_size, 3)
    x_test = x_test.reshape(-1, img_size, img_size, 3)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

    y_train = tf.one_hot(y_train, num_classes)
    y_val = tf.one_hot(y_val, num_classes)
    y_test = tf.one_hot(y_test, num_classes)

    return x_train, y_train, x_val, y_val, x_test, y_test


# def get_data(data_dir, img_size=128, labels=None):
#     if labels is None:
#         labels = ['PNEUMONIA', 'NORMAL']
#     data = []
#     for label in labels:
#         path = join(data_dir, label)
#         class_num = labels.index(label)
#         for img in tqdm(listdir(path)):
#             try:
#                 img_arr = cv2.imread(join(path, img))
#                 resized_arr = tf.image.resize(img_arr, (img_size, img_size))  # Reshaping images to preferred size
#                 data.append((resized_arr, class_num))
#             except Exception as e:
#                 print(e)
#     return data
#
#
# def get_data_array(train_dir, test_dir, img_size=128, num_classes=2):
#     train_data = get_data(train_dir)
#     test_data = get_data(test_dir)
#     x_train, y_train = zip(*train_data)
#     x_test, y_test = zip(*test_data)
#
#     x_train = np.array(x_train)
#     x_test = np.array(x_test)
#
#     x_train = x_train.reshape(-1, img_size, img_size, 3)
#     x_test = x_test.reshape(-1, img_size, img_size, 3)
#
#     x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
#
#     y_train = tf.one_hot(y_train, num_classes)
#     y_val = tf.one_hot(y_val, num_classes)
#     y_test = tf.one_hot(y_test, num_classes)
#
#     return x_train, y_train, x_val, y_val, x_test, y_test

'''This function take in a list of images (images) and an optional seed value (seed) for reproducibility. 
It uses the RandAugment function from the imgaug library (imported as iaa) to apply random data augmentation to the 
images. The n parameter is the number of operations to apply and the m parameter is the magnitude of the operation. 
It then casts the images as tf.uint8 tensors to make sure that the data type of the images is correct for the 
operation. It applies the rand_aug operation on the images.numpy() and returns the augmented images.'''


def augment(images, seed=0):
    rand_aug = iaa.RandAugment(n=2, m=(0, 7), seed=seed)
    images = tf.cast(images, tf.uint8)
    return rand_aug(images=images.numpy())


'''It creates three different datasets: train_ds_rand, test_ds, val_ds.

First, it creates the train_ds_rand dataset by using the tf.data.Dataset.from_tensor_slices method to convert the 
training data and labels to a dataset, then shuffles it using a buffer size of batch_size * 100, splits it into 
batches of the specified size and applies the augment function using tf.py_function and map method, this applies 
random data augmentation to the images in the training set, it then caches and prefetches the data for performance.

The test_ds and val_ds datasets are created in a similar way, but without applying data augmentation.

Finally, it returns the three datasets: train_ds_rand, val_ds, test_ds'''


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

    return train_ds_rand, val_ds, test_ds