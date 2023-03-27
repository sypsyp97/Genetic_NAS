from src.Evolutionary_Algorithm import start_evolution, create_next_population
import tensorflow_datasets as tfds
import gc

from datetime import datetime
from get_datasets.Data_for_TFLITE import prepare_dataset
import random
import numpy as np
import tensorflow as tf

seed_value = 666  # You can choose any number as your fixed seed value

random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

image_size = 256
batch_size = 64
auto = tf.data.AUTOTUNE
resize_bigger = 280
num_classes = 5

# with open('results_14032023164450/generation_11/best_model_arrays.pkl', 'rb') as f:
#     data = pickle.load(f)
#     f.close()
#
# next = create_next_population(parent_arrays=data, population=20, num_classes=5)


if __name__ == '__main__':
    gc.enable()

    now = datetime.now()
    formatted_date = now.strftime("%d%m%Y%H%M%S")

    train_dataset, val_dataset, test_dataset = tfds.load("tf_flowers", shuffle_files=False,
                                                         split=["train[:85%]", "train[85%:95%]", "train[95%:]"],
                                                         download=True, as_supervised=True)

    train_dataset = prepare_dataset(train_dataset, is_training=True)
    val_dataset = prepare_dataset(val_dataset, is_training=False)
    test_dataset = prepare_dataset(test_dataset, is_training=False)

    population_array, max_fitness_history, average_fitness_history, best_models_arrays = start_evolution(
        train_ds=train_dataset,
        val_ds=val_dataset,
        test_ds=test_dataset,
        generations=16,
        population=24,
        num_classes=5,
        epochs=30,
        # population_array=next,
        time=formatted_date)
