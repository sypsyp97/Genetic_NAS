# calculate the fitness
from math import pi

import numpy as np


def calculate_fitness(accuracy, inference_time=1, memory_footprint_tflite=1):
    """Calculate the fitness of a model based on its accuracy and inference time.

    The fitness is defined as a weighted combination of the accuracy and the normalized inference time,
    with a higher accuracy and a lower inference time leading to a higher fitness. The inference time
    is normalized using an arctangent function, which helps to limit its impact on the overall fitness.

    Parameters
    ----------
    accuracy : float
        The accuracy of the model on a given dataset. This should be a value between 0 (no correct predictions)
        and 1 (all predictions correct).

    inference_time : float, optional (default=1)
        The time taken by the model to make a prediction. This is usually measured in milliseconds.
        A lower inference time indicates a faster model.

    memory_footprint_tflite : float, optional (default=1)
        The memory footprint of the TensorFlow Lite model. This parameter is defined but not used in the function.

    Returns
    -------
    fitness : float
        The fitness of the model, calculated as a weighted combination of the accuracy and the normalized
        inference time. A higher fitness indicates a better model.
    """
    # Normalize the inference time using an arctangent function
    normalized_inference_time = np.arctan(inference_time / 500) / (pi / 2)

    # Calculate the fitness as the accuracy weighted by the normalized inference time
    fitness = (1 - normalized_inference_time) * accuracy

    return fitness
