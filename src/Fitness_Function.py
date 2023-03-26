# calculate the fitness
import numpy as np
from math import pi


def calculate_fitness(accuracy, inference_time=1, memory_footprint_tflite=1):
    fitness = (1 - np.arctan(inference_time/500) / (pi / 2)) * accuracy

    return fitness
