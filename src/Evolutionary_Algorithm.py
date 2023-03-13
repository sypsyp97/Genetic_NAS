import numpy as np

from tools.Create_Model import create_model
from tools.Evaluate_Model import evaluate_tflite_model
from tools.Create_Model import train_model
from tools.Model_Checker import check_model
from tools.TFLITE_Converter import convert_to_tflite
from tools.Compile_Edge_TPU import compile_edgetpu
from tools.Inference_Speed_TPU import inference_time_tpu
from src.Fitness_Function import calculate_fitness
import pickle

"""Function Signature:
def create_first_population(population: int = 100, num_classes: int = 5) -> np.ndarray

Parameters:
population: An integer representing the number of models in the initial population. Defaults to 100.
num_classes: An integer representing the number of classes in the classification task. Defaults to 5.

Returns:
A NumPy array of shape (population, 9, 18) representing the initial population of models.


Description: The "create_first_population" function creates an initial population of "population" models, where each 
model is represented as a binary array of shape (9, 18). Each element of the binary array represents a parameter for 
a specific layer operation.
The function first generates a random binary array of shape (9, 18) for each model in the initial population. It then 
creates a model from each binary array and checks if it satisfies certain constraints using the "check_model" 
function. If the model does not satisfy the constraints, the function generates a new random binary array for the 
model and tries again until a valid model is found.

The function returns a NumPy array representing the initial population of models.
"""


def create_first_population(population=100, num_classes=5):
    first_population_array = np.random.randint(0, 2, (population, 9, 18))

    for i in range(population):
        model = create_model(first_population_array[i], num_classes=num_classes)
        while check_model(model):
            del model
            first_population_array[i] = np.random.randint(0, 2, (9, 18))
            model = create_model(first_population_array[i], num_classes=num_classes)

        del model

    return first_population_array


def select_models(train_ds,
                  val_ds,
                  test_ds,
                  time,
                  population_array,
                  generation,
                  epochs=30,
                  num_classes=5):
    fitness_list = []
    tflite_accuracy_list = []
    tpu_time_list = []

    for i in range(population_array.shape[0]):
        model = create_model(population_array[i], num_classes=num_classes)
        model, _ = train_model(train_ds, val_ds, model=model, epochs=epochs)

        try:
            tflite_model, tflite_name = convert_to_tflite(keras_model=model, generation=generation, i=i, time=time)
            tflite_accuracy = evaluate_tflite_model(tflite_model=tflite_model, tfl_int8=True)
            edgetpu_name = compile_edgetpu(tflite_name)
            tpu_time = inference_time_tpu(edgetpu_model_name=edgetpu_name)
        except:
            tflite_accuracy = 0
            tpu_time = 9999

        fitness = calculate_fitness(tflite_accuracy, tpu_time)

        tflite_accuracy_list.append(tflite_accuracy)
        fitness_list.append(fitness)
        tpu_time_list.append(tpu_time)

    max_fitness = np.max(fitness_list)
    average_fitness = np.average(fitness_list)

    best_models_indices = sorted(range(len(fitness_list)), key=lambda j: fitness_list[j], reverse=True)[:5]
    best_models_arrays = [population_array[k] for k in best_models_indices]

    with open(f'results_{time}/generation_{generation}/best_model_arrays.pkl', 'wb') as f:
        pickle.dump(best_models_arrays, f)
    with open(f'results_{time}/generation_{generation}/fitness_list.pkl', 'wb') as f:
        pickle.dump(fitness_list, f)
    with open(f'results_{time}/generation_{generation}/tflite_accuracy_list.pkl', 'wb') as f:
        pickle.dump(tflite_accuracy_list, f)
    with open(f'results_{time}/generation_{generation}/tpu_time_list.pkl', 'wb') as f:
        pickle.dump(tpu_time_list, f)

    return best_models_arrays, max_fitness, average_fitness


def crossover(parent_arrays):
    parent_indices = np.random.randint(0, 5, size=parent_arrays[0].shape)
    child_array = np.choose(parent_indices, parent_arrays)
    return child_array


def mutate(model_array, mutate_prob=0.05):
    prob = np.random.uniform(size=(9, 18))
    mutated_array = np.where(prob < mutate_prob, np.logical_not(model_array), model_array)

    return mutated_array


def create_next_population(parent_arrays, population=10, num_classes=5):
    next_population_array = np.random.randint(0, 2, (population, 9, 18))

    for individual in range(population):
        next_population_array[individual] = crossover(parent_arrays)
        next_population_array[individual] = mutate(next_population_array[individual], mutate_prob=0.03)

    for individual in range(population):
        model = create_model(next_population_array[individual], num_classes=num_classes)
        while check_model(model):
            del model
            next_population_array[individual] = crossover(parent_arrays)
            next_population_array[individual] = mutate(next_population_array[individual], mutate_prob=0.03)
            model = create_model(next_population_array[individual], num_classes=num_classes)
        del model

    return next_population_array


def start_evolution(train_ds, val_ds, test_ds, generations, population, num_classes, epochs, population_array=None,
                    time=None):
    max_fitness_history = []
    average_fitness_history = []
    if population_array is None:
        population_array = create_first_population(population=population, num_classes=num_classes)

    for generation in range(generations):
        best_models_arrays, max_fitness, average_fitness = select_models(train_ds=train_ds, val_ds=val_ds,
                                                                         test_ds=test_ds, time=time,
                                                                         population_array=population_array,
                                                                         generation=generation, epochs=epochs,
                                                                         num_classes=num_classes)
        population_array = create_next_population(parent_arrays=best_models_arrays, population=population,
                                                  num_classes=num_classes)
        max_fitness_history.append(max_fitness)
        average_fitness_history.append(average_fitness)

    with open(f'results_{time}/next_population_array.pkl', 'wb') as f:
        pickle.dump(population_array, f)
    with open(f'results_{time}/max_fitness_history.pkl', 'wb') as f:
        pickle.dump(max_fitness_history, f)
    with open(f'results_{time}/average_fitness_history.pkl', 'wb') as f:
        pickle.dump(average_fitness_history, f)
    with open(f'results_{time}//best_model_arrays.pkl', 'wb') as f:
        pickle.dump(best_models_arrays, f)

    return population_array, max_fitness_history, average_fitness_history, best_models_arrays
