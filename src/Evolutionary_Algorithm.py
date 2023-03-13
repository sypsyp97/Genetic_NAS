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
import os

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


def create_first_population(population=10, num_classes=5):
    first_population_array = np.random.randint(0, 2, (population, 9, 18))

    for i in range(population):
        model = create_model(first_population_array[i], num_classes=num_classes)
        while check_model(model):
            del model
            first_population_array[i] = np.random.randint(0, 2, (9, 18))
            model = create_model(first_population_array[i], num_classes=num_classes)

        del model

    return first_population_array


"""Function Signature:
def select_models(train_ds: tf.data.Dataset,
val_ds: tf.data.Dataset,
test_ds: tf.data.Dataset,
time: str,
population_array: np.ndarray,
generation: int,
epochs: int = 30,
num_classes: int = 5) -> Tuple[List[np.ndarray], float, float]

Parameters:
train_ds: A TensorFlow dataset representing the training data.
val_ds: A TensorFlow dataset representing the validation data.
test_ds: A TensorFlow dataset representing the test data.
time: A string representing the current time.
population_array: A NumPy array of shape (population, 9, 18) representing the initial population of models.
generation: An integer representing the current generation.
epochs: An integer representing the number of epochs to train each model. Defaults to 30.
num_classes: An integer representing the number of classes in the classification task. Defaults to 5.

Returns:
A tuple containing: A list of NumPy arrays representing the binary arrays of the top 5 models with the highest 
fitness values in the current generation. A float representing the highest fitness value in the current generation. A 
float representing the average fitness value in the current generation. 

Description: The "select_models" function trains the models in the current population, evaluates their fitness 
values, and selects the top 5 models with the highest fitness values to pass on to the next generation. For each 
model in the population, the function creates a Keras model from the binary array representation, trains the model on 
the training and validation datasets, converts the model to a TensorFlow Lite model, evaluates the accuracy of the 
model on the test dataset using the "evaluate_tflite_model" function, and measures the inference time of the model on 
the Edge TPU using the "inference_time_tpu" function. The fitness value of the model is then calculated using the 
"calculate_fitness" function.

The function returns a tuple containing:

A list of the top 5 models with the highest fitness values in the current generation. Each model is represented as a 
binary array of shape (9, 18). The highest fitness value in the current generation. The average fitness value in the 
current generation."""


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
    result_dir = f'results_{time}'
    generation_dir = result_dir + f'/generation_{generation}'
    best_models_arrays_dir = generation_dir + '/best_model_arrays.pkl'
    fitness_list_dir = generation_dir + '/fitness_list.pkl'
    tflite_accuracy_list_dir = generation_dir + '/tflite_accuracy_list.pkl'
    tpu_time_list_dir = generation_dir + '/tpu_time_list.pkl'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(generation_dir):
        os.makedirs(generation_dir)

    with open(best_models_arrays_dir, 'wb') as f:
        pickle.dump(best_models_arrays, f)
    with open(fitness_list_dir, 'wb') as f:
        pickle.dump(fitness_list, f)
    with open(tflite_accuracy_list_dir, 'wb') as f:
        pickle.dump(tflite_accuracy_list, f)
    with open(tpu_time_list_dir, 'wb') as f:
        pickle.dump(tpu_time_list, f)

    return best_models_arrays, max_fitness, average_fitness


"""Function Signature:
def crossover(parent_arrays: List[np.ndarray]) -> np.ndarray

Parameters:
parent_arrays: A list of NumPy arrays representing the binary arrays of the parents.

Returns:
A NumPy array representing the binary array of the child resulting from the crossover operation. 

Description: The "crossover" function takes a list of binary arrays representing the parents and performs a crossover 
operation to create a new binary array representing the child. The crossover operation is performed by randomly 
selecting one of the binary values from each parent for each layer in the child's binary array.

The function returns a NumPy array representing the binary array of the child resulting from the crossover operation."""


def crossover(parent_arrays):
    parent_indices = np.random.randint(0, 5, size=parent_arrays[0].shape)
    child_array = np.choose(parent_indices, parent_arrays)
    return child_array


"""Function Signature:
def mutate(model_array: np.ndarray, mutate_prob: float = 0.05) -> np.ndarray

Parameters:
model_array: A NumPy array representing the binary array of the model to mutate.
mutate_prob: A float value representing the probability of a gene being mutated. Default is 0.05.

Returns:
A NumPy array representing the binary array of the mutated model.

Description: The "mutate" function takes a NumPy array representing the binary array of the model and performs a 
mutation operation with a given probability. The mutation operation randomly selects genes and inverts them (from 0 
to 1 or from 1 to 0) with the given probability.

The function returns a NumPy array representing the binary array of the mutated model."""


def mutate(model_array, mutate_prob=0.05):
    prob = np.random.uniform(size=(9, 18))
    mutated_array = np.where(prob < mutate_prob, np.logical_not(model_array), model_array)

    return mutated_array


"""Function Signature:
def create_next_population(parent_arrays: List[np.ndarray], population: int = 10, num_classes: int = 5) -> np.ndarray

Parameters:
parent_arrays: A list of NumPy arrays representing the binary arrays of the parents.
population: An integer value representing the size of the next population to create. Default is 10.
num_classes: An integer value representing the number of classes in the classification task. Default is 5.

Returns: A NumPy array representing the binary arrays of the individuals in the next population. 

Description: The "create_next_population" function takes a list of binary arrays representing the parents and 
performs crossover and mutation operations to create a new population of binary arrays representing individuals. The 
function first creates a new empty population array with the given size. Then, for each individual in the population, 
the function performs the crossover and mutation operations on the parent arrays to create a new binary array for the 
individual. The function also checks the validity of the new binary array by calling the "check_model" function and 
generates a new binary array if it is invalid.

The function returns a NumPy array representing the binary arrays of the individuals in the next population."""


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

    result_dir = f'results_{time}'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

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



        next_population_array_dir = result_dir + '/next_population_array.pkl'
        max_fitness_history_dir = result_dir + '/max_fitness_history.pkl'
        average_fitness_history_dir = result_dir + '/average_fitness_history.pkl'
        best_model_arrays_dir = result_dir + '/best_model_arrays.pkl'

        with open(next_population_array_dir, 'wb') as f:
            pickle.dump(population_array, f)
        with open(max_fitness_history_dir, 'wb') as f:
            pickle.dump(max_fitness_history, f)
        with open(average_fitness_history_dir, 'wb') as f:
            pickle.dump(average_fitness_history, f)
        with open(best_model_arrays_dir, 'wb') as f:
            pickle.dump(best_models_arrays, f)

    return population_array, max_fitness_history, average_fitness_history, best_models_arrays
