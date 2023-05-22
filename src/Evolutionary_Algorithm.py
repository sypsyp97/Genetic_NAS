import gc

import numpy as np

from tools.Create_Model import create_model
from tools.Evaluate_Model import evaluate_tflite_model
from tools.Create_Model import train_model
from tools.Model_Checker import model_has_problem
from tools.TFLITE_Converter import convert_to_tflite
from tools.Compile_Edge_TPU import compile_edgetpu

from src.Fitness_Function import calculate_fitness
import pickle
import os


def create_first_population(population, num_classes=5):
    """
    Generate the initial population of models for a genetic algorithm.

    Parameters:
    population : int
        The number of models to generate.
    num_classes : int, optional
        The number of output classes in the model, defaults to 5.

    Returns:
    np.ndarray
        A 3D numpy array representing the initial population of models.
    """

    # Generate a 3D numpy array of random binary digits,
    # where the first dimension is the number of models,
    # and the second and third dimensions are the characteristics of each model.
    first_population_array = np.random.randint(0, 2, (population, 9, 18))

    # Loop over each model in the population
    for i in range(population):
        # Create a model based on the characteristics encoded in the array
        model = create_model(first_population_array[i], num_classes=num_classes)

        # If the model has a problem, delete it and create a new one with different random characteristics
        while model_has_problem(model):
            del model
            first_population_array[i] = np.random.randint(0, 2, (9, 18))
            model = create_model(first_population_array[i], num_classes=num_classes)

        # Delete the model to free up memory
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
    """
    Train and evaluate a population of models, and select the best ones based on their fitness.

    Parameters:
    train_ds : tf.data.Dataset
        The training dataset.
    val_ds : tf.data.Dataset
        The validation dataset.
    test_ds : tf.data.Dataset
        The test dataset.
    time : datetime or str
        A timestamp used in directory names for saving results.
    population_array : np.ndarray
        A 3D numpy array representing the population of models.
    generation : int
        The generation number of the models, used in directory names for saving results.
    epochs : int, optional
        The number of epochs to train each model for, defaults to 30.
    num_classes : int, optional
        The number of output classes in the model, defaults to 5.

    Returns:
    tuple
        A tuple containing a list of the best model arrays, the maximum fitness, and the average fitness of the population.
    """

    # Initialize lists to store the fitness, accuracy, and inference time of each model
    fitness_list = []
    tflite_accuracy_list = []
    tpu_time_list = []

    # Define directory names for saving results
    result_dir = f'results_{time}'
    generation_dir = result_dir + f'/generation_{generation}'
    best_models_arrays_dir = generation_dir + '/best_model_arrays.pkl'
    fitness_list_dir = generation_dir + '/fitness_list.pkl'
    tflite_accuracy_list_dir = generation_dir + '/tflite_accuracy_list.pkl'
    tpu_time_list_dir = generation_dir + '/tpu_time_list.pkl'

    # Create the directories if they do not exist
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(generation_dir):
        os.makedirs(generation_dir)

    # Loop over each model in the population
    for i in range(population_array.shape[0]):
        # Create and train a model based on the characteristics encoded in the array
        model = create_model(population_array[i], num_classes=num_classes)
        model, _ = train_model(train_ds, val_ds, model=model, epochs=epochs)

        try:
            # Convert the model to TensorFlow Lite format and compile it for Edge TPU
            _, tflite_name = convert_to_tflite(keras_model=model, generation=generation, i=i, time=time)
            edgetpu_name = compile_edgetpu(tflite_name)

            # Evaluate the TensorFlow Lite model and get its accuracy and inference time
            tflite_accuracy, tpu_time = evaluate_tflite_model(tflite_model=edgetpu_name, tfl_int8=True)

        except:
            # If any error occurs during the conversion or evaluation process, set the accuracy to 0 and the inference time to a large value
            tflite_accuracy = 0
            tpu_time = 9999

        # Calculate the fitness of the model
        fitness = calculate_fitness(tflite_accuracy, tpu_time)

        # Add the accuracy, fitness, and inference time to their respective lists
        tflite_accuracy_list.append(tflite_accuracy)
        fitness_list.append(fitness)
        tpu_time_list.append(tpu_time)

        # Save the accuracy, fitness, and inference time lists to files
        with open(tflite_accuracy_list_dir, 'wb') as f:
            pickle.dump(tflite_accuracy_list, f)
        with open(tpu_time_list_dir, 'wb') as f:
            pickle.dump(tpu_time_list, f)

    # Calculate the maximum and average fitness of the population
    max_fitness = np.max(fitness_list)
    average_fitness = np.average(fitness_list)

    # Select the indices of the 5 models with the highest fitness
    best_models_indices = sorted(range(len(fitness_list)), key=lambda j: fitness_list[j], reverse=True)[:5]
    # Use the indices to select the model arrays from the population
    best_models_arrays = [population_array[k] for k in best_models_indices]

    # Save the array of the best models to a file
    with open(best_models_arrays_dir, 'wb') as f:
        pickle.dump(best_models_arrays, f)

    # Return a tuple containing the array of the best models, the maximum fitness, and the average fitness
    return best_models_arrays, max_fitness, average_fitness


def crossover(parent_arrays):
    """
    Performs crossover operation on a list of parent arrays to generate a child array.

    Parameters:
    parent_arrays : list of np.ndarray
        A list of parent arrays.

    Returns:
    np.ndarray
        A child array that is a combination of the parent arrays.
    """

    # Generate a same-sized array filled with random integers between 0 and 4 (inclusive),
    # which will be used as indices to select elements from the parent arrays
    parent_indices = np.random.randint(0, 5, size=parent_arrays[0].shape)

    # Use the indices array to select elements from the parent arrays and form a new child array
    child_array = np.choose(parent_indices, parent_arrays)

    return child_array


def mutate(model_array, mutate_prob=0.05):
    """
    Performs mutation operation on a given model array.

    Parameters:
    model_array : np.ndarray
        The model array to be mutated.
    mutate_prob : float, optional
        The probability of mutation for each element in the array, defaults to 0.05.

    Returns:
    np.ndarray
        The mutated model array.
    """

    # Generate a same-sized array filled with random floats between 0 and 1 (inclusive)
    prob = np.random.uniform(size=(9, 18))

    # Perform mutation operation: if the randomly generated number for a position is less than mutation probability,
    # flip the bit at that position in the model array; else, keep the original bit
    mutated_array = np.where(prob < mutate_prob, np.logical_not(model_array), model_array)

    return mutated_array


def create_next_population(parent_arrays, population=20, num_classes=5):
    """
    Creates the next generation of model arrays by performing crossover and mutation operations.

    Parameters:
    parent_arrays : list of np.ndarray
        A list of parent arrays.
    population : int, optional
        The size of the population to be generated, defaults to 20.
    num_classes : int, optional
        The number of classes for the model, defaults to 5.

    Returns:
    np.ndarray
        The next generation of model arrays.
    """

    # Initialize the next generation with random integers between 0 and 1
    next_population_array = np.random.randint(0, 2, (population, 9, 18))

    # For each individual in the population
    for individual in range(population):
        # Perform crossover operation using parent arrays
        next_population_array[individual] = crossover(parent_arrays)
        # Perform mutation operation with a mutation probability of 0.03
        next_population_array[individual] = mutate(next_population_array[individual], mutate_prob=0.03)

    # For each individual in the population
    for individual in range(population):
        # Create a model using the individual's model array
        model = create_model(next_population_array[individual], num_classes=num_classes)
        # If the model has a problem
        while model_has_problem(model):
            # Delete the model
            del model
            # Perform crossover operation using parent arrays
            next_population_array[individual] = crossover(parent_arrays)
            # Perform mutation operation with a mutation probability of 0.03
            next_population_array[individual] = mutate(next_population_array[individual], mutate_prob=0.03)
            # Create a new model using the updated individual's model array
            model = create_model(next_population_array[individual], num_classes=num_classes)
        # Delete the model after checking
        del model

    # Return the next generation of model arrays
    return next_population_array


def start_evolution(train_ds, val_ds, test_ds, generations, population, num_classes, epochs, population_array=None,
                    time=None):
    """
    Starts the evolutionary process for model optimization.

    Parameters:
    train_ds : tensorflow Dataset
        The training dataset.
    val_ds : tensorflow Dataset
        The validation dataset.
    test_ds : tensorflow Dataset
        The testing dataset.
    generations : int
        The number of generations to evolve through.
    population : int
        The size of the population in each generation.
    num_classes : int
        The number of classes in the target variable.
    epochs : int
        The number of epochs to train each model.
    population_array : np.ndarray, optional
        The initial population array.
    time : datetime or str, optional
        The timestamp to append to the result directory name.

    Returns:
    tuple
        The final population array, the history of maximum fitness score,
        the history of average fitness score, and the best model arrays.
    """

    # Initialize fitness histories
    max_fitness_history = []
    average_fitness_history = []

    # If no initial population is provided, create one
    if population_array is None:
        population_array = create_first_population(population=32, num_classes=num_classes)

    # Define the results directory and create it if it does not exist
    result_dir = f'results_{time}'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # For each generation
    for generation in range(generations):
        # Select the best models, calculate the maximum and average fitness
        best_models_arrays, max_fitness, average_fitness = select_models(train_ds=train_ds, val_ds=val_ds,
                                                                         test_ds=test_ds, time=time,
                                                                         population_array=population_array,
                                                                         generation=generation, epochs=epochs,
                                                                         num_classes=num_classes)

        # Create the next generation population
        population_array = create_next_population(parent_arrays=best_models_arrays, population=population,
                                                  num_classes=num_classes)

        # Record the maximum and average fitness
        max_fitness_history.append(max_fitness)
        average_fitness_history.append(average_fitness)

        # Define the file paths for saving results
        next_population_array_dir = result_dir + '/next_population_array.pkl'
        max_fitness_history_dir = result_dir + '/max_fitness_history.pkl'
        average_fitness_history_dir = result_dir + '/average_fitness_history.pkl'
        best_model_arrays_dir = result_dir + '/best_model_arrays.pkl'

        # Save the results
        with open(next_population_array_dir, 'wb') as f:
            pickle.dump(population_array, f)
        with open(max_fitness_history_dir, 'wb') as f:
            pickle.dump(max_fitness_history, f)
        with open(average_fitness_history_dir, 'wb') as f:
            pickle.dump(average_fitness_history, f)
        with open(best_model_arrays_dir, 'wb') as f:
            pickle.dump(best_models_arrays, f)

    # Return the final population array, the history of maximum fitness score,
    # the history of average fitness score, and the best model arrays
    return population_array, max_fitness_history, average_fitness_history, best_models_arrays
