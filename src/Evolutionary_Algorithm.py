import concurrent.futures

import numpy as np

from src.Create_Model import create_model, model_summary
from src.Evaluate_Model import model_evaluation
from src.Fitness_Function import calculate_fitness
from src.Train_Model import train_model
from tools.Model_Checker import check_large_model


def create_first_population(population=10, num_classes=2):
    first_population_array = np.random.randint(0, 2, (population, 9, 18))

    for i in range(population):
        model = create_model(first_population_array[i], num_classes=num_classes)
        while check_large_model(model):
            first_population_array[i] = np.random.randint(0, 2, (9, 18))
            model = create_model(first_population_array[i], num_classes=num_classes)

    return first_population_array


# def train_and_evaluate(i, population_array, train_ds, val_ds, test_ds, epochs, num_classes):
#     model = create_model(population_array[i], num_classes=num_classes)
#     trained_model, history = train_model(train_ds, val_ds, model=model, epochs=epochs)
#     acc = model_evaluation(trained_model, test_ds)
#     fitness = calculate_fitness(acc)
#     return i, fitness
#
#
# def select_best_2_model(train_ds, val_ds, test_ds, population_array, epochs=20, num_classes=2):
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         fitness_results = [
#             executor.submit(train_and_evaluate, i, population_array, train_ds, val_ds, test_ds, epochs, num_classes) for
#             i in range(population_array.shape[0])]
#         fitness_list = [r.result()[1] for r in fitness_results]
#
#     best_models_indices = sorted(range(len(fitness_list)), key=lambda i: fitness_list[i], reverse=True)[:2]
#     best_models_array = [population_array[i] for i in best_models_indices]
#
#     return best_models_array[0], best_models_array[1]


def select_best_2_model(train_ds,
                        val_ds,
                        test_ds,
                        population_array,
                        epochs=20,
                        num_classes=2):
    fitness_list = []
    # tflite_accuracies = []
    for i in range(population_array.shape[0]):
        model = create_model(population_array[i], num_classes=num_classes)
        model_summary(model)
        trained_model, _ = train_model(train_ds, val_ds, model=model, epochs=epochs)
        acc = model_evaluation(trained_model, test_ds)

        # TODO: Calculate the memory_footprint_edge and inference_time
        #       Need a Linux

        fitness = calculate_fitness(acc)
        fitness_list.append(fitness)

    best_models_indices = sorted(range(len(fitness_list)), key=lambda i: fitness_list[i], reverse=True)[:2]
    best_models_array = [population_array[i] for i in best_models_indices]

    return best_models_array[0], best_models_array[1]


def crossover(parent_1_array, parent_2_array, probability_of_1=0.5):
    mask = np.random.binomial(1, probability_of_1, size=(9, 18)).astype(np.bool_)
    child_array = np.where(mask, parent_1_array, parent_2_array)
    return child_array


# def crossover(parent_1_array, parent_2_array):
#     mask = np.random.randint(0, 2, size=(9, 18), dtype=np.bool_)
#     child_array = np.where(mask, parent_1_array, parent_2_array)
#
#     return child_array


def mutate(model_array, mutate_prob=0.02):
    prob = np.random.uniform(size=(9, 18))
    mutated_array = np.where(prob < mutate_prob, np.logical_not(model_array), model_array)

    return mutated_array


# TODO: Optimize the code, do not use for loop
def create_next_population(parent_1_array, parent_2_array, population=10, num_classes=2):
    next_population_array = np.random.randint(0, 2, (population, 9, 18))

    for i in range(population):
        next_population_array[i] = crossover(parent_1_array, parent_2_array)
        next_population_array[i] = mutate(next_population_array[i], mutate_prob=0.02)

    for i in range(population):
        model = create_model(next_population_array[i], num_classes=num_classes)
        while check_large_model(model):
            next_population_array[i] = crossover(parent_1_array, parent_2_array)
            next_population_array[i] = mutate(next_population_array[i], mutate_prob=0.02)
            model = create_model(next_population_array[i], num_classes=num_classes)

    return next_population_array
