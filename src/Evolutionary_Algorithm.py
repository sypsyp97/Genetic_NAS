import numpy as np

from src.Create_Model import create_model, model_summary
from src.Evaluate_Model import model_evaluation
from src.Fitness_Function import calculate_fitness
from src.Create_Model import train_model
from tools.Model_Checker import check_large_model


def create_first_population(population=10, num_classes=5):
    first_population_array = np.random.randint(0, 2, (population, 9, 18))

    for i in range(population):
        model = create_model(first_population_array[i], num_classes=num_classes)
        while check_large_model(model):
            first_population_array[i] = np.random.randint(0, 2, (9, 18))
            model = create_model(first_population_array[i], num_classes=num_classes)

    return first_population_array


def roulette_wheel_selection(fitness_list):
    # Normalize the fitness values to obtain probabilities
    fitness_sum = np.sum(fitness_list)
    probabilities = [fitness / fitness_sum for fitness in fitness_list]

    # Generate a random number between 0 and 1
    rand_num = np.random.rand()

    # Select the individual based on the roulette wheel
    prob_sum = 0
    for i, prob in enumerate(probabilities):
        prob_sum += prob
        if rand_num <= prob_sum:
            return i


def select_best_2_model(train_ds,
                        val_ds,
                        test_ds,
                        population_array,
                        epochs=30,
                        num_classes=5):
    fitness_list = []
    for i in range(population_array.shape[0]):
        model = create_model(population_array[i], num_classes=num_classes)
        model_summary(model)
        trained_model, _ = train_model(train_ds, val_ds, model=model, epochs=epochs)
        acc = model_evaluation(trained_model, test_ds)
        # fitness = calculate_fitness(acc)
        fitness = acc
        fitness_list.append(fitness)

    max_fitness = np.max(fitness_list)
    average_fitness = np.average(fitness_list)

    # Select the two best individuals using Roulette Wheel Selection
    best_models_indices = [roulette_wheel_selection(fitness_list) for _ in range(2)]
    best_models_array = [population_array[i] for i in best_models_indices]
    print(best_models_array[0])
    print(best_models_array[1])
    print("max_fitness: ", max_fitness, "\n", "average_fitness: ", average_fitness)

    return best_models_array[0], best_models_array[1], max_fitness, average_fitness


# def select_best_2_model(train_ds,
#                         val_ds,
#                         test_ds,
#                         population_array,
#                         epochs=30,
#                         num_classes=5):
#     fitness_list = []
#     # tflite_accuracies = []
#     for i in range(population_array.shape[0]):
#         model = create_model(population_array[i], num_classes=num_classes)
#         model_summary(model)
#         trained_model, _ = train_model(train_ds, val_ds, model=model, epochs=epochs)
#         acc = model_evaluation(trained_model, test_ds)
#
#         # TODO: Calculate the memory_footprint_edge and inference_time
#         #       Need a Linux
#
#         # fitness = calculate_fitness(acc)
#         fitness = acc
#         fitness_list.append(fitness)
#
#     max_fitness = np.max(fitness_list)
#     average_fitness = np.average(fitness_list)
#
#     best_models_indices = sorted(range(len(fitness_list)), key=lambda i: fitness_list[i], reverse=True)[:2]
#     best_models_array = [population_array[i] for i in best_models_indices]
#     print(best_models_array[0])
#     print(best_models_array[1])
#     print("max_fitness: ", max_fitness, "\n", "average_fitness: ", average_fitness)
#
#     return best_models_array[0], best_models_array[1], max_fitness, average_fitness


def crossover(parent_1_array, parent_2_array):
    mask = np.random.binomial(1, 0.5, size=(9, 18)).astype(np.bool_)
    # mask = np.random.randint(0, 2, size=(9, 18), dtype=np.bool_)
    child_array = np.where(mask, parent_1_array, parent_2_array)

    return child_array


def mutate(model_array, mutate_prob=0.025):
    prob = np.random.uniform(size=(9, 18))
    mutated_array = np.where(prob < mutate_prob, np.logical_not(model_array), model_array)

    return mutated_array


# TODO: Optimize the code, do not use for loop
def create_next_population(parent_1_array, parent_2_array, population=10, num_classes=5):
    next_population_array = np.random.randint(0, 2, (population, 9, 18))

    for i in range(population):
        next_population_array[i] = crossover(parent_1_array, parent_2_array)
        next_population_array[i] = mutate(next_population_array[i], mutate_prob=0.025)

    for i in range(population):
        model = create_model(next_population_array[i], num_classes=num_classes)
        while check_large_model(model):
            next_population_array[i] = crossover(parent_1_array, parent_2_array)
            next_population_array[i] = mutate(next_population_array[i], mutate_prob=0.025)
            model = create_model(next_population_array[i], num_classes=num_classes)

    return next_population_array


def start_evolution(train_ds, val_ds, test_ds, generations, population, num_classes, epochs, population_array=None):
    max_fitness_history = []
    average_fitness_history = []
    if population_array is None:
        population_array = create_first_population(population=population, num_classes=num_classes)

    for i in range(generations):
        a, b, max_fitness, average_fitness = select_best_2_model(train_ds, val_ds, test_ds, population_array,
                                                                 epochs=epochs, num_classes=num_classes)
        population_array = create_next_population(a, b, population=population, num_classes=num_classes)
        max_fitness_history.append(max_fitness)
        average_fitness_history.append(average_fitness)
        print('Generations: ', i)
        print("max_fitness_history: ", max_fitness_history, "\n", "average_fitness_history: ", average_fitness_history)
        print("best_parents_1: ", a)
        print("best_parents_2: ", b)
        print("Next population: ", population_array)

    return population_array, max_fitness_history, average_fitness_history, a, b
