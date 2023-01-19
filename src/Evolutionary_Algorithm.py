import numpy as np

from src.Create_Model import create_model, model_summary
from src.Evaluate_Model import model_evaluation
from src.Train_Model import train_model


def create_first_population(population=10):
    first_population_array = np.random.randint(0, 2, (population, 9, 18))

    return first_population_array


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
        trained_model, history = train_model(train_ds, val_ds, model=model, epochs=epochs)
        acc = model_evaluation(trained_model, test_ds)
        fitness_list.append(acc)

    # TODO: Use custom fitness function to select the individual.

    best_models_indices = sorted(range(len(fitness_list)), key=lambda i: fitness_list[i], reverse=True)[:2]
    best_models_array = [population_array[i] for i in best_models_indices]

    return best_models_array[0], best_models_array[1]


def crossover(parent_1_array, parent_2_array):
    mask = np.random.permutation(np.concatenate((np.zeros(9*18//2),
                                                 np.ones(9*18//2)))).reshape(9, 18).astype(np.bool_)
    child_array = np.where(mask, parent_1_array, parent_2_array)
    return child_array


def mutate(model_array, mutate_prob=0.02):
    prob = np.random.uniform(size=(9, 18))
    mutated_array = np.where(prob < mutate_prob, np.logical_not(model_array), model_array)

    return mutated_array


def create_next_population(parent_1_array, parent_2_array, population=10):
    next_population_array = np.random.randint(0, 2, (population, 9, 18))
    for i in range(population):
        next_population_array[i] = crossover(parent_1_array, parent_2_array)
        next_population_array[i] = mutate(next_population_array[i], mutate_prob=0.1)

    return next_population_array
