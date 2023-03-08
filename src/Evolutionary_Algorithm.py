import numpy as np

from tools.Create_Model import create_model, model_summary
from tools.Evaluate_Model import evaluate_tflite_model
from tools.Create_Model import train_model
from tools.Model_Checker import check_model
from tools.TFLITE_Converter import convert_to_tflite
from tools.Compile_Edge_TPU import compile_edgetpu
from tools.Inference_Speed_TPU import inference_time_tpu
from src.Fitness_Function import calculate_fitness


def create_first_population(population=100, num_classes=5):
    first_population_array = np.random.randint(0, 2, (population, 9, 18))

    for i in range(population):
        model = create_model(first_population_array[i], num_classes=num_classes)
        while check_model(model):
            first_population_array[i] = np.random.randint(0, 2, (9, 18))
            model = create_model(first_population_array[i], num_classes=num_classes)

    return first_population_array


def select_models(train_ds,
                  val_ds,
                  test_ds,
                  time,
                  population_array,
                  generation,
                  epochs=30,
                  num_classes=5,):

    fitness_list = []
    tflite_accuracy_list = []
    tpu_time_list = []

    for i in range(population_array.shape[0]):
        model = create_model(population_array[i], num_classes=num_classes)
        trained_model, _ = train_model(train_ds, val_ds, model=model, epochs=epochs)
        try:
            tflite_model, tflite_name= convert_to_tflite(keras_model=trained_model, generation=generation, i=i, time=time)
            tflite_accuracy = evaluate_tflite_model(tflite_model=tflite_model, tfl_int8=True)
        except:
            tflite_accuracy = 0

        try:
            edgetpu_name = compile_edgetpu(tflite_name)
            tpu_time = inference_time_tpu(edgetpu_model_name=edgetpu_name)
        except:
            tpu_time = 9999

        fitness = calculate_fitness(tflite_accuracy, tpu_time)

        tflite_accuracy_list.append(tflite_accuracy)
        fitness_list.append(fitness)
        tpu_time_list.append(tpu_time)

    max_fitness = np.max(fitness_list)
    average_fitness = np.average(fitness_list)

    best_models_indices = sorted(range(len(fitness_list)), key=lambda j: fitness_list[j], reverse=True)[:5]
    best_models_arrays = [population_array[k] for k in best_models_indices]
    print("best_parent_1: ", best_models_arrays[0])
    print("best_parent_2: ", best_models_arrays[1])
    print("best_parent_3: ", best_models_arrays[2])
    print("best_parent_4: ", best_models_arrays[3])
    print("best_parent_5: ", best_models_arrays[4])
    print("Fitness in Generation: ", fitness_list)
    print("Accuracy in Generation: ", tflite_accuracy_list)
    print("Inference Time in Generation: ", tpu_time_list)

    print("max_fitness: ", max_fitness, "\n", "average_fitness: ", average_fitness)

    return best_models_arrays, max_fitness, average_fitness


def crossover(parent_arrays):
    parent_indices = np.random.randint(0, 5, size=parent_arrays[0].shape)
    child_array = np.choose(parent_indices, parent_arrays)
    return child_array


def mutate(model_array, mutate_prob=0.025):
    prob = np.random.uniform(size=(9, 18))
    mutated_array = np.where(prob < mutate_prob, np.logical_not(model_array), model_array)

    return mutated_array


# TODO: Optimize the code, do not use for loop
def create_next_population(parent_arrays, population=10, num_classes=5):
    next_population_array = np.random.randint(0, 2, (population, 9, 18))

    for i in range(population):
        next_population_array[i] = crossover(parent_arrays)
        next_population_array[i] = mutate(next_population_array[i], mutate_prob=0.025)

    for i in range(population):
        model = create_model(next_population_array[i], num_classes=num_classes)
        while check_model(model):
            next_population_array[i] = crossover(parent_arrays)
            next_population_array[i] = mutate(next_population_array[i], mutate_prob=0.025)
            model = create_model(next_population_array[i], num_classes=num_classes)

    return next_population_array


def start_evolution(train_ds, val_ds, test_ds, generations, population, num_classes, epochs, population_array=None, time=None):
    max_fitness_history = []
    average_fitness_history = []
    if population_array is None:
        population_array = create_first_population(population=population, num_classes=num_classes)

    for i in range(generations):
        best_models_arrays, max_fitness, average_fitness = select_models(train_ds, val_ds, test_ds, population_array,
                                                                         generation=i, epochs=epochs, num_classes=num_classes, time=time)
        population_array = create_next_population(parent_arrays=best_models_arrays, population=population, num_classes=num_classes)
        max_fitness_history.append(max_fitness)
        average_fitness_history.append(average_fitness)
        print('Generations: ', i)
        print("max_fitness_history: ", max_fitness_history, "\n", "average_fitness_history: ", average_fitness_history)

    return population_array, max_fitness_history, average_fitness_history, best_models_arrays
