import numpy as np


from src.Create_Model import create_model, model_summary
from src.Evaluate_Model import model_evaluation
from src.Fitness_Function import calculate_fitness
from src.Train_Model import train_model
from tools.Model_Checker import check_large_model

'''This function creates the first population of models for a genetic algorithm, by default it creates 10 models. The 
function takes two inputs, population and num_classes. population is the number of models to be created and 
num_classes is the number of output classes for each model.

The function first creates an array called first_population_array, which is filled with random integers between 0 and 
1 of shape (population, 9, 18).

Then it iterates through the population and creates a model using the create_model function, passing in the array 
element of first_population_array and the number of output classes. It then uses the check_large_model function to 
check if the model has any MultiHeadAttention layers with an output size greater than 1024. If it finds such a layer, 
it regenerates the random array element and creates the model again, until it finds a model that is not violating the 
rule.

Finally it returns the first_population_array with all the models that don't have a MultiHeadAttention layer with 
output size greater than 1024.'''


def create_first_population(population=10, num_classes=5):
    first_population_array = np.random.randint(0, 2, (population, 9, 18))

    for i in range(population):
        model = create_model(first_population_array[i], num_classes=num_classes)
        while check_large_model(model):
            first_population_array[i] = np.random.randint(0, 2, (9, 18))
            model = create_model(first_population_array[i], num_classes=num_classes)

    return first_population_array


'''This function takes in several inputs including the training, validation and test datasets, an array of models (
population_array), the number of output classes and the number of training epochs. It then iterates through the 
population_array, creating a model for each array element using the create_model function and passing in the array 
element and number of output classes. It then trains the model using the train_model function, passing in the 
train_ds, val_ds and the number of epochs. Then it evaluates the model on the test_ds using the model_evaluation 
function.

It then uses the accuracy of the model to calculate the fitness of the model using the calculate_fitness function. It 
appends the fitness of each model to the fitness_list.

After evaluating all models in population_array, it finds the indices of the two models with the highest fitness 
values by sorting the fitness_list in descending order and taking the first two indices. It then creates an array of 
the two best models using these indices and returns this array.'''


def select_best_2_model(train_ds,
                        val_ds,
                        test_ds,
                        population_array,
                        epochs=30,
                        num_classes=5):
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

    max_fitness = np.max(fitness_list)
    average_fitness = np.average(fitness_list)

    best_models_indices = sorted(range(len(fitness_list)), key=lambda i: fitness_list[i], reverse=True)[:2]
    best_models_array = [population_array[i] for i in best_models_indices]

    print("max_fitness: ", max_fitness, "\n", "average_fitness: ", average_fitness)

    return best_models_array[0], best_models_array[1], max_fitness, average_fitness


# def crossover(parent_1_array, parent_2_array, probability_of_1=0.5):
#     mask = np.random.binomial(1, probability_of_1, size=(9, 18)).astype(np.bool_)
#     child_array = np.where(mask, parent_1_array, parent_2_array)
#     return child_array

'''This function takes in two inputs, parent_1_array and parent_2_array. The function performs a crossover operation 
on these two arrays, creating a new child array.

It starts by creating a boolean mask of shape (9,18) filled with random 0s and 1s using np.random.randint(). Then it 
uses np.where() function with the mask, parent_1_array and parent_2_array as inputs to create a new child_array. 
Elements of the child array are taken from parent_1_array where the mask is True, and from parent_2_array where the 
mask is False.

This way, the child array inherits certain characteristics from parent_1_array and certain characteristics from 
parent_2_array, simulating the crossover operation in genetic algorithms.'''


def crossover(parent_1_array, parent_2_array):
    mask = np.random.randint(0, 2, size=(9, 18), dtype=np.bool_)
    child_array = np.where(mask, parent_1_array, parent_2_array)

    return child_array


'''This function takes in two inputs, model_array and mutate_prob. The function performs a mutation operation on the 
model_array, creating a new mutated_array.

It starts by creating an array of random float values between 0 and 1 of shape (9,18) using np.random.uniform(). It 
then uses np.where() function with this array, mutate_prob and the input model_array as inputs. It compares each 
element of the random float array with mutate_prob, if the element is less than mutate_prob, it applies the numpy 
function logical_not to the corresponding element of model_array, otherwise it keeps the element unchanged.

Thus, this function will flip a certain percentage of elements in the model_array, simulating the mutation operation 
in genetic algorithms. The percentage of elements flipped is determined by the mutate_prob input, which is set to 
0.01 by default.'''


def mutate(model_array, mutate_prob=0.01):
    prob = np.random.uniform(size=(9, 18))
    mutated_array = np.where(prob < mutate_prob, np.logical_not(model_array), model_array)

    return mutated_array


'''This function takes in three inputs: parent_1_array, parent_2_array, population and num_classes. It creates the 
next population of models for a genetic algorithm by default, it creates 10 models.

The function starts by creating an array called next_population_array, which is filled with random integers between 0 
and 1 of shape (population, 9, 18).

Then it iterates through the population and applies the crossover function to the parent_1_array and parent_2_array, 
creating a new child array. Then it applies the mutate function to this child array, flipping a certain percentage of 
elements in the array.

After this, it iterates through the population again and creates a model using the create_model function, passing in 
the array element of next_population_array and the number of output classes. It then uses the check_large_model 
function to check if the model has any MultiHeadAttention layers with an output size greater than 1024. If it finds 
such a layer, it regenerates the child array by applying crossover and mutate again, until it finds a model that is 
not violating the rule.

Finally, it returns the next_population_array with all the models that don't have a MultiHeadAttention layer with 
output size greater than 1024.

It also uses the mutate_prob=0.01 as default value, which means it will flip 1% of elements'''


# TODO: Optimize the code, do not use for loop
def create_next_population(parent_1_array, parent_2_array, population=10, num_classes=5):
    next_population_array = np.random.randint(0, 2, (population, 9, 18))

    for i in range(population):
        next_population_array[i] = crossover(parent_1_array, parent_2_array)
        next_population_array[i] = mutate(next_population_array[i], mutate_prob=0.01)

    for i in range(population):
        model = create_model(next_population_array[i], num_classes=num_classes)
        while check_large_model(model):
            next_population_array[i] = crossover(parent_1_array, parent_2_array)
            next_population_array[i] = mutate(next_population_array[i], mutate_prob=0.01)
            model = create_model(next_population_array[i], num_classes=num_classes)

    return next_population_array


'''This function implements a genetic algorithm for training a neural network. The function takes in 
several parameters such as a training dataset, validation dataset, test dataset, number of generations, population 
size, number of classes, and number of epochs. If a population array is not provided, it creates an initial 
population of models using the create_first_population function. The function then loops through the specified number 
of generations, where in each iteration it selects the best 2 models using the select_best_2_model function, 
and creates a new population using the create_next_population function. The final population array is returned.'''


def start_evolution(train_ds, val_ds, test_ds, generations, population, num_classes, epochs, population_array=None):
    max_fitness_history = []
    average_fitness_history = []
    if population_array is None:
        population_array = create_first_population(population=population, num_classes=num_classes)

    for i in range(generations):
        print('Generations: ', i)
        a, b, max_fitness, average_fitness = select_best_2_model(train_ds, val_ds, test_ds, population_array, epochs=epochs, num_classes=num_classes)
        population_array = create_next_population(a, b, population=population, num_classes=num_classes)
        max_fitness_history.append(max_fitness)
        average_fitness_history.append(average_fitness)

    print("max_fitness_history: ", max_fitness_history, "\n", "average_fitness_history: ", average_fitness_history)

    return population_array, max_fitness_history, average_fitness_history, a, b
