# Práctica 1

# Imports
import requests
import numpy as np
import random
import math
import matplotlib.pyplot as plt

# Constantes
WEBSITE = "http://memento.evannai.inf.uc3m.es/age/test?c="
NUM_SENSORS = 4 * 16 # tamaño cromosoma, 4 estaciones con 16 sensores cada una

# Petición al servidor
def getRequest(chromosome):

    # Valor de evaluación
    request = float(requests.get(WEBSITE + chromosome).text)

    return request


# Inicializar población
def initPopulation(size_popu, num_sensors):
    population = []
    for i in range(size_popu):
        population.append(list(np.random.randint(2, size=num_sensors)))

    # print(population)
    return population


# Evaluar población
def evaluatePopulation(population):
    evaluations = []
    for chromosome in population:
        evaluation = getRequest(''.join(map(str, chromosome)))
        evaluations.append(evaluation)

    # print(evaluations)
    return evaluations


# Seleccionar a los mejores (ruleta, jerárquica, torneos)
# Torneos
def tournamentSelection(population, population_evaluation, p_tournament=0.05):
    size_popu = len(population)
    tournament_size = max(math.floor(p_tournament * size_popu), 2)
    # print(tournament_size)
    new_population = []
    index_winner = 0

    for i in range(size_popu):
        aux_eval = math.inf

        index_selected_individuals = random.sample(range(size_popu), tournament_size)
        # print(index_selected_individuals)

        for index in index_selected_individuals:
            if aux_eval > population_evaluation[index]:
                index_winner = index
                aux_eval = population_evaluation[index_winner]
        # print("index_winner ", index_winner)
        new_population.append(population[index_winner])

    return new_population


# Operador de cruce (simple, de dos puntos, uniforme)
def crossingUniform(parent_1, parent_2):
    child_1 = []
    child_2 = []

    for gene in range(len(parent_1)):
        random_1  = random.random()
        random_2  = random.random()

        if random_1 >= 0.5:
            child_1.append(parent_1[gene])
        elif random_1 < 0.5:
            child_1.append(parent_2[gene])

        if random_2 >= 0.5:
            child_2.append(parent_1[gene])
        elif random_2 < 0.5:
            child_2.append(parent_2[gene])

    return child_1, child_2

def crossing(population):
    childs = []

    while len(population) > 0:
        parent_1 = population.pop(random.randint(0, len(population)-1))
        parent_2 = population.pop(random.randint(0, len(population)-1))

        child_1, child_2 = crossingUniform(parent_1, parent_2)

        childs.append(child_1)
        childs.append(child_2)

    return childs


# Operador de mutación
def mutation(population):
    probability = 1/NUM_SENSORS
    mutated_population = []

    for i in range(len(population)):
        mutated_population.append([])
        for j in range(len(population[i])):
            if random.random() <= probability:
                mutated_population[i].append(1 - population[i][j])
            else:
                mutated_population[i].append(population[i][j])

    return mutated_population





# MAIN
popu =initPopulation(10, NUM_SENSORS)
# print("Initial population: ", popu)

best_by_generation = []

# Bucle principal
for i in range(10):
    popu_evaluation = evaluatePopulation(popu)
    best_by_generation.append(min(popu_evaluation))
    print("Generation ", i + 1)
    # print("Evaluation initial population: ", popu_evaluation)

    selected_popu = tournamentSelection(popu, popu_evaluation)
    # print("Selected population from TOURNAMENT: ", selected_popu)
    # print("Evaluation selected population: ", evaluatePopulation(selected_popu))

    childs = crossing(selected_popu)
    # print("Childs of selected population: ", childs)

    mutated_popu = mutation(childs)
    # print("Childs mutated: ", mutated_popu)

    # for i in range(len(childs)):
    #     print(childs[i] == mutated_popu[i])

    popu = mutated_popu


print(best_by_generation)

plt.plot(best_by_generation)
plt.xlabel('Generations')
plt.ylabel('Fitness value')
plt.show()
