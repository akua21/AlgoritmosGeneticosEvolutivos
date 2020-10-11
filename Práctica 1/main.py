# Práctica 1 AGE - Optimización de Sensores en Smart Cities

# Imports
import requests
import numpy as np
import random
import math
import matplotlib.pyplot as plt

# Constantes
WEBSITE = "http://memento.evannai.inf.uc3m.es/age/test?c="
NUM_SENSORS = 4 * 16 # tamaño cromosoma, 4 estaciones con 16 sensores cada una

SIZE_POPULATION = 50 # tamaño población, número de individuos de una población
CYCLES = 10 # número de ciclos(generaciones)



# Funciones

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
        evaluation = getRequest(''.join(map(str, chromosome)))  # ?
        evaluations.append(evaluation)

    # print(evaluations)
    return evaluations


# Seleccionar a los mejores
# Torneos (posibilidad de elitismo)
def tournamentSelection(population, population_evaluation, p_tournament=0.05, elitism=False, size_elite=0):
    size_popu = len(population)
    tournament_size = max(math.floor(p_tournament * size_popu), 2)

    selected_population = []
    index_winner = 0

    # Si se quiere elitismo entonces nos guardamos los mejores individuos(size_elite) antes del torneo
    if elitism:
        sorted_population_eval = sorted(population_evaluation) # sorted no modifica population_evaluation # ?

        for i in range(size_elite):
            selected_population.append(population[popu_evaluation.index(sorted_population_eval[i])])


    # print("selected_population antes del torneo: ", selected_population)
    for i in range(size_popu - size_elite):
        aux_eval = math.inf

        index_selected_individuals = random.sample(range(size_popu), tournament_size)
        # print(index_selected_individuals)

        for index in index_selected_individuals:
            if aux_eval > population_evaluation[index]:
                index_winner = index
                aux_eval = population_evaluation[index_winner]

        selected_population.append(population[index_winner])

    # print("selected_population después del torneo: ", selected_population)
    return selected_population


# Operadores de cruce
# Uniforme
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

# Simple
def crossingSimple(parent_1, parent_2):
    child_1 = []
    child_2 = []

    divide_number = random.randint(0, len(parent_1)) # este numero va a ser distinto cada vez que llame a crossingSimple, no sé si tiene que ser así o no ¿?
    for i in range(len(parent_1)):
        if i < divide_number:
            child_1.append(parent_1[i])
            child_2.append(parent_2[i])

        elif i >= divide_number:
            child_1.append(parent_2[i])
            child_2.append(parent_1[i])

    return child_1, child_2

# Dos puntos
def crossing2Points(parent_1, parent_2):
    child_1 = []
    child_2 = []

    divide_number_1 = random.randint(0, len(parent_1)) # este numero va a ser distinto cada vez que llame a crossingSimple, no sé si tiene que ser así o no ¿?
    divide_number_2 = random.randint(0, len(parent_1))

    for i in range(len(parent_1)):
        if i < divide_number_1:
            child_1.append(parent_1[i])
            child_2.append(parent_2[i])

        elif i >= divide_number_1 and i < divide_number_2:
            child_1.append(parent_2[i])
            child_2.append(parent_1[i])

        elif i >= divide_number_2:
            child_1.append(parent_1[i])
            child_2.append(parent_2[i])

    return child_1, child_2

# Cruzamiento
def crossing(population, uniform=False, simple=False, two_points=False):
    childs = []

    while len(population) > 0:
        parent_1 = population.pop(random.randint(0, len(population)-1))
        parent_2 = population.pop(random.randint(0, len(population)-1))

        if uniform:
            child_1, child_2 = crossingUniform(parent_1, parent_2)

            childs.append(child_1)
            childs.append(child_2)

        elif simple:
            child_1, child_2 = crossingSimple(parent_1, parent_2)

            childs.append(child_1)
            childs.append(child_2)

        elif two_points:
            child_1, child_2 = crossing2Points(parent_1, parent_2)

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





# MAIN--------------------------------------------------------------------------

# INICIALIZAR POBLACIÓN
popu =initPopulation(SIZE_POPULATION, NUM_SENSORS)
# print("Initial population: ", popu)

# Lista con el mejor individuo de cada generación
best_of_generation = []
# Lista con la evaluación(fitness) del mejor individuo de cada generación
best_of_generation_eval = []

# BUCLE PRINCIPAL
for i in range(CYCLES):

    # EVALUACIÓN
    popu_evaluation = evaluatePopulation(popu)
    best_of_generation_eval.append(min(popu_evaluation))
    best_of_generation.append(popu[popu_evaluation.index(min(popu_evaluation))])
    print("Generation ", i)
    # print("Evaluation initial population: ", popu_evaluation)


    # SELECCIÓN POR TORNEO
    selected_popu = tournamentSelection(popu, popu_evaluation)
    # print("Selected population from TOURNAMENT: ", selected_popu)
    # print("Evaluation selected population: ", evaluatePopulation(selected_popu))

    # SELECCIÓN POR TORNEO CON ELITISMO
    # selected_popu = tournamentSelection(popu, popu_evaluation, elitism=True, size_elite=3)


    # CRUZAMIENTO POR CRUCE UNIFORME
    childs = crossing(selected_popu, uniform=True)
    # print("Childs of selected population: ", childs)

    # CRUZAMIENTO POR CRUCE SIMPLE
    childs = crossing(selected_popu, simple=True)

    # CRUZAMIENTO POR CRUCE DOS PUNTOS
    childs = crossing(selected_popu, two_points=True)


    # MUTACIÓN
    mutated_popu = mutation(childs)
    # print("Childs mutated: ", mutated_popu)

    # for i in range(len(childs)):
    #     print(childs[i] == mutated_popu[i])

    popu = mutated_popu




print(best_of_generation_eval)
print()
# print(best_of_generation)

plt.plot(best_of_generation_eval)
plt.xlabel('Generations')
plt.ylabel('Fitness value')
plt.show()
