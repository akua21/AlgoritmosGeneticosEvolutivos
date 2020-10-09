# Práctica 1

# Imports
import requests
import numpy as np
import random
import math

# Constantes
WEBSITE = "http://memento.evannai.inf.uc3m.es/age/test?c="
NUM_SENSORS = 64 # tamaño cromosoma

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
p_tournament = 0.4
def tournamentSelection(population, size_popu, population_evaluation):
    tournament_size = math.floor(p_tournament * size_popu)
    print(tournament_size)
    new_population = []
    best_individual = 0
    aux_eval = 0

    for i in range(size_popu):
        selected_individuals = random.sample(range(size_popu), tournament_size)
        print(selected_individuals)
        for individual in selected_individuals:
            if aux_eval < population_evaluation[individual]:
                best_individual = individual
                aux_eval = best_individual

        new_population.append(population[best_individual])

    return new_population



# Operador de cruce (simple, de dos puntos, uniforme)

# Operador de mutación





# MAIN
popu =initPopulation(5, NUM_SENSORS)
popu_evaluation = evaluatePopulation(popu)
new_popu = tournamentSelection(popu, 5, popu_evaluation)
