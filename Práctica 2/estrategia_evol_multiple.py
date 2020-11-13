# Práctica 2 AGE - Calibración de motores automática mediante estrategias evolutivas

# Imports
import requests
import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt

# Constantes
# Website para los 4 rotores
WEBSITE_4 = "http://163.117.164.219/age/robot4?c1=%s&c2=%s&c3=%s&c4=%s"

# Website para los 10 rotores
WEBSITE_10 = "http://163.117.164.219/age/robot4?c1=%s&c2=%s&c3=%s&c4=%s&c5=%s&c6=%s&c7=%s&c8=%s&c9=%s&c10=%s"

NUM_MOTORS = 4
SIZE_POPULATION = 4
NUM_CHILDS = 6

S = 10
TAU = 1/math.sqrt(2 * math.sqrt(NUM_MOTORS))
TAU_0 = 1/math.sqrt(2 * NUM_MOTORS)

# Funciones

# Petición al servidor
def getRequest(rotations):
    if len(rotations) == 4:
        str_value = WEBSITE_4 % (rotations[0], rotations[1], rotations[2], rotations[3])
    elif len(rotations) == 10:
        str_value = WEBSITE_10 % (rotations[0], rotations[1], rotations[2], rotations[3], rotations[4], rotations[5], rotations[6], rotations[7], rotations[8], rotations[9])

    request = float(requests.get(str_value).text)
    return request

# Inicializar población
def initPopulation(size_population, num_motors):
    population = []

    for individual in range(size_population):
        # Vector codificación
        x = []
        # Vector varianzas
        variances = []
        for i in range(num_motors):
            x.append(np.random.uniform(-180, 180))
            variances.append(np.random.uniform(200, 400))

        population.append([x, variances, None])
    return population

# Evaluar población
def evaluatePopulation(population):
    for individual in population:
        individual[2] = getRequest(individual[0])

# Ordenar población por menor evaluación(fitness)
def auxFunctionSort(individual):
    return individual[2]

def sortPopulation(population):
    population.sort(key=auxFunctionSort)


# Seleccionar los padres por torneo
def tournamentSelection(population, p_tournament=0.05, sorted_population=False, num_parents=2):
    size_popu = len(population)
    tournament_size = max(math.floor(p_tournament * size_popu), 2)
    parents = []

    for i in range(num_parents):
        index_selected_individuals = random.sample(range(size_popu), tournament_size)

        # Cuando se va a realizar el reemplazo por inserción la población ya está ordenada por su evaluación
        if sorted_population:
            parents.append(population[min(index_selected_individuals)])

        # Cuando se va a realizar el reemplazo por inclusión la población no está ordenada por su evaluación
        else:
            index_winner = 0
            aux_eval = math.inf

            for index in index_selected_individuals:
                if aux_eval > population[index][2]:
                    index_winner = index
                    aux_eval = population[index_winner][2]

            parents.append(population[index_winner])

    return parents

# Cruzamiento
def crossing(parents):
    # Vector de codificación
    x_vector_child = []
    # Vector de varianzas
    variances_vector_child = []
    for i in range(len(parents[0][0])):
        x_vector_child.append((sum(parent[0][i] for parent in parents))/len(parents[0][0]))
        variances_vector_child.append(random.choice([parent[1][i] for parent in parents]))

    child = [x_vector_child, variances_vector_child, None]
    return child

# Mutación
def mutation(individual):
    individual_mutated = [mutation_x(individual), mutation_variances(individual), None]
    return individual_mutated

# Mutación de la parte funcional
def mutation_x(individual):
    x_mutated = []

    x_vector = individual[0]
    variances_vector = individual[1]

    for i in range(len(x_vector)):
        gauss_value = random.gauss(0, variances_vector[i])
        x_mutated.append(x_vector[i] + gauss_value)
    return x_mutated

# Mutación de las varianzas
def mutation_variances(individual, scale=False):
    variances_vector = individual[1]
    variances_mutated = []

    for variance in variances_vector:
        gauss_value = random.gauss(0, TAU)

        if scale:
            gauss_value_scale = random.gauss(0, TAU_0)
            variances_mutated.append(math.exp(gauss_value_scale) * variance * math.exp(gauss_value))
        else:
            variances_mutated.append(variance * math.exp(gauss_value))

    return variances_mutated

# Generación de los hijos
def generateChildren(population, num_childs):
    children = []
    for i in range(num_childs):
        parents = tournamentSelection(population, sorted_population=True)

        child = crossing(parents)

        child_mutated = mutation(child)

        children.append(child_mutated)
    return children

# Nueva población - Reemplazo por Inserción o Inclusión
def newPopulation(population, children, replace="insertion"):
    new_population = []

    if replace == "insertion":
        if len(population) <= len(children):
            evaluatePopulation(children)
            sortPopulation(children)
            new_population = children[:len(population)]
        else:
            new_population = population[:-len(children)] + children
            evaluatePopulation(new_population)
            sortPopulation(new_population)

    elif replace == "inclusion":
        evaluatePopulation(children)

        new_population = population + children
        sortPopulation(new_population)

        new_population = new_population[:-len(children)]

    return new_population


# EJECUCIÓN---------------------------------------------------------------------

popu = initPopulation(SIZE_POPULATION, NUM_MOTORS)
print("POBLACIÓN INICIAL: ", popu)
print()

evaluatePopulation(popu)
print("POBLACIÓN EVALUADA: ", popu)
print()

sortPopulation(popu)
print("POBLACIÓN ORDENADA: ", popu)
print()

best_individual = popu[0]

# Bucle (converge en un número de ciclos)
for i in range(500):
    print("Generación ", i)

    children = generateChildren(popu, NUM_CHILDS)

    popu = newPopulation(popu, children)

    if best_individual[2] > popu[0][2]:
        best_individual = popu[0]

        print("Evaluación nuevo mejor individuo: ", best_individual[2])
        print("Codificación nuevo mejor individuo: ", best_individual[0])
