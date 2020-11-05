# Práctica 1 AGE - Optimización de Sensores en Smart Cities

# Imports
import requests
import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt

# Constantes
WEBSITE = "http://memento.evannai.inf.uc3m.es/age/test?c="
WEBSITE_ALFA = "http://163.117.164.219/age/alfa?c="
WEBSITE_MADRIZ = "http://163.117.164.219/age/madriz?c="

NUM_SENSORS = 4 * 16 # tamaño cromosoma, 4 estaciones con 16 sensores cada una

SIZE_POPULATION = 100 # tamaño población, número de individuos de una población
CYCLES = 5 # número máximo de ciclos(generaciones) sin mejora que permitimos
NUM_TIMES_EXPERIMENT = 8 # número de veces que se va a ejecutar el experimento



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
        evaluation = getRequest(''.join(map(str, chromosome)))
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

        size_popu = size_popu - size_elite

        sorted_population_eval = sorted(population_evaluation)

        for i in range(size_elite):
            selected_population.append(population[population_evaluation.index(sorted_population_eval[i])])


    # print("selected_population antes del torneo: ", selected_population)
    for i in range(size_popu):
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

    divide_number = random.randint(1, len(parent_1) - 2)

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

    divide_number_1 = random.randint(1, len(parent_1) - 2)
    divide_number_2 = random.randint(1, len(parent_1) - 2)

    divide_number_min = min(divide_number_1, divide_number_2)
    divide_number_max = max(divide_number_1, divide_number_2)

    for i in range(len(parent_1)):
        if i < divide_number_min:
            child_1.append(parent_1[i])
            child_2.append(parent_2[i])

        elif i >= divide_number_min and i < divide_number_max:
            child_1.append(parent_2[i])
            child_2.append(parent_1[i])

        elif i >= divide_number_max:
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

            childs.extend([child_1, child_2])

        elif simple:
            child_1, child_2 = crossingSimple(parent_1, parent_2)

            childs.extend([child_1, child_2])

        elif two_points:
            child_1, child_2 = crossing2Points(parent_1, parent_2)

            childs.extend([child_1, child_2])

    return childs


# Operador de mutación
# Mutación de todos con poca probabilidad (1/NUM_SENSORS) para cada gen
def mutation(population, prob):
    probability = 1/prob
    mutated_population = []

    for i in range(len(population)):
        mutated_population.append([])
        for j in range(len(population[i])):
            if random.random() <= probability:
                mutated_population[i].append(1 - population[i][j])
            else:
                mutated_population[i].append(population[i][j])

    return mutated_population


# Mutación de los peores con mayor probabilidad (1/NUM_SENSORS)*2 , los mejores (20%) no mutan
def mutation_worst_ones(population, population_evaluation):
    probability = (1/NUM_SENSORS)*2
    mutated_population = []
    size_popu = len(population)

    # Añado a los mejores a la nueva población y los elimino de la actual para que luego no entren en la mutación
    sorted_population_eval = sorted(population_evaluation)
    size_best = int(len(population) * 0.2)

    last_best_one_eval = sorted_population_eval[size_best - 1]

    for i in range(size_popu):
        if popu_evaluation[i] > last_best_one_eval:
            mutated_population.append([])
            for j in range(len(population[i])):
                if random.random() <= probability:
                    mutated_population[i].append(1 - population[i][j])
                else:
                    mutated_population[i].append(population[i][j])
        else:
            mutated_population.append(population[i])

    return mutated_population




def exploration_exploitation(population):
    i = 0   # Contador generaciones

    # Lista con el mejor individuo de cada generación
    best_of_generation = []
    # Lista con la evaluación(fitness) del mejor individuo de cada generación
    best_of_generation_eval = []
    # Contador del número de evaluaciones
    counter_eval = 0

    # Bucle de EXPLORACIÓN (se realiza mutación con alta probabilidad 1/16)
    for cycle in range(15):
        # EVALUACIÓN
        popu_evaluation = evaluatePopulation(population)
        counter_eval += 1 * len(population)

        best_of_generation_eval.append(min(popu_evaluation))

        best_of_generation.append(population[popu_evaluation.index(min(popu_evaluation))])
        print("Generation ", i)

        # SELECCIÓN POR TORNEO
        selected_popu = tournamentSelection(population, popu_evaluation)

        # CRUZAMIENTO POR CRUCE UNIFORME
        childs = crossing(selected_popu, uniform=True)

        # MUTACIÓN DE TODOS CON MUCHA PROBABILIDAD
        mutated_popu = mutation(childs, 16)

        population = mutated_popu

        i += 1


    # Bucle de EXPLOTACIÓN (se realiza mutación con baja probabilidad 1/NUM_SENSORS)
    for cycle in range(10):
        # EVALUACIÓN
        popu_evaluation = evaluatePopulation(population)
        counter_eval += 1 * len(population)

        best_of_generation_eval.append(min(popu_evaluation))

        best_of_generation.append(population[popu_evaluation.index(min(popu_evaluation))])
        print("Generation ", i)

        # SELECCIÓN POR TORNEO CON ELITISMO
        selected_popu = tournamentSelection(population, popu_evaluation, elitism=True, size_elite=5)

        # CRUZAMIENTO POR CRUCE SIMPLE
        childs = crossing(selected_popu, simple=True)

        # MUTACIÓN DE TODOS CON POCA PROBABILIDAD
        mutated_popu = mutation(childs, NUM_SENSORS)

        population = mutated_popu

        i += 1

    return counter_eval, best_of_generation, best_of_generation_eval



# MAIN--------------------------------------------------------------------------
execution_time = 0
counter_eval_total = 0

best_of_experiment = []
best_of_experiment_eval = []
list_best_of_generation_eval = []

# BUCLE para realizar varias veces el EXPERIMENTO-------------------------------
for num in range(NUM_TIMES_EXPERIMENT):
    start_time = time.time()

    # INICIALIZAR POBLACIÓN
    popu =initPopulation(SIZE_POPULATION, NUM_SENSORS)
    # print("Initial population: ", popu)

    # Lista con el mejor individuo de cada generación
    best_of_generation = []
    # Lista con la evaluación(fitness) del mejor individuo de cada generación
    best_of_generation_eval = []

    # Contador del número de evaluaciones
    counter_eval = 0



    # Si se utiliza la función exploration_exploitation hay que comentar todo lo de abajo (BUCLE PRINCIPAL)
    # counter_eval, best_of_generation, best_of_generation_eval = exploration_exploitation(popu)



    # """
    # BUCLE PRINCIPAL-----------------------------------------------------------
    cycles_no_better = 0    # Contador de ciclos sin mejora
    min_popu_eval = math.inf   # Guarda la evaluación mínima para saber si la nueva evaluación es mejor o peor

    i = 0   # Contador generaciones

    while cycles_no_better < CYCLES:

        # EVALUACIÓN
        print("INICIO EVAL")
        popu_evaluation = evaluatePopulation(popu)
        print("FIN EVAL")
        counter_eval += 1 * len(popu)

        # Se aumenta el contador de la condición de parada si la evalución mínima es peor
        if min(popu_evaluation) < min_popu_eval:
            min_popu_eval = min(popu_evaluation)
            cycles_no_better = 0
        else:
            cycles_no_better += 1

        print("c, min eval ", cycles_no_better, min(popu_evaluation))


        best_of_generation_eval.append(min(popu_evaluation))

        best_of_generation.append(popu[popu_evaluation.index(min(popu_evaluation))])
        print("Generation ", i)
        # print("Evaluation initial population: ", popu_evaluation)


        # SELECCIÓN POR TORNEO
        # selected_popu = tournamentSelection(popu, popu_evaluation)
        # print("Selected population from TOURNAMENT: ", selected_popu)
        # print("Evaluation selected population: ", evaluatePopulation(selected_popu))

        # SELECCIÓN POR TORNEO CON ELITISMO
        selected_popu = tournamentSelection(popu, popu_evaluation, elitism=True, size_elite=5)


        # CRUZAMIENTO POR CRUCE UNIFORME
        childs = crossing(selected_popu, uniform=True)
        # print("Childs of selected population: ", childs)

        # CRUZAMIENTO POR CRUCE SIMPLE
        # childs = crossing(selected_popu, simple=True)

        # CRUZAMIENTO POR CRUCE DOS PUNTOS
        # childs = crossing(selected_popu, two_points=True)


        # MUTACIÓN DE TODOS CON POCA PROBABILIDAD
        mutated_popu = mutation(childs, NUM_SENSORS)

        # MUTACIÓN DE LOS PEORES
        # childs_evaluation = evaluatePopulation(childs)
        # counter_eval += 1 * len(childs)
        # mutated_popu = mutation_worst_ones(childs, childs_evaluation)

        # print("Childs mutated: ", mutated_popu)

        # for i in range(len(childs)):
        #     print(childs[i] == mutated_popu[i])

        popu = mutated_popu

        i += 1

    # """

    print()
    current_time = time.time()
    # print("Tiempo ejecución: ", current_time - start_time, "segundos")
    execution_time += current_time - start_time
    print()
    # print("Número de evaluaciones: ", counter_eval)
    counter_eval_total += counter_eval
    print()
    # print("Evaluaciones de los mejores individuos de cada generación: ", best_of_generation_eval)
    list_best_of_generation_eval.append(best_of_generation_eval)
    print()
    # print("Eval. del mejor individuo total: ", min(best_of_generation_eval))
    best_of_experiment_eval.append(min(best_of_generation_eval))
    # print("Mejores individuos de cada generación: ", best_of_generation)
    print()
    # print("Mejor individuo: ", best_of_generation[best_of_generation_eval.index(min(best_of_generation_eval))])
    best_of_experiment.append(best_of_generation[best_of_generation_eval.index(min(best_of_generation_eval))])


print()
print("Tiempo de ejecución de media: ", execution_time / NUM_TIMES_EXPERIMENT)
print()
print("Número de evaluaciones de media: ", counter_eval_total / NUM_TIMES_EXPERIMENT)
print()
print("Evaluaciones de los mejores individuos de cada vez que se ha ejecutado el experimento: ", best_of_experiment_eval)
print()
print("Eval. del mejor individuo de todas las veces que se ha ejecutado el experimento: ", min(best_of_experiment_eval))
print()
print("Mejor individuo del experimento: ", best_of_generation[best_of_generation_eval.index(min(best_of_generation_eval))])


# plt.plot(list_best_of_generation_eval[0], list_best_of_generation_eval[1])
plt.plot(list_best_of_generation_eval[0], label='Ex. 1º vez')
plt.plot(list_best_of_generation_eval[1], label='Ex. 2º vez')
plt.plot(list_best_of_generation_eval[2], label='Ex. 3º vez')
plt.plot(list_best_of_generation_eval[3], label='Ex. 4º vez')
plt.plot(list_best_of_generation_eval[4], label='Ex. 5º vez')
plt.plot(list_best_of_generation_eval[5], label='Ex. 6º vez')
plt.plot(list_best_of_generation_eval[6], label='Ex. 7º vez')
plt.plot(list_best_of_generation_eval[7], label='Ex. 8º vez')

plt.xlabel('Generaciones')
plt.ylabel('Valor de fitness')
plt.legend()
plt.show()
