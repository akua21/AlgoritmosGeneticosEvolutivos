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

S = 10

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
def initIndividual(num_motors):
    # Vector codificación
    x = []
    # Vector varianzas
    variances = []

    for i in range(num_motors):
        x.append(np.random.uniform(-180, 180))
        variances.append(np.random.uniform(50, 100))

    individual = [x, variances]
    return individual

# Evaluar individuo
def evaluateIndividual(individual):
    evaluation = getRequest(individual[0])
    return evaluation

# MUTACIÓN
def mutation(individual, evaluation_individual, success_in_generation):
    individual_mutated = [mutation_x(individual), individual[1]]
    evaluation_mutated = evaluateIndividual(individual_mutated)

    if evaluation_individual <= evaluation_mutated:
        success_in_generation.append(0)

        if len(success_in_generation) > S:
            success_in_generation.pop(0)
            individual_selected = [individual[0], mutation_variances(individual, success_in_generation)]
        else:
            individual_selected = individual

    else:
        success_in_generation.append(1)

        if len(success_in_generation) > S:
            success_in_generation.pop(0)
            individual_selected = [individual_mutated[0], mutation_variances(individual_mutated, success_in_generation)]
        else:
            individual_selected = individual_mutated

        print("Nuevo mejor valor:", evaluation_mutated)
        print("Codificación nuevo valor:", individual_selected[0])

    return individual_selected


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
def mutation_variances(individual, success_in_generation):
    proportion_success = sum(success_in_generation) / len(success_in_generation)
    cd = 0.82
    ci = 1.18
    variances_mutated = []

    for variance in individual[1]:
        if proportion_success < 1/5:
            variances_mutated.append(cd * variance)
        elif proportion_success > 1/5:
            variances_mutated.append(ci * variance)
        elif proportion_success == 1/5:
            variances_mutated.append(variance)

    return variances_mutated



# EJECUCIÓN---------------------------------------------------------------------

indi = initIndividual(NUM_MOTORS)
print("INDIVIDUO: ", indi)
print()

evaluation_indi = evaluateIndividual(indi)
print("EVALUACIÓN INDIVIDUO: ", evaluation_indi)
print()

# Bucle (converge en un número de ciclos)
success_in_generation = []
for i in range(500):
    print("Generación ", i)

    evaluation_indi = evaluateIndividual(indi)

    indi = mutation(indi, evaluation_indi, success_in_generation)

print("EVALUACIÓN FINAL: ", evaluation_indi)
print("CODIFICACIÓN FINAL: ", indi[0])
