# Práctica 2 AGE - Calibración de motores automática mediante estrategias evolutivas

# Imports
import requests
import numpy as np
import random
import math
import matplotlib.pyplot as plt

# Constantes
# Website para los 4 rotores
WEBSITE_4 = "http://163.117.164.219/age/robot4?c1=%s&c2=%s&c3=%s&c4=%s"

# Website para los 10 rotores
WEBSITE_10 = "http://163.117.164.219/age/robot4?c1=%s&c2=%s&c3=%s&c4=%s&c5=%s&c6=%s&c7=%s&c8=%s&c9=%s&c10=%s"

NUM_MOTORS = 4
S = 10

COUNTER_EVALS = 0
NUM_TIMES_EXPERIMENT = 5

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
        variances.append(np.random.uniform(200, 400))

    individual = [x, variances]
    return individual

# Evaluar individuo
def evaluateIndividual(individual):
    global COUNTER_EVALS
    COUNTER_EVALS += 1
    evaluation = getRequest(individual[0])
    return evaluation

# MUTACIÓN
def mutation(individual, evaluation_individual, success_in_generation):
    individual_mutated = [mutation_x(individual), individual[1]]
    evaluation_mutated = evaluateIndividual(individual_mutated)

    if evaluation_individual < evaluation_mutated:
        success_in_generation.append(0)

        if len(success_in_generation) > S:
            success_in_generation.pop(0)
            individual_selected = [individual[0], mutation_variances(individual, success_in_generation)]
        else:
            individual_selected = individual
        individual_selected_eval = evaluation_individual

    else:
        success_in_generation.append(1)

        if len(success_in_generation) > S:
            success_in_generation.pop(0)
            individual_selected = [individual_mutated[0], mutation_variances(individual_mutated, success_in_generation)]
        else:
            individual_selected = individual_mutated
        individual_selected_eval = evaluation_mutated

        print("Nuevo mejor valor:", evaluation_mutated)
        print("Codificación nuevo valor:", individual_selected[0])

    return individual_selected, individual_selected_eval


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

# BUCLE para realizar varias veces el EXPERIMENTO-------------------------------
list_best_individual_eval_exp = []

best_of_all = [0, math.inf, 0]

for num in range(NUM_TIMES_EXPERIMENT):
    print("EXP ", num, "-------------------------")
    COUNTER_EVALS = 0

    indi = initIndividual(NUM_MOTORS)
    print("INDIVIDUO: ", indi)
    print()

    evaluation_indi = evaluateIndividual(indi)
    print("EVALUACIÓN INDIVIDUO: ", evaluation_indi)
    print()

    counter_evals_best = COUNTER_EVALS
    best_individual = indi
    best_individual_eval = evaluation_indi
    list_best_individual_eval = [best_individual_eval]

    # Bucle (converge en un número de ciclos)
    success_in_generation = []
    for i in range(50):
        print("Generación ", i)

        indi, eval = mutation(indi, evaluation_indi, success_in_generation)

        evaluation_indi = eval


        if best_individual_eval > evaluation_indi:
            best_individual = indi
            best_individual_eval = evaluation_indi

            counter_evals_best = COUNTER_EVALS

            print("Codificación nuevo mejor individuo: ", best_individual)
            print("Evaluación nuevo mejor individuo: ", best_individual_eval)
            print("Número de evaluaciones: ", COUNTER_EVALS)

        list_best_individual_eval.append(evaluation_indi)

    if best_individual_eval < best_of_all[1]:
        best_of_all = [best_individual, best_individual_eval, counter_evals_best]
    list_best_individual_eval_exp.append(list_best_individual_eval)
    print()



print("Codificación mejor individuo: ", best_of_all[0])
print("Evaluación mejor individuo: ", best_of_all[1])
print("Número de evaluaciones mejor individuo: ", best_of_all[2])
print("Media de evaluaciones de los mejores individuos: ", sum(list_b[-1] for list_b in list_best_individual_eval_exp) / NUM_TIMES_EXPERIMENT)

plt.plot(list_best_individual_eval_exp[0], label='Ex. 1º vez')
plt.plot(list_best_individual_eval_exp[1], label='Ex. 2º vez')
plt.plot(list_best_individual_eval_exp[2], label='Ex. 3º vez')
plt.plot(list_best_individual_eval_exp[3], label='Ex. 4º vez')
plt.plot(list_best_individual_eval_exp[4], label='Ex. 5º vez')

plt.xlabel('Generaciones')
plt.ylabel('Valor de fitness')
plt.legend()
plt.show()
