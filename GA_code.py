# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 17:20:13 2023

@author: ASUS
"""
import time
import numpy as np
from numpy import random
#import random
# matrix = np.loadtxt(r"C:\Users\Boroush\Documents\University courses\Term 6\Bio-computing\project\geekfile.txt")
# matrix=matrix.astype(int)


def mat_open(location):
    matrix = np.loadtxt(location)
    matrix = matrix.astype(int)
    (row, coloumn) = (matrix.shape)
    return (matrix,row,coloumn)


#number of 0s for each coloumn is cost, this way the coloumn that covers  most of the rows has a lower cost.
def costing(num_col,matrix):
    cost_list=[]
    for i in range(coloumn):
        c = 0
        for j in matrix[:, i]:
            if j == 0:
                c+= 1
        cost_list.append(c)
    cost = np.array(cost_list)
    return cost


def population_generation(num_col):
    chromosome = []
    chromosomes = []
    while len(chromosome)<100:
        probability=random.random()
        chr = random.choice([1, 0], p =[probability,1-probability], size=( num_col))
        b=True
        for a in chromosome:
            if (a - chr).any() == False:
                b=False
        if b==False:
            continue
        else:
            print(chr)
            chromosome.append(chr)
            chromosomes.append([0, chr])

    return chromosomes


# score hasel as f.f har chi kamtar behtare
def fitness(coloumn, population_un, costs):
    for chr in population_un:
        s = 0
        for j in range(coloumn):
            s = costs[j] * chr[1][j] + s
        chr[0]=s
   # print(population_un)
    return


# p_tournoment = pulation.copy() use copy of the population for thus part
def parent_selection(p_tournoment):
    #binary tournoment
    for x in zip(p_tournoment[::2], p_tournoment[1::2]):
        if x[0][0] > x[1][0]:
            p_tournoment.remove(x[0])
        else:
            p_tournoment.remove(x[1])
    return p_tournoment


def cross_over(p_population,num_coloumn):
    childeren = []
    i = 0
    parent1 = p_population[i]
    while True:
        child_list = []
        parent2 = p_population[i + 1]
        parray1 = list(parent1[1])
        parray2 = list(parent2[1])
        for j in range(num_coloumn):
            if parray1[j] == parray2[j]:
                child_list.append(parent1[1][j])
            else:
                probability = random.random()
                fp1 = 1 - (parent1[0] / (parent2[0] + parent1[0]))
                fp2 = 1 - fp1
                if probability > fp1:
                    child_list.append(parent2[1][j])
                else:
                    child_list.append(parent1[1][j])
        c_n = np.array(child_list)
        childeren.append([0, c_n])
        i = i + 2
        if i < len(p_population):
            parent1 = p_population[i]
        else:
            break
    return childeren


# main
(matrix, row, coloumn) = mat_open(r"C:\Users\Boroush\Documents\University courses\Term 6\Bio-computing\project\4.txt")
# population =[[score,chromosome], [[4, array([1, 0, 1, 1, 1])],,,,,,,]
#population = population_generation(coloumn)
#p_tournoment = p.copy()
#for i in range(len(population)):
    #population[i][0]=fitness_function(population[i][1])
    
    
####################################################################


import math

# Mutation function
def mutate(child, t, m_f, m_c, m_g):
  num_mutations = math.ceil(1 + math.exp(-4 * m_g * (t - m_c) / m_f))
  
  for i in range(num_mutations):
    index = random.randint(0, len(child) - 1)
    child[index] = random.randint(0, 1)

  return child

####################################################################
# Encode solution as string
def encode_solution(solution):
  encoding = ''
  for gene in solution:
    encoding += str(gene)
  return encoding

# Check if child is duplicate
def is_duplicate(population, child):

  # Encode population
  population_set = set()
  for solution in population:
    encoding = encode_solution(solution) 
    population_set.add(encoding)

  # Encode child
  child_encoding = encode_solution(child)

  # Check if child encoding exists
  return child_encoding in population_set

# Steady state replacement  
def steady_state_replace(population, child):

  # Check for duplicate
  is_duplicate = False
  for individual in population:
    if (individual[1] == child[1]).all():
      is_duplicate = True
      break 

  if is_duplicate:
    return False
  
  # Select random individual with above average fitness
  fitnesses = [x[0] for x in population]
  avg_fitness = sum(fitnesses) / len(population)
  
  replace_idx = -1
  for i, individual in enumerate(population):
    if individual[0] >= avg_fitness:
      if replace_idx == -1:
        replace_idx = i
      elif random.random() < 0.5:
        replace_idx = i

  # Replace selected individual with child
  population[replace_idx] = child
  
  return True

###################################################################

# Main GA loop  
def ga():

  population = population_generation(coloumn)
  best = None
  
  MAX_GENS = 100
  
  for g in range(MAX_GENS):
  
    p1, p2 = parent_selection(population)  
    child = cross_over(p1, p2)
    child = mutate(child)  
    #child = make_feasible(child)
    
    if steady_state_replace(population, child):
      # Update best
      if fitness(child) < fitness(best):
        best = child

  return best

# Run multiple trials  
TRIALS = 10

total_time = 0
for i in range(TRIALS):
  start = time.time()
  best = ga()
  end = time.time()
  total_time += (end - start)

avg_time = total_time / TRIALS

print("Avg time:", avg_time)
print("Best solution:", best)





