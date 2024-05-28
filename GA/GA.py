import random
import numpy.random
import torch


def SelectChild(childs, select_rate=0.1):
    sorted_child = sorted(childs, key=lambda x: x.fitness, reverse=True)
    del sorted_child[int(len(childs) * select_rate):]
    return sorted_child[:int(len(childs) * select_rate)]


def UniformCrossover(parent1, parent2, f_max, f_avg, P_max=0.5, P_min=0.2):
    # 自适应交叉概率
    if max(parent1.fitness, parent2.fitness) < f_avg:
        cross_rate = P_max
    else:
        cross_rate = P_max - (P_max - P_min) * (max(parent1.fitness, parent2.fitness) - f_avg) / (f_max - f_avg + 1)
    gene1 = parent1.gene.contiguous()
    gene2 = parent2.gene.contiguous()
    temp1 = gene1.numpy()
    temp2 = gene2.numpy()
    for i in range(len(temp1)):
        rate = random.uniform(0, 1)
        if rate <= cross_rate:
            temp1[i], temp2[i] = temp2[i], temp1[i]
    return gene1, gene2


def Variation(individual, total_N, cur_n):
    mutation_rate = 1.0 - random.random() ** (1.0 - cur_n / total_N)
    gene = individual.gene.numpy()
    for i in range(len(gene)):
        rate = random.uniform(0, 1)
        if rate <= mutation_rate:
            gene[i] = numpy.random.standard_cauchy() * 2 - 1
    return individual
