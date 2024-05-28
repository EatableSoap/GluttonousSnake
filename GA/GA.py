import random

import torch


def CalFitness(score, Time):
    # TODO 某一个函数，待定
    return (1+200*score+1.0/(10+Time))*10000


def SelectChild(childs, select_rate=0.1):
    sorted_child = sorted(childs, key=lambda x: x.fitness, reverse=True)
    return sorted_child[:int(len(childs) * select_rate)]


def UniformCrossover(parent1: torch.tensor, parent2, cross_rate=0.1):
    gene1 = parent1.gene.contiguous()
    gene2 = parent2.gene.contiguous()
    temp1 = gene1.numpy()
    temp2 = gene2.numpy()
    for i in range(len(temp1)):
        rate = random.uniform(0, 1)
        if rate <= cross_rate:
            temp1[i], temp2[i] = temp2[i], temp1[i]
    return gene1, gene2


def Variation(individual: torch.tensor, mutation_rate=0.01):
    gene = individual.gene.contiguous()
    temp = gene.numpy()
    for i in range(len(temp)):
        rate = random.uniform(0, 1)
        if rate <= mutation_rate:
            temp[i] += random.uniform(-1, 1)
    return gene
