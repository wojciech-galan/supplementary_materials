#! /usr/bin/python
# -*- coding: utf-8 -*-

import random


def checkPopulation(population):
    '''Ma być conajmniej jedna jedynka w wektorze cech, jeśli nie ma to jest przydzielana losowo'''
    for i in range(len(population)):
        if not any(population[i]):
            population[i][random.choice(range(len(population[i])))] = 1
    return population


eval_population = lambda pop: sum(x[0] for x in pop)


def individual_fitness(a_tuple):
    return (a_tuple[0] + a_tuple[2] - 0.25 * a_tuple[1] - 0.25 * a_tuple[3], )
