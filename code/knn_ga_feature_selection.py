#! /usr/bin/python
# -*- coding: utf-8 -*-

import copy
import argparse
import os
import time
import multiprocessing
import cPickle as pickle
from deap import base
from deap import creator
from deap import tools
from ga_stuff import checkPopulation, individual_fitness, eval_population
from lib import get_feature_indices
from ml_stuff import knn_for_given_splits_and_features

CACHE = {}


def wrapper(ml_function, ml_arguments, result_processing_function, binary_vector):
    try:
        res = CACHE[tuple(binary_vector)]
        return res
    except KeyError:
        indexes = [i for i, x in enumerate(binary_vector) if x]
        return result_processing_function(ml_function(indexes, *ml_arguments))


def print_summary(generation, population, fitnesses, evaluation_time):
    m = max([(k, v) for k, v in zip(population, fitnesses)], key=lambda x: x[1])
    different_individuals = set(tuple(x) for x in population)
    print generation, eval_population(fitnesses), m[1], sum(m[0]), len(different_individuals), len(
        CACHE), evaluation_time


def evaluate_and_measure_time(pool, func, population):
    t = time.time()
    fitnesses = pool.map(func, population)
    elapsed_time = time.time() - t
    for individual, fit in zip(population, fitnesses):
        individual.fitness.values = fit
        CACHE[tuple(individual)] = fit
    return fitnesses, elapsed_time


def ga(num_of_possible_feats, mean_initial_num_of_feats, std_initial_num_of_feats, cx, mut_pb, tournsize,
       num_of_elitte_individuals, pop_size, num_of_neighbours, cv_splits, max_turns, res_dir, cpus_num):
    # GENETIC ALGORITHMS: DECLARATIONS
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attribute", get_feature_indices, num_of_possible_feats, mean_initial_num_of_feats,
                     std_initial_num_of_feats)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", eval('tools.%s' % cx))
    toolbox.register("mutate", tools.mutFlipBit, indpb=float(mut_pb) / num_of_possible_feats)  # zmiana!
    toolbox.register("select", tools.selTournament, tournsize=tournsize)
    toolbox.register("select_best", tools.selBest, k=num_of_elitte_individuals)
    toolbox.register("evaluate", wrapper, knn_for_given_splits_and_features, [cv_splits, 0, num_of_neighbours],
                     individual_fitness)

    pop = toolbox.population(n=pop_size)  # generate initial population
    pop = [l[0] for l in pop]
    pop = checkPopulation(pop)  # check for zero-vectors

    # Evaluate the entire population
    pool = multiprocessing.Pool(cpus_num)
    parents_fitnesses, eval_time = evaluate_and_measure_time(pool, toolbox.evaluate, pop)
    generation = 0
    print_summary(generation, pop, parents_fitnesses, eval_time)

    # START EVOLUTION
    while not all(pop[0] == pop[i] for i in range(1, len(pop))) and generation < max_turns:
        parents = toolbox.select(pop, pop_size)  # Select the next generation individuals

        # elitism
        best_individuals = sorted(pop, key=lambda x: x.fitness.values, reverse=True)[:num_of_elitte_individuals]

        offspring = []
        # Apply crossover and mutation on the offspring
        for i in range(0, len(parents) - 1, 2):
            child1 = copy.deepcopy(parents[i])
            child2 = copy.deepcopy(parents[i + 1])
            if cx == 'cxUniform':
                toolbox.mate(child1, child2, 0.5)
            else:
                toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
            offspring.append(child1)
            offspring.append(child2)

        for mutant in offspring:
            toolbox.mutate(mutant)
            del mutant.fitness.values

        # Evaluate the offspring
        offspring = checkPopulation(offspring)
        fitnesses, eval_time = evaluate_and_measure_time(pool, toolbox.evaluate, offspring)

        pop = offspring
        generation += 1
        # dodanie elitarnych osobników
        for best_individual in best_individuals:
            if best_individual not in pop:
                pop.append(best_individual)
                fitnesses.append(best_individual.fitness.values)
        print_summary(generation, pop, fitnesses, eval_time)
    out_name = '_'.join([str(x) for x in mean_initial_num_of_feats, std_initial_num_of_feats, cx, mut_pb, tournsize,
       num_of_elitte_individuals, pop_size, num_of_neighbours, max_turns])
    pickle.dump(pop, open(os.path.join(res_dir, out_name), 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Genetic algorithms for feature selection")
    parser.add_argument('mean_initial_feats', action='store', type=int, help='mean initial number of features')
    parser.add_argument('std_initial_feats', action='store', type=int,
                        help='Standard deviation of initial number of features')
    parser.add_argument('cx', action='store', type=str, choices=['cxUniform', 'cxOnePoint', 'cxTwoPoint'],
                        help='crossover type')
    parser.add_argument('mut_pb', action='store', type=float, help='mutation probability')
    parser.add_argument('tournsize', action='store', type=int, help='number of individuals in tournament')
    parser.add_argument('elitte_individuals', action='store', type=int, help='number of elitte individuals')
    parser.add_argument('neighbours', action='store', type=int, help='number of neighbours for knn')
    parser.add_argument('pop_size', action='store', type=int, help='ga population size')
    parser.add_argument('max_turns', action='store', type=int, help='max number of ga generations')
    parser.add_argument('res_dir', action='store', type=str, help='directory containing results')
    parser.add_argument('--cpus', action='store', type=int, default=multiprocessing.cpu_count() - 1,
                        help='number of logical cpus used for evaluation of individuals')
    splits = pickle.load(open(os.path.join('..', 'datasets', 'splits.dump')))
    num_of_possible_features = len(splits[0][0][0])
    args = parser.parse_args()
    print "starting computuations using %d processes for evaluation" %args.cpus
    ga(num_of_possible_features, args.mean_initial_feats, args.std_initial_feats, args.cx, args.mut_pb, args.tournsize,
       args.elitte_individuals, args.pop_size, args.neighbours, splits, args.max_turns, args.res_dir, args.cpus)
