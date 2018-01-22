#!/usr/bin/env python
# -*- coding: utf-8 -*-
# imports
import sys
import time
from argparse import ArgumentParser, RawTextHelpFormatter
import glob
import cPickle as pickle
import os
import pandas as pd
from deap import base
from deap import creator
from deap import tools
from lib import get_feature_indices

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)




if __name__ == '__main__':
    col_names = ["rep", "mean_1", "std_1", "cx", "mut_pb", "t_size", "elitism", "pop_size", "max_t", "val",
                 "vector"]
    df = pd.DataFrame(columns=col_names)

    for d in glob.glob(os.path.join('..', 'ga_res', 'qda', "*")):  # over all repetitions
        repetition = d.split("/")[-1]
        for f in glob.glob(os.path.join(d, "*")):  # over all final populations for all params runs

            params = f.split("/")[-1].split("_")

            toolbox = base.Toolbox()
            toolbox.register("attribute", get_feature_indices, 101, params[0], params[1])
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=1)

            with open(f, 'rb') as temp: population = pickle.load(temp)

            # get the fittest individual from a given population
            fittest = max(population, key=lambda x: x.fitness.getValues()[0])
            df.loc[len(df)] = [repetition] + params + [fittest.fitness.getValues()[0]] + [fittest]  # this is ugly

    df.to_csv(os.path.join(os.getcwd(), "results.tsv"), sep="\t")

