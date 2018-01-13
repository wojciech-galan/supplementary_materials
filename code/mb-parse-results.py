#!/usr/bin/env python
"""
This is the shit I do in my free time, honestly...
"""

__date__ = "2017-11-17"
__author__ = "Maciej Bak"
__email__ = "wsciekly.maciek@gmail.com"
__license__ = "GPL"

# imports
import sys
import time
import logging
from argparse import ArgumentParser, RawTextHelpFormatter
import glob
import cPickle as pickle
import os
import pandas as pd
from deap import base
from deap import creator
from deap import tools

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


def parse_arguments():
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument("-v",
                    "--verbosity",
                    dest="verbosity",
                    choices=('DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL'),
                    default='ERROR',
                    help="Verbosity/Log level. Defaults to ERROR")
    parser.add_argument("-l",
                    "--logfile",
                    dest="logfile",
                    help="Store log to this file.")
    parser.add_argument("--result-dir",
                    dest="results",
                    required=True,
                    help="Path to the results directory.")
    return parser


def get_feature_indices(feats_cardinality, mean_num_of_choosen_feats, std_num_of_choosen_feats):
    '''Tworzy liste indeksow cech dla danego osobnika. Dlugosc listy ma rozklad normalny o sredniej mean i odchyleniu std'''
    length = int(random.gauss(mean_num_of_choosen_feats + 0.5, std_num_of_choosen_feats))
    curr_feat_indices = []
    possible_indices = range(feats_cardinality)
    if length >= feats_cardinality: # more features than possible
        curr_feat_indices = range(feats_cardinality)
    else:
        while len(curr_feat_indices) < length:
            possible_indices, curr_feat_indices = get_feature_index(possible_indices, curr_feat_indices)
    return creator.Individual([int(i in curr_feat_indices) for i in range(feats_cardinality)])


def main(options, logger):

    col_names = ["rep", "mean_1","std_1","cx","mut_pb","t_size","elitism","pop_size","k","max_t","val","vector"]
    df = pd.DataFrame(columns = col_names)

    for d in glob.glob(os.path.join(options.results,"*")): #over all repetitions
        repetition = d.split("/")[-1]
        for f in glob.glob(os.path.join(d,"*")): #over all final populations for all params runs

            params = f.split("/")[-1].split("_")

            toolbox = base.Toolbox()
            toolbox.register("attribute", get_feature_indices, 101, params[0], params[1])
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=1)

            with open(f, 'rb') as temp: population = pickle.load(temp)

            #get the fittest individual from a given population
            fittest = max(population, key=lambda x:x.fitness.getValues()[0])
            df.loc[len(df)] = [repetition] + params + [fittest.fitness.getValues()[0]] + [fittest] #this is ugly

    df.to_csv(os.path.join(os.getcwd(),"results.tsv"),sep="\t")


if __name__ == '__main__':
    try:
        try:
            options = parse_arguments().parse_args()
        except Exception as e:
            parser.print_help()
            sys.exit()
        #
        ########################################################################
        # Set up logging
        ########################################################################
        #
        formatter = logging.Formatter(fmt="[%(asctime)s] %(levelname)s - %(message)s",
                                      datefmt="%d-%b-%Y %H:%M:%S")
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger = logging.getLogger('uniprot_to_json')
        logger.setLevel(logging.getLevelName(options.verbosity))
        logger.addHandler(console_handler)
        if options.logfile is not None:
            logfile_handler = logging.handlers.RotatingFileHandler(options.logfile, maxBytes=50000, backupCount=2)
            logfile_handler.setFormatter(formatter)
            logger.addHandler(logfile_handler)
        #
        ########################################################################
        #
        start_time = time.time()
        start_date = time.strftime("%d-%m-%Y at %H:%M:%S")
        #
        ########################################################################
        # Run main
        ########################################################################
        #
        logger.info("Starting script")
        main(options, logger)
        #
        ########################################################################
        seconds = time.time() - start_time
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        logger.info("Successfully finished in {hours} hour(s) {minutes} minute(s) and {seconds} second(s)".format(
            hours=int(hours),
            minutes=int(minutes),
            seconds=int(seconds) if seconds > 1.0 else 1
        ))
    except KeyboardInterrupt as e:
        logger.warn("Interrupted by user after {hours} hour(s) {minutes} minute(s) and {seconds} second(s)".format(
            hours=int(hours),
            minutes=int(minutes),
            seconds=int(seconds) if seconds > 1.0 else 1
        ))
        sys.exit(-1)
    except Exception as e:
        logger.exception(str(e))
        raise e

