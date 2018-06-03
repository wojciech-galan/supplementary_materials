#! /usr/bin/python
# -*- coding: utf-8 -*-
import math
import random
import time
import cPickle as pickle
from functools import wraps
from deap import creator


def rename_host_lineage(hlin):
    if 'Archaea' in hlin or 'Bacteria' in hlin:
        return 'phage'
    elif 'Eukaryota' in hlin:
        return 'Eucaryota-infecting'


def groupping(hlineages):
    ret = []
    pairs = set([x[:2] for x in hlineages])
    for pair in pairs:
        ret.append((pair[0], pair[1], sum(z[2] for z in hlineages if z[:2] == pair)), )
    return ret


def dataset_creation(container, lineage_fragments, what):
    # lineage_fragments np. 'Microviridae'
    # what - either phage or 'Eucaryota-infecting'
    blind_set = []
    for lineage_fragment in lineage_fragments:
        blind_set.extend([virus for virus in container if
                          lineage_fragment in virus.lineage and rename_host_lineage(virus.host_lineage) == what])
    return [virus for virus in container if not virus in blind_set], blind_set


def f(x, power):
    return math.pow(math.e, (-math.pow(x, power)))


def g(power, xs, cardinality):
    return math.fabs(sum(sum(f(i, power) for i in range(x)) for x in xs) - cardinality)


def get_virus(left_viruses, lin, power):
    '''
    Gets
    '''
    get_or_not = lambda x: x >= random.random()
    viruses_for_the_lineage = [virus for virus in left_viruses if tuple(virus.lineage) == lin]
    # print lin, [(i, f(i, power), get_or_not(f(i, power))) for i, virus in enumerate(viruses_for_the_lineage)]
    choosen_viruses = [virus for i, virus in enumerate(viruses_for_the_lineage) if get_or_not(f(i, power))]
    left_viruses = [virus for virus in left_viruses if virus not in set(choosen_viruses)]
    return choosen_viruses, left_viruses


def get_feature_index(possible_indices, curr_indices):
    index = random.choice(possible_indices)
    possible_indices.remove(index)
    curr_indices.append(index)
    return possible_indices, curr_indices


def get_feature_indices(feats_cardinality, mean_num_of_choosen_feats, std_num_of_choosen_feats):
    '''Tworzy listę indeksów cech dla danego osobnika. Długość listy ma rozkład normalny o średniej mean i odchyleniu std'''
    length = int(random.gauss(mean_num_of_choosen_feats + 0.5, std_num_of_choosen_feats))
    curr_feat_indices = []
    possible_indices = range(feats_cardinality)
    if length >= feats_cardinality: # more features than possible
        curr_feat_indices = range(feats_cardinality)
    else:
        while len(curr_feat_indices) < length:
            possible_indices, curr_feat_indices = get_feature_index(possible_indices, curr_feat_indices)
    return creator.Individual([int(i in curr_feat_indices) for i in range(feats_cardinality)])


def get_best_params_for_selectkbest(selectkbest_results_pickled):
    # https://stackoverflow.com/questions/44999289/print-feature-names-for-selectkbest-where-k-value-is-inside-param-grid-of-gridse
    with open(selectkbest_results_pickled) as f:
        selectkbest_results = pickle.load(f)
    scores = selectkbest_results.best_estimator_.steps[0][1].scores_
    p_values = selectkbest_results.best_estimator_.steps[0][1].pvalues_
    indices = [x[-1] for x in sorted(zip(scores, range(len(p_values))), reverse=True)]
    return sorted(indices[:selectkbest_results.best_params_['kbest__k']]), selectkbest_results.best_params_['svc__C']


def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        print time.time() - start, "seconds"
        return result
    return wrapper