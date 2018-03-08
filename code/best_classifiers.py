#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import glob
import cPickle as pickle
from feature_selection_bottom_up import number_to_indices
from feature_selection_for_svc import scorer_function
from ga_stuff import individual_fitness


def get_bottom_up(directory):
    results = {}
    # for name in glob.glob(os.path.join(directory, 'results.dump*')):
    #     print name, len(pickle.load(open(name)))
    names = glob.glob(os.path.join(directory, 'results.dump*'))
    best_name = max(names, key=lambda x: int(x.rsplit('_', 1)[1]))
    best_res = max(pickle.load(open(best_name)).items(), key=lambda x: individual_fitness(x[1]))
    print individual_fitness(best_res[1])
    return number_to_indices(best_res[0])


def get_best_params_for_selectkbest(selectkbest_results_pickled):
    with open(selectkbest_results_pickled) as f:
        selectkbest_results = pickle.load(f)
    scores = selectkbest_results.best_estimator_.steps[0][1].scores_
    p_values = selectkbest_results.best_estimator_.steps[0][1].pvalues_
    indices = [x[-1] for x in sorted(zip(scores, range(len(p_values))), reverse=True)]
    return sorted(indices[:selectkbest_results.best_params_['kbest__k']]), selectkbest_results.best_params_['svc__C']


if __name__ == '__main__':
    # svm_RFE
    svc_RFE_results = pickle.load(open(os.path.join('..', 'svm_res', 'RFE.dump')))
    best_result = max(svc_RFE_results.items(), key=lambda item: item[1][0])
    svc_RFE_best_features = [i for i, b in enumerate(best_result[1][1].support_) if b]
    svc_RFE_best_C = best_result[1][1].estimator.C
    # svm_SelectKBest
    # https://stackoverflow.com/questions/44999289/print-feature-names-for-selectkbest-where-k-value-is-inside-param-grid-of-gridse
    svc_SelectKBest_best_features, svc_RFE_SelectKBest_C = get_best_params_for_selectkbest(
        os.path.join('..', 'svm_res', 'grid_search.dump'))
    raise
    svc_SelectKBest_best_features = []
    # bottom up QDA
    get_bottom_up(os.path.join('..', 'bottom_up_feature_selection_results_qda'))

    for i in range(1, 10, 2):
        print i
        get_bottom_up(os.path.join('..', 'bottom_up_feature_selection_results_knn_%d' % i))
