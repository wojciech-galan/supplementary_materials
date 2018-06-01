#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import glob
import cPickle as pickle
from deap import creator, base
from ga_stuff import individual_fitness
from feature_selection_bottom_up import number_to_indices

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


def get_best_params_for_selectkbest(selectkbest_results_pickled):
    # https://stackoverflow.com/questions/44999289/print-feature-names-for-selectkbest-where-k-value-is-inside-param-grid-of-gridse
    with open(selectkbest_results_pickled) as f:
        selectkbest_results = pickle.load(f)
    scores = selectkbest_results.best_estimator_.steps[0][1].scores_
    p_values = selectkbest_results.best_estimator_.steps[0][1].pvalues_
    indices = [x[-1] for x in sorted(zip(scores, range(len(p_values))), reverse=True)]
    return sorted(indices[:selectkbest_results.best_params_['kbest__k']]), selectkbest_results.best_params_['svc__C']


def get_bottom_up(directory):
    # for name in glob.glob(os.path.join(directory, 'results.dump*')):
    #     print name, len(pickle.load(open(name)))
    names = glob.glob(os.path.join(directory, 'results.dump*'))
    best_name = max(names, key=lambda x: int(x.rsplit('_', 1)[1]))
    best_res = max(pickle.load(open(best_name)).items(), key=lambda x: individual_fitness(x[1]))
    # print individual_fitness(best_res[1])
    return number_to_indices(best_res[0])


def get_ga(directory, neighbours=None):
    names = glob.glob(os.path.join(directory, '*', '*'))
    if neighbours:
        neighbours = str(neighbours)
        names = [name for name in names if name.rsplit('_', 2)[-2] == neighbours]
    res = max([max(pickle.load(open(name)), key=lambda x: x.fitness) for name in names], key=lambda x: x.fitness)
    return [x for x in range(len(res)) if res[x]]


# LogisticRegression with lasso- feature selection
lasso_features = pickle.load(open(os.path.join('..', 'lr_res', 'best_features_LogisticRegression.dump')))
lasso_c = pickle.load(open(os.path.join('..', 'lr_res', 'best_C_LogisticRegression.dump')))[0]

# svm_RFE
svc_RFE_results = pickle.load(open(os.path.join('..', 'svm_res', 'RFE.dump')))
best_result = max(svc_RFE_results.items(), key=lambda item: item[1][0])
svc_RFE_best_features = [i for i, b in enumerate(best_result[1][1].support_) if b]
svc_RFE_best_C = best_result[1][1].estimator.C

# svm_SelectKBest
svc_SelectKBest_best_features, svc_RFE_SelectKBest_C = get_best_params_for_selectkbest(
    os.path.join('..', 'svm_res', 'grid_search.dump'))

# svm biogram
svc_biogram_best_features = pickle.load(open(os.path.join('..', 'svm_res', 'best_features_biogram.dump'), 'rb'))
svc_biogram_best_c = pickle.load(open(os.path.join('..', 'svm_res', 'best_C_biogram.dump'), 'rb'))

# penalized svm
svc_penalized_best_features = pickle.load(
    open(os.path.join('..', 'svm_res', 'best_features_penalizedSVM.dump'), 'rb'))
svc_penalized_best_c = pickle.load(open(os.path.join('..', 'svm_res', 'best_C_penalizedSVM.dump'), 'rb'))

# bottom up QDA
qda_bottomup_best_features = get_bottom_up(os.path.join('..', 'bottom_up_feature_selection_results_qda'))

# bottom up kNN
feats_bottomup_knn = {}
for i in range(1, 10, 2):
    feats_bottomup_knn[i] = get_bottom_up(os.path.join('..', 'bottom_up_feature_selection_results_knn_%d' % i))

# GA knn
names = glob.glob(os.path.join('..', 'ga_res', 'knn', '*', '*'))
ks = [int(name.rsplit('_', 2)[-2]) for name in names]
feats_ga_knn = {}
for k in sorted(set(ks)):
    feats_ga_knn[k] = get_ga(os.path.join('..', 'ga_res', 'knn'), k)
names = glob.glob(os.path.join('..', 'ga_res', 'knn_500', '*', '*'))
ks = [int(name.rsplit('_', 2)[-2]) for name in names]
feats_ga_knn_500 = {}
for k in sorted(set(ks)):
    feats_ga_knn_500[k] = get_ga(os.path.join('..', 'ga_res', 'knn_500'), k)

# GA qda
feats_ga_qda = get_ga(os.path.join('..', 'ga_res', 'qda'))
feats_ga_qda_500 = get_ga(os.path.join('..', 'ga_res', 'qda_500'))



