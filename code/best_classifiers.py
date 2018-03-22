#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import glob
import functools
import cPickle as pickle
import numpy as np
from deap import creator, base
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from feature_selection_bottom_up import number_to_indices
from feature_selection_for_svc import scorer_function
from ga_stuff import individual_fitness
from ml_stuff import generic_cv_for_given_splits_and_features
from ml_stuff import svc_for_given_splits_and_features
from ml_stuff import qda_for_given_splits_and_features
from ml_stuff import knn_for_given_splits_and_features
from ml_stuff import binary_classification_evaluation_extended

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


class Res(object):
    def __init__(self, method, features, params, cv_splits, pos_class, method_to_compute_cv_fitness,
                 x_learn, x_test, y_learn, y_test, description=None, seed=77):
        self.method = method
        self.features = features
        self.params = params
        np.random.seed(seed)
        cv_res = method_to_compute_cv_fitness(features, cv_splits, pos_class, **params)
        self.cv_fitness = individual_fitness(cv_res)
        # print cv_res
        # print self.cv_fitness
        np.random.seed(seed)
        classifier = eval('%s(**params)' % method)
        print classifier
        #print method_to_compute_cv_fitness(features, cv_splits, pos_class, **params)
        self.cv_res = generic_cv_for_given_splits_and_features(classifier, features, cv_splits, pos_class)
        # print self.cv_res
        classifier.fit(x_learn, y_learn)
        probas = classifier.predict_proba(x_test)
        #print zip(y_test, probas[:, 0])
        self.blind_res = binary_classification_evaluation_extended(y_test, probas, positive_class, classifier.classes_)
        self.description = description

    def __str__(self):
        return str(self.__dict__)


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


def get_best_params_for_selectkbest(selectkbest_results_pickled):
    # https://stackoverflow.com/questions/44999289/print-feature-names-for-selectkbest-where-k-value-is-inside-param-grid-of-gridse
    with open(selectkbest_results_pickled) as f:
        selectkbest_results = pickle.load(f)
    scores = selectkbest_results.best_estimator_.steps[0][1].scores_
    p_values = selectkbest_results.best_estimator_.steps[0][1].pvalues_
    indices = [x[-1] for x in sorted(zip(scores, range(len(p_values))), reverse=True)]
    return sorted(indices[:selectkbest_results.best_params_['kbest__k']]), selectkbest_results.best_params_['svc__C']


if __name__ == '__main__':
    cv_splits = pickle.load(open(os.path.join('..', 'datasets', 'splits.dump')))
    attributes_learn = pickle.load(open(os.path.join('..', 'datasets', 'attributes_learn.dump')))
    classes_learn = pickle.load(open(os.path.join('..', 'datasets', 'classes_learn.dump')))
    attributes_test = pickle.load(open(os.path.join('..', 'datasets', 'attributes_test.dump')))
    classes_test = pickle.load(open(os.path.join('..', 'datasets', 'classes_test.dump')))
    positive_class = 1
    results = []
    num_of_feats = cv_splits[0][0].shape[1]
    # simple svm
    results.append(
        Res('SVC', range(num_of_feats), {'kernel': 'linear', 'probability': True}, cv_splits, positive_class,
            svc_for_given_splits_and_features, attributes_learn, attributes_test, classes_learn, classes_test, description='simple svc'))
    # simple kNN
    results.append(Res('KNeighborsClassifier', range(num_of_feats), {},
                       cv_splits, positive_class, knn_for_given_splits_and_features,
                       attributes_learn, attributes_test,
                       classes_learn, classes_test, description='simple kNN'))

    # simple QDA
    results.append(Res('QuadraticDiscriminantAnalysis', range(num_of_feats), {'n_neighbors': 5},
                       cv_splits, positive_class, qda_for_given_splits_and_features, attributes_learn,
                       attributes_test,
                       classes_learn, classes_test, description='simple QDA'))
    # svm_RFE
    svc_RFE_results = pickle.load(open(os.path.join('..', 'svm_res', 'RFE.dump')))
    best_result = max(svc_RFE_results.items(), key=lambda item: item[1][0])
    svc_RFE_best_features = [i for i, b in enumerate(best_result[1][1].support_) if b]
    svc_RFE_best_C = best_result[1][1].estimator.C
    results.append(
        Res('SVC', svc_RFE_best_features, {'C': svc_RFE_best_C, 'kernel': 'linear', 'probability': True}, cv_splits,
            positive_class, svc_for_given_splits_and_features, attributes_learn, attributes_test, classes_learn,
            classes_test, description='svm_RFE'))

    # svm_SelectKBest
    svc_SelectKBest_best_features, svc_RFE_SelectKBest_C = get_best_params_for_selectkbest(
        os.path.join('..', 'svm_res', 'grid_search.dump'))
    results.append(
        Res('SVC', svc_SelectKBest_best_features, {'C': svc_RFE_SelectKBest_C, 'kernel': 'linear', 'probability': True},
            cv_splits, positive_class, svc_for_given_splits_and_features, attributes_learn, attributes_test,
            classes_learn, classes_test, description='svm_SelectKBest'))

    # svm biogram
    svc_biogram_best_features = pickle.load(open(os.path.join('..', 'svm_res', 'best_features_biogram.dump'), 'rb'))
    results.append(
        Res('SVC', svc_biogram_best_features, {'kernel': 'linear', 'probability': True},
            cv_splits, positive_class, svc_for_given_splits_and_features, attributes_learn, attributes_test,
            classes_learn, classes_test, description='svm_biogram'))

    # penalized svm
    svc_penalized_best_features = pickle.load(open(os.path.join('..', 'svm_res', 'best_features_penalizedSVM.dump'), 'rb'))
    results.append(
        Res('SVC', svc_biogram_best_features, {'kernel': 'linear', 'probability': True},
            cv_splits, positive_class, svc_for_given_splits_and_features, attributes_learn, attributes_test,
            classes_learn, classes_test, description='svm_biogram'))

    # bottom up QDA
    qda_bottomup_best_features = get_bottom_up(os.path.join('..', 'bottom_up_feature_selection_results_qda'))
    results.append(
        Res('QuadraticDiscriminantAnalysis', qda_bottomup_best_features, {},
            cv_splits, positive_class, qda_for_given_splits_and_features, attributes_learn, attributes_test,
            classes_learn, classes_test, description='bottom up QDA'))

    # bottom up kNN
    for i in range(1, 10, 2):
        # print i
        feats = get_bottom_up(os.path.join('..', 'bottom_up_feature_selection_results_knn_%d' % i))
        results.append(Res('KNeighborsClassifier', feats, {'n_neighbors': i},
                           cv_splits, positive_class, knn_for_given_splits_and_features, attributes_learn,
                           attributes_test,
                           classes_learn, classes_test, description='bottom up kNN'))

    names = glob.glob(os.path.join('..', 'ga_res', 'knn', '*', '*'))
    ks = [int(name.rsplit('_', 2)[-2]) for name in names]
    for k in sorted(set(ks)):
        print k
        feats = get_ga(os.path.join('..', 'ga_res', 'knn'), k)
        results.append(Res('KNeighborsClassifier', feats, {'n_neighbors': k},
                           cv_splits, positive_class, knn_for_given_splits_and_features, attributes_learn,
                           attributes_test, classes_learn, classes_test, description='GA knn 100'))

    names = glob.glob(os.path.join('..', 'ga_res', 'knn_500', '*', '*'))
    ks = [int(name.rsplit('_', 2)[-2]) for name in names]
    for k in sorted(set(ks)):
        print k
        feats = get_ga(os.path.join('..', 'ga_res', 'knn_500'), k)
        results.append(Res('KNeighborsClassifier', feats, {'n_neighbors': k},
                           cv_splits, positive_class, knn_for_given_splits_and_features, attributes_learn,
                           attributes_test,
                           classes_learn, classes_test, description='GA knn 500'))

    feats = get_ga(os.path.join('..', 'ga_res', 'qda'))
    results.append(Res('QuadraticDiscriminantAnalysis', feats, {},
                       cv_splits, positive_class, qda_for_given_splits_and_features, attributes_learn,
                       attributes_test,
                       classes_learn, classes_test, description='GA qda 100'))
    feats500 = get_ga(os.path.join('..', 'ga_res', 'qda_500'))
    results.append(Res('QuadraticDiscriminantAnalysis', feats500, {},
                       cv_splits, positive_class, qda_for_given_splits_and_features, attributes_learn,
                       attributes_test,
                       classes_learn, classes_test, description='GA qda 500'))

    print feats
    print feats500
    for res in results:
        print res.description, res.cv_fitness
        print res.cv_res, res.blind_res
        print '--------------------------------------------------'
