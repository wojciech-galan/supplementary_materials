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
from sklearn.linear_model import LogisticRegression
from feature_selection_bottom_up import number_to_indices
from feature_selection_for_svc import scorer_function
from ga_stuff import individual_fitness
from ml_stuff import generic_cv_for_given_splits_and_features
from ml_stuff import svc_for_given_splits_and_features
from ml_stuff import qda_for_given_splits_and_features
from ml_stuff import knn_for_given_splits_and_features
from ml_stuff import binary_classification_evaluation_extended
from best_features_and_params import lasso_c, lasso_features
from best_features_and_params import svc_RFE_best_C, svc_RFE_best_features
from best_features_and_params import svc_SelectKBest_best_features, svc_RFE_SelectKBest_C
from best_features_and_params import svc_biogram_best_c, svc_biogram_best_features
from best_features_and_params import svc_penalized_best_c, svc_penalized_best_features
from best_features_and_params import qda_bottomup_best_features
from best_features_and_params import feats_ga_knn
from best_features_and_params import feats_ga_knn_500
from best_features_and_params import feats_bottomup_knn
from best_features_and_params import feats_ga_qda
from best_features_and_params import feats_ga_qda_500

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
        # print method_to_compute_cv_fitness(features, cv_splits, pos_class, **params)
        self.cv_res = generic_cv_for_given_splits_and_features(classifier, features, cv_splits, pos_class)
        # assert cv_res == self.cv_res
        # print self.cv_res
        classifier.fit(x_learn[:, features], y_learn)
        probas = classifier.predict_proba(x_test[:, features])
        # print zip(y_test, probas[:, 0])
        self.blind_res = binary_classification_evaluation_extended(y_test, probas, positive_class, classifier.classes_)
        self.description = description

    def __str__(self):
        return str(self.__dict__)


if __name__ == '__main__':
    cv_splits = pickle.load(open(os.path.join('..', 'datasets', 'splits.dump')))
    attributes_learn = pickle.load(open(os.path.join('..', 'datasets', 'attributes_learn.dump')))
    classes_learn = pickle.load(open(os.path.join('..', 'datasets', 'classes_learn.dump')))
    attributes_test = pickle.load(open(os.path.join('..', 'datasets', 'attributes_test.dump')))
    classes_test = pickle.load(open(os.path.join('..', 'datasets', 'classes_test.dump')))
    positive_class = 1
    results = []
    num_of_feats = cv_splits[0][0].shape[1]
    # simple LogisticRegression
    results.append(
        Res('LogisticRegression', range(num_of_feats), {}, cv_splits,
            positive_class,
            functools.partial(generic_cv_for_given_splits_and_features, LogisticRegression()),
            attributes_learn, attributes_test, classes_learn, classes_test,
            description='simple LogisticRegression'))
    # LogisticRegression with lasso- feature selection
    results.append(
        Res('LogisticRegression', lasso_features, {'C': lasso_c}, cv_splits, positive_class,
            functools.partial(generic_cv_for_given_splits_and_features, LogisticRegression(C=lasso_c)),
            attributes_learn, attributes_test, classes_learn, classes_test,
            description='LogisticRegression with lasso- feature selection'))
    # simple svm
    results.append(
        Res('SVC', range(num_of_feats), {'kernel': 'linear', 'probability': True}, cv_splits, positive_class,
            svc_for_given_splits_and_features, attributes_learn, attributes_test, classes_learn, classes_test,
            description='simple svc'))
    # simple kNN
    results.append(Res('KNeighborsClassifier', range(num_of_feats), {},
                       cv_splits, positive_class, knn_for_given_splits_and_features,
                       attributes_learn, attributes_test,
                       classes_learn, classes_test, description='simple kNN'))

    # simple QDA
    results.append(Res('QuadraticDiscriminantAnalysis', range(num_of_feats), {},
                       cv_splits, positive_class, qda_for_given_splits_and_features, attributes_learn,
                       attributes_test,
                       classes_learn, classes_test, description='simple QDA'))
    # svm_RFE
    results.append(
        Res('SVC', svc_RFE_best_features, {'C': svc_RFE_best_C, 'kernel': 'linear', 'probability': True}, cv_splits,
            positive_class, svc_for_given_splits_and_features, attributes_learn, attributes_test, classes_learn,
            classes_test, description='svm_RFE'))

    # svm_SelectKBest
    results.append(
        Res('SVC', svc_SelectKBest_best_features, {'C': svc_RFE_SelectKBest_C, 'kernel': 'linear', 'probability': True},
            cv_splits, positive_class, svc_for_given_splits_and_features, attributes_learn, attributes_test,
            classes_learn, classes_test, description='svm_SelectKBest'))

    # svm biogram
    results.append(
        Res('SVC', svc_biogram_best_features, {'C': svc_biogram_best_c, 'kernel': 'linear', 'probability': True},
            cv_splits, positive_class, svc_for_given_splits_and_features, attributes_learn, attributes_test,
            classes_learn, classes_test, description='svm_biogram'))

    # penalized svm
    results.append(
        Res('SVC', svc_biogram_best_features, {'kernel': 'linear', 'probability': True, 'C': svc_penalized_best_c},
            cv_splits, positive_class, svc_for_given_splits_and_features, attributes_learn, attributes_test,
            classes_learn, classes_test, description='svm_penalized'))

    # bottom up QDA
    results.append(
        Res('QuadraticDiscriminantAnalysis', qda_bottomup_best_features, {},
            cv_splits, positive_class, qda_for_given_splits_and_features, attributes_learn, attributes_test,
            classes_learn, classes_test, description='bottom up QDA'))

    # bottom up kNN
    for i in range(1, 10, 2):
        results.append(Res('KNeighborsClassifier', feats_bottomup_knn[i], {'n_neighbors': i},
                           cv_splits, positive_class, knn_for_given_splits_and_features, attributes_learn,
                           attributes_test,
                           classes_learn, classes_test, description='bottom up kNN'))

    # GA knn
    names = glob.glob(os.path.join('..', 'ga_res', 'knn', '*', '*'))
    ks = [int(name.rsplit('_', 2)[-2]) for name in names]
    for k in sorted(set(ks)):
        results.append(Res('KNeighborsClassifier', feats_ga_knn[k], {'n_neighbors': k},
                           cv_splits, positive_class, knn_for_given_splits_and_features, attributes_learn,
                           attributes_test, classes_learn, classes_test, description='GA knn 100'))

    names = glob.glob(os.path.join('..', 'ga_res', 'knn_500', '*', '*'))
    ks = [int(name.rsplit('_', 2)[-2]) for name in names]
    for k in sorted(set(ks)):
        results.append(Res('KNeighborsClassifier', feats_ga_knn_500[k], {'n_neighbors': k},
                           cv_splits, positive_class, knn_for_given_splits_and_features, attributes_learn,
                           attributes_test,
                           classes_learn, classes_test, description='GA knn 500, %d neighbours' % k))

    # GA qda
    results.append(Res('QuadraticDiscriminantAnalysis', feats_ga_qda, {},
                       cv_splits, positive_class, qda_for_given_splits_and_features, attributes_learn,
                       attributes_test,
                       classes_learn, classes_test, description='GA qda 100'))
    results.append(Res('QuadraticDiscriminantAnalysis', feats_ga_qda_500, {},
                       cv_splits, positive_class, qda_for_given_splits_and_features, attributes_learn,
                       attributes_test,
                       classes_learn, classes_test, description='GA qda 500'))

    for res in results:
        print res.description, res.cv_fitness[0]
        print res.cv_res, res.blind_res
        print '--------------------------------------------------'
