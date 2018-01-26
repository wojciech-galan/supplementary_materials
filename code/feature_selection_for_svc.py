#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import cPickle as pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_selection import *
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from ml_stuff import binary_classification_evaluation
from ml_stuff import preprocess_results_for_given_splits_and_features
from feature_selection_bottom_up import number_to_indices, indices_to_number


# class Result(object):
#     def __init__(self, C, res, feature_indices):
#         super(Result, self).__init__()
#         self.C = C
#         self.res = res
#         self.fitness = individual_fitness(res)
#         self.features = indices_to_number(feature_indices)
#
#     def __repr__(self):
#         return 'Result(%f, %s, %s)' % (self.C, str(self.res), str(number_to_indices(self.features)))


def scorer_function(y_true, y_predicted):
    evaluated = np.array([binary_classification_evaluation(y_true, y_predicted, None, 0, [0, 1])])
    return np.sum(preprocess_results_for_given_splits_and_features(evaluated))

def cv_for_given_splits_and_features(estimator, X, y, split_indices, feature_indices):
    results = []
    for learn_indices, test_indices in split_indices:
        l = X[learn_indices, feature_indices]
        t = X[test_indices, feature_indices]
        y_learn = y[learn_indices]
        y_test = y[test_indices]
        estimator.fit(l, y_learn)
        probas = estimator.predict_proba(t)
        results.append(scorer_function(y_test, probas))
    return np.mean(results)

attributes_learn = pickle.load(open(os.path.join('..', 'datasets', 'attributes_learn.dump')))
classes_learn = pickle.load(open(os.path.join('..', 'datasets', 'classes_learn.dump')))
attributes_test = pickle.load(open(os.path.join('..', 'datasets', 'attributes_test.dump')))
classes_test = pickle.load(open(os.path.join('..', 'datasets', 'classes_test.dump')))
indices = pickle.load(open(os.path.join('..', 'datasets', 'cv_indices.dump')))

scorer = make_scorer(scorer_function, needs_proba=True)
estimator = SVC(kernel='linear', probability=True)
selector = RFECV(estimator, cv=indices, scoring=scorer)
selector.fit(attributes_learn, classes_learn)
support  = selector.get_support()
print cv_for_given_splits_and_features(selector.estimator, attributes_learn, classes_learn, indices, support) #featureindices potrzebne?
print roc_auc_score(classes_test, selector.predict_proba(attributes_test)[:,1])