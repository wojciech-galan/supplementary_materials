#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import warnings
import cPickle as pickle
import numpy as np
from rpy2.robjects import numpy2ri
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from rpy2.robjects import r
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import binarize
from ml_stuff import binary_classification_evaluation
from ml_stuff import preprocess_results_for_given_splits_and_features


def scorer_function(y_true, y_predicted):
    evaluated = np.array([binary_classification_evaluation(y_true, y_predicted, None, 0, [0, 1])])
    return np.sum(preprocess_results_for_given_splits_and_features(evaluated))


def cv_for_given_splits_and_features(estimator, X, y, split_indices, feature_indices):
    results = []
    for learn_indices, test_indices in split_indices:
        l = X[learn_indices, :][:, feature_indices]
        t = X[test_indices, :][:, feature_indices]
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

res_dir = os.path.join('..', 'svm_res')
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

# scorer = make_scorer(scorer_function, needs_proba=True)
# results = {}
# for C in 2 ** np.linspace(-5, 15, 21):
#     estimator = SVC(C=C, kernel='linear', probability=True)
#     selector = RFECV(estimator, cv=indices, scoring=scorer)
#     selector.fit(attributes_learn, classes_learn)
#     support = selector.get_support(indices=True)
#     results[C] = (
#     cv_for_given_splits_and_features(selector.estimator, attributes_learn, classes_learn, indices, support), selector)
#
# pickle.dump(results, open(os.path.join(results, 'RFE.dump')))
# best_result = max(results.items(), key=lambda item: item[1][0])
# best_selector = best_result[1][1]
# print "RFE AUC:", roc_auc_score(classes_test, best_selector.predict_proba(attributes_test)[:, 1])

# TODo zmienić w publikacji na linear kernel

# TODO zmienić w publikacji chi2 na f_classif

# kbest = SelectKBest(f_classif)  # TODO zmienić funkcję!
# pipeline = Pipeline([('kbest', kbest), ('svc', SVC(kernel='linear'))])  # TODO probas!
# grid_search = GridSearchCV(pipeline,
#                            {'kbest__k': range(attributes_learn.shape[1] - 1, 0, -1),
#                             'svc__C': 2 ** np.linspace(-5, 15, 21)},
#                            scoring=None, cv=indices, n_jobs=2)  # TODO zmienic scoring
# grid_search.fit(attributes_learn, classes_learn)

binarized_attributes_learn = binarize(attributes_learn)
binarized_attributes_test = binarize(attributes_test)

with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	r.library( 'penalizedSVM' )
