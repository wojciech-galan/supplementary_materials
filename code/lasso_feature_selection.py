#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import psutil
import numpy as np
import cPickle as pickle
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from ga_stuff import individual_fitness
from ml_stuff import binary_classification_evaluation
from ml_stuff import preprocess_results_for_given_splits_and_features
from feature_selection_for_svc import cv_for_given_splits_and_features


def scorer_function(y_true, y_predicted):
    # LogisticRegressionCV transfomts class label 0 into -1, this is addresed below
    mask = (y_true == -1)
    new_y_true = np.ones(y_true.shape, dtype=y_true.dtype)
    new_y_true[mask] = 0
    evaluated = np.array([binary_classification_evaluation(new_y_true, y_predicted, None, 0, [0, 1])])
    return np.sum(preprocess_results_for_given_splits_and_features(evaluated))


if __name__ == '__main__':
    attributes_learn = pickle.load(open(os.path.join('..', 'datasets', 'attributes_learn.dump')))
    classes_learn = pickle.load(open(os.path.join('..', 'datasets', 'classes_learn.dump')))
    attributes_test = pickle.load(open(os.path.join('..', 'datasets', 'attributes_test.dump')))
    classes_test = pickle.load(open(os.path.join('..', 'datasets', 'classes_test.dump')))
    split_indices = pickle.load(open(os.path.join('..', 'datasets', 'cv_indices.dump')))
    features = pickle.load(open(os.path.join('..', 'datasets', 'features.dump')))

    scorer = make_scorer(scorer_function, needs_proba=True)
    num_of_jobs = psutil.cpu_count() - 1 or 1
    model = LogisticRegressionCV(Cs=2 ** np.linspace(-5, 5, 11), cv=split_indices, scoring=scorer, solver='liblinear',
                                 penalty='l1', n_jobs=num_of_jobs)
    model.fit(attributes_learn, classes_learn)
    feature_indices = [i for i, element in enumerate(model.coef_[0] != np.zeros(model.coef_[0].shape)) if element]
    print feature_indices
    print model.C_

    best_model = LogisticRegression(C=model.C_[0])
    result = cv_for_given_splits_and_features(best_model, attributes_learn, classes_learn, split_indices, feature_indices)

    print result, individual_fitness(result)
    pos_probas = model.predict_proba(attributes_test)[:, 1]
    auc = roc_auc_score(classes_test, pos_probas, None)
    print "Acc =", model.score(attributes_test, classes_test)
    print 'AUC =', auc
    print model.coef_
    print model.intercept_
    print np.where(model.coef_)[1]
    print np.array(features)[np.where(model.coef_)[1]]
