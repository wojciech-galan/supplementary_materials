#! /usr/bin/python
# -*- coding: utf-8 -*-

import cPickle as pickle
import os
import psutil
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.svm import SVC


def transform_classes(classes, chosen_class):
    '''
    Transforms multiclass classification problem to a binary one
    :param classes:
    :param chosen_class:
    :return:
    '''
    classes = np.array(classes)
    classes[classes != chosen_class] = np.inf
    classes[classes == chosen_class] = 1
    classes[classes == np.inf] = 0
    return classes


def select_features(attributes, coefficient_of_features):
    feature_indices = np.where(coefficient_of_features != np.zeros(coefficient_of_features.shape))[0]
    return attributes[:, feature_indices]


def roc_auc_scorer(y_true, y_pred):
    return roc_auc_score(y_true, y_pred[:, 1])


if __name__ == '__main__':
    # loading data
    attributes_learn = pickle.load(open(os.path.join('..', 'datasets', 'extension_attributes_learn.dump')))
    attributes_test = pickle.load(open(os.path.join('..', 'datasets', 'extension_attributes_test.dump')))
    classes_learn = pickle.load(open(os.path.join('..', 'datasets', 'extension_classes_learn.dump')))
    classes_test = pickle.load(open(os.path.join('..', 'datasets', 'extension_classes_test.dump')))
    indices = pickle.load(open(os.path.join('..', 'datasets', 'extension_cv_indices.dump')))

    num_of_jobs = max(psutil.cpu_count() - 2, 1)
    random_state = 666
    c_range = 2 ** np.linspace(-5, 5, 11)
    lr = LogisticRegression(C=np.iinfo(np.int32).max, random_state=random_state)

    for class_num in range(4):
        print 'LR with Lasso'
        transformed_classes_learn = transform_classes(classes_learn, class_num)
        transformed_classes_test = transform_classes(classes_test, class_num)
        model = LogisticRegressionCV(Cs=c_range, cv=indices, scoring=make_scorer(roc_auc_scorer, needs_proba=True),
                                     solver='liblinear', random_state=random_state,
                                     penalty='l1', n_jobs=num_of_jobs)
        model.fit(attributes_learn, transformed_classes_learn)
        print 'C =', model.C_
        selected_attributes_train = select_features(attributes_learn, model.coef_[0])
        selected_attributes_test = select_features(attributes_test, model.coef_[0])
        lr.fit(selected_attributes_train, transformed_classes_learn)
        probas = lr.predict_proba(selected_attributes_test)
        print class_num, np.mean(cross_val_score(lr, selected_attributes_train, transformed_classes_learn,
                                                 scoring=make_scorer(roc_auc_scorer, needs_proba=True),
                                                 n_jobs=num_of_jobs,
                                                 cv=indices)), roc_auc_scorer(transformed_classes_test, probas)
        print 'SVC'
        svc_results = {}
        for C in c_range:
            print C
            estimator = SVC(C=C, kernel='linear', probability=True)
            selector = RFECV(estimator, cv=indices, scoring=make_scorer(roc_auc_scorer, needs_proba=True),
                             n_jobs=num_of_jobs)
            selector.fit(attributes_learn, transformed_classes_learn)
            support = selector.get_support(indices=True)
            selected_attributes_train = attributes_learn[:, support]
            result = np.mean(cross_val_score(estimator, selected_attributes_train, transformed_classes_learn,
                                             scoring=make_scorer(roc_auc_scorer, needs_proba=True), n_jobs=num_of_jobs,
                                             cv=indices))
            svc_results[C] = (result, support)
        best = max(svc_results.items(), key=lambda x:x[1][0])
        best_c = best[0]
        best_cv_res, best_indices = best[1]
        print best_c, best_indices
        selected_attributes_train = attributes_learn[:, best_indices]
        selected_attributes_test = attributes_test[:, best_indices]
        svc = SVC(C=best_c, kernel='linear', probability=True)
        svc.fit(selected_attributes_train, transformed_classes_learn)
        probas = svc.predict_proba(selected_attributes_test)
        print class_num, best_cv_res, roc_auc_scorer(transformed_classes_test, probas)
        print '-----------------------------------------------------------'
