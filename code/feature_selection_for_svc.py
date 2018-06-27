#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import psutil
import cPickle as pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from ml_stuff import binary_classification_evaluation
from ml_stuff import preprocess_results_for_given_splits_and_features
from ga_stuff import individual_fitness


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
        evaluated = binary_classification_evaluation(y_test, probas, None, 0, [0, 1])
        results.append(evaluated)
    results = preprocess_results_for_given_splits_and_features(np.array(results))
    #print results
    return np.mean(results[:, 0]), np.std(results[:, 0]), np.mean(results[:, 1]), np.std(results[:, 1])
if __name__ == '__main__':
    attributes_learn = pickle.load(open(os.path.join('..', 'datasets', 'attributes_learn.dump')))
    classes_learn = pickle.load(open(os.path.join('..', 'datasets', 'classes_learn.dump')))
    attributes_test = pickle.load(open(os.path.join('..', 'datasets', 'attributes_test.dump')))
    classes_test = pickle.load(open(os.path.join('..', 'datasets', 'classes_test.dump')))
    indices = pickle.load(open(os.path.join('..', 'datasets', 'cv_indices.dump')))

    res_dir = os.path.join('..', 'svm_res')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    scorer = make_scorer(scorer_function, needs_proba=True)
    results = {}
    c_range = 2 ** np.linspace(-5, 5, 11)
    num_of_jobs = psutil.cpu_count()-1 or 1
    for C in c_range:
        print C
        estimator = SVC(C=C, kernel='linear', probability=True)
        selector = RFECV(estimator, cv=indices, scoring=scorer, n_jobs=num_of_jobs)
        selector.fit(attributes_learn, classes_learn)
        support = selector.get_support(indices=True)
        result = individual_fitness(cv_for_given_splits_and_features(selector.estimator, attributes_learn, classes_learn, indices, support))
        print result
        results[C] = (result, selector)

    pickle.dump(results, open(os.path.join(res_dir, 'RFE.dump'), 'w'))
    best_result = max(results.items(), key=lambda item: item[1][0])
    best_selector = best_result[1][1]
    print "RFE AUC:", roc_auc_score(classes_test, best_selector.predict_proba(attributes_test)[:, 1])
    joblib.dump(best_selector, os.path.join(res_dir, 'RFE_best.dump'))


    kbest = SelectKBest(f_classif)
    pipeline = Pipeline([('kbest', kbest), ('svc', SVC(kernel='linear', probability=True))])
    grid_search = GridSearchCV(pipeline,
                               {'kbest__k': range(attributes_learn.shape[1] - 1, 0, -1),
                                'svc__C': c_range},
                               scoring=scorer, cv=indices, n_jobs=num_of_jobs)
    grid_search.fit(attributes_learn, classes_learn)


    print grid_search.best_score_
    print '-----------------------'
    print grid_search.best_params_

    pickle.dump(grid_search, open(os.path.join(res_dir, 'grid_search.dump'), 'w'))
    selection = SelectKBest(f_classif, k=grid_search.best_params_['kbest__k'])
    selection.fit(attributes_learn, classes_learn)
    feat_indices = selection.get_support(indices=True)
    best_clf = SVC(kernel='linear', probability=True, C=grid_search.best_params_['svc__C'])
    result = individual_fitness(cv_for_given_splits_and_features(best_clf, attributes_learn, classes_learn, indices, feat_indices))
    print 'SelectKBest result in CV:', result
    best_clf.fit(attributes_learn, classes_learn)
    probas = best_clf.predict_proba(attributes_test)
    print "SelectKBest AUC:", roc_auc_score(classes_test, probas[:, 1])

