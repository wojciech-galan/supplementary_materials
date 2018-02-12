#! /usr/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def binary_classification_evaluation(classes_proper, classes_probas, ids, positive_num, class_order, pos_ind=1):
    '''Bierze:
        - listę własciwych klas
        - listę prawdopodobieństw klas
        - listę id
        - identyfikator klasy "pozytywnej"
        - porządek, w jakim zwracane są klasy
    Zwraca:
        - correctness
        - sensitivity
        - specificity
        - precision (PPV)
        - NPV
        - FPR
        - FDR
        - F1 score
        - Mathews correlation coefficient
        - dla każdego niepoprawnie sklasyfikowanego elementu obiekt typu ImproperlyClassified. Obiekty te będa zwracane w liście'''
    # na podstawie http://en.wikipedia.org/wiki/Sensitivity_and_specificity
    assert len(classes_proper) == len(classes_probas)
    assert set(classes_proper) == set([0, 1])
    pos_probas = classes_probas[:, pos_ind]
    classes_predicted = [class_order[x.argmax()] for x in classes_probas]

    try:
        auc = roc_auc_score(classes_proper, pos_probas, None)
    except ValueError:
        auc = 0
    if matthews_corrcoef:
        mcc = matthews_corrcoef(classes_proper, classes_predicted)
    else:
        l = len(classes_proper)
        tp = float(len([x for x in range(l) if classes_proper[x] == classes_predicted[x] == positive_num]))
        fp = float(len([x for x in range(l) if classes_proper[x] != classes_predicted[x] == positive_num]))
        tn = float(len([x for x in range(l) if classes_proper[x] == classes_predicted[x] != positive_num]))
        fn = float(len([x for x in range(l) if classes_proper[x] != classes_predicted[x] != positive_num]))
        if any([tp + fp == 0, tp + fn == 0, tn + fp == 0, tn + fn == 0]):
            mcc = 0
        else:
            mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return mcc, auc


def knn(X_train, X_test, y_train, neighbours):
    clf = neighbors.KNeighborsClassifier(neighbours)
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test), clf.classes_


def qda(X_train, X_test, y_train):
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test), clf.classes_

def svc(X_train, X_test, y_train, probability=True, **kwargs):
    clf = SVC(probability=probability, **kwargs)
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test), clf.classes_


def preprocess_results_for_given_splits_and_features(results):
    '''
    Scales MCC to range [0,1]
    :param results: numpy array [[mcc, auc], [mcc, auc],...]
    :return: scaled results
    '''
    results[:, 0] = (results[:, 0] + 1) / 2
    return results


def knn_for_given_splits_and_features(features_indexes, splits, positive_class, neighbours):
    results = []
    for split in splits:
        l, t, l_classes, t_classes, l_ids, t_ids = split
        l = l[:, features_indexes]
        t = t[:, features_indexes]
        res, class_order = knn(l, t, l_classes, neighbours)
        evaluated = binary_classification_evaluation(t_classes, res, t_ids, positive_class, class_order)
        results.append(evaluated)
    results = preprocess_results_for_given_splits_and_features(np.array(results))
    return np.mean(results[:, 0]), np.std(results[:, 0]), np.mean(results[:, 1]), np.std(results[:, 1])


def qda_for_given_splits_and_features(features_indexes, splits, positive_class):
    results = []
    for split in splits:
        l, t, l_classes, t_classes, l_ids, t_ids = split
        l = l[:, features_indexes]
        t = t[:, features_indexes]
        res, class_order = qda(l, t, l_classes)
        evaluated = binary_classification_evaluation(t_classes, res, t_ids, positive_class, class_order)
        results.append(evaluated)
    results = preprocess_results_for_given_splits_and_features(np.array(results))
    return np.mean(results[:, 0]), np.std(results[:, 0]), np.mean(results[:, 1]), np.std(results[:, 1])

def svc_for_given_splits_and_features(features_indexes, splits, positive_class, **kwargs):
    results = []
    for split in splits:
        l, t, l_classes, t_classes, l_ids, t_ids = split
        l = l[:, features_indexes]
        t = t[:, features_indexes]
        res, class_order = svc(l, t, l_classes, **kwargs)
        evaluated = binary_classification_evaluation(t_classes, res, t_ids, positive_class, class_order)
        results.append(evaluated)
    results = preprocess_results_for_given_splits_and_features(np.array(results))
    return np.mean(results[:, 0]), np.std(results[:, 0]), np.mean(results[:, 1]), np.std(results[:, 1])