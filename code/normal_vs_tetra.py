#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import cPickle as pickle
from sklearn.svm import SVC
from ga_stuff import individual_fitness
from feature_selection_for_svc import cv_for_given_splits_and_features
from sklearn.metrics import roc_auc_score

attributes_learn = pickle.load(open(os.path.join('..', 'datasets', 'attributes_learn.dump')))
attributes_test = pickle.load(open(os.path.join('..', 'datasets', 'attributes_test.dump')))
attributes_learn_tetra = pickle.load(open(os.path.join('..', 'datasets', 'attributes_learn_with_tetra.dump')))
attributes_test_tetra = pickle.load(open(os.path.join('..', 'datasets', 'attributes_test_with_tetra.dump')))

classes_learn = pickle.load(open(os.path.join('..', 'datasets', 'classes_learn.dump')))
classes_test = pickle.load(open(os.path.join('..', 'datasets', 'classes_test.dump')))
print attributes_learn.shape, attributes_learn_tetra.shape
indices = pickle.load(open(os.path.join('..', 'datasets', 'cv_indices.dump')))

svc_clf = SVC(probability=True)
num_of_feats = attributes_learn.shape[1]
print individual_fitness(
    cv_for_given_splits_and_features(svc_clf, attributes_learn, classes_learn, indices,
                                     range(num_of_feats)))  # (1.901320716712803,)


num_of_feats = attributes_learn_tetra.shape[1]
print individual_fitness(
    cv_for_given_splits_and_features(svc_clf, attributes_learn_tetra, classes_learn, indices,
                                     range(num_of_feats)))  # (1.9444894000271329,)

svc_clf.fit(attributes_learn, classes_learn)
pos_proba = svc_clf.predict_proba(attributes_test)[:, 1]
print roc_auc_score(classes_test, pos_proba, None)
svc_clf.fit(attributes_learn_tetra, classes_learn)
pos_proba = svc_clf.predict_proba(attributes_test_tetra)[:, 1]
print roc_auc_score(classes_test, pos_proba, None)
#todo sprawdzić, czy rzeczywićie pierwsze atrybuty się zgadzają