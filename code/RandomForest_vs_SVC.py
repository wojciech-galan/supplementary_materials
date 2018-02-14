#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import cPickle as pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from ga_stuff import individual_fitness
from feature_selection_for_svc import cv_for_given_splits_and_features

attributes_learn = pickle.load(open(os.path.join('..', 'datasets', 'attributes_learn.dump')))
classes_learn = pickle.load(open(os.path.join('..', 'datasets', 'classes_learn.dump')))
attributes_test = pickle.load(open(os.path.join('..', 'datasets', 'attributes_test.dump')))
classes_test = pickle.load(open(os.path.join('..', 'datasets', 'classes_test.dump')))
indices = pickle.load(open(os.path.join('..', 'datasets', 'cv_indices.dump')))
splits = pickle.load(open(os.path.join('..', 'datasets', 'splits.dump')))

rf_clf = RandomForestClassifier(random_state=1)
num_of_feats = attributes_learn.shape[1]
rf_clf.fit(attributes_learn, classes_learn)
probas = rf_clf.predict_proba(attributes_test)
print individual_fitness(
    cv_for_given_splits_and_features(rf_clf, attributes_learn, classes_learn, indices,
                                     range(num_of_feats)))  # (1.842500502502306,)

svc_clf = SVC(probability=True)
svc_clf.fit(attributes_learn, classes_learn)
probas = svc_clf.predict_proba(attributes_test)
print individual_fitness(
    cv_for_given_splits_and_features(svc_clf, attributes_learn, classes_learn, indices,
                                     range(num_of_feats)))  # (1.901320716712803,)
