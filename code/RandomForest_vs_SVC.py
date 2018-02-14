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
indices = pickle.load(open(os.path.join('..', 'datasets', 'cv_indices.dump')))

rf_clf = RandomForestClassifier(random_state=1)
num_of_feats = attributes_learn.shape[1]
print individual_fitness(
    cv_for_given_splits_and_features(rf_clf, attributes_learn, classes_learn, indices,
                                     range(num_of_feats)))  # (1.842500502502306,)

svc_clf = SVC(probability=True)
print individual_fitness(
    cv_for_given_splits_and_features(svc_clf, attributes_learn, classes_learn, indices,
                                     range(num_of_feats)))  # (1.901320716712803,)
