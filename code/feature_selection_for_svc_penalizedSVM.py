#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import warnings
import cPickle as pickle
import numpy as np
from rpy2.robjects import numpy2ri
from rpy2.robjects import r

attributes_learn = pickle.load(open(os.path.join('..', 'datasets', 'attributes_learn.dump')))
classes_learn = pickle.load(open(os.path.join('..', 'datasets', 'classes_learn.dump')))
attributes_test = pickle.load(open(os.path.join('..', 'datasets', 'attributes_test.dump')))
classes_test = pickle.load(open(os.path.join('..', 'datasets', 'classes_test.dump')))
indices = pickle.load(open(os.path.join('..', 'datasets', 'cv_indices.dump')))
splits = pickle.load(open(os.path.join('..', 'datasets', 'splits.dump')))

lambdas = np.linspace( 0.01, 0.05, 41 )

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r.library('penalizedSVM')

numpy2ri.activate()
feature_sets = []
features_left = []
for split in splits:
    l_classes_negative = np.asarray([x or -1 for x in split[2]])
    a_res = r['svm.fs'](split[0], l_classes_negative, fs_method='scad', lambda1_set=lambdas)
    model = a_res.rx2('model')
    print model.rx('xind')[0]

    features_indices = [x - 1 for x in model.rx('xind')[0]]
    feature_sets.append(np.array(features_indices))
    # p_values = r.test_features(split[2], binarized_attributes_learn, quick=True)
    # significant_features = [x for x in range(len(p_values)) if p_values[x] < 0.0001]
    # feature_sets.append(np.array(significant_features))
    # features_left.append(np.array(list(set(range(len(p_values))) - set(significant_features))))