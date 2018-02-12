#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import warnings
import cPickle as pickle
import numpy as np
from sklearn.preprocessing import binarize
from rpy2.robjects import numpy2ri
from rpy2.robjects import r

attributes_learn = pickle.load(open(os.path.join('..', 'datasets', 'attributes_learn.dump')))
classes_learn = pickle.load(open(os.path.join('..', 'datasets', 'classes_learn.dump')))
attributes_test = pickle.load(open(os.path.join('..', 'datasets', 'attributes_test.dump')))
classes_test = pickle.load(open(os.path.join('..', 'datasets', 'classes_test.dump')))
indices = pickle.load(open(os.path.join('..', 'datasets', 'cv_indices.dump')))

binarized_attributes_learn = binarize(attributes_learn)
binarized_attributes_test = binarize(attributes_test)

r.library( 'biogram' )
numpy2ri.activate()
p_values = r.test_features(classes_learn, binarized_attributes_learn)
print r.summary( p_values )
#print p_values
groups = r.cut(p_values)
print groups
