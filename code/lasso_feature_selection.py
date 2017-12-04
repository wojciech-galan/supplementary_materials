#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
import cPickle as pickle
from sklearn.linear_model import LogisticRegressionCV


attributes_learn = pickle.load(open(os.path.join('..', 'datasets', 'attributes_learn.dump')))
classes_learn = pickle.load(open(os.path.join('..', 'datasets', 'classes_learn.dump')))
attributes_test = pickle.load(open(os.path.join('..', 'datasets', 'attributes_test.dump')))
classes_test = pickle.load(open(os.path.join('..', 'datasets', 'classes_test.dump')))
indices = pickle.load(open(os.path.join('..', 'datasets', 'cv_indices.dump')))
features = pickle.load(open(os.path.join('..', 'datasets', 'features.dump')))

model = LogisticRegressionCV(Cs=100, cv=indices, solver='liblinear', penalty='l1')
model.fit(attributes_learn, classes_learn)
print model.score(attributes_test, classes_test)
print model.coef_
print model.intercept_
print np.where(model.coef_)[1]
print np.array(features)[np.where(model.coef_)[1]]

