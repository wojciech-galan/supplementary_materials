#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import warnings
import cPickle as pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score
from rpy2.robjects import numpy2ri
from rpy2.robjects import r
from feature_selection_for_svc_biogram import svc_for_given_splits_and_features
from ga_stuff import individual_fitness

attributes_learn = pickle.load(open(os.path.join('..', 'datasets', 'attributes_learn.dump')))
classes_learn = pickle.load(open(os.path.join('..', 'datasets', 'classes_learn.dump')))
attributes_test = pickle.load(open(os.path.join('..', 'datasets', 'attributes_test.dump')))
classes_test = pickle.load(open(os.path.join('..', 'datasets', 'classes_test.dump')))
indices = pickle.load(open(os.path.join('..', 'datasets', 'cv_indices.dump')))
splits = pickle.load(open(os.path.join('..', 'datasets', 'splits.dump')))

lambdas = np.linspace(0.01, 0.05, 41)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r.library('penalizedSVM')

numpy2ri.activate()
# in the first round of cross-validation 5 possible feature sets are selected
feature_sets = []
features_left = []
for split in splits:
    l_classes_negative = np.asarray([x or -1 for x in split[2]])
    a_res = r['svm.fs'](split[0], l_classes_negative, fs_method='scad', lambda1_set=lambdas)
    import pdb
    pdb.set_trace()
    model = a_res.rx2('model')
    print model.rx('xind')[0]

    features_indices = [x - 1 for x in model.rx('xind')[0]]
    feature_sets.append(np.array(features_indices))
    # p_values = r.test_features(split[2], binarized_attributes_learn, quick=True)
    # significant_features = [x for x in range(len(p_values)) if p_values[x] < 0.0001]
    # feature_sets.append(np.array(significant_features))
    # features_left.append(np.array(list(set(range(len(p_values))) - set(significant_features))))

# in the second round of cross-validation 5 possible feature sets are evaluated
scores = []
for features_indexes in feature_sets:
    scores.append(individual_fitness(
        svc_for_given_splits_and_features(features_indexes, splits, 0, kernel='linear', probability=True)))
print scores  #
max_score = max(scores)
index = scores.index(max_score)
max_features = feature_sets[index]
res_dir = os.path.join('..', 'svm_res')
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

pickle.dump(max_features, open(os.path.join(res_dir, 'best_features_penalizedSVM.dump'), 'w'))
scores_for_left_features = []

clf = SVC(kernel='linear', probability=True)
clf.fit(attributes_learn, classes_learn)
probas = clf.predict_proba(attributes_test)
predicted = clf.predict(attributes_test)

print roc_auc_score(classes_test, probas[:,1])   # 0.982733136254
print accuracy_score(classes_test, predicted)    # 0.944672131148
print matthews_corrcoef(classes_test, predicted) # 0.889351731351