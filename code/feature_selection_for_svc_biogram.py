#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import cPickle as pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score
from sklearn.preprocessing import binarize
from rpy2.robjects import numpy2ri
from rpy2.robjects import r
from ml_stuff import svc_for_given_splits_and_features
from ga_stuff import individual_fitness

attributes_learn = pickle.load(open(os.path.join('..', 'datasets', 'attributes_learn.dump')))
classes_learn = pickle.load(open(os.path.join('..', 'datasets', 'classes_learn.dump')))
attributes_test = pickle.load(open(os.path.join('..', 'datasets', 'attributes_test.dump')))
classes_test = pickle.load(open(os.path.join('..', 'datasets', 'classes_test.dump')))
indices = pickle.load(open(os.path.join('..', 'datasets', 'cv_indices.dump')))
splits = pickle.load(open(os.path.join('..', 'datasets', 'splits.dump')))

# in the first round of cross-validation 5 possible feature sets are selected
r.library('slam')
r.library('biogram')
numpy2ri.activate()
feature_sets = []
features_left = []
for split in splits:
    binarized_attributes_learn = binarize(split[0])
    p_values = r.test_features(split[2], binarized_attributes_learn, quick=True)
    significant_features = [x for x in range(len(p_values)) if p_values[x] < 0.0001]
    feature_sets.append(np.array(significant_features))
    features_left.append(np.array(list(set(range(len(p_values))) - set(significant_features))))

# in the second round of cross-validation 5 possible feature sets are evaluated
scores = []
for features_indexes in feature_sets:
    scores.append(individual_fitness(
        svc_for_given_splits_and_features(features_indexes, splits, 0, kernel='linear', probability=True)))
print scores  # [(1.8262310714659573,), (1.8389146555594416,), (1.8396458851692163,), (1.8275493008041139,), (1.8213059895311217,)]
max_score = max(scores)
index = scores.index(max_score)
max_features = feature_sets[index]
res_dir = os.path.join('..', 'svm_res')
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

pickle.dump(max_features, open(os.path.join(res_dir, 'best_features_biogram.dump'), 'w'))
scores_for_left_features = []

clf = SVC(kernel='linear', probability=True)
clf.fit(attributes_learn, classes_learn)
probas = clf.predict_proba(attributes_test)
predicted = clf.predict(attributes_test)

print roc_auc_score(classes_test, probas[:,1])   # 0.982733136254
print accuracy_score(classes_test, predicted)    # 0.944672131148
print matthews_corrcoef(classes_test, predicted) # 0.889351731351
