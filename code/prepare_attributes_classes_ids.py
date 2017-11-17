#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import cPickle as pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


def virus_to_attributes(virus, choosen_attributes):
    l = []
    for att in choosen_attributes:
        if type(virus.__dict__[att]) == dict:
            l.extend(virus.__dict__[att][key] for key in sorted(virus.__dict__[att]))
        else:
            l.append(virus.__dict__[att])
    return l


# loading datasets
test_eu = pickle.load(open(os.path.join('..', 'datasets', 'viruses_blind_set_eu.dump')))
learn_eu = pickle.load(open(os.path.join('..', 'datasets', 'viruses_crossvalidation_eu.dump')))
test_phages = pickle.load(open(os.path.join('..', 'datasets', 'viruses_blind_set_phages.dump')))
learn_phages = pickle.load(open(os.path.join('..', 'datasets', 'viruses_crossvalidation_phages.dump')))

# preparing classes
classes_test = np.array([0] * len(test_eu) + [1] * len(test_phages))
classes_learn = np.array([0] * len(learn_eu) + [1] * len(learn_phages))

# preparing attributes
attribs = ('molecule', 'nuc_frequencies', 'relative_nuc_frequencies_one_strand', 'relative_trinuc_freqs_one_strand')
attributes_learn = [virus_to_attributes(virus, attribs) for virus in learn_eu]
attributes_learn.extend([virus_to_attributes(virus, attribs) for virus in learn_phages])
attributes_learn = np.array(attributes_learn)
attributes_test = [virus_to_attributes(virus, attribs) for virus in test_eu]
attributes_test.extend([virus_to_attributes(virus, attribs) for virus in test_phages])
attributes_test = np.array(attributes_test)

# preparing ids
ids_learn = [virus.gi for virus in learn_eu]
ids_learn.extend([virus.gi for virus in learn_phages])
ids_learn = np.array(ids_learn)
ids_test = [virus.gi for virus in test_eu]
ids_test.extend([virus.gi for virus in test_phages])
ids_test = np.array(ids_test)

# scaling features
scaler = StandardScaler()
attributes_learn = scaler.fit_transform(attributes_learn)
attributes_test = scaler.transform(attributes_test)

# preparing crossvalidation splits
skf = StratifiedKFold(n_splits=5)
cv_splits = []
for i, (train_index, test_index) in enumerate(skf.split(attributes_learn, classes_learn)):
    cv_splits.append([])
    cv_splits[i].append(attributes_learn[train_index])
    cv_splits[i].append(attributes_learn[test_index])
    cv_splits[i].append(classes_learn[train_index])
    cv_splits[i].append(classes_learn[test_index])
    cv_splits[i].append(ids_learn[train_index])
    cv_splits[i].append(ids_learn[test_index])

# serializing data
pickle.dump(cv_splits, open(os.path.join('..', 'datasets', 'splits.dump'), 'w'))
pickle.dump(attributes_learn, open(os.path.join('..', 'datasets', 'attributes_learn.dump'), 'w'))
pickle.dump(attributes_test, open(os.path.join('..', 'datasets', 'attributes_test.dump'), 'w'))
pickle.dump(classes_learn, open(os.path.join('..', 'datasets', 'classes_learn.dump'), 'w'))
pickle.dump(classes_test, open(os.path.join('..', 'datasets', 'classes_test.dump'), 'w'))
pickle.dump(ids_learn, open(os.path.join('..', 'datasets', 'ids_learn.dump'), 'w'))
pickle.dump(ids_test, open(os.path.join('..', 'datasets', 'ids_test.dump'), 'w'))
