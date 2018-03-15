#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import warnings
import cPickle as pickle
import numpy as np
import argparse
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score
from rpy2.robjects import numpy2ri
from rpy2.robjects import r
from ml_stuff import svc_for_given_splits_and_features
from ga_stuff import individual_fitness

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Genetic algorithms for feature selection")
    parser.add_argument('--infile', action='store', type=str, default=os.path.join('..', 'datasets', 'splits.dump'),
                        help='input file containing cv splits')
    parser.add_argument('--outdir', action='store', type=str, default=os.path.join('..', 'svm_res'),
                        help='directory for results')
    args = parser.parse_args()

    # attributes_learn = pickle.load(open(os.path.join('..', 'datasets', 'attributes_learn.dump')))
    # classes_learn = pickle.load(open(os.path.join('..', 'datasets', 'classes_learn.dump')))
    # attributes_test = pickle.load(open(os.path.join('..', 'datasets', 'attributes_test.dump')))
    # classes_test = pickle.load(open(os.path.join('..', 'datasets', 'classes_test.dump')))
    # indices = pickle.load(open(os.path.join('..', 'datasets', 'cv_indices.dump')))
    splits = pickle.load(open(args.infile))

    lambdas = np.linspace(0.01, 0.05, 41)

    r.library('e1071')
    r.library('MASS')
    r.library('corpcor')
    r.library('statmod')
    r.library('tgp')
    r.library('mlegp')
    r.library('lhs')
    r.library('penalizedSVM')

    numpy2ri.activate()
    # in the first round of cross-validation 5 possible feature sets are selected
    feature_sets = []
    models=[]
    results = []
    models_fname = os.path.join(args.outdir, 'penalized_svm_models.dump')
    results_fname = os.path.join(args.outdir, 'pealized_svm_results.dump')
    for split in splits:
        l_classes_negative = np.asarray([x or -1 for x in split[2]])
        a_res = r['svm.fs'](split[0], l_classes_negative, fs_method='scad', lambda1_set=lambdas)
        model = a_res.rx2('model')
        print model.rx('xind')[0]

        features_indices = [x - 1 for x in model.rx('xind')[0]]
        feature_sets.append(np.array(features_indices))
        models.append(model)
        results.append((features_indices, model.rx('lam.opt')))
    pickle.dump(models, open(models_fname, 'w'))
    pickle.dump(results, open(results_fname, 'w'))

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
