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

    splits = pickle.load(open(args.infile))
    c_range = 2 ** np.linspace(-5, 5, 11)
    lambdas = 1/c_range


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
        results.append((np.array(features_indices), model.rx('lam.opt')))
    pickle.dump(models, open(models_fname, 'w'))
    pickle.dump(results, open(results_fname, 'w'))

    # in the second round of cross-validation 5 possible feature sets are evaluated
    scores = {}
    for res in results:
        for c in c_range:
            features_indices, _ = res
            scores[(tuple(features_indices), c)] = individual_fitness(svc_for_given_splits_and_features(features_indices, splits, 0, kernel='linear', C=c, probability=True))
    print scores  #
    max_score = max(scores.items(), key=lambda x: x[1])
    max_features = max_score[0][0]
    max_c = max_score[0][1]
    res_dir = os.path.join('..', 'svm_res')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    pickle.dump(max_features, open(os.path.join(res_dir, 'best_features_penalizedSVM.dump'), 'w'))
    pickle.dump(max_c, open(os.path.join(res_dir, 'best_C_penalizedSVM.dump'), 'w'))
