#! /usr/bin/python
# -*- coding: utf-8 -*-

import cPickle as pickle
import os
from matplotlib import pyplot as plt
import itertools
from scipy.stats import pearsonr
from feature_selection_for_svc import scorer_function
from lib import get_best_params_for_selectkbest

beautification_dict = {'relative_trinuc_freqs_one_strand': 'relative frequence',
                       'nuc_frequencies': 'frequence',
                       'relative_nuc_frequencies_one_strand': 'relative frequence'}

def read_importances(path, indices):
    with open(path) as f:
        return {indices[i]:float(x) for i, x in enumerate(f.read().split())}
def feat_beautify(name, d):
    try:
        feat, nuc = name.split('__')
        return'%s %s'%(nuc, d[feat])
    except:
        return x
# lasso
lasso_features = pickle.load(open(os.path.join('..', 'lr_res', 'best_features_LogisticRegression.dump')))

# svm biogram
svc_biogram_best_features = pickle.load(open(os.path.join('..', 'svm_res', 'best_features_biogram.dump'), 'rb'))

# penalized svm
svc_penalized_best_features = pickle.load(
    open(os.path.join('..', 'svm_res', 'best_features_penalizedSVM.dump'), 'rb'))

svc_RFE_results = pickle.load(open(os.path.join('..', 'svm_res', 'RFE.dump')))
best_result = max(svc_RFE_results.items(), key=lambda item: item[1][0])
svc_RFE_best_features = [i for i, b in enumerate(best_result[1][1].support_) if b]
svc_RFE_best_C = best_result[1][1].estimator.C

# svm_SelectKBest
svc_SelectKBest_best_features, svc_RFE_SelectKBest_C = get_best_params_for_selectkbest(
    os.path.join('..', 'svm_res', 'grid_search.dump'))

feat_names = pickle.load(open(os.path.join('..', 'datasets', 'features.dump')))

lasso_importances = read_importances(os.path.join('..', 'datasets', 'lasso_importances'), lasso_features)
kbest_importances = read_importances(os.path.join('..', 'datasets', 'kbest_importances'), svc_SelectKBest_best_features)
rfe_importances = read_importances(os.path.join('..', 'datasets', 'rfe_importances'), svc_RFE_best_features)
penalized_importances = read_importances(os.path.join('..', 'datasets', 'penalized_importances'), svc_penalized_best_features)
biogram_importances = read_importances(os.path.join('..', 'datasets', 'biogram_importances'), svc_biogram_best_features)
Lasso = []
SelectKBest = []
RFE = []
SCAD = []
QuiPT = []
for i, x in enumerate(feat_names):
    print feat_beautify(x, beautification_dict), '\t', lasso_importances.get(i, 0), '\t', kbest_importances.get(i, 0), \
    '\t', rfe_importances.get(i, 0), '\t', penalized_importances.get(i, 0), '\t', biogram_importances.get(i, 0)
    Lasso.append(lasso_importances.get(i, 0))
    SelectKBest.append(kbest_importances.get(i, 0))
    RFE.append(rfe_importances.get(i, 0))
    SCAD.append(penalized_importances.get(i, 0))
    QuiPT.append(biogram_importances.get(i, 0))

plt.figure(figsize=(10, 7))
for x, y in itertools.combinations(("Lasso", "SelectKBest", "RFE", "SCAD", "QuiPT"), 2):
    r = pearsonr(eval(x), eval(y))
    plt.scatter(eval(x), eval(y), label='%s vs %s, r=%.2f, pval=%.1e'%(x, y, r[0], r[1]), s=14)
plt.legend()
plt.xlim(right=2.8)
plt.ylim(bottom=-1.5)
plt.xlabel("Feature weights set1")
plt.ylabel("Feature weights set2")
plt.title("Correlation between feature weights")
plt.savefig(os.path.join('..', 'figures', 'correlation_between_feature_sets.eps'), bbox_inches='tight')
# cross on the figure centered at (0, 0) is an effect of some feature importances after feature selection being set to 0