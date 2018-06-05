#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import cPickle as pickle
import numpy as np
from lib import get_best_params_for_selectkbest
from feature_selection_for_svc import scorer_function
from best_features_and_params import lasso_features
from best_features_and_params import svc_biogram_best_features
from best_features_and_params import svc_penalized_best_features
from best_features_and_params import qda_bottomup_best_features
from best_features_and_params import feats_ga_knn
from best_features_and_params import feats_ga_knn_500
from best_features_and_params import feats_bottomup_knn
from best_features_and_params import feats_ga_qda
from best_features_and_params import feats_ga_qda_500

if __name__ == '__main__':
    # svm_RFE
    svc_RFE_results = pickle.load(open(os.path.join('..', 'svm_res', 'RFE.dump')))
    best_result = max(svc_RFE_results.items(), key=lambda item: item[1][0])
    svc_RFE_best_features = [i for i, b in enumerate(best_result[1][1].support_) if b]

    # svm_SelectKBest
    svc_SelectKBest_best_features, _ = get_best_params_for_selectkbest(
        os.path.join('..', 'svm_res', 'grid_search.dump'))

    with open('../datasets/features.dump') as f:
        feat_names = pickle.load(f)

    all_features = lasso_features + list(svc_penalized_best_features) + list(
        svc_biogram_best_features) + svc_RFE_best_features + svc_SelectKBest_best_features + feats_bottomup_knn[
                       9] + qda_bottomup_best_features + feats_ga_knn_500[9] + feats_ga_qda_500

    counts = [(feature, all_features.count(feature)) for feature in sorted(list(set(all_features)))]
    common = set(lasso_features) & set(svc_penalized_best_features) & set(svc_biogram_best_features) & set(
        svc_RFE_best_features) & set(svc_SelectKBest_best_features) & set(
        feats_bottomup_knn[9]) & set(qda_bottomup_best_features) & set(feats_ga_knn_500[9]) & set(feats_ga_qda_500)
    print [feat_names[i] for i in common]
    print '-------------------'
    for c in sorted(counts, key=lambda x:x[1], reverse=True):
        print c[1], feat_names[c[0]]

    # co jest istotne?
    # po 1 sprawdzić ilość cech, która powstałaby po utworzeniu randomowych setów o tej długości
    # po 2 - które cechy sa istotne? zastanowić się, jak to sprawdzić
