#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import random
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


def create_random_feature_list(all_features, num_of_features_to_pick):
    return random.sample(all_features, num_of_features_to_pick)


def get_feature_counts(container_of_lists_of_features, ordered_feature_list):
    all_features = reduce(lambda x, y: x+y, (container_of_lists_of_features))
    counts = [(feature, all_features.count(i)) for i, feature in enumerate(ordered_feature_list)]
    return counts


def get_common_features(container_of_containers_of_features):
    return reduce(lambda x, y: set(x) & set(y), container_of_containers_of_features)


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

    all_features_after_selection = [lasso_features, list(svc_penalized_best_features), list(svc_biogram_best_features),
                    svc_RFE_best_features, svc_SelectKBest_best_features, feats_bottomup_knn[9],
                    qda_bottomup_best_features, feats_ga_knn_500[9], feats_ga_qda_500]

    counts = get_feature_counts(all_features_after_selection, feat_names)
    common = get_common_features(all_features_after_selection)
    print [feat_names[i] for i in common]
    print '-------------------'
    for c in sorted(counts, key=lambda x: x[1], reverse=True):
        print c #print c[1], feat_names[c[0]]

    lengths = [len(x) for x in
               [lasso_features, svc_penalized_best_features, svc_biogram_best_features, svc_RFE_best_features,
                svc_SelectKBest_best_features, feats_bottomup_knn[9], qda_bottomup_best_features, feats_ga_knn_500[9],
                feats_ga_qda_500]]

    # co jest istotne?
    # po 1 sprawdzić ilość cech, która powstałaby po utworzeniu randomowych setów o tej długości
    # po 2 - które cechy sa istotne? zastanowić się, jak to sprawdzić
