#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import random
import cPickle as pickle
import numpy as np
from matplotlib import pyplot as plt
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
    all_features = reduce(lambda x, y: x + y, (container_of_lists_of_features))
    counts = [(feature, all_features.count(i)) for i, feature in enumerate(ordered_feature_list)]
    return counts


def get_common_features(container_of_containers_of_features):
    return reduce(lambda x, y: set(x) & set(y), container_of_containers_of_features)


def make_histogram(objects):
    '''
    :param objects: container of objects
    :return: dict {object:count}
    '''
    return {o: objects.count(o) for o in set(objects)}


def statistics_for_histograms(hist_dict_container):
    '''

    :param hist_dict_container: container of {object:count}-type dictionaries
    :return:
    '''
    sorted_keys = sorted(
        list(reduce(lambda x, y: set(x) | set(y), [hist_dict.keys() for hist_dict in hist_dict_container])))
    data_list = []
    means = []
    stds = []
    for key in sorted_keys:
        data = [hist_dict.get(key, 0) for hist_dict in hist_dict_container]
        means.append(np.mean(data))
        stds.append(np.std(data))
        print key, np.mean(data), np.std(data)
        data_list.extend([key]*sum(data))
    return sorted_keys, means, stds, data_list


if __name__ == '__main__':
    random.seed(77)
    with open('../datasets/features.dump') as f:
        feat_names = pickle.load(f)

    # svm_RFE
    svc_RFE_results = pickle.load(open(os.path.join('..', 'svm_res', 'RFE.dump')))
    best_result = max(svc_RFE_results.items(), key=lambda item: item[1][0])
    svc_RFE_best_features = [i for i, b in enumerate(best_result[1][1].support_) if b]

    # svm_SelectKBest
    svc_SelectKBest_best_features, _ = get_best_params_for_selectkbest(
        os.path.join('..', 'svm_res', 'grid_search.dump'))

    all_features_after_selection = [lasso_features, list(svc_penalized_best_features), list(svc_biogram_best_features),
                                    svc_RFE_best_features, svc_SelectKBest_best_features, feats_bottomup_knn[9],
                                    qda_bottomup_best_features, feats_ga_knn_500[9], feats_ga_qda_500]

    counts = get_feature_counts(all_features_after_selection, feat_names)
    common = get_common_features(all_features_after_selection)
    print [feat_names[i] for i in common]
    print '-------------------'
    for c in sorted(counts, key=lambda x: x[1], reverse=True):
        print c

    lengths = [len(x) for x in
               [lasso_features, svc_penalized_best_features, svc_biogram_best_features, svc_RFE_best_features,
                svc_SelectKBest_best_features, feats_bottomup_knn[9], qda_bottomup_best_features, feats_ga_knn_500[9],
                feats_ga_qda_500]]
    container_of_random_common_features = []
    container_or_random_feature_histograms = []
    for x in range(10000):
        all_random_features = [create_random_feature_list(range(len(feat_names)), length) for length in lengths]
        container_of_random_common_features.append(get_common_features(all_random_features))
        container_or_random_feature_histograms.append(
            make_histogram([v for k, v in get_feature_counts(all_random_features, range(len(feat_names)))]))
    lengths = [len(x) for x in container_of_random_common_features]
    lengths_set = set(lengths)
    print [(x, lengths.count(x)) for x in lengths_set]  # [(0, 985703), (1, 14214), (2, 81), (3, 2)]
    # import pdb
    # pdb.set_trace()
    random_keys, random_means, random_stds, data_list = statistics_for_histograms(container_or_random_feature_histograms)
    from scipy.stats import ks_2samp
    print ks_2samp([x[1] for x in counts], data_list)
    print data_list

    feat_sel_histogram = make_histogram([x[1] for x in counts])
    print feat_sel_histogram
    print random_keys, random_means, random_stds
    plt.bar(np.array(random_keys)+0.4, random_means, yerr=random_stds, alpha=0.5, width=0.5)
    plt.bar(feat_sel_histogram.keys(), feat_sel_histogram.values(), alpha=0.5, width=0.5)
    plt.show()

    # co jest istotne?
    # po 1 sprawdzić ilość cech, która powstałaby po utworzeniu randomowych setów o tej długości
    # po 2 - które cechy sa istotne? zastanowić się, jak to sprawdzić
