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
                                    svc_RFE_best_features, svc_SelectKBest_best_features, # feats_bottomup_knn[9], ga and qda returned the same set of features
                                    qda_bottomup_best_features, feats_ga_knn_500[9], feats_ga_qda_500]

    counts = get_feature_counts(all_features_after_selection, feat_names)
    common = get_common_features(all_features_after_selection)
    print [feat_names[i] for i in common]
    print '-------------------'
    for c in sorted(counts, key=lambda x: x[1], reverse=True):
        print c

    lengths = [len(x) for x in
               [lasso_features, svc_penalized_best_features, svc_biogram_best_features, svc_RFE_best_features,
                svc_SelectKBest_best_features, qda_bottomup_best_features, feats_ga_knn_500[9], # feats_bottomup_knn[9], ga and qda returned the same set of features
                feats_ga_qda_500]]
    container_of_random_common_features = []
    container_or_random_feature_histograms = []
    for x in range(1000000):
        all_random_features = [create_random_feature_list(range(len(feat_names)), length) for length in lengths]
        container_of_random_common_features.append(get_common_features(all_random_features))
        container_or_random_feature_histograms.append(
            make_histogram([v for k, v in get_feature_counts(all_random_features, range(len(feat_names)))]))
    lengths = [len(x) for x in container_of_random_common_features]
    lengths_set = set(lengths)
    print [(x, lengths.count(x)) for x in lengths_set]  # [(0, 917675), (1, 79291), (2, 2965), (3, 67), (4, 2)]
    plt.hist([0] * 917675 + [1] * 79291 + [2] * 2965 + [3] * 67 + [4] * 2, 4, align='left')  # , log=True)
    plt.axis([-0.5, 4.5, 0, 1000000])
    plt.grid(True)
    print range(0, 1200000, 200000)
    plt.yticks(range(0, 1200000, 200000), np.array(range(0, 12, 2)) / 10.)
    plt.ylim([0, 10 ** 6])
    plt.xlabel('Number of common elements in the intersection')
    plt.ylabel('Fraction of intersections in each group')
    plt.savefig(os.path.join('..', 'figures', 'intersections_histogram.svg'), bbox_inches='tight')

    # random_keys, random_means, random_stds, data_list = statistics_for_histograms(container_or_random_feature_histograms)
    # from scipy.stats import ks_2samp
    # print ks_2samp([x[1] for x in counts], data_list)
    # pdb.set_trace()
    # print data_list
    #
    # feat_sel_histogram = make_histogram([x[1] for x in counts])
    # print feat_sel_histogram
    # print random_keys, random_means, random_stds
    # plt.bar(np.array(random_keys)+0.4, random_means, yerr=random_stds, alpha=0.5, width=0.5)
    # plt.bar(feat_sel_histogram.keys(), feat_sel_histogram.values(), alpha=0.5, width=0.5)
    # plt.show()

# ('molecule', 8)
# ('relative_trinuc_freqs_one_strand__GGA', 8)
# ('relative_nuc_frequencies_one_strand__TG', 7)
# ('relative_trinuc_freqs_one_strand__TAT', 7)
# ('relative_trinuc_freqs_one_strand__TGT', 7)
# ('relative_trinuc_freqs_one_strand__TTA', 7)
# ('relative_trinuc_freqs_one_strand__TTC', 7)
# ('nuc_frequencies__CC', 6)
# ('nuc_frequencies__TA', 6)
# ('relative_nuc_frequencies_one_strand__AG', 6)
# ('relative_nuc_frequencies_one_strand__CC', 6)
# ('relative_nuc_frequencies_one_strand__GC', 6)
# ('relative_trinuc_freqs_one_strand__AGA', 6)
# ('relative_trinuc_freqs_one_strand__CTA', 6)
# ('relative_trinuc_freqs_one_strand__TTT', 6)
# ('nuc_frequencies__CA', 5)
# ('relative_nuc_frequencies_one_strand__CG', 5)
# ('relative_trinuc_freqs_one_strand__AAG', 5)
# ('relative_trinuc_freqs_one_strand__AAT', 5)
# ('relative_trinuc_freqs_one_strand__ACG', 5)
# ('relative_trinuc_freqs_one_strand__GAA', 5)
# ('relative_trinuc_freqs_one_strand__GAG', 5)
# ('relative_trinuc_freqs_one_strand__GCC', 5)
# ('relative_trinuc_freqs_one_strand__GGC', 5)
# ('relative_trinuc_freqs_one_strand__GGG', 5)
# ('relative_trinuc_freqs_one_strand__GTC', 5)
# ('relative_trinuc_freqs_one_strand__TCT', 5)
# ('relative_trinuc_freqs_one_strand__TGA', 5)
# ('relative_trinuc_freqs_one_strand__TGG', 5)
# ('nuc_frequencies__AA', 4)
# ('nuc_frequencies__AG', 4)
# ('nuc_frequencies__CT', 4)
# ('nuc_frequencies__GG', 4)
# ('nuc_frequencies__GT', 4)
# ('nuc_frequencies__TG', 4)
# ('relative_nuc_frequencies_one_strand__AA', 4)
# ('relative_nuc_frequencies_one_strand__AT', 4)
# ('relative_nuc_frequencies_one_strand__GA', 4)
# ('relative_nuc_frequencies_one_strand__TA', 4)
# ('relative_nuc_frequencies_one_strand__TC', 4)
# ('relative_nuc_frequencies_one_strand__TT', 4)
# ('relative_trinuc_freqs_one_strand__AGC', 4)
# ('relative_trinuc_freqs_one_strand__ATC', 4)
# ('relative_trinuc_freqs_one_strand__ATG', 4)
# ('relative_trinuc_freqs_one_strand__CAC', 4)
# ('relative_trinuc_freqs_one_strand__CGT', 4)
# ('relative_trinuc_freqs_one_strand__CTC', 4)
# ('relative_trinuc_freqs_one_strand__GAC', 4)
# ('relative_trinuc_freqs_one_strand__GTA', 4)
# ('relative_trinuc_freqs_one_strand__TAC', 4)
# ('relative_trinuc_freqs_one_strand__TAG', 4)
# ('relative_trinuc_freqs_one_strand__TCA', 4)
# ('relative_trinuc_freqs_one_strand__TTG', 4)
# ('nuc_frequencies__AC', 3)
# ('nuc_frequencies__AT', 3)
# ('nuc_frequencies__CG', 3)
# ('nuc_frequencies__G', 3)
# ('nuc_frequencies__GA', 3)
# ('nuc_frequencies__GC', 3)
# ('nuc_frequencies__T', 3)
# ('relative_nuc_frequencies_one_strand__AC', 3)
# ('relative_nuc_frequencies_one_strand__CT', 3)
# ('relative_nuc_frequencies_one_strand__GG', 3)
# ('relative_trinuc_freqs_one_strand__AGG', 3)
# ('relative_trinuc_freqs_one_strand__ATT', 3)
# ('relative_trinuc_freqs_one_strand__CAT', 3)
# ('relative_trinuc_freqs_one_strand__CCT', 3)
# ('relative_trinuc_freqs_one_strand__CTG', 3)
# ('relative_trinuc_freqs_one_strand__CTT', 3)
# ('relative_trinuc_freqs_one_strand__GAT', 3)
# ('relative_trinuc_freqs_one_strand__GCT', 3)
# ('relative_trinuc_freqs_one_strand__GGT', 3)
# ('relative_trinuc_freqs_one_strand__GTG', 3)
# ('relative_trinuc_freqs_one_strand__TAA', 3)
# ('relative_trinuc_freqs_one_strand__TCC', 3)
# ('relative_trinuc_freqs_one_strand__TGC', 3)
# ('nuc_frequencies__A', 2)
# ('nuc_frequencies__TC', 2)
# ('relative_nuc_frequencies_one_strand__CA', 2)
# ('relative_nuc_frequencies_one_strand__GT', 2)
# ('relative_trinuc_freqs_one_strand__AAA', 2)
# ('relative_trinuc_freqs_one_strand__ACA', 2)
# ('relative_trinuc_freqs_one_strand__ACC', 2)
# ('relative_trinuc_freqs_one_strand__ATA', 2)
# ('relative_trinuc_freqs_one_strand__CAA', 2)
# ('relative_trinuc_freqs_one_strand__CAG', 2)
# ('relative_trinuc_freqs_one_strand__CCA', 2)
# ('relative_trinuc_freqs_one_strand__CCC', 2)
# ('relative_trinuc_freqs_one_strand__CCG', 2)
# ('relative_trinuc_freqs_one_strand__CGA', 2)
# ('relative_trinuc_freqs_one_strand__CGG', 2)
# ('relative_trinuc_freqs_one_strand__GCA', 2)
# ('relative_trinuc_freqs_one_strand__GTT', 2)
# ('nuc_frequencies__C', 1)
# ('nuc_frequencies__TT', 1)
# ('relative_trinuc_freqs_one_strand__AAC', 1)
# ('relative_trinuc_freqs_one_strand__ACT', 1)
# ('relative_trinuc_freqs_one_strand__AGT', 1)
# ('relative_trinuc_freqs_one_strand__CGC', 1)
# ('relative_trinuc_freqs_one_strand__GCG', 1)
# ('relative_trinuc_freqs_one_strand__TCG', 1)