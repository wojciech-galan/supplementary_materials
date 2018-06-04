#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import random
import cPickle as pickle
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import mannwhitneyu
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


class Data(object):
    def __init__(self):
        pass


def get_distances_from_point(point, other_points):
    """
    Computes distances between a point and other points in multidimensional space
    :param point:
    :param other_points:
    :return: dictionary frozenset(point1, point2):distance
    """
    distances = {}
    for i in range(len(other_points)):
        distances[frozenset((point[0], other_points[i][0]))] = euclidean(point[1], other_points[i][1])
    return distances


def get_distances(points):
    """
    Computes distances between points in multidimensional space
    :param points: iterable containing tuples (point_id, coordinates)
    :return: dictionary frozenset(point1, point2):distance
    """
    distances = {}
    for i in range(len(points) - 1):
        distances.update(get_distances_from_point(points[i], points[i + 1:]))
    return distances


def get_distances_between_two_groups_of_points(points1, points2):
    """
    Computes distances between points from group1 and group2
    :param points1:
    :param points2:
    :return:dictionary point1:sorted(distances)
    """
    distances = {}
    for point in points1:
        distances[point[0]] = sorted(get_distances_from_point(point, points2).values())
    return distances


def k_nearest(distances, k):
    """
    Selects distances between a point and its k nearest neighbours
    :param distances: dictionary frozenset(point1, point2):distance
    :param k: (integer) num of nearest neighbours
    :return: dictionary point:[distances]
    """
    res = {}
    points = set()
    for key in distances:
        points.update(key)
    for point in points:
        temp = []
        for key, value in distances.iteritems():
            if point in key:
                temp.append(value)
        res[point] = sorted(temp)[:k]
    return res


def compute_distances_for_given_indices(attributes, indices, ids, classes):
    attribs = attributes[:, indices]
    points = zip(ids, attribs)
    distances_0 = get_distances([points[x] for x in range(len(points)) if classes[x] == 0])
    distances_1 = get_distances([points[x] for x in range(len(points)) if classes[x] == 1])
    return points, distances_0, distances_1


def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def compute_statistics(indices, k_lowest, k_highest, attrs, classes, ids, description):
    print description
    random_indices_list = []
    possible_indices = range(attrs.shape[1])
    for _ in range(5):
        random_indices = random.sample(possible_indices, len(indices))
        random_indices_list.append(random_indices)
    points_ga, distances_0, distances_1 = compute_distances_for_given_indices(attrs, indices, ids,
                                                                              classes)
    distances_0_1 = get_distances_between_two_groups_of_points(
        [points_ga[x] for x in range(len(points_ga)) if classes[x] == 0],
        [points_ga[x] for x in range(len(points_ga)) if classes[x] == 1]).values()

    random_distances_0 = []
    random_distances_1 = []
    points_random = []
    for random_indices in random_indices_list:
        p, d0, d1 = compute_distances_for_given_indices(attrs, random_indices, ids, classes)
        points_random.append(p)
        random_distances_0.append(d0)
        random_distances_1.append(d1)

    random_distances_0_1 = [
        get_distances_between_two_groups_of_points([points[x] for x in range(len(points)) if classes[x] == 0],
                                                   [points[x] for x in range(len(points)) if classes[x] == 1]).values()
        for points in points_random]

    for k in range(k_lowest, k_highest + 1):
        print k
        distances_0_1_k = flatten([sorted_distances[:k] for sorted_distances in distances_0_1])

        k_nearest_0 = flatten([value for value in k_nearest(distances_0, k).values()])
        k_nearest_1 = flatten([value for value in k_nearest(distances_1, k).values()])
        print np.mean(k_nearest_0), np.std(k_nearest_0)
        print np.mean(k_nearest_1), np.std(k_nearest_1)
        print np.mean(distances_0_1_k), np.std(distances_0_1_k)

        random_distances_0_1_k = flatten(flatten([sorted_distances[:k] for sorted_distances in random_distances_0_1]))
        k_nearest_0_random = flatten(flatten([k_nearest(distances, k).values() for distances in random_distances_0]))
        k_nearest_1_random = flatten(flatten([k_nearest(distances, k).values() for distances in random_distances_1]))
        print np.mean(k_nearest_0_random), np.std(k_nearest_0_random), \
            mannwhitneyu(k_nearest_0, k_nearest_0_random, alternative='less')
        print np.mean(k_nearest_1_random), np.std(k_nearest_1_random), \
            mannwhitneyu(k_nearest_1, k_nearest_1_random, alternative='less')
        print np.mean(random_distances_0_1_k), np.std(random_distances_0_1_k), \
            mannwhitneyu(distances_0_1_k, random_distances_0_1_k, alternative='greater')
        print '----------------------------------------------------'


if __name__ == '__main__':
    attributes = np.concatenate((pickle.load(open(os.path.join('..', 'datasets', 'attributes_learn.dump'))),
                            pickle.load(open(os.path.join('..', 'datasets', 'attributes_test.dump')))))
    ids = np.concatenate((pickle.load(open(os.path.join('..', 'datasets', 'ids_learn.dump'))),
                          pickle.load(open(os.path.join('..', 'datasets', 'ids_test.dump')))))
    classes = np.concatenate((pickle.load(open(os.path.join('..', 'datasets', 'classes_learn.dump'))),
                              pickle.load(open(os.path.join('..', 'datasets', 'classes_test.dump')))))
    np.random.seed(77)
    compute_statistics(range(2), 1, 3, 'blah')
    raise

    # svm_RFE
    svc_RFE_results = pickle.load(open(os.path.join('..', 'svm_res', 'RFE.dump')))
    best_result = max(svc_RFE_results.items(), key=lambda item: item[1][0])
    svc_RFE_best_features = [i for i, b in enumerate(best_result[1][1].support_) if b]

    # svm_SelectKBest
    svc_SelectKBest_best_features, _ = get_best_params_for_selectkbest(
        os.path.join('..', 'svm_res', 'grid_search.dump'))

    k_low = 1
    k_high = 5
    compute_statistics(lasso_features, k_low, k_high, attributes, classes, ids, "LogisticRegression, lasso")
    compute_statistics(svc_RFE_best_features, k_low, k_high, attributes, classes, ids, "svm_RFE")
    compute_statistics(svc_SelectKBest_best_features, k_low, k_high, attributes, classes, ids, "SelectKBEst")
    compute_statistics(svc_biogram_best_features, k_low, k_high, attributes, classes, ids, "svc_biogram_best_features")
    compute_statistics(svc_penalized_best_features, k_low, k_high, attributes, classes, ids, "svc_penalized_best_features")
    compute_statistics(qda_bottomup_best_features, k_low, k_high, attributes, classes, ids, "qda_bottomup_best_features")
    for neighbours, indices in feats_ga_knn.iteritems():
        compute_statistics(indices, k_low, k_high, attributes, classes, ids, "feats_ga_knn%d" % neighbours)
    for neighbours, indices in feats_ga_knn_500.iteritems():
        compute_statistics(indices, k_low, k_high, attributes, classes, ids, "feats_ga_knn_500%d" % neighbours)
    compute_statistics(feats_bottomup_knn, k_low, k_high, attributes, classes, ids, "feats_bottomup_knn")
    compute_statistics(feats_ga_qda, k_low, k_high, attributes, classes, ids, "feats_ga_qda")
    compute_statistics(feats_ga_qda_500, k_low, k_high, attributes, classes, ids, "feats_ga_qda_500")

# 1
# 2.01410542882 0.750136990695
# 0.942203967972 0.858834239322
# 3.36332346587 0.763919705254
# 2.51975681454 0.964933695152 MannwhitneyuResult(statistic=2499611.0, pvalue=3.3987296567595326e-73)
# 1.18831138547 0.941468394807 MannwhitneyuResult(statistic=2932059.0, pvalue=1.4102227497033813e-31)
# 3.12431090192 1.04891741948 MannwhitneyuResult(statistic=4607020.0, pvalue=2.0791909549775136e-40)
# 2
# 2.14876217109 0.759486194753
# 1.07552938141 0.90695994108
# 3.42438860184 0.773626055892
# 2.63978998935 0.978089363577 MannwhitneyuResult(statistic=10233096.0, pvalue=8.4127563157121121e-131)
# 1.32978669989 0.977819795839 MannwhitneyuResult(statistic=11937057.0, pvalue=1.6601414820764725e-53)
# 3.19272114979 1.06075105817 MannwhitneyuResult(statistic=18335444.0, pvalue=8.4646847255111211e-75)
# 3
# 2.24487500323 0.769114148979
# 1.18708535771 0.937663315691
# 3.46660995348 0.781010507052
# 2.72403519706 0.991645413365 MannwhitneyuResult(statistic=23463217.0, pvalue=3.8614008456949143e-179)
# 1.44459004009 1.00175857715 MannwhitneyuResult(statistic=27308036.0, pvalue=5.1072990727392633e-69)
# 3.24020903074 1.06858208916 MannwhitneyuResult(statistic=41095637.0, pvalue=1.03728778262495e-106)
# 4
# 2.31908904386 0.777782680865
# 1.28724237794 0.966924790597
# 3.50012119765 0.78690885354
# 2.78808284724 1.00198109756 MannwhitneyuResult(statistic=42329559.0, pvalue=6.191631154316216e-222)
# 1.54633146822 1.02103368039 MannwhitneyuResult(statistic=49124786.0, pvalue=4.187549994932281e-82)
# 3.27751950242 1.07500817803 MannwhitneyuResult(statistic=72847216.0, pvalue=3.1477396586631834e-137)
# 5
# 2.38141359026 0.785160894758
# 1.37108461457 0.989918004099
# 3.52780959131 0.791780644494
# 2.84117980354 1.00998310132 MannwhitneyuResult(statistic=66967945.0, pvalue=4.4415570478270689e-260)
# 1.63083139347 1.03634504926 MannwhitneyuResult(statistic=77411355.0, pvalue=3.8667265116986017e-94)
# 3.3084440909 1.07981268538 MannwhitneyuResult(statistic=113537447.0, pvalue=2.2333757820126036e-166)
# 6
# 2.43387881664 0.791744692243
# 1.44498817565 1.00736867499
# 3.55369802371 0.793896132152
# 2.8858306558 1.01670749117 MannwhitneyuResult(statistic=97450125.0, pvalue=5.465587941928016e-295)
# 1.70361229028 1.0461978028 MannwhitneyuResult(statistic=112223627.0, pvalue=3.3370401129639703e-105)
# 3.33512052525 1.08359356123 MannwhitneyuResult(statistic=163372467.0, pvalue=1.4100068331979734e-197)
# 7
# 2.47933373728 0.79764811283
# 1.51013165977 1.02103595493
# 3.57755072432 0.795383958207
# 2.92394721128 1.00920790264 MannwhitneyuResult(statistic=132232832.0, pvalue=0.0)
# 1.74621000011 1.044924194 MannwhitneyuResult(statistic=155362543.0, pvalue=7.083272856760456e-102)
# 3.37546200871 1.05655601747 MannwhitneyuResult(statistic=217789803.0, pvalue=2.3791375742728515e-181)
# 8
# 2.51928248716 0.803182032923
# 1.56852898067 1.032017512
# 3.59945677337 0.796610580551
# 2.95872431 1.01445124212 MannwhitneyuResult(statistic=173925348.0, pvalue=0.0)
# 1.8002669206 1.05129841881 MannwhitneyuResult(statistic=204266243.0, pvalue=1.7099771976586359e-107)
# 3.39714169093 1.0596692219 MannwhitneyuResult(statistic=284442793.0, pvalue=9.1400298636531595e-207)
# 9
# 2.55513563553 0.807826678619
# 1.62119730408 1.04001365046
# 3.61974761422 0.797854496897
# 2.98985469213 1.01907398362 MannwhitneyuResult(statistic=221509039.0, pvalue=0.0)
# 1.84912667221 1.05559024427 MannwhitneyuResult(statistic=259905892.0, pvalue=4.6796812374638126e-113)
# 3.41684114034 1.06253350139 MannwhitneyuResult(statistic=360063050.0, pvalue=8.104870949790262e-233)
# 10
# 2.58782449581 0.811933667867
# 1.67085226989 1.04595129896
# 3.63853261096 0.799003367992
# 3.01821657945 1.02330896856 MannwhitneyuResult(statistic=275001515.0, pvalue=0.0)
# 1.89460112155 1.05828573331 MannwhitneyuResult(statistic=322401164.0, pvalue=4.8734900560421573e-118)
# 3.43488330709 1.06500240674 MannwhitneyuResult(statistic=444634717.0, pvalue=3.3213814290361134e-259)
