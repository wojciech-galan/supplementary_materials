#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import random
import cPickle as pickle
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import mannwhitneyu


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

def get_distances_between_two_groups_of_points(points1, points2, k):
    """
    Computes distances between points from group1 and group2
    :param points1:
    :param points2:
    :param k: num of nearest neighbours
    :return:dictionary point1:sorted(distances)
    """
    distances = {}
    for point in points1:
        distances[point[0]] = sorted(get_distances_from_point(point, points2).values())[:k]
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


if __name__ == '__main__':
    attrs = np.concatenate((pickle.load(open(os.path.join('..', 'datasets', 'attributes_learn.dump'))),
            pickle.load(open(os.path.join('..', 'datasets', 'attributes_test.dump')))))
    ids = np.concatenate((pickle.load(open(os.path.join('..', 'datasets', 'ids_learn.dump'))),
          pickle.load(open(os.path.join('..', 'datasets', 'ids_test.dump')))))
    classes = np.concatenate((pickle.load(open(os.path.join('..', 'datasets', 'classes_learn.dump'))),
              pickle.load(open(os.path.join('..', 'datasets', 'classes_test.dump')))))
    ga_indices = [0, 4, 5, 8, 22, 23, 24, 26, 27, 30, 34, 35, 36, 39, 77, 93, 98]
    random_indices_list = []
    possible_indices = range(attrs.shape[1])
    for i in range(5):
        random_indices = random.sample(possible_indices, len(ga_indices))
        random_indices_list.append(random_indices)

    def compute_distances_for_given_indices(attributes, indices, ids, classes):
        attribs = attributes[:,indices]
        points = zip(ids, attribs)
        distances_0 = get_distances([points[x] for x in range(len(points)) if classes[x] == 0])
        distances_1 = get_distances([points[x] for x in range(len(points)) if classes[x] == 1])
        return points, distances_0, distances_1

    def flatten(list_of_lists):
        return [item for sublist in list_of_lists for item in sublist]

    points_ga, distances_0_ga, distances_1_ga = compute_distances_for_given_indices(attrs, ga_indices, ids,
                                                                                    classes)

    random_distances_0 = []
    random_distances_1 = []
    points_random = []
    for random_indices in random_indices_list:
        p, d0, d1 = compute_distances_for_given_indices(attrs, random_indices, ids, classes)
        points_random.append(p)
        random_distances_0.append(d0)
        random_distances_1.append(d1)

    for k in range(1, 11):
        print k
        distances_0_1 = get_distances_between_two_groups_of_points(
            [points_ga[x] for x in range(len(points_ga)) if classes[x] == 0],
            [points_ga[x] for x in range(len(points_ga)) if classes[x] == 1],
            k
        )
        k_nearest_0_ga = flatten([value for value in k_nearest(distances_0_ga, k).values()])
        k_nearest_1_ga = flatten([value for value in k_nearest(distances_1_ga, k).values()])
        distances_0_1_all_values = flatten(distances_0_1.values())
        print np.mean(k_nearest_0_ga), np.std(k_nearest_0_ga)
        print np.mean(k_nearest(distances_1_ga, k).values()), np.std(k_nearest(distances_1_ga, k).values())
        print np.mean(distances_0_1_all_values), np.std(distances_0_1_all_values)

        random_distances_0_1_all_values = flatten(flatten([get_distances_between_two_groups_of_points
            ([points[x] for x in range(len(points)) if classes[x] == 0],
             [points[x] for x in range(len(points)) if classes[x] == 1], k).values() for points in points_random]))
        k_nearest_0_random = flatten(flatten([k_nearest(distances, k).values() for distances in random_distances_0]))
        k_nearest_1_random = flatten(flatten([k_nearest(distances, k).values() for distances in random_distances_1]))
        print np.mean(k_nearest_0_random), np.std(k_nearest_0_random), \
            mannwhitneyu(k_nearest_0_ga, k_nearest_0_random, alternative='less')
        print np.mean(k_nearest_1_random), np.std(k_nearest_1_random), \
            mannwhitneyu(k_nearest_1_ga, k_nearest_1_random, alternative='less')
        print np.mean(random_distances_0_1_all_values), np.std(random_distances_0_1_all_values), \
            mannwhitneyu(distances_0_1_all_values, random_distances_0_1_all_values, alternative='greater')

