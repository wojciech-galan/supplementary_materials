#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import cPickle as pickle
import numpy as np
from scipy.spatial.distance import euclidean


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
    attrs_test = pickle.load(open(os.path.join('..', 'datasets', 'attributes_test.dump')))
    ids_test = pickle.load(open(os.path.join('..', 'datasets', 'ids_test.dump')))
    classes_test = pickle.load(open(os.path.join('..', 'datasets', 'classes_test.dump')))
    points = zip(ids_test, attrs_test)
    print classes_test
    distances_0 = get_distances([points[x] for x in range(len(points)) if classes_test[x] == 0])
    distances_1 = get_distances([points[x] for x in range(len(points)) if classes_test[x] == 1])
    for k in range(1, 11):
        print k
        distances_0_1 = get_distances_between_two_groups_of_points(
            [points[x] for x in range(len(points)) if classes_test[x] == 0],
            [points[x] for x in range(len(points)) if classes_test[x] == 1],
            k
        )
        distances_1_0 = get_distances_between_two_groups_of_points(
            [points[x] for x in range(len(points)) if classes_test[x] == 0],
            [points[x] for x in range(len(points)) if classes_test[x] == 1],
            k
        )
        print np.mean(k_nearest(distances_0, k).values()), np.std(k_nearest(distances_0, k).values())
        print np.mean(k_nearest(distances_1, k).values()), np.std(k_nearest(distances_1, k).values())
        print np.mean(distances_0_1.values()), np.std(distances_0_1.values())
        print np.mean(distances_1_0.values()), np.std(distances_1_0.values())
