#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import random
import argparse
import itertools
import cPickle as pickle
import numpy as np
from funcy import rpartial
from ga_stuff import individual_fitness
from ml_stuff import knn_for_given_splits_and_features, qda_for_given_splits_and_features


def check_initial_combinations(classifier_for_given_splits_and_features, all_feature_indices, splits,
                               current_results={}):
    for i in range(1, 4):
        for indices_combination in itertools.combinations(all_feature_indices, i):
            key = set(indices_combination)
            if key in current_results:
                continue
            current_results[key] = classifier_for_given_splits_and_features(indices_combination, splits, 0)
            print current_results
    return current_results


def append_feature(all_feature_indices, splits, positive_class, curr_feature_indices, curr_results,
                   classifier_for_given_splits_and_features):
    # curr_feature_indices to powinien być set
    curr_feature_indices = curr_feature_indices[:]
    not_present_features_indices = [index for index in all_feature_indices if not index in curr_feature_indices]
    while not_present_features_indices:
        feature_index = random.choice(not_present_features_indices)
        curr_feature_indices.append(feature_index)
        key = set(curr_feature_indices)
        if not key in curr_results:
            curr_results[key] = classifier_for_given_splits_and_features(curr_feature_indices, splits, positive_class)
            remove_feature(curr_feature_indices, splits, positive_class, key, curr_results,
                          classifier_for_given_splits_and_features)
            print key, curr_results[key][-4:]
        not_present_features_indices.remove(feature_index)


def remove_feature(splits, positive_class, neighbours, curr_feature_indices_stable_copy,
                  curr_results, classifier_for_given_splits_and_features):
    for index in curr_feature_indices_stable_copy:
        curr_feature_indices = curr_feature_indices_stable_copy[:]
        curr_feature_indices.remove(index)
        key = set(curr_feature_indices)
        if key not in curr_results:
            curr_results[key] = classifier_for_given_splits_and_features(curr_feature_indices, splits, positive_class)
            print '(removing)', key, curr_results[key]


def climb_up(classifier_for_given_splits_and_features, all_feature_indices, splits, current_results):
    while True:
        sorted_results = sorted(current_results.iteritems(), key=lambda x: individual_fitness(x[1])[0])
        ch = random.choice([0, 1, 2])
        if ch == 0:
            i = 0
        elif ch == 1:
            i = random.choice(range(1, 10))
        else:
            i = random.choice(range(10, 100))
        append_feature(all_feature_indices, splits, 0, sorted_results[i][0], current_results, '%sres.dump' % res_dir)
        if not random.choice(range(20)):
            pickle.dump(curr_results, open('%sres_copy.dump' % res_dir, 'wb'))
            pickle.dump(curr_results, open('%sres.dump' % res_dir, 'wb'))


def main(): pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Genetic algorithms for feature selection")
    parser.add_argument('res_dir', action='store', type=str, help='directory containing results')
    parser.add_argument('--infile', action='store', type=str, default=os.path.join('..', 'datasets', 'splits.dump'),
                        help='input file containing cv splits')
    parser.add_argument('--classifier', action='store', type=str, default='qda', choices=['qda', 'knn'],
                        help='classifier (either knn or qda)')
    parser.add_argument('--neighbours', action='store', type=int, default=0,
                        help='Number od nearest neighbours. Valid only for knn classifier.')
    args = parser.parse_args()
    splits = pickle.load(open(args.infile))
    num_of_possible_features = len(splits[0][0][0])
    if args.classifier == 'qda':
        classifier_func = qda_for_given_splits_and_features
    elif args.classifier == 'knn':
        def is_odd(number):
            return number % 2


        if args.neighbours < 1 or not is_odd(args.neighbours):
            raise Exception("Number of neighbours has to be an odd number >= 1")
        classifier_func = rpartial(knn_for_given_splits_and_features, args.neighbours)

    num_of_features = splits[0][0].shape[1]
    initial_results = check_initial_combinations(classifier_func, range(num_of_features), splits)

