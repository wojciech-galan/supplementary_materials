#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import random
import argparse
import itertools
import cPickle as pickle
import numpy as np
from funcy import rpartial
from joblib import Parallel, delayed
from ga_stuff import individual_fitness
from ml_stuff import knn_for_given_splits_and_features, qda_for_given_splits_and_features


def check_initial_combinations(classifier_for_given_splits_and_features, all_feature_indices, splits,
                               current_results={}):
    for i in range(1, 4):
        for indices_combination in itertools.combinations(all_feature_indices, i):
            key = frozenset(indices_combination)
            if key in current_results:
                continue
            current_results[key] = classifier_for_given_splits_and_features(indices_combination, splits, 0)
            # print current_results
    return current_results


def append_feature(all_feature_indices, splits, positive_class, curr_feature_indices, curr_results,
                   classifier_for_given_splits_and_features):
    # curr_feature_indices to powinien być set
    curr_feature_indices = curr_feature_indices[:]
    not_present_features_indices = [index for index in all_feature_indices if not index in curr_feature_indices]
    while not_present_features_indices:
        feature_index = random.choice(not_present_features_indices)
        curr_feature_indices.append(feature_index)
        key = frozenset(curr_feature_indices)
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
        key = frozenset(curr_feature_indices)
        if key not in curr_results:
            curr_results[key] = classifier_for_given_splits_and_features(curr_feature_indices, splits, positive_class)
            print '(removing)', key, curr_results[key]


def sort_results_according_to_values(results, function_to_process_values):
    return sorted(results.iteritems(), key=lambda x: function_to_process_values(x[1]), reverse=True)


def climb_up(classifier_for_given_splits_and_features, all_feature_indices, splits, current_results):
    while True:
        sorted_results = sort_results_according_to_values(current_results, individual_fitness)
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


def indices_to_number(list_of_indexes):
    return sum([2 ** index for index in list_of_indexes])


def number_to_indices(num):
    l = []
    while num:
        l.append(num % 2)
        num /= 2
    return [i for i, x in enumerate(l) if x]

def determine_last_result_num(directory):
    res_list = []
    for f in os.listdir(directory):
        if f.startswith('results.dump'):
            res_list.append(int(f.split('_')[1]))
    try:
        return max(res_list)
    except ValueError:  # res_list is empty
        return None

def save_results(results, directory, pool_of_workers, keys=None):
    if keys == None:
        keys = results.keys()
    numbers_from_corresponding_keys = pool_of_workers(delayed(indices_to_number)(key) for key in keys)
    with open(os.path.join(directory, 'keys.dump'), 'w') as f:
        pickle.dump(numbers_from_corresponding_keys, f)
    with open(os.path.join(directory, 'keys_copy.dump'), 'w') as f:
        pickle.dump(numbers_from_corresponding_keys, f)
    # pattern for results filename:results.dump_%d, results_copy.dump_%d
    last_result_num = determine_last_result_num(directory)
    if last_result_num is None:
        result_num = 0
    else:
        result_num = last_result_num + 1
    czy najpierw zapisywać klucze, czy wyniki?


def load_results(directory, pool_of_workers):

    print "Loading results from disk"
    last_results_num = determine_last_result_num(directory)
    if last_results_num is None:
        print "no results"
        return [], []
    try:
        with open(os.path.join(directory, "results.dump_%d"%last_results_num)) as f:
            results = pickle.load(f)
    except:
        print "cannot load original result file, loading copy"
        os.remove(os.path.join(directory, "results.dump_%d"%last_results_num))
        try:
            with open(os.path.join(directory, "results_copy.dump_%d"%last_results_num)) as f:
                results = pickle.load(f)
        except:
            print "cannot load copy, loading prev results"
            with open(os.path.join(directory, "results.dump_%d" % last_results_num-1)) as f:
                results = pickle.load(f)
    try:
        dumped_numbers = pickle.load(open(os.path.join(directory, 'keys.dump')))
    except:
        dumped_numbers = pickle.load(open(os.path.join(directory, 'keys_copy.dump')))
    keys = pool_of_workers(delayed(number_to_indices)(number) for number in dumped_numbers)
    return results, keys


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

    num = indices_to_number([1, 2, 4, 5, 7, 10])
    print num, number_to_indices(num)
    raise
    num_of_features = splits[0][0].shape[1]
    initial_results = check_initial_combinations(classifier_func, range(num_of_features), splits)
    print len(initial_results)
    print initial_results
