#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import random
import datetime
import argparse
import itertools
import cPickle as pickle
import numpy as np
from funcy import rpartial
from joblib import Parallel, delayed
from ga_stuff import individual_fitness
from ml_stuff import knn_for_given_splits_and_features, qda_for_given_splits_and_features


def check_initial_combinations(classifier_for_given_splits_and_features, all_feature_indices, splits, positive_class,
                               processed_combinations, pool_of_workers):
    results = {}
    for i in range(1, 4):
        keys_to_be_processed = []
        for indices_combination in itertools.combinations(all_feature_indices, i):
            key = indices_to_number(set(indices_combination))
            if key not in processed_combinations:
                keys_to_be_processed.append(key)
        print len(keys_to_be_processed), "keys will be processed"
        processed_results = pool_of_workers(
            delayed(classifier_for_given_splits_and_features)(number_to_indices(key), splits, positive_class) for key in
            keys_to_be_processed)
        results.update({keys_to_be_processed[i]: processed_results[i] for i in range(len(keys_to_be_processed))})
        processed_combinations.update(keys_to_be_processed)
    return results


def append_feature(classifier_for_given_splits_and_features, all_feature_indices, splits, positive_class, curr_feature_indices, processed_keys, best_results, pool_of_workers):
    curr_feature_indices = curr_feature_indices[:]
    not_present_features_indices = [index for index in all_feature_indices if not index in curr_feature_indices]
    for feature_index in not_present_features_indices:
        curr_feature_indices.append(feature_index)
        key = indices_to_number(set(curr_feature_indices))
        if not key in processed_keys:
            best_results[key] = classifier_for_given_splits_and_features(curr_feature_indices, splits, positive_class)
            processed_keys.update(key)
            print "num of results before removing", len(best_results)
            remove_feature(curr_feature_indices, splits, positive_class, key, curr_results,
                           classifier_for_given_splits_and_features)
            print "num of results after removing", len(best_results)
            print number_to_indices(key), curr_results[key][-4:], datetime.datetime.now()


def remove_feature(classifier_for_given_splits_and_features, splits, positive_class, curr_feature_indices,
                   curr_results):
    keys_to_be_processed = []
    for index in curr_feature_indices:  # todo ale to mają być słownik + set
        indices = curr_feature_indices[:]
        indices.remove(index)
        key = indices_to_number(set(indices))
        if key not in curr_results:
            keys_to_be_processed.append(key)
    curr_results[key] = classifier_for_given_splits_and_features(curr_feature_indices, splits, positive_class)
    processed_results = pool_of_workers(
        delayed(classifier_for_given_splits_and_features)(number_to_indices(key), splits, positive_class) for key in
        keys_to_be_processed)
    results.update({keys_to_be_processed[i]: processed_results[i] for i in range(len(keys_to_be_processed))})
    processed_combinations.update(keys_to_be_processed)
    print '(removing)', key, curr_results[key]


def sort_results_according_to_values(results, function_to_process_values):
    return sorted(results.iteritems(), key=lambda x: function_to_process_values(x[1]), reverse=True)


def climb_up(classifier_for_given_splits_and_features, all_feature_indices, splits, positive_class, procesed_keys, current_results,
             max_best_results, pool_of_workers, res_dir):
    while True:
        sorted_results = dict(sort_results_according_to_values(current_results, individual_fitness)[:max_best_results])
        ch = random.choice([0, 1, 2])
        if ch == 0:
            i = 0
        elif ch == 1:
            i = random.choice(range(1, 10))
        else:
            i = random.choice(range(10, 100))
        length = len(sorted_results)# todo remove
        #               wszystkie indeksy                            konkretny zestaw cech                    klucze
        #(classifier_for_given_splits_and_features, all_feature_indices, splits, positive_class, curr_feature_indices, processed_keys, best_results, pool_of_workers)
        append_feature(classifier_for_given_splits_and_features, all_feature_indices, splits, positive_class, number_to_indices(sorted_results[i][0]), procesed_keys, sorted_results, pool_of_workers) # todo sprawdzic argumenty
        assert len(sorted_results) > length
        print len(sorted_results), length
        #todo zapis
        if not random.choice(range(20)):
            save_results(sorted_results, res_dir, procesed_keys)


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


def save_results(results, directory, keys=None):
    # pattern for results filename:results.dump_%d, results_copy.dump_%d
    last_result_num = determine_last_result_num(directory)
    if last_result_num is None:
        result_num = 0
    else:
        result_num = last_result_num + 1
    with open(os.path.join(directory, 'results.dump_%d' % result_num), 'w') as f:
        pickle.dump(results, f)
    with open(os.path.join(directory, 'results_copy.dump_%d' % result_num), 'w') as f:
        pickle.dump(results, f)
    if keys == None:
        keys = results.keys()
    with open(os.path.join(directory, 'keys.dump'), 'w') as f:
        pickle.dump(keys, f)
    with open(os.path.join(directory, 'keys_copy.dump'), 'w') as f:
        pickle.dump(keys, f)


def load_results(directory):
    print "Loading results from disk"
    last_results_num = determine_last_result_num(directory)
    if last_results_num is None:
        print "no results"
        return {}, set()
    try:
        with open(os.path.join(directory, "results.dump_%d" % last_results_num)) as f:
            results = pickle.load(f)
    except:
        print "cannot load original result file, loading copy"
        os.remove(os.path.join(directory, "results.dump_%d" % last_results_num))
        try:
            with open(os.path.join(directory, "results_copy.dump_%d" % last_results_num)) as f:
                results = pickle.load(f)
        except:
            print "cannot load copy, loading prev results"
            with open(os.path.join(directory, "results.dump_%d" % last_results_num - 1)) as f:
                results = pickle.load(f)
    try:
        keys = pickle.load(open(os.path.join(directory, 'keys.dump')))
    except:
        keys = pickle.load(open(os.path.join(directory, 'keys_copy.dump')))
    return results, keys


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Genetic algorithms for feature selection")
    parser.add_argument('res_dir', action='store', type=str, help='directory containing results')
    parser.add_argument('n_proc', action='store', type=int, help='number_of_additional_processes')
    parser.add_argument('--infile', action='store', type=str, default=os.path.join('..', 'datasets', 'splits.dump'),
                        help='input file containing cv splits')
    parser.add_argument('--classifier', action='store', type=str, default='qda', choices=['qda', 'knn'],
                        help='classifier (either knn or qda)')
    parser.add_argument('--neighbours', action='store', type=int, default=0,
                        help='Number od nearest neighbours. Valid only for knn classifier.')
    args = parser.parse_args()
    splits = pickle.load(open(args.infile))
    num_of_possible_features = len(splits[0][0][0])
    num_of_best_res_after_sort = 100
    if args.classifier == 'qda':
        classifier_func = qda_for_given_splits_and_features
    elif args.classifier == 'knn':
        def is_odd(number):
            return number % 2


        if args.neighbours < 1 or not is_odd(args.neighbours):
            raise Exception("Number of neighbours has to be an odd number >= 1")
        classifier_func = rpartial(knn_for_given_splits_and_features, args.neighbours)

    num_of_features = splits[0][0].shape[1]
    results, keys = load_results(args.res_dir)
    # print results, keys
    print len(results), len(keys)
    import time
    t = time.time()
    pos_class_num = 0
    with Parallel(n_jobs=args.n_proc) as parallel:
        initial_results = check_initial_combinations(classifier_func, range(num_of_features), splits, pos_class_num, keys, parallel)
        results.update(initial_results)
        save_results(results, args.res_dir, keys)
        results = dict(sort_results_according_to_values(results, individual_fitness)[:num_of_best_res_after_sort])
        print len(results), len(keys)
        print time.time() -t
    raise
    climb_up(classifier_func, range(num_of_features), splits, pos_class_num, keys, results, num_of_best_res_after_sort, None, args.res_dir) #TODO wrzucić to do context managera
    # TODO zmienić None na parallel
