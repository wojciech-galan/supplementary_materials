#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import csv
import cPickle as pickle
import numpy as np


def read_dump_write_newline_separated(dump_path, out_path):
    with open(dump_path) as f:
        data = pickle.load(f)
    np.savetxt(out_path, data, fmt='%s')


def process_cv_indices(dump_path, out_path):
    with open(dump_path) as f:
        data = pickle.load(f)
    with open(out_path, 'w') as f_out:
        for split_num, split in enumerate(data):
            f_out.write('cv_split number: %d%s' % (split_num, os.linesep))
            f_out.write('train set indices: %s%s' % (', '.join([str(x) for x in split[0]]), os.linesep))
            f_out.write('test set indices: %s%s' % (', '.join([str(x) for x in split[1]]), os.linesep))
            f_out.write('---------------------------------------------------%s'% os.linesep)


read_dump_write_newline_separated(os.path.join('..', 'datasets', 'ids_learn.dump'),
                                  os.path.join('..', 'datasets', 'ids_learn'))
read_dump_write_newline_separated(os.path.join('..', 'datasets', 'ids_test.dump'),
                                  os.path.join('..', 'datasets', 'ids_test'))
read_dump_write_newline_separated(os.path.join('..', 'datasets', 'extension_ids_learn.dump'),
                                  os.path.join('..', 'datasets', 'extension_ids_learn'))
read_dump_write_newline_separated(os.path.join('..', 'datasets', 'extension_ids_test.dump'),
                                  os.path.join('..', 'datasets', 'extension_ids_test'))

process_cv_indices(os.path.join('..', 'datasets', 'cv_indices.dump'), os.path.join('..', 'datasets', 'cv_indices'))
process_cv_indices(os.path.join('..', 'datasets', 'extension_cv_indices.dump'),
                   os.path.join('..', 'datasets', 'extension_cv_indices'))
