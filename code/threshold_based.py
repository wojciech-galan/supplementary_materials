#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import cPickle as pickle
import matplotlib.pyplot as plt
from check_new_viruses_analyze_results import compute_fpr_tpr_auc

class ThresholdBasedClassifier(object):
    '''
    Predicts on of two possible classes based on threshold
    '''

    def __init__(self, threshold=0, predict_when_below_threshold=0, possible_classes={0, 1}):
        assert predict_when_below_threshold in possible_classes
        assert len(possible_classes) == 2
        super(ThresholdBasedClassifier, self).__init__()
        self.threshold = threshold
        self.predict_when_below_threshold = predict_when_below_threshold
        self.possible_classes = possible_classes

    def predict(self, X_vector):
        def predict_one_object(x, threshold, predict_when_below_threshold, possible_classes):
            if x < threshold:
                return predict_when_below_threshold
            else:
                return list({predict_when_below_threshold} ^ possible_classes)[0]

        return [predict_one_object(x, self.threshold, self.predict_when_below_threshold, self.possible_classes) for x in
                X_vector]


def get_fpr_tpr_auc_one_feature(feature_num, x_learn, x_test, y_learn, y_test, threshold=0):
    '''Classifies an object to one of two classes based only on one feature'''
    tbc = ThresholdBasedClassifier(threshold)
    fpr, tpr, auc = compute_fpr_tpr_auc(y_test, tbc.predict(x_test[:, feature_num]))
    return fpr, tpr, auc

if __name__ == '__main__':
    attributes_learn = pickle.load(open(os.path.join('..', 'datasets', 'attributes_learn.dump')))
    attributes_test = pickle.load(open(os.path.join('..', 'datasets', 'attributes_test.dump')))
    classes_learn = pickle.load(open(os.path.join('..', 'datasets', 'classes_learn.dump')))
    classes_test = pickle.load(open(os.path.join('..', 'datasets', 'classes_test.dump')))
    tb_fpr, tb_tpr, tb_auc = get_fpr_tpr_auc_one_feature(0, attributes_learn, attributes_test, classes_learn,
                                                         classes_test)
