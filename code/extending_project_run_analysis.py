#! /usr/bin/python
# -*- coding: utf-8 -*-

import cPickle as pickle
import os
import psutil
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.svm import SVC


def transform_classes(classes, chosen_class):
    '''
    Transforms multiclass classification problem to a binary one
    :param classes:
    :param chosen_class:
    :return:
    '''
    classes = np.array(classes)
    classes[classes != chosen_class] = np.inf
    classes[classes == chosen_class] = 1
    classes[classes == np.inf] = 0
    return classes


def select_features(attributes, coefficient_of_features):
    feature_indices = np.where(coefficient_of_features != np.zeros(coefficient_of_features.shape))[0]
    return attributes[:, feature_indices]


def roc_auc_scorer(y_true, y_pred):
    return roc_auc_score(y_true, y_pred[:, 1])


if __name__ == '__main__':
    # loading data
    attributes_learn = pickle.load(open(os.path.join('..', 'datasets', 'extension_attributes_learn.dump')))
    attributes_test = pickle.load(open(os.path.join('..', 'datasets', 'extension_attributes_test.dump')))
    classes_learn = pickle.load(open(os.path.join('..', 'datasets', 'extension_classes_learn.dump')))
    classes_test = pickle.load(open(os.path.join('..', 'datasets', 'extension_classes_test.dump')))
    indices = pickle.load(open(os.path.join('..', 'datasets', 'extension_cv_indices.dump')))

    num_of_jobs = max(psutil.cpu_count() - 2, 1)
    random_state = 666
    c_range = 2 ** np.linspace(-5, 5, 11)
    lr = LogisticRegression(C=np.iinfo(np.int32).max, random_state=random_state)

    for class_num in range(4):
        print 'LR with Lasso'
        transformed_classes_learn = transform_classes(classes_learn, class_num)
        transformed_classes_test = transform_classes(classes_test, class_num)
        model = LogisticRegressionCV(Cs=c_range, cv=indices, scoring=make_scorer(roc_auc_scorer, needs_proba=True),
                                     solver='liblinear', random_state=random_state,
                                     penalty='l1', n_jobs=num_of_jobs)
        model.fit(attributes_learn, transformed_classes_learn)
        print 'C =', model.C_, np.where(model.coef_[0] != np.zeros(model.coef_[0].shape))[0]
        selected_attributes_train = select_features(attributes_learn, model.coef_[0])
        selected_attributes_test = select_features(attributes_test, model.coef_[0])
        lr.fit(selected_attributes_train, transformed_classes_learn)
        probas = lr.predict_proba(selected_attributes_test)
        cv_score = cross_val_score(lr, selected_attributes_train, transformed_classes_learn,
                                                 scoring=make_scorer(roc_auc_scorer, needs_proba=True),
                                                 n_jobs=num_of_jobs,
                                                 cv=indices)
        print class_num, cv_score, np.mean(cv_score), roc_auc_scorer(transformed_classes_test, probas)
        print 'SVC'
        svc_results = {}
        for C in c_range:
            print C
            estimator = SVC(C=C, kernel='linear', probability=True)
            selector = RFECV(estimator, cv=indices, scoring=make_scorer(roc_auc_scorer, needs_proba=True),
                             n_jobs=num_of_jobs)
            selector.fit(attributes_learn, transformed_classes_learn)
            support = selector.get_support(indices=True)
            selected_attributes_train = attributes_learn[:, support]
            result = cross_val_score(estimator, selected_attributes_train, transformed_classes_learn,
                                             scoring=make_scorer(roc_auc_scorer, needs_proba=True), n_jobs=num_of_jobs,
                                             cv=indices)
            svc_results[C] = (result, support)
        best = max(svc_results.items(), key=lambda x:np.mean(x[1][0]))
        best_c = best[0]
        best_cv_res, best_indices = best[1]
        print best_c, best_indices
        selected_attributes_train = attributes_learn[:, best_indices]
        selected_attributes_test = attributes_test[:, best_indices]
        svc = SVC(C=best_c, kernel='linear', probability=True)
        svc.fit(selected_attributes_train, transformed_classes_learn)
        probas = svc.predict_proba(selected_attributes_test)
        print class_num, best_cv_res, np.mean(best_cv_res), roc_auc_scorer(transformed_classes_test, probas)
        print '-----------------------------------------------------------'

# LR with Lasso
# C = [16.] [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  17  18
#   19  21  22  23  24  25  26  27  28  29  30  31  32  34  35  36  37  38
#   39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56
#   57  58  59  60  61  62  63  64  65  66  67  68  69  70  71  72  73  74
#   75  76  77  78  79  80  81  82  83  84  85  86  87  88  89  90  91  92
#   93  94  95  96  97  98  99 100]
# 0 [0.64302812 0.9505896  0.92217928 0.6112381  0.59316577] 0.7440401725526131 0.8235357122723943
# SVC
# 0.03125
# 0.0625
# 0.125
# 0.25
# 0.5
# 1.0
# 2.0
# 4.0
# 8.0
# 16.0
# 32.0
# 0.25 [ 0  2  3  4  8  9 12 13 14 15 16 18 19 20 22 23 24 25 26 27 28 29 30 31
#  32 33 34 35 36 37 39 41 42 43 44 45 46 48 49 50 53 54 57 59 60 61 63 66
#  67 68 70 71 75 76 77 78 79 83 85 87 91 92 93 94 95 96 99]
# 0 [0.61892134 0.9607501  0.93502692 0.59205079 0.58116449] 0.7375827295358635 0.7535233102204184
# -----------------------------------------------------------
# LR with Lasso
# C = [0.03125] [  0   8  13  16  20  22  26  27  28  29  30  31  37  39  40  41  45  46
#   51  60  61  62  63  69  70  71  73  74  78  79  81  82  83  84  91  94
#   96 100]
# 1 [0.84657319 0.82276811 0.81872704 0.72103234 0.65253233] 0.772326601117051 0.8140501076227966
# SVC
# 0.03125
# 0.0625
# 0.125
# 0.25
# 0.5
# 1.0
# 2.0
# 4.0
# 8.0
# 16.0
# 32.0
# 0.03125 [ 0  8 22 26 27 57]
# 1 [0.87019833 0.78255538 0.82230014 0.81260709 0.83072468] 0.8236771236671139 0.8090277104461984
# -----------------------------------------------------------
# LR with Lasso
# C = [0.0625] [ 0  1  4 10 12 13 14 20 21 22 26 27 37 38 40 44 45 46 48 49 50 54 55 57
#  58 60 61 62 63 66 67 68 69 71 74 76 77 82 84 86 87 89 92 93 94 97 99]
# 2 [0.57402771 0.79391104 0.85298381 0.57570874 0.74084362] 0.7074949837468876 0.6635284048357761
# SVC
# 0.03125
# 0.0625
# 0.125
# 0.25
# 0.5
# 1.0
# 2.0
# 4.0
# 8.0
# 16.0
# 32.0
# 0.03125 [ 0 20 21 26 27 48 93]
# 2 [0.59027103 0.72256928 0.86734557 0.72092399 0.78462781] 0.7371475337725257 0.6589079383759494
# -----------------------------------------------------------
# LR with Lasso
# C = [1.] [  0   2   3   4   7   8   9  10  13  14  18  20  22  26  27  29  30  31
#   32  33  34  35  36  37  38  39  41  42  44  45  46  47  48  49  50  51
#   52  53  54  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69
#   70  71  72  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88
#   89  90  91  92  93  94  96  97  98 100]
# 3 [0.83174609 0.78055288 0.74480685 0.77372249 0.73965142] 0.7740959446209442 0.7947207876536498
# SVC
# 0.03125
# 0.0625
# 0.125
# 0.25
# 0.5
# 1.0
# 2.0
# 4.0
# 8.0
# 16.0
# 32.0
# 0.125 [ 0  3  8  9 10 12 13 14 20 21 22 25 26 29 30 31 32 33 34 35 36 37 41 45
#  46 47 49 51 64 67 69 72 76 77 78 79 80 81 85 86 87 96 98]
# 3 [0.8599459  0.78268361 0.75845106 0.73483485 0.74198032] 0.7755791475093309 0.8288031167889824
# -----------------------------------------------------------