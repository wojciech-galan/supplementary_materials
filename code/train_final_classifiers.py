#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
import cPickle as pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from feature_selection_for_svc import scorer_function
from prepare_attributes_classes_ids import virus_to_attributes, test_eu, learn_eu, test_phages, learn_phages, attribs
from best_features_and_params import lasso_features, feats_ga_qda_500, feats_ga_knn_500


def train_classifier(classifier, attributes, classes, features):
    selected_attributes = attributes[:, features]
    classifier.fit(selected_attributes, classes)
    return classifier

def print_feature_names(all_feature_names, feature_numbers):
    print [all_feature_names[feat_num] for feat_num in feature_numbers]


if __name__ == '__main__':
    features = pickle.load(open(os.path.join('..', 'datasets', 'features.dump')))
    classes_learn = pickle.load(open(os.path.join('..', 'datasets', 'classes_learn.dump')))
    classes_test = pickle.load(open(os.path.join('..', 'datasets', 'classes_test.dump')))

    # preparing attributes
    attributes_learn = [virus_to_attributes(virus, attribs) for virus in learn_eu]
    attributes_learn.extend([virus_to_attributes(virus, attribs) for virus in learn_phages])
    attributes_learn = np.array(attributes_learn)
    attributes_test = [virus_to_attributes(virus, attribs) for virus in test_eu]
    attributes_test.extend([virus_to_attributes(virus, attribs) for virus in test_phages])
    attributes_test = np.array(attributes_test)

    attributes_all = np.concatenate((attributes_learn, attributes_test))
    classes_all = np.concatenate((classes_learn, classes_test))

    # scaling
    scaler = StandardScaler()
    attributes_all = scaler.fit_transform(attributes_all)
    joblib.dump(scaler, os.path.join('..', 'datasets', 'scaler.pkl'))

    # lr, lasso
    trained_lr = train_classifier(LogisticRegression(), attributes_all, classes_all, lasso_features)
    joblib.dump(trained_lr, os.path.join('..', 'datasets', 'classifier_lr.pkl'))
    print_feature_names(features, lasso_features)

    # svc, rfe
    svc_RFE_results = pickle.load(open(os.path.join('..', 'svm_res', 'RFE.dump')))
    best_result = max(svc_RFE_results.items(), key=lambda item: item[1][0])
    svc_RFE_best_features = [i for i, b in enumerate(best_result[1][1].support_) if b]
    svc_RFE_best_C = best_result[1][1].estimator.C
    trained_svc = train_classifier(SVC(C=svc_RFE_best_C, probability=True), attributes_all, classes_all, svc_RFE_best_features)
    joblib.dump(trained_svc, os.path.join('..', 'datasets', 'classifier_svc.pkl'))
    print_feature_names(features, svc_RFE_best_features)

    # qda, ga 500
    trained_qda = train_classifier(QuadraticDiscriminantAnalysis(), attributes_all, classes_all, feats_ga_qda_500)
    joblib.dump(trained_qda, os.path.join('..', 'datasets', 'classifier_qda.pkl'))
    print_feature_names(features, feats_ga_qda_500)

    # knn, ga 500
    trained_knn = train_classifier(KNeighborsClassifier(n_neighbors=9), attributes_all, classes_all, feats_ga_knn_500[9])
    joblib.dump(trained_knn, os.path.join('..', 'datasets', 'classifier_knn.pkl'))
    print_feature_names(features, feats_ga_knn_500[9])
