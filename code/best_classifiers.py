#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import glob
import functools
import cPickle as pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from feature_selection_bottom_up import number_to_indices
from lib import get_best_params_for_selectkbest
from ga_stuff import individual_fitness
from feature_selection_for_svc import scorer_function
from ml_stuff import generic_cv_for_given_splits_and_features
from ml_stuff import svc_for_given_splits_and_features
from ml_stuff import qda_for_given_splits_and_features
from ml_stuff import knn_for_given_splits_and_features
from ml_stuff import binary_classification_evaluation_extended
from best_features_and_params import lasso_c, lasso_features
from best_features_and_params import svc_biogram_best_c, svc_biogram_best_features
from best_features_and_params import svc_penalized_best_c, svc_penalized_best_features
from best_features_and_params import qda_bottomup_best_features
from best_features_and_params import feats_ga_knn
from best_features_and_params import feats_ga_knn_500
from best_features_and_params import feats_bottomup_knn
from best_features_and_params import feats_ga_qda
from best_features_and_params import feats_ga_qda_500


class Res(object):
    def __init__(self, method, features, params, cv_splits, pos_class, method_to_compute_cv_fitness,
                 x_learn, x_test, y_learn, y_test, description=None, seed=77):
        self.method = method
        self.features = features
        self.params = params
        np.random.seed(seed)
        cv_res = method_to_compute_cv_fitness(features, cv_splits, pos_class, **params)
        self.cv_fitness = individual_fitness(cv_res)
        # print cv_res
        # print self.cv_fitness
        np.random.seed(seed)
        classifier = eval('%s(**params)' % method)
        print classifier
        # print method_to_compute_cv_fitness(features, cv_splits, pos_class, **params)
        self.cv_res = generic_cv_for_given_splits_and_features(classifier, features, cv_splits, pos_class)
        # assert cv_res == self.cv_res
        # print self.cv_res
        classifier.fit(x_learn[:, features], y_learn)
        probas = classifier.predict_proba(x_test[:, features])
        # print zip(y_test, probas[:, 0])
        self.blind_res = binary_classification_evaluation_extended(y_test, probas, positive_class, classifier.classes_,
                                                                   fpr=True)
        self.description = description

    def __str__(self):
        return str(self.__dict__)


if __name__ == '__main__':
    cv_splits = pickle.load(open(os.path.join('..', 'datasets', 'splits.dump')))
    attributes_learn = pickle.load(open(os.path.join('..', 'datasets', 'attributes_learn.dump')))
    classes_learn = pickle.load(open(os.path.join('..', 'datasets', 'classes_learn.dump')))
    attributes_test = pickle.load(open(os.path.join('..', 'datasets', 'attributes_test.dump')))
    classes_test = pickle.load(open(os.path.join('..', 'datasets', 'classes_test.dump')))
    positive_class = 1
    results = []
    num_of_feats = cv_splits[0][0].shape[1]
    # simple LogisticRegression
    results.append(
        Res('LogisticRegression', range(num_of_feats), {}, cv_splits,
            positive_class,
            functools.partial(generic_cv_for_given_splits_and_features, LogisticRegression()),
            attributes_learn, attributes_test, classes_learn, classes_test,
            description='simple LogisticRegression'))
    # LogisticRegression with lasso- feature selection
    results.append(
        Res('LogisticRegression', lasso_features, {'C': lasso_c}, cv_splits, positive_class,
            functools.partial(generic_cv_for_given_splits_and_features, LogisticRegression(C=lasso_c)),
            attributes_learn, attributes_test, classes_learn, classes_test,
            description='LogisticRegression with lasso- feature selection'))
    # simple svm
    results.append(
        Res('SVC', range(num_of_feats), {'kernel': 'linear', 'probability': True}, cv_splits, positive_class,
            svc_for_given_splits_and_features, attributes_learn, attributes_test, classes_learn, classes_test,
            description='simple svc'))
    # simple kNN
    results.append(Res('KNeighborsClassifier', range(num_of_feats), {},
                       cv_splits, positive_class, knn_for_given_splits_and_features,
                       attributes_learn, attributes_test,
                       classes_learn, classes_test, description='simple kNN'))

    # simple QDA
    results.append(Res('QuadraticDiscriminantAnalysis', range(num_of_feats), {},
                       cv_splits, positive_class, qda_for_given_splits_and_features, attributes_learn,
                       attributes_test,
                       classes_learn, classes_test, description='simple QDA'))

    # svm_RFE
    svc_RFE_results = pickle.load(open(os.path.join('..', 'svm_res', 'RFE.dump')))
    best_result = max(svc_RFE_results.items(), key=lambda item: item[1][0])
    svc_RFE_best_features = [i for i, b in enumerate(best_result[1][1].support_) if b]
    svc_RFE_best_C = best_result[1][1].estimator.C
    results.append(
        Res('SVC', svc_RFE_best_features, {'C': svc_RFE_best_C, 'kernel': 'linear', 'probability': True}, cv_splits,
            positive_class, svc_for_given_splits_and_features, attributes_learn, attributes_test, classes_learn,
            classes_test, description='svm_RFE'))

    # svm_SelectKBest
    svc_SelectKBest_best_features, svc_RFE_SelectKBest_C = get_best_params_for_selectkbest(
        os.path.join('..', 'svm_res', 'grid_search.dump'))
    results.append(
        Res('SVC', svc_SelectKBest_best_features, {'C': svc_RFE_SelectKBest_C, 'kernel': 'linear', 'probability': True},
            cv_splits, positive_class, svc_for_given_splits_and_features, attributes_learn, attributes_test,
            classes_learn, classes_test, description='svm_SelectKBest'))

    # svm biogram
    results.append(
        Res('SVC', svc_biogram_best_features, {'C': svc_biogram_best_c, 'kernel': 'linear', 'probability': True},
            cv_splits, positive_class, svc_for_given_splits_and_features, attributes_learn, attributes_test,
            classes_learn, classes_test, description='svm_biogram'))

    # penalized svm
    results.append(
        Res('SVC', svc_biogram_best_features, {'kernel': 'linear', 'probability': True, 'C': svc_penalized_best_c},
            cv_splits, positive_class, svc_for_given_splits_and_features, attributes_learn, attributes_test,
            classes_learn, classes_test, description='svm_penalized'))

    # bottom up QDA
    results.append(
        Res('QuadraticDiscriminantAnalysis', qda_bottomup_best_features, {},
            cv_splits, positive_class, qda_for_given_splits_and_features, attributes_learn, attributes_test,
            classes_learn, classes_test, description='bottom up QDA'))

    # bottom up kNN
    for i in range(1, 10, 2):
        results.append(Res('KNeighborsClassifier', feats_bottomup_knn[i], {'n_neighbors': i},
                           cv_splits, positive_class, knn_for_given_splits_and_features, attributes_learn,
                           attributes_test,
                           classes_learn, classes_test, description='bottom up kNN'))

    # GA knn
    names = glob.glob(os.path.join('..', 'ga_res', 'knn', '*', '*'))
    ks = [int(name.rsplit('_', 2)[-2]) for name in names]
    for k in sorted(set(ks)):
        results.append(Res('KNeighborsClassifier', feats_ga_knn[k], {'n_neighbors': k},
                           cv_splits, positive_class, knn_for_given_splits_and_features, attributes_learn,
                           attributes_test, classes_learn, classes_test, description='GA knn 100'))

    names = glob.glob(os.path.join('..', 'ga_res', 'knn_500', '*', '*'))
    ks = [int(name.rsplit('_', 2)[-2]) for name in names]
    for k in sorted(set(ks)):
        results.append(Res('KNeighborsClassifier', feats_ga_knn_500[k], {'n_neighbors': k},
                           cv_splits, positive_class, knn_for_given_splits_and_features, attributes_learn,
                           attributes_test,
                           classes_learn, classes_test, description='GA knn 500, %d neighbours' % k))

    # GA qda
    results.append(Res('QuadraticDiscriminantAnalysis', feats_ga_qda, {},
                       cv_splits, positive_class, qda_for_given_splits_and_features, attributes_learn,
                       attributes_test,
                       classes_learn, classes_test, description='GA qda 100'))
    results.append(Res('QuadraticDiscriminantAnalysis', feats_ga_qda_500, {},
                       cv_splits, positive_class, qda_for_given_splits_and_features, attributes_learn,
                       attributes_test,
                       classes_learn, classes_test, description='GA qda 500'))

    for res in results:
        print res.description, res.cv_fitness[0]
        print res.cv_res, res.blind_res
        print '--------------------------------------------------'

# simple LogisticRegression 1.7058370026
# [0.80471537 0.13094322 0.94647732 0.05047953 0.89692308 0.07191199] (0.8893517313513327, 0.9837241332975007, 0.944672131147541)
# --------------------------------------------------
# LogisticRegression with lasso- feature selection 1.76776251503
# [0.83908334 0.10589763 0.9634977  0.03337646 0.91538462 0.05806433] (0.881214149649768, 0.9833546089760817, 0.9405737704918032)
# --------------------------------------------------
# simple svc 1.80934918635
# [0.80069186 0.12213433 0.93870874 0.05775479 0.89487179 0.06750947] (0.8852459016393442, 0.9827499328137597, 0.9426229508196722)
# --------------------------------------------------
# simple kNN 1.74486911885
# [0.69691884 0.12717094 0.92739776 0.0603668  0.82974359 0.07835589] (0.7937408962226535, 0.9565137059930126, 0.8934426229508197)
# --------------------------------------------------
# simple QDA 1.87238156491
# [0.85396711 0.12763589 0.96995924 0.03442696 0.92051282 0.07214202] (0.7831473308133717, 0.9515839156140822, 0.889344262295082)
# --------------------------------------------------
# svm_RFE 1.87009392493
# [0.84707059 0.08402142 0.96517554 0.03245694 0.92051282 0.04534911] (0.8524876545946632, 0.9809359043267939, 0.9262295081967213)
# --------------------------------------------------
# svm_SelectKBest 1.84347436991
# [0.82494944 0.09448447 0.95317817 0.04147185 0.90769231 0.05297215] (0.877314466009143, 0.9842196318194034, 0.9385245901639344)
# --------------------------------------------------
# svm_biogram 1.84809158484
# [0.8302313  0.08352212 0.95408284 0.04266656 0.91025641 0.04711255] (0.8114754098360656, 0.9717649825315775, 0.9057377049180327)
# --------------------------------------------------
# svm_penalized 1.84050672102
# [0.81862072 0.09042821 0.95319921 0.04279729 0.90358974 0.05165505] (0.8239435026742628, 0.973998925020156, 0.9118852459016393)
# --------------------------------------------------
# bottom up QDA 1.94885300517
# [0.94490988 0.03898014 0.98594346 0.0186915  0.97179487 0.02025479] (0.8383498716702922, 0.9664572695511959, 0.9180327868852459)
# --------------------------------------------------
# bottom up kNN 1.85735494711
# [0.8833471  0.08106968 0.9374359  0.04648316 0.9374359  0.04648316] (0.8984544500914721, 0.9487704918032787, 0.9487704918032787)
# --------------------------------------------------
# bottom up kNN 1.89004855892
# [0.879614   0.08713009 0.96751874 0.02554366 0.93538462 0.05033484] (0.8567372739249797, 0.9558082504703036, 0.9282786885245902)
# --------------------------------------------------
# bottom up kNN 1.79424694526
# [0.75979126 0.16401616 0.9474188  0.05026187 0.86461538 0.10130509] (0.7895420339517228, 0.9549600241870465, 0.8934426229508197)
# --------------------------------------------------
# bottom up kNN 1.89947441858
# [0.87640392 0.0750631  0.97526101 0.01842267 0.93435897 0.04218241] (0.877167055361037, 0.9784080220370868, 0.9385245901639344)
# --------------------------------------------------
# bottom up kNN 1.90481140889
# [0.8775582  0.0772424  0.97960026 0.01565062 0.93487179 0.04368316] (0.873545260423432, 0.9811206664875034, 0.9364754098360656)
# --------------------------------------------------
# GA knn 100 1.86224919238
# [0.88172892 0.06299493 0.93794872 0.03475849 0.93794872 0.03475849] (0.8649357167376589, 0.9323770491803278, 0.9323770491803278)
# --------------------------------------------------
# GA knn 100 1.88866935097
# [0.87379781 0.08828387 0.96804734 0.02096563 0.93230769 0.05083394] (0.8606846512734581, 0.9565137059930127, 0.930327868852459)
# --------------------------------------------------
# GA knn 100 1.8934754997
# [0.87622086 0.07738122 0.97000131 0.01985437 0.93435897 0.04350217] (0.8569100816064156, 0.9694974469228701, 0.9282786885245902)
# --------------------------------------------------
# GA knn 100 1.89947441858
# [0.87640392 0.0750631  0.97526101 0.01842267 0.93435897 0.04218241] (0.877167055361037, 0.9784080220370868, 0.9385245901639344)
# --------------------------------------------------
# GA knn 100 1.90481140889
# [0.8775582  0.0772424  0.97960026 0.01565062 0.93487179 0.04368316] (0.873545260423432, 0.9811206664875034, 0.9364754098360656)
# --------------------------------------------------
# GA knn 100 1.90076284849
# [0.86819905 0.07955639 0.98052071 0.01565135 0.92974359 0.04527947] (0.8953697055304286, 0.9804656006449879, 0.9467213114754098)
# --------------------------------------------------
# GA knn 100 1.89775305241
# [0.86413218 0.09068788 0.98122814 0.01682076 0.92717949 0.05201021] (0.8920528817818068, 0.9854793738242408, 0.944672131147541)
# --------------------------------------------------
# GA knn 100 1.89826473524
# [0.85995403 0.09049712 0.9826693  0.01227775 0.92410256 0.05266345] (0.824942172000179, 0.981313826928245, 0.9118852459016393)
# --------------------------------------------------
# GA knn 500, 7 neighbours 1.89947441858
# [0.87640392 0.0750631  0.97526101 0.01842267 0.93435897 0.04218241] (0.877167055361037, 0.9784080220370868, 0.9385245901639344)
# --------------------------------------------------
# GA knn 500, 9 neighbours 1.90481140889
# [0.8775582  0.0772424  0.97960026 0.01565062 0.93487179 0.04368316] (0.873545260423432, 0.9811206664875034, 0.9364754098360656)
# --------------------------------------------------
# GA knn 500, 11 neighbours 1.90076284849
# [0.86819905 0.07955639 0.98052071 0.01565135 0.92974359 0.04527947] (0.8953697055304286, 0.9804656006449879, 0.9467213114754098)
# --------------------------------------------------
# GA knn 500, 13 neighbours 1.89973792762
# [0.86835251 0.09100684 0.98118869 0.01700467 0.92974359 0.05147655] (0.848709985131713, 0.9789119188390217, 0.9241803278688525)
# --------------------------------------------------
# GA knn 500, 15 neighbours 1.89785695696
# [0.86393306 0.07709575 0.97950033 0.01589173 0.92717949 0.04428109] (0.8335437792420599, 0.9764680193496373, 0.9159836065573771)
# --------------------------------------------------
# GA qda 100 1.95343454705
# [0.95556637 0.04828545 0.98645628 0.01907695 0.97692308 0.02517526] (0.8531758384844114, 0.972352862133835, 0.9262295081967213)
# --------------------------------------------------
# GA qda 500 1.95409342574
# [0.95745555 0.04610066 0.98594083 0.01925039 0.97794872 0.02397676] (0.8581805411901011, 0.9700013437248053, 0.9282786885245902)
# --------------------------------------------------
