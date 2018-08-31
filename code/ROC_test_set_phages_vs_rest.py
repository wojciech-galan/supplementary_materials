#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import cPickle as pickle
import numpy as np
from matplotlib import pyplot as plt
from lib import get_best_params_for_selectkbest
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from check_new_viruses_analyze_results import compute_fpr_tpr_auc


def get_fpr_tpr_auc_for_classifier_and_feats(classifier_name, params, features, x_learn, x_test, y_learn, y_test):
    classifier = eval('%s(**params)' % classifier_name)
    classifier.fit(x_learn[:, features], y_learn)
    probas = classifier.predict_proba(x_test[:, features])
    return compute_fpr_tpr_auc(y_test, probas[:, 1])


if __name__ == '__main__':
    attributes_learn = pickle.load(open(os.path.join('..', 'datasets', 'attributes_learn.dump')))
    attributes_test = pickle.load(open(os.path.join('..', 'datasets', 'attributes_test.dump')))
    classes_learn = pickle.load(open(os.path.join('..', 'datasets', 'classes_learn.dump')))
    classes_test = pickle.load(open(os.path.join('..', 'datasets', 'classes_test.dump')))
    num_of_feats = attributes_learn.shape[1]
    np.random.seed(77)

    image_data_path = os.path.join('..', 'datasets', 'ROC_test_set_phages_vs_rest.dump')
    if not os.path.exists(image_data_path):
        from feature_selection_for_svc import scorer_function
        from best_features_and_params import lasso_features
        from best_features_and_params import svc_biogram_best_c, svc_biogram_best_features
        from best_features_and_params import svc_penalized_best_c, svc_penalized_best_features
        from best_features_and_params import qda_bottomup_best_features
        from best_features_and_params import feats_ga_knn_500
        from best_features_and_params import feats_bottomup_knn
        from best_features_and_params import feats_ga_qda_500

        # svm_RFE
        svc_RFE_results = pickle.load(open(os.path.join('..', 'svm_res', 'RFE.dump')))
        best_result = max(svc_RFE_results.items(), key=lambda item: item[1][0])
        svc_RFE_best_features = [i for i, b in enumerate(best_result[1][1].support_) if b]
        svc_RFE_best_C = best_result[1][1].estimator.C

        # svm_SelectKBest
        svc_SelectKBest_best_features, svc_RFE_SelectKBest_C = get_best_params_for_selectkbest(
            os.path.join('..', 'svm_res', 'grid_search.dump'))

        # Logistic Regression
        lr_fpr, lr_tpr, lr_auc = get_fpr_tpr_auc_for_classifier_and_feats('LogisticRegression', {}, range(num_of_feats),
                                                                          attributes_learn, attributes_test,
                                                                          classes_learn,
                                                                          classes_test)
        lr_lasso_fpr, lr_lasso_tpr, lr_lasso_auc = get_fpr_tpr_auc_for_classifier_and_feats('LogisticRegression', {},
                                                                                            lasso_features,
                                                                                            attributes_learn,
                                                                                            attributes_test,
                                                                                            classes_learn,
                                                                                            classes_test)
        # QDA
        qda_fpr, qda_tpr, qda_auc = get_fpr_tpr_auc_for_classifier_and_feats('QuadraticDiscriminantAnalysis', {},
                                                                             range(num_of_feats),
                                                                             attributes_learn, attributes_test,
                                                                             classes_learn,
                                                                             classes_test)
        qda_bottomup_fpr, qda_bottomup_tpr, qda_bottomup_auc = get_fpr_tpr_auc_for_classifier_and_feats(
            'QuadraticDiscriminantAnalysis', {},
            qda_bottomup_best_features,
            attributes_learn,
            attributes_test, classes_learn,
            classes_test)
        qda_ga_fpr, qda_ga_tpr, qda_ga_auc = get_fpr_tpr_auc_for_classifier_and_feats('QuadraticDiscriminantAnalysis',
                                                                                      {},
                                                                                      feats_ga_qda_500,
                                                                                      attributes_learn,
                                                                                      attributes_test, classes_learn,
                                                                                      classes_test)
        # kNN
        knn_fpr, knn_tpr, knn_auc = get_fpr_tpr_auc_for_classifier_and_feats('KNeighborsClassifier', {},
                                                                             range(num_of_feats),
                                                                             attributes_learn, attributes_test,
                                                                             classes_learn,
                                                                             classes_test)
        knn_bottomup_fpr, knn_bottomup_tpr, knn_bottomup_auc = get_fpr_tpr_auc_for_classifier_and_feats(
            'KNeighborsClassifier', {'n_neighbors': 9},
            feats_bottomup_knn[9],
            attributes_learn,
            attributes_test, classes_learn,
            classes_test)
        knn_ga_fpr, knn_ga_tpr, knn_ga_auc = get_fpr_tpr_auc_for_classifier_and_feats('KNeighborsClassifier',
                                                                                      {'n_neighbors': 9},
                                                                                      feats_ga_knn_500[9],
                                                                                      attributes_learn,
                                                                                      attributes_test, classes_learn,
                                                                                      classes_test)
        print feats_bottomup_knn[9]
        print feats_ga_knn_500[9]  # they are the same!
        # SVC
        svc_fpr, svc_tpr, svc_auc = get_fpr_tpr_auc_for_classifier_and_feats('SVC',
                                                                             {'kernel': 'linear', 'probability': True},
                                                                             range(num_of_feats),
                                                                             attributes_learn, attributes_test,
                                                                             classes_learn,
                                                                             classes_test)
        svc_rfe_fpr, svc_rfe_tpr, svc_rfe_auc = get_fpr_tpr_auc_for_classifier_and_feats('SVC', {'C': svc_RFE_best_C,
                                                                                                 'kernel': 'linear',
                                                                                                 'probability': True},
                                                                                         svc_RFE_best_features,
                                                                                         attributes_learn,
                                                                                         attributes_test, classes_learn,
                                                                                         classes_test)
        svc_kbest_fpr, svc_kbest_tpr, svc_kbest_auc = get_fpr_tpr_auc_for_classifier_and_feats('SVC', {
            'C': svc_RFE_SelectKBest_C, 'kernel': 'linear', 'probability': True},
                                                                                               svc_SelectKBest_best_features,
                                                                                               attributes_learn,
                                                                                               attributes_test,
                                                                                               classes_learn,
                                                                                               classes_test)
        svc_biogram_fpr, svc_biogram_tpr, svc_biogram_auc = get_fpr_tpr_auc_for_classifier_and_feats('SVC', {
            'C': svc_biogram_best_c, 'kernel': 'linear', 'probability': True},
                                                                                                     svc_biogram_best_features,
                                                                                                     attributes_learn,
                                                                                                     attributes_test,
                                                                                                     classes_learn,
                                                                                                     classes_test)
        svc_penalized_fpr, svc_penalized_tpr, svc_penalized_auc = get_fpr_tpr_auc_for_classifier_and_feats('SVC', {
            'kernel': 'linear', 'probability': True, 'C': svc_penalized_best_c},
                                                                                                           svc_penalized_best_features,
                                                                                                           attributes_learn,
                                                                                                           attributes_test,
                                                                                                           classes_learn,
                                                                                                           classes_test)
        res_dict = {x: eval(x) for x in
                    'lr_fpr, lr_tpr, lr_auc, lr_lasso_fpr, lr_lasso_tpr, lr_lasso_auc, qda_fpr, qda_tpr, qda_auc, qda_bottomup_fpr, qda_bottomup_tpr, qda_bottomup_auc, qda_ga_fpr, qda_ga_tpr, qda_ga_auc, knn_fpr, knn_tpr, knn_auc, knn_bottomup_fpr, knn_bottomup_tpr, knn_bottomup_auc, knn_ga_fpr, knn_ga_tpr, knn_ga_auc, svc_fpr, svc_tpr, svc_auc, svc_rfe_fpr, svc_rfe_tpr, svc_rfe_auc, svc_kbest_fpr, svc_kbest_tpr, svc_kbest_auc, svc_biogram_fpr, svc_biogram_tpr, svc_biogram_auc, svc_penalized_fpr, svc_penalized_tpr, svc_penalized_auc'.split(
                        ', ')}
        with open(image_data_path, 'w') as f:
            pickle.dump(res_dict, f)
    else:
        with open(image_data_path) as f:
            res_dict = pickle.load(f)

    f, axarr = plt.subplots(2, 2, figsize=(16, 12))
    axarr[0, 0].plot(res_dict['lr_fpr'], res_dict['lr_tpr'], label='LR  AUC = %0.3f' % res_dict['lr_auc'])
    axarr[0, 0].plot(res_dict['lr_lasso_fpr'], res_dict['lr_lasso_tpr'],
                     label='LR_Lasso AUC = %0.3f' % res_dict['lr_lasso_auc'])
    axarr[0, 0].plot([0, 1], [0, 1], 'r--', label='random')
    axarr[0, 0].set_title('LR')
    axarr[0, 0].legend(loc='lower right')
    axarr[0, 0].set_xlim([0, 0.6])
    axarr[0, 0].set_ylim([0.2, 1])
    axarr[0, 0].set_ylabel('True Positive Rate')
    axarr[0, 0].set_xlabel('False Positive Rate')
    axarr[0, 1].set_title('QDA')
    axarr[0, 1].plot(res_dict['qda_fpr'], res_dict['qda_tpr'], label='QDA AUC = %0.3f' % res_dict['qda_auc'])
    axarr[0, 1].plot(res_dict['qda_bottomup_fpr'], res_dict['qda_bottomup_tpr'],
                     label='QDA_bottom-up AUC = %0.3f' % res_dict['qda_bottomup_auc'])
    axarr[0, 1].plot(res_dict['qda_ga_fpr'], res_dict['qda_ga_tpr'], label='QDA GA AUC = %0.3f' % res_dict['qda_ga_auc'])
    axarr[0, 1].plot([0, 1], [0, 1], 'r--', label='random')
    axarr[0, 1].legend(loc='lower right')
    axarr[0, 1].set_xlim([0, 0.9])
    axarr[0, 1].set_ylim([0.2, 1])
    axarr[0, 1].set_ylabel('True Positive Rate')
    axarr[0, 1].set_xlabel('False Positive Rate')
    axarr[1, 0].set_title('kNN')
    axarr[1, 0].plot(res_dict['knn_fpr'], res_dict['knn_tpr'], label='kNN AUC = %0.3f' % res_dict['knn_auc'])
    axarr[1, 0].plot(res_dict['knn_bottomup_fpr'], res_dict['knn_bottomup_tpr'],
                     label='kNN bottom-up and GA AUC = %0.3f' % res_dict['knn_bottomup_auc'])
    axarr[1, 0].plot([0, 1], [0, 1], 'r--', label='random')
    axarr[1, 0].legend(loc='lower right')
    axarr[1, 0].set_xlim([0, 1])
    axarr[1, 0].set_ylim([0, 1])
    axarr[1, 0].set_ylabel('True Positive Rate')
    axarr[1, 0].set_xlabel('False Positive Rate')
    axarr[1, 1].set_title('SVC')
    axarr[1, 1].plot(res_dict['svc_fpr'], res_dict['svc_tpr'], label='SVC AUC = %0.3f' % res_dict['svc_auc'])
    axarr[1, 1].plot(res_dict['svc_rfe_fpr'], res_dict['svc_rfe_tpr'], label='SVC RFE AUC = %0.3f' % res_dict['svc_rfe_auc'])
    axarr[1, 1].plot(res_dict['svc_kbest_fpr'], res_dict['svc_kbest_tpr'],
                     label='SVC SelectKBest AUC = %0.3f' % res_dict['svc_kbest_auc'])
    axarr[1, 1].plot(res_dict['svc_biogram_fpr'], res_dict['svc_biogram_tpr'],
                     label='SVC biogram AUC = %0.3f' % res_dict['svc_biogram_auc'])
    axarr[1, 1].plot(res_dict['svc_penalized_fpr'], res_dict['svc_penalized_tpr'],
                     label='SVC penalizedSVM AUC = %0.3f' % res_dict['svc_penalized_auc'])
    axarr[1, 1].plot([0, 1], [0, 1], 'r--', label='random')
    axarr[1, 1].legend(loc='lower right')
    axarr[1, 1].set_xlim([0, 0.77])
    axarr[1, 1].set_ylim([0, 1])
    axarr[1, 1].set_ylabel('True Positive Rate')
    axarr[1, 1].set_xlabel('False Positive Rate')
    # Fine-tune figure; hide y ticks for right plots
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
    plt.savefig(os.path.join('..', 'figures', 'ROC_test_set_phages_vs_rest.svg'), bbox_inches='tight')
