#! /usr/bin/python
# -*- coding: utf-8 -*-

from check_new_viruses import Result

import os
import cPickle as pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import binarize
from lib import compute_fpr_tpr_auc

CLASS_TO_NUM = {'Eucaryota-infecting': 0, 'phage': 1}


def evaluate_string(string):
    try:
        return eval(string)
    except SyntaxError:
        print 'Cannot evaluate string' # sometimes string is empty


if __name__ == '__main__':
    results = pickle.load(open(os.path.join('..', 'datasets', 'check_new_viruses_results.dump')))
    results = [res for res in results if evaluate_string(res.lr_proba)]

    proper = [CLASS_TO_NUM[result.proper] for result in results]
    predicted_lr = [evaluate_string(result.lr_proba)['phage'] for result in results]
    predicted_svm = [evaluate_string(result.svm_proba)['phage'] for result in results]
    predicted_knn = [evaluate_string(result.knn_proba)['phage'] for result in results]
    predicted_qda = [evaluate_string(result.qda_proba)['phage'] for result in results]

    print 'AUC LR  %0.4f, MCC LR  %0.4f' % (
    roc_auc_score(proper, predicted_lr), matthews_corrcoef(proper, binarize([predicted_lr], 0.5)[0]))
    print 'AUC SVM %0.4f, MCC SVM %0.4f' % (
    roc_auc_score(proper, predicted_svm), matthews_corrcoef(proper, binarize([predicted_svm], 0.5)[0]))
    print 'AUC KNN %0.4f, MCC KNN %0.4f' % (
    roc_auc_score(proper, predicted_knn), matthews_corrcoef(proper, binarize([predicted_knn], 0.5)[0]))
    print 'AUC QDA %0.4f, MCC QDA %0.4f' % (
    roc_auc_score(proper, predicted_qda), matthews_corrcoef(proper, binarize([predicted_qda], 0.5)[0]))

    fpr_tpr_dict = {}
    plt.title('Receiver Operating Characteristic')
    fpr, tpr, auc = compute_fpr_tpr_auc(proper, predicted_lr)
    fpr_tpr_dict['fpr_lr'] = fpr
    fpr_tpr_dict['tpr_lr'] = tpr
    fpr_tpr_dict['auc_lr'] = auc
    plt.plot(fpr, tpr, label='LR  AUC = %0.3f' % auc)
    fpr, tpr, auc = compute_fpr_tpr_auc(proper, predicted_svm)
    fpr_tpr_dict['fpr_svc'] = fpr
    fpr_tpr_dict['tpr_svc'] = tpr
    fpr_tpr_dict['auc_svc'] = auc
    plt.plot(fpr, tpr, label='SVC AUC = %0.3f' % auc)
    fpr, tpr, auc = compute_fpr_tpr_auc(proper, predicted_knn)
    fpr_tpr_dict['fpr_knn'] = fpr
    fpr_tpr_dict['tpr_knn'] = tpr
    fpr_tpr_dict['auc_knn'] = auc
    plt.plot(fpr, tpr, label='kNN AUC = %0.3f' % auc)
    fpr, tpr, auc = compute_fpr_tpr_auc(proper, predicted_qda)
    fpr_tpr_dict['fpr_qda'] = fpr
    fpr_tpr_dict['tpr_qda'] = tpr
    fpr_tpr_dict['auc_qda'] = auc
    plt.plot(fpr, tpr, label='QDA AUC = %0.3f' % auc)
    plt.plot([0, 1], [0, 1], 'r--', label='random')
    plt.legend(loc='lower right')
    plt.xlim([0, 0.31])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(os.path.join('..', 'figures', 'check_new_viruses_auc.eps'), bbox_inches='tight')

    with open(os.path.join('..', 'datasets', 'new_viruses_fpr_tpr.dump'), 'w') as f:
        pickle.dump(fpr_tpr_dict, f)

    plt.title('Krzywe ROC')
    fpr = fpr_tpr_dict['fpr_lr']
    tpr = fpr_tpr_dict['tpr_lr']
    auc = fpr_tpr_dict['auc_lr']
    plt.plot(fpr, tpr, label='LR  AUC = %0.3f' % auc)
    fpr, tpr, auc = compute_fpr_tpr_auc(proper, predicted_svm)
    fpr = fpr_tpr_dict['fpr_svc']
    tpr = fpr_tpr_dict['tpr_svc']
    auc = fpr_tpr_dict['auc_svc']
    plt.plot(fpr, tpr, label='SVC AUC = %0.3f' % auc)
    fpr, tpr, auc = compute_fpr_tpr_auc(proper, predicted_knn)
    fpr = fpr_tpr_dict['fpr_knn']
    tpr = fpr_tpr_dict['tpr_knn']
    auc = fpr_tpr_dict['auc_knn']
    plt.plot(fpr, tpr, label='kNN AUC = %0.3f' % auc)
    fpr, tpr, auc = compute_fpr_tpr_auc(proper, predicted_qda)
    fpr = fpr_tpr_dict['fpr_qda']
    tpr = fpr_tpr_dict['tpr_qda']
    auc = fpr_tpr_dict['auc_qda']
    plt.plot(fpr, tpr, label='QDA AUC = %0.3f' % auc)
    plt.plot([0, 1], [0, 1], 'r--', label='klasyfikator losowy')
    plt.legend(loc='lower right')
    plt.xlim([0, 0.31])
    plt.ylim([0, 1])
    plt.ylabel(u'Czułość')
    plt.xlabel(u'1 - specyficzność')
    plt.savefig(os.path.join('..', 'figures', 'check_new_viruses_auc_polish.png'), bbox_inches='tight')

