#! /usr/bin/python
# -*- coding: utf-8 -*-

import cPickle as pickle
import os
import glob
import re
from sklearn.metrics import roc_auc_score, matthews_corrcoef, roc_curve
from sklearn.preprocessing import binarize
import matplotlib.pyplot as plt
from check_new_viruses import Result
from check_new_viruses_analyze_results import evaluate_string
from check_new_viruses_analyze_results import CLASS_TO_NUM


def process_result_file(path):
    with open(path) as f:
        content = pickle.load(f)
    proper, predicted_lr, predicted_svm, predicted_knn, predicted_qda = get_proper_and_predicted_lr_svm_knn_qda(content)
    fpr_lr, tpr_lr, auc_lr, mcc_lr = compute_fpr_tpr_auc_mcc(proper, predicted_lr)
    fpr_svm, tpr_svm, auc_svm, mcc_svm = compute_fpr_tpr_auc_mcc(proper, predicted_svm)
    fpr_knn, tpr_knn, auc_knn, mcc_knn = compute_fpr_tpr_auc_mcc(proper, predicted_knn)
    fpr_qda, tpr_qda, auc_qda, mcc_qda = compute_fpr_tpr_auc_mcc(proper, predicted_qda)
    print 'AUC LR  %0.4f, MCC LR  %0.4f' % (auc_lr, mcc_lr)
    print 'AUC SVM %0.4f, MCC SVM %0.4f' % (auc_svm, mcc_svm)
    print 'AUC KNN %0.4f, MCC KNN %0.4f' % (auc_knn, mcc_knn)
    print 'AUC QDA %0.4f, MCC QDA %0.4f' % (auc_qda, mcc_qda)
    return fpr_lr, tpr_lr, auc_lr, fpr_svm, tpr_svm, auc_svm, fpr_knn, tpr_knn, auc_knn, fpr_qda, tpr_qda, auc_qda


def compute_fpr_tpr_auc_mcc(proper, predicted):
    fpr, tpr, _ = roc_curve(proper, predicted)
    auc = roc_auc_score(proper, predicted)
    mcc = matthews_corrcoef(proper, binarize([predicted], 0.5)[0])
    return fpr, tpr, auc, mcc


def get_err_rate_and_len_rate(regex, string):
    error_rate, length = re.findall(regex, string)[0]
    return float(error_rate), int(length)


def get_proper_and_predicted_lr_svm_knn_qda(result_iterable):
    proper = [CLASS_TO_NUM[result.proper] for result in result_iterable]
    predicted_lr = [evaluate_string(result.lr_proba)['phage'] for result in result_iterable]
    predicted_svm = [evaluate_string(result.svm_proba)['phage'] for result in result_iterable]
    predicted_knn = [evaluate_string(result.knn_proba)['phage'] for result in result_iterable]
    predicted_qda = [evaluate_string(result.qda_proba)['phage'] for result in result_iterable]
    return proper, predicted_lr, predicted_svm, predicted_knn, predicted_qda


def plot_results_in_one_subplot(subplot_obj, title, fpr_lr, tpr_lr, auc_lr, fpr_svm, tpr_svm, auc_svm, fpr_knn, tpr_knn,
                                auc_knn, fpr_qda, tpr_qda, auc_qda):
    subplot_obj.plot(fpr_lr, tpr_lr, label='LR  AUC = %0.3f' % auc_lr)
    subplot_obj.plot(fpr_svm, tpr_svm, label='SVM AUC = %0.3f' % auc_svm)
    subplot_obj.plot(fpr_knn, tpr_knn, label='kNN AUC = %0.3f' % auc_knn)
    subplot_obj.plot(fpr_qda, tpr_qda, label='QDA AUC = %0.3f' % auc_qda)
    subplot_obj.plot([0, 1], [0, 1], 'r--', label='random')
    subplot_obj.set_title(title)
    subplot_obj.legend(loc='lower right')
    subplot_obj.set_xlim([0, 1])
    subplot_obj.set_ylim([0, 1])
    subplot_obj.set_ylabel('True Positive Rate')
    subplot_obj.set_xlabel('False Positive Rate')


def comparator(a, b):
    if a[1][1] != b[1][1]:
        return a[1][1] - b[1][1]
    else:
        return int(100*(a[1][0] - b[1][0]))


def sort_paths(paths, regex):
    d = {path: get_err_rate_and_len_rate(regex, path) for path in paths}
    return [x[0] for x in sorted(d.items(), cmp=comparator)]


if __name__ == '__main__':
    num_of_columns = 2
    num_of_rows = 4
    f, axarr = plt.subplots(num_of_rows, num_of_columns, figsize=(15, 25))
    re_path = os.path.join('..', 'datasets', 'check_simmulated_metagenomics_results_(.+)_(\d+).dump')
    print sort_paths(glob.glob(os.path.join('..', 'datasets', 'check_simmulated_metagenomics_results_*')), re_path)
    for i, path in enumerate(sort_paths(glob.glob(os.path.join('..', 'datasets', 'check_simmulated_metagenomics_results_*')), re_path)):
        error_rate, length = get_err_rate_and_len_rate(re_path, path)
        print error_rate, length
        fpr_lr, tpr_lr, auc_lr, fpr_svm, tpr_svm, auc_svm, fpr_knn, tpr_knn, auc_knn, fpr_qda, tpr_qda, auc_qda = process_result_file(
            path)
        print i / num_of_columns, i % num_of_columns
        plot_results_in_one_subplot(axarr[i / num_of_columns, i % num_of_columns],
                                    'Sequence length = %d, error rate = %.2f' % (length, error_rate), fpr_lr, tpr_lr,
                                    auc_lr, fpr_svm, tpr_svm, auc_svm, fpr_knn, tpr_knn, auc_knn, fpr_qda, tpr_qda,
                                    auc_qda)
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
    #plt.setp([a.get_yticklabels() for a in axarr[:, 2]], visible=False)
    plt.setp([a.get_xticklabels() for a in axarr[2, :]], visible=False)
    plt.setp([a.get_xticklabels() for a in axarr[1, :]], visible=False)
    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    axarr[-1, -1].axis('off')
    plt.savefig(os.path.join('..', 'figures', 'check_simmulated_metagenomics.svg'), bbox_inches='tight')
