#! /usr/bin/python
# -*- coding: utf-8 -*-


import os
import json_tricks
import matplotlib.pyplot as plt


class Res(object):
    def __init__(self, error, length, lr_res, svm_res, knn_res, qda_res):
        self.error = float(error)
        self.length = int(length)
        self.lr_auc = float(lr_res[0])
        self.lr_mcc = float(lr_res[1])
        self.svm_auc = float(svm_res[0])
        self.svm_mcc = float(svm_res[1])
        self.knn_auc = float(knn_res[0])
        self.knn_mcc = float(knn_res[1])
        self.qda_auc = float(qda_res[0])
        self.qda_mcc = float(qda_res[1])


def create_basic_image(random_label):
    f, axarr = plt.subplots(1, 2, figsize=(10, 5))
    axarr[0].semilogx(lengths, lr_aucs, '-o', label='LR')
    axarr[0].semilogx(lengths, svm_aucs, '-x', label='SVC')
    axarr[0].semilogx(lengths, knn_aucs, '-v', label='kNN')
    axarr[0].semilogx(lengths, qda_aucs, '-s', label='QDA')
    axarr[0].semilogx([100, 10 ** 4], [0.5, 0.5], 'r--', label=random_label)
    axarr[1].semilogx(lengths, lr_aucs_, '-o', label='LR')
    axarr[1].semilogx(lengths, svm_aucs_, '-x', label='SVC')
    axarr[1].semilogx(lengths, knn_aucs_, '-v', label='kNN')
    axarr[1].semilogx(lengths, qda_aucs_, '-s', label='QDA')
    axarr[1].semilogx([100, 10 ** 4], [0.5, 0.5], 'r--', label=random_label)
    axarr[0].set_ylabel('AUC', fontsize=axes_label_fontsize)
    for x in range(2):
        axarr[x].set_xlim([90, 11000])
        axarr[x].set_ylim([0.48, 1])
        axarr[x].legend(loc='upper left', fontsize=12)
    f.subplots_adjust(wspace=0.07)
    return f, axarr


simulated_metagenomics_all_results_path = os.path.join('..', 'datasets', 'results_for_stimulated_metagenomics.json')
with open(simulated_metagenomics_all_results_path) as handle:
    res_dict = {eval(key): value for key, value in json_tricks.load(handle).iteritems()}

results = [Res(k[0], k[1], (v['auc_lr'], v['mcc_lr']), (v['auc_svm'], v['mcc_svm']), (v['auc_knn'], v['mcc_knn']),
               (v['auc_qda'], v['mcc_qda'])) for k, v in sorted(res_dict.iteritems())]
print results[0].__dict__
print results[6].__dict__
lr_aucs = [r.lr_auc for r in results if r.error == 0]
svm_aucs = [r.svm_auc for r in results if r.error == 0]
knn_aucs = [r.knn_auc for r in results if r.error == 0]
qda_aucs = [r.qda_auc for r in results if r.error == 0]
lr_aucs_ = [r.lr_auc for r in results if r.error != 0]
svm_aucs_ = [r.svm_auc for r in results if r.error != 0]
knn_aucs_ = [r.knn_auc for r in results if r.error != 0]
qda_aucs_ = [r.qda_auc for r in results if r.error != 0]
lengths = [r.length for r in results if r.error == 0]

axes_label_fontsize = 12
title_fontsize = 14
f, axarr = create_basic_image('random')
axarr[0].set_title('Substitution rate = 0')
axarr[1].set_title('Substitution rate = 0.02')
axarr[0].set_xlabel('Fragment length', fontsize=axes_label_fontsize)
axarr[1].set_xlabel('Fragment length', fontsize=axes_label_fontsize)
plt.setp(axarr, xticks=lengths, xticklabels=lengths)
plt.setp(axarr[1].get_yticklabels(), visible=False)
plt.savefig(os.path.join('..', 'figures', 'check_simmulated_metagenomics_auc_as_a_function_of_len.eps'),
            bbox_inches='tight')

# the same as above, but in polish
f, axarr = create_basic_image('klasyfikator losowy')
axarr[0].set_title(u'Odsetek podstawionych nukleotydów = 0')
axarr[1].set_title(u'Odsetek podstawionych nukleotydów = 0.02')
axarr[0].set_xlabel(u'Długość fragmentu', fontsize=axes_label_fontsize)
axarr[1].set_xlabel(u'Długość fragmentu', fontsize=axes_label_fontsize)
plt.setp(axarr, xticks=lengths, xticklabels=lengths)
plt.setp(axarr[1].get_yticklabels(), visible=False)
plt.savefig(os.path.join('..', 'figures', 'check_simmulated_metagenomics_auc_as_a_function_of_len_polish.png'),
            bbox_inches='tight')
