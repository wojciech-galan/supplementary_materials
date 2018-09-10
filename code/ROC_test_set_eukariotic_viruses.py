#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import cPickle as pickle
import numpy as np
from matplotlib import pyplot as plt
from ROC_test_set_phages_vs_rest import get_fpr_tpr_auc_for_classifier_and_feats
from extending_project_run_analysis import transform_classes


def feat_str_to_indices(feat_string):
    return np.array(feat_string.split(), dtype=int)


if __name__ == '__main__':
    attributes_learn = pickle.load(open(os.path.join('..', 'datasets', 'extension_attributes_learn.dump')))
    attributes_test = pickle.load(open(os.path.join('..', 'datasets', 'extension_attributes_test.dump')))
    classes_learn = pickle.load(open(os.path.join('..', 'datasets', 'extension_classes_learn.dump')))
    classes_test = pickle.load(open(os.path.join('..', 'datasets', 'extension_classes_test.dump')))
    np.random.seed(77)
    seed_plants_feat_ind_lr = feat_str_to_indices('''0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  17  18
  19  21  22  23  24  25  26  27  28  29  30  31  32  34  35  36  37  38
  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56
  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71  72  73  74
  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89  90  91  92
  93  94  95  96  97  98  99 100''')
    vertebrates_feat_ind_lr = feat_str_to_indices('''0   8  13  16  20  22  26  27  28  29  30  31  37  39  40  41  45  46
  51  60  61  62  63  69  70  71  73  74  78  79  81  82  83  84  91  94
  96 100''')
    arthropods_feat_ind_lr = feat_str_to_indices(''' 0  1  4 10 12 13 14 20 21 22 26 27 37 38 40 44 45 46 48 49 50 54 55 57
 58 60 61 62 63 66 67 68 69 71 74 76 77 82 84 86 87 89 92 93 94 97 99''')
    seed_plants_feat_ind_svc = feat_str_to_indices('''0  2  3  4  8  9 12 13 14 15 16 18 19 20 22 23 24 25 26 27 28 29 30 31
 32 33 34 35 36 37 39 41 42 43 44 45 46 48 49 50 53 54 57 59 60 61 63 66
 67 68 70 71 75 76 77 78 79 83 85 87 91 92 93 94 95 96 99''')
    vertebrates_feat_ind_svc = feat_str_to_indices('''0  8 22 26 27 57''')
    arthropods_feat_ind_svc = feat_str_to_indices('''0 20 21 26 27 48 93''')
    seed_plants_c_svc = .25
    vertebrates_c_svc = 0.03125
    arthropods_c_svc = 0.03125
    fpr_plants_lr, tpr_plants_lr, auc_plants_lr = get_fpr_tpr_auc_for_classifier_and_feats('LogisticRegression', {},
                                                                                           seed_plants_feat_ind_lr,
                                                                                           attributes_learn,
                                                                                           attributes_test,
                                                                                           transform_classes(
                                                                                               classes_learn, 0), transform_classes(classes_test, 0))
    fpr_plants_svc, tpr_plants_svc, auc_plants_svc = get_fpr_tpr_auc_for_classifier_and_feats('SVC',
                                                                                              {'kernel': 'linear',
                                                                                               'probability': True,
                                                                                               'C': seed_plants_c_svc},
                                                                                              seed_plants_feat_ind_svc,
                                                                                              attributes_learn,
                                                                                              attributes_test,
                                                                                              transform_classes(
                                                                                                  classes_learn, 0),
                                                                                              transform_classes(
                                                                                                  classes_test, 0))
    fpr_vertebrates_lr, tpr_vertebrates_lr, auc_vertebrates_lr = get_fpr_tpr_auc_for_classifier_and_feats(
        'LogisticRegression', {}, vertebrates_feat_ind_lr, attributes_learn, attributes_test, transform_classes(classes_learn, 1), transform_classes(classes_test, 1))
    fpr_vertebrates_svc, tpr_vertebrates_svc, auc_vertebrates_svc = get_fpr_tpr_auc_for_classifier_and_feats(
        'SVC', {'kernel': 'linear', 'probability': True, 'C': vertebrates_c_svc},
        vertebrates_feat_ind_svc, attributes_learn, attributes_test, transform_classes(classes_learn, 1), transform_classes(classes_test, 1))
    fpr_arthropods_lr, tpr_arthropods_lr, auc_arthropods_lr = get_fpr_tpr_auc_for_classifier_and_feats(
        'LogisticRegression', {}, arthropods_feat_ind_lr, attributes_learn, attributes_test, transform_classes(classes_learn, 2), transform_classes(classes_test, 2))
    fpr_arthropods_svc, tpr_arthropods_svc, auc_arthropods_svc = get_fpr_tpr_auc_for_classifier_and_feats(
        'SVC', {'kernel': 'linear', 'probability': True, 'C': arthropods_c_svc}, arthropods_feat_ind_svc,
        attributes_learn, attributes_test, transform_classes(classes_learn, 2), transform_classes(classes_test, 2))

    f, axarr = plt.subplots(3, 1, figsize=(6, 12))
    axarr[0].plot(fpr_plants_lr, tpr_plants_lr, label='LR  AUC = %0.3f' % auc_plants_lr)
    axarr[0].plot(fpr_plants_svc, tpr_plants_svc, label='SVC AUC = %0.3f' % auc_plants_svc)
    axarr[0].plot([0, 1], [0, 1], 'r--', label='random')
    axarr[0].set_title('Viruses infecting seed plants')
    axarr[0].legend(loc='lower right')
    axarr[0].set_xlim([0, 1])
    axarr[0].set_ylim([0, 1])
    axarr[0].set_ylabel('True Positive Rate')
    #axarr[0].set_xlabel('False Positive Rate')
    axarr[1].plot(fpr_vertebrates_lr, tpr_vertebrates_lr, label='LR  AUC = %0.3f' % auc_vertebrates_lr)
    axarr[1].plot(fpr_vertebrates_svc, tpr_vertebrates_svc, label='SVC AUC = %0.3f' % auc_vertebrates_svc)
    axarr[1].plot([0, 1], [0, 1], 'r--', label='random')
    axarr[1].set_title('Viruses infecting vertebrates')
    axarr[1].legend(loc='lower right')
    axarr[1].set_xlim([0, 1])
    axarr[1].set_ylim([0, 1])
    axarr[1].set_ylabel('True Positive Rate')
    #axarr[1].set_xlabel('False Positive Rate')
    axarr[2].plot(fpr_arthropods_lr, tpr_arthropods_lr, label='LR  AUC = %0.3f' % auc_arthropods_lr)
    axarr[2].plot(fpr_arthropods_svc, tpr_arthropods_svc, label='SVC AUC = %0.3f' % auc_arthropods_svc)
    axarr[2].plot([0, 1], [0, 1], 'r--', label='random')
    axarr[2].set_title('Viruses infecting arthropods')
    axarr[2].legend(loc='lower right')
    axarr[2].set_xlim([0, 1])
    axarr[2].set_ylim([0, 1])
    axarr[2].set_ylabel('True Positive Rate')
    axarr[2].set_xlabel('False Positive Rate')
    plt.setp([a.get_xticklabels() for a in axarr[:-1]], visible=False)
    f.subplots_adjust(hspace=0.14)
    plt.savefig(os.path.join('..', 'figures', 'ROC_test_set_eukariotic_viruses.svg'), bbox_inches='tight')