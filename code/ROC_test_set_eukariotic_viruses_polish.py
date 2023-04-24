#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import cPickle as pickle
import matplotlib.pyplot as plt

if __name__ == '__main__':
	with open(os.path.join('..', 'datasets', 'eukaryotic_viruses_fpr_tpr.dump')) as f:
		fpr_tpr_dict = pickle.load(f)

	f, axarr = plt.subplots(3, 1, figsize=(6, 12))
	axarr[0].plot(fpr_tpr_dict['fpr_plants_lr'], fpr_tpr_dict['tpr_plants_lr'], label='LR  AUC = %0.3f' % fpr_tpr_dict['auc_plants_lr'])
	axarr[0].plot(fpr_tpr_dict['fpr_plants_svc'], fpr_tpr_dict['tpr_plants_svc'], label='SVC AUC = %0.3f' % fpr_tpr_dict['auc_plants_svc'])
	axarr[0].plot([0, 1], [0, 1], 'r--', label='klasyfikator losowy')
	axarr[0].set_title(u'Wirusy infekujące rośliny nasienne')
	axarr[0].legend(loc='lower right')
	axarr[0].set_xlim([0, 1])
	axarr[0].set_ylim([0, 1])
	axarr[0].set_ylabel(u'Czułość')
	# axarr[0].set_xlabel('False Positive Rate')
	axarr[1].plot(fpr_tpr_dict['fpr_vertebrates_lr'], fpr_tpr_dict['tpr_vertebrates_lr'], label='LR  AUC = %0.3f' % fpr_tpr_dict['auc_vertebrates_lr'])
	axarr[1].plot(fpr_tpr_dict['fpr_vertebrates_svc'], fpr_tpr_dict['tpr_vertebrates_svc'], label='SVC AUC = %0.3f' % fpr_tpr_dict['auc_vertebrates_svc'])
	axarr[1].plot([0, 1], [0, 1], 'r--', label='klasyfikator losowy')
	axarr[1].set_title(u'Wirusy infekujące kręgowce')
	axarr[1].legend(loc='lower right')
	axarr[1].set_xlim([0, 1])
	axarr[1].set_ylim([0, 1])
	axarr[1].set_ylabel(u'Czułość')
	# axarr[1].set_xlabel('False Positive Rate')
	axarr[2].plot(fpr_tpr_dict['fpr_arthropods_lr'], fpr_tpr_dict['tpr_arthropods_lr'], label='LR  AUC = %0.3f' % fpr_tpr_dict['auc_arthropods_lr'])
	axarr[2].plot(fpr_tpr_dict['fpr_arthropods_svc'], fpr_tpr_dict['tpr_arthropods_svc'], label='SVC AUC = %0.3f' % fpr_tpr_dict['auc_arthropods_svc'])
	axarr[2].plot([0, 1], [0, 1], 'r--', label='klasyfikator losowy')
	axarr[2].set_title(u'Wirusy infekujące stawonogi')
	axarr[2].legend(loc='lower right')
	axarr[2].set_xlim([0, 1])
	axarr[2].set_ylim([0, 1])
	axarr[2].set_ylabel(u'Czułość')
	axarr[2].set_xlabel(u'1 - specyficzność')
	plt.setp([a.get_xticklabels() for a in axarr[:-1]], visible=False)
	f.subplots_adjust(hspace=0.14)
	plt.savefig(os.path.join('..', 'figures', 'ROC_test_set_eukariotic_viruses_polish.png'), bbox_inches='tight')
