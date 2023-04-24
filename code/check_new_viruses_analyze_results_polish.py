#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import cPickle as pickle
import matplotlib.pyplot as plt


if __name__ == '__main__':
	with open(os.path.join('..', 'datasets', 'new_viruses_fpr_tpr.dump')) as f:
		fpr_tpr_dict = pickle.load(f)

	plt.title('Krzywe ROC')
	fpr = fpr_tpr_dict['fpr_lr']
	tpr = fpr_tpr_dict['tpr_lr']
	auc = fpr_tpr_dict['auc_lr']
	plt.plot(fpr, tpr, label='LR  AUC = %0.3f' % auc)
	fpr = fpr_tpr_dict['fpr_svc']
	tpr = fpr_tpr_dict['tpr_svc']
	auc = fpr_tpr_dict['auc_svc']
	plt.plot(fpr, tpr, label='SVC AUC = %0.3f' % auc)
	fpr = fpr_tpr_dict['fpr_knn']
	tpr = fpr_tpr_dict['tpr_knn']
	auc = fpr_tpr_dict['auc_knn']
	plt.plot(fpr, tpr, label='kNN AUC = %0.3f' % auc)
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