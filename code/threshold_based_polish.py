#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import itertools
import cPickle as pickle
import matplotlib.pyplot as plt
from threshold_based import get_fpr_tpr_auc_one_feature

if __name__ == '__main__':
	# evaluation on test set
	attributes_learn = pickle.load(open(os.path.join('..', 'datasets', 'attributes_learn.dump')))
	attributes_test = pickle.load(open(os.path.join('..', 'datasets', 'attributes_test.dump')))
	classes_learn = pickle.load(open(os.path.join('..', 'datasets', 'classes_learn.dump')))
	classes_test = pickle.load(open(os.path.join('..', 'datasets', 'classes_test.dump')))
	tb_fpr, tb_tpr, tb_auc = get_fpr_tpr_auc_one_feature(0, attributes_learn, attributes_test, classes_learn,
	                                                     classes_test)
	# evaluation on new viruses
	new_viruses_res_name = os.path.join('..', 'datasets', 'check_new_viruses_threshold_based_results.dump')
	with open(new_viruses_res_name) as f:
		fpr_new_viruses, tpr_new_viruses, auc_new_viruses = pickle.load(f)

	# evaluation on simulated subsequences
	simulated_metagenomics_res_name = os.path.join('..', 'datasets',
	                                               'check_simulated_metagenomics_threshold_based_results.dump')
	simulated_metagenomics_confusion_matrices_name = os.path.join('..', 'datasets',
	                                                              'check_simulated_metagenomics_threshold_based_confusion-matrices.dump')
	with open(simulated_metagenomics_res_name) as f:
		res_simulated = pickle.load(f)
	with open(simulated_metagenomics_confusion_matrices_name) as f:
		simulated_metagenomics_confusion_matrices = pickle.load(f)

	# plots
	f, axarr = plt.subplots(1, 2)
	axarr[0].plot(tb_fpr, tb_tpr, label=u'klasyfikator bazujący tylko na rodzaju\nkwasu nukleinowego, AUC = %0.3f' % tb_auc)
	axarr[1].plot(fpr_new_viruses, tpr_new_viruses,
	              label=u'klasyfikator bazujący tylko na rodzaju kwasu\nnukleinowego, AUC = %0.3f' % auc_new_viruses)
	for x in range(2):
		axarr[x].plot([0, 1], [0, 1], 'r--', label='klasyfikator losowy')
		axarr[x].legend(loc='lower right', fontsize=6)
		axarr[x].set_xlim([0, 1])
		axarr[x].set_ylim([0, 1])
	plt.setp(axarr[1].get_yticklabels(), visible=False)
	f.subplots_adjust(wspace=0.1)
	axes_label_fontsize = 8
	title_label_fontsize = 9
	axarr[0].set_ylabel(u'Czułość', fontsize=axes_label_fontsize)
	axarr[0].set_xlabel(u'1 - specyficzność', fontsize=axes_label_fontsize)
	axarr[1].set_xlabel(u'1 - specyficzność', fontsize=axes_label_fontsize)
	axarr[0].set_title('AUC mierzone na zbiorze testowym', fontsize=title_label_fontsize)
	axarr[1].set_title(u'AUC mierzone na zbiorze nowych wirusów', fontsize=title_label_fontsize)
	plt.savefig(os.path.join('..', 'figures', 'nucleic_acid_based_polish.png'), bbox_inches='tight')

	ticks = ('eukariotyczny', 'bakteriofag')
	f, axarr = plt.subplots(6, 2, gridspec_kw={'width_ratios': [5, 2]})
	label_size = 9
	for i, length in enumerate([100, 250, 500, 1000, 3000, 10000]):
		axarr[i, 0].plot(res_simulated[length]['fpr'], res_simulated[length]['tpr'],
		                 label=u'klasyfikator bazujący tylko na rodzaju\nkwasu nukleinowego, AUC = %0.3f' % res_simulated[length]['auc'])
		axarr[i, 0].plot([0, 1], [0, 1], 'r--', label='klasyfikator losowy')
		axarr[i, 0].legend(loc='lower right', fontsize=7)
		axarr[i, 0].set_xlim([0, 1])
		axarr[i, 0].set_ylim([0, 1])
		axarr[i, 0].set_ylabel(u'Czułość', fontsize=9)
		cm = simulated_metagenomics_confusion_matrices[length]
		conf_matrix_plot = axarr[i, 1].imshow(cm, cmap=plt.cm.Blues)
		axarr[i, 1].set_yticks(range(len(ticks)))
		axarr[i, 1].set_yticklabels(ticks, size=label_size)
		axarr[i, 1].set_ylabel('Klasa rzeczywista', size=label_size)
		for k, l in itertools.product(range(2), range(2)):
			axarr[i, 1].text(l, k, format(cm[k, l], 'd'), horizontalalignment="center", size=7,
			                 color="white" if cm[k, l] > 20000 else "black")
		f.colorbar(conf_matrix_plot, ax=axarr[i, 1])
		axarr[i, 0].text(1.1, 1.05, u'Długość fragmentu = %d' % length, horizontalalignment='right', size=9)
	axarr[-1, 0].set_xlabel(u'1 - specyficzność', fontsize=9)
	axarr[-1, 1].set_xticks(range(len(ticks)))
	axarr[-1, 1].set_xticklabels(ticks, rotation=45, size=label_size)
	axarr[-1, 1].set_xlabel('Klasa przewidziana', size=label_size)
	# [tick in ax.xaxis.get_major_ticks()
	plt.setp([axarr[i, j].get_xticklabels() for i in range(5) for j in range(2)], visible=False)
	f.subplots_adjust(hspace=0.25)
	f.subplots_adjust(wspace=0.5)
	f.set_size_inches(9, 15)
	plt.savefig(os.path.join('..', 'figures', 'nucleic_acid_based_simulated_polish.png'), bbox_inches='tight')