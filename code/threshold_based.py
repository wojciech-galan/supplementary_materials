#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import cPickle as pickle
import matplotlib.pyplot as plt
from check_new_viruses_analyze_results import compute_fpr_tpr_auc
from check_new_viruses import download_fasta
from check_new_viruses import translate_host_to_infecting
from check_new_viruses import NUMBER_TO_ACID
from check_new_viruses_analyze_results import CLASS_TO_NUM
from lib import write_fastas_to_a_file


class ThresholdBasedClassifier(object):
    '''
    Predicts on of two possible classes based on threshold
    '''

    def __init__(self, threshold=0, predict_when_below_threshold=0, possible_classes={0, 1}):
        assert predict_when_below_threshold in possible_classes
        assert len(possible_classes) == 2
        super(ThresholdBasedClassifier, self).__init__()
        self.threshold = threshold
        self.predict_when_below_threshold = predict_when_below_threshold
        self.possible_classes = possible_classes

    def predict(self, X_vector):
        def predict_one_object(x, threshold, predict_when_below_threshold, possible_classes):
            if x < threshold:
                return predict_when_below_threshold
            else:
                return list({predict_when_below_threshold} ^ possible_classes)[0]

        return [predict_one_object(x, self.threshold, self.predict_when_below_threshold, self.possible_classes) for x in
                X_vector]


def get_fpr_tpr_auc_one_feature(feature_num, x_learn, x_test, y_learn, y_test, threshold=0):
    '''Classifies an object to one of two classes based only on one feature'''
    tbc = ThresholdBasedClassifier(threshold)
    fpr, tpr, auc = compute_fpr_tpr_auc(y_test, tbc.predict(x_test[:, feature_num]))
    return fpr, tpr, auc


def get_diff_between_two_viral_containers(newer_cont_path, older_cont_path):
    from viral_seq_fetcher.src.SeqContainer import Container
    from viral_seq_fetcher import src
    container_old = Container.fromFile(os.path.join('..', 'datasets', 'container_Fri_Oct__6_14:26:35_2017.dump'))
    with_host_old = [v for v in container_old if v.host_lineage]
    ids_old = [v.gi for v in with_host_old]
    sys.modules['src'] = src
    container_new = pickle.load(open(os.path.join('..', 'datasets', 'container_Mon_Aug_27_16:51:14_2018.dump')))
    with_host_new = [v for v in container_new if v.host_lineage]
    with_host_diff = [v for v in with_host_new if v.gi not in ids_old]
    with_proper_host_lineage_diff = [v for v in with_host_diff if
                                     translate_host_to_infecting(v.host_lineage) in ('Eucaryota-infecting', 'phage')]
    return with_proper_host_lineage_diff


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
    try:
        with open(new_viruses_res_name) as f:
            fpr_new_viruses, tpr_new_viruses, auc_new_viruses = pickle.load(f)
    except IOError:
        with_proper_host_lineage_diff = get_diff_between_two_viral_containers(
            os.path.join('..', 'datasets', 'container_Mon_Aug_27_16:51:14_2018.dump'),
            os.path.join('..', 'datasets', 'container_Fri_Oct__6_14:26:35_2017.dump'))
        proper_results = [CLASS_TO_NUM[translate_host_to_infecting(v.host_lineage)] for v in
                          with_proper_host_lineage_diff]
        # coded as in viral_seq_fetcher.src.constants.ACID_TO_NUMBER, so 1 means 'dna' as well as 'phage'
        results = [v.molecule for v in with_proper_host_lineage_diff]

        fpr_new_viruses, tpr_new_viruses, auc_new_viruses = compute_fpr_tpr_auc(proper_results, results)
        pickle.dump((fpr_new_viruses, tpr_new_viruses, auc_new_viruses), open(new_viruses_res_name, 'w'))

    # evaluation on simulated subsequences
    simulated_metagenomics_res_name = os.path.join('..', 'datasets',
                                                   'check_simulated_metagenomics_threshold_based_results.dump')
    try:
        with open(simulated_metagenomics_res_name) as f:
            fpr_simulated, tpr_simulated, auc_simulated = pickle.load(f)
    except IOError:
        with_proper_host_lineage_diff = get_diff_between_two_viral_containers(
            os.path.join('..', 'datasets', 'container_Mon_Aug_27_16:51:14_2018.dump'),
            os.path.join('..', 'datasets', 'container_Fri_Oct__6_14:26:35_2017.dump'))
        gi_host_map = {v.gi: translate_host_to_infecting(v.host_lineage) for v in with_proper_host_lineage_diff}
        gi_molecule_map = {v.gi: v.molecule for v in with_proper_host_lineage_diff}
        # assuming, that fasta sequences are stored in /tmp/seqs
        old_dir = '/tmp/seqs'
        new_dir = '/tmp/seqs2'
        fasta_dir = '/tmp/fastas'
        try:
            os.makedirs(new_dir)
        except OSError:
            pass
        try:
            os.makedirs(fasta_dir)
        except OSError:
            pass

        for gi in gi_host_map:
            fasta = read_fasta_file(os.path.join(old_dir, gi))
            write_fastas_to_a_file([fasta], os.path.join(new_dir, gi))  # change seq description to gi
        subprocess.call('cat %s > /tmp/seq' % ' '.join([os.path.join(new_dir, gi) for gi in gi_host_map]), shell=True)

        evaluation_results = {}
        simulator = 'mysim'
        for length in (100, 250, 500, 1000, 3000, 10000):
            for error_rate in (0.00, 0.02):
                simulated_seq_path = 'blah_mysim_%f_%d.fastq' % (error_rate, length)
                if not os.path.exists(simulated_seq_path):
                    cmd = 'python /home/wojtek/PycharmProjects/supplementary_materials/code/my_read_simulator.py /tmp/seq %d 100000 %f %s' % (
                            length, error_rate, simulated_seq_path)
                    print cmd
                    subprocess.call(cmd, shell=True)
                fastas = read_multifasta_file(simulated_seq_path)
