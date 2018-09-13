#! /usr/bin/python
# -*- coding: utf-8 -*-

import cPickle as pickle
import sys
import os
import subprocess
from joblib import Parallel, delayed
from viral_seq_fetcher.src.SeqContainer import Container
from viral_seq_fetcher import src
from lib import write_fastas_to_a_file, read_fasta_file, read_fastq_file, fastq_to_fasta
from my_read_simulator import read_fasta_file as read_multifasta_file

NUMBER_TO_ACID = {1:'dna', 0:'rna'}


class Result(object):
    def __init__(self, lr_p, knn_p, qda_p, svm_p, proper, gi):
        self.lr_proba = lr_p
        self.knn_proba = knn_p
        self.svm_proba = svm_p
        self.qda_proba = qda_p
        self.proper = proper
        self.id = gi


def translate_host_to_infecting(host_lineage):
    try:
        if host_lineage[1] == 'Eukaryota':
            return 'Eucaryota-infecting'
        elif host_lineage[1] in ('Bacteria', 'Archaea'):
            return 'phage'
    except IndexError:
        pass


def predict_based_on_fasta_file(fasta_seq, fasta_dir, gi_molecule, gi_host, i):
    command = 'viruses_classifier %s --nucleic_acid %s --classifier %s --probas'
    gi = fasta_seq.description.replace('>', '')
    path = os.path.join(fasta_dir, gi+'_'+str(i))
    write_fastas_to_a_file([fasta_seq], path)
    result = []
    for classifier in ('lr', 'knn', 'qda', 'svc'):
        p = subprocess.Popen(command%(path, NUMBER_TO_ACID[gi_molecule[gi]], classifier), shell=True, stdout=subprocess.PIPE)
        p.wait()
        result.append(p.communicate()[0].strip())
    return (result[0], result[1], result[2], result[3], gi_host[gi], gi)

if __name__ == '__main__':
    container_old = Container.fromFile(os.path.join('..', 'datasets', 'container_Fri_Oct__6_14:26:35_2017.dump'))
    with_host_old = [v for v in container_old if v.host_lineage]
    ids_old = [v.gi for v in with_host_old]
    sys.modules['src'] = src
    container_new = pickle.load(open(os.path.join('..', 'datasets', 'container_Mon_Aug_27_16:51:14_2018.dump')))
    with_host_new = [v for v in container_new if v.host_lineage]
    with_host_diff = [v for v in with_host_new if v.gi not in ids_old]
    with_proper_host_ineage_diff = [v for v in with_host_diff if translate_host_to_infecting(v.host_lineage) in ('Eucaryota-infecting', 'phage')]
    gi_host_map = {v.gi:translate_host_to_infecting(v.host_lineage) for v in with_proper_host_ineage_diff}
    gi_molecule_map = {v.gi:v.molecule for v in with_proper_host_ineage_diff}

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
        write_fastas_to_a_file([fasta], os.path.join(new_dir, gi)) # change seq description to gi
    subprocess.call('cat %s > /tmp/seq' % ' '.join([os.path.join(new_dir, gi) for gi in gi_host_map]), shell=True)

    evaluation_results = {}
    simulator = 'mysim'
    for length in (100, 250, 500, 1000, 3000, 10000):
        for error_rate in (0.00, 0.02):
            simulated_seq_path = 'blah_mysim_%f_%d.fastq'%(error_rate, length)
            if not os.path.exists(simulated_seq_path):
                if simulator == 'wgsim':
                    cmd = 'wgsim /tmp/seq %s /dev/null -h -e %f -S 77 -d 0 -1 %d -N 100000'%(simulated_seq_path, error_rate, length)
                elif simulator == 'mysim':
                    cmd = 'python /home/wojtek/PycharmProjects/supplementary_materials/code/my_read_simulator.py /tmp/seq %d 100000 %f %s' %(length, error_rate, simulated_seq_path)
                print cmd
                subprocess.call(cmd, shell=True)
            if simulator == 'wgsim':
                fastqs = read_fastq_file(simulated_seq_path)
                fastas = [fastq_to_fasta(fastq) for fastq in fastqs]
            elif simulator == 'mysim':
                fastas = read_multifasta_file(simulated_seq_path)
            print "starting classifier evaluation"
            results = Parallel(n_jobs=8)(delayed(predict_based_on_fasta_file)(fasta_seq=fasta, fasta_dir=fasta_dir, gi_molecule=gi_molecule_map, gi_host=gi_host_map, i=i) for i, fasta in enumerate(fastas))
            evaluation_results[(length, error_rate)] = [Result(result[0], result[1], result[2], result[3], result[4], result[5]) for result in results]
            pickle.dump(evaluation_results[(length, error_rate)], open(os.path.join('..', 'datasets', 'check_simmulated_metagenomics_results_mysim_%.2f_%d.dump'%(error_rate, length)), 'w'))
            pickle.dump(evaluation_results[(length, error_rate)], open(os.path.join('..', 'datasets',
                                                                                    'check_simmulated_metagenomics_results_mysim_copy_%.2f_%d.dump' % (
                                                                                    error_rate, length)), 'w'))