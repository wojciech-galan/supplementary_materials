#! /usr/bin/python
# -*- coding: utf-8 -*-

import cPickle as pickle
import sys
import os
import subprocess
from joblib import Parallel, delayed
from viral_seq_fetcher.src.SeqContainer import Container
from viral_seq_fetcher import src

NUMBER_TO_ACID = {1:'dna', 0:'rna'}


class Result(object):
    def __init__(self, lr_p, knn_p, qda_p, svm_p, proper, gi):
        self.lr_proba = lr_p
        self.knn_proba = knn_p
        self.svm_proba = svm_p
        self.qda_proba = qda_p
        self.proper = proper
        self.id = gi


class AbstractSequence(object):
    def __init__(self, seq):
        super(AbstractSequence, self).__init__()
        self.seq = seq.strip()

    def __len__(self):
        return len(self.seq)

    def __contains__(self, item):
        return item.seq in self.seq

    def __bool__(self):
        return bool(self.seq)

    def __eq__(self, other):
        return self.seq == other.seq

    def __hash__(self):
        return hash(self.seq)


class FastQ(AbstractSequence):

    def __init__(self, lines):
        assert len(lines) == 4
        super(FastQ, self).__init__(lines[1])
        self.description = lines[0][1:].split('_')[0]
        self.quality = lines[3]


class Fasta(AbstractSequence):

    def __init__(self, descr, seq):
        super(Fasta, self).__init__(seq)
        descr = str(descr)
        if not descr.startswith('>'):
            descr = '>' + descr
        self.description = descr

    def __str__(self):
        return self.description + '\r\n' + self.seq


def read_fastq_file(path):
    ret_list = []
    lines = []
    for line in open(path):
        lines.append(line.strip())
        if len(lines) == 4:
            ret_list.append(FastQ(lines))
            lines = []
    return ret_list


def write_fastas_to_a_file(fastas, path):
    with open(path, 'w') as f:
        f.write('\r\n'.join([str(fasta) for fasta in fastas]))
        f.write('\r\n')


def fastq_to_fasta(fastq_obj):
    return Fasta(fastq_obj.description, fastq_obj.seq)


def read_fasta_file(path):
    with open(path) as f:
        f.readline()
        return Fasta(os.path.basename(path), ''.join(f.readlines()))

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
for length in (100, 250, 500):
    for error_rate in (0, 0.02):
        simulated_seq_path = 'blah_%f_%d.fastq'%(error_rate, length)
        if not os.path.exists(simulated_seq_path):
            cmd = 'wgsim /tmp/seq %s /dev/null -h -e %f -S 77 -d 0 -1 %d -N 100000'%(simulated_seq_path, error_rate, length)
            print cmd
            subprocess.call(cmd, shell=True)
        fastqs = read_fastq_file(simulated_seq_path)
        fastas = [fastq_to_fasta(fastq) for fastq in fastqs]
        print "starting classifier evaluation"
        results = Parallel(n_jobs=6)(delayed(predict_based_on_fasta_file)(fasta_seq=fasta, fasta_dir=fasta_dir, gi_molecule=gi_molecule_map, gi_host=gi_host_map, i=i) for i, fasta in enumerate(fastas))
        evaluation_results[(length, error_rate)] = [Result(result[0], result[1], result[2], result[3], result[4], result[5]) for result in results]
        pickle.dump(evaluation_results, open(os.path.join('..', 'datasets', 'check_simmulated_metagenomics_results_%.1f_%d.dump'%(error_rate, length)), 'w'))