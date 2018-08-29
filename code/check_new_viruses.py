#! /usr/bin/python
# -*- coding: utf-8 -*-

import cPickle as pickle
import sys
import os
import time
import urllib2
import socket
import subprocess
from Bio import Entrez
from viral_seq_fetcher.src.SeqContainer import Container
from viral_seq_fetcher import src

NUMBER_TO_ACID = {1:'dna', 0:'rna'}

def download_fasta(directory, gi):
    path = os.path.join(directory, gi)
    while not os.path.exists(path):
        try:
            handle = Entrez.efetch(db="nuccore", id=gi, rettype='fasta', retmode="text")
            print handle.geturl()
            with open(path, 'w') as file_handle:
                file_handle.write(handle.read())
        except (urllib2.URLError, socket.timeout, socket.error), e:
            print e
            time.sleep(1)
    return path


def translate_host_to_infecting(host_lineage):
    try:
        if host_lineage[1] == 'Eukaryota':
            return 'Eucaryota-infecting'
        elif host_lineage[1] in ('Bacteria', 'Archaea'):
            return 'phage'
    except IndexError:
        pass
    print host_lineage[:2]


class Result(object):
    def __init__(self, lr_p, knn_p, qda_p, svm_p, proper, gi):
        self.lr_proba = lr_p
        self.knn_proba = knn_p
        self.svm_proba = svm_p
        self.qda_proba = qda_p
        self.proper = proper
        self.id = gi

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

    # save the gi list
    with open(os.path.join('..', 'datasets', 'new_viruses_ids'), 'w') as f:
        f.write(os.linesep.join(gi_host_map.keys()))


    new_hosts = [tuple(v.host_lineage[:2]) for v in with_host_diff]
    new_hosts_dict = {}
    for key in set(new_hosts):
        new_hosts_dict[key] = new_hosts.count(key)
    for k, v in new_hosts_dict.items():
        print k, v

    # ('cellular organisms', 'Archaea') 9
    # (u'cellular organisms', u'Bacteria') 72
    # (u'cellular organisms', u'Eukaryota') 756
    # ('unclassified sequences',) 25


    command = 'viruses_classifier %s --nucleic_acid %s --classifier %s --probas'
    seq_directory = '/home/wojtek/PycharmProjects/Viral_feature_extractor/viral_seq_fetcher/files/sequences/'
    results = []
    for i, v in enumerate(with_proper_host_ineage_diff):
        if not i%10: print i, 'Processed'
        temp_dir = '/tmp/seqs'
        proper_result = translate_host_to_infecting(v.host_lineage)
        #print "Proper result for", v.gi, proper_result
        f_name = download_fasta(temp_dir, v.gi)
        result = []
        for classifier in ('lr', 'knn', 'qda', 'svc'):
            p = subprocess.Popen(command%(f_name, NUMBER_TO_ACID[v.molecule], classifier), shell=True, stdout=subprocess.PIPE)
            p.wait()
            result.append(p.communicate()[0].strip())
        #print result
        results.append(Result(result[0], result[1], result[2], result[3], proper_result, v.gi))
    pickle.dump(results, open(os.path.join('..', 'datasets', 'check_new_viruses_results.dump'), 'w'))
