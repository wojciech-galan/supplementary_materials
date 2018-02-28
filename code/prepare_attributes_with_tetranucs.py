#! /usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os
import cPickle as pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from viral_seq_fetcher.src.hostProcessing import *
from viral_seq_fetcher.src.codonUsageBias import *
from viral_seq_fetcher.src.SeqEntrySeq import StandaloneSeqEntrySeq, SeqEntrySeqException
from viral_seq_fetcher.src.SeqEntrySet import SeqEntrySet
from viral_seq_fetcher.src.UnifiedSeq import UnifiedSeq
from viral_seq_fetcher.src.constants import *
from viral_seq_fetcher.src.commonFunctions import *
from viral_seq_fetcher.src.ParserClasses import processFile


class SeqRepresentation(object):
    '''Klasa licząca współczynniki, które będą potem używane przez AI'''

    def __init__(self, uniSeq, taxonomy_dir, debug, verbose):
        '''Argumenty:
            - uniSeq - obiekt typu UnifiedSequence'''
        super(SeqRepresentation, self).__init__()
        self.gi = uniSeq.gi
        seq_rscu = {}
        seq_cai = {}
        seq_enc = {}
        seq_codons = {}
        # pdb.set_trace()

        # ogólne właściwości
        if uniSeq.strand in strands_to_number:
            self.strand = strands_to_number[uniSeq.strand]
        else:
            self.strand = None
        if uniSeq.molecule in ACID_TO_NUMBER:
            self.molecule = ACID_TO_NUMBER[uniSeq.molecule]
        else:
            self.molecule = None
            logger.info("For gi=%s molecule=%s" % (uniSeq.gi, uniSeq.molecule))
            pdb.set_trace()  # TODO wyjebać?
        if uniSeq.seq:
            self.length = len(uniSeq.seq)
        else:
            self.length = None
            logger.info("For gi=%s lengt=%s" % (uniSeq.gi, self.length))
        self.lineage = [s.strip() for s in uniSeq.lineage.split(';')]
        if uniSeq.host:
            self.host = uniSeq.host
            self.host_lineage = findHostLineage(uniSeq.host, taxonomy_dir, debug=debug, verbose=verbose)
        else:
            logger.info("No host for %s" % uniSeq.gi)
            self.host = None
            self.host_lineage = None

        # liczenie mono, di i tri
        if not uniSeq.seq:
            self.nuc_frequencies = None
            self.relative_nuc_frequencies = None
            self.relative_trinuc_freqs = None
            self.relative_nuc_frequencies_one_strand = None
            self.relative_trinuc_freqs_one_strand = None
            cd_regions = None
            logger.info("No seq for %s" % uniSeq.gi)
        else:
            self.nuc_frequencies = nucFrequencies(uniSeq.seq, 2)
            self.relative_nuc_frequencies_one_strand = relativeNucFrequencies(self.nuc_frequencies, 1)
            self.relative_trinuc_freqs_one_strand = thirdOrderBias(uniSeq.seq, 1)
            if self.strand in (1, 2):
                self.relative_nuc_frequencies = relativeNucFrequencies(self.nuc_frequencies,
                                                                       strands_to_number[uniSeq.strand])
                self.relative_trinuc_freqs = thirdOrderBias(uniSeq.seq, strands_to_number[uniSeq.strand])
            # print self.relative_nuc_frequencies
            # pdb.set_trace()
            else:
                logger.info("For gi=%s strand=%s" % (uniSeq.gi, uniSeq.strand))
                self.relative_nuc_frequencies = None
                self.relative_trinuc_freqs = None
            cd_regions = uniSeq.getCdRegions()
            # tetra
            self.nuc_frequencies.update(nucFrequencies(uniSeq.seq, 4))

        # liczenie codon usage bias
        if not cd_regions:
            logger.info("No cd_regions for %s" % uniSeq.gi)
            self.avg_rscu_all = None
            self.std_rscu_all = None
            self.avg_cai_all = None
            self.std_cai_all = None
            self.avg_rscu_proper = None
            self.std_rscu_proper = None
            self.avg_cai_proper = None
            self.std_cai_proper = None
            self.avg_enc_all = None
            self.std_enc_all = None
            self.avg_enc_proper = None
            self.std_enc_proper = None
            self.avg_codons_all = None
            self.std_codons_all = None
            self.avg_codons_proper = None
            self.std_codons_proper = None
            self.num_of_all_cd_regions = None
            self.num_of_proper_cd_regions = None
            self.nuc_frequencies_inside_cd_regions = None
            self.nuc_frequencies_outside_cd_regions = None
            self.relative_nuc_frequencies_inside_cd_regions = None
            self.relative_nuc_frequencies_outside_cd_regions = None
            self.relative_trinuc_freqs_inside_cd_regions = None
            self.relative_trinuc_freqs_outside_cd_regions = None
            return
        none_prod_id = 0
        self.num_of_all_cd_regions = len(cd_regions)
        nuc_freqs_cd = []
        relative_nuc_freqs_cd = []
        relative_trinuc_freqs_cd = []
        weights = []
        weights_rel = []
        for cd_region in cd_regions:
            # liczymy i dla proper cd_regions i dla niewłaściwych
            if cd_region.frame == 'one':
                codon_usage = codons(cd_region.seq)
            elif cd_region.frame == 'two':
                codon_usage = codons(cd_region.seq[1:])
            elif cd_region.frame == 'three':
                codon_usage = codons(cd_region.seq[2:])
            else:
                logger.info("For gi=%s cd_region.frame=%s" % (uniSeq.gi, cd_region.frame))
                pdb.set_trace()  # TODO wyjebać?
            if not cd_region.prod_id:  # TODO what?
                cd_region.prod_id = 'no_id_%d' % none_prod_id
                none_prod_id += 1
            seq_rscu[cd_region.prod_id] = rscu(codon_usage, eval('gencode_SG%s' % cd_region.code))
            seq_cai[cd_region.prod_id] = cai(codon_usage, eval('gencode_SG%s' % cd_region.code))
            seq_enc[cd_region.prod_id] = enc(codon_usage, eval('gencode_SG%s' % cd_region.code))
            seq_codons[cd_region.prod_id] = codonFreqs(cd_region.seq)
            nuc_freqs_one_seq = nucFrequencies(cd_region.seq, 2)
            nuc_freqs_cd.append(nuc_freqs_one_seq)
            weights.append(len(cd_region.seq))
            if self.strand in (1, 2):
                relative_nuc_freqs_cd.append(
                    relativeNucFrequencies(nuc_freqs_one_seq, strands_to_number[uniSeq.strand]))
                relative_trinuc_freqs_cd.append(thirdOrderBias(cd_region.seq, strands_to_number[uniSeq.strand]))
                weights_rel.append(len(cd_region.seq))
        self.nuc_frequencies_inside_cd_regions = {}
        self.relative_nuc_frequencies_inside_cd_regions = {}
        self.relative_trinuc_freqs_inside_cd_regions = {}
        for nuc in nuc_freqs_cd[0]:
            nuc_freqs = [cd_region[nuc] for cd_region in nuc_freqs_cd]
            self.nuc_frequencies_inside_cd_regions[nuc] = weightedArithmeticMean(values=nuc_freqs, weights=weights)
        if len(relative_nuc_freqs_cd) > 0:
            for nuc in relative_nuc_freqs_cd[0]:
                nuc_freqs = [cd_region[nuc] for cd_region in relative_nuc_freqs_cd]
                self.relative_nuc_frequencies_inside_cd_regions[nuc] = weightedArithmeticMean(values=nuc_freqs,
                                                                                              weights=weights_rel)
            for nuc in relative_trinuc_freqs_cd[0]:
                trinuc_freqs = [cd_region[nuc] for cd_region in relative_trinuc_freqs_cd]
                self.relative_trinuc_freqs_inside_cd_regions = weightedArithmeticMean(values=trinuc_freqs,
                                                                                      weights=weights_rel)
        else:
            self.relative_nuc_frequencies_inside_cd_regions = None
            self.relative_trinuc_freqs_inside_cd_regions = None

        nuc_freqs_outside_cd = []
        relative_nuc_freqs_outside_cd = []
        relative_trinuc_freqs_outside_cd = []
        weights = []
        weights_rel = []
        weights_rel_trinuc = []
        self.nuc_frequencies_outside_cd_regions = {}
        self.relative_nuc_frequencies_outside_cd_regions = {}
        self.relative_trinuc_freqs_outside_cd_regions = {}
        for outside_cd_reg in uniSeq.getSeqsOutsideCdRegions():
            if len(outside_cd_reg) > 10:  # krótszych odcinków nie ma sensu brać pod uwagę, zaburzają tylko wyniki
                nuc_freqs_one_seq = nucFrequencies(outside_cd_reg, 2)
                nuc_freqs_outside_cd.append(nuc_freqs_one_seq)
                weights.append(len(outside_cd_reg))
                if self.strand in (1, 2):
                    relative_nuc_freqs_outside_cd.append(
                        relativeNucFrequencies(nuc_freqs_one_seq, strands_to_number[uniSeq.strand]))
                    weights_rel.append(len(outside_cd_reg))
                    relative_trinuc_freqs_outside_cd.append(
                        thirdOrderBias(outside_cd_reg, strands_to_number[uniSeq.strand]))

        if len(nuc_freqs_outside_cd) > 0:
            for nuc in nuc_freqs_outside_cd[0]:
                nuc_freqs = [outside_cd_reg[nuc] for outside_cd_reg in nuc_freqs_outside_cd]
                self.nuc_frequencies_outside_cd_regions[nuc] = weightedArithmeticMean(values=nuc_freqs, weights=weights)
        else:
            self.nuc_frequencies_outside_cd_regions = None
        if len(relative_nuc_freqs_outside_cd) > 0:
            for nuc in relative_nuc_freqs_outside_cd[0]:
                nuc_freqs = [outside_cd_reg[nuc] for outside_cd_reg in relative_nuc_freqs_outside_cd]
                self.relative_nuc_frequencies_outside_cd_regions[nuc] = weightedArithmeticMean(values=nuc_freqs,
                                                                                               weights=weights_rel)
            for nuc in relative_trinuc_freqs_outside_cd[0]:
                trinuc_freqs = [cd_region[nuc] for cd_region in relative_trinuc_freqs_outside_cd]
                self.relative_trinuc_freqs_outside_cd_regions = weightedArithmeticMean(values=trinuc_freqs,
                                                                                       weights=weights_rel)
        else:
            self.relative_nuc_frequencies_outside_cd_regions = None
            self.relative_trinuc_freqs_outside_cd_regions = None

        self.avg_rscu_all = {}
        self.avg_cai_all = {}
        self.std_rscu_all = {}
        self.std_cai_all = {}
        self.avg_codons_all = {}
        self.std_codons_all = {}
        for codon in codon_usage:
            l1 = zip(*[(seq_rscu[cd_region.prod_id][codon], len(cd_region.seq)) for cd_region in cd_regions])
            l2 = zip(*[(seq_cai[cd_region.prod_id][codon], len(cd_region.seq)) for cd_region in cd_regions])
            l3 = zip(*[(seq_codons[cd_region.prod_id][codon], len(cd_region.seq)) for cd_region in cd_regions])
            self.avg_rscu_all[codon] = float(weightedArithmeticMean(values=l1[0], weights=l1[1]))
            self.avg_cai_all[codon] = float(weightedArithmeticMean(values=l2[0], weights=l2[1]))
            self.avg_codons_all[codon] = float(weightedArithmeticMean(values=l3[0], weights=l3[1]))
            if len(cd_regions) > 1:
                self.std_rscu_all[codon] = weightedStandardDeviation(values=l1[0], weights=l1[1])
                self.std_cai_all[codon] = weightedStandardDeviation(values=l2[0], weights=l2[1])
                self.std_codons_all[codon] = weightedStandardDeviation(values=l3[0], weights=l3[1])
            else:
                self.std_rscu_all = None
                self.std_cai_all = None
                self.std_codons_all = None
        l = zip(*[(seq_enc[cd_region.prod_id], len(cd_region.seq)) for cd_region in cd_regions])
        self.avg_enc_all = float(weightedArithmeticMean(values=l[0], weights=l[1]))
        if len(cd_regions) > 1:
            self.std_enc_all = weightedStandardDeviation(values=l[0], weights=l[1])
        else:
            self.std_enc_all = None
        # jeśli wszystkie są proper - to przepisujemy
        if all(cd_region.proper for cd_region in cd_regions):
            self.avg_rscu_proper = self.avg_rscu_all
            self.avg_cai_proper = self.avg_cai_all
            self.avg_enc_proper = self.avg_enc_all
            self.std_rscu_proper = self.std_rscu_all
            self.std_cai_proper = self.std_cai_all
            self.std_enc_proper = self.std_enc_all
            self.num_of_proper_cd_regions = self.num_of_all_cd_regions
            self.avg_codons_proper = self.avg_codons_all
            self.std_codons_proper = self.std_codons_all
        # jeśli nie, to liczymy
        else:
            self.avg_rscu_proper = {}
            self.avg_cai_proper = {}
            self.std_rscu_proper = {}
            self.std_cai_proper = {}
            self.avg_codons_proper = {}
            self.std_codons_proper = {}
            self.num_of_proper_cd_regions = len([cd_region.proper for cd_region in cd_regions])
            if any(cd_region.proper for cd_region in cd_regions):
                for codon in codon_usage:
                    l1 = zip(*[(seq_rscu[cd_region.prod_id][codon], len(cd_region.seq)) for cd_region in cd_regions if
                               cd_region.proper])
                    l2 = zip(*[(seq_cai[cd_region.prod_id][codon], len(cd_region.seq)) for cd_region in cd_regions if
                               cd_region.proper])
                    l3 = zip(*[(seq_codons[cd_region.prod_id][codon], len(cd_region.seq)) for cd_region in cd_regions if
                               cd_region.proper])
                    self.avg_rscu_proper[codon] = float(weightedArithmeticMean(values=l1[0], weights=l1[1]))
                    self.avg_cai_proper[codon] = float(weightedArithmeticMean(values=l2[0], weights=l2[1]))
                    self.avg_codons_proper[codon] = float(weightedArithmeticMean(values=l3[0], weights=l3[1]))
                    if len([cd_region for cd_region in cd_regions if cd_region.proper]) > 1:
                        self.std_rscu_proper[codon] = weightedStandardDeviation(values=l1[0], weights=l1[1])
                        self.std_cai_proper[codon] = weightedStandardDeviation(values=l2[0], weights=l2[1])
                        self.std_codons_proper[codon] = weightedStandardDeviation(values=l3[0], weights=l3[1])
                    else:
                        self.std_rscu_proper = None
                        self.std_cai_proper = None
                        self.std_codons_proper = None
                self.avg_enc_proper = float(weightedArithmeticMean(
                    {seq_enc[cd_region.prod_id]: len(cd_region.seq) for cd_region in cd_regions if cd_region.proper}))
                l = zip(
                    *[(seq_enc[cd_region.prod_id], len(cd_region.seq)) for cd_region in cd_regions if cd_region.proper])
                if len([cd_region for cd_region in cd_regions if cd_region.proper]) > 1:
                    self.std_enc_proper = weightedStandardDeviation(values=l[0], weights=l[1])
                else:
                    self.std_enc_proper = None
            else:
                self.avg_rscu_proper = None
                self.avg_cai_proper = None
                self.avg_enc_proper = None
                self.std_rscu_proper = None
                self.std_cai_proper = None
                self.std_enc_proper = None
                self.avg_codons_proper = None
            self.std_codons_proper = None


class LittleParser(object):
    '''Klasa ma znalezc gospodarza i organizm (w postaci Viruses; ssRNA positive-strand viruses, no DNA stage; Picornavirales; Picornaviridae; Enterovirus
    dla odpowiedniego pliku xml, oraz id, nazwe i sekwencje'''

    def __init__(self, **kwargs):
        super(LittleParser, self).__init__()
        for key, value in kwargs.iteritems():
            self.__dict__[key] = value

    @classmethod
    def fromDBRecord(cls, record, keys):
        object_dict = {}
        for nr, key in enumerate(keys, start=0):
            object_dict[key] = record[nr]
        return cls(**object_dict)

    @classmethod
    def fromHandle(cls, handle, taxonomy_dir, debug, verbose):
        '''handle - uchwyt do pliku lub zasobu w sieci'''
        object_dict = {}
        try:
            object_dict['_url'] = handle.geturl()
        except AttributeError:
            object_dict['_name'] = handle.name

        object_dict['_za_ktorym_razem'] = 0
        object_dict['_host'] = []
        # file is empty - throws XMLSyntaxError
        seq = _parse(handle, taxonomy_dir, debug, verbose)
        return seq

    @constantIfProper
    def length(self):
        return self._seq_len

    def hasHost(self):
        if self._host:
            return True
        return False

    def setHost(new_host_id):
        '''jeśli podczas sprawdzania okazało się że skryp nie ustalił gospodarza lub zrobił to niepoprawnie
        to podajemy prawidłowego. Parametry:
        - new_host_id - prawidłowy gospodarz - jego id w bazie taksonomy'''
        pass


def _parse(handle, taxonomy_dir, debug, verbose):
    object_dict = {}
    if verbose:
        print handle.name
    gi = os.path.basename(handle.name)
    int(gi)  # upewniam się, że to zawsze będzie numer
    root = processFile(handle)

    if root.bioseq_set_seq_set__.__dict__.keys() != ['to_evaluate', 'seq_entry_', 'name']:
        raise LittleParserException(
            "root.bioseq_set_seq_set__.__dict__.keys() != ['to_evaluate', 'seq_entry_', 'name']")
    if len(root.bioseq_set_seq_set__.seq_entry_) != 1:
        raise LittleParserException("len(root.bioseq_set_seq_set__.seq_entry_) != 1")

    seq_entry_ = root.bioseq_set_seq_set__.seq_entry_[0]

    if 'seq_entry_seq_' in seq_entry_:
        try:
            seq_entry_seq = StandaloneSeqEntrySeq(seq_entry_.seq_entry_seq_, gi)
        except SeqEntrySeqException, processFilesese:
            logger.debug("While processing %s: %s"(str(sese), gi))
            print sese.message
        uniSeq = UnifiedSeq(seq_entry_seq)
    elif 'seq_entry_set_' in seq_entry_:
        seq_entry_set = SeqEntrySet(seq_entry_.seq_entry_set_, gi)
        uniSeq = UnifiedSeq(seq_entry_set)
    else:
        raise LittleParserException("No seq_entry_seq_ nor seq_entry_set_ in seq_entry_")
    ssseq = SeqRepresentation(uniSeq, taxonomy_dir, debug, verbose)
    return ssseq


def create_new_set(old_set, all_new):
    new_set = []
    for element in old_set:
        for element2 in all_new:
            if element.gi == element2.gi:
                new_set.append(element2)
                break
    assert len(old_set) == len(new_set)
    return new_set


def virus_to_attributes(virus, choosen_attributes):
    l = []
    for att in choosen_attributes:
        if type(virus.__dict__[att]) == dict:
            l.extend(virus.__dict__[att][key] for key in sorted(virus.__dict__[att]))
        else:
            l.append(virus.__dict__[att])
    return l


if __name__ == '__main__':
    seq_representations = []
    seq_dir = sys.argv[1]
    taxonomy_dir = sys.argv[2]
    cv_eu = pickle.load(open(
        os.path.join('..', 'datasets', 'viruses_crossvalidation_eu.dump')))
    cv_phage = pickle.load(open(os.path.join(
        '..', 'datasets', 'viruses_crossvalidation_phages.dump')))
    test_eu = pickle.load(
        open(os.path.join('..', 'datasets', 'viruses_blind_set_eu.dump')))
    test_phage = pickle.load(open(
        os.path.join('..', 'datasets', 'viruses_blind_set_phages.dump')))

    for seq_file in os.listdir(seq_dir):
        print seq_file
        representation = LittleParser.fromHandle(open(os.path.join(seq_dir, seq_file)), taxonomy_dir, False, False)
        seq_representations.append(representation)
    new_cv_eu = create_new_set(cv_eu, seq_representations)
    new_cv_phage = create_new_set(cv_phage, seq_representations)
    new_test_eu = create_new_set(test_eu, seq_representations)
    new_test_phage = create_new_set(test_phage, seq_representations)

    # preparing attributes
    attribs = ('molecule', 'nuc_frequencies', 'relative_nuc_frequencies_one_strand', 'relative_trinuc_freqs_one_strand')
    attributes_learn = [virus_to_attributes(virus, attribs) for virus in new_cv_eu]
    attributes_learn.extend([virus_to_attributes(virus, attribs) for virus in new_cv_phage])
    attributes_learn = np.array(attributes_learn, dtype=float)
    attributes_test = [virus_to_attributes(virus, attribs) for virus in new_test_eu]
    attributes_test.extend([virus_to_attributes(virus, attribs) for virus in new_test_phage])
    attributes_test = np.array(attributes_test, dtype=float)

    # scaling features
    scaler = StandardScaler()
    attributes_learn = scaler.fit_transform(attributes_learn)
    attributes_test = scaler.transform(attributes_test)
    pickle.dump(attributes_learn,
                open(os.path.join('..', 'datasets', 'attributes_learn_with_tetra.dump', 'w')))
    pickle.dump(attributes_test,
                open(os.path.join('..', 'datasets', 'attributes_test_with_tetra.dump', 'w')))