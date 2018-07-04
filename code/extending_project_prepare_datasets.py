#! /usr/bin/python
# -*- coding: utf-8 -*-

import cPickle as pickle
import os
import operator
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from prepare_attributes_classes_ids import virus_to_attributes


# suppose I know, that a virus infect eukaryotic organism.
# I wanna know, whether the organism is a seed plants (Spermatophyta), proto- or vertebrates


def prepare_attributes_classes_ids_for_learn_and_test(container, test_set_fraction, class_number,
                                                      virus_attributes_to_be_considered):
    container.sort(key=operator.attrgetter('lineage'))
    classes = np.ones(len(container)) * class_number
    ids = [virus.gi for virus in container]
    num_of_elements_in_test_set = int(len(container) * test_set_fraction)
    attributes = [virus_to_attributes(virus, virus_attributes_to_be_considered) for virus in container]
    return np.array(attributes[num_of_elements_in_test_set:]), np.array(attributes[:num_of_elements_in_test_set]), \
           classes[num_of_elements_in_test_set:], classes[:num_of_elements_in_test_set], \
           ids[num_of_elements_in_test_set:], ids[:num_of_elements_in_test_set]


with open(os.path.join('..', 'datasets', 'all_viruses_with_desired_attributes.dump')) as f:
    viruses_container = pickle.load(f)

print len(viruses_container)
seed_plants_infecting = [virus for virus in viruses_container if 'Spermatophyta' in virus.host_lineage]
vertebrates_infecting = [virus for virus in viruses_container if 'Vertebrata' in virus.host_lineage]
arthropods_infecting = [virus for virus in viruses_container if 'Arthropoda' in virus.host_lineage]
other_eukaryotic_viruses = [virus for virus in viruses_container if virus not in set(
    seed_plants_infecting + vertebrates_infecting + arthropods_infecting) and 'Eukaryota' in virus.host_lineage]
print len(seed_plants_infecting), len(vertebrates_infecting), len(arthropods_infecting), len(other_eukaryotic_viruses)

test_set_fraction = 0.2
attribs = ('molecule', 'nuc_frequencies', 'relative_nuc_frequencies_one_strand', 'relative_trinuc_freqs_one_strand')
learn_set_seed_plants, test_set_seed_plants, learn_classes_seed_plants, test_classes_seed_plants, learn_ids_seed_plants, test_ids_seed_plants = prepare_attributes_classes_ids_for_learn_and_test(
    seed_plants_infecting, test_set_fraction, 0, attribs)
learn_set_vertebrates, test_set_vertebrates, learn_classes_vertebrates, test_classes_vertebrates, learn_ids_vertebrates, test_ids_vertebrates = prepare_attributes_classes_ids_for_learn_and_test(
    vertebrates_infecting, test_set_fraction, 1, attribs)
learn_set_arthropods, test_set_arthropods, learn_classes_arthropods, test_classes_arthropods, learn_ids_arthropods, test_ids_arthropods = prepare_attributes_classes_ids_for_learn_and_test(
    arthropods_infecting, test_set_fraction, 2, attribs)
learn_set_other, test_set_other, learn_classes_other, test_classes_other, learn_ids_other, test_ids_other = prepare_attributes_classes_ids_for_learn_and_test(
    other_eukaryotic_viruses, test_set_fraction, 3, attribs)

attributes_test = np.concatenate((test_set_seed_plants, test_set_vertebrates, test_set_arthropods, test_set_other))
print attributes_test.shape
attributes_learn = np.concatenate((learn_set_seed_plants, learn_set_vertebrates, learn_set_arthropods, learn_set_other))
print attributes_learn.shape
classes_test = np.concatenate(
    (test_classes_seed_plants, test_classes_vertebrates, test_classes_arthropods, test_classes_other))
print classes_test.shape
classes_learn = np.concatenate(
    (learn_classes_seed_plants, learn_classes_vertebrates, learn_classes_arthropods, learn_classes_other))
print classes_learn.shape
ids_test = np.concatenate(
    (test_ids_seed_plants, test_ids_vertebrates, test_ids_arthropods, test_ids_other))
print ids_test.shape
ids_learn = np.concatenate(
    (learn_ids_seed_plants, learn_ids_vertebrates, learn_ids_arthropods, learn_ids_other))
print ids_learn.shape

# scaling features
scaler = StandardScaler()
attributes_learn = scaler.fit_transform(attributes_learn)
attributes_test = scaler.transform(attributes_test)

# preparing crossvalidation splits
skf = KFold(n_splits=5)
seed_plants_indices = []
indices = []
np.set_printoptions(threshold=np.nan)
print len(learn_set_seed_plants), len(learn_classes_vertebrates), len(learn_classes_arthropods), len(
    learn_classes_other)
for train_indices, test_indices in skf.split(learn_set_seed_plants):
    indices.append([train_indices, test_indices])
for i, (train_indices, test_indices) in enumerate(skf.split(learn_set_vertebrates)):
    indices[i][0] = np.concatenate((indices[i][0], train_indices + len(learn_set_seed_plants)))
    indices[i][1] = np.concatenate((indices[i][1], test_indices + len(learn_set_seed_plants)))
for i, (train_indices, test_indices) in enumerate(skf.split(learn_set_arthropods)):
    indices[i][0] = np.concatenate(
        (indices[i][0], train_indices + len(learn_set_seed_plants) + len(learn_set_vertebrates)))
    indices[i][1] = np.concatenate(
        (indices[i][1], test_indices + len(learn_set_seed_plants) + len(learn_set_vertebrates)))
for i, (train_indices, test_indices) in enumerate(skf.split(learn_set_other)):
    indices[i][0] = np.concatenate((indices[i][0],
                                    train_indices + len(learn_set_seed_plants) + len(learn_set_vertebrates) + len(
                                        learn_set_arthropods)))
    indices[i][1] = np.concatenate((indices[i][1],
                                    test_indices + len(learn_set_seed_plants) + len(learn_set_vertebrates) + len(
                                        learn_set_arthropods)))

# check
all_data = []
for i, (l, t) in enumerate(indices):
    assert len(set(l) & set(t)) == 0
    assert len(set(l) | set(t)) == len(attributes_learn)
    all_data.extend(l)
    all_data.extend(t)
assert all([all_data.count(i) == 5 for i in range(len(attributes_learn))])

# serializing data
pickle.dump(attributes_learn, open(os.path.join('..', 'datasets', 'extension_attributes_learn.dump'), 'w'))
pickle.dump(attributes_test, open(os.path.join('..', 'datasets', 'extension_attributes_test.dump'), 'w'))
pickle.dump(classes_learn, open(os.path.join('..', 'datasets', 'extension_classes_learn.dump'), 'w'))
pickle.dump(classes_test, open(os.path.join('..', 'datasets', 'extension_classes_test.dump'), 'w'))
pickle.dump(ids_learn, open(os.path.join('..', 'datasets', 'extension_ids_learn.dump'), 'w'))
pickle.dump(ids_test, open(os.path.join('..', 'datasets', 'extension_ids_test.dump'), 'w'))
pickle.dump(indices, open(os.path.join('..', 'datasets', 'extension_cv_indices.dump'), 'w'))

# host_lineages = [tuple(virus.host_lineage[:10]) for virus in other_eukaryotic_viruses]
# host_lineages_cardinality = [(lineage, host_lineages.count(lineage)) for lineage in set(host_lineages)]
# acc = 0
# for lineage, cardinality in sorted(host_lineages_cardinality, key=lambda x: x[1], reverse=True):
#     acc += cardinality
#     print lineage, cardinality, acc

# host_lineages = [tuple(virus.host_lineage[:11]) for virus in viruses_container if 'Eukaryota' in virus.host_lineage]
# host_lineages_cardinality = [(lineage, host_lineages.count(lineage)) for lineage in set(host_lineages)]
# for lineage, cardinality in sorted(host_lineages_cardinality, key=lambda x: x[1], reverse=True):
#     print lineage, cardinality
# 5282
# (u'cellular organisms', u'Eukaryota', u'Opisthokonta', u'Metazoa', u'Eumetazoa', u'Bilateria', u'Deuterostomia', u'Chordata', u'Craniata', u'Vertebrata', u'Gnathostomata') 1175
# (u'cellular organisms', u'Eukaryota', u'Viridiplantae', u'Streptophyta', u'Streptophytina', u'Embryophyta', u'Tracheophyta', u'Euphyllophyta', u'Spermatophyta', u'Magnoliophyta', u'Mesangiospermae') 1147
# (u'cellular organisms', u'Eukaryota', u'Opisthokonta', u'Metazoa', u'Eumetazoa', u'Bilateria', u'Protostomia', u'Ecdysozoa', u'Panarthropoda', u'Arthropoda', u'Mandibulata') 978
# (u'cellular organisms', u'Eukaryota', u'Opisthokonta', u'Metazoa', u'Eumetazoa', u'Bilateria', u'Protostomia', u'Ecdysozoa', u'Panarthropoda', u'Arthropoda', u'Chelicerata') 149
# (u'cellular organisms', u'Eukaryota', u'Opisthokonta', u'Fungi', u'Dikarya', u'Ascomycota', u'saccharomyceta', u'Pezizomycotina', u'leotiomyceta', u'sordariomyceta', u'Sordariomycetes') 106
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Fungi', 'Dikarya', 'Ascomycota', 'saccharomyceta', 'Pezizomycotina', 'leotiomyceta', 'sordariomyceta', 'Leotiomycetes') 57
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Metazoa', 'Eumetazoa', 'Bilateria', 'Protostomia', 'Lophotrochozoa', 'Mollusca', 'Cephalopoda', 'Coleoidea') 53
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Metazoa', 'Eumetazoa', 'Bilateria', 'Protostomia', 'Lophotrochozoa', 'Mollusca', 'Bivalvia', 'Heteroconchia') 49
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Metazoa', 'Eumetazoa', 'Bilateria', 'Protostomia', 'Lophotrochozoa', 'Mollusca') 42
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Metazoa', 'Eumetazoa', 'Bilateria', 'Protostomia', 'Lophotrochozoa', 'Mollusca', 'Gastropoda', 'Caenogastropoda') 40
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Fungi', 'Dikarya', 'Ascomycota', 'saccharomyceta', 'Pezizomycotina', 'leotiomyceta', 'dothideomyceta', 'Dothideomycetes') 26
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Metazoa', 'Eumetazoa', 'Bilateria', 'Protostomia', 'Ecdysozoa', 'Panarthropoda') 26
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Metazoa', 'Eumetazoa', 'Bilateria', 'Protostomia', 'Lophotrochozoa', 'Annelida', 'Polychaeta') 25
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Fungi', 'Dikarya', 'Ascomycota', 'saccharomyceta', 'Pezizomycotina', 'leotiomyceta', 'Eurotiomycetes', 'Eurotiomycetidae') 22
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Metazoa', 'Eumetazoa', 'Bilateria', 'Protostomia', 'Ecdysozoa', 'Panarthropoda', 'Arthropoda') 16
# ('cellular organisms', 'Eukaryota', 'Viridiplantae', 'Chlorophyta', 'prasinophytes', 'Mamiellophyceae', 'Mamiellales', 'Mamiellaceae', 'Micromonas') 11
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Metazoa', 'Eumetazoa', 'Bilateria', 'Protostomia', 'Ecdysozoa') 11
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Metazoa', 'Eumetazoa', 'Bilateria', 'Protostomia', 'Ecdysozoa', 'Nematoda', 'Chromadorea') 10
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Fungi', 'Dikarya', 'Basidiomycota', 'Agaricomycotina', 'Agaricomycetes', 'Agaricomycetes incertae sedis', 'Cantharellales', 'Ceratobasidiaceae') 10
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Metazoa', 'Eumetazoa', 'Cnidaria', 'Anthozoa', 'Hexacorallia') 9
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Metazoa', 'Eumetazoa', 'Bilateria', 'Protostomia', 'Lophotrochozoa', 'Mollusca', 'Gastropoda', 'Heterobranchia') 8
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Metazoa', 'Eumetazoa', 'Ctenophora', 'Tentaculata', 'Lobata', 'Bolinopsidae', 'Mnemiopsis') 7
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Metazoa', 'Eumetazoa', 'Bilateria', 'Protostomia', 'Lophotrochozoa', 'Annelida', 'Clitellata', 'Oligochaeta') 7
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Metazoa', 'Eumetazoa', 'Bilateria', 'Protostomia', 'Ecdysozoa', 'Nematoda', 'Chromadorea', 'Rhabditida') 6
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Fungi', 'Dikarya', 'Basidiomycota', 'Pucciniomycotina', 'Pucciniomycetes', 'Pucciniales', 'Cronartiaceae', 'Cronartium') 5
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Metazoa', 'Eumetazoa', 'Bilateria', 'Protostomia', 'Ecdysozoa', 'Nematoda', 'Chromadorea', 'Ascaridida') 5
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Metazoa', 'Eumetazoa', 'Bilateria', 'Protostomia', 'Ecdysozoa', 'Nematoda', 'Enoplea', 'Dorylaimia') 4
# ('cellular organisms', 'Eukaryota', 'Stramenopiles', 'Oomycetes', 'Peronosporales', 'Phytophthora') 4
# ('cellular organisms', 'Eukaryota', 'Parabasalia', 'Trichomonadida', 'Trichomonadidae', 'Trichomonas') 4
# ('cellular organisms', 'Eukaryota', 'Stramenopiles', 'Bacillariophyta', 'Coscinodiscophyceae', 'Chaetocerotophycidae', 'Chaetocerotales', 'Chaetocerotaceae', 'Chaetoceros') 4
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Metazoa', 'Eumetazoa', 'Bilateria', 'Deuterostomia', 'Chordata') 3
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Fungi', 'Dikarya', 'Ascomycota', 'saccharomyceta', 'Saccharomycotina', 'Saccharomycetes', 'Saccharomycetales', 'Saccharomycetaceae') 3
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Metazoa', 'Eumetazoa', 'Bilateria', 'Protostomia', 'Lophotrochozoa', 'Annelida', 'Polychaeta', 'Palpata') 3
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Fungi', 'Dikarya', 'Basidiomycota', 'Agaricomycotina', 'Agaricomycetes', 'Agaricomycetidae', 'Agaricales', 'Pleurotaceae') 2
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Metazoa', 'Eumetazoa', 'Bilateria', 'Protostomia', 'Lophotrochozoa', 'Mollusca', 'Gastropoda', 'Vetigastropoda') 2
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Fungi', 'Dikarya', 'Ascomycota', 'saccharomyceta', 'Pezizomycotina', 'Pezizomycetes', 'Pezizales', 'Tuberaceae') 2
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Metazoa', 'Eumetazoa', 'Bilateria', 'Platyhelminthes', 'Rhabditophora', 'Seriata', 'Tricladida', 'Continenticola') 2
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Fungi', 'Dikarya', 'Basidiomycota', 'Agaricomycotina', 'Agaricomycetes', 'Agaricomycetes incertae sedis', 'Thelephorales', 'Typhulaceae') 2
# (u'cellular organisms', u'Eukaryota', u'Opisthokonta', u'Fungi', u'Dikarya', u'Basidiomycota', u'Agaricomycotina', u'Agaricomycetes', u'Agaricomycetes incertae sedis', u'Russulales', u'Bondarzewiaceae') 2
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Metazoa', 'Eumetazoa', 'Bilateria', 'Protostomia', 'Ecdysozoa', 'Nematoda', 'Chromadorea', 'Tylenchida') 2
# ('cellular organisms', 'Eukaryota', 'Rhodophyta', 'Florideophyceae', 'Bonnemaisoniales', 'Bonnemaisoniaceae', 'Delisea') 1
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Metazoa', 'Eumetazoa', 'Bilateria', 'Deuterostomia', 'Echinodermata', 'Eleutherozoa', 'Echinozoa', 'Echinoidea') 1
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Fungi', 'Dikarya', 'Basidiomycota', 'Agaricomycotina', 'Tremellomycetes', 'Cystofilobasidiales', 'Mrakiaceae', 'Xanthophyllomyces') 1
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Metazoa', 'Porifera', 'Demospongiae', 'Heteroscleromorpha', 'Poecilosclerida', 'Microcionidae', 'Artemisina') 1
# ('cellular organisms', 'Eukaryota', 'Euglenozoa', 'Kinetoplastida', 'Trypanosomatidae', 'Phytomonas') 1
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Fungi', 'Dikarya', 'Basidiomycota', 'Agaricomycotina', 'Agaricomycetes', 'Agaricomycetes incertae sedis', 'Thelephorales', 'Thelephoraceae') 1
# ('cellular organisms', 'Eukaryota', 'Alveolata', 'Apicomplexa', 'Conoidasida', 'Coccidia', 'Eucoccidiorida', 'Eimeriorina', 'Eimeriidae', 'Eimeria') 1
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Metazoa', 'Eumetazoa', 'Cnidaria', 'Anthozoa', 'Octocorallia', 'Alcyonacea', 'Calcaxonia', 'Primnoidae') 1
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Metazoa', 'Eumetazoa', 'Cnidaria', 'Anthozoa', 'Hexacorallia', 'Actiniaria', 'Aiptasiidae') 1
# ('cellular organisms', 'Eukaryota', 'Euglenozoa', 'Kinetoplastida', 'Trypanosomatidae', 'Leishmaniinae', 'Leishmania', 'Leishmania', 'Leishmania aethiopica species complex') 1
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Metazoa', 'Eumetazoa', 'Bilateria', 'Deuterostomia', 'Chordata', 'Tunicata', 'Ascidiacea', 'Enterogona') 1
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Fungi', 'Dikarya', 'Basidiomycota', 'Agaricomycotina', 'Agaricomycetes', 'Agaricomycetidae', 'Agaricales', 'Tricholomataceae') 1
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Fungi', 'Microsporidia', 'Microsporidia incertae sedis', 'Antonospora') 1
# (u'cellular organisms', u'Eukaryota', u'Stramenopiles', u'Bicosoecida', u'Cafeteriaceae', u'Cafeteria') 1
# ('cellular organisms', 'Eukaryota', 'Euglenozoa', 'Kinetoplastida', 'Trypanosomatidae', 'Leishmaniinae', 'Leishmania', 'Leishmania', 'Leishmania major species complex') 1
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Metazoa', 'Eumetazoa', 'Cnidaria', 'Anthozoa', 'Octocorallia', 'Alcyonacea', 'Holaxonia', 'Paramuriceidae') 1
# ('cellular organisms', 'Eukaryota', 'Opisthokonta', 'Metazoa', 'Eumetazoa', 'Bilateria', 'Protostomia', 'Lophotrochozoa', 'Mollusca', 'Bivalvia', 'Pteriomorphia') 1

# for x in viruses_container:
#     print x.__dict__
