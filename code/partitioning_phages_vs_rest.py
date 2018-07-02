#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import sqlite3
import cPickle as pickle

from scipy.optimize import minimize
from viral_seq_fetcher.src import SeqContainer

from code.lib import *

dirname = os.path.join('..', 'datasets')
fname = 'container_Fri_Oct__6_14:26:35_2017.dump'
container = SeqContainer.Container.fromFile(os.path.join(dirname, fname))
attribs = ('host_lineage', 'lineage', 'nuc_frequencies', 'relative_nuc_frequencies_one_strand',
           'relative_trinuc_freqs_one_strand')
desired_container = SeqContainer.Container([x for x in container if all(x.__dict__[attrib] for attrib in attribs)])
container = None

with sqlite3.connect(':memory:') as conn:
    conn.execute('''CREATE TABLE lineages (id INT PRIMAMY KEY, lin VARCHAR)''')
    conn.execute('''CREATE TABLE host_lineages (id INT PRIMAMY KEY, hlin VARCHAR)''')
    for virus in desired_container:
        conn.execute('''INSERT INTO lineages VALUES(?, ?);''', (virus.gi, str(virus.lineage[:5])))
    conn.commit()
    for virus in desired_container:
        conn.execute('''INSERT INTO host_lineages VALUES(?, ?);''', (virus.gi, str(virus.host_lineage[:2])))
    conn.commit()
    cur = conn.cursor()
    cur.execute('''SELECT lineages.lin, host_lineages.hlin, COUNT(*) FROM
                        lineages JOIN host_lineages ON lineages.id==host_lineages.id
                        GROUP BY lineages.lin, host_lineages.hlin''')
    results = cur.fetchall()

processed_results = []  # get rid of viruses infecting viruses, puting phage in one group
for result in results:
    if not 'Viruses' in result[1]:
        processed_results.append((tuple(eval(result[0])), rename_host_lineage(eval(result[1])), result[2]), )

pickle.dump(processed_results, open(os.path.join('..', 'datasets', 'viruses_without_viral_renamed_lineage.dump'), 'w'))


phage_cardinality = sum([x[2] for x in processed_results if x[1] == 'phage'])
eucarya_cardinality = sum([x[2] for x in processed_results if x[1] == 'Eucaryota-infecting'])
group_cardinality = min(phage_cardinality, eucarya_cardinality)
blind_cardinality = round(0.2 * group_cardinality)
cv_cardinality = 0.2 * (group_cardinality - blind_cardinality)
blind_set_phages = []
print blind_cardinality, cv_cardinality
print phage_cardinality, 'phages,', eucarya_cardinality, 'eucaryota-infecting'

# phages
container_left, blind_set_phages = dataset_creation(desired_container, [
    'Microviridae', 'Cystovirus', 'Allolevivirus',  # ssDNA, dsRNA, ssRNA
    'Picovirinae', 'P22virus', 'Autographivirinae',
    'Phietavirus', 'Sk1virus', 'Andromedavirus',
    'unclassified archaeal viruses', 'unclassified bacterial viruses',
    'Peduovirinae', 'Vequintavirinae'  # Myoviridae
], 'phage')
cv_phage = sorted([virus for virus in container_left if rename_host_lineage(virus.host_lineage) == 'phage'],
                  key=lambda x: x.lineage)
print len(cv_phage), len(blind_set_phages), len(cv_phage) + len(blind_set_phages)
print len([x for x in desired_container if rename_host_lineage(x.host_lineage) == 'phage'])

# Eukaryota-invecting
blind_set_eu, cv_eu = [], []
container_left, blind_set_eu = dataset_creation(desired_container, [
    'Virus families not assigned to an order',
    'Virus-associated RNAs',
    'unclassified ssRNA viruses',
    'Togaviridae', 'Idaeovirus', 'Nodaviridae', 'Dicistroviridae',  # ssRNA+ viruses
    'Ophioviridae', 'Pneumoviridae', 'Filoviridae', 'Bornaviridae', 'Arenaviridae',  # ssRNA- viruses
    'Gammatorquevirus', 'Babuvirus', 'unclassified Geminiviridae', 'Densovirinae',
    'unclassified Parvoviridae', 'Curtovirus', 'Turncurtovirus',  # ssDNA viruses
    'Hypoviridae', 'Aquareovirus',  # dsRNA viruses
    'Nudiviridae', 'Lambdapapillomavirus', 'Aviadenovirus', 'Iridoviridae', 'Betaherpesvirinae',
    'unclassified dsDNA viruses', 'unclassified Betapolyomavirus',  # dsDNA viruses, no RNA stage
    'Babu- and nanovirus-associated alphasatellites', 'Double-stranded RNA satellites',  # Satellites
    'Hepadnaviridae',  # Retro-transcribing viruses
    'Gemykibivirus'  # Genomoviridae
], 'Eucaryota-infecting')
print len(desired_container), len(container_left)
eu_lineages = []
for virus in container_left:
    if rename_host_lineage(virus.host_lineage) == 'Eucaryota-infecting':
        eu_lineages.append(tuple(virus.lineage))
eu_lineages_cardinalities = [eu_lineages.count(x) for x in set(eu_lineages)]
power = minimize(g, 1, args=(eu_lineages_cardinalities, cv_cardinality * 5), bounds=((0, None),)).x
eu_viruses_left = [virus for virus in container_left if
                   rename_host_lineage(virus.host_lineage) == 'Eucaryota-infecting']
for x in sorted([(x, eu_lineages.count(x)) for x in set(eu_lineages)], key=lambda x: x[1])[:-1]:
    chosen, eu_viruses_left = get_virus(eu_viruses_left, x[0], power)
    cv_eu.extend(chosen)
    # print x, sum(f(i, power) for i in range(x[1])), len(chosen)
last_group_cardinality = int(5 * cv_cardinality - len(cv_eu))
x = max([(x, eu_lineages.count(x)) for x in set(eu_lineages)], key=lambda x: x[1])
viruses_belonging_to_the_biggest_group = [virus for virus in eu_viruses_left if tuple(virus.lineage) == x[0]]
print last_group_cardinality, 5 * cv_cardinality, blind_cardinality, len(cv_eu)
assert last_group_cardinality > 0
random.shuffle(viruses_belonging_to_the_biggest_group)
cv_eu.extend(viruses_belonging_to_the_biggest_group[:last_group_cardinality])
assert len(cv_eu) == 5 * cv_cardinality
cv_eu.sort(key=lambda x:x.lineage)

pickle.dump(blind_set_eu, open(os.path.join('..', 'datasets', 'viruses_blind_set_eu.dump'), 'w'))
pickle.dump(blind_set_phages, open(os.path.join('..', 'datasets', 'viruses_blind_set_phages.dump'), 'w'))
pickle.dump(cv_eu, open(os.path.join('..', 'datasets', 'viruses_crossvalidation_eu.dump'), 'w'))
pickle.dump(cv_phage, open(os.path.join('..', 'datasets', 'viruses_crossvalidation_phages.dump'), 'w'))
