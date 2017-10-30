#! /usr/bin/python
# -*- coding: utf-8 -*-
import math
import random


def rename_host_lineage(hlin):
    if 'Archaea' in hlin or 'Bacteria' in hlin:
        return 'phage'
    elif 'Eukaryota' in hlin:
        return 'Eucaryota-infecting'


def groupping(hlineages):
    ret = []
    pairs = set([x[:2] for x in hlineages])
    for pair in pairs:
        ret.append((pair[0], pair[1], sum(z[2] for z in hlineages if z[:2] == pair)), )
    return ret


def dataset_creation(container, lineage_fragments, what):
    # lineage_fragments np. 'Microviridae'
    # what - either phage or 'Eucaryota-infecting'
    blind_set = []
    for lineage_fragment in lineage_fragments:
        blind_set.extend([virus for virus in container if
                          lineage_fragment in virus.lineage and rename_host_lineage(virus.host_lineage) == what])
    return [virus for virus in container if not virus in blind_set], blind_set


def f(x, power):
    return math.pow(math.e, (-math.pow(x, power)))


def g(power, xs, cardinality):
    return math.fabs(sum(sum(f(i, power) for i in range(x)) for x in xs) - cardinality)


def get_virus(left_viruses, lin, power):
    '''
    Gets
    '''
    get_or_not = lambda x: x >= random.random()
    viruses_for_the_lineage = [virus for virus in left_viruses if tuple(virus.lineage) == lin]
    # print lin, [(i, f(i, power), get_or_not(f(i, power))) for i, virus in enumerate(viruses_for_the_lineage)]
    choosen_viruses = [virus for i, virus in enumerate(viruses_for_the_lineage) if get_or_not(f(i, power))]
    left_viruses = [virus for virus in left_viruses if virus not in set(choosen_viruses)]
    return choosen_viruses, left_viruses
