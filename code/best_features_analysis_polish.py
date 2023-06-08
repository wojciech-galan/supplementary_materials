#! /usr/bin/python

# run with Python3 !!!

import os
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    plt.hist([0] * 917675 + [1] * 79291 + [2] * 2965 + [3] * 67 + [4] * 2, 4, align='left', log=True)
    fig, ax = plt.subplots()
    ax.set_yscale('log', base=10)
    ax.hist([0] * 917675 + [1] * 79291 + [2] * 2965 + [3] * 67 + [4] * 2, [-.5, .5, 1.5, 2.5, 3.5, 4.5], align='mid')
    ax.set_xlabel(u'Liczba wspólnych elementów')
    ax.set_ylabel(u'P wystąpienia danej ilości elementów wspólnych')
    ax.grid(True)
    ax.set_ylim([1, 10 ** 6])
    ax.set_xlim([-.5, 4.5])
    nums = [917675, 79291, 2965, 67, 2]
    ax.set_yticks(nums, labels=np.array(nums)/1000000)
    plt.savefig(os.path.join('..', 'figures', 'intersections_histogram_polish.png'), bbox_inches='tight')
