#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import warnings
import cPickle as pickle
import numpy as np
from rpy2.robjects import numpy2ri
from rpy2.robjects import r

with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	r.library('penalizedSVM')
