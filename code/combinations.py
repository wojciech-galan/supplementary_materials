#! /usr/bin/python
# -*- coding: utf-8 -*-

from math import factorial
from decimal import Decimal

s = 0
# https://en.wikipedia.org/wiki/Combination
n = 101 # 101 features
for k in range(1, n + 1):
    s += factorial(n)/((factorial(k)*factorial(n-k)))
print '%.2E' % Decimal(s) # 2.54E+30

s = 0
# https://en.wikipedia.org/wiki/Combination
n = 357 # 101 + 256 features
for k in range(1, n + 1):
    s += factorial(n)/((factorial(k)*factorial(n-k)))
print '%.2E' % Decimal(s) # 2.94E+107