#!/usr/bin/env python3

'''
Copyright (C) 2020 Simon D. Levy

MIT License
'''

import numpy as np
from tuningcurves import curve

N = 10

e = np.random.uniform(-1, +1, (2,N))

alpha = np.random.uniform(0, 100, N)

b = np.random.uniform(-20,+20, N)

np.set_printoptions(precision=3)

d = np.random.uniform(-1, +1, (N,1))

for x in (0,0), (0,1), (1,0), (1,1):
    a  = curve(alpha, e, x, b)
    fx = np.dot(a, d)

    print(fx, np.tanh(fx))

