#!/usr/bin/env python3
'''
Use the Neural Engineering framework to solve Pendulum via an elitist GA

Copyright (C) 2020 Simon D. Levy

MIT License
'''

from lib import NefGym
from sys import argv
import pickle
import numpy as np

from sueap.elitist import Elitist

class NefPendulum(NefGym):

    def __init__(self, neurons=20, seed=None):

        NefGym.__init__(self, 'Pendulum-v0', neurons, seed)

    def activate(self, x):

        return np.clip(x, -2, +2)

if __name__ == '__main__':

    problem = NefPendulum()

    if len(argv) < 2:
        ga = Elitist(problem, 2048, save_name='pendulum')
        ga.run(80, max_fitness=-1500)
    else:
        net = pickle.load(open(argv[1], 'rb'))
        print('Got reward %.3f in %d steps' % problem.test(net))


