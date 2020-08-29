#!/usr/bin/env python3
'''
Use the Neural Engineering framework to solve CartPole via an elitist GA

Copyright (C) 2020 Simon D. Levy

MIT License
'''

from lib import NefGym
from sys import argv
import pickle
import numpy as np

from sueap.algorithms.elitist import Elitist

class NefCartPole(NefGym):

    def __init__(self, neurons=10, seed=None):

        NefGym.__init__(self, 'CartPole-v0', neurons, seed)

    def activate(self, x):

        return 1 if np.tanh(x) > 0 else 0

if __name__ == '__main__':

    if len(argv) < 2:
        print('Usage: python3 %s FILE' % argv[0])
        exit(0)
    
    problem = NefCartPole()
    net = pickle.load(open(argv[1], 'rb'))
    print('Got reward %.3f in %d steps' % problem.test(net, render=True))
