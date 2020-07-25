#!/usr/bin/env python3
'''
Use the Neural Engineering framework to solve CartPole via an elitist GA

Copyright (C) 2020 Simon D. Levy

MIT License
'''

from lib import NefGym
import numpy as np

from sueap.elitist import Elitist

class NefCartPole(NefGym):

    def __init__(self, neurons=10, seed=None):

        NefGym.__init__(self, 'CartPole-v0', neurons, seed)

    def activate(self, x):

        return 1 if np.tanh(x) > 0 else 0

if __name__ == '__main__':

    problem = NefCartPole()

    ga = Elitist(problem, 2048, save_name='cartpole')

    best = ga.run(10, max_fitness=2000)

    #print('Got reward %.3f in %d steps' % problem.run_episode(best, render=True))

