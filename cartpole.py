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

    def __init__(self, neurons=10):

        NefGym.__init__(self, 'CartPole-v0', 4, neurons)

    def _get_action(self, params, obs):

        a  = self._curve(obs)

        d = params

        return self.activate(np.dot(a, d))

    def activate(self, x):

        return 1 if np.tanh(x) > 0 else 0

if __name__ == '__main__':

    import random
    random.seed(0)
    np.random.seed(0)

    problem = NefCartPole()

    ga = Elitist(problem, 2048)

    best = ga.run(2, max_fitness=2000)

    #print('Got reward %.3f in %d steps' % problem.run_episode(best, render=True))

