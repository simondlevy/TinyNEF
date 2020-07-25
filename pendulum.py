#!/usr/bin/env python3
'''
Use the Neural Engineering framework to solve Pendulum via an elitist GA

Copyright (C) 2020 Simon D. Levy

MIT License
'''

from lib import NefGym
import numpy as np

from sueap.elitist import Elitist

class NefPendulum(NefGym):

    def __init__(self, neurons=20):

        NefGym.__init__(self, 'Pendulum-v0', 3, neurons)

    def _get_action(self, params, obs):

        return self.activate(np.dot(self._curve(obs), params))

    def activate(self, x):

        return np.clip(x, -2, +2)

if __name__ == '__main__':

    problem = NefPendulum()

    ga = Elitist(problem, 2048)

    best = ga.run(3, max_fitness=-1500)

    #print('Got reward %.3f in %d steps' % problem.run_episode(best, render=True))

