#!/usr/bin/env python3
'''
Use the Neural Engineering framework to solve Pendulum via an elitist GA

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import numpy as np

from lib import NefGym

class NefPendulum(NefGym):

    def __init__(self, neurons=20):

        NefGym.__init__(self, 'Pendulum-v0', 1, neurons=neurons)

    def activation(self, x):

        return np.clip(x, -2, +2)

if __name__ == '__main__':

    problem = NefPendulum()

    best = problem.learn(80)

    print('Got reward %.3f in %d steps' % problem.run_episode(best, render=True))
