#!/usr/bin/env python3
'''
Use the Neural Engineering framework to solve CartPole via an elitist GA

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import numpy as np

from lib import NefGym

class NefCartPole(NefGym):

    def __init__(self):

        NefGym.__init__(self, 'CartPole-v0', 1)

    def activation(self, x):

        return 1 if np.tanh(x) > 0 else 0

if __name__ == '__main__':

    problem = NefCartPole()

    print('Got reward %.3f in %d steps' % problem.run_episode(problem.new_params(), render=True))

    best = problem.learn(10)

    print('Got reward %.3f in %d steps' % problem.run_episode(best, render=True))
