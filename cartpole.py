#!/usr/bin/env python3
'''
Use the Neural Engineering framework to solve CartPole via an elitist GA

Copyright (C) 2020 Simon D. Levy

MIT License
'''

from lib import NefGym
import numpy as np
import gym

from sueap.elitist import Elitist

class NefCartPole(NefGym):

    def __init__(self, neurons=10):

        NefGym.__init__(self, 4, neurons)


    def run_episode(self, params, env='CartPole-v0', render=False):

        # Build env
        env = gym.make(env)
        obs = env.reset()

        episode_reward, episode_steps = 0,0

        # Simulation loop
        while True:

            action = self._get_action(params, obs)

            # Optional render of environment
            if render:
                env.render()

            # Do environment step
            obs, reward, done, _ = env.step(action)

            episode_reward += reward
            episode_steps += 1

            # Episode end
            if done:
                break

        # Cleanup
        env.close()

        return episode_reward, episode_steps

    def _get_action(self, params, obs):

        a  = self._curve(obs)

        d = params

        return 1 if np.tanh(np.dot(a, d)) > 0 else 0

    def _curve(self, x):

        return NefCartPole._G(self.alpha * np.dot(x, self.e) + self.b)

    @staticmethod
    def _G(v):

        v[v<=0] = np.finfo(float).eps

        g = 10 * np.log(np.abs(v))

        g[g<0] = 0

        return  g

if __name__ == '__main__':

    problem = NefCartPole()

    ga = Elitist(problem, 2048)

    best = ga.run(10, max_fitness=2000)

    print('Got reward %.3f in %d steps' % problem.run_episode(best, render=True))

