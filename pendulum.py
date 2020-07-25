#!/usr/bin/env python3
'''
Use the Neural Engineering framework to solve Pendulum via an elitist GA

Copyright (C) 2020 Simon D. Levy

MIT License
'''

from lib import NefGym
import numpy as np
import gym

from sueap.elitist import Elitist

class NefPendulum(NefGym):

    def __init__(self, neurons=20):

        NefGym.__init__(self, 3, neurons)

    def run_episode(self, params, env='Pendulum-v0', render=False):

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

        return np.clip(np.dot(a, d), -2, +2)

    def _curve(self, x):

        return NefGym._G(self.alpha * np.dot(x, self.e) + self.b)

if __name__ == '__main__':

    problem = NefPendulum()

    ga = Elitist(problem, 2048)

    best = ga.run(80, max_fitness=-1500)

    print('Got reward %.3f in %d steps' % problem.run_episode(best, render=True))

