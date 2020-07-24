'''
Abstract class for using TinyNEF and a genetic algorithm to solve problems in OpenAI gym

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import numpy as np
import gym

from sueap.elitist import Elitist

class NefGym:

    def __init__(self, env_name, action_size, neurons=10):

        env = gym.make(env_name)

        # Encoder
        self.alpha = np.random.uniform(0, 100, neurons) # tuning parameter alpha
        self.b = np.random.uniform(-20,+20, neurons)    # tuning parameter b
        self.e = np.random.uniform(-1, +1, (env.observation_space.shape[0],neurons)) # encoder weights

        # Genetic Algorithm
        self.ga = Elitist(self, 2000)

        # Stuff for later
        self.neurons = neurons
        self.action_size = action_size
        self.env_name = env_name

    def new_params(self):

       return np.random.uniform(-1, +1, (self.neurons,self.action_size)) # decoder weights

    def eval_params(self, params, episodes=10):

        total_reward = 0
        total_steps = 0

        for _ in range(episodes):

            episode_reward, episode_steps = self.run_episode(params)

            total_reward += episode_reward
            total_steps += episode_steps

        return total_reward, total_steps


    def mutate_params(self, params, noise_std):

        d = params

        return d+noise_std*np.random.randn(*d.shape)

    def run_episode(self, params, render=False):

        # Build env
        env = gym.make(self.env_name)
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

        return self.activation(np.tanh(np.dot(a, d)))

    def _curve(self, x):

        return NefGym._G(self.alpha * np.dot(x, self.e) + self.b)

    @staticmethod
    def _G(v):

        v[v<=0] = np.finfo(float).eps

        g = 10 * np.log(np.abs(v))

        g[g<0] = 0

        return  g

    def learn(self, ngen):

        return self.ga.run(ngen, max_fitness=2000)
