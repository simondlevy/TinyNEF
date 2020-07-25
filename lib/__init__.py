'''
A superclass for working with OpenAI Gym environments using the Neural Engineering Framework

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import gym
import numpy as np

class NefGym:

    def __init__(self, env, neurons, seed):

        # Seed random-number generator if indicated
        if seed is not None:
            np.random.seed(seed)

        # Get observation size from environment
        obs_size = gym.make(env).observation_space.shape[0]

        # Encoder
        self.alpha = np.random.uniform(0, 100, neurons) # tuning parameter alpha
        self.b = np.random.uniform(-20,+20, neurons)    # tuning parameter b
        self.e = np.random.uniform(-1, +1, (obs_size,neurons)) # encoder weights

        self.seed = seed
        self.env = env
        self.neurons = neurons

    def new_params(self):

        return np.random.uniform(-1, +1, (self.neurons,1)) # decoder weights

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
        env = gym.make(self.env)

        if self.seed is not None:
            env.seed(self.seed)

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

        return self.activate(np.dot(self._curve(obs), params))

    def _curve(self, x):

        return NefGym._G(self.alpha * np.dot(x, self.e) + self.b)

    @staticmethod
    def _G(v):

        v[v<=0] = np.finfo(float).eps

        g = 10 * np.log(np.abs(v))

        g[g<0] = 0

        return  g

