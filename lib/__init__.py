import numpy as np

class NefGym:

    def __init__(self, obs_size, neurons):

        # Encoder
        self.alpha = np.random.uniform(0, 100, neurons) # tuning parameter alpha
        self.b = np.random.uniform(-20,+20, neurons)    # tuning parameter b
        self.e = np.random.uniform(-1, +1, (obs_size,neurons)) # encoder weights

        self.neurons = neurons

    def new_params(self):

       return np.random.uniform(-1, +1, (self.neurons,1)) # decoder weights
