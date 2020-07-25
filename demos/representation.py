#!/usr/bin/env python3
'''
Illustrates NEF Principle 1: A population of neurons collectively represents a
time-varying vector of real numbers through non-linear encoding and linear
decoding.

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import numpy as np
import matplotlib.pyplot as plt

# Parameters to experiment with
N_NEURONS = 200
N_SAMPLES = 100
RADIUS    = 30

# Scalar version of nonlinear response function
def G(x):

    if x <= 0:

        x = np.finfo(float).eps

    return max(0, 10 * np.log(np.abs(x)))

# Randomly samlples points from a circle
theta = np.random.uniform(0,2*np.pi,N_SAMPLES)
x = np.zeros((N_SAMPLES,2))
x[:,0] = np.cos(theta) * RADIUS
x[:,1] = np.sin(theta) * RADIUS

# Display the samples
plt.scatter(x[:,0], x[:,1], s=1)
plt.axis('equal')
plt.show()

# Generate response-curve samples for each neuron
a = np.zeros((N_NEURONS,N_SAMPLES))
for i in range(N_NEURONS):

    # Create random encoder, parameters for this neuron
    e = np.random.randint(0,2,2)*2-1
    alpha = np.random.uniform(0, 100)
    b = np.random.uniform(-20,+20)

    # Generate samples for this neuron
    for k in range(N_SAMPLES):
        a[i,k] = G(alpha * np.dot(e, x[k]) + b)
