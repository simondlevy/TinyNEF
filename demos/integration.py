#!/usr/bin/env python3
'''
Leaky integration vs. standard integration

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import numpy as np
import matplotlib.pyplot as plt

def integrate(x, leak=0):
    n = len(x)
    y = np.zeros(n)
    for k in range(1,n):
        y[k] = y[k-1] + x[k] - leak*y[k-1] 
    return y

N = 50

# Generate a constant signal
x = .3*np.ones(N)

# Plot the signal
plt.subplot(3,1,1)
plt.ylim([0,1.1])
plt.plot(x)

# Plot standard integration
plt.subplot(3,1,2)
plt.plot(integrate(x))

# Plot leaky integration
plt.subplot(3,1,3)
plt.plot(integrate(x,.1))

plt.show()
    

