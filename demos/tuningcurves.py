#!/usr/bin/env python3
'''
Generate and plot some tuning curves for
Leaky Integrate-and-Fire neurons

Simon D. Levy

CSCI 252

Fall 2018
'''

import numpy as np
import matplotlib.pyplot as plt

def G(v):

    v[v<=0] = np.finfo(float).eps

    g = 10 * np.log(np.abs(v))

    g[g<0] = 0

    return  g

def curve(alpha, e, x, b):

    return G(alpha * np.dot(x, e) + b)

def integrate(x, leak=0):
    n = len(x)
    y = np.zeros(n)
    for k in range(1,n):
        y[k] = y[k-1] + x[k] - leak*y[k-1] 
    return y

def plotcurve(alpha, e, x, b, style='b'):

    plt.plot(x, curve(alpha, e, x, b), style)

def finishplot():

    plt.xlim([-1,+1])
    plt.ylim([.1,50])
    plt.xlabel('x')
    plt.ylabel('Firing rate (Hz)')
    plt.show()    

def plotcurves(n):

    x = np.linspace(-1,+1,500)

    for k in range(n):

        # +1 or -1
        e = 2 * np.random.randint(2) - 1

        alpha = np.random.uniform(0, 100)

        b = np.random.uniform(-20,+20)

        plt.plot(x, curve(alpha, e, x, b))
            
    finishplot()

def getspikes(alpha, e, b, t, x):

    # Get firing rates from Equation (1)
    rates = curve(alpha, e, x, b)

    # Ignore rates below a reasonable value
    okay = rates > 20
    rates = rates[okay]
    times = np.copy(t[okay])
    
    # We'll keep track of the time at which a spike happens
    spiked = False
    tnext = 0
    spiketimes = np.array([])

    # Loop over each time and its corresponding rate
    for rcurr,tcurr in zip(rates, times):

        # If we've spiked within the current interval
        if spiked:

            # and it's time for a new spike
            if tcurr >= tnext:

                # reset the already-spiked flag
                spiked = False

        # If we've haven't spiked within the current interval
        else:

            # add the spike at the current time
            spiketimes = np.append(spiketimes, tcurr)

            # flag that we've spiked
            spiked = True

            # compute the end of the current interval
            tnext = tcurr + 1/rcurr

    # Convert spike times into spike train
    n = len(t)
    spikes = np.zeros(n)
    spikes[(spiketimes*n).astype(int)] = 1
    return spikes

def plotspikes(t, spikes):

    plt.subplot(2,1,1)
    plt.plot(t, spikes)
    plt.subplot(2,1,2)
    plt.plot(t, x)
    plt.ylabel('x')
    plt.xlabel('Time (sec)')
    plt.show()

def plotspikes2(t, spikes1, spikes2, x):

    plt.subplot(3,1,1)
    plt.plot(t, spikes1, 'b')
    plt.subplot(3,1,2)
    plt.plot(t, spikes2, 'g')
    plt.subplot(3,1,3)
    plt.plot(t, x)
    plt.plot([0,t[-1]], [0,0], 'k--')
    plt.ylabel('x')
    plt.xlabel('Time (sec)')
    plt.show()        

def plotspikes3(t, spikes1, spikes2, integrated, orig):

    plt.subplot(3,1,1)
    plt.plot(t, spikes1, 'b')
    plt.subplot(3,1,2)
    plt.plot(t, spikes2, 'g')
    plt.subplot(3,1,3)
    plt.plot(integrated)
    plt.plot(orig, 'r--')
    plt.ylabel('x')
    plt.xlabel('Time (sec)')
    plt.show()

if __name__ == '__main__':

    plotcurves(10)

    FS = 1000
    t = np.linspace(0.01, .99, FS)
    x = np.linspace(+1,-1,FS)

    alpha = 50
    b = +10
    
    plotcurve(alpha, +1, x, b, 'b')
    plotcurve(alpha, -1, x, b, 'g')
    finishplot()

    x = np.sin(2*np.pi*t)
    spikes1 =  getspikes(alpha, +1, b, t, x)
    spikes2 = -getspikes(alpha, -1, b, t, x)
    plotspikes2(t, spikes1, spikes2, x)
    
    plotspikes3(t, spikes1, spikes2, integrate(spikes1+spikes2, .05), x)

