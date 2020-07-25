#!/usr/bin/env python3
'''
lif.py Leaky Integrate and Fire model of neural spiking

Adapted from http://tips.vhlab.org/techniques-and-tricks/matlab/integrate-and-fire

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Params to fiddle with
dur     = 0.1    # duration in seconds
dt      = 0.0002 # delta-t in seconds
V_reset = -0.080 # -80mV
V_th    = -0.040 # -40mV
V_e     = -0.075 # -75mV
Rm      = 10e6   # membrane resistance
tau_m   = 10e-3  # membrane time constant
    
def lif(T, Im):
    '''
    Leaky Integrate-and-Fire function
    
    Input:  T  = time values
            Im = input current in amps
    Output: Vm = volts
    '''

    Vm = np.array([V_reset])

    for t in range(len(T)-1):
            
        Vm = np.append(Vm, V_reset if Vm[t] > V_th else Vm[t] + dt * (-(Vm[t]-V_e)+Im*Rm) / tau_m)

    return Vm
    
def main():

    # Generate time values
    T = np.arange(0,dur+dt,dt) 

    # Part 1: Plot voltage over time for a constant input current
    Im = 5e-9   # 5 nA
    Vm = lif(T, Im)
    plt.plot(T, Vm)
    plt.xlabel('Time(s)')
    plt.ylabel('Voltage (V)')
    plt.title('Im = ' + str(Im) + 'A')
    plt.show()

    # Part 2: Plot spiking-rate respose for various input currents
    Imvals = np.linspace(5e-9, 2e-8, 400)
    Hzvals = np.array([])
    for Im in Imvals:
        Vm = lif(T, Im)
        mn = np.mean(Vm)
        Hzvals = np.append(Hzvals, len(signal.find_peaks(Vm)[0])/dur)
    plt.plot(Imvals, Hzvals)
    plt.xlabel('Input current (Amps)')
    plt.ylabel('Firing rate (Hz)')
    plt.show()

if __name__ == '__main__':

    main()    

    
    
    
        


    
