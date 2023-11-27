"""
Module: visualizer
Author: Petr Nemec
License: 

Some functions were taken from or inspired by the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP) and libfmp.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_signal_in_time(x, Fs):
    """Plots the inputed signal in time

    Args:
        x(np.ndarray): array with signal values in time
        Fs (int): sample rate        
    """

    L = np.size(x[0])#only left channel
    t = np.arange(L) / Fs

    plt.plot(t, x[0])
    plt.show()



