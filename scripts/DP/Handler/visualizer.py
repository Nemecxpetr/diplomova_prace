"""
Module: visualizer
Author: Petr Nemec
License: 

Some functions were taken from or inspired by the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP) and libfmp.
"""

from math import log10
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

def plot_spectrograph(x, Fs):
    """Plots spectrograph of inputet signal
    """
    X = np.fft.rfft(x)
    f_fft = np.fft.rfftfreq(len(x), 1/Fs)
    plt.semilogx(f_fft, 20*np.log10(np.abs(X)))
    plt.ylabel('Modul [dB]')
    plt.xlabel('Frekvence [Hz]')
    #plt.ylabel('Log amplitude [dB]')
    #plt.xlabel('Frequency [Hz]')
    plt.xlim([20, Fs/2])
    plt.show()

def plot_spectrograph_phase(x, Fs):
    """Plots spectrograph of inputet signal
    """
    # lets use only left channel
    x = x.T[0]
    X = np.fft.rfft(x)
    f_fft = np.fft.rfftfreq(len(x), 1/Fs)

    fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True)
    ax[0].semilogx(f_fft, X.real)
    ax[0].set_ylabel('Realna slozka')
    #plt.ylabel('Log amplitude [dB]')
    #plt.xlabel('Frequency [Hz]')


    ax[1].semilogx(f_fft, X.imag)
    plt.ylabel('Faze [deg]')
    plt.xlabel('Frekvence [Hz]')
    #plt.ylabel('Log amplitude [dB]')
    #plt.xlabel('Frequency [Hz]')
    plt.xlim([20, Fs/2])
    plt.show()