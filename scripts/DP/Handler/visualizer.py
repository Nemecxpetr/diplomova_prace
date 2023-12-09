"""
Module: visualizer
Author: Petr Nemec
License: 

Some functions were taken from or inspired by the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP) and libfmp.
"""

from math import log10
from matplotlib.figure import figaspect

import numpy as np
import matplotlib.pyplot as plt
import librosa

from libfmp.c1.c1s2_symbolic_rep import visualize_piano_roll
from libfmp.b.b_plot import plot_matrix 
from Handler.MIDI_handler import csv_to_list

import soundfile as sf

def plot_signal_in_time(x, Fs):
    """Plots the inputed signal in time

    Args:
        x(np.ndarray): array with signal values in time
        Fs (int): sample rate        
    """

    L = np.size(x)#only left channel
    t = np.arange(L) / Fs

    fig, ax = plt.subplots( figsize=(10, 6) ,layout='constrained')
    ax.plot(t, x)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    
    return fig, ax

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
    
def plot_spectrogram(x, Fs):
    """Visualize spektrogram
    
    Args: 
    x (np.ndarray) - signal
    Fs (int) - sampling frequency
    """

    N=4096//2
    H=N//2
    X = librosa.stft(x, n_fft = N, hop_length = H, win_length=N, window = 'hann', center = True, pad_mode = "constant")
    gamma = 100
    Y = 20*np.log10(1+gamma*abs(X))
    fig = plot_matrix(Y, Fs)
    plt.yscale('log')
    plt.ylim([20, Fs/2])
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
    

def compare_midi(df_original, df_synced, audio_chroma):
    """Plot two piano-rolls together with the audio interpretation chroma
    Args:
        df_original
        df_synced
        audio_chroma
    
    """
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8,5), sharex=False)
    
    visualize_piano_roll(csv_to_list(df_original),
                         xlabel='Time (seconds)',
                         ylabel='Chroma pitch',
                         colors='FMP_1',
                         velocity_alpha=True,
                         figsize=(8,5),
                         ax=axs[0])    
    axs[0].set(title='Original MIDI')     
    axs[0].label_outer()
    
    visualize_piano_roll(csv_to_list(df_synced),
                         xlabel='Time (seconds)',
                         ylabel='Chroma pitch',
                         colors='FMP_1',
                         velocity_alpha=True,
                         figsize=(8,5),
                         ax=axs[1])    
    axs[1].set(title='Synchronized MIDI')    
    axs[1].label_outer()

    fig.tight_layout()
    
    plt.show()
        
    return fig, axs