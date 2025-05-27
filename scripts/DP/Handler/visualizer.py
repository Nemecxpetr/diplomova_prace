"""
Module: visualizer
Author: Petr Nemec
License: 

Some functions were taken from or inspired by the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP) and libfmp.
"""

from math import e, log10
from re import I
from libfmp.b import plot_chromagram
from matplotlib.figure import figaspect

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import librosa
import pandas as pd
from SYNC.DTW import compute_chroma_feature

import libfmp
from libfmp.c1.c1s2_symbolic_rep import visualize_piano_roll
from libfmp.b.b_plot import plot_matrix 
from Handler.MIDI_handler import df_to_list, load_midi, midi_to_csv

import soundfile as sf

def plot_signal_in_time(x, Fs):
    """Plots the inputed signal in time

    Args:
        x(np.ndarray): array with signal values in time
        Fs (int): sample rate        
    """

    L = np.size(x)#only left channel
    t = np.arange(L) / Fs

    fig, ax = plt.subplots( figsize=(12, 4) ,layout='constrained')
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
    #plt.ylabel('Modul [dB]')
    #plt.xlabel('Frekvence [Hz]')
    plt.ylabel('Log amplitude [dB]')
    plt.xlabel('Frequency [Hz]')
    plt.xlim([20, Fs/2])
    plt.show()

def print_spectrum_for_thesis(filename):
    """PRINT SPECTRUM
    Prints spectrum of an audio file four times with increasing length
    Arg:
        filename - name of the file in the dataset audio folder
    """
    input_audio_path = f'../../data/input/audio/{filename}.wav'
    Fs = 48000
    x, Fs = librosa.load(path=input_audio_path, sr=Fs)
    
    for i in range(4): 
        win = int(len(x)/(4*i+1))
        X = np.fft.rfft(x[0:win])
        f_fft = np.fft.rfftfreq(win, 1/Fs)
        plt.semilogx(f_fft, 20*np.log10(np.abs(X)), label=f'{round(win/len(x), 3)} part of signal', linestyle='-', linewidth=1.2)
        plt.ylabel('Log amplitude [dB]')
        plt.xlabel('Frequency [Hz]')
        plt.xlim([20, Fs/2])
        plt.ylim([0, 60])
        plt.legend()
        plt.show()
    
def plot_spectrogram(x, Fs, feature_type: str = 'stft'):
    """Visualize spektrogram
    
    Args: 
    x (np.ndarray) - signal
    Fs (int) - sampling frequency
    """
    if x.ndim > 1 and x.shape[0] > 1:
        x = np.mean(x, axis=0)

    N=4096//2
    H=N//2
    if feature_type == 'stft':
        X = librosa.stft(x, n_fft = N, hop_length = H, win_length=N, window = 'hann', center = True, pad_mode = "constant")
        title = 'Short-time Fourier transform spectrum'
    elif feature_type == 'cqt':
        X = librosa.cqt(x, sr = Fs, hop_length=H, window = 'hann', pad_mode = 'constant')
        title = 'Constant-Q power spectrum'
    Y = np.abs(X)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(Y, ref=np.max),
                                   sr=Fs, x_axis='time', y_axis='cqt_note', ax=ax)
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
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
    ax[0].set_ylabel('Real part')
    plt.ylabel('Log amplitude [dB]')
    plt.xlabel('Frequency [Hz]')


    ax[1].semilogx(f_fft, X.imag)
   # plt.ylabel('Faze [deg]')
   # plt.xlabel('Frekvence [Hz]')
    plt.ylabel('Log amplitude [dB]')
    plt.xlabel('Frequency [Hz]')
    plt.xlim([20, Fs/2])
    plt.show()
    
def visualize_piano_roll(score, xlabel='Time (seconds)', ylabel='Pitch', colors='FMP_1', velocity_alpha=False,
                         figsize=(12, 4), ax=None, dpi=72):
    """Plot a pianoroll visualization

    Inspired by: Notebook: C1/C1S2_CSV.ipynb from Meinard Mueller
    Adapted for different score structure

    Args:
        score: List of note events
        xlabel: Label for x axis (Default value = 'Time (seconds)')
        ylabel: Label for y axis (Default value = 'Pitch')
        colors: Several options: 1. string of FMP_COLORMAPS, 2. string of matplotlib colormap,
            3. list or np.ndarray of matplotlib color specifications,
            4. dict that assigns labels  to colors (Default value = 'FMP_1')
        velocity_alpha: Use the velocity value for the alpha value of the corresponding rectangle
            (Default value = False)
        figsize: Width, height in inches (Default value = (12)
        ax: The Axes instance to plot on (Default value = None)
        dpi: Dots per inch (Default value = 72)

    Returns:
        fig: The created matplotlib figure or None if ax was given.
        ax: The used axes
    """
    fig = None
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.subplot(1, 1, 1)

    # EDIT: The labels are at different position
    labels_set = sorted(set([note[5] for note in score]))
    colors = libfmp.b.color_argument_to_dict(colors, labels_set)
    
    # EDIT: Also the pitch is somewhere else:
    pitch_min = min(note[3] for note in score)
    pitch_max = max(note[3] for note in score)
    time_min = min(note[0] for note in score)
    time_max = max(note[0] + note[2] for note in score)

    # The values are not needed but we need to iterate correctly
    for start, _, duration, pitch, velocity, instr, _, _ in score:
        if velocity_alpha is False:
            velocity = None
        rect = patches.Rectangle((start, pitch - 0.5), duration, 1, linewidth=1,
                                 # EDIT: expects velocity normalized in interval of 0:1
                                 edgecolor='k', facecolor=colors[instr], alpha=velocity/128) 
        ax.add_patch(rect)

    ax.set_ylim([pitch_min - 1.5, pitch_max + 1.5])
    ax.set_xlim([min(time_min, 0), time_max + 0.5])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.set_axisbelow(True)
    ax.legend([patches.Patch(linewidth=1, edgecolor='k', facecolor=colors[key]) for key in labels_set],
              labels_set, loc='upper right', framealpha=1)

    if fig is not None:
        plt.tight_layout()

    return fig, ax

def compare_midi(df_original : pd.DataFrame or str, 
                 df_synced : pd.DataFrame or str, 
                 audio_chroma = None, 
                 audio_hop=None,
                 title_original_midi  : str = 'Original MIDI',
                 title_new_midi : str = 'New MIDI' ,
                 title_audio : str = 'Original audio chroma features',
                 verbose     : bool = False):

    """Plot two piano-rolls together with the audio interpretation chroma
    Args:
        df_original:  original midi data or string with path to them
        df_synced:    synced midi data or string with path to them
        audio_chroma (optional): audio chroma data to see what was the midi chromagram synced with
        audio_chroma_settings:
    Returns:
        fig: 
        axs: 
    """
    if audio_chroma is not None: 
        assert audio_hop is not None, "The hop size is not set"
        rows = 3 
    else: rows = 2 
    
    
    if isinstance(df_original, str):  df_original = midi_to_csv(df_original, None)
    if isinstance(df_synced, str):    df_synced = midi_to_csv(df_synced, None)
    
    fig, axs = plt.subplots(nrows=rows, ncols=1, figsize=(8,2.5*rows), sharex=True)
    
    visualize_piano_roll(df_to_list(df_original),
                         xlabel='Time (seconds)',
                         ylabel='MIDI pitch',
                         colors='FMP_1',
                         velocity_alpha=True,
                         figsize=(8,5),
                         ax=axs[0])    
    axs[0].set(title=title_original_midi)     
    axs[0].label_outer()
    
    visualize_piano_roll(df_to_list(df_synced),
                         xlabel='Time (seconds)',
                         ylabel='MIDI pitch',
                         colors='FMP_1',
                         velocity_alpha=True,
                         figsize=(8,5),
                         ax=axs[1])    
    axs[1].set(title=title_new_midi)    
    axs[1].label_outer()
    
    if audio_chroma is not None:
    
        img = librosa.display.specshow(audio_chroma, x_axis='s', y_axis='chroma', cmap='gray_r', hop_length=audio_hop, ax=axs[2])
        axs[2].set(title=title_audio)    
        axs[2].set_xlabel('Time (seconds)')
        axs[2].set_ylabel('Chroma')  
        axs[2].label_outer()

    fig.tight_layout()
    
    if verbose: plt.show()

        
    return fig, axs

def compare_chroma(chromas      : list, 
                   titles       : list, 
                   audio_hop    : float,
                   verbose      : bool = False):
    """Plot Three Chroma Features
    """
    if len(chromas) != len(titles):
        raise ValueError("The number of chromas and titles must be the same.")

    rows = len(chromas)
    fig, axs = plt.subplots(nrows=rows, ncols=1, figsize=(8,2.5*rows), sharex=True)

    for i, image in enumerate(chromas):
        img = librosa.display.specshow(image, x_axis='s', y_axis='chroma', cmap='gray_r', hop_length=audio_hop, ax=axs[i])
        axs[i].set(title=titles[i])    
        axs[i].set_xlabel('Time (seconds)')
        axs[i].set_ylabel('Chroma')  
        axs[i].label_outer()

    fig.tight_layout()
    
    if verbose: plt.show()
    
    return fig, axs

def compare_chroma_features(x, fs, H, N, verbose : bool = False):
    """ Compare different chroma features for one audio file

    """
    chroma_types = ['stft', 'cens', 'cqt_1']
    labels = ['STFT chroma features', 'CENS chroma features', 'Constant-Q chroma features']
    chromas = []
    for i, chroma in enumerate(chroma_types):
        X = compute_chroma_feature(x, fs, H, N, feature_type=chroma)
        chromas.append(X)

    rows = len(chromas)
    fig, axs = plt.subplots(nrows=rows, ncols=1, figsize=(8,2.5*rows), sharex=True)

    for i, image in enumerate(chromas):
        img = librosa.display.specshow(image, x_axis='s', y_axis='chroma', cmap='gray_r', hop_length=H, ax=axs[i])
        axs[i].set(title=labels[i])    
        axs[i].set_xlabel('Time (seconds)')
        axs[i].set_ylabel('Chroma')  
        axs[i].label_outer()

    fig.tight_layout()

    if verbose: plt.show()

    return fig, axs