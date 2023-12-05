"""
Module: DTW
Author: Bc. Petr Němec

Part of my master's thesis (https://github.com/Nemecxpetr/diplomova_prace)

Cílem semestrálního projektu je popis principů hudební synchronizace, sestavení datasetu a základní implementace score-to-audio synchronizace

Aim of this script is basic implementation of score-to-audio synchronization

"""
import os
import Handler as handle
import pretty_midi as pm
import librosa
from matplotlib import pyplot as plt
import numpy as np
import statistics
import pandas as pd

import libfmp.c3
from libfmp.c3.c3s2_dtw_plot import plot_matrix_with_points

import soundfile as sf

from synctoolbox.dtw.core import compute_warping_path
from synctoolbox.dtw.cost import cosine_distance



# libfmp or librosa has implemented some basic DTW - Multi-scale or memory restricted I will have to implement myself

# TODO: implement cost matrix, warping path etc. or use library (libfmp or librosa)
# NOTE: for now using implementations from libraries
def warping_path(X, Y, mueller=False, show=False):
    """ Computes warping path between two chromavectors
    Args:
        X, Y (np.ndarray): chroma vectors ( of shape(12, num_of_time_indices))   
    Returns: 
        # TODO: what data is wp?
        wp (np.ndarray [shape=(N, 2)]): Warping path with index pairs.
    """
    # choose to use librosa or mueller implementation
    mueller = mueller

    # with librosa dtw implementation
    C = cosine_distance(X, Y) # from synctoolbox
    _, wp = librosa.sequence.dtw(C=C)

    if show and not mueller:
        fig, ax = plt.subplots(nrows=2, sharex=False)
        img = librosa.display.specshow(D, x_axis='frames', y_axis='frames',
                                       ax=ax[0])
        ax[0].set(title='DTW cost', xlabel='Noisy sequence', ylabel='Target')
        ax[0].plot(wp[:, 1], wp[:, 0], label='Optimal path', color='y')
        ax[0].legend()
        fig.colorbar(img, ax=ax[0])
        ax[1].plot(D[-1, :] / wp.shape[0])
        ax[1].set(xlim=[0, Y.shape[1]], #ylim=[0, 2],
                  title='Matching cost function')

        plt.show()
    
    # with FMP (Mueller) implementation
    C = libfmp.c3.compute_cost_matrix(X, Y)
    D = libfmp.c3.compute_accumulated_cost_matrix(C)
    P = libfmp.c3.compute_optimal_warping_path(D)

    P = np.array(P)

    # Synctoolbox
    _, _, wp_full = compute_warping_path(C=C)

    plot_matrix_with_points(C, wp_full.T, linestyle='-',  marker='', aspect='equal',
                            title='Cost matrix and warping path computed using full DTW',

                            xlabel='MIDI - CSV (frames)', ylabel='Audio (frames)', figsize=(9, 5))
    plt.show()

    if show:
        plt.figure(figsize=(9, 3))
        ax = plt.subplot(1, 2, 1)
        plot_matrix_with_points(C, P, linestyle='-', 
            ax=[ax], aspect='equal', clim=[0, np.max(C)],
            title='$C$ with optimal warping path', xlabel='Sequence Y', ylabel='Sequence X');

        ax = plt.subplot(1, 2, 2)
        plot_matrix_with_points(D, P, linestyle='-', 
            ax=[ax], aspect='equal', clim=[0, np.max(D)],
            title='$D$ with optimal warping path', xlabel='Sequence Y', ylabel='Sequence X');

        plt.show()

    if mueller:
        wp = P

    return wp


def dtw_test(show=True):
    #test_audio, Fs = handle.read_audio(os.path.join('..', '..', 'data', 'audio', 'test.wav'))
    #handle.plot_signal_in_time(test_audio, Fs)

    # I don't know why but I decided to downsample it
    Fs = 22050
    N = 2048
    H = N//2
    fn_wav_x = os.path.join('..', '..', 'data', 'audio', 'DTW_test.wav')

    x_wav, Fs = librosa.load(fn_wav_x, sr=Fs)
    X = librosa.feature.chroma_stft(y=x_wav, sr=Fs, hop_length=H, n_fft=N)

    path_midi =os.path.join('..', '..', 'data', 'MIDI', 'DTW_test.mid')
    midi_data = handle.load_midi(path_midi)
    Y = pm.PrettyMIDI.get_chroma(midi_data)

    if show:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=False)
        img = librosa.display.specshow(X, x_axis='frames', y_axis='chroma', cmap='gray_r', hop_length=H, ax=ax[0])
        ax[0].set(title='Sequence $X$')    
        ax[0].set_xlabel('Time (frames)')
        ax[0].set_ylabel('Chroma')  
        ax[0].label_outer()

        librosa.display.specshow(Y, x_axis='frames', y_axis='chroma', cmap='gray_r', ax=ax[1])
        ax[1].set(title='Sequence $Y$')
        ax[1].set_xlabel('Time (frames)')
        ax[1].set_ylabel('Chroma')
        ax[0].label_outer()
    
        fig.colorbar(img, ax=ax)
        plt.show()

    #NOTE: X = audio chroma, Y = score chroma
    wp = warping_path(X, Y, show=show)   

    #TODO: Move to a separate function
    """
    Creates synchronized midi and csv object at specified locations
    
    Args: 
        original_midi_data
        wp
        path_midi
        path_csv
    """
    score = handle.midi_to_list(midi_data)
    synced_score = score

    for note in score:
         # 
         start = note[0]
         duration = note[1]
         pitch = note[2]
         velocity = note[-2]
         label = note[-1]
         
         
         # fínd start in wp on position of midi (y) and change it for the position of interpolated wp audio (x) 
         start_frame = librosa.time_to_frames(start, sr=Fs, n_fft=N, hop_length=H) # compute frame index of original note start, equivallently: #start*Fs/H
         end_frame = librosa.time_to_frames(start+duration, sr=Fs, n_fft=N, hop_length=H)
         
         for point in wp:
             new_start_frame = statistics.median(wp[:,start_frame]) # TODO: instead of median interpolated wp before
             new_end_frame = statistics.median(wp[:,end_frame])
         
         new_start = librosa.frames_to_time(new_start_frame, sr=Fs, hop_length=H, n_fft = N)
         new_duration = librosa.frames_to_time(new_end_frame-new_start_frame, sr=Fs, hop_length=H, n_fft=N)

         synced_score.append([new_start, new_duration, pitch, velocity, label])
        
    #NOTE: problem!! warping path of the midi data is for the midi data of the chromagram which we need to adjust and then transform back into midi.
    # Question? How to do that?
    #  
    # I aksed Prof. Mueller and he said one way to synchronize the two objects is to interpolate the warping path
    # the best thing to do would be to interpolate the warping path so that it is "continuous" 
    # (more on the struggle of interpolation later -> TODO)
    # then find new values for midi notes (original time values in the csv_from_midi object)
    #
    # If I don't want to interpolate other idea could be to use different warping path conditions

    path_csv = os.path.join('..', '..', 'data','CSV', 'dtw_test_synced.csv')
    synced_csv = handle.list_to_csv(synced_score, path_csv)
    
    path_midi = os.path.join('..', '..', 'data', 'MIDI', 'from_csv', 'dtw_test_synced.mid')
    synced_midi = handle.csv_to_midi(synced_csv, path_midi)

    # TODO: create compare midi function that will plot piano roll of original and new midi
    #handle.compare_midi(synced_midi, midi_data)

    # question how to adjust if more tones that originally start at the same time are now all different? How does the warping path reflect that?
    # Now that I think musically about it how would I even align that as a human listener? so I guess it doesn't really matter. If it picks the best match (The WP with least cost)
    # It should probably work


dtw_test(show=False)


"""
Pomocné funkce pro tisk do diplomové práce
"""
#TODO: přesunout do jiného souboru (nejlépe DP.py)

def ukazka_spektrogramu():
    path = os.path.join('..', '..', 'data', 'audio', 'test.wav')
    x, Fs = sf.read(path)
    # lets use only left channel
    x = x.T[0]

    N=4096//2
    H=N//2
    X = librosa.stft(x, n_fft = N, hop_length = H, win_length=N, window = 'hann', center = True, pad_mode = "constant")
    gamma = 100
    Y = 20*np.log10(1+gamma*abs(X))
    fig = fmpplot.plot_matrix(Y, Fs)
    plt.yscale('log')
    plt.ylim([20, Fs/2])
    plt.show()