"""
Module: DTW
Author: Petr Němec

Part of my master's thesis (https://github.com/Nemecxpetr/diplomova_prace)

Cílem semestrálního projektu je popis principů hudební synchronizace, sestavení datasetu a základní implementace score-to-audio synchronizace

Aim of this script is basic implementation of score-to-audio synchronization

"""
import os
import sys
import Handler as handle
import pretty_midi as pm
import librosa
from matplotlib import pyplot as plt
import numpy as np
import libfmp.c3



# libfmp or librosa has implemented some basic DTW - Multi-scale or memory restricted I will have to implement myself

# TODO: implement cost matrix, warping path etc. 
def warping_path(X, Y, show=False):
    """ Computes warping path between two chromavectors
    Args:
        X, Y (np.ndarray): chroma vectors ( of shape(12, num_of_time_indices))   
    Returns: 
        # TODO: what data is wp?
        wp (np.ndarray [shape=(N, 2)]): Warping path with index pairs.
    """
    # choose to use librosa or mueller implementation
    mueller = True

    # with librosa dtw implementation
    # TODO: there's something wrong with this
    D, wp = librosa.sequence.dtw(X, Y, subseq=True)

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
    
    if show:
        plt.figure(figsize=(9, 3))
        ax = plt.subplot(1, 2, 1)
        libfmp.c3.plot_matrix_with_points(C, P, linestyle='-', 
            ax=[ax], aspect='equal', clim=[0, np.max(C)],
            title='$C$ with optimal warping path', xlabel='Sequence Y', ylabel='Sequence X');

        ax = plt.subplot(1, 2, 2)
        libfmp.c3.plot_matrix_with_points(D, P, linestyle='-', 
            ax=[ax], aspect='equal', clim=[0, np.max(D)],
            title='$D$ with optimal warping path', xlabel='Sequence Y', ylabel='Sequence X');

        plt.show()

    if mueller:
        wp = P

    return wp


def test():
    #test_audio, Fs = handle.read_audio(os.path.join('..', '..', 'data', 'audio', 'test.wav'))
    #handle.plot_signal_in_time(test_audio, Fs)

    Fs = 22050
    N = 4410//2
    H = N//4
    fn_wav_X = os.path.join('..', '..', 'data', 'audio', 'test.wav')

    X_wav, Fs = librosa.load(fn_wav_X, sr=Fs)
    X = librosa.feature.chroma_stft(y=X_wav, sr=Fs, hop_length=H, n_fft=N)

    path_midi =os.path.join('..', '..', 'data', 'MIDI', 'test.mid')
    Y = pm.PrettyMIDI.get_chroma(handle.load_midi(path_midi))

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

    wp = warping_path(X, Y, show=True)

    print(wp)


handle.test()

#handle.test_tempo()

#test()

