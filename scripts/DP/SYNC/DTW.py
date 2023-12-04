"""
Module: DTW
Author: Bc. Petr Němec

Part of my master's thesis (https://github.com/Nemecxpetr/diplomova_prace)

Cílem semestrálního projektu je popis principů hudební synchronizace, sestavení datasetu a základní implementace score-to-audio synchronizace

Aim of this script is basic implementation of score-to-audio synchronization

"""
import os
import sys
from turtle import shape
import Handler as handle
import pretty_midi as pm
import librosa
from matplotlib import pyplot as plt
import numpy as np
import libfmp.c3
import libfmp.b.b_plot as fmpplot
import soundfile as sf

from Handler import visualizer



# libfmp or librosa has implemented some basic DTW - Multi-scale or memory restricted I will have to implement myself

# TODO: implement cost matrix, warping path etc. or use library (libfmp or librosa)
# NOTE: for now using implementations from libraries
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
    midi_data = handle.load_midi(path_midi)
    Y = pm.PrettyMIDI.get_chroma(midi_data)

    show = False
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

    #print(wp[0:10])
    print(len(wp))
    #NOTE: wp is ((x1, y1), (x2, y2), ..., (xn, ym))

    #TODO: now we need to adjust the times in the original midi
    # this could be done by turning the original midi to csv adjusting the note start positions and then turning the adjusted csv back to midi
    # the "turning" functions are already created in MIDI_handler.py in Handler package

    score = handle.midi_to_list(midi_data)
    print (len(score))
    #TODO: problem!! warping path of the midi data is for the midi data of the chromagram which we need to adjust and then transform back into midi.
    # Question? How to do that?

    for n in range(len(score[0])):
         print(score[0][n])

    # question how to adjust if more tones that originally start at the same time are now all different? How does the warping path reflect that?

#handle.test()

#handle.test_tempo()

#test()


path = os.path.join('..', '..', 'data', 'audio', 'test.wav')
x, Fs = sf.read(path)
# lets use only left channel
x = x.T[0]

#visualizer.plot_spectrograph(x, Fs)
#visualizer.plot_spectrograph_phase(x, Fs)
N=4096//2
H=N//2
X = librosa.stft(x, n_fft = N, hop_length = H, win_length=N, window = 'hann', center = True, pad_mode = "constant")
gamma = 100
Y = 20*np.log10(1+gamma*abs(X))
fig = fmpplot.plot_matrix(Y, Fs)
plt.yscale('log')
plt.ylim([20, Fs/2])
plt.show()