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



# libfmp or librosa has implemented some basic DTW - Multi-scale or memory restricted I will have to implement myself

# TODO: implement cost matrix, warping path etc. 
def warping_path(X, Y):
    D, wp = librosa.sequence.dtw(X, Y, subseq=True)


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

    return 0


def test():
    #test_audio, Fs = handle.read_audio(os.path.join('..', '..', 'data', 'audio', 'test.wav'))
    

    #handle.plot_signal_in_time(test_audio, Fs)

    Fs = 22050
    N = 4410
    H = N//4
    fn_wav_X = os.path.join('..', '..', 'data', 'audio', 'nev_gon_give_you_up.webm')

    X_wav, Fs = librosa.load(fn_wav_X, sr=Fs)
    X = librosa.feature.chroma_stft(y=X_wav, sr=Fs, hop_length=H, n_fft=N)

    path_midi =os.path.join('..', '..', 'data', 'MIDI', 'nev_gon_giv_you_up.mid')
    Y = pm.PrettyMIDI.get_chroma(handle.load_midi(path_midi))


    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=False)
    img = librosa.display.specshow(X, x_axis='frames', y_axis='chroma', cmap='gray_r', hop_length=H, ax=ax[0])
    ax[0].set(title='Sequence $X$')    
    ax[0].set_xlabel('Time (frames)')
    ax[0].set_ylabel('Chroma')  
    ax[0].label_outer()

    librosa.display.specshow(Y, x_axis='frames', y_axis='chroma', cmap='gray_r', hop_length=H, ax=ax[1])
    ax[1].set(title='Sequence $Y$')
    ax[1].set_xlabel('Time (frames)')
    ax[1].set_ylabel('Chroma')
    ax[0].label_outer()
    
    fig.colorbar(img, ax=ax)
    plt.show()

    warping_path(X, Y)








test()

