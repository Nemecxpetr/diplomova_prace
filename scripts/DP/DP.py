"""
Module: DP
Author: Bc. Petr Nemec

Part of my master's thesis (https://github.com/Nemecxpetr/diplomova_prace)

Aim of this script is providing help funcition for visualization of graphs etc. in the thesis
"""
import Handler as handle
from Handler.visualizer import plot_signal_in_time, plot_spectrogram
import librosa
from matplotlib import pyplot as plt

import os

Fs = 48000
fn_wav_x = os.path.join('..', '..', 'data', 'audio', 'test.wav')
x, Fs = librosa.load(fn_wav_x, sr=Fs, mono=True)
fig, ax = plot_signal_in_time(x=x, Fs=Fs)
#plt.show()

plot_spectrogram(x, Fs)


