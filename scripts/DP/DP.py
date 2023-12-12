"""
Module: DP
Author: Bc. Petr Nemec

Part of my master's thesis (https://github.com/Nemecxpetr/diplomova_prace)

Aim of this script is providing help funcition for visualization of graphs etc. in the thesis
"""
import Handler as handle
from Handler.visualizer import plot_signal_in_time, plot_spectrogram
import librosa
from libfmp.c1.c1s2_symbolic_rep import visualize_piano_roll
from matplotlib import pyplot as plt
from SYNC.DTW import dtw_test

import os

# Fs = 48000
# fn_wav_x = os.path.join('..', '..', 'data', 'audio', 'test.wav')
# x, Fs = librosa.load(fn_wav_x, sr=Fs, mono=True)
#fig, ax = plot_signal_in_time(x=x, Fs=Fs)
#plt.show()

#plot_spectrogram(x, Fs)

# plot piano-roll 
# fn_midi = os.path.join('..', '..', 'data', 'MIDI', 'test_100bpm.mid')
# midi = handle.load_midi(fn_midi)
# fig, ax = visualize_piano_roll(handle.midi_to_list(midi), velocity_alpha=True)
# ax.set_title('Test 100 bmp')
# plt.tight_layout()
# plt.show()

#handle.MIDI_handler.test()

#dtw_test(show=False)


path_csv_100 = os.path.join('..', '..', 'data','CSV', 'test_100bpm.csv')
path_csv_80 = os.path.join('..', '..', 'data','CSV', 'test_80bpm.csv')

df_80bpm = handle.MIDI_handler.read_csv(path_csv_80, separator=',')
df_100bpm = handle.MIDI_handler.read_csv(path_csv_100, separator=',')

handle.visualizer.compare_midi(df_80bpm, df_100bpm, title_original_midi='MIDI 80 BPM', title_new_midi='MIDI 100 BPM')