""" 
Handler package

Author: Petr Němec
License: 

Containing functions for easier MIDI and audio files manipulation. Realizing conversion between midi, csv and audio (yet TODO). 
Creating chroma vectors and other visual representations. (yet TODO)

Also offers some classes to create note objects for playing - I'm just playin' and learning to programe at this point.

Some functions were taken from or inspired by the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)

This package is part of my master's thesis (https://github.com/Nemecxpetr/diplomova_prace)
"""
# MIDI handler functions to create midi and convert it to csv and back
from numba.core.ir_utils import convert_code_obj_to_function
from .MIDI_handler import load_midi,\
    midi_to_list, \
    list_to_csv, \
    read_csv ,\
    create_midi_from_csv_experimental, \
    midi_to_csv,\
    df_to_list,\
    midi_test

from .audio_handler import read_audio, \
    write_audio, \
    convert_to_wav

from .visualizer import plot_signal_in_time, \
    plot_spectrograph, \
    plot_spectrograph_phase, \
    compare_midi, \
    plot_spectrogram, \
    print_spectrum_for_thesis
