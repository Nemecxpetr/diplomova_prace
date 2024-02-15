"""
Module: DP
Author: Bc. Petr Nemec

Part of my master's thesis (https://github.com/Nemecxpetr/diplomova_prace)

Aim of this script is providing help funcition for visualization of graphs etc. in the thesis

Task:

"""
from time import sleep
import Handler as handle
from Handler.visualizer import plot_signal_in_time, plot_spectrogram
import librosa
from libfmp.c1.c1s2_symbolic_rep import visualize_piano_roll
from matplotlib import pyplot as plt
from SYNC.DTW import dtw_test, create_synced_object_from_MIDIfile

import os
import sf2_loader as sf

# What we try to achieve is a function synchronize_MIDI_with_audio()
if __name__ == "__main__":

    filenames = ['dtw_test', 'test']
    debug = False # debuging argument passed to other functions 
    verbose = False # argument passed to other functions for activating graph visualization of sync process

    for filename in filenames:
        # 1. STEP - choose destinations for input and output MIDI and AUDIO data
        
        input_midi_path = f'../../data/input/MIDI/tests/{filename}.mid'
        input_audio_path = f'../../data/iput/audio/{filename}.wav' #TODO: adapt to different audio formats?
                                                                   #TODO: also make sure that the names are really the same
        output_midi_path = f'../../data/output/{filename}.mid'
        csv = f'../../data/csv/{filename}.csv'
        
        # 2. STEP - SYNCHRONIZE 
        # TODO: padd input midi with some zero notes at beggining
        handle.midi_to_csv(midi=input_midi_path, csv_path=csv, debug=debug)
        synced_midi, audio_chroma = create_synced_object_from_MIDIfile(output_midi_path, input_audio_path, output_midi_path, csv, verbose)

        # 3. STEP - is it working? VISUAL COMPARISON
        # export it to chroma representation
        handle.compare_midi(input_midi_path, output_midi_path, audio_chroma)
        

# EXPERIMENTING with the musicpy and sound font loader
# possible usage for later synthesis of sounds (comparing audio-to-audio with score-to-audio synth aproaches)

# filename = "tests/test"
# soundfont = "EX115"

# # examples
# loader = sf.sf2_loader(f'../../data/sf2/{soundfont}.sf2')

# loader.play_midi_file(f'../../data/input/MIDI/{filename}.mid')

# sleep(5)