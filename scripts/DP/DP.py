"""
Module: DP
Author: Bc. Petr Nemec

Part of my master's thesis (https://github.com/Nemecxpetr/diplomova_prace)

Aim of this script is providing help funcition for visualization of graphs etc. in the thesis

Task:


Reference:

[1] NÌMEC, Petr. Score-to-audio synchronization of music interpretations [online]. 
    Brno, 2024 [cit. 2024-03-01]. Available from: https://www.vut.cz/studenti/zav-prace/detail/159304. 
    Master's Thesis. Vysoké uèení technické v Brnì, Fakulta elektrotechniky a komunikaèních technologií, 
    Department of Telecommunications. Supervisor Matìj Ištvánek.
    
[2] MUELLER, Meinard and ZALKOW, Frank: FMP Notebooks: Educational Material for Teaching and Learning Fundamentals of Music Processing.
    Proceedings of the International Conference on Music Information Retrieval (ISMIR), Delft, The Netherlands, 2019.
"""
import Handler as handle
from SYNC.DTW import create_synced_object_from_MIDIfile
from pathlib import Path

import os

# What we try to achieve is a function synchronize_MIDI_with_audio()
if __name__ == "__main__":

    filenames = ['dtw_test', 'dtw_test_whistle', 'dtw_test_voice_eq', 'dtw_test_voice_slow']
    debug = False # debuging argument passed to other functions 
    verbose = False # argument passed to other functions for activating graph visualization of sync process

    for filename in filenames:
        # 1. STEP - choose destinations for input and output MIDI and AUDIO data        
        input_midi_path = f'../../data/input/MIDI/tests/dtw_test.mid'
        input_audio_path = f'../../data/input/audio/{filename}.wav' #TODO: adapt to different audio formats?
                                                                   #TODO: also make sure that the names are really the same
        output_midi_path = f'../../data/output/{filename}.mid'
        csv = f'../../data/csv/{filename}.csv'
        
        # 2. STEP - SYNCHRONIZE 
        # TODO: padd input midi with some zero notes at beggining
        handle.midi_to_csv(midi=input_midi_path, csv_path=csv, debug=debug)
        synced_midi, audio_chroma, audio_hop= create_synced_object_from_MIDIfile(input_midi_path, input_audio_path, output_midi_path, csv, verbose)

        # 3. STEP - is it working? VISUAL COMPARISON
        # export it to chroma representation
        Fs = 48000
        handle.compare_midi( input_midi_path, output_midi_path, audio_chroma, audio_hop=audio_hop)
        

# EXPERIMENTING with the musicpy and sound font loader
# possible usage for later synthesis of sounds (comparing audio-to-audio with score-to-audio synth aproaches)

# filename = "tests/test"
# soundfont = "EX115"

path_csv_100 = os.path.join('..', '..', 'data','CSV', 'test_100bpm.csv')
path_csv_80 = os.path.join('..', '..', 'data','CSV', 'test_80bpm.csv')

# loader.play_midi_file(f'../../data/input/MIDI/{filename}.mid')

# sleep(5)