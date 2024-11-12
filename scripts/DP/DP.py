"""
Module: DP
Author: Bc. Petr Nemec

This script is realizing the testing and accesing of the pipeline functions of my master's thesis [1]
(https://github.com/Nemecxpetr/diplomova_prace).

Task: 


Reference:

[1] NĚMEC, Petr. Score-to-audio synchronization of music interpretations [online]. 
    Brno, 2024 [cit. 2024-03-01]. Available from: https://www.vut.cz/studenti/zav-prace/detail/159304. 
    Master's Thesis. Vysoké učení technické v Brně, Fakulta elektrotechniky a komunikačních technologií, 
    Department of Telecommunications. Supervisor Matěj Ištvánek.
    
[2] MUELLER, Meinard and ZALKOW, Frank: FMP Notebooks: Educational Material for Teaching and Learning Fundamentals of Music Processing.
    Proceedings of the International Conference on Music Information Retrieval (ISMIR), Delft, The Netherlands, 2019.
"""
import Handler as handle
from SYNC.DTW import create_synced_object_from_MIDIfile
from pathlib import Path

import os

# What we try to achieve is a function synchronize_MIDI_with_audio()
if __name__ == "__main__":

    # 1. choose the testfile names 
    #    # Naming convention in the dataset is that the first audio file has the same name as the midifile to be synced with.
    #    # The others are numbered
    filenames = ['gymnopedie no1',
                 'gymnopedie no1_3', 
                 'gymnopedie no1_4']
    # NOTE: first we should test it on some short files to see if it actually works
    
  
    debug = False # debuging argument passed to other functions 
    verbose = True # argument passed to other functions for activating graph visualization of sync process

    for filename in filenames:
        # 1. STEP - choose destinations for input and output MIDI and AUDIO data        
        input_midi_path = f'../../data/input/MIDI/gymnopedie/{filenames[0]}.mid'
        input_audio_path = f'../../data/input/audio/gymnopedie/{filename}.wav' #TODO: adapt to different audio formats?
                                                                   #TODO: also make sure that the names are really the same
        output_midi_path = f'../../data/output/s_{filename}.mid'
        csv = f'../../data/csv/{filename}.csv'
        
        # 2. STEP - SYNCHRONIZE 
        # TODO: padd input midi with some zero notes at beggining and at end
        handle.midi_to_csv(midi=input_midi_path, csv_path=csv, debug=debug)
        synced_midi, audio_chroma, audio_hop= create_synced_object_from_MIDIfile(input_midi_path, input_audio_path, output_midi_path, csv, verbose)

        # 3. STEP - is it working? VISUAL COMPARISON
        # export it to chroma representation
        Fs = 48000
        handle.compare_midi( input_midi_path, output_midi_path, audio_chroma, audio_hop=audio_hop)
        