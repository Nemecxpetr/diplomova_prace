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
from Handler.MIDI_handler import midi_test
from SYNC.DTW import create_synced_object_from_MIDIfile, dtw_test
from pathlib import Path
from libfmp.b.b_sonification import sonify_chromagram_with_signal, list_to_chromagram
from synctoolbox.feature.csv_tools import df_to_pitch_features
from synctoolbox.feature.chroma import pitch_to_chroma
from scipy.io.wavfile import write
import numpy as np
from IPython.display import Audio

import librosa
import os

def pipeline(filenames, debug, verbose):
    """ MAIN PIPELINE
    This is the function where the main pipeline is created as described in the thesis [1]
        1. Load input files as sequences 𝑥, 𝑦 of some sort;
        2. Convert the sequences 𝑋, 𝑌 to a common mid-representation format – chroma
            features;
        3. Compute the optimal warping path;
        4. Extrapolate the obtained path and gain exact beginning and end times/tics
            of musical elements in the original symbolic representation;
        5. Change the times of the musical instances in a copy of the original symbolic
            data and export the sequence as a new synchronized MIDI file.
    """
    for filename in filenames:
        # 1. STEP - choose destinations for input and output MIDI and AUDIO data        
        input_midi_path = f'../../data/input/MIDI/{filenames[0]}.mid'
        input_audio_path = f'../../data/input/audio/{filename}.wav' #TODO: adapt to different audio formats?
                                                                   #TODO: also make sure that the names are really the same
        output_midi_path = f'../../data/output/s_{filename}.mid'
        csv = f'../../data/csv/{filename}.csv'
        output_audio_path = f'../../output/audio/{filename}.wav'
        
        # 2. STEP - SYNCHRONIZE 
        # TODO: pad input midi with some zero notes at beggining (done) and at end
        handle.midi_to_csv(midi=input_midi_path, csv_path=csv, debug=debug)
        synced_midi, audio_chroma, audio_hop= create_synced_object_from_MIDIfile(input_midi_path, input_audio_path, output_midi_path, csv, verbose)

        # 3. STEP - is it working? VISUAL COMPARISON
        # export it to chroma representation
        handle.compare_midi(input_midi_path, output_midi_path, audio_chroma, audio_hop=audio_hop)
        

        # Sonification evaluation     
        Fs = 48000
        x, Fs = librosa.load(path=input_audio_path, sr=Fs)
        feature_rate = Fs / audio_hop   
        f_pitch = df_to_pitch_features(synced_midi, feature_rate=feature_rate)
        f_chroma = pitch_to_chroma(f_pitch=f_pitch)
        #f_chroma_quantized = quantize_chroma(f_chroma=f_chroma)
        chroma_midi = f_chroma

        x_chroma_ann, x_chroma_ann_stereo = sonify_chromagram_with_signal(chroma_midi, x, feature_rate, Fs)
        # Write out audio as 24bit PCM WAV
        audio = Audio(x_chroma_ann_stereo, rate=Fs, autoplay=False)

        with open(output_audio_path, "wb") as f:
           f.write(audio.data)


# What we try to achieve is a function synchronize_MIDI_with_audio()
if __name__ == "__main__":

    # 1. choose the testfile names 
    #    # Naming convention in the dataset is that the first audio file has the same name 
    #    as the midifile to be synced with.
    #    # The others are numbered
    filenames = ['unravel']
    # NOTE: first we should test it on some short files to see if it actually works
    
  
    debug = False # debuging argument passed to other functions 
    verbose = True # argument passed to other functions for activating graph visualization of sync process
    
    pipeline(filenames, debug, verbose)
        
    #dtw_test(filenames[0], True)
    #midi_test(debug=True)

    #filename = 'tests/test'
    #handle.visualizer.print_spectrum_for_thesis(filename)



    
