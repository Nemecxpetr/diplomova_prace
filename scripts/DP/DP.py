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
import matplotlib.pyplot as plt

import librosa
import os

def pipeline(filenames, folder, debug, verbose):
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
        input_midi_path = f'../../data/input/MIDI/{folder}/{filenames[0]}.mid'
        input_audio_path = f'../../data/input/audio/{folder}/{filename}.wav'
        output_midi_path = f'../../data/output/midi/{folder}/s_{filename}.mid'
        csv = f'../../data/csv/{filename}.csv'
        output_audio_path = f'../../data/output/audio/{folder}/{filename}.wav'

        # Ensure parent directories exist
        Path(input_midi_path).parent.mkdir(parents=True, exist_ok=True)
        Path(input_audio_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_midi_path).parent.mkdir(parents=True, exist_ok=True)
        Path(csv).parent.mkdir(parents=True, exist_ok=True)
        Path(output_audio_path).parent.mkdir(parents=True, exist_ok=True)

        # 2. STEP - SYNCHRONIZE 
        # TODO: pad input midi with some zero notes at beggining (done) and at end
        handle.midi_to_csv(midi=input_midi_path, csv_path=csv, debug=debug)
        synced_midi, audio_chroma, audio_hop= create_synced_object_from_MIDIfile(input_midi_path, input_audio_path, output_midi_path, csv, verbose)

        # 3. STEP - is it working? VISUAL COMPARISON
        # export it to chroma representation
        fig, _ = handle.compare_midi(input_midi_path, output_midi_path, audio_chroma, audio_hop=audio_hop)
        if fig is not None:
            save_path = f'../../data/output/figures/{folder}/{filename}_comparison.png'
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
            plt.close(fig)

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

def dataset_preset(preset, conv=False):
    """Choose filenames and folder presets for the dataset described in [1]
    Args:   
        preset (str): The name of the dataset preset to use.
            # Dataset Presets:
            #    #    preset = 'gymnopedie'
            #    #    preset = 'unravel'
            #    #    preset = 'albeniz'
            #    #    preset = 'summertime'
            #    #    preset = 'messiaen'
        conv (bool): Whether to convert audio files to wav format.
    Returns:
        tuple: A tuple containing the folder name and a list of filenames.
    """
    if preset == 'gymnopedie':         
            #folder = 'gymnopedie'
            #filenames = ['gymnopedie no1_1', 'gymnopedie no1_2', 'gymnopedie no1 khatia']
            # Convert audio files to wav format
            #for filename in filenames:
            #    handle.audio_handler.convert_to_wav(f'../../data/input/audio/{folder}/{filename}.m4a', f'../../data/input/audio/{folder}/{filename}.wav', format='m4a')

            folder = 'gymnopedie'
            filenames = ['gymnopedie no1', 'gymnopedie no1_1', 'gymnopedie no1_2', 'gymnopedie no1 khatia', 'gymnopedie no1_3', 'gymnopedie no1_4']       
    elif preset == 'unravel':
            folder = 'unravel'
            filenames = ['unravel']
            conv_format = None
    elif preset ==  'albeniz':
            folder = 'albeniz'
            filenames = ['alb_se5', 'alb_se5_1', 'alb_se5_2', 'alb_se5_3']
            conv_format = 'mp3'
            # for filename in filenames:
            #    handle.audio_handler.convert_to_wav(f'../../data/input/audio/{folder}/{filename}.mp3', f'../../data/input/audio/{folder}/{filename}.wav', format='mp3')
    elif preset == 'summertime':
        folder = 'summertime'
        filenames = ['summertime','summertime_1', 'summertime_2']
        conv_format = 'mp3'
        #for filename in filenames:
        #    handle.audio_handler.convert_to_wav(f'../../data/input/audio/{folder}/{filename}.mp3', f'../../data/input/audio/{folder}/{filename}.wav', format='mp3')
    elif preset == 'messiaen':
        folder = 'messiaen'
        filenames = ['messiaen_le_banquet_celeste', 'messiaen_le_banquet_celeste_1', 'messiaen_le_banquet_celeste_2']
        conv_format = 'm4a' 
    else:
        print('Unknown preset')
        return None, None

    if (conv_format is not None) & conv:
        for filename in filenames:
            handle.audio_handler.convert_to_wav(f'../../data/input/audio/{folder}/{filename}.{conv_format}', f'../../data/input/audio/{folder}/{filename}.wav', format=conv_format)
  
    return folder, filenames

# What we try to achieve is a function synchronize_MIDI_with_audio()
if __name__ == "__main__":

    
    # 1. choose the testfile names 
    #    # Naming convention in the dataset is that the first audio file has the same name 
    #    as the midifile to be synced with.
    #    # The others are numbered
    # Dataset Presets:
    #    #    preset = 'gymnopedie'
    #    #    preset = 'unravel'
    #    #    preset = 'albeniz'
    #    #    preset = 'summertime'
    #    #    preset = 'messiaen'
    folder, filenames = dataset_preset('messiaen', True)

  
    debug = False # debuging argument passed to other functions 
    verbose = False # argument passed to other functions for activating graph visualization of sync process
    
    # Pipeline in a function for better 
    pipeline(filenames,folder, debug, verbose)
        
    #dtw_test(filenames[0], True)
    #midi_test(debug=True)

    #filename = 'tests/test'
    #handle.visualizer.print_spectrum_for_thesis(filename)



    
