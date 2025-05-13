"""
Module: DTW
Author: Bc. Petr Němec

Part of my master's thesis (https://github.com/Nemecxpetr/diplomova_prace)

Cílem semestrálního projektu je popis principů hudební synchronizace, sestavení datasetu a základní implementace score-to-audio synchronizace

Aim of this script is basic implementation of score-to-audio synchronization

[1] NĚMEC, Petr. Score-to-audio synchronization of music interpretations [online]. 
    Brno, 2024 [cit. 2024-03-01]. Available from: https://www.vut.cz/studenti/zav-prace/detail/159304. 
    Master's Thesis. Vysoké učení technické v Brně, Fakulta elektrotechniky a komunikačních technologií, 
    Department of Telecommunications. Supervisor Matěj Ištvánek.
    
[2] MUELLER, Meinard and ZALKOW, Frank: FMP Notebooks: Educational Material for Teaching and Learning Fundamentals of Music Processing.
    Proceedings of the International Conference on Music Information Retrieval (ISMIR), Delft, The Netherlands, 2019.
"""
import os
from pathlib import Path
import string

import Handler as handle
import librosa
from matplotlib import pyplot as plt
import scipy
import numpy as np

from synctoolbox.dtw.utils import make_path_strictly_monotonic
from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
from synctoolbox.feature.csv_tools import df_to_pitch_features
from synctoolbox.feature.chroma import pitch_to_chroma


def create_synced_object_from_MIDIfile(path_midi : string or Path, 
                 path_audio : string or Path,
                 path_output : string or Path, 
                 path_csv : string or Path,
                 verbose : bool = False):
    """ Creates a synchronized MIDI file at specified location:
    
    Given path to: 
                   1) audio interpratation recording file 
                   2) MIDI score file
    creates a NEW SYNCHRONIZED MIDI file, at given output location.
    
    The synchronization is score-to-audio as described in the master thesis [1].
   
    Also a CSV file that acts as mid-level representation is stored at given location. 
    
    Args:
        path_MIDI_data (string or Path) : path to the tempo stable MIDI file
        path_audio (string or Path) : path to the audio file with the interpretation according to which the MIDI file should be aligned
        path_output (string or Path) : path to the output file where the synced MIDI will be saved
        path_CSV (string or Path) : path to the csv file # TODO delete and instead  of path make the function to handle only filenames
    Returns: 
        synced_midi_df (pd.DataFrame) : data frame of synchronized MIDI data
        audio_chroma ()
        feature_rate
    """
    
    ### Audio conversion parameters parameters
    # Time-frequency analysis parameters:
    Fs = 48000
    N = 2048*2
    H = N//2
    feature_rate = Fs/H

    # Load audio
    x_wav, Fs = librosa.load(path=path_audio, sr=Fs)
    X_wav = librosa.stft(x_wav, n_fft = N, hop_length=H, window='hann')
      
    # Different chroma vector approaches
    #chroma_audio_cens = librosa.feature.chroma_cens(y=x_wav, sr=Fs, hop_length=H)
    #chroma_audio = librosa.feature.chroma_cqt(y=x_wav, sr=Fs, C=X_wav, hop_length=H)
    #chroma_audio_cqt = librosa.feature.chroma_cqt(y=x_wav, sr=Fs, hop_length=H, threshold=0.1)

    # Aproach 2. - first aply stft and separate harmonic and percussive elements and use harmonics to compute the spectrogram and percusives to compute transient curve
    # Does it make sense though? because the exact number of notes is important for the sync and by separation there will be a difference in the files.
    # More sense makes the novelty detection to improve time precision.
   
    # DEBUG - try different chroma audio aproaches
    # TODO: make this selectable to be able to compare in the text
    chroma_audio = librosa.feature.chroma_stft(y=x_wav, sr=Fs, n_fft=N, hop_length=H)
    # chroma_audio = chroma_audio_cens # this one looks horrible but for some reason it seems to work quite well
    #chroma_audio = chroma_audio_cqt
    # chroma_audio = chroma_audio_harmonic
    # chroma_audio = chroma_smooth


    # Load midi and export it to chroma representation
    df_midi =  handle.midi_to_csv(midi=path_midi, csv_path=path_csv, debug=False)
    # Experimental - pad the midi df with shadow zero note at begining for better synch. performance - happens inside the df_to_list function from midi handler
    f_pitch = df_to_pitch_features(df_midi, feature_rate=feature_rate)
    f_chroma = pitch_to_chroma(f_pitch=f_pitch)
    #f_chroma_quantized = quantize_chroma(f_chroma=f_chroma)
    chroma_midi = f_chroma
    
    # show the input audio and midi chroma representations
    if verbose:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=True)
        img = librosa.display.specshow(chroma_audio, x_axis='frames', y_axis='chroma', cmap='gray_r', hop_length=H, ax=ax[0])
        ax[0].set(title='Audio chroma representation')    
        ax[0].set_xlabel('Time (frames)')
        ax[0].set_ylabel('Chroma')  
        ax[0].label_outer()

        librosa.display.specshow(chroma_midi, x_axis='frames', y_axis='chroma', cmap='gray_r', ax=ax[1])
        ax[1].set(title='MIDI chroma representation')
        ax[1].set_xlabel('Time (frames)')
        ax[1].set_ylabel('Chroma')
        ax[0].label_outer()
    
        fig.colorbar(img, ax=ax)
        plt.show()
                
    synced_midi = create_synced_object(df_midi, warping_path(chroma_audio, chroma_midi, feature_rate = feature_rate), feature_rate, path_output,path_csv)
        
    return synced_midi, chroma_audio, H

def log_compression(v, gamma=1.0):
    """Logarithmically compresses a value or array

    From [2]
    Notebook: C3/C3S1_LogCompression.ipynb

    Args:
        v (float or np.ndarray): Value or array
        gamma (float): Compression factor (Default value = 1.0)

    Returns:
        v_compressed (float or np.ndarray): Compressed value or array
    """
    return np.log(1 + gamma * v)
    
def warping_path(X, Y, feature_rate=50, show=False):
    """ Computes warping path between two chromavectors
    Args:
        X, Y (np.ndarray): chroma vectors (of shape(12, num_of_time_indices)) 
            now we use synctoolbox which uses multiple resolutions for MrMsDTW
    Returns: 
        wp (np.ndarray [shape=(N, 2)]): Warping path with index pairs.
    """

    # Synctoolbox implementation
    wp_full = sync_via_mrmsdtw(f_chroma1=X,
                      #f_onset1=f_DLNCO_audio, 
                      f_chroma2=Y, 
                      #f_onset2=f_DLNCO_annotation, 
                      input_feature_rate=feature_rate, 
                      #step_weights=step_weights, 
                      #threshold_rec=threshold_rec, 
                      verbose=show)

    # For application we need the path to be strictly monotonic!
    # Theres multiple ways to do so - interpolation, omiting some values, etc. ...
    wp = make_path_strictly_monotonic(wp_full)

    return wp

def create_synced_object(df_original_midi_data, wp, feature_rate, path_midi, path_csv):
    """
    Creates synchronized midi and csv object at specified locations
    
    Args: 
        original_midi_data
        wp
        Fs
        H
        path_midi
        path_csv
    Returns:
        df_warped(pd.dataFrame): data frame with "midi-csv" formated data warped with the provided warping path
    """
    
    df_annotated = df_original_midi_data
        
    #NOTE: problem!! warping path of the midi data is for the midi data of the chromagram which we need to adjust and then transform back into midi.
    # Question? How to do that?
    #  
    # I aksed Prof. Mueller and he said one way to synchronize the two objects is to interpolate the warping path
    # the best thing to do would be to interpolate the warping path so that it is "continuous" 
    # (more on the struggle of interpolation later -> TODO)
    # then find new values for midi notes (original time values in the csv_from_midi object)

    df_warped = df_annotated.copy(deep=True)
    # since we added end to the default data frame this is unnecessary
    #df_warped["end"] = df_warped["start"] + df_warped["duration"]
    df_warped[['start', 'end']] = scipy.interpolate.interp1d(wp[1] / feature_rate, 
                               wp[0] / feature_rate, kind='linear', fill_value="extrapolate")(df_annotated[['start', 'end']])
    df_warped["duration"] = df_warped["end"] - df_warped["start"]
    
    handle.MIDI_handler.create_midi_from_csv_experimental(path_output_file=path_midi,csv=df_warped)
    handle.MIDI_handler.midi_to_csv(path_midi, path_csv, max_duration = 100)
    
    return df_warped

def dtw_test(filename='dtw_test', show=False):
    """ Testing function for the DTW script 
    """
    #test_audio, Fs = handle.read_audio(os.path.join('..', '..', 'data', 'audio', 'test.wav'))
    #handle.plot_signal_in_time(test_audio, Fs)

    # This settings showed to be crucial!!
    Fs = 48000
    N = 2048
    H = N//2
    fn_wav_x = os.path.join('..', '..', 'data', 'input', 'audio','tests', f'{filename}.wav')
    # TODO: is this correct?
    feature_rate = Fs/H

    # Load audio
    x_wav, Fs = librosa.load(fn_wav_x, sr=Fs)
    # export it to chroma representation
    chroma_audio = librosa.feature.chroma_stft(y=x_wav, sr=Fs, hop_length=H, n_fft=N)

    # Load midi and export it to chroma representation
    path_midi =os.path.join('..', '..', 'data', 'input', 'MIDI', 'tests', f'{filename}.mid')
    path_csv = os.path.join('..', '..', 'data', 'CSV', f'{filename}.csv')
    df_midi = handle.midi_to_csv(midi=path_midi, csv_path=path_csv)
    f_pitch = df_to_pitch_features(df_midi, feature_rate=feature_rate)
    f_chroma = pitch_to_chroma(f_pitch=f_pitch)
    #f_chroma_quantized = quantize_chroma(f_chroma=f_chroma)
    chroma_midi = f_chroma

    # show the audio and midi chroma representations
    if show:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=False)
        img = librosa.display.specshow(chroma_audio, x_axis='frames', y_axis='chroma', cmap='gray_r', hop_length=H, ax=ax[0])
        ax[0].set(title='Audio chroma representation')    
        ax[0].set_xlabel('Time (frames)')
        ax[0].set_ylabel('Chroma')  
        ax[0].label_outer()

        librosa.display.specshow(chroma_midi, x_axis='frames', y_axis='chroma', cmap='gray_r', ax=ax[1])
        ax[1].set(title='MIDI chroma representation')
        ax[1].set_xlabel('Time (frames)')
        ax[1].set_ylabel('Chroma')
        ax[0].label_outer()
    
        fig.colorbar(img, ax=ax)
        plt.show()

    # compute optimal warping path
    wp = warping_path(chroma_audio, chroma_midi, feature_rate=feature_rate, show=show)
    # create synchronized midi with the computed warping path
    new_midi_path = os.path.join('..', '..', 'data', 'output', f's_{filename}.mid')    
    new_csv_path = os.path.join('..', '..', 'data', 'CSV',  f'{filename}_synced.csv')    
    synced_midi = create_synced_object(df_midi, wp, feature_rate=feature_rate, path_midi = new_midi_path, path_csv = new_csv_path)
    # Compare the original midi with the new midi and audio representation
    handle.compare_midi(df_midi, synced_midi, audio_chroma=chroma_audio, audio_hop=H)
    
    different_files= False
    if different_files:
        #TODO: create function doing this whole process:
        # Load different audio 
        fn_wav_x = os.path.join('..', '..', 'data', 'audio', 'dtw_test_whistle.wav')
        x_wav, Fs = librosa.load(fn_wav_x, sr=Fs)
        # export it to chroma
        X_audio = librosa.feature.chroma_stft(y=x_wav, sr=Fs, hop_length=H, n_fft=N)
        # compute optimal wp
        wp_piano = warping_path(X_audio, chroma_midi, feature_rate=feature_rate, show=show)
        # create synced object
        midi_path = os.path.join('..', '..', 'data', 's_dtw_test_synced.mid')    
        csv_path = os.path.join('..', '..', 'data', 'CSV', 'dtw_test_synced.csv')    
        synced_midi=create_synced_object(df_midi, wp_piano, feature_rate=feature_rate, path_midi=midi_path, path_csv = csv_path )
        # compare 
        handle.compare_midi(df_midi, synced_midi, audio_chroma=X_audio, audio_hop=H)    
    
    
    # question how to adjust if more tones that originally start at the same time are now all different? How does the warping path reflect that?
    # Now that I think musically about it how would I even align that as a human listener? so I guess it doesn't really matter. If it picks the best match (The WP with least cost)
    # It should probably work