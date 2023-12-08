"""
Module: DTW
Author: Bc. Petr Němec

Part of my master's thesis (https://github.com/Nemecxpetr/diplomova_prace)

Cílem semestrálního projektu je popis principů hudební synchronizace, sestavení datasetu a základní implementace score-to-audio synchronizace

Aim of this script is basic implementation of score-to-audio synchronization

"""
import os
import Handler as handle
import pretty_midi as pm
import librosa
from matplotlib import pyplot as plt
import numpy as np
import scipy

import soundfile as sf

from synctoolbox.dtw.utils import make_path_strictly_monotonic
from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
from synctoolbox.feature.csv_tools import read_csv_to_df


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

def create_synced_object(original_midi_data, wp, feature_rate, path_midi, path_csv):
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
    
    df_annotated = handle.list_to_csv(handle.midi_to_list(original_midi_data))
        
    #NOTE: problem!! warping path of the midi data is for the midi data of the chromagram which we need to adjust and then transform back into midi.
    # Question? How to do that?
    #  
    # I aksed Prof. Mueller and he said one way to synchronize the two objects is to interpolate the warping path
    # the best thing to do would be to interpolate the warping path so that it is "continuous" 
    # (more on the struggle of interpolation later -> TODO)
    # then find new values for midi notes (original time values in the csv_from_midi object)

    df_warped = df_annotated.copy(deep=True)
    df_warped["end"] = df_warped["start"] + df_warped["duration"]
    df_warped[['start', 'end']] = scipy.interpolate.interp1d(wp[1] / feature_rate, 
                               wp[0] / feature_rate, kind='linear', fill_value="extrapolate")(original_midi_data[['start', 'end']])
    df_warped["duration"] = df_warped["end"] - df_warped["start"]
    note_list = df_warped[['start', 'duration', 'pitch', 'velocity']].values.tolist()

    
    synced_csv = handle.list_to_csv(note_list, path_csv)  
    synced_midi = handle.csv_to_midi(df_warped, path_midi)
    
    return df_warped


def dtw_test(show=True):
    #test_audio, Fs = handle.read_audio(os.path.join('..', '..', 'data', 'audio', 'test.wav'))
    #handle.plot_signal_in_time(test_audio, Fs)

    # This settings showed to be crucial!!
    Fs = 48000
    N = 4096
    H = N//8
    fn_wav_x = os.path.join('..', '..', 'data', 'audio', 'dtw_test_whistle.wav')
    # TODO: is this correct?
    feature_rate = Fs/H

    x_wav, Fs = librosa.load(fn_wav_x, sr=Fs)
    X = librosa.feature.chroma_stft(y=x_wav, sr=Fs, hop_length=H, n_fft=N)

    path_midi =os.path.join('..', '..', 'data', 'MIDI', 'DTW_test.mid')
    midi_data = handle.load_midi(path_midi)
    Y = pm.PrettyMIDI.get_chroma(midi_data)

    if show:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=False)
        img = librosa.display.specshow(X, x_axis='frames', y_axis='chroma', cmap='gray_r', hop_length=H, ax=ax[0])
        ax[0].set(title='Sequence $X$')    
        ax[0].set_xlabel('Time (frames)')
        ax[0].set_ylabel('Chroma')  
        ax[0].label_outer()

        librosa.display.specshow(Y, x_axis='frames', y_axis='chroma', cmap='gray_r', ax=ax[1])
        ax[1].set(title='Sequence $Y$')
        ax[1].set_xlabel('Time (frames)')
        ax[1].set_ylabel('Chroma')
        ax[0].label_outer()
    
        fig.colorbar(img, ax=ax)
        plt.show()

    #NOTE: X = audio chroma, Y = score chroma
    wp = warping_path(X, Y, feature_rate=feature_rate, show=show)
    midi_path = os.path.join('..', '..', 'data', 'MIDI', 'from_csv', 'dtw_test_synced_with_whistle.mid')    
    csv_path = os.path.join('..', '..', 'data', 'CSV',  'dtw_test_synced_with_whistle.csv')    
    create_synced_object(midi_data, wp, feature_rate=feature_rate, path_midi = midi_path, path_csv = csv_path)

    fn_wav_x = os.path.join('..', '..', 'data', 'audio', 'dtw_test.wav')
    x_wav, Fs = librosa.load(fn_wav_x, sr=Fs)
    X_whistle = librosa.feature.chroma_stft(y=x_wav, sr=Fs, hop_length=H, n_fft=N)
    wp_whistle = warping_path(X_whistle, Y, feature_rate=feature_rate, show=show)

    midi_path = os.path.join('..', '..', 'data', 'MIDI', 'from_csv', 'dtw_test_synced_with_ms_piano.mid')    
    csv_path = os.path.join('..', '..', 'data', 'CSV', 'dtw_test_synced_with_ms_piano.csv')    
    create_synced_object(midi_data, wp_whistle, feature_rate=feature_rate, path_midi=midi_path, path_csv = csv_path )
    
    fn_wav_x = os.path.join('..', '..', 'data', 'audio', 'dtw_test_voice_slow.wav')
    x_wav, Fs = librosa.load(fn_wav_x, sr=Fs)
    X_whistle = librosa.feature.chroma_stft(y=x_wav, sr=Fs, hop_length=H, n_fft=N)
    wp_whistle = warping_path(X_whistle, Y,feature_rate=feature_rate, show=show)

    midi_path = os.path.join('..', '..', 'data', 'MIDI', 'from_csv', 'dtw_test_synced_with_voice_slow.mid')    
    csv_path = os.path.join('..', '..', 'data', 'CSV', 'dtw_test_synced_with_voice_slow.csv')    
    create_synced_object(midi_data, wp_whistle, feature_rate=feature_rate, path_midi=midi_path, path_csv = csv_path )
    

    # TODO: create compare midi function that will plot piano roll of original and new midi
    #handle.compare_midi(synced_midi, midi_data)

    # question how to adjust if more tones that originally start at the same time are now all different? How does the warping path reflect that?
    # Now that I think musically about it how would I even align that as a human listener? so I guess it doesn't really matter. If it picks the best match (The WP with least cost)
    # It should probably work


dtw_test(show=True)


"""
Pomocné funkce pro tisk do diplomové práce
"""
#TODO: přesunout do jiného souboru (nejlépe DP.py)

def ukazka_spektrogramu():
    path = os.path.join('..', '..', 'data', 'audio', 'test.wav')
    x, Fs = sf.read(path)
    # lets use only left channel
    x = x.T[0]

    N=4096//2
    H=N//2
    X = librosa.stft(x, n_fft = N, hop_length = H, win_length=N, window = 'hann', center = True, pad_mode = "constant")
    gamma = 100
    Y = 20*np.log10(1+gamma*abs(X))
    fig = fmpplot.plot_matrix(Y, Fs)
    plt.yscale('log')
    plt.ylim([20, Fs/2])
    plt.show()