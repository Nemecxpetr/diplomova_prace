﻿"""
Module: audio_handler
Author: Petr Němec
License: 

Some functions were taken from or inspired by the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP) and libfmp.
"""

import librosa
import soundfile as sf
from pydub import AudioSegment


def read_audio(path, Fs=None, mono=False):
    """Read an audio file into a np.ndarray.

    Args:
        path (str): Path to audio file
        Fs (scalar): Resample audio to given sampling rate. Use native sampling rate if None. (Default value = None)
        mono (bool): Convert multi-channel file to mono. (Default value = False)

    Returns:
        x (np.ndarray): Waveform signal
        Fs (scalar): Sampling rate
    """
    return librosa.load(path, sr=Fs, mono=mono)


def write_audio(path, x, Fs):
    """Write a signal (as np.ndarray) to an audio file.

    Args:
        path (str): Path to audio file
        x (np.ndarray): Waveform signal
        Fs (scalar): Sampling rate
    """
    sf.write(path, x, Fs)



def convert_to_wav(input_path, output_path, format='m4a'):
    """Converts input files from various formats to WAV format.
    Args:
        input_path (str): Path to input file
        output_path (str): Path to output WAV file
    Returns:
        None
    """

    sound = AudioSegment.from_file(input_path, format=format)
    file_handle = sound.export(output_path, format='wav')