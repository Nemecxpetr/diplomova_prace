"""
Module: MIDI_handler
Author: Petr Němec
License: 

Some functions were taken from or inspired by the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""

from copy import deepcopy
import os
from tkinter import N
import pandas as pd
import pretty_midi
import librosa.display
import regex

from midiutil import MIDIFile
from matplotlib import pyplot as plt
from matplotlib import patches
from tabulate import tabulate

import libfmp.b.b_plot
from pretty_midi.utilities import qpm_to_bpm

INITIAL_TEMPO = 120

def __compare_midi(df_original : pd.DataFrame,
                   df_synced : pd.DataFrame,
                   audio_chroma = None):
    """Plot two piano-rolls together with the audio interpretation chroma
    Args:
        df_original
        df_synced
        audio_chroma
    
    """
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8,5), sharex=True)
    
    __visualize_piano_roll(df_to_list(df_original),
                         xlabel='Time (seconds)',
                         ylabel='Chroma pitch',
                         colors='FMP_1',
                         velocity_alpha=True,
                         figsize=(8,5),
                         ax=axs[0])    
    axs[0].set(title='Original MIDI')     
    axs[0].label_outer()
    
    __visualize_piano_roll(df_to_list(df_synced),
                         xlabel='Time (seconds)',
                         ylabel='Chroma pitch',
                         colors='FMP_1',
                         velocity_alpha=True,
                         figsize=(8,5),
                         ax=axs[1])    
    axs[1].set(title='New MIDI')    
    axs[1].label_outer()

    fig.tight_layout()
    
    plt.show()
        
    return fig, axs

def __visualize_piano_roll(score, xlabel='Time (seconds)', ylabel='Pitch', colors='FMP_1', velocity_alpha=False,
                         figsize=(12, 4), ax=None, dpi=72):
    """Plot a pianoroll visualization

    Inspired by: Notebook: C1/C1S2_CSV.ipynb from Meinard Müller
    Adapted for different score structure

    Args:
        score: List of note events
        xlabel: Label for x axis (Default value = 'Time (seconds)')
        ylabel: Label for y axis (Default value = 'Pitch')
        colors: Several options: 1. string of FMP_COLORMAPS, 2. string of matplotlib colormap,
            3. list or np.ndarray of matplotlib color specifications,
            4. dict that assigns labels  to colors (Default value = 'FMP_1')
        velocity_alpha: Use the velocity value for the alpha value of the corresponding rectangle
            (Default value = False)
        figsize: Width, height in inches (Default value = (12)
        ax: The Axes instance to plot on (Default value = None)
        dpi: Dots per inch (Default value = 72)

    Returns:
        fig: The created matplotlib figure or None if ax was given.
        ax: The used axes
    """
    fig = None
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.subplot(1, 1, 1)

    # EDIT: The labels are at different position
    labels_set = sorted(set([note[5] for note in score]))
    colors = libfmp.b.color_argument_to_dict(colors, labels_set)
    
    # EDIT: Also the pitch is somewhere else:
    pitch_min = min(note[3] for note in score)
    pitch_max = max(note[3] for note in score)
    time_min = min(note[0] for note in score)
    time_max = max(note[0] + note[2] for note in score)

    # The values are not needed but we need to iterate correctly
    for start, _, duration, pitch, velocity, instr, _, _ in score:
        if velocity_alpha is False:
            velocity = None
        rect = patches.Rectangle((start, pitch - 0.5), duration, 1, linewidth=1,
                                 # EDIT: expects velocity normalized in interval of 0:1
                                 edgecolor='k', facecolor=colors[instr], alpha=velocity/128) 
        ax.add_patch(rect)

    ax.set_ylim([pitch_min - 1.5, pitch_max + 1.5])
    ax.set_xlim([min(time_min, 0), time_max + 0.5])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.set_axisbelow(True)
    ax.legend([patches.Patch(linewidth=1, edgecolor='k', facecolor=colors[key]) for key in labels_set],
              labels_set, loc='upper right', framealpha=1)

    if fig is not None:
        plt.tight_layout()

    return fig, ax

def load_midi(fn=os.path.join('..', '..', 'data', 'MIDI', 'test.mid')):
    """Load midi file into the midi_data var
        Args:
            fn (os.path): path to the file to be loaded
                - default is the test file provided: os.path.join('..', '..', 'data', 'MIDI', 'test.mid')
        Returns: 
            midi_data (pretty_midi.PrettyMIDI): loaded midi data
    
        #TODO NOTE: this function is basically pointless so could be instead incorporated into other functions
            
    """    
    midi_data = pretty_midi.PrettyMIDI(fn)
    return midi_data

def midi_to_list(midi: str or pretty_midi.pretty_midi.PrettyMIDI,
                 max_duration: int = 10,
                 debug: bool = False) -> list:
    """
    Convert a midi file to a list of note events.
    Inspired by: Notebook: C1/C1S2_MIDI.ipynb from Meinard Müller

    Args:
        midi (str or pretty_midi.pretty_midi.PrettyMIDI):       path to a midi file or PrettyMIDI object
        debug (bool):                                           if True, additional info is printed

    Returns:
        score (list):                                           a list of note events
                                                                (start, end, duration, pitch, velocity, instr, instr_program, midi_channel)
    """

    if isinstance(midi, str):
        midi_data = pretty_midi.pretty_midi.PrettyMIDI(midi)
    elif isinstance(midi, pretty_midi.pretty_midi.PrettyMIDI):
        midi_data = midi
    else:
        raise RuntimeError('midi must be a path to a midi file or pretty_midi.PrettyMIDI')

    score = []
    midi_channel = 0
    previous_instr_program = ''
    offset = 0

    for i, instrument in enumerate(midi_data.instruments):
        instr = regex.sub(r'[^\p{Latin} ]', u'', instrument.name)
        instr_program = instrument.program

        #TODO NOTE: Adding round for visualisation in the thesis
        # measure to be deleted or accesed by some print_thesis command or smth
        ndigits = 3
        for note in instrument.notes:
            start = round(note.start, ndigits)
            end = round(note.end, ndigits)
            duration = round(note.end - start, ndigits)
            if duration > max_duration:
                print(f'Max duration: {duration}, skipping this note.')
                continue
            pitch = note.pitch
            velocity = note.velocity

            start = start - offset
            end = end - offset

            if i != 0:
                if previous_instr_program != instr_program:
                    midi_channel += 1
                    if midi_channel == 9:
                        midi_channel = 10

            previous_instr_program = instrument.program

            if 'Drum' in instr:
                instr = 'Drumset'
                midi_channel = 9

            if instrument.is_drum:
                if debug:
                    print('drums track now')
                midi_channel = 9

            if debug:
                print(f'Instrument: {instr}, start: {start}, instr_program: {instr_program}, pitch: {pitch}, duration: {duration}, '
                      f'velocity: {velocity}, midi_channel: {midi_channel}')
            score.append([start, end, duration, pitch, velocity, instr, instr_program, midi_channel])
    return score

def list_to_csv(note_list, fn_out=None):
    """Write a list of note events (comprising a start time, duration, pitch, velocity, and label for each note event)
    to a CSV file

    Inspired by: Notebook: C1/C1S2_MIDI.ipynb from Meinard Müller
    
    Args:
        score (list): List of note events
        fn_out (str): The path of the csv file to be created

    Returns: 
        df (pd.DataFrame): data frame with the information saved to csv
    """
    df = pd.DataFrame(note_list,  columns=['start', 'end', 'duration', 'pitch',
                                            'velocity', 'instrument', 'instr_program', 'midi_channel'])
  
    # NOTE: ideally, I would like to use float_format='%.3f', but then the numeric columns are considered as strings and,
    # therefore, are quoted
    if fn_out is not None:  df.to_csv(fn_out, sep=';', index=False, quoting=2)

    return df

def read_csv(fn : str or os.path, separator : str = ';', header=True, add_label=False):
    """Read a CSV file in table format and creates a pd.DataFrame from it, with observations in the
    rows and variables in the columns.

    Args:
        fn (str): Filename
        header (bool): Boolean (Default value = True)
        add_label (bool): Add column with constant value of `add_label` (Default value = False)

    Returns:
        df (pd.DataFrame): Pandas DataFrame
    """
    df = pd.read_csv(fn, sep=separator, keep_default_na=False, header=0 if header else None)
    if add_label:
        assert 'label' not in df.columns, 'Label column must not exist if `add_label` is True'
        df = df.assign(label=[add_label] * len(df.index))
    return df

def df_to_list(csv : str or pd.DataFrame):
    """Convert a data frame score file to a list of note events

    Notebook: C1/C1S2_CSV.ipynb

    Args:
        csv (str or pd.DataFrame): Either a path to a csv file or a data frame

    Returns:
        score (list): A list of note events where each note is specified as
            ``[start, end, duration, pitch, velocity, instr, instr_program, midi_channel]``
    """

    if isinstance(csv, str):
        df = read_csv(csv)
    elif isinstance(csv, pd.DataFrame):
        df = csv
    else:
        raise RuntimeError('csv must be a path to a csv file or pd.DataFrame')

    score = []
    for i, (start, end, duration, pitch, velocity, instr, instr_program, midi_channel) in df.iterrows():
        score.append([start, end, duration, pitch, velocity, instr, instr_program, midi_channel])
    return score

def convert_seconds_to_quarter(seconds : float,
                               bpm : float):
    """Converts seconds to quaters with set bpm in 4/4
    
    # NOTE: pretty midi has function qpm_to_bpm that does opposite to this?
    # TODO: will this work with songs in different measures? like 6/12, 3/4, 5/4, 7/4, etc. ...??
    
    Args: seconds (float) : time in seconds
          bpm (float) : tempo in beats per minute
    Return: 
        qpm (float) : quaters per minute
    """
    return seconds * (4 / (bpm/60))

def create_midi_from_csv_experimental(path_output_file: str,
                                      csv: str or pd.DataFrame,
                                      bpm: int = 120,
                                      debug: bool = False):
    if isinstance(csv, str):
        df_csv = pd.read_csv(csv)
    elif isinstance(csv, pd.DataFrame):
        df_csv = csv
    else:
        raise RuntimeError('csv must be a path to a csv file or pd.DataFrame')
    
    unique_instruments = df_csv['midi channel'].unique()
    my_midi_file = MIDIFile(len(unique_instruments), adjust_origin=False)
    if debug:
        print(f'Unique instruments: {unique_instruments}')

    previous_track = None
    for i, row in df_csv.iterrows():

        channel = int(row['midi channel'])
        instr_program = int(row['instr program'])
        pitch = int(row['pitch'])

        track = 0
        if track != previous_track:
            my_midi_file.addTempo(track=track, time=0, tempo=bpm)
        previous_track = track

        if i == 0:
            my_midi_file.addProgramChange(0, channel, 0, instr_program)
        else:
            if previous_channel != channel:
                my_midi_file.addProgramChange(0, channel, 0, instr_program)

        if debug:
            print(
                f"Note {pitch}, start at {convert_seconds_to_quarter(row['start'], bpm)} and duration "
                f"{convert_seconds_to_quarter(row['duration'], bpm)}, bpm: {bpm}, volume: {row['velocity']}, "
                f"instr program: {instr_program}, channel: {channel}, track: {track}")

        my_midi_file.addNote(track=track, channel=channel, time=convert_seconds_to_quarter(row['start'], bpm),
                             pitch=pitch, volume=int(row['velocity']),
                             duration=convert_seconds_to_quarter(row['duration'], bpm))
        previous_channel = channel

    # create and save the midi file itself
    with open(f'{path_output_file}', "wb") as output_file:
        my_midi_file.writeFile(output_file)
   
def midi_to_csv(midi: str or pretty_midi.pretty_midi.PrettyMIDI,
                csv_path: str,
                debug: bool = False):
    """
    Convert a midi file to a csv file and save it.

    Args:
        midi_path:          path of the input .mid file or the data of midi file
        csv_path:           path of the output .csv
        debug:              if True, additional info is printed
    """
    
    if isinstance(midi, str):
        midi_data = pretty_midi.pretty_midi.PrettyMIDI(midi)
    elif isinstance(midi, pretty_midi.pretty_midi.PrettyMIDI):
        midi_data = midi
    else:
        raise RuntimeError('midi must be a path to a midi file or pretty_midi.PrettyMIDI')
   
    score = midi_to_list(midi_data, debug=debug)
    final_df = pd.DataFrame(score, columns=['start', 'end', 'duration', 'pitch',
                                            'velocity', 'instrument', 'instr program', 'midi channel'])
    
    if csv_path is not None:    final_df.to_csv(csv_path, index=False)
    
    return final_df

# TODO: try different aproach with changing the original midi instead of creating a completely new midi from csv
def midi_and_csv_to_midi(pm_original_midi, df_warped, fn_out):
    # deep copy original midi
    pm_new_midi = deepcopy(pm_original_midi)
    new_times = []
    pm_new_midi.adjust_times(all, new_times)
    return pm_new_midi
    

# Testing functions for optimizing the MIDI handler functions
def test(debug=False):
    input_midi_path = os.path.join('..', '..', 'data', 'MIDI', 'test_100bpm.mid')
    midi_data = load_midi(fn=input_midi_path)
    path_csv = os.path.join('..', '..', 'data','CSV', 'test_100bpm.csv')

    output_midi_path= os.path.join('..', '..', 'data', 'MIDI', 'from_csv','test.mid')

    df_new = midi_to_csv(midi=midi_data, csv_path=path_csv, debug=debug)
    print('Original midi data - returned by function midi_to_list')
    print(tabulate(midi_to_list(midi_data), headers=['start', 'end', 'duration', 'pitch', 'velocity', 'instr', 'instr_program', 'midi_channel']))
    
    create_midi_from_csv_experimental(path_output_file=output_midi_path, csv=df_new, bpm=INITIAL_TEMPO, debug=debug)
    
    new_midi = load_midi(fn=output_midi_path)
    print('New midi data loaded from csv')
    print(tabulate(midi_to_list(new_midi), headers=['start', 'end', 'duration', 'pitch', 'velocity', 'instr', 'instr_program', 'midi_channel']))   
    print('')
    print('Note that the original and new data are not sorted the same way')
    
    __compare_midi(midi_to_csv(input_midi_path, csv_path=None, debug=debug), midi_to_csv(output_midi_path, csv_path=None, debug=debug), None)
    plt.show()

    

def test_drums():
    path_drums = os.path.join('..', '..', 'data', 'MIDI', 'test_with_drums.mid')
    midi_data = load_midi(path_drums)    

    score = midi_to_list(midi_data)

    path_csv = os.path.join('..', '..', 'data','CSV', 'test_with_drums.csv')
    list_to_csv(score, path_csv)
    
    fig, ax = visualize_piano_roll(score, velocity_alpha=True)

    path_midi= os.path.join('..', '..', 'data', 'MIDI', 'test_drums_new.mid')
    midi_data_from_csv = csv_to_midi(path_csv, path_midi)

    # using the variable ax for single a Axes

    fig, ax = visualize_piano_roll(midi_to_list(midi_data_from_csv), velocity_alpha=True)
    plt.show()

def test_tempo():

    md = load_midi(fn=os.path.join('..', '..', 'data', 'MIDI', 'test.mid'))
    score = midi_to_list(md)
    
    path_csv = os.path.join('..', '..', 'data', 'CSV', 'test.csv')
    csv = list_to_csv(score, path_csv)

    path_midi = os.path.join('..', '..', 'data', 'MIDI','from_csv', 'test_tempo.mid')
    new_midi = create_midi_from_csv_experimental(csv, path_midi)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
    __visualize_piano_roll(midi_to_list(md), velocity_alpha=True, ax=ax[0])
    ax[0].set(title='Original MIDI')
    __visualize_piano_roll(midi_to_list(new_midi), velocity_alpha=True, ax=ax[1])
    ax[1].set(title='MIDI from CSV')
    fig.tight_layout()
    

    Y = pretty_midi.PrettyMIDI.get_chroma(md)

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8,5), sharex=False)
    librosa.display.specshow(Y, x_axis='s', y_axis='chroma', cmap='gray_r', ax=axs[0])
    axs[0].set(title='Original MIDI chroma')    
    axs[0].set_xlabel('Time (seconds)')
    axs[0].set_ylabel('Chroma')  
    axs[0].label_outer()
    

    librosa.display.specshow(pretty_midi.PrettyMIDI.get_chroma(new_midi), x_axis='s', y_axis='chroma', cmap='gray_r', ax=axs[1])
    axs[1].set(title='MIDI from CSV chroma')    
    axs[1].set_xlabel('Time (seconds)')
    axs[1].set_ylabel('Chroma')  
    axs[1].label_outer()

    fig.tight_layout()
    
    plt.show()


