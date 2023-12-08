"""
Module: MIDI_handler
Author: Petr Němec
License: 

Some functions were taken from or inspired by the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""

from copy import deepcopy
import os
import sys
import numpy as np
import pandas as pd
import pretty_midi
import librosa.display

from matplotlib import pyplot as plt
from matplotlib import patches
from tabulate import tabulate

# FROM FMP NOTEBOOKS - TODO: move it to another file?
FMP_COLORMAPS = {
    'FMP_1': np.array([[1.0, 0.5, 0.0], [0.33, 0.75, 0.96], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0],
                       [1.0, 0.0, 1.0],  [0.99, 0.51, 0.71], [0.53, 0.0, 0.46], [0.56, 0.93, 0.72], [0, 0, 0.9]])
}
def color_argument_to_dict(colors, labels_set, default='gray'):
    """Create a dictionary that maps labels to colors.

    Args:
        colors: Several options: 1. string of ``FMP_COLORMAPS``, 2. string of matplotlib colormap,
            3. list or np.ndarray of matplotlib color specifications, 4. dict that assigns labels  to colors
        labels_set: List of all labels
        default: Default color, used for labels that are in labels_set, but not in colors

    Returns:
        color_dict: Dictionary that maps labels to colors
    """

    if isinstance(colors, str):
        # FMP colormap
        if colors in FMP_COLORMAPS:
            color_dict = {l: c for l, c in zip(labels_set, FMP_COLORMAPS[colors])}
        # matplotlib colormap
        else:
            cm = plt.get_cmap(colors)
            num_labels = len(labels_set)
            colors = [cm(i / (num_labels + 1)) for i in range(num_labels)]
            color_dict = {l: c for l, c in zip(labels_set, colors)}

    # list/np.ndarray of colors
    elif isinstance(colors, (list, np.ndarray, tuple)):
        color_dict = {l: c for l, c in zip(labels_set, colors)}

    # is already a dict, nothing to do
    elif isinstance(colors, dict):
        color_dict = colors

    else:
        raise ValueError('`colors` must be str, list, np.ndarray, or dict')

    for key in labels_set:
        if key not in color_dict:
            color_dict[key] = default

    return color_dict
def visualize_piano_roll(score, xlabel='Time (seconds)', ylabel='Pitch', colors='FMP_1', velocity_alpha=False,
                         figsize=(12, 4), ax=None, dpi=72):
    """Plot a pianoroll visualization

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

    labels_set = sorted(set([note[4] for note in score]))
    colors = color_argument_to_dict(colors, labels_set)

    pitch_min = min(note[2] for note in score)
    pitch_max = max(note[2] for note in score)
    time_min = min(note[0] for note in score)
    time_max = max(note[0] + note[1] for note in score)

    for start, duration, pitch, velocity, label in score:
        if velocity_alpha is False:
            velocity = None
        rect = patches.Rectangle((start, pitch - 0.5), duration, 1, linewidth=1,
                                 edgecolor='k', facecolor=colors[label], alpha=velocity)
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
# END OF Part to be moved


def load_midi(fn=os.path.join('..', '..', 'data', 'MIDI', 'test.mid')):
    """Load midi file into the midi_data var
        Args:
            fn (os.path): path to the file to be loaded
                - default is the test file provided: os.path.join('..', '..', 'data', 'MIDI', 'test.mid')

        Returns: 
            midi_data (pretty_midi.PrettyMIDI): loaded midi data
    """
    
    midi_data = pretty_midi.PrettyMIDI(fn)
    return midi_data

def midi_to_list(midi):
    """Convert a midi file to a list of note events
        For compensating different tempo the start and duration parameters are recalculated for 120 bpm

    Args:
        midi (str or pretty_midi.pretty_midi.PrettyMIDI): Either a path to a midi file or PrettyMIDI object

    Returns:
        score (list): A list of note events where each note is specified as
            ``[start, duration, pitch, velocity, label]``
    """

    if isinstance(midi, str):                                   #check if midi arg is string
        midi_data = pretty_midi.pretty_midi.PrettyMIDI(midi)
    elif isinstance(midi, pretty_midi.pretty_midi.PrettyMIDI):  #check if midi arg is prettyMidi object
        midi_data = midi
    else:
        raise RuntimeError('midi must be a path to a midi file or pretty_midi.PrettyMIDI')
    
    # TODO: adjust the start and duration times even if there are tempo changes
    tempo_changes, tempo = midi_data.get_tempo_changes()
    
    score = []
        
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            t = tempo[0] # get the tempo from the tempo list by getting the index of current tempo change index
                
            start = note.start*(t/120)
            end = note.end*(t/120)
            duration = -start+end                

            pitch = note.pitch
            velocity = note.velocity / 128.
            score.append([start, end, duration, pitch, velocity, instrument.name.lower()])

    return score

def list_to_csv(note_list, fn_out=None):
    """Write a list of note events (comprising a start time, duration, pitch, velocity, and label for each note event)
    to a CSV file

    Args:
        score (list): List of note events
        fn_out (str): The path of the csv file to be created

    Returns: 
        df (pd.DataFrame): data frame with the information saved to csv
    """
    df = pd.DataFrame(note_list, columns=['start', 'end', 'duration', 'pitch', 'velocity', 'instrument'])
    # ideally, I would like to use float_format='%.3f', but then the numeric columns are considered as strings and,
    # therefore, are quoted
    if fn_out is not None:  df.to_csv(fn_out, sep=';', index=False, quoting=2)

    return df

def read_csv(fn, header=True, add_label=False):
    """Read a CSV file in table format and creates a pd.DataFrame from it, with observations in the
    rows and variables in the columns.

    Args:
        fn (str): Filename
        header (bool): Boolean (Default value = True)
        add_label (bool): Add column with constant value of `add_label` (Default value = False)

    Returns:
        df (pd.DataFrame): Pandas DataFrame
    """
    df = pd.read_csv(fn, sep=';', keep_default_na=False, header=0 if header else None)
    if add_label:
        assert 'label' not in df.columns, 'Label column must not exist if `add_label` is True'
        df = df.assign(label=[add_label] * len(df.index))
    return df

def csv_to_list(csv):
    """Convert a csv score file to a list of note events

    Notebook: C1/C1S2_CSV.ipynb

    Args:
        csv (str or pd.DataFrame): Either a path to a csv file or a data frame

    Returns:
        score (list): A list of note events where each note is specified as
            ``[start, duration, pitch, velocity, label]``
    """

    if isinstance(csv, str):
        df = read_csv(csv)
    elif isinstance(csv, pd.DataFrame):
        df = csv
    else:
        raise RuntimeError('csv must be a path to a csv file or pd.DataFrame')

    score = []
    for i, (start, duration, pitch, velocity, label) in df.iterrows():
        score.append([start, duration, pitch, velocity, label])
    return score

def csv_to_midi(csv, fn_out):
    """Convert a csv to midi file

    Args: 
        csv (str or pd.DataFrame): Either a path to a csv file or a data frame with note information
        fn_out (str): The path of the midi file to be created (including ../name.mid)

    Returns: 
        midi_out (pretty_midi.PrettyMIDI()): a pretty midi object 
    """

    # First load the CSV file as data frame
    if isinstance(csv, str):
        df = pd.read_csv(csv, sep=';', keep_default_na=False)
    elif isinstance(csv, pd.DataFrame):
        df = csv
    else:
        raise RuntimeError('csv must be a path to a csv file or pd.DataFrame')

    # Create a PrettyMIDI object
    midi_out = pretty_midi.PrettyMIDI(initial_tempo=120)

    # Create a dictionary to keep track of instruments by name
    instruments = {}

    # Iterate over the rows of the DataFrame and add MIDI information to midi_out
    for _, row in df.iterrows():
        start_time = float(row['start'])
        end_time = float(row['end'])
        duration = float(row['duration'])
        pitch = int(row['pitch'])
        velocity = int(row['velocity']*128)
        try:
            instrument_name = row['instrument']
            instrument_id = pretty_midi.instrument_name_to_program(instrument_name)
        except ValueError:
            print('Not a valid instrument name - instrument set to piano')
            instrument_id = 1

        # Check if the instrument already exists
        if instrument_name in instruments:
            instrument = instruments[instrument_name]
        else:
            # Create a new Instrument object for the note
            instrument = pretty_midi.Instrument(program=instrument_id)
            instruments[instrument_name] = instrument

        # Create a Note object with the extracted MIDI information
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start_time,
            end=end_time
        )

        # Add the note to the instrument
        instrument.notes.append(note)

    # Add the instruments to the PrettyMIDI object
    for instrument in instruments.values():
        midi_out.instruments.append(instrument)

    # Write out the MIDI data
    midi_out.write(fn_out)

    return midi_out

# TODO: try different aproach with changing the original midi instead of creating a completely new midi from csv
def midi_and_csv_to_midi(pm_original_midi, df_warped, fn_out):
    # deep copy original midi
    pm_new_midi = deepcopy(pm_original_midi)
    new_times = []
    pm_new_midi.adjust_times(all, new_times)
    return pm_new_midi
    


# Testing functions for optimizing the MIDI handler functions
def test():
    print('First lets check the loading of midi data: ')
    midi_data = load_midi(fn=os.path.join('..', '..', 'data', 'MIDI', 'test.mid'))
    score = midi_to_list(midi_data)

    print(score)

    print(tabulate(score, headers=['start', 'duration', 'pitch', 'velocity', 'instrument name']))
    print('')

    print('Now lets generate csv file from this list')
    print('')

    path_csv = os.path.join('..', '..', 'data','CSV', 'test.csv')
    list_to_csv(score, path_csv)


    print('We can also create a midi back from the csv file - even though there are some problems and it won_t be the same')

    path_midi= os.path.join('..', '..', 'data', 'MIDI', 'from_csv','test.mid')
    midi_data_from_csv = csv_to_midi(path_csv, path_midi)

    print(score)

    # using the variable ax for single a Axes

    fig, ax = visualize_piano_roll(midi_to_list(midi_data_from_csv), velocity_alpha=True)
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
    new_midi = csv_to_midi(csv, path_midi)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
    visualize_piano_roll(midi_to_list(md), velocity_alpha=True, ax=ax[0])
    ax[0].set(title='Original MIDI')
    visualize_piano_roll(midi_to_list(new_midi), velocity_alpha=True, ax=ax[1])
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


