"""
Module: MIDI_handler
Author: Petr Němec
License: 

Some functions were taken from or inspired by the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""

from copy import deepcopy
import os
import pandas as pd
import pretty_midi
import librosa.display

from matplotlib import pyplot as plt
from matplotlib import patches
from tabulate import tabulate

from libfmp.c1.c1s2_symbolic_rep import visualize_piano_roll

INITIAL_TEMPO = 60

def __compare_midi(df_original, df_synced, audio_chroma):
    """Plot two piano-rolls together with the audio interpretation chroma
    Args:
        df_original
        df_synced
        audio_chroma
    
    """
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8,5), sharex=False)
    
    visualize_piano_roll(csv_to_list(df_original),
                         xlabel='Time (seconds)',
                         ylabel='Chroma pitch',
                         colors='FMP_1',
                         velocity_alpha=True,
                         figsize=(8,5),
                         ax=axs[0])    
    axs[0].set(title='Original MIDI')     
    axs[0].label_outer()
    
    visualize_piano_roll(csv_to_list(df_synced),
                         xlabel='Time (seconds)',
                         ylabel='Chroma pitch',
                         colors='FMP_1',
                         velocity_alpha=True,
                         figsize=(8,5),
                         ax=axs[1])    
    axs[1].set(title='Synchronized MIDI')    
    axs[1].label_outer()

    fig.tight_layout()
    
    plt.show()
        
    return fig, axs

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
                
            start = note.start*(t/INITIAL_TEMPO)
            end = note.end*(t/INITIAL_TEMPO)
            duration = end - start                

            pitch = note.pitch
            velocity = note.velocity / 128.
            score.append([start, duration, pitch, velocity, instrument.name.lower()])

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
    df = pd.DataFrame(note_list, columns=['start', 'duration', 'pitch', 'velocity', 'instrument'])
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
    midi_out = pretty_midi.PrettyMIDI(initial_tempo=INITIAL_TEMPO)

    # Create a dictionary to keep track of instruments by name
    instruments = {}

    # Iterate over the rows of the DataFrame and add MIDI information to midi_out
    for _, row in df.iterrows():
        start_time = float(row['start'])
        #end_time = float(row['end'])
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
            end = start_time + duration
        )

        # Add the note to the instrument
        instrument.notes.append(note)

    # Add the instruments to the PrettyMIDI object
    for instrument in instruments.values():
        midi_out.instruments.append(instrument)

    # Write out the MIDI data
    midi_out.write(fn_out)

    return midi_out

def load_midi_as_df(fn):
    """ Loads MIDI file at specified path
    converts to a list
    Recalculates all times for one single tempo (INITIAL_TEMPO = 120)
    Args:
        fn(path): path to midi file
    Returns:
        (pd.DataFrame ): converted midi 
    """
    return list_to_csv(midi_to_list(load_midi(fn)))

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
    original = list_to_csv(score, path_csv)


    print('We can also create a midi back from the csv file - even though there are some problems and it won_t be the same')

    path_midi= os.path.join('..', '..', 'data', 'MIDI', 'from_csv','test.mid')
    csv_to_midi(path_csv, path_midi)
    new_midi = load_midi_as_df(path_midi)

    __compare_midi(original, new_midi, None)
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


