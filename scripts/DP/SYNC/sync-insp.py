
import pretty_midi
import pandas as pd
import regex
from glob import glob
from pathlib import Path
from midiutil import MIDIFile


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
                                                                (start, end, duration, pitch, velocity, instrument)
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

        for note in instrument.notes:
            start = note.start
            end = note.end
            duration = note.end - start
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


def midi_to_csv(midi_path: str,
                csv_path: str,
                debug: bool = False):
    """
    Convert a midi file to a csv file and save it.

    Args:
        midi_path:          path of the input .mid file or the data of midi file
        csv_path:           path of the output .csv
        debug:              if True, additional info is printed
    """

    midi_data = pretty_midi.PrettyMIDI(midi_path)
    score = midi_to_list(midi_data, debug=debug)
    final_df = pd.DataFrame(score, columns=['start', 'end', 'duration', 'pitch',
                                            'velocity', 'instrument', 'instr_program', 'midi_channel'])
    final_df.to_csv(csv_path, index=False)


def create_midi_from_csv_experimental(path_output_file: str,
                                      csv: str,
                                      bpm: int = 120,
                                      debug: bool = False):
    df_csv = pd.read_csv(csv)
    unique_instruments = df_csv['midi_channel'].unique()
    my_midi_file = MIDIFile(len(unique_instruments), adjust_origin=False)
    if debug:
        print(f'Unique instruments: {unique_instruments}')

    previous_track = None
    for i, row in df_csv.iterrows():

        channel = int(row['midi_channel'])
        instr_program = int(row['instr_program'])
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


if __name__ == "__main__":

    filenames = ['MySong']
    debug = False

    for filename in filenames:
        input_midi_path = f'D:/data/input/{filename}.mid'
        output_midi_path = f'D:/data/output/{filename}.mid'
        csv = f'D:/data/csv/{filename}.csv'

        midi_to_csv(midi_path=input_midi_path, csv_path=csv, debug=debug)
        create_midi_from_csv_experimental(path_output_file=output_midi_path, csv=csv, bpm=120, debug=debug)