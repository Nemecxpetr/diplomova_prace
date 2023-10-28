# Package MIDI
""" Package with functions to easy handling midi in the synchronization process
Also offers some classes to create note objects for playing - I'm just playin' and learning to programe at this point.
"""
# MIDI handler functions to create midi and convert it to csv and back
from .MIDI_handler import load_midi, \
    midi_to_list, \
    list_to_csv, \
    read_csv, \
    csv_to_list, \
    csv_to_midi, \
    test

# symbolic note objects for calculating freq pitch of notes by names etc. ...
from symbolic_notes import Note, \
    NoteDetunable