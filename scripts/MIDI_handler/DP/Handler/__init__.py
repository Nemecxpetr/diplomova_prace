# Package Handler
""" 
Handler package

Author: Petr Nìmec
License: 

Containing functions for easier handling MIDI and audio files and conversing them to chroma vectors and other visual representations.

Also offers some classes to create note objects for playing - I'm just playin' and learning to programe at this point.

Some functions were taken from or inspired by the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)

This package is part of my master's thesis (https://github.com/Nemecxpetr/diplomova_prace)
"""
# MIDI handler functions to create midi and convert it to csv and back
from .MIDI_handler import load_midi, \
    midi_to_list, \
    list_to_csv, \
    read_csv, \
    csv_to_list, \
    csv_to_midi, \
    test

from .audio_handler import *

# symbolic note objects for calculating freq pitch of notes by names etc. ...
from .symbolic_notes import Note, \
    NoteDetunable