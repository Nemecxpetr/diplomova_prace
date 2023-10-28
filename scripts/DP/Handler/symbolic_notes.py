"""

"""
import numpy as np

class Note:
    """Representation of a musical note with a single pitch
    """
    
    _names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    def __init__(self, note_string, ref_A4=440.):
        """Initialize a new note

        Arguments
        =========
            note_string: str
                name of the tone in format "X[#/b]N" where X is a letter from A to G and N is the octave number (e.g. A3 or C#5) 
            ref_A4: float
                reference frequency for A4 in Hz (default: 440 Hz)
        """
        self.note_string = note_string
        self.midi_pitch = self._string2midi(note_string)
        self.ref_A4 = ref_A4

    def _string2midi(self, note_string):
        """Helper function to convert a note string into a MIDI pitch (A4 = 69)

        Arguments
        =========
            note_string: str
                name of the tone in format "X[#/b]N" where X is a letter from A to G and N is the octave number (e.g. A3 or C#5) 

        Returns
        =======
            midi_pitch: float
        """ 
        base_tone = note_string[0]  # First character of the note string
        octave = int(note_string[-1])  # Last character of the note string

        pitch_adjustment = note_string.count('#') - note_string.count('b')

        p = int(self._names.index(base_tone)) + pitch_adjustment    

        midi_pitch = p + (octave+1)*12

        return midi_pitch
    
    def increment_pitch(self, increment):
        self.midi_pitch += increment        
          
    def get_freq(self):
        """Returns the frequency of the tone in Hz
        """
        freq_center = 2 ** ((self.midi_pitch - 69) / 12) * self.ref_A4
        return freq_center
    
    def get_midi(self):
        """Returns the MIDI pitch of the tone (A4 = 69)
        """
        return self.midi_pitch
    
    def get_string(self):
        """Returns the note string representation 
           in format "X[#/b]N" where X is a letter from A to G and N is the octave number (e.g. A3 or C#5)
        """
        string = self._names[(int(self.midi_pitch) - 60) % 12] + str(int(self.midi_pitch)//(12)-1)
        
        return string
    
    def __str__(self):
        return self.get_string()  

class NoteDetunable(Note):
    
    def __init__(self, note_string, ref_A4=440.):
        super().__init__(note_string, ref_A4)
    
    def detune(self, delta_cents):
        """Detune the note by a certain amount
        
        Arguments
        =========
            delta_cents: float
                desired detuning of the tone in cents (i.e., 1/100 of a semitone)
        """
        d = delta_cents/100
        self.increment_pitch(d)


