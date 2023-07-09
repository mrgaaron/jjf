from astropy import units as u
from .base import MusicSystem
from itertools import cycle
from ..pitch import Pitch, PitchSystem, PitchCollection, EqualTemperament
import numpy as np


LOWEST = Pitch(8.1758 * u.Hz)
HIGHEST = Pitch(12543.85 * u.Hz)
NOTE_NAMES = [
    "C",
    "DFlat",
    "D",
    "EFlat",
    "E",
    "F",
    "GFlat",
    "G",
    "AFlat",
    "A",
    "B",
    "BFlat",
]
BETA = np.power(2, 1 / len(NOTE_NAMES))


class WesternSystem(MusicSystem):
    def __init__(self) -> None:
        self.pitch_system = PitchSystem(
            LOWEST, HIGHEST, EqualTemperament(BETA, len(NOTE_NAMES))
        )
        self.pitches = self._pitches()

    def _pitches(self) -> PitchCollection:
        note_cycle = cycle(NOTE_NAMES)
        octave = -1
        pitches = []
        for i, pitch in enumerate(self.pitch_system.pitches):
            if i > 0 and i % 12 == 0:
                octave += 1
            # it is only when we have a music system identified that we can assign note names
            pitch.name = f"{next(note_cycle)}{octave}"
            pitches.append(pitch)
        return PitchCollection(pitches)
