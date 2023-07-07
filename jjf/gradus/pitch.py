from abc import ABC, abstractmethod
from astropy import units as u
from collections.abc import Coroutine
from dataclasses import dataclass
from functools import total_ordering
from enum import StrEnum
import numpy as np
from typing import Tuple


class MusicError(Exception):
    pass


@total_ordering
class Pitch:
    @u.quantity_input(frequency=u.Hz)
    def __init__(self, frequency, name=None):
        self.frequency = frequency
        self.name = name

    def ratio(self, other):
        return np.round(other.frequency / self.frequency, 2)

    def __lt__(self, other):
        return self.frequency < other.frequency

    def __eq__(self, other):
        return self.frequency == other.frequency

    def __hash__(self):
        return hash(self.frequency)

    def __repr__(self):
        if self.name is None:
            return f"Pitch({self.frequency})"
        else:
            return f"Pitch({self.frequency}{{{self.name}}})"


class Temperament(ABC):
    @abstractmethod
    def pitches(
        self, lowest: Pitch, highest: Pitch
    ) -> Coroutine[Pitch, Pitch, list[Pitch]]:
        ...

    @abstractmethod
    def interval_spacing(self) -> int:
        ...


class EqualTemperament(Temperament):
    def __init__(self, beta: float, notes_per_octave: int):
        self.beta = beta
        self.notes_per_octave = notes_per_octave

    def pitches(
        self, lowest: Pitch, highest: Pitch
    ) -> Coroutine[Pitch, Pitch, list[Pitch]]:
        current = lowest
        while current <= highest:
            yield current
            current = Pitch(current.frequency * self.beta)
        yield current  # make sure we yield the final value

    def interval_spacing(self, octaves=2) -> Tuple[int, float]:
        return [
            (i, np.power(np.power(2, 1 / 12), i))
            for i in range(-self.notes_per_octave, 1 + self.notes_per_octave * octaves)
        ]


class UnequalTemperament(Temperament):
    def __init__(self, ratios: list[float]):
        self.ratios = ratios

    def pitches(self, lowest: Pitch, highest: Pitch) -> list[Pitch]:
        ...


@dataclass
class Dyad:
    pitch1: Pitch
    pitch2: Pitch

    @property
    def pitches(self):
        return [self.pitch1, self.pitch2]


@dataclass
class Triad:
    pitch1: Pitch
    pitch2: Pitch
    pitch3: Pitch

    @property
    def pitches(self):
        return [self.pitch1, self.pitch2, self.pitch3]


@dataclass
class Tetrad:
    pitch1: Pitch
    pitch2: Pitch
    pitch3: Pitch
    pitch4: Pitch

    @property
    def pitches(self):
        return [self.pitch1, self.pitch2, self.pitch3, self.pitch4]


@dataclass
class Pentad:
    pitch1: Pitch
    pitch2: Pitch
    pitch3: Pitch
    pitch4: Pitch
    pitch5: Pitch

    @property
    def pitches(self):
        return [self.pitch1, self.pitch2, self.pitch3, self.pitch4, self.pitch5]


@dataclass
class Hexad:
    pitch1: Pitch
    pitch2: Pitch
    pitch3: Pitch
    pitch4: Pitch
    pitch5: Pitch
    pitch6: Pitch

    @property
    def pitches(self):
        return [
            self.pitch1,
            self.pitch2,
            self.pitch3,
            self.pitch4,
            self.pitch5,
            self.pitch6,
        ]


@dataclass
class ToneRow:
    pitches: list[Pitch]


class PitchSystem:
    def __init__(self, lowest: Pitch, highest: Pitch, temperament: Temperament):
        self.lowest = lowest
        self.highest = highest
        self.temperament = temperament

    @property
    def pitches(self) -> Coroutine[Pitch, Pitch, list[Pitch]]:
        return self.temperament.pitches(self.lowest, self.highest)
