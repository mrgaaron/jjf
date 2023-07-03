from abc import ABC, abstractmethod
from astropy import units as u
from collections.abc import Coroutine
from functools import total_ordering
from enum import StrEnum
import numpy as np


@total_ordering
class Pitch:
    @u.quantity_input(frequency=u.Hz)
    def __init__(self, frequency, name=None):
        self.frequency = frequency
        self.name = name

    def ratio(self, other):
        return 1200 * np.log2(other.frequency / self.frequency)

    def __lt__(self, other):
        return self.frequency < other.frequency

    def __eq__(self, other):
        return self.frequency == other.frequency

    def __repr__(self):
        if self.name is None:
            return f"Pitch({self.frequency})"
        else:
            return f"Pitch({self.frequency}{{{self.name}}})"


class DistributionType(StrEnum):
    GEOMETRIC = "geometric"
    UNEQUAL = "unequal"


class Temperament(ABC):
    @abstractmethod
    def pitches(
        self, lowest: Pitch, highest: Pitch
    ) -> Coroutine[Pitch, Pitch, list[Pitch]]:
        ...


class EqualTemperament(Temperament):
    def __init__(self, beta: float):
        self.beta = beta

    def pitches(
        self, lowest: Pitch, highest: Pitch
    ) -> Coroutine[Pitch, Pitch, list[Pitch]]:
        current = lowest
        while current <= highest:
            yield current
            current = Pitch(current.frequency * self.beta)
        yield current  # make sure we yield the final value


class UnequalTemperament(Temperament):
    def __init__(self, ratios: list[float]):
        self.ratios = ratios

    def pitches(self, lowest: Pitch, highest: Pitch) -> list[Pitch]:
        ...


class MusicError(Exception):
    pass


class PitchSystem:
    def __init__(self, lowest: Pitch, highest: Pitch, temperament: Temperament):
        self.lowest = lowest
        self.highest = highest
        self.temperament = temperament

    @property
    def pitches(self) -> Coroutine[Pitch, Pitch, list[Pitch]]:
        return self.temperament.pitches(self.lowest, self.highest)
