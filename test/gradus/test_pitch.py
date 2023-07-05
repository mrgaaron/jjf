from astropy import units as u
from astropy.units.quantity import isclose
from jjf.gradus.pitch import Pitch, EqualTemperament
import numpy as np
import pytest


def test_equal_temperament_pitches():
    # octave from A440 to A880
    lowest = Pitch(440 * u.Hz)
    highest = Pitch(880 * u.Hz)

    # equal temperament with beta = 2^(1/12), i.e. the standard 12-tone scale or 12-TET
    beta = np.power(2, 1 / 12)
    temperament = EqualTemperament(beta, 12)
    all_pitches = list(temperament.pitches(lowest, highest))
    assert len(all_pitches) == 13
    assert isclose(all_pitches[0].frequency, 440 * u.Hz)
    assert isclose(all_pitches[-1].frequency, 880 * u.Hz)
    for i in range(1, len(all_pitches)):
        assert all_pitches[i].frequency / all_pitches[i - 1].frequency == beta


def test_pitch_ratios():
    # octave from A440 to A880
    pitch1 = Pitch(440 * u.Hz)
    pitch2 = Pitch(659 * u.Hz)
    pitch3 = Pitch(880 * u.Hz)

    assert pitch1.ratio(pitch2) == 3 / 2
    assert pitch1.ratio(pitch3) == 2 / 1
    assert isclose(pitch2.ratio(pitch3), 4 / 3, rtol=0.01)
