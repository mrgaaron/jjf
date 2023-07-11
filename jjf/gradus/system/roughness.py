from .base import MusicSystem
from itertools import combinations
from scipy.signal import convolve
from scipy.stats import norm
import numpy as np


def pairwise_roughness(pitch1, pitch2):
    a1, a2 = 1.0, 1.0  # assume equal amplitudes
    f1, f2 = pitch1.frequency.value, pitch2.frequency.value
    fmax = max(f1, f2)
    fmin = min(f1, f2)
    # magic constants, here be dragons
    b1 = 3.5
    b2 = 5.75
    s1 = 0.0207
    s2 = 18.96
    s = 0.24 / (s1 * min(f1, f2) + s2)

    # R = X^0.1*0.5(Y^3.11)*Z --> http://www.acousticslab.org/learnmoresra/moremodel.html
    X = a1 * a2
    Y = (2 * min(a1, a2)) / (a1 + a2)
    Z = np.e ** (-b1 * s * (fmax - fmin)) - np.e ** (-b2 * s * (f1 + f2))
    return X**0.1 * 0.5 * (Y**3.11) * Z


def calculate_roughness_curve(pitches):
    N = len(pitches)
    roughness = []
    for i, p1 in enumerate(pitches):
        for j, p2 in enumerate(pitches):
            roughness.append(pairwise_roughness(p1, p2))
    return roughness


class RoughnessMatrix:
    def __init__(self, system: MusicSystem):
        self.system = system
        self.pitches = list(self.system.pitches)
        self.pitch_map = {p: i for i, p in enumerate(self.pitches)}
        self.inv_pitch_map = {i: p for i, p in enumerate(self.pitches)}

        self._matrix = self.matrix()

    def matrix(self):
        mtx = np.empty((len(self.pitches), len(self.pitches)))
        roughness_curve = calculate_roughness_curve(self.pitches)
        mtx[:] = np.max(roughness_curve)

        for i, _ in enumerate(self.pitches):
            for k, _ in enumerate([self.pitches]):
                if k + i < 0:
                    continue
                try:
                    mtx[k + i, i] = roughness_curve[k]
                except IndexError:
                    continue
        mtx[np.where(mtx < 0.1)] = np.max(mtx)
        return mtx

    def indices_to_pitches(self, pitches):
        return [self.inv_pitch_map[p] for p in pitches]

    def pitches_to_indices(self, pitches):
        return [self.pitch_map[p] for p in pitches]

    def total_pairwise_roughness(self, pitches):
        s = 0.0
        for pair in combinations(pitches, 2):
            col_idx = self.pitch_map[pair[0]]
            row_idx = self.pitch_map[pair[1]]
            s += self._matrix[col_idx, row_idx]
        return s / len(pitches)
