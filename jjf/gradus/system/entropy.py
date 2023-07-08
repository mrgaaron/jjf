from .base import MusicSystem
from itertools import combinations
from scipy.signal import convolve
from scipy.stats import norm
import numpy as np


def compute_coprimes():
    numerators = []
    denominators = []
    # N = 420
    N = 100
    for den in range(1, N):
        for num in range(1, 3 * den):
            if np.gcd(num, den) == 1:
                numerators.append(num)
                denominators.append(den)
    numerators = np.array(numerators)
    denominators = np.array(denominators)
    return numerators, denominators


def ensure_delta(ratios):
    M = len(ratios)
    delta = 0.00001
    indices = np.ones(M, dtype=bool)

    for i in range(M - 2):
        ind = abs(ratios[i + 1 :] - ratios[i]) > delta
        indices[i + 1 :] = indices[i + 1 :] * ind

    return ratios[indices]


def bendetti_heights(numerators, denominators):
    return numerators * denominators


def tenney_heights(bendetti_heights):
    return np.log2(bendetti_heights)


def harmonic_entropy(ratio_interval, spread=0.01, min_tol=1e-15):
    numerators, denominators = compute_coprimes()
    ratios = numerators / denominators
    ratios = ensure_delta(ratios)

    ind = np.argsort(ratios)
    weight_ratios = ratios[ind]

    centers = (weight_ratios[:-1] + weight_ratios[1:]) / 2

    ratio_interval = np.array(ratio_interval)
    N = len(ratio_interval)
    HE = np.zeros(N)
    for i, x in enumerate(ratio_interval):
        P = np.diff(
            np.concatenate(
                ([0], norm.cdf(np.log2(centers), loc=np.log2(x), scale=spread), [1])
            )
        )
        ind = P > min_tol
        HE[i] = -np.sum(P[ind] * np.log2(P[ind]))

    return weight_ratios, HE


def shift(arr, num, fill_value=np.nan):
    arr = np.roll(arr, num)
    if num < 0:
        arr[num:] = fill_value
    elif num > 0:
        arr[:num] = fill_value
    return arr


class EntropyMatrix:
    def __init__(self, system: MusicSystem):
        self.system = system
        self.pitches = list(self.system.pitches)
        self.pitch_map = {p: i for i, p in enumerate(self.pitches)}

        self._matrix = self.matrix()

    def matrix(self):
        interval_spacing = self.system.pitch_system.temperament.interval_spacing(
            octaves=2
        )
        mtx = np.empty((len(self.pitches), len(self.pitches)))
        _, entropy_curve = harmonic_entropy(
            [i[1] for i in interval_spacing], spread=0.005
        )
        mtx[:] = np.max(entropy_curve)

        for i, _ in enumerate(self.pitches):
            for k, j in enumerate([idx[0] for idx in interval_spacing]):
                if j + i < 0:
                    continue
                try:
                    mtx[j + i, i] = entropy_curve[k]
                except IndexError:
                    continue
        mtx[np.where(mtx < 0.1)] = np.max(mtx)
        return mtx

    def total_pairwise_entropy(self, pitches):
        s = 0.0
        for pair in combinations(pitches, 2):
            col_idx = self.pitch_map[pair[0]]
            row_idx = self.pitch_map[pair[1]]
            s += self._matrix[col_idx, row_idx]
        return s / len(pitches)
