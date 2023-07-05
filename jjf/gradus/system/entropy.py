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


def bendetti_heights(numerators, denominators):
    return numerators * denominators


def tenney_heights(bendetti_heights):
    return np.log2(bendetti_heights)


def harmonic_entropy(ratio_interval, spread=0.01, min_tol=1e-15):
    numerators, denominators = compute_coprimes()
    ratios = numerators / denominators

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


class EntropyMatrix:
    def __init__(self, system):
        self.system = system
        self.interval = self.system.pitch_system.lowest.ratio(
            self.system.pitch_system.highest
        )
        self.pitches = list(self.system.pitches)

    def matrix(self):
        interval_size = 3  # no more than 2 octaves or the calculation loses coherence
        interval_spacing = self.system.pitch_system.temperament.interval_spacing()
        arange = np.arange(0.5, interval_size, interval_spacing)
        below_unison = len(arange[arange < 1])
        above_unison = len(arange[arange > 1])
        mtx = np.zeros((len(self.pitches), len(self.pitches)))

        for i, pitch in enumerate(self.pitches):
            _, entropy_curve = harmonic_entropy(
                np.arange(0.5, interval_size, interval_spacing)
            )
            # don't include pitches aboved or below unison if they aren't in the musical system
            if i < below_unison:
                corrected = entropy_curve[min(below_unison - i, 0) :]
                padded = np.pad(
                    entropy_curve,
                    (i, mtx.shape[0] - corrected.shape[0] - i),
                    "constant",
                    constant_values=0,
                )
            else:
                padded = np.pad(
                    entropy_curve,
                    (i, max(mtx.shape[0] - entropy_curve.shape[0] - i, 0)),
                    "constant",
                    constant_values=0,
                )[: mtx.shape[0]]

            mtx[0:, i] = padded
        return mtx
