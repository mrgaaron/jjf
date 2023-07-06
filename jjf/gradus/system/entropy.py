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


def shift(arr, num, fill_value=np.nan):
    arr = np.roll(arr, num)
    if num < 0:
        arr[num:] = fill_value
    elif num > 0:
        arr[:num] = fill_value
    return arr


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
        mtx = np.zeros((len(self.pitches), len(self.pitches)))
        _, entropy_curve = harmonic_entropy(
            np.arange(0.5, interval_size, interval_spacing)
        )

        for i, pitch in enumerate(self.pitches):
            # don't include pitches aboved or below unison if they aren't in the musical system
            if i < below_unison:
                padded = np.pad(
                    entropy_curve,
                    (0, mtx.shape[0] - entropy_curve.shape[0]),
                    "constant",
                    constant_values=0,
                )
                corrected = shift(padded, -(below_unison - i), fill_value=0.0)
            else:
                # TODO: this is bugged for notes over A7, not handling the upper bound correctly
                if mtx.shape[0] - entropy_curve.shape[0] - i + below_unison < 0:
                    padded = np.pad(
                        entropy_curve,
                        (mtx.shape[0] - below_unison, 0),
                        "constant",
                        constant_values=0,
                    )
                    corrected = shift(
                        padded,
                        abs(mtx.shape[0] - entropy_curve.shape[0] - i - below_unison),
                        fill_value=0.0,
                    )[-128:]
                else:
                    corrected = np.pad(
                        entropy_curve,
                        (
                            i - below_unison,
                            mtx.shape[0] - entropy_curve.shape[0] - i + below_unison,
                        ),
                        "constant",
                        constant_values=0,
                    )
            mtx[0:, i] = corrected
        return mtx
