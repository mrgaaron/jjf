from jjf.gradus.pitch import Pitch
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
