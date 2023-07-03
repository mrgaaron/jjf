from jjf.gradus.pitch import Pitch
from scipy.signal import convolve


class HarmonicHartleyEntropy:
    def __init__(self, pitches: list[Pitch]):
        self.pitches = pitches
