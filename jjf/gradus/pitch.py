class Pitch:
    def __init__(self, frequency):
        self.frequency = frequency

    def ratio(self, other):
        return self.frequency / other.frequency


A = Pitch(440)
