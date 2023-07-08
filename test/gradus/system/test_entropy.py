from astropy.units.quantity import isclose
from jjf.gradus.pitch import Pitch
from jjf.gradus.system.entropy import EntropyMatrix
from jjf.gradus.system.western import WesternSystem


def test_entropy_matrix_setup():
    system = WesternSystem()
    entropy = EntropyMatrix(system)
    mtx = entropy.matrix()
    assert True


def test_entropy_dyads():
    system = WesternSystem()
    entropy = EntropyMatrix(system)
    p1 = system.pitches["C4"]
    p2 = system.pitches["G4"]
    total_entropy = entropy.total_pairwise_entropy([p1, p2])
    assert isclose(total_entropy, 1.23, rtol=0.01)


def test_entropy_triads():
    system = WesternSystem()
    entropy = EntropyMatrix(system)
    major = [
        system.pitches["C4"],
        system.pitches["E4"],
        system.pitches["G4"],
    ]
    minor = [
        system.pitches["C4"],
        system.pitches["EFlat4"],
        system.pitches["G4"],
    ]
    six_chord = [
        system.pitches["C4"],
        system.pitches["E4"],
        system.pitches["G4"],
        system.pitches["A4"],
    ]
    wide_chord = [
        system.pitches["E3"],
        system.pitches["C4"],
        system.pitches["G5"],
    ]
    random_chord = [
        system.pitches["C4"],
        system.pitches["GFlat4"],
        system.pitches["AFlat4"],
    ]
    major_entropy = entropy.total_pairwise_entropy(major)
    minor_entropy = entropy.total_pairwise_entropy(minor)
    six_entropy = entropy.total_pairwise_entropy(six_chord)
    wide_entropy = entropy.total_pairwise_entropy(wide_chord)
    random_entropy = entropy.total_pairwise_entropy(random_chord)

    assert major_entropy == minor_entropy
    assert major_entropy < six_entropy
    assert major_entropy < wide_entropy
    assert major_entropy < random_entropy
