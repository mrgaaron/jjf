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
    p1 = system.pitch_by_name("C4")
    p2 = system.pitch_by_name("G4")
    total_entropy = entropy.total_pairwise_entropy([p1, p2])
    assert isclose(total_entropy, 2.47, rtol=0.01)


def test_entropy_triads():
    system = WesternSystem()
    entropy = EntropyMatrix(system)
    major = [
        system.pitch_by_name("C4"),
        system.pitch_by_name("E4"),
        system.pitch_by_name("G4"),
    ]
    minor = [
        system.pitch_by_name("C4"),
        system.pitch_by_name("EFlat4"),
        system.pitch_by_name("G4"),
    ]
    sixth_chord = [
        system.pitch_by_name("C4"),
        system.pitch_by_name("E4"),
        system.pitch_by_name("A4"),
    ]
    wide_chord = [
        system.pitch_by_name("E3"),
        system.pitch_by_name("C4"),
        system.pitch_by_name("G5"),
    ]
    random_chord = [
        system.pitch_by_name("C4"),
        system.pitch_by_name("GFlat4"),
        system.pitch_by_name("AFlat4"),
    ]
    major_entropy = entropy.total_pairwise_entropy(major)
    minor_entropy = entropy.total_pairwise_entropy(minor)
    sixth_entropy = entropy.total_pairwise_entropy(sixth_chord)
    wide_entropy = entropy.total_pairwise_entropy(wide_chord)
    random_entropy = entropy.total_pairwise_entropy(random_chord)

    assert major_entropy == minor_entropy
    assert major_entropy < sixth_entropy
    assert major_entropy < wide_entropy
    assert major_entropy < random_entropy
