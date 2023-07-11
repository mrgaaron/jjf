from astropy.units.quantity import isclose
from jjf.gradus.system.roughness import RoughnessMatrix
from jjf.gradus.system.western import WesternSystem


def test_roughness_matrix_setup():
    system = WesternSystem()
    roughness = RoughnessMatrix(system)
    mtx = roughness.matrix()
    assert True


def test_roughness_dyads():
    system = WesternSystem()
    roughness = RoughnessMatrix(system)
    p1 = system.pitches["C4"]
    p2 = system.pitches["G4"]
    total_roughness = roughness.total_pairwise_roughness([p1, p2])
    assert isclose(total_roughness, 0.25, rtol=0.01)


def test_roughness_triads():
    system = WesternSystem()
    roughness = RoughnessMatrix(system)
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
    major_roughness = roughness.total_pairwise_roughness(major)
    minor_roughness = roughness.total_pairwise_roughness(minor)
    six_roughness = roughness.total_pairwise_roughness(six_chord)
    wide_roughness = roughness.total_pairwise_roughness(wide_chord)
    random_roughness = roughness.total_pairwise_roughness(random_chord)
    print(roughness._matrix[:, 0])

    assert major_roughness == minor_roughness
    assert major_roughness < six_roughness
    assert major_roughness < wide_roughness
    assert major_roughness < random_roughness
