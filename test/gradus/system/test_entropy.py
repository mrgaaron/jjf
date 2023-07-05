from jjf.gradus.system.entropy import EntropyMatrix
from jjf.gradus.system.western import WesternSystem


def test_entropy_matrix_setup():
    system = WesternSystem()
    entropy = EntropyMatrix(system)
    mtx = entropy.matrix()
    assert True
