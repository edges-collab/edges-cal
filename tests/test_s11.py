import numpy as np

from edges_cal import reflection_coefficient as rc


def test_gamma_shift_zero():
    s11 = np.random.normal(size=100)
    assert np.all(s11 == rc.gamma_shifted(s11, 0, 0, 0))
