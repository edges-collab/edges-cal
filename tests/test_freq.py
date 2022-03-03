"""Test frequency range classes."""
import pytest

import numpy as np
from astropy import units as u

from edges_cal import FrequencyRange


def test_freq_class():
    """Ensure frequencies are with low/high."""
    freq = FrequencyRange(
        np.linspace(0, 10, 101) * u.MHz, f_low=1 * u.MHz, f_high=7 * u.MHz
    )
    assert freq.freq.max() <= 7 * u.MHz
    assert freq.freq.min() >= 1 * u.MHz
    assert freq.n == len(freq.freq)
    assert np.isclose(freq.df, 0.1 * u.MHz)


def test_edges_freq():
    freq = FrequencyRange.from_edges()
    assert freq.min == 0.0 * u.MHz
    assert freq.max < 200.0 * u.MHz
    assert len(freq.freq) == 32768
    assert np.isclose(freq.df, 200 * u.MHz / 32768.0, atol=1e-7 * u.MHz)


def test_edges_freq_limited():
    freq = FrequencyRange.from_edges(f_low=50.0 * u.MHz, f_high=100.0 * u.MHz)
    assert len(freq.freq) == 8193
    assert freq.min == 50.0 * u.MHz
    assert freq.max == 100.0 * u.MHz


def test_freq_irregular():
    freq = FrequencyRange(np.logspace(1, 2, 25) * u.MHz)
    with pytest.warns(UserWarning):
        assert freq.df == freq.freq[1] - freq.freq[0]


def test_freq_normalize():
    freq = FrequencyRange(np.linspace(0, 10, 101) * u.MHz)
    assert freq.normalize(0 * u.MHz) == -1
    assert freq.normalize(10 * u.MHz) == 1
    assert freq.normalize(5 * u.MHz) == 0

    assert freq.denormalize(-1) == 0 * u.MHz
    assert freq.denormalize(1) == 10 * u.MHz
    assert freq.denormalize(0) == 5 * u.MHz

    f = np.linspace(-2, 12, 50) * u.MHz
    assert np.allclose(freq.denormalize(freq.normalize(f)), f)
