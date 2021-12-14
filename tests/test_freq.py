"""Test frequency range classes."""
import pytest

import numpy as np

from edges_cal import FrequencyRange


def test_freq_class():
    """Ensure frequencies are with low/high."""
    freq = FrequencyRange(np.linspace(0, 10, 101), f_low=1, f_high=7)
    assert freq.freq.max() <= 7
    assert freq.freq.min() >= 1
    assert freq.n == len(freq.freq)
    assert np.isclose(freq.df, 0.1)


def test_edges_freq():
    freq = FrequencyRange.from_edges()
    assert freq.min == 0.0
    assert freq.max < 200.0
    assert len(freq.freq) == 32768
    assert np.isclose(freq.df, 200 / 32768.0, atol=1e-7)


def test_edges_freq_limited():
    freq = FrequencyRange.from_edges(f_low=50.0, f_high=100.0)
    assert len(freq.freq) == 8193
    assert freq.min == 50.0
    assert freq.max == 100.0


def test_freq_irregular():
    freq = FrequencyRange(np.logspace(1, 2, 25))
    with pytest.warns(UserWarning):
        assert freq.df == freq.freq[1] - freq.freq[0]


def test_freq_normalize():
    freq = FrequencyRange(np.linspace(0, 10, 101))
    assert freq.normalize(0) == -1
    assert freq.normalize(10) == 1
    assert freq.normalize(5) == 0

    assert freq.denormalize(-1) == 0
    assert freq.denormalize(1) == 10
    assert freq.denormalize(0) == 5

    f = np.linspace(-2, 12, 50)
    assert np.allclose(freq.denormalize(freq.normalize(f)), f)
