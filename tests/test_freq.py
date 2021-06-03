"""Test frequency range classes."""
import numpy as np

from edges_cal import EdgesFrequencyRange, FrequencyRange


def test_freq_class():
    """Ensure frequencies are with low/high."""
    freq = FrequencyRange(np.linspace(0, 10, 100), f_low=1, f_high=7)
    assert freq.freq.max() <= 7
    assert freq.freq.min() >= 1


def test_edges_freq():
    freq = EdgesFrequencyRange()
    assert freq.min == 0.0
    assert freq.max < 200.0
    assert len(freq.freq) == 32768
    assert np.isclose(freq.df, 200 / 32768.0, atol=1e-7)


def test_edges_freq_limited():
    freq = EdgesFrequencyRange(f_low=50.0, f_high=100.0)
    assert len(freq.freq) == 8193
    assert freq.min == 50.0
    assert freq.max == 100.0
