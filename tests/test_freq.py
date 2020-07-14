"""Test frequency range classes."""
import numpy as np

from edges_cal import FrequencyRange


def test_freq_class():
    """Ensure frequencies are with low/high."""
    freq = FrequencyRange(np.linspace(0, 10, 100), f_low=1, f_high=7)
    assert freq.freq.max() <= 7
    assert freq.freq.min() >= 1
