import pytest

import numpy as np
from astropy import units as u

from edges_cal import tools


def test_dct_to_list():
    """Ensure simple dictionary is dealt with correctly."""
    dct_of_lists = {"a": [1, 2], "b": [3, 4]}

    list_of_dicts = tools.dct_of_list_to_list_of_dct(dct_of_lists)

    assert list_of_dicts == [
        {"a": 1, "b": 3},
        {"a": 1, "b": 4},
        {"a": 2, "b": 3},
        {"a": 2, "b": 4},
    ]


def test_bin_array_simple_1d():
    x = np.array([1, 1, 2, 2, 3, 3])
    out = tools.bin_array(x, 2)
    assert np.all(out == np.array([1, 2, 3]))


def test_bin_array_remainder():
    x = np.array([1, 1, 2, 2, 3, 3, 4])
    out = tools.bin_array(x, 2)
    assert np.all(out == np.array([1, 2, 3]))


def test_bin_array_2d():
    x = np.array(
        [
            [1, 1, 2, 2, 3, 3, 4],
            [4, 4, 5, 5, 6, 6, 7],
        ]
    )
    out = tools.bin_array(x, 2)
    assert np.all(out == np.array([[1, 2, 3], [4, 5, 6]]))


def test_bad_bin_size():
    with pytest.raises(ValueError, match="Cannot use bin_size < 1"):
        tools.FrequencyRange(np.linspace(0, 1, 100) * u.MHz, bin_size=0)
