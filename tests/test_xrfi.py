import itertools

import pytest

import numpy as np
from pytest_cases import fixture_ref as fxref
from pytest_cases import parametrize_plus

from edges_cal import xrfi

NFREQ = 1000


@pytest.fixture("module")
def freq():
    return np.linspace(50, 150, NFREQ)


@pytest.fixture("module")
def sky_pl_1d(freq):
    return 1750 * (freq / 75.0) ** -2.55


@pytest.fixture("module")
def sky_flat_1d():
    return np.ones(NFREQ)


@pytest.fixture("module")
def sky_linpoly_1d(freq):
    p = np.poly1d([1000, -400, 4000, -8000, 7000, -2000, 400][:-1])
    return (freq / 75.0) ** -2.55 * p(freq / 75.0)


def add_noise(spec, scale=1, seed=None):
    if seed:
        np.random.seed(seed)
    return spec + np.random.normal(0, spec / scale)


@pytest.fixture("module")
def rfi_regular_1d():
    a = np.zeros(NFREQ)
    a[::50] = 10000
    return a


@pytest.fixture("module")
def rfi_random_1d():
    a = np.zeros(NFREQ)
    a[np.random.randint(0, len(a), 100)] = 10000
    return a


@pytest.fixture("module")
def rfi_null_1d():
    return np.zeros(NFREQ)


@parametrize_plus(
    "sky_model", [fxref(sky_flat_1d), fxref(sky_pl_1d), fxref(sky_linpoly_1d)]
)
@parametrize_plus(
    "rfi_model", [fxref(rfi_null_1d), fxref(rfi_regular_1d), fxref(rfi_random_1d)]
)
@pytest.mark.parametrize(
    "rfi_func, scale",
    list(
        itertools.product(
            (xrfi.xrfi_medfilt, xrfi.xrfi_poly, xrfi.xrfi_poly_filter),
            (1000, 100, 10),  # Note that realistic noise should be ~250.
        )
    ),
)
def test_1d_default(sky_model, rfi_model, rfi_func, scale):
    sky = add_noise(sky_model, scale=scale, seed=1010) + rfi_model

    flags = rfi_func(sky)
    assert not np.any(flags ^ (rfi_model > 0))
