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
    p = np.poly1d([1750, 0, 3, -2, 7, 5][::-1])
    f = np.linspace(-1, 1, len(freq))
    return (freq / 75.0) ** -2.55 * p(f)


def thermal_noise(spec, scale=1, seed=None):
    if seed:
        np.random.seed(seed)
    return np.random.normal(0, spec / scale)


@pytest.fixture("module")
def rfi_regular_1d():
    a = np.zeros(NFREQ)
    a[::50] = 1
    return a


@pytest.fixture("module")
def rfi_random_1d():
    a = np.zeros(NFREQ)
    np.random.seed(12345)
    a[np.random.randint(0, len(a), 100)] = 1
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
    "scale, amplitude",
    list(
        itertools.product(
            (1000,),  # 100, 50),  # Note that realistic noise should be ~250.
            (200,),  # 100, 50), # amplitude of RFI wrt the largest noise.
        )
    ),
)
def test_1d_default(sky_model, rfi_model, scale, amplitude):
    std = sky_model / scale
    amp = std.max() * amplitude
    noise = thermal_noise(sky_model, scale=scale, seed=1010)
    rfi = rfi_model * amp
    sky = sky_model + noise + rfi

    true_flags = rfi_model > 0
    flags, significance = xrfi.xrfi_medfilt(
        sky, max_iter=15, threshold=10, kf=5, poly_order=3
    )

    wrong = np.where(true_flags != flags)[0]

    if len(wrong) > 0:
        print("RFI false positive(0)/negative(1): ")
        print(true_flags[wrong])
        print("Corrupted sky at wrong flags: ")
        print(sky[wrong])
        print("Std. dev away from model at wrong flags: ")
        print((sky[wrong] - sky_model[wrong]) / std[wrong])
        print("Std. dev of noise away from model at wrong flags: ")
        print(noise[wrong] / std[wrong])
        print("Significance at wrong flags: ")
        print(significance[wrong] / std[wrong])

        print("Std dev of RFI away from model at wrong flags: ")
        print(rfi[wrong] / std[wrong])

    assert len(wrong) == 0
