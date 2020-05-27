import itertools

import pytest

import numpy as np
from pytest_cases import fixture_ref as fxref
from pytest_cases import parametrize_plus

from edges_cal import xrfi

NFREQ = 1000


@pytest.fixture(scope="module")
def freq():
    return np.linspace(50, 150, NFREQ)


@pytest.fixture(scope="module")
def sky_pl_1d(freq):
    return 1750 * (freq / 75.0) ** -2.55


@pytest.fixture(scope="module")
def sky_flat_1d():
    return np.ones(NFREQ)


@pytest.fixture(scope="module")
def sky_linpoly_1d(freq):
    p = np.poly1d([1750, 0, 3, -2, 7, 5][::-1])
    f = np.linspace(-1, 1, len(freq))
    return (freq / 75.0) ** -2.55 * p(f)


def thermal_noise(spec, scale=1, seed=None):
    if seed:
        np.random.seed(seed)
    return np.random.normal(0, spec / scale)


@pytest.fixture(scope="module")
def rfi_regular_1d():
    a = np.zeros(NFREQ)
    a[50::50] = 1
    return a


@pytest.fixture(scope="module")
def rfi_random_1d():
    a = np.zeros(NFREQ)
    np.random.seed(12345)
    a[np.random.randint(0, len(a), 40)] = 1
    return a


@pytest.fixture(scope="module")
def rfi_null_1d():
    return np.zeros(NFREQ)


def test_flagged_filter(sky_pl_1d, rfi_regular_1d):
    flags = rfi_regular_1d.astype("bool")
    in_data = sky_pl_1d.copy()
    detrended = xrfi.flagged_filter(in_data, size=5, flags=flags, interp_flagged=False)
    assert not np.any(np.isnan(detrended))
    assert np.all(in_data == sky_pl_1d)

    # Anything close to a flag will not be identical, as the
    # median of an even number of items is the average of the middle two (and with a flag
    # the total number of items is reduced by one).
    assert np.all(detrended[flags] == sky_pl_1d[flags])

    padded_flags = np.zeros_like(flags)
    for index in np.where(flags)[0]:
        padded_flags[index - 2 : index + 3] = True
        padded_flags[index] = False

    # Ensure everything away from flags is exactly the same.
    assert np.all(detrended[~padded_flags] == sky_pl_1d[~padded_flags])

    # An unflagged filter should be an identity operation.
    unflagged = xrfi.flagged_filter(in_data, size=5)
    assert np.all(unflagged == sky_pl_1d)

    # But not quite, when mode = 'reflect':
    unflagged = xrfi.flagged_filter(in_data, size=5, mode="reflect")
    assert not np.all(unflagged[:2] == sky_pl_1d[:2])

    # An unflagged filter with RFI should be very close to the original
    sky = sky_pl_1d + 100000 * rfi_regular_1d
    detrended = xrfi.flagged_filter(sky, size=5)
    assert np.allclose(detrended, sky_pl_1d, rtol=1e-1)


@parametrize_plus(
    "sky_model", [fxref(sky_flat_1d), fxref(sky_pl_1d), fxref(sky_linpoly_1d)]
)
@parametrize_plus(
    "rfi_model", [fxref(rfi_null_1d), fxref(rfi_regular_1d), fxref(rfi_random_1d)]
)
@pytest.mark.parametrize(
    "scale",
    list(itertools.product((1000, 100))),  # Note that realistic noise should be ~250.
)
def test_1d_medfilt(sky_model, rfi_model, scale):
    std = sky_model / scale
    amp = std.max() * 200
    noise = thermal_noise(sky_model, scale=scale, seed=1010)
    rfi = rfi_model * amp
    sky = sky_model + noise + rfi

    true_flags = rfi_model > 0
    flags, significance = xrfi.xrfi_medfilt(
        sky, max_iter=1, threshold=10, kf=5, use_meanfilt=True
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


@parametrize_plus(
    "sky_model", [fxref(sky_flat_1d), fxref(sky_pl_1d), fxref(sky_linpoly_1d)]
)
@parametrize_plus(
    "rfi_model", [fxref(rfi_null_1d), fxref(rfi_regular_1d), fxref(rfi_random_1d)]
)
@pytest.mark.parametrize("scale", [1000, 100])
@pytest.mark.skip("Not working yet...")
def test_poly(sky_model, rfi_model, scale):
    std = sky_model / scale
    amp = std.max() * 200
    noise = thermal_noise(sky_model, scale=scale, seed=1010)
    rfi = rfi_model * amp
    sky = sky_model + noise + rfi

    true_flags = rfi_model > 0
    flags = xrfi.xrfi_model(sky)

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
        print("Std dev of RFI away from model at wrong flags: ")
        print(rfi[wrong] / std[wrong])

    assert len(wrong) == 0
