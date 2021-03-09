"""Tests of the xrfi module."""
import pytest

import itertools
import numpy as np
from pytest_cases import fixture_ref as fxref
from pytest_cases import parametrize_plus

from edges_cal import xrfi

NFREQ = 1000


@pytest.fixture(scope="module")
def freq():
    """Default frequencies."""
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
def rfi_regular_leaky():
    """RFI that leaks into neighbouring bins"""
    a = np.zeros(NFREQ)
    a[50:-30:50] = 1
    a[49:-30:50] = (
        1.0 / 1000
    )  # needs to be smaller than 200 or else it will be flagged outright.
    a[51:-30:50] = 1.0 / 1000
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
    sky, std, noise, rfi = make_sky(sky_model, rfi_model, scale)

    true_flags = rfi_model > 0
    flags, significance = xrfi.xrfi_medfilt(
        sky, max_iter=1, threshold=10, kf=5, use_meanfilt=True
    )

    wrong = np.where(true_flags != flags)[0]

    print_wrongness(wrong, std, {}, noise, true_flags, sky, rfi)

    assert len(wrong) == 0


@parametrize_plus(
    "sky_model", [fxref(sky_flat_1d), fxref(sky_pl_1d), fxref(sky_linpoly_1d)]
)
@parametrize_plus(
    "rfi_model", [fxref(rfi_null_1d), fxref(rfi_regular_1d), fxref(rfi_random_1d)]
)
@pytest.mark.parametrize("scale", [1000, 100])
def test_xrfi_model(sky_model, rfi_model, scale, freq):
    sky, std, noise, rfi = make_sky(sky_model, rfi_model, scale)

    true_flags = rfi_model > 0
    flags, info = xrfi.xrfi_model(sky, freq=freq)

    wrong = np.where(true_flags != flags)[0]

    print_wrongness(wrong, std, info, noise, true_flags, sky, rfi)

    assert len(wrong) == 0


@parametrize_plus(
    "sky_model", [fxref(sky_flat_1d), fxref(sky_pl_1d), fxref(sky_linpoly_1d)]
)
@parametrize_plus("rfi_model", [fxref(rfi_regular_leaky)])
@pytest.mark.parametrize("scale", [1000, 100])
def test_poly_watershed_strict(sky_model, rfi_model, scale, freq):
    sky, std, noise, rfi = make_sky(sky_model, rfi_model, scale, rfi_amp=200)

    true_flags = rfi_model > 0
    flags, info = xrfi.xrfi_model(
        sky, freq=freq, watershed=1, threshold=5, min_threshold=4, max_iter=10
    )

    wrong = np.where(true_flags != flags)[0]

    print_wrongness(wrong, std, info, noise, true_flags, sky, rfi)

    assert len(wrong) == 0


@parametrize_plus(
    "sky_model", [fxref(sky_flat_1d), fxref(sky_pl_1d), fxref(sky_linpoly_1d)]
)
@parametrize_plus("rfi_model", [fxref(rfi_regular_leaky)])
@pytest.mark.parametrize("scale", [1000, 100])
def test_poly_watershed_relaxed(sky_model, rfi_model, scale, freq):
    sky, std, noise, rfi = make_sky(sky_model, rfi_model, scale, rfi_amp=500)

    true_flags = rfi_model > 0
    flags, info = xrfi.xrfi_model(sky, freq=freq, watershed=1, threshold=6)

    # here we just assert no *missed* RFI
    wrong = np.where(true_flags & ~flags)[0]

    print_wrongness(wrong, std, info, noise, true_flags, sky, rfi)

    assert len(wrong) == 0


def test_watershed():
    rfi = np.zeros((10, 10), dtype=bool)
    out, _ = xrfi.xrfi_watershed(flags=rfi)
    assert not np.any(out)

    rfi = np.ones((10, 10), dtype=bool)
    out, _ = xrfi.xrfi_watershed(flags=rfi)
    assert np.all(out)

    rfi = np.repeat([0, 1], 48).reshape((3, 32))
    out, _ = xrfi.xrfi_watershed(flags=rfi, tol=0.2)
    assert np.all(out)


@parametrize_plus(
    "sky_model", [fxref(sky_flat_1d), fxref(sky_pl_1d), fxref(sky_linpoly_1d)]
)
@parametrize_plus(
    "rfi_model", [fxref(rfi_null_1d), fxref(rfi_regular_1d), fxref(rfi_random_1d)]
)
@pytest.mark.parametrize("scale", [1000, 100])
def test_xrfi_model_sweep(sky_model, rfi_model, scale):
    sky, std, noise, rfi = make_sky(sky_model, rfi_model, scale)

    true_flags = rfi_model > 0
    flags, info = xrfi.xrfi_model_sweep(
        sky, max_iter=10, threshold=5, use_median=True, which_bin="last",
    )

    # Only consider flags after bin 100 (since that's the bin width)
    wrong = np.where(true_flags[100:] != flags[100:])[0]

    print_wrongness(wrong, std, info, noise, true_flags, sky, rfi)
    assert len(wrong) == 0


@parametrize_plus(
    "sky_model", [fxref(sky_flat_1d), fxref(sky_pl_1d), fxref(sky_linpoly_1d)]
)
@parametrize_plus(
    "rfi_model", [fxref(rfi_null_1d), fxref(rfi_regular_1d), fxref(rfi_random_1d)]
)
@pytest.mark.parametrize("scale", [1000, 100])
def test_xrfi_model_sweep_all(sky_model, rfi_model, scale):
    sky, std, noise, rfi = make_sky(sky_model, rfi_model, scale)

    true_flags = rfi_model > 0
    flags, info = xrfi.xrfi_model_sweep(
        sky, max_iter=10, which_bin="all", threshold=5, use_median=True
    )

    # Only consider flags after bin 100 (since that's the bin width)
    wrong = np.where(true_flags[100:] != flags[100:])[0]

    print_wrongness(wrong, std, info, noise, true_flags, sky, rfi)
    assert len(wrong) == 0


@parametrize_plus(
    "sky_model", [fxref(sky_flat_1d), fxref(sky_pl_1d), fxref(sky_linpoly_1d)]
)
@parametrize_plus(
    "rfi_model", [fxref(rfi_null_1d), fxref(rfi_regular_1d), fxref(rfi_random_1d)]
)
@pytest.mark.parametrize("scale", [1000, 100])
def test_xrfi_model_sweep_watershed(sky_model, rfi_model, scale):
    sky, std, noise, rfi = make_sky(sky_model, rfi_model, scale)

    true_flags = rfi_model > 0
    flags, info = xrfi.xrfi_model_sweep(
        sky, max_iter=10, which_bin="all", threshold=5, use_median=True, watershed=3
    )

    # Only consider flags after bin 100 (since that's the bin width)
    wrong = np.where(true_flags[100:] & ~flags[100:])[0]

    print_wrongness(wrong, std, info, noise, true_flags, sky, rfi)
    assert len(wrong) == 0


def test_xrfi_model_too_many_nans():
    spec = np.nan * np.ones(500)

    with pytest.raises(xrfi.NoDataError):
        xrfi.xrfi_model_sweep(spec)


@parametrize_plus(
    "rfi_model", [fxref(rfi_null_1d), fxref(rfi_regular_1d), fxref(rfi_random_1d)]
)
@pytest.mark.parametrize("scale", [1000, 100])
def test_xrfi_model_sweep_median(sky_flat_1d, rfi_model, scale):
    rfi = rfi_model.copy()
    rfi[:100] = 0
    sky, std, noise, rfi = make_sky(sky_flat_1d, rfi_model, scale)

    true_flags = rfi_model > 0
    flags, info = xrfi.xrfi_model_sweep(
        sky, max_iter=10, threshold=5, use_median=False, which_bin="all"
    )

    # Only consider flags after bin 100 (since that's the bin width)
    wrong = np.where(true_flags[100:] != flags[100:])[0]

    print_wrongness(wrong, std, info, noise, true_flags, sky, rfi)

    assert len(wrong) == 0


def print_wrongness(wrong, std, info, noise, true_flags, sky, rfi):
    if len(wrong) > 0:
        print("Indices of WRONG flags:")
        print(100 + wrong)
        print("RFI false positive(0)/negative(1): ")
        print(true_flags[wrong])
        print("Corrupted sky at wrong flags: ")
        print(sky[wrong])
        print("Std. dev away from model at wrong flags: ")
        print((sky[wrong] - sky[wrong]) / std[wrong])
        print("Std. dev of noise away from model at wrong flags: ")
        print(noise[wrong] / std[wrong])
        print("Std dev of RFI away from model at wrong flags: ")
        print(rfi[wrong] / std[wrong])
        print("Measured Std Dev: ")
        print(min(info.get("std", [0])), max(info.get("std", [0])))
        print("Actual Std Dev (for uniform):", np.std(noise))


def make_sky(sky_model, rfi_model=np.zeros(NFREQ), scale=1000, rfi_amp=200):
    std = sky_model / scale
    amp = std.max() * rfi_amp
    noise = thermal_noise(sky_model, scale=scale, seed=1010)
    rfi = rfi_model * amp
    return sky_model + noise + rfi, std, noise, rfi


def test_xrfi_explicit(freq, sky_flat_1d, rfi_regular_1d):
    flags = xrfi.xrfi_explicit(freq, extra_rfi=[(60, 70), (80, 90)])
    assert flags[105]
    assert not flags[0]
    assert flags[350]


def test_xrfi_model_sweep_watershed_last(sky_flat_1d):
    with pytest.raises(ValueError):
        xrfi.xrfi_model_sweep(sky_flat_1d, which_bin="last", watershed=4)


def test_giving_weights(sky_flat_1d):
    sky, std, noise, rfi = make_sky(sky_flat_1d)

    flags, info = xrfi.xrfi_model_sweep(
        sky,
        weights=np.ones_like(sky),
        max_iter=10,
        which_bin="all",
        threshold=5,
        use_median=True,
    )

    flags2, info2 = xrfi.xrfi_model_sweep(
        sky, max_iter=10, which_bin="all", threshold=5, use_median=True
    )

    assert np.all(flags == flags2)


def test_visualisation(sky_pl_1d, rfi_random_1d):
    sky, std, noise, rfi = make_sky(sky_pl_1d)
    flags, info = xrfi.xrfi_model(sky, return_models=True, max_iter=3)
    xrfi.visualise_model_info(sky, flags, info)
