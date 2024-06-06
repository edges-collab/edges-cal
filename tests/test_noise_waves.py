"""Tests of the noise-wave fitting (iterative) procedure."""

from collections import deque

import numpy as np
import pytest
from edges_cal import noise_waves as nw

N = 501
FREQ = np.linspace(50, 100, N)


@pytest.mark.parametrize(
    ("true_sca", "true_off", "true_t_unc", "true_t_cos", "true_t_sin"),
    [
        (np.ones(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)),
        (np.linspace(3.5, 4.5, N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)),
        (np.linspace(3.5, 4.5, N), -np.ones(N), np.zeros(N), np.zeros(N), np.zeros(N)),
        (
            np.linspace(3.5, 4.5, N),
            -np.ones(N),
            np.linspace(19.0, 20.0, N),
            np.zeros(N),
            np.zeros(N),
        ),
        (
            np.linspace(3.5, 4.5, N),
            -np.ones(N),
            np.linspace(19.0, 20.0, N),
            np.linspace(4, 5, N),
            np.linspace(10, 15, N),
        ),
    ],
    ids=[
        "trivial",
        "non-zero scale",
        "non-zero scale and off",
        "non-zero tunc",
        "all non-zero",
    ],
)
@pytest.mark.parametrize(
    "gamma_rec",
    [
        np.zeros(N, dtype=complex),
        1e-5 * np.exp(-1j * FREQ / 12),
        1e-2 * np.exp(1j * FREQ / 6),
    ],
    ids=["perfect_rx", "low-level-rx", "high-level-rx"],
)
@pytest.mark.parametrize(
    "gamma_amb",
    [np.zeros(N, dtype=complex), 1e-5 * np.exp(-1j * FREQ / 12)],
    ids=["perfect_ra", "low-level-ra"],
)
def test_fit_perfect_receiver(
    true_sca, true_off, true_t_unc, true_t_cos, true_t_sin, gamma_rec, gamma_amb
):
    """Test that noise-wave fits work."""
    gamma_ant = {
        "ambient": gamma_amb,
        "hot_load": gamma_amb,
        "short": (FREQ / 75) ** -1 * np.exp(1j * FREQ / 12),
        "open": (FREQ / 75) ** -1 * np.exp(-1j * FREQ / 12),
    }

    temp = {"ambient": 300, "hot_load": 400, "short": 300, "open": 300}

    uncal_temp = {
        k: nw.decalibrate_antenna_temperature(
            temp=temp[k],
            gamma_ant=gamma_ant[k],
            gamma_rec=gamma_rec,
            sca=true_sca,
            off=true_off,
            t_unc=true_t_unc,
            t_cos=true_t_cos,
            t_sin=true_t_sin,
        )
        for k in temp
    }

    result = deque(
        nw.get_calibration_quantities_iterative(
            freq=FREQ,
            temp_raw=uncal_temp,
            gamma_rec=gamma_rec,
            gamma_ant=gamma_ant,
            temp_ant=temp,
            cterms=5,
            wterms=5,
        ),
        maxlen=1,
    )
    sca, off, tu, tc, ts = result.pop()

    assert np.allclose(sca(FREQ), true_sca)
    assert np.allclose(off(FREQ), true_off)
    assert np.allclose(tu(FREQ), true_t_unc)
    assert np.allclose(tc(FREQ), true_t_cos)
    assert np.allclose(ts(FREQ), true_t_sin)
