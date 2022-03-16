"""
Test spectrum reading.
"""
import pytest

import numpy as np
from astropy import units as u
from edges_io.io import CalibrationObservation

from edges_cal import LoadSpectrum


def test_read(io_obs: CalibrationObservation):
    for load in ["ambient", "hot_load", "open", "short"]:
        spec = LoadSpectrum.from_io(io_obs, load, f_high=100 * u.MHz, f_low=50 * u.MHz)

        print(spec.averaged_Q.shape)
        print(np.where(np.isnan(spec.averaged_Q)))

        mask = ~np.isnan(spec.averaged_Q)
        assert np.sum(~mask) < 100
        assert spec.averaged_Q.ndim == 1

        assert np.all(~np.isnan(spec.variance_Q[mask]))
        assert np.all(spec.averaged_spectrum[mask] > spec.averaged_Q[mask])
        assert ~np.isinf(spec.temp_ave)
        assert ~np.isnan(spec.temp_ave)

        assert np.all(spec.variance_spectrum[mask] > spec.variance_Q[mask])


def test_equality(io_obs: CalibrationObservation):
    spec1 = LoadSpectrum.from_io(
        io_obs, "ambient", f_high=100 * u.MHz, f_low=50 * u.MHz
    )
    spec2 = LoadSpectrum.from_io(
        io_obs, "ambient", f_high=100 * u.MHz, f_low=50 * u.MHz
    )

    assert spec1 == spec2
    assert spec1.metadata["hash"] == spec2.metadata["hash"]

    spec3 = LoadSpectrum.from_io(
        io_obs, "ambient", f_high=100 * u.MHz, f_low=60 * u.MHz
    )
    assert spec3.metadata["hash"] != spec2.metadata["hash"]


def test_temperature_range(io_obs):
    with pytest.raises(RuntimeError, match="The temperature range has masked"):
        # Fails only because our test data is awful. spectra and thermistor measurements
        # don't overlap.
        LoadSpectrum.from_io(io_obs, "ambient", temperature_range=0.5)

    with pytest.raises(RuntimeError, match="The temperature range has masked"):
        LoadSpectrum.from_io(io_obs, "ambient", temperature_range=(20, 40))
