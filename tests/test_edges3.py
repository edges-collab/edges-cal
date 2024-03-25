"""Tests of creating EDGES-3 style calibration."""

import pytest
from astropy import units as un
from edges_cal import CalibrationObservation
from edges_io import TEST_DATA_PATH, io3


@pytest.fixture(scope="module")
def smallcal() -> io3.CalibrationObservation:
    return io3.CalibrationObservation.from_date(
        root_dir=TEST_DATA_PATH / "edges3-mock-root",
        year=2023,
        day=70,
    )


@pytest.fixture(scope="module")
def calobs(smallcal):
    return CalibrationObservation.from_edges3(
        smallcal,
        f_low=50 * un.MHz,
        f_high=100 * un.MHz,
        spectrum_kwargs={"default": {"allow_closest_time": True}},
    )


def test_calobs_creation(calobs):
    assert calobs is not None
