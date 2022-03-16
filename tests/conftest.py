"""
Conftest.
"""

import pytest

from astropy import units as u
from edges_io import io
from pathlib import Path

from edges_cal import CalibrationObservation
from edges_cal.config import config


@pytest.fixture(scope="session", autouse=True)
def data_path() -> Path:
    """Path to test data."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def cal_data(data_path: Path):
    return data_path / "Receiver01_25C_2019_11_26_040_to_200MHz"


@pytest.fixture(scope="session", autouse=True)
def tmpdir(tmp_path_factory):
    return tmp_path_factory.mktemp("edges-cal")


@pytest.fixture(scope="session")
def io_obs(cal_data):
    return io.CalibrationObservation(cal_data)


@pytest.fixture(scope="session", autouse=True)
def set_cache_dir(tmpdir):
    config["cal"]["cache_dir"] = str(tmpdir / "cal-cache")


@pytest.fixture(scope="session")
def calobs(io_obs):
    return CalibrationObservation.from_io(io_obs, f_low=50 * u.MHz, f_high=100 * u.MHz)
