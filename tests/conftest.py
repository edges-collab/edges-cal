"""
Conftest.
"""

import pytest

from pathlib import Path


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
