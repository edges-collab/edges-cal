"""
Test spectrum reading.
"""
import pytest

from edges_cal import LoadSpectrum


@pytest.fixture(scope="module")
def ambient(data_path, tmpdir) -> LoadSpectrum:
    calpath = data_path / "Receiver01_25C_2019_11_26_040_to_200MHz"

    return LoadSpectrum.from_load_name("ambient", calpath, cache_dir=tmpdir)


def test_read(ambient: LoadSpectrum):
    assert ambient.averaged_Q.ndim == 1


def test_datetimes(ambient: LoadSpectrum):
    assert len(ambient.thermistor_timestamps) == len(ambient.thermistor)
