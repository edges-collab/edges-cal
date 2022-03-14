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


def test_temperature_range(ambient: LoadSpectrum, data_path, tmpdir):
    calpath = data_path / "Receiver01_25C_2019_11_26_040_to_200MHz"

    with pytest.raises(RuntimeError, match="The temperature range has masked"):
        # Fails only because our test data is awful. spectra and thermistor measurements
        # don't overlap.
        new = LoadSpectrum.from_load_name(
            "ambient", calpath, cache_dir=tmpdir, temperature_range=0.5
        )
        new.n_integrations

    with pytest.raises(RuntimeError, match="The temperature range has masked"):
        new = LoadSpectrum.from_load_name(
            "ambient", calpath, cache_dir=tmpdir, temperature_range=(20, 40)
        )
        new.n_integrations
