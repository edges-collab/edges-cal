"""
Test spectrum reading.
"""
from pathlib import Path

from edges_cal import LoadSpectrum


def test_read(data_path: Path):
    calpath = data_path / "Receiver01_2019_11_26_040_to_200MHz/25C"

    spec = LoadSpectrum.from_load_name("ambient", calpath)

    assert spec.averaged_Q.ndim == 1
