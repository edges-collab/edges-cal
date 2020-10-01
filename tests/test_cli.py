import pytest

from click.testing import CliRunner
from pathlib import Path

from edges_cal.cli import run


def test_run(data_path: Path, tmpdir: Path):
    runner = CliRunner()
    outdir = tmpdir / "cli-out"
    outdir.mkdir()
    result = runner.invoke(
        run,
        [
            str(data_path / "settings.yaml"),
            str(data_path / "Receiver01_25C_2019_11_26_040_to_200MHz"),
            "--out",
            str(outdir),
            "--cache-dir",
            str(outdir),
            "--plot",
        ],
    )

    assert result.exit_code == 0
