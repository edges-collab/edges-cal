"""
Integration tests that ensure we can reproduce Alan's C-code results.
"""

import traceback
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from edges_cal import alanmode as am
from edges_cal.cli import alancal


def test_edges3_2022_316_against_alan(data_path, tmp_path_factory):
    # Skip this test if we are not on enterprise where the actual data is.
    datadir = Path("/data5/edges/data/EDGES3_data/MRO")
    if not datadir.exists():
        pytest.skip("This text can only be executed on enterprise")
    out = tmp_path_factory.mktemp("day316")

    runner = CliRunner()

    result = runner.invoke(
        alancal,
        [
            "2022_319_14",
            "2022",
            "316",
            "--out",
            str(out.absolute()),
            "-res",
            "49.8",
            "-ps",
            "33",
            "-cablen",
            "4.26",
            "-cabloss",
            "-91.5",
            "-cabdiel",
            "-1.24",
            "-fstart",
            "48",
            "-fstop",
            "198",
            "-smooth",
            "8",
            "-tload",
            "300",
            "-tcal",
            "1000",
            "-Lh",
            "-1",
            "-wfstart",
            "50.0",
            "-wfstop",
            "190.0",
            "-tcold",
            "306.5",
            "-thot",
            393.22,
            "-tcab",
            "306.5",
            "-cfit",
            "7",
            "-wfit",
            "7",
            "-nfit3",
            "10",
            "-nfit2",
            "27",
        ],
    )

    if result.exit_code:
        print(result.exception)
        print(traceback.print_exception(*result.exc_info))

    print(result.output)
    assert result.exit_code == 0

    loads = ["amb", "hot", "open", "short"]

    alandata = data_path / "edges3-2022-316-alan"

    # Test the raw spectra
    alanspec = {}
    ourspec = {}
    for load in loads:
        data = am.read_spec_txt(f"{alandata}/sp{load}.txt")
        spfreq = data["freq"]
        alanspec[load] = data["spectra"]
        data = am.read_spec_txt(f"{out}/sp{load}.txt")
        ourspec[load] = data["spectra"]
        np.testing.assert_allclose(spfreq, data["freq"])

    for load, val in alanspec.items():
        print(f"Raw Spectrum from {load}")
        # We don't currently get the very edges of the smoothing correct, but it doesn't
        # matter because we never use the very edges anyway. We test within these edges.
        np.testing.assert_allclose(val[20:-20], ourspec[load][20:-20], atol=1e-6)

    # Test the calibrated (unmodelled) S11s
    for load in [*loads, "lna"]:
        print(f"Raw S11 {load}")
        s11freq, alans11 = am.read_s11_csv(f"{alandata}/s11{load}.csv")
        ourfreq, ours11 = am.read_s11_csv(f"{out}/s11{load}.csv")

        np.testing.assert_allclose(s11freq, ourfreq)
        np.testing.assert_allclose(alans11.real, ours11.real, atol=1e-10)
        np.testing.assert_allclose(alans11.imag, ours11.imag, atol=1e-10)

    # Test modelled S11s
    _alans11m = np.genfromtxt(f"{alandata}/s11_modelled.txt", comments="#", names=True)
    _ours11m = np.genfromtxt(f"{out}/s11_modelled.txt", comments="#", names=True)

    for k in _alans11m.dtype.names:
        print(f"Modelled S11 {k}")

        # We clip the ends here, because they are slightly extrapolated in the default
        # case.
        np.testing.assert_allclose(_alans11m[k], _ours11m[k], atol=3e-9, rtol=0)

    # Test final calibration
    acal = am.read_specal(f"{alandata}/specal_316test.txt")
    ourcal = am.read_specal(f"{out}/specal.txt")

    np.testing.assert_allclose(acal["freq"], ourcal["freq"])
    np.testing.assert_allclose(acal["C1"], ourcal["C1"], atol=1e-8)
    np.testing.assert_allclose(acal["C2"], ourcal["C2"], atol=1e-8)
    np.testing.assert_allclose(acal["Tunc"], ourcal["Tunc"], atol=1e-8)
    np.testing.assert_allclose(acal["Tcos"], ourcal["Tcos"], atol=1e-8)
    np.testing.assert_allclose(acal["Tsin"], ourcal["Tsin"], atol=1e-8)
