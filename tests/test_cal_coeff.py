import pytest

import h5py
import logging
import numpy as np
from edges_io import io
from pathlib import Path

from edges_cal import cal_coefficients as cc


def test_vna_from_file(data_path):
    s1p = cc.S1P(
        data_path
        / "Receiver01_25C_2019_11_26_040_to_200MHz/S11/Ambient01/External01.s1p"
    )
    assert hasattr(s1p, "s1p")


def test_vna_from_s1p(data_path):
    s1p = io.S1P(
        data_path
        / "Receiver01_25C_2019_11_26_040_to_200MHz/S11/Ambient01/External01.s1p"
    )
    s1p = cc.S1P(s1p)
    assert hasattr(s1p, "s1p")


def test_vna_bad_input():
    with pytest.raises(TypeError):
        cc.S1P(3)


def test_even_nterms_s11(cal_data):
    s11 = cc.LoadS11.from_path("ambient", cal_data, n_terms=40)

    with pytest.raises(ValueError):
        s11.n_terms


def test_lna_from_path(cal_data):
    lna = cc.LNA.from_path(cal_data)
    assert lna.repeat_num == 1


def test_1d_rfi_removal(cal_data, tmpdir, caplog):
    cache = tmpdir / "cal-coeff-cache"
    calobs = cc.CalibrationObservation(
        cal_data,
        load_kwargs={"rfi_removal": "1D", "cache_dir": cache},
        compile_from_def=False,
        include_previous=False,
    )

    assert calobs.ambient.spectrum.averaged_Q.ndim == 1
    assert calobs.ambient.spectrum.averaged_p0.ndim == 1
    assert calobs.ambient.spectrum.averaged_p1.ndim == 1
    assert calobs.ambient.spectrum.averaged_p2.ndim == 1

    assert calobs.ambient.spectrum.variance_Q.ndim == 1
    assert calobs.ambient.spectrum.variance_p0.ndim == 1
    assert calobs.ambient.spectrum.variance_p1.ndim == 1
    assert calobs.ambient.spectrum.variance_p2.ndim == 1

    assert calobs.ambient.spectrum.variance_spectrum.ndim == 1
    assert isinstance(calobs.ambient.spectrum.ancillary, list)
    assert isinstance(calobs.ambient.spectrum.ancillary[0], dict)

    calobs2 = cc.CalibrationObservation(
        cal_data,
        load_kwargs={"rfi_removal": "1D", "cache_dir": cache},
        compile_from_def=False,
        include_previous=False,
    )

    print(list(cache.glob("*")))
    # Access an averaged quantity

    caplog.set_level(logging.INFO)
    prev_level = cc.logger.getEffectiveLevel()
    cc.logger.setLevel(logging.INFO)
    assert np.allclose(
        calobs.ambient.spectrum.averaged_Q, calobs2.ambient.spectrum.averaged_Q
    )
    assert "Reading in previously-created integrated ambient" in caplog.text
    cc.logger.setLevel(prev_level)


def test_spec_write(cal_data: Path, tmpdir: Path):
    spec = cc.LoadSpectrum.from_load_name("ambient", cal_data, cache_dir=tmpdir)

    spec.write(tmpdir)

    with h5py.File(tmpdir / "ambient_averaged_spectrum.h5", "r") as fl:
        mask = ~np.isnan(spec.averaged_spectrum)
        assert np.allclose(
            fl["averaged_raw_spectrum"][...][mask], spec.averaged_spectrum[mask]
        )


def test_load_from_path(cal_data: Path, tmpdir: Path):
    cache = tmpdir / "cal-coeff-cache"

    load = cc.Load.from_path(
        cal_data, load_name="hot_load", spec_kwargs={"cache_dir": cache}
    )

    assert load.spectrum.load_name == "hot_load"
    mask = ~np.isnan(load.averaged_Q)
    assert np.all(load.averaged_Q[mask] == load.spectrum.averaged_Q[mask])


def test_calobs_bad_input(cal_data: Path):
    with pytest.raises(TypeError):
        cc.CalibrationObservation(cal_data, load_spectra={"ambient": "derp"})

    with pytest.raises(ValueError):
        cc.CalibrationObservation(cal_data, f_low=100, f_high=40)


def test_new_load(cal_data: Path):
    calobs = cc.CalibrationObservation(cal_data, compile_from_def=False)

    new_open = calobs.new_load("open", spec_kwargs={"ignore_times_percent": 50.0})
    assert len(new_open.spectrum.thermistor_temp) < len(
        calobs.open.spectrum.thermistor_temp
    )


@pytest.mark.skip("too slow...")
def test_2d_rfi_removal(cal_data, tmpdir, caplog):
    cache = tmpdir / "cal-coeff-cache-new"

    calobs = cc.CalibrationObservation(
        cal_data,
        load_kwargs={
            "rfi_removal": "2D",
            "cache_dir": cache,
            "rfi_kernel_width_time": 2,
            "rfi_kernel_width_freq": 2,
        },
        compile_from_def=False,
    )

    assert calobs.ambient.spectrum.averaged_Q.ndim == 1
    assert calobs.ambient.spectrum.averaged_p0.ndim == 1
    assert calobs.ambient.spectrum.averaged_p1.ndim == 1
    assert calobs.ambient.spectrum.averaged_p2.ndim == 1

    assert calobs.ambient.spectrum.variance_Q.ndim == 1
    assert calobs.ambient.spectrum.variance_p0.ndim == 1
    assert calobs.ambient.spectrum.variance_p1.ndim == 1
    assert calobs.ambient.spectrum.variance_p2.ndim == 1

    assert calobs.ambient.spectrum.variance_spectrum.ndim == 1
    assert isinstance(calobs.ambient.spectrum.ancillary, list)
    assert isinstance(calobs.ambient.spectrum.ancillary[0], dict)


def test_bad_fminmax(cal_data: Path):
    with pytest.raises(ValueError):
        cc.CalibrationObservation(cal_data, f_low=100, f_high=50)


def test_cal_uncal_round_trip(cal_data: Path, tmpdir: Path):
    cache = tmpdir / "cal-coeff-cache"
    calobs = cc.CalibrationObservation(
        cal_data, load_kwargs={"cache_dir": cache}, compile_from_def=False
    )

    tcal = calobs.calibrate("ambient")
    raw = calobs.decalibrate(tcal, "ambient")
    mask = ~np.isnan(raw)
    assert np.allclose(raw[mask], calobs.ambient.averaged_spectrum[mask])

    with pytest.warns(UserWarning):
        calobs.decalibrate(tcal, "ambient", freq=np.linspace(30, 120, 50))


def test_load_resids(cal_data: Path, tmpdir: Path):
    cache = tmpdir / "cal-coeff-cache"
    calobs = cc.CalibrationObservation(
        cal_data, load_kwargs={"cache_dir": cache}, compile_from_def=False
    )

    cal = calobs.calibrate("ambient")

    out = calobs.get_load_residuals()
    mask = ~np.isnan(cal)
    assert np.allclose(out["ambient"][mask], cal[mask] - calobs.ambient.temp_ave)


def test_rms(cal_data: Path, tmpdir: Path):
    cache = tmpdir / "cal-coeff-cache"
    calobs = cc.CalibrationObservation(
        cal_data, load_kwargs={"cache_dir": cache}, compile_from_def=False
    )

    rms = calobs.get_rms()
    assert isinstance(rms, dict)
    assert isinstance(rms["ambient"], float)


def test_write_coefficients(cal_data: Path, tmpdir: Path):
    cache = tmpdir / "cal-coeff-cache"
    calobs = cc.CalibrationObservation(
        cal_data, load_kwargs={"cache_dir": cache}, compile_from_def=False
    )

    calobs.write_coefficients(tmpdir)

    assert any(
        path.name.startswith("calibration_parameters") for path in tmpdir.glob("*")
    )


def test_update(cal_data: Path, tmpdir: Path):
    cache = tmpdir / "cal-coeff-cache"
    calobs = cc.CalibrationObservation(
        cal_data, load_kwargs={"cache_dir": cache}, wterms=5, compile_from_def=False
    )

    assert len(calobs.Tcos_poly) == 4

    c2 = calobs.clone(wterms=7)

    assert len(c2.Tcos_poly) == 6


def test_calibration_init(cal_data: Path, tmpdir: Path):
    cache = tmpdir / "cal-coeff-cache"
    calobs = cc.CalibrationObservation(
        cal_data, load_kwargs={"cache_dir": cache}, cterms=5, compile_from_def=False
    )

    calobs.write(tmpdir / "calfile.h5")

    cal = cc.Calibration(tmpdir / "calfile.h5")

    assert np.allclose(cal.lna_s11(), calobs.lna.s11_model(calobs.freq.freq))
    assert np.allclose(cal.C1(), calobs.C1())
    assert np.allclose(cal.C2(), calobs.C2())
    assert np.allclose(cal.Tunc(), calobs.Tunc())
    assert np.allclose(cal.Tcos(), calobs.Tcos())
    assert np.allclose(cal.Tsin(), calobs.Tsin())

    temp = calobs.ambient.averaged_spectrum
    s11 = calobs.ambient.reflections.s11_model(calobs.freq.freq)
    cal_temp = cal.calibrate_temp(calobs.freq.freq, temp, s11)
    mask = ~np.isnan(cal_temp)
    assert np.allclose(cal_temp[mask], calobs.calibrate("ambient")[mask])

    mask = ~np.isnan(temp)
    assert np.allclose(
        cal.decalibrate_temp(calobs.freq.freq, cal_temp, s11)[mask], temp[mask]
    )

    with h5py.File(tmpdir / "calfile.h5", "a") as fl:
        fl.attrs["switch_path"] = "/doesnt/exist"


def test_term_sweep(cal_data: Path, tmpdir: Path):
    cache = tmpdir / "cal-coeff-cache"
    calobs = cc.CalibrationObservation(
        cal_data,
        load_kwargs={"cache_dir": cache},
        cterms=5,
        wterms=7,
        f_low=60,
        f_high=80,
    )

    calobs_opt = cc.perform_term_sweep(
        calobs,
        max_cterms=6,
        max_wterms=8,
        explore_run_nums=True,
        explore_repeat_nums=True,
        direc=tmpdir,
    )

    assert isinstance(calobs_opt, cc.CalibrationObservation)


def test_2017_semi_rigid():
    hlc = cc.HotLoadCorrection(path=":semi_rigid_s_parameters_2017.txt")
    assert hlc.s12_model(hlc.freq.freq).dtype == complex


def test_calobs_equivalence(cal_data):
    calobs1 = cc.CalibrationObservation(cal_data, compile_from_def=True)
    calobs2 = cc.CalibrationObservation(cal_data, compile_from_def=True)

    assert calobs1.open == calobs2.open
    assert calobs1.open.spectrum == calobs2.open.spectrum
    assert hash(calobs1.open.spectrum) == hash(calobs2.open.spectrum)


def test_basic_s11_properties(cal_data):
    calobs = cc.CalibrationObservation(cal_data, compile_from_def=False)

    assert calobs.open.reflections.match.load_name == "Match"
    assert calobs.open.reflections.match.repeat_num == 1
