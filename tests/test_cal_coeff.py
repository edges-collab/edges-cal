import pytest

import numpy as np
from astropy import units as u
from pathlib import Path

from edges_cal import cal_coefficients as cc


def test_load_from_io(io_obs, tmpdir: Path):
    load = cc.Load.from_io(io_obs, load_name="hot_load", ambient_temperature=300.0)

    assert load.load_name == "hot_load"
    mask = ~np.isnan(load.averaged_Q)
    assert np.all(load.averaged_Q[mask] == load.spectrum.averaged_Q[mask])


def test_calobs_bad_input(io_obs):
    with pytest.raises(ValueError):
        cc.CalibrationObservation.from_io(io_obs, f_low=100 * u.MHz, f_high=40 * u.MHz)


def test_new_load(calobs, io_obs):
    new_open = calobs.new_load(
        "open", io_obs, spec_kwargs={"ignore_times_percent": 50.0}
    )
    assert new_open.spectrum.temp_ave != calobs.open.spectrum.temp_ave


def test_cal_uncal_round_trip(calobs):

    tcal = calobs.calibrate("ambient")
    raw = calobs.decalibrate(tcal, "ambient")
    mask = ~np.isnan(raw)
    assert np.allclose(raw[mask], calobs.ambient.averaged_spectrum[mask])

    with pytest.warns(UserWarning):
        calobs.decalibrate(tcal, "ambient", freq=np.linspace(30, 120, 50) * u.MHz)


def test_load_resids(calobs):
    cal = calobs.calibrate("ambient")

    out = calobs.get_load_residuals()
    mask = ~np.isnan(cal)
    assert np.allclose(out["ambient"][mask], cal[mask] - calobs.ambient.temp_ave)


def test_rms(calobs):
    rms = calobs.get_rms()
    assert isinstance(rms, dict)
    assert isinstance(rms["ambient"], float)


def test_update(calobs):
    c2 = calobs.clone(wterms=10)

    assert len(c2.Tcos_poly) > len(calobs.Tcos_poly)
    assert len(c2.Tcos_poly) == 9


def test_calibration_init(calobs, tmpdir: Path):

    calobs.write(tmpdir / "calfile.h5")

    cal = cc.Calibration(tmpdir / "calfile.h5")

    assert np.allclose(
        cal.receiver_s11(), calobs.receiver.s11_model(calobs.freq.freq.to_value("MHz"))
    )
    assert np.allclose(cal.C1(), calobs.C1())
    assert np.allclose(cal.C2(), calobs.C2())
    assert np.allclose(cal.Tunc(), calobs.Tunc())
    assert np.allclose(cal.Tcos(), calobs.Tcos())
    assert np.allclose(cal.Tsin(), calobs.Tsin())

    temp = calobs.ambient.averaged_spectrum
    s11 = calobs.ambient.reflections.s11_model(calobs.freq.freq.to_value("MHz"))
    cal_temp = cal.calibrate_temp(calobs.freq.freq, temp, s11)
    mask = ~np.isnan(cal_temp)
    assert np.allclose(cal_temp[mask], calobs.calibrate("ambient")[mask])

    mask = ~np.isnan(temp)
    assert np.allclose(
        cal.decalibrate_temp(calobs.freq.freq, cal_temp, s11)[mask], temp[mask]
    )


def test_term_sweep(io_obs):
    print("Making it")
    calobs = cc.CalibrationObservation.from_io(
        io_obs,
        cterms=5,
        wterms=7,
        f_low=60 * u.MHz,
        f_high=80 * u.MHz,
    )
    print("Made")

    calobs_opt = cc.perform_term_sweep(
        calobs,
        max_cterms=6,
        max_wterms=8,
    )

    assert isinstance(calobs_opt, cc.CalibrationObservation)


def test_2017_semi_rigid():
    hlc = cc.HotLoadCorrection.from_file(path=":semi_rigid_s_parameters_2017.txt")
    assert hlc.s12s21_model(hlc.freq.freq.to_value("MHz")).dtype == complex


def test_calobs_equivalence(calobs, io_obs):
    calobs1 = cc.CalibrationObservation.from_io(
        io_obs, f_low=50 * u.MHz, f_high=100 * u.MHz
    )

    assert calobs1.open.spectrum == calobs.open.spectrum
    assert calobs1.open.reflections == calobs.open.reflections
    assert calobs1.open == calobs.open


def test_basic_s11_properties(calobs):
    assert calobs.open.reflections.load_s11.load_name == "open"


def test_inject(calobs):
    new = calobs.inject(
        lna_s11=calobs.receiver_s11 * 2,
    )

    np.testing.assert_allclose(new.receiver_s11, 2 * calobs.receiver_s11)
    assert not np.allclose(
        new.get_linear_coefficients("open")[0],
        calobs.get_linear_coefficients("open")[0],
    )
