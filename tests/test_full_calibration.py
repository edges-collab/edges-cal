"""Test that a full calibraiton gives results that don't change over time."""

import pytest

import h5py
import hickle
import numpy as np
from astropy import units as u
from edges_io import io

from edges_cal import s11
from edges_cal.cal_coefficients import CalibrationObservation, HotLoadCorrection, Load


@pytest.fixture(scope="module")
def data(data_path):
    return data_path / "2015-09-data"


@pytest.fixture(scope="module")
def s11dir(data) -> io.S11Dir:
    return io.S11Dir(data / "S11", run_num={"receiver_reading": 1})


@pytest.fixture(scope="module")
def calobs_2015(data, s11dir):
    f_low = 50 * u.MHz
    f_high = 100 * u.MHz

    receiver = s11.Receiver.from_io(
        s11dir.receiver_reading,
        resistance=49.98 * u.ohm,
        n_terms=11,
        model_type="polynomial",
    )

    refl = {}
    for src in ["ambient", "hot_load", "open", "short"]:
        refl[src] = s11.LoadS11.from_io(
            s11dir,
            src,
            internal_switch_kwargs={"resistance": 50.12 * u.ohm},
            f_low=f_low,
            f_high=f_high,
        )

    spec = {}
    for src in refl:
        spec[src] = hickle.load(data / f"spectra/{src}.h5")

    loads = {}
    for src in refl:
        if src != "hot_load":
            loads[src] = Load(spectrum=spec[src], reflections=refl[src])
        else:
            loads[src] = Load(
                spectrum=spec[src],
                reflections=refl[src],
                loss_model=HotLoadCorrection.from_file(f_low=f_low, f_high=f_high),
                ambient_temperature=spec["ambient"].temp_ave,
            )

    return CalibrationObservation(loads=loads, receiver=receiver, cterms=6, wterms=5)


@pytest.mark.parametrize("p", ["C1", "C2", "Tunc", "Tcos", "Tsin"])
def test_cal_params(data, calobs_2015, p):
    with h5py.File(data / "reference.h5", "r") as fl:
        f = fl["freq"][...]
        np.testing.assert_allclose(
            getattr(calobs_2015, p)(f * u.MHz), fl[p.lower()][...], rtol=1e-4
        )


def test_receiver(data, calobs_2015):
    with h5py.File(data / "reference.h5", "r") as fl:
        f = fl["freq"][...]
        np.testing.assert_allclose(
            calobs_2015.receiver.raw_s11, fl["receiver_raw"][...], atol=1e-8, rtol=1e-5
        )
        np.testing.assert_allclose(
            calobs_2015.receiver.s11_model(f),
            fl["receiver_modeled"][...],
            atol=1e-8,
            rtol=1e-5,
        )


@pytest.mark.parametrize("p", ["s11", "s12", "s22"])
def test_internal_switch(data, calobs_2015, p):
    with h5py.File(data / "reference.h5", "r") as fl:
        f = fl["freq"][...]
        np.testing.assert_allclose(
            getattr(calobs_2015.internal_switch, f"{p}_data"),
            fl[f"isw_raw_{p}"][...],
            atol=1e-8,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            getattr(calobs_2015.internal_switch, f"{p}_model")(f),
            fl[f"isw_mdl_{p}"][...],
            atol=1e-8,
            rtol=1e-5,
        )


@pytest.mark.parametrize("load", ["ambient", "hot_load", "short", "open"])
def test_load_s11(data, calobs_2015, load):
    with h5py.File(data / "reference.h5", "r") as fl:
        f = fl["freq"][...]
        np.testing.assert_allclose(
            getattr(calobs_2015, load).s11_model(f),
            fl[f"{load}_s11"][...],
            atol=1e-8,
            rtol=1e-5,
        )


def make_comparison_data(obspath):
    calio = io.CalibrationObservation(
        obspath,
        run_num={"receiver_reading": 1},
        repeat_num=1,
    )

    calobs = CalibrationObservation.from_io(
        calio,
        f_low=50.0 * u.MHz,
        f_high=100.0 * u.MHz,
        cterms=6,
        wterms=5,
        spectrum_kwargs={
            "default": {"t_load": 300, "t_load_ns": 350, "ignore_times_percent": 7},
            "hot_load": {"ignore_times_percent": 10},
        },
        receiver_kwargs={"n_terms": 11, "model_type": "polynomial"},
    )

    f = np.linspace(50, 100, 100) * u.MHz

    with h5py.File("data/2015-09-data/reference.h5", "w") as fl:

        fl["freq"] = f

        fl["c1"] = calobs.C1(f)
        fl["c2"] = calobs.C2(f)
        fl["tunc"] = calobs.Tunc(f)
        fl["tcos"] = calobs.Tcos(f)
        fl["tsin"] = calobs.Tsin(f)

        fl["receiver_raw"] = calobs.receiver.raw_s11
        fl["receiver_modeled"] = calobs.receiver.s11_model(f)

        fl["isw_raw_s11"] = calobs.internal_switch.s11_data
        fl["isw_raw_s12"] = calobs.internal_switch.s12_data
        fl["isw_raw_s22"] = calobs.internal_switch.s22_data
        fl["isw_mdl_s11"] = calobs.internal_switch.s11_model(f.to_value("MHz"))
        fl["isw_mdl_s12"] = calobs.internal_switch.s12_model(f.to_value("MHz"))
        fl["isw_mdl_s22"] = calobs.internal_switch.s22_model(f.to_value("MHz"))

        for src, load in calobs.loads.items():
            fl[f"{src}_s11"] = load.s11_model(f)

    for src, load in calobs.loads.items():
        hickle.dump(load.spectrum, f"data/2015-09-data/spectra/{src}.h5")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        obsdir = sys.argv[1]
    else:
        obsdir = (
            "/data5/edges/data/CalibrationObservations/Receiver01/"
            "Receiver01_25C_2015_09_02_040_to_200MHz"
        )

    make_comparison_data(obsdir)
