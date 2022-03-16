import pytest

import numpy as np
from astropy import units as u
from edges_io import io

from edges_cal import modelling as mdl
from edges_cal import reflection_coefficient as rc
from edges_cal import s11


def test_gamma_shift_zero():
    s11 = np.random.normal(size=100)
    assert np.all(s11 == rc.gamma_embed(s11, 0, 0, 0))


def test_gamma_impedance_roundtrip():
    z0 = 50
    z = np.random.normal(size=10)

    np.testing.assert_allclose(rc.gamma2impedance(rc.impedance2gamma(z, z0), z0), z)


def test_gamma_embed_rountrip():
    s11 = np.random.uniform(0, 1, size=10)
    s12s21 = np.random.uniform(0, 1, size=10)
    s22 = np.random.uniform(0, 1, size=10)

    gamma = np.random.uniform(0, 1, size=10)

    np.testing.assert_allclose(
        rc.gamma_de_embed(s11, s12s21, s22, rc.gamma_embed(s11, s12s21, s22, gamma)),
        gamma,
    )


def test_calkit_standard_name():
    assert rc.CalkitStandard(resistance=50).name == "match"

    with pytest.raises(ValueError, match="capacitance_model is required"):
        rc.CalkitStandard(resistance=np.inf)

    assert (
        rc.CalkitStandard(
            resistance=np.inf,
            capacitance_model=rc.AGILENT_85033E.open.capacitance_model,
        ).name
        == "open"
    )

    with pytest.raises(ValueError, match="inductance_model is required"):
        rc.CalkitStandard(resistance=0)

    assert (
        rc.CalkitStandard(
            resistance=0, inductance_model=rc.AGILENT_85033E.short.inductance_model
        ).name
        == "short"
    )


def test_calkit_termination_impedance():
    with pytest.raises(TypeError, match="freq must be a frequency quantity!"):
        # requires frequency to be in units
        rc.AGILENT_85033E.open.termination_impedance(np.linspace(50, 100, 100))

    assert (
        rc.AGILENT_85033E.match.termination_impedance(50 * u.MHz)
        == rc.AGILENT_85033E.match.resistance
    )


def test_calkit_units():
    freq = np.linspace(50, 100, 100) * u.MHz

    ag = rc.AGILENT_85033E.open

    assert ag.termination_impedance(freq).unit == u.ohm
    assert ag.termination_gamma(freq).unit == u.dimensionless_unscaled
    assert ag.lossy_characteristic_impedance(freq).unit == u.ohm
    assert u.get_physical_type(ag.gl(freq)) == "dimensionless"
    assert ag.offset_gamma(freq).unit == u.dimensionless_unscaled
    assert ag.reflection_coefficient(freq).unit == u.dimensionless_unscaled


def test_calkit_quantities_match_trivial():
    """A test that for a simple calkit definition, the outputs are correct."""

    std = rc.CalkitStandard(
        resistance=50.0 * u.Ohm,
        offset_impedance=50 * u.Ohm,
        offset_delay=0 * u.ps,
        offset_loss=0.0,
    )

    assert std.intrinsic_gamma == 0.0
    assert std.capacitance_model is None
    assert std.inductance_model is None
    assert std.termination_gamma(freq=150 * u.MHz) == 0.0
    assert std.termination_impedance(freq=150 * u.MHz) == 50 * u.Ohm
    assert std.lossy_characteristic_impedance(freq=150 * u.MHz) == 50 * u.Ohm
    assert std.gl(freq=150 * u.MHz) == 0.0
    assert std.reflection_coefficient(150 * u.MHz) == 0.0


def test_calkit_quantities_match_with_delay():
    """A test that for a simple calkit definition, the outputs are correct."""

    std = rc.CalkitStandard(
        resistance=50.0 * u.Ohm,
        offset_impedance=50 * u.Ohm,
        offset_delay=50 * u.ps,
        offset_loss=0.0,
    )

    assert std.intrinsic_gamma == 0.0
    assert std.capacitance_model is None
    assert std.inductance_model is None
    assert std.termination_gamma(freq=150 * u.MHz) == 0.0
    assert std.termination_impedance(freq=150 * u.MHz) == 50 * u.Ohm
    assert std.lossy_characteristic_impedance(freq=150 * u.MHz) == 50 * u.Ohm
    assert std.gl(freq=200 * u.MHz) == 1e-2 * 2j * np.pi
    assert std.reflection_coefficient(150 * u.MHz) == 0.0


def test_calkit_quantities_match_with_loss():
    """A test that for a simple calkit definition, the outputs are correct."""

    std = rc.CalkitStandard(
        resistance=50.0 * u.Ohm,
        offset_impedance=50 * u.Ohm,
        offset_delay=(25 / np.pi) * u.ps,
        offset_loss=4 * np.pi * u.Gohm / u.s,
    )

    assert std.intrinsic_gamma == 0.0
    assert std.capacitance_model is None
    assert std.inductance_model is None
    assert std.termination_gamma(freq=150 * u.MHz) == 0.0
    assert std.termination_impedance(freq=150 * u.MHz) == 50 * u.Ohm
    assert std.lossy_characteristic_impedance(freq=1 * u.GHz) == (51 - 1j) * u.Ohm
    assert std.gl(freq=1 * u.GHz) == 5e-2j + (1 + 1j) * 1e-3
    assert std.reflection_coefficient(150 * u.MHz) != 0.0


def test_calkit_quantities_open_trivial():
    """A test that for a simple calkit definition, the outputs are correct."""

    std = rc.CalkitStandard(
        resistance=np.inf * u.Ohm,
        offset_impedance=50 * u.Ohm,
        offset_delay=0 * u.ps,
        offset_loss=0 * u.Gohm / u.s,
        capacitance_model=mdl.Polynomial(parameters=[1e-9 / (100 * np.pi), 0, 0, 0]),
    )

    assert std.intrinsic_gamma == 1.0
    assert std.capacitance_model(1e9) == 1e-9 / (100 * np.pi)
    assert std.inductance_model is None
    assert std.termination_impedance(freq=1 * u.GHz) == -50j * u.Ohm
    assert std.termination_gamma(freq=1 * u.GHz) == -1j
    assert std.lossy_characteristic_impedance(freq=1 * u.GHz) == 50 * u.Ohm
    assert std.gl(freq=1 * u.GHz) == 0.0
    assert std.reflection_coefficient(1 * u.GHz) == -1j


def test_s1p_freq(cal_data):

    vna = s11.VNAReading.from_s1p(cal_data / "S11/ReceiverReading01/Match01.s1p")
    assert vna.freq.n > 0
    assert np.all(vna.freq.freq > 20.0 * u.MHz)
    assert np.all(vna.freq.freq < 210.0 * u.MHz)


def test_receiver(cal_data):
    ioobj = io.S11Dir(cal_data / "S11")

    rcv = s11.Receiver.from_io(
        device=ioobj.receiver_reading[0], f_low=50 * u.MHz, f_high=100 * u.MHz
    )
    assert rcv.n_terms == 37
    assert np.iscomplexobj(rcv.raw_s11)
    assert np.all(np.abs(rcv.raw_s11) < 1)
    assert len(np.unique(rcv.raw_s11)) > 25

    s11mdl = rcv.s11_model(rcv.freq.freq.to_value("MHz"))
    assert np.iscomplexobj(s11mdl)
    assert np.all(np.abs(s11mdl) <= 1)
    assert len(np.unique(s11mdl)) > 25


def test_even_nterms_s11(cal_data):
    fl = io.S11Dir(cal_data / "S11").receiver_reading

    with pytest.raises(ValueError, match="n_terms must be odd"):
        s11.Receiver.from_io(fl, n_terms=40)


def test_s1p_converter(io_obs):
    s1p = io_obs.s11.ambient[0].match

    assert s11._s1p_converter(s1p.path, check=True) == s1p
    assert s11._s1p_converter(s1p) == s1p

    with pytest.raises(TypeError, match="s1p must be a path"):
        s11._s1p_converter(3)

    assert np.allclose(s11.VNAReading.from_s1p(s1p).s11, s1p.s11)


def test_tuplify():
    assert s11._tuplify((3, 4, 5, 3)) == (3, 4, 5, 3)
    assert s11._tuplify((3.0, 4.0)) == (3, 4)
    assert s11._tuplify(3) == (3, 3, 3) == s11._tuplify(3.0)

    with pytest.raises(ValueError):
        s11._tuplify("hey")


def test_bad_s11_input_to_vna():
    with pytest.raises(ValueError, match="s11 must be a complex quantity"):
        s11.VNAReading(
            freq=np.linspace(50, 100, 100) * u.MHz, s11=np.linspace(0, 1, 100)
        )

    with pytest.raises(ValueError, match="freq and s11 must have the same length"):
        s11.VNAReading(
            freq=np.linspace(50, 100, 100) * u.MHz, s11=np.linspace(0, 1, 70) + 0j
        )


def test_different_freqs_in_standards():
    freq = np.linspace(50, 100, 100) * u.MHz
    s = np.linspace(0, 1, 100) + 0j

    vna1 = s11.VNAReading(freq=freq, s11=s)
    vna2 = s11.VNAReading(freq=freq[:80], s11=s[:80])

    with pytest.raises(
        ValueError, match="short standard does not have same frequencies"
    ):
        s11.StandardsReadings(open=vna1, short=vna2, match=vna1)

    with pytest.raises(
        ValueError, match="match standard does not have same frequencies"
    ):
        s11.StandardsReadings(open=vna1, short=vna1, match=vna2)

    sr = s11.StandardsReadings(open=vna1, short=vna1, match=vna1)
    assert sr.freq == vna1.freq


def test_init_internal_switch():
    s = np.linspace(0, 1, 100) + 0j
    freq = np.linspace(50, 100, 100) * u.MHz

    with pytest.raises(TypeError, match="n_terms must be an integer or tuple of three"):
        s11.InternalSwitch(
            s11_data=s,
            s12_data=s,
            s22_data=s,
            freq=freq,
            n_terms=(5, 5, 5, 5),
        )


def test_receiver_clone(calobs):
    assert calobs.receiver.clone() == calobs.receiver
    ck = rc.get_calkit(rc.AGILENT_85033E, resistance_of_match=49.0)
    new = calobs.receiver.with_new_calkit(ck)

    assert not np.allclose(calobs.receiver.raw_s11, new.raw_s11)
