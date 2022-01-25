import pytest

import numpy as np
from astropy import units as u

from edges_cal import modelling as mdl
from edges_cal import reflection_coefficient as rc


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
    assert std.lossy_characteristic_impedance(freq=1 * u.GHz) == 50 * u.Ohm + (1 - 1j)
    assert std.gl(freq=1 * u.GHz) == 5e-2j + (1 + 1j) * 1e-3
    assert std.reflection_coefficient(150 * u.MHz) == 0.0


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
