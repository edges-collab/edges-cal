import pytest

import numpy as np
from astropy import units as u

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
    assert ag.gamma_offset(freq).unit == u.dimensionless_unscaled
    assert ag.reflection_coefficient(freq).unit == u.dimensionless_unscaled
