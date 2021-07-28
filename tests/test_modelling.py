import pytest

import numpy as np
import yaml
from typing import Type

from edges_cal import modelling as mdl


def test_pass_params():
    pl = mdl.PhysicalLin(parameters=[1, 2, 3])

    epl = pl.at(x=np.linspace(0, 1, 10))

    assert epl.n_terms == 3

    with pytest.raises(ValueError):
        mdl.PhysicalLin(parameters=[1, 2, 3], n_terms=4)


def test_bad_get_mdl():
    with pytest.raises(ValueError):
        mdl.get_mdl(3)


@pytest.mark.parametrize(
    "model",
    [mdl.PhysicalLin, mdl.Polynomial, mdl.EdgesPoly, mdl.Fourier, mdl.FourierDay],
)
def test_basis(model: Type[mdl.Model]):
    x = np.linspace(0, 1, 10)
    pl = model(parameters=[1, 2, 3]).at(x=x)

    assert pl.basis.shape == (3, 10)
    assert pl().shape == (10,)
    assert pl(x=np.linspace(0, 1, 20)).shape == (20,)

    pl2 = model(n_terms=4).at(x=x)
    with pytest.raises(ValueError):
        pl2()

    assert pl2(parameters=[1, 2, 3]).shape == (10,)

    with pytest.raises(ValueError):
        pl2(parameters=[1, 2, 3, 4, 5])


def test_model_fit():
    pl = mdl.PhysicalLin()
    fit = mdl.ModelFit(pl.at(x=np.linspace(0, 1, 10)), ydata=np.linspace(0, 1, 10),)
    assert isinstance(fit.model.model, mdl.PhysicalLin)


def test_simple_fit():
    pl = mdl.PhysicalLin(parameters=[1, 2, 3])
    model = pl.at(x=np.linspace(50, 100, 10))

    data = model()
    print(data)
    fit = mdl.ModelFit(model, ydata=data)

    assert np.allclose(fit.model_parameters, [1, 2, 3])
    assert np.allclose(fit.residual, 0)
    assert np.allclose(fit.weighted_chi2, 0)
    assert np.allclose(fit.reduced_weighted_chi2(), 0)


def test_weighted_fit():
    np.random.seed(1234)
    four = mdl.Fourier(parameters=[1, 2, 3])
    model = four.at(x=np.linspace(50, 100, 10))

    sigmas = np.abs(model() / 100)
    data = model() + np.random.normal(scale=sigmas)

    fit = mdl.ModelFit(model, ydata=data, weights=1 / sigmas)

    assert np.allclose(fit.model_parameters, [1, 2, 3], rtol=0.05)


def test_wrong_params():
    with pytest.raises(ValueError):
        mdl.Polynomial(n_terms=5, parameters=(1, 2, 3, 4, 5, 6))


def test_no_nterms():
    with pytest.raises(ValueError):
        mdl.Polynomial()


def test_model_fit_intrinsic():
    m = mdl.Polynomial(n_terms=2).at(x=np.linspace(0, 1, 10))
    fit = m.fit(ydata=np.linspace(0, 1, 10))
    assert np.allclose(fit.evaluate(), fit.ydata)


def test_physical_lin():
    m = mdl.PhysicalLin(f_center=1, n_terms=5).at(x=np.array([1 / np.e, 1, np.e]))

    basis = m.basis
    assert np.allclose(basis[0], [np.e ** 2.5, 1, np.e ** -2.5])
    assert np.allclose(basis[1], [-np.e ** 2.5, 0, np.e ** -2.5])
    assert np.allclose(basis[2], [np.e ** 2.5, 0, np.e ** -2.5])
    assert np.allclose(basis[3], [np.e ** 4.5, 1, np.e ** -4.5])
    assert np.allclose(basis[4], [np.e ** 2, 1, np.e ** -2])


def test_linlog():
    m = mdl.LinLog(f_center=1, n_terms=3).at(x=np.array([0.5, 1, 2]))
    assert m.basis.shape == (3, 3)


@pytest.mark.xfail(reason="Hasn't been updated to new API.")
def test_noise_waves():
    nw = mdl.NoiseWaves(
        freq=np.linspace(50, 100, 100), gamma_coeffs=(np.ones((4, 100)),) * 3
    )

    p = [
        1,
        0,
        0,
        0,
        0,
        0,  # For T_load
        1,
        0,
        0,
        0,
        0,
        0,  # For T_unc
        1,
        0,
        0,
        0,
        0,
        0,  # For T_cos
        1,
        0,
        0,
        0,
        0,
        0,  # For T_sin
    ]

    poly = mdl.Polynomial(n_terms=6, default_x=nw.freq.freq)

    np.testing.assert_allclose(nw.t_load(p[:6]), poly(parameters=p[:6]))
    np.testing.assert_allclose(nw.t_load(p), poly(parameters=p[:6]))

    np.testing.assert_allclose(nw.t_unc(p[6:12]), poly(parameters=p[6:12]))
    np.testing.assert_allclose(nw.t_unc(p), poly(parameters=p[6:12]))

    np.testing.assert_allclose(nw.t_cos(p[12:18]), poly(parameters=p[12:18]))
    np.testing.assert_allclose(nw.t_cos(p), poly(parameters=p[12:18]))

    np.testing.assert_allclose(nw.t_sin(p[18:24]), poly(parameters=p[18:24]))
    np.testing.assert_allclose(nw.t_sin(p), poly(parameters=p[18:24]))

    np.testing.assert_allclose(nw.default_basis[0], -1)
    assert np.all(nw.default_basis[6] > 0)


@pytest.mark.xfail(reason="Hasn't been updated to new API.")
def test_noise_waves_with_fg():
    nw = mdl.NoiseWaves(
        freq=np.linspace(50, 100, 100),
        gamma_coeffs=(np.ones((5, 100)),) * 3,
        fg_terms=5,
    )

    p = [
        1,
        0,
        0,
        0,
        0,
        0,  # For T_load
        1,
        0,
        0,
        0,
        0,
        0,  # For T_unc
        1,
        0,
        0,
        0,
        0,
        0,  # For T_cos
        1,
        0,
        0,
        0,
        0,
        0,  # For T_sin
        1,
        0,
        0,
        0,
        0,  # For T_fg
    ]

    np.testing.assert_allclose(
        nw.t_fg(p[24:]),
        mdl.LinLog(n_terms=5, default_x=nw.freq.freq)(parameters=p[24:]),
    )
    np.testing.assert_allclose(
        nw.t_fg(p), mdl.LinLog(n_terms=5, default_x=nw.freq.freq)(parameters=p[24:])
    )


def test_yaml_roundtrip():
    p = mdl.Polynomial(n_terms=5)
    s = yaml.dump(p)
    pp = yaml.load(s)
    assert p == pp
    assert "!Model" in s
