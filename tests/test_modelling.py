import pytest

import numpy as np
from typing import Type

from edges_cal import modelling as mdl


def test_pass_params():
    pl = mdl.PhysicalLin(parameters=[1, 2, 3])

    assert pl.n_terms == 3

    with pytest.raises(ValueError):
        mdl.PhysicalLin(parameters=[1, 2, 3], n_terms=4)


@pytest.mark.parametrize(
    "model", [mdl.PhysicalLin, mdl.Polynomial, mdl.EdgesPoly, mdl.Fourier]
)
def test_basis(model: Type[mdl.Model]):
    pl = model(parameters=[1, 2, 3], default_x=np.linspace(0, 1, 10))
    assert pl.default_basis.shape == (3, 10)
    assert pl().shape == (10,)
    assert pl(x=np.linspace(0, 1, 20)).shape == (20,)

    pl2 = model(n_terms=4)
    with pytest.raises(ValueError):
        pl2()

    with pytest.raises(ValueError):
        pl2(parameters=[1, 2, 3, 4])

    assert pl2(parameters=[1, 2, 3], x=np.linspace(0, 1, 10)).shape == (10,)


def test_cached_basis():
    pl = mdl.PhysicalLin(parameters=[1, 2, 3], default_x=np.linspace(0, 1, 10))

    df_basis = pl.default_basis.copy()

    assert np.all(pl.get_basis_term(0) == df_basis[0])
    del pl.default_basis

    assert np.all(pl.get_basis_term(1) == df_basis[1])


def test_model_fit():

    fit = mdl.ModelFit(
        mdl.PhysicalLin,
        xdata=np.linspace(0, 1, 10),
        ydata=np.linspace(0, 1, 10),
        n_terms=3,
    )
    assert isinstance(fit.model, mdl.PhysicalLin)

    model = mdl.PhysicalLin(n_terms=3)
    fit = mdl.ModelFit(model, xdata=np.linspace(0, 1, 10), ydata=np.linspace(0, 1, 10),)
    assert isinstance(fit.model, mdl.PhysicalLin)

    with pytest.raises(ValueError):
        mdl.ModelFit(
            3, xdata=np.linspace(0, 1, 10), ydata=np.linspace(0, 1, 10), n_terms=3
        )


def test_simple_fit():
    model = mdl.PhysicalLin(parameters=[1, 2, 3], default_x=np.linspace(50, 100, 10))

    data = model()
    print(data)
    fit = mdl.ModelFit(model, xdata=model.default_x, ydata=data)

    assert np.allclose(fit.model_parameters, [1, 2, 3])

    assert np.allclose(fit.residual, 0)
    assert np.allclose(fit.weighted_chi2, 0)
    assert np.allclose(fit.reduced_weighted_chi2(), 0)


def test_weighted_fit():
    np.random.seed(1234)
    model = mdl.Fourier(parameters=[1, 2, 3], default_x=np.linspace(50, 100, 10))

    sigmas = np.abs(model() / 100)
    data = model() + np.random.normal(scale=sigmas)

    fit = mdl.ModelFit(model, xdata=model.default_x, ydata=data, weights=1 / sigmas)

    assert np.allclose(fit.model_parameters, [1, 2, 3], rtol=0.05)


def test_wrong_params():
    with pytest.raises(ValueError):
        mdl.Polynomial(n_terms=5, parameters=(1, 2, 3, 4, 5, 6))


def test_no_nterms():
    with pytest.raises(ValueError):
        mdl.Polynomial()


def test_del_default_basis():
    m = mdl.Polynomial(n_terms=3, default_x=np.linspace(0, 1, 10))
    assert m.default_basis.shape == (3, 10)
    del m.default_basis
    m.update_nterms(4)
    assert m.default_basis.shape == (4, 10)
    m.update_nterms(2)
    assert m.default_basis.shape == (2, 10)
    m.update_nterms(2)
    assert m.default_basis.shape == (2, 10)


def test_get_bad_indx():
    m = mdl.Polynomial(n_terms=3)

    with pytest.raises(ValueError):
        m.get_basis(x=np.linspace(0, 1, 10), indices=list(range(4)))


def test_model_fit_intrinsic():
    m = mdl.Polynomial(n_terms=2, default_x=np.linspace(0, 1, 10))
    fit = m.fit(ydata=np.linspace(0, 1, 10))
    assert np.allclose(fit.evaluate(m.default_x), fit.ydata)


def test_physical_lin():
    m = mdl.PhysicalLin(f_center=1, n_terms=5, default_x=np.array([1 / np.e, 1, np.e]))

    basis = m.default_basis
    assert np.allclose(basis[0], [np.e ** 2.5, 1, np.e ** -2.5])
    assert np.allclose(basis[1], [-np.e ** 2.5, 0, np.e ** -2.5])
    assert np.allclose(basis[2], [np.e ** 2.5, 0, np.e ** -2.5])
    assert np.allclose(basis[3], [np.e ** 4.5, 1, np.e ** -4.5])
    assert np.allclose(basis[4], [np.e ** 2, 1, np.e ** -2])


def test_linlog():
    m = mdl.LinLog(f_center=1, n_terms=3, default_x=np.array([0.5, 1, 2]))
    assert m.default_basis.shape == (3, 3)


def test_bad_xdata():
    with pytest.raises(ValueError):
        mdl.ModelFit(model_type="linlog", ydata=np.linspace(0, 1, 10))
