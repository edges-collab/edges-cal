import pytest

import numpy as np

from edges_cal import modelling as mdl


def test_pass_params():
    pl = mdl.PhysicalLin(parameters=[1, 2, 3])

    assert pl.n_terms == 3

    with pytest.raises(ValueError):
        mdl.PhysicalLin(parameters=[1, 2, 3], n_terms=4)


@pytest.mark.parametrize(
    "model", [mdl.PhysicalLin, mdl.Polynomial, mdl.EdgesPoly, mdl.Fourier]
)
def test_basis(model: mdl.Model):
    pl = model(parameters=[1, 2, 3], default_x=np.linspace(0, 1, 10))
    assert pl.default_basis.shape == (3, 10)
    assert pl().shape == (10,)
    assert pl(x=np.linspace(0, 1, 20)).shape == (20,)

    pl2 = model(n_terms=4)
    with pytest.raises(ValueError):
        pl2()

    with pytest.raises(ValueError):
        pl2(parameters=[1, 2, 3, 4])

    with pytest.raises(ValueError):
        pl2(parameters=[1, 2, 3], x=np.linspace(0, 1, 10))


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
