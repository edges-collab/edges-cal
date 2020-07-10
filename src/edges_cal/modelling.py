# -*- coding: utf-8 -*-
"""
Functions for generating least-squares model fits for linear models(and evaluating the models).
"""
import numpy as np
import scipy as sp
from abc import abstractmethod
from cached_property import cached_property
from typing import Sequence, Type

F_CENTER = 75.0


class Model:
    _models = {}
    n_terms = None

    def __init__(
        self,
        parameters: [None, Sequence] = None,
        n_terms: [int, None] = None,
        default_x=None,
    ):
        if parameters:
            self.parameters = list(parameters)
            if self.n_terms and len(self.parameters) != self.n_terms:
                raise ValueError(
                    f"wrong number of parameters! Should be {self.n_terms}."
                )
            self.n_terms = len(self.parameters)
        else:
            self.parameters = None

        if n_terms:
            if self.n_terms and n_terms != self.n_terms:
                raise ValueError(f"n_terms must be {self.n_terms}")

            self.n_terms = n_terms

        if not self.n_terms:
            raise ValueError("Need to supply either parameters or n_terms!")

        self.default_x = default_x

    def __init_subclass__(cls, is_meta=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not is_meta:
            cls._models[cls.__name__.lower()] = cls

    @cached_property
    def default_basis(self):
        return self.get_basis(self.default_x) if self.default_x is not None else None

    def get_basis(self, x):
        out = np.zeros((self.n_terms, len(x)))
        self._fill_basis_terms(x, out)
        return out

    def __call__(self, x=None, basis=None):
        if x is None and basis is None and self.default_basis is None:
            raise ValueError("You need to provide either 'x' or 'basis'.")
        elif x is None and basis is None:
            basis = self.default_basis
        elif x is not None:
            basis = self.get_basis(x)

        return np.dot(self.parameters, basis)

    @abstractmethod
    def _fill_basis_terms(self, x, out):
        pass


class Foreground(Model, is_meta=True):
    def __init__(self, parameters=None, f_center=F_CENTER, with_cmb=False, **kwargs):
        super().__init__(parameters, **kwargs)
        self.f_center = f_center
        self.with_cmb = with_cmb

    def __call__(self, x):
        t = 2.725 if self.with_cmb else 0
        return t + super().__call__(x)


class PhysicalLin(Foreground):
    def _fill_basis_terms(self, x, out):
        y = x / self.f_center

        out[0] = y ** -2.5
        out[1] = y ** -2.5 * np.log(y)
        out[2] = y ** -2.5 * np.log(y) ** 2

        if self.n_terms >= 4:
            out[3] = y ** -4.5
            if self.n_terms == 5:
                out[4] = y ** -2
            if self.n_terms > 5:
                raise ValueError("too many terms supplied!")


class Polynomial(Foreground):
    def __init__(self, log_x=False, offset=0, **kwargs):
        super().__init__(**kwargs)
        self.log_x = log_x
        self.offset = offset

    def _fill_basis_terms(self, x, out):
        y = x / self.f_center
        if self.log_x:
            y = np.log(y)

        for i in range(self.n_terms):
            out[i] = y ** (i + self.offset)


class EdgesPoly(Polynomial):
    def __init__(self, offset=-2.5, **kwargs):
        super().__init__(offset=offset, **kwargs)


class Fourier(Model):
    def _fill_basis_terms(self, x, out):
        out[0] = np.ones_like(x)

        for i in range((self.n_terms - 1) // 2):
            out[2 * i + 1] = np.cos((i + 1) * x)
            out[2 * i + 2] = np.sin((i + 1) * x)


class ModelFit:
    def __init__(
        self,
        model_type: [str, Type[Model], Model],
        xdata: np.ndarray,
        ydata: np.ndarray,
        weights: [None, np.ndarray] = None,
        n_terms: int = None,
        **kwargs,
    ):
        if isinstance(model_type, str):
            self.model = Model._models[model_type.lower()](
                default_x=xdata, n_terms=n_terms, **kwargs
            )
        elif issubclass(model_type, Model):
            self.model = model_type(default_x=xdata, n_terms=n_terms, **kwargs)
        elif isinstance(model_type, Model):
            self.model = model_type
            self.model.default_x = xdata
        else:
            raise ValueError(
                "model_type must be str, Model subclass or Model instance."
            )

        self.xdata = xdata
        self.ydata = ydata
        self.weights = weights

        if weights is None:
            self.weights = np.eye(len(self.xdata))
        elif weights.ndim == 1:
            # if a vector is given
            assert weights.shape == self.xdata.shape
            self.weights = np.diag(weights)
        elif weights.ndim == 2:
            assert weights.shape == (len(self.xdata), len(self.xdata))
            self.weights = weights

        self.n_terms = self.model.n_terms

        self.degrees_of_freedom = len(self.xdata) - self.n_terms - 1

    @cached_property
    def model_parameters(self):
        return self.get_model_params()

    @cached_property
    def qr(self):
        AT = self.model.default_basis

        # sqrt of weight matrix
        sqrt_w = np.sqrt(self.weights)

        # A and ydata "tilde"
        WA = np.dot(sqrt_w, AT.T)

        # solving system using 'short' QR decomposition (see R. Butt, Num. Anal. Using MATLAB)
        q, r = sp.linalg.qr(WA, mode="economic")
        return q, r

    def get_model_params(self):
        # transposing matrices so data is along columns
        ydata = np.reshape(self.ydata, (-1, 1))

        Wydata = np.dot(np.sqrt(self.weights), ydata)
        q, r = self.qr
        return sp.linalg.solve(r, np.dot(q.T, Wydata)).flatten()

    def evaluate(self, x=None):
        # Set the parameters on the underlying object (solves for them if not solved yet)
        self.model.parameters = list(self.model_parameters)
        if x is None:
            x = self.xdata
        return self.model(x)

    @cached_property
    def residual(self):
        return self.ydata - self.evaluate()

    @cached_property
    def weighted_chi2(self):
        return np.dot(self.residual.T, np.dot(self.weights, self.residual))

    def reduced_weighted_chi2(self):
        return (1 / self.degrees_of_freedom) * self.weighted_chi2

    def weighted_rms(self):
        return np.sqrt(self.weighted_chi2) / np.sum(np.diag(self.weights))

    def get_covariance(self):
        r = self.qr[1]
        inv_pre_cov = np.linalg.inv(np.dot(r.T, r))
        return self.reduced_weighted_chi2() * inv_pre_cov
