# -*- coding: utf-8 -*-
"""Functions for generating least-squares model fits for linear models."""
from __future__ import annotations

import numpy as np
from abc import abstractmethod
from cached_property import cached_property
from copy import deepcopy
from typing import Optional, Sequence, Type, Union

F_CENTER = 75.0


class Model:
    _models = {}
    n_terms = None

    def __init__(
        self,
        parameters: [None, Sequence] = None,
        n_terms: [int, None] = None,
        default_x: np.ndarray = None,
    ):
        """
        A base class for a linear model.

        Parameters
        ----------
        parameters : list of float, optional
            If provided, a set of parameters at which to evaluate the model.
        n_terms : int, optional
            The number of terms the model has (useful for models with an arbitrary
            number of terms).
        default_x : np.ndarray, optional
            A set of default co-ordinates at which to evaluate the model.

        Raises
        ------
        ValueError
            If number of parameters is not consistent with n_terms.
        """
        if n_terms:
            if self.n_terms and n_terms != self.n_terms:
                raise ValueError(f"n_terms must be {self.n_terms}")

            self.n_terms = n_terms

        if parameters:
            self.parameters = list(parameters)
            if self.n_terms and len(self.parameters) != self.n_terms:
                raise ValueError(
                    f"wrong number of parameters! Should be {self.n_terms}."
                )
            self.n_terms = len(self.parameters)
        else:
            self.parameters = None

        if not self.n_terms:
            raise ValueError("Need to supply either parameters or n_terms!")

        self.default_x = default_x
        self.__basis_terms = {}

    def __init_subclass__(cls, is_meta=False, **kwargs):
        """Initialize a subclass and add it to the registered models."""
        super().__init_subclass__(**kwargs)
        if not is_meta:
            cls._models[cls.__name__.lower()] = cls

    @property
    def default_basis(self) -> [None, np.ndarray]:
        """The (cached) basis functions at default_x.

        If it exists, a 2D array shape (n_terms, x).
        """
        try:
            return self.__default_basis
        except AttributeError:
            self.__default_basis = (
                self.get_basis(self.default_x) if self.default_x is not None else None
            )
            return self.__default_basis

    @default_basis.setter
    def default_basis(self, val):
        assert isinstance(val, np.ndarray)
        assert val.ndim == 2
        self.__default_basis = val

    @default_basis.deleter
    def default_basis(self):
        del self.__default_basis
        self.__basis_terms = {}

    def get_basis(self, x: np.ndarray, indices: [None, list] = None) -> np.ndarray:
        """Obtain the basis functions.

        Parameters
        ----------
        x : np.ndarray
            Co-ordinates at which to evaluate the basis functions.

        Returns
        -------
        basis : np.ndarray
            A 2D array, shape ``(n_terms, len(x))`` with the computed basis functions
            for each term.
        """
        if indices is None:
            indices = list(range(self.n_terms))

        if len(indices) > self.n_terms:
            raise ValueError("Cannot get more indices than n_terms.")

        return np.array([self.get_basis_term(indx, x) for indx in indices])

    def update_nterms(self, n_terms: int):
        """Update the number of terms in the model.

        This does it more quickly, without too many repeated calculations.
        """
        # Important: get the default basis upfront, before changing n_terms.
        # If it was never created yet, or has been deleted, it will get appended to.
        # We don't want to append to the wrong thing.
        db = self.default_basis
        old_terms = self.n_terms * 1
        self.n_terms = n_terms
        self.parameters = None

        if self.default_x is None or n_terms == old_terms:
            pass
        elif n_terms < old_terms:
            self.default_basis = db[:n_terms]
        else:
            self.default_basis = np.vstack(
                (db, self.get_basis(self.default_x, list(range(old_terms, n_terms))),)
            )

        return

    def get_basis_term(self, indx: int, x: Optional[np.ndarray] = None):
        """Get a specific basis function term."""
        # If using a new passed-in x, don't cache.
        if x is not None:
            return self._get_basis_term(indx, x)

        if indx not in self.__basis_terms:
            self.__basis_terms[indx] = self._get_basis_term(indx, self.default_x)

        return self.__basis_terms[indx]

    def __call__(
        self,
        x: [np.ndarray, None] = None,
        basis: [np.ndarray, None] = None,
        parameters: [np.ndarray, list, None] = None,
    ) -> np.ndarray:
        """Evaluate the model.

        Parameters
        ----------
        x : np.ndarray, optional
            The co-ordinates at which to evaluate the model (by default, use ``default_x``).
        basis : np.ndarray, optional
            The basis functions at which to evaluate the model. This is useful if calling
            the model multiple times, as the basis itself can be cached and re-used.
        parameters :
            A list/array of parameters at which to evaluate the model. Will use the
            instance's parameters if available.

        Returns
        -------
        model : np.ndarray
            The model evaluated at the input ``x`` or ``basis``.
        """
        if parameters is None:
            parameters = self.parameters

        if parameters is None:
            raise ValueError("You must supply parameters to evaluate the model!")

        if x is None and basis is None and self.default_basis is None:
            raise ValueError("You need to provide either 'x' or 'basis'.")
        elif x is None and basis is None:
            basis = self.default_basis
        elif x is not None:
            basis = self.get_basis(x)

        if len(parameters) != len(basis):
            raise ValueError(
                f"number of parameters ({len(parameters)}) does not match "
                f"the number of basis terms ({len(basis)})."
            )

        return np.dot(parameters, basis)

    @abstractmethod
    def _get_basis_term(self, indx: int, x: np.ndarray) -> np.ndarray:
        pass

    def fit(
        self, ydata: np.ndarray, weights: [None, np.ndarray, float] = None, xdata=None
    ):
        """Create a linear-regression fit object."""
        return ModelFit(
            self,
            ydata=ydata,
            xdata=xdata if xdata is not None else None,
            weights=weights,
        )

    @staticmethod
    def get_mdl(name) -> Type[Model]:
        """Get a specific linear model from a string name."""
        return Model._models[name.lower()]


class Foreground(Model, is_meta=True):
    def __init__(
        self,
        parameters=None,
        f_center: float = F_CENTER,
        with_cmb: bool = False,
        **kwargs,
    ):
        """
        Base class for Foreground models.

        Parameters
        ----------
        parameters : list of float, optional
            If provided, a set of parameters at which to evaluate the model.
        f_center : float
            A "center" or "reference" frequency. Typically models will have their
            co-ordindates divided by this frequency before solving for the co-efficients.
        with_cmb : bool
            Whether to add a simply CMB component to the foreground.
        kwargs
            All other arguments passed through to :class:`Model`.
        """
        super().__init__(parameters, **kwargs)
        self.f_center = f_center
        self.with_cmb = with_cmb

    def __call__(
        self,
        x: [np.ndarray, None] = None,
        basis: [np.ndarray, None] = None,
        parameters: [np.ndarray, list, None] = None,
    ) -> np.ndarray:
        """Evaluate the model.

        Parameters
        ----------
        x : np.ndarray, optional
            The co-ordinates at which to evaluate the model (by default, use ``default_x``).
        basis : np.ndarray, optional
            The basis functions at which to evaluate the model. This is useful if calling
            the model multiple times, as the basis itself can be cached and re-used.
        parameters :
            A list/array of parameters at which to evaluate the model. Will use the
            instance's parameters if available.

        Returns
        -------
        model : np.ndarray
            The model evaluated at the input ``x`` or ``basis``.
        """
        t = 2.725 if self.with_cmb else 0
        return t + super().__call__(x=x, basis=basis, parameters=parameters)


class PhysicalLin(Foreground):
    """Foreground model using a linearized physical model of the foregrounds."""

    def _get_basis_term(self, indx: int, x: np.ndarray) -> np.ndarray:
        y = x / self.f_center
        if indx < 3:
            logy = np.log(y)
            y25 = y ** -2.5
            return y25 * logy ** indx

        elif indx == 3:
            return y ** -4.5
        elif indx == 4:
            return 1 / (y * y)
        else:
            raise ValueError("too many terms supplied!")


class Polynomial(Foreground):
    def __init__(self, log_x: bool = False, offset: float = 0, **kwargs):
        r"""A polynomial foreground model.

        Parameters
        ----------
        log_x : bool
            Whether to fit the poly coefficients with log-space co-ordinates.
        offset : float
            An offset to use for each index in the polynomial model.
        kwargs
            All other arguments passed through to :class:`Foreground`.

        Notes
        -----
        The polynomial model can be written

        .. math:: \sum_{i=0}^{n} c_i y^{i + offset},

        where ``y`` is ``log(x)`` if ``log_x=True`` and simply ``x`` otherwise.
        """
        super().__init__(**kwargs)
        self.log_x = log_x
        self.offset = offset

    def _get_basis_term(self, indx: int, x: np.ndarray) -> np.ndarray:
        y = x / self.f_center
        if self.log_x:
            y = np.log(y)

        return y ** (indx + self.offset)


class EdgesPoly(Polynomial):
    def __init__(self, offset: float = -2.5, **kwargs):
        """
        Polynomial foregrounds with an offset corresponding to approximate galaxy spectral index.

        Parameters
        ----------
        offset : float
            The offset to use. Default is close to the Galactic spectral index.
        kwargs
            All other arguments are passed through to :class:`Polynomial`.
        """
        super().__init__(offset=offset, **kwargs)


class LinLog(Polynomial):
    def __init__(self, beta: float = -2.5, **kwargs):
        self.beta = beta
        kwargs["log_x"] = True
        kwargs["offset"] = 0

        super().__init__(**kwargs)

    def _get_basis_term(self, indx: int, x: np.ndarray) -> np.ndarray:
        term = super()._get_basis_term(indx, x)
        return term * (x / self.f_center) ** self.beta


class Fourier(Model):
    """A Fourier-basis model."""

    def _get_basis_term(self, indx: int, x: np.ndarray) -> np.ndarray:
        if indx == 0:
            return np.ones_like(x)
        elif indx % 2:
            return np.cos((indx + 1) // 2 * x)
        else:
            return np.sin((indx + 1) // 2 * x)


class ModelFit:
    def __init__(
        self,
        model_type: [str, Type[Model], Model],
        *,
        ydata: np.ndarray,
        xdata: [None, np.ndarray] = None,
        weights: [None, np.ndarray] = None,
        n_terms: int = None,
        **kwargs,
    ):
        """A class representing a fit of model to data.

        Parameters
        ----------
        model_type
            The type of model to fit to the data.
        xdata
            The co-ordinates of the measured data.
        ydata
            The values of the measured data.
        weights
            The weight of the measured data at each point. This corresponds to the
            *variance* of the measurement (not the standard deviation). This is appropriate
            if the weights represent the number of measurements going into each piece
            of data.
        n_terms
            The number of terms to use in the model (useful for models with an
            arbitrary number of terms).
        kwargs
            All other arguments are passed to the chosen model.

        Raises
        ------
        ValueError
            If model_type is not str, or a subclass of :class:`Model`.
        """
        if not isinstance(model_type, Model) and xdata is None:
            raise ValueError(
                "You must pass xdata unless the model_type is an instance."
            )

        if isinstance(model_type, str):
            self.model = Model._models[model_type.lower()](
                default_x=xdata, n_terms=n_terms, **kwargs
            )
        elif np.issubclass_(model_type, Model):
            self.model = model_type(default_x=xdata, n_terms=n_terms, **kwargs)
        elif isinstance(model_type, Model):
            self.model = model_type
            if xdata is not None:
                self.model.default_x = xdata
        else:
            raise ValueError(
                "model_type must be str, Model subclass or Model instance."
            )

        self.xdata = self.model.default_x
        self.ydata = ydata
        self.weights = weights

        if weights is None:
            self.weights = 1
        else:
            # if a vector is given
            assert weights.shape == self.xdata.shape
            self.weights = weights

        self.n_terms = self.model.n_terms
        self.degrees_of_freedom = self.xdata.size - self.n_terms - 1

    @cached_property
    def fit(self) -> Model:
        """The model fit."""
        if np.isscalar(self.weights):
            pars = self._ls(self.model.default_basis, self.ydata)
        else:
            pars = self._wls(self.model.default_basis, self.ydata, w=self.weights)
        self.model.parameters = pars
        return self.model

    def _wls(self, van, y, w):
        """Ripped straight outta numpy for speed.

        Note: this function is written purely for speed, and is intended to *not*
        be highly generic. Don't replace this by statsmodels or even np.polyfit. They
        are significantly slower (>4x for statsmodels, 1.5x for polyfit).
        """
        # set up the least squares matrices and apply weights.
        # Don't use inplace operations as they
        # can cause problems with NA.
        mask = w > 0

        lhs = van[:, mask] * w[mask]
        rhs = y[mask] * w[mask]

        rcond = y.size * np.finfo(y.dtype).eps

        # Determine the norms of the design matrix columns.
        scl = np.sqrt(np.square(lhs).sum(1))
        scl[scl == 0] = 1

        # Solve the least squares problem.
        c, resids, rank, s = np.linalg.lstsq((lhs.T / scl), rhs.T, rcond)
        c = (c.T / scl).T

        return c

    def _ls(self, van, y):
        """Ripped straight outta numpy for speed.

        Note: this function is written purely for speed, and is intended to *not*
        be highly generic. Don't replace this by statsmodels or even np.polyfit. They
        are significantly slower (>4x for statsmodels, 1.5x for polyfit).
        """
        rcond = y.size * np.finfo(y.dtype).eps

        # Determine the norms of the design matrix columns.
        scl = np.sqrt(np.square(van).sum(1))

        # Solve the least squares problem.
        return np.linalg.lstsq((van.T / scl), y.T, rcond)[0] / scl

    @cached_property
    def model_parameters(self):
        """The best-fit model parameters."""
        # Parameters need to be copied into this object, otherwise a new fit on the
        # parent model will change the model_parameters of this fit!
        return self.fit.parameters.copy()

    def evaluate(self, x: [np.ndarray, None] = None) -> np.ndarray:
        """Evaluate the best-fit model.

        Parameters
        ----------
        x : np.ndarray, optional
            The co-ordinates at which to evaluate the model. By default, use the input
            data co-ordinates.

        Returns
        -------
        y : np.ndarray
            The best-fit model evaluated at ``x``.
        """
        return self.model(x, parameters=self.model_parameters)

    @cached_property
    def residual(self) -> np.ndarray:
        """Residuals of data to model."""
        return self.ydata - self.evaluate()

    @cached_property
    def weighted_chi2(self) -> float:
        """The chi^2 of the weighted fit."""
        return np.dot(self.residual.T, self.weights * self.residual)

    def reduced_weighted_chi2(self) -> float:
        """The weighted chi^2 divided by the degrees of freedom."""
        return (1 / self.degrees_of_freedom) * self.weighted_chi2

    def weighted_rms(self) -> float:
        """The weighted root-mean-square of the residuals."""
        return np.sqrt(self.weighted_chi2) / np.sum(self.weights)

    def get_covariance(self) -> np.ndarray:
        """The covariance of the parameter estimates at the solution."""
        return self.fit.normalized_cov_params

    def reset(self):
        """Resets the fit."""
        del self.residual
        del self.weighted_chi2
        del self.model_parameters
        del self.fit
