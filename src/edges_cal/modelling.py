# -*- coding: utf-8 -*-
"""Functions for generating least-squares model fits for linear models."""
import numpy as np
import scipy as sp
from abc import abstractmethod
from cached_property import cached_property
from typing import Sequence, Tuple, Type, Union

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
        """Initialize a subclass and add it to the registered models."""
        super().__init_subclass__(**kwargs)
        if not is_meta:
            cls._models[cls.__name__.lower()] = cls

    @cached_property
    def default_basis(self) -> Union[np.ndarray, None]:
        """Basis functions defined at the default parameters and co-ordinates."""
        return self.get_basis(self.default_x) if self.default_x is not None else None

    def get_basis(self, x: np.ndarray) -> np.ndarray:
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
        n_terms = (
            self.n_terms if isinstance(self.n_terms, int) else np.prod(self.n_terms)
        )
        x = np.array(x)
        out = np.zeros((n_terms, x.shape[-1]))
        self._fill_basis_terms(x, out)
        return out

    def __call__(
        self, x: [np.ndarray, None] = None, basis: [np.ndarray, None] = None
    ) -> np.ndarray:
        """Evaluate the model.

        Parameters
        ----------
        x : np.ndarray, optional
            The co-ordinates at which to evaluate the model (by default, use ``default_x``).
        basis : np.ndarray, optional
            The basis functions at which to evaluate the model. This is useful if calling
            the model multiple times, as the basis itself can be cached and re-used.

        Returns
        -------
        model : np.ndarray
            The model evaluated at the input ``x`` or ``basis``.
        """
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
        self, x: [np.ndarray, None] = None, basis: [np.ndarray, None] = None
    ) -> np.ndarray:
        """Evaluate the model.

        Parameters
        ----------
        x : np.ndarray, optional
            The co-ordinates at which to evaluate the model (by default, use ``default_x``).
        basis : np.ndarray, optional
            The basis functions at which to evaluate the model. This is useful if calling
            the model multiple times, as the basis itself can be cached and re-used.

        Returns
        -------
        model : np.ndarray
            The model evaluated at the input ``x`` or ``basis``.
        """
        t = 2.725 if self.with_cmb else 0
        return t + super().__call__(x)


class PhysicalLin(Foreground):
    """Foreground model using a linearized physical model of the foregrounds."""

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

    def _fill_basis_terms(self, x, out):
        y = x / self.f_center
        if self.log_x:
            y = np.log(y)

        for i in range(self.n_terms):
            out[i] = y ** (i + self.offset)


class Polynomial2D(Model):
    def __init__(self, offset: Tuple[float, float] = (0, 0), **kwargs):
        self.offset = offset
        super().__init__(**kwargs)

    def _fill_basis_terms(self, x, out):
        x, y = x

        for i in range(self.n_terms[0]):
            for j in range(self.n_terms[1]):
                out[i * self.n_terms[1] + j] = x.flatten() ** (
                    i + self.offset[0]
                ) * y.flatten() ** (j + self.offset[1])


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


class Fourier(Model):
    """A Fourier-basis model."""

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
        n_terms: [None, int, Tuple[int, int]] = None,
        **kwargs,
    ):
        """A class representing a fit of model to data.

        Parameters
        ----------
        model_type : str or :class:`Model`
            The type of model to fit to the data.
        xdata : np.ndarray
            The co-ordinates of the measured data.
        ydata : np.ndarray
            The values of the measured data.
        weights : np.ndarray, optional
            The weight of the measured data at each point.
        n_terms : int, optional
            The number of terms to use in the model (useful for models with an
            arbitrary number of terms).
        kwargs
            All other arguments are passed to the chosen model.

        Raises
        ------
        ValueError
            If model_type is not str, or a subclass of :class:`Model`.
        """
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
        self.ydata = ydata.flatten()
        self.weights = weights

        if weights is None:
            self.weights = 1
        elif weights.ndim == 1:
            # if a vector is given
            assert weights.shape == self.ydata.shape
            self.weights = weights
        elif weights.ndim == 2:
            assert weights.shape == (len(self.ydata), len(self.ydata))
            self.weights = weights

        self.n_terms = self.model.n_terms

        self.degrees_of_freedom = (
            len(self.ydata)
            - (self.n_terms if isinstance(self.n_terms, int) else np.prod(self.n_terms))
            - 1
        )

    @cached_property
    def model_parameters(self):
        """The best-fit model parameters."""
        return self.get_model_params()

    @cached_property
    def qr(self):
        """The QR-decomposition."""
        AT = self.model.default_basis

        # sqrt of weight matrix
        sqrt_w = np.sqrt(self.weights)

        if np.isscalar(self.weights):
            WA = AT.T
        elif self.weights.ndim == 1:
            WA = (sqrt_w * AT).T
        else:
            # A and ydata "tilde"
            WA = np.dot(sqrt_w, AT.T)

        # solving system using 'short' QR decomposition (see R. Butt, Num. Anal. Using MATLAB)
        q, r = sp.linalg.qr(WA, mode="economic")
        return q, r

    def get_model_params(self) -> np.ndarray:
        """Obtain the best-fit model parameters."""
        # transposing matrices so data is along columns
        ydata = np.reshape(self.ydata, (-1, 1))

        if np.isscalar(self.weights):
            weighted_ydata = ydata
        elif self.weights.ndim == 1:
            weighted_ydata = (np.sqrt(self.weights) * ydata.T).T
        else:
            weighted_ydata = np.dot(np.sqrt(self.weights), ydata)

        q, r = self.qr
        return sp.linalg.solve(r, np.dot(q.T, weighted_ydata)).flatten()

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
        # Set the parameters on the underlying object (solves for them if not solved yet)
        self.model.parameters = list(self.model_parameters)
        if x is None:
            x = self.xdata
        return self.model(x)

    @cached_property
    def residual(self) -> np.ndarray:
        """Residuals of data to model."""
        return self.ydata - self.evaluate()

    @cached_property
    def weighted_chi2(self) -> float:
        """The chi^2 of the weighted fit."""
        if np.isscalar(self.weights):
            return np.sum(self.residual ** 2)
        elif self.weights.ndim == 1:
            return np.sum(self.residual * np.sqrt(self.weights)) ** 2
        else:
            return np.dot(self.residual.T, np.dot(self.weights, self.residual))

    def reduced_weighted_chi2(self) -> float:
        """The weighted chi^2 divided by the degrees of freedom."""
        return (1 / self.degrees_of_freedom) * self.weighted_chi2

    def weighted_rms(self) -> float:
        """The weighted root-mean-square of the residuals."""
        if np.isscalar(self.weights):
            return np.sqrt(self.weighted_chi2)
        elif self.weights.ndim == 1:
            return np.sqrt(self.weighted_chi2) / np.sum(self.weights)
        else:
            return np.sqrt(self.weighted_chi2) / np.sum(np.diag(self.weights))

    def get_covariance(self) -> np.ndarray:
        """The covariance of the parameter estimates at the solution."""
        r = self.qr[1]
        inv_pre_cov = np.linalg.inv(np.dot(r.T, r))
        return self.reduced_weighted_chi2() * inv_pre_cov
