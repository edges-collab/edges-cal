# -*- coding: utf-8 -*-
"""Functions for generating least-squares model fits for linear models."""
from __future__ import annotations

import attr
import numpy as np
import yaml
from abc import ABCMeta, abstractmethod
from cached_property import cached_property
from typing import Optional, Sequence, Tuple, Type, Union

from .tools import FrequencyRange, as_readonly

F_CENTER = 75.0
_MODELS = {}


@attr.s(frozen=True, kw_only=True)
class FixedLinearModel:
    """
    A base class for a linear model fixed at a certain set of co-ordinates.

    Using this class caches the basis functions at the particular coordinates, and thus
    speeds up the fitting of multiple sets of data at those co-ordinates.

    Parameters
    ----------
    model
        The linear model to evaluate at the co-ordinates
    x
        A set of co-ordinates at which to evaluate the model.
    init_basis
        If the basis functions of the model, evaluated at x, are known already, they
        can be input directly to save computation time.
    """

    model: Model = attr.ib()
    x: np.ndarray = attr.ib(converter=np.asarray)
    _init_basis: np.ndarray | None = attr.ib(
        default=None, converter=attr.converters.optional(np.asarray)
    )

    @model.validator
    def _model_vld(self, att, val):
        assert isinstance(val, Model)

    @_init_basis.validator
    def _init_basis_vld(self, att, val):
        if val is None:
            return None

        if val.shape[1] != len(self.x):
            raise ValueError("The init_basis values must be the same shape as x.")

    @property
    def n_terms(self):
        """The number of terms/parameters in the model."""
        return self.model.n_terms

    @cached_property
    def basis(self) -> np.ndarray:
        """The (cached) basis functions at default_x.

        Shape ``(n_terms, x)``.
        """
        out = np.zeros((self.model.n_terms, len(self.x)))
        for indx in range(self.model.n_terms):
            if self._init_basis is not None and indx < len(self._init_basis):
                out[indx] = self._init_basis[indx]
            else:
                out[indx] = self.model.get_basis_term(indx, self.x)

        return out

    def __call__(
        self,
        x: np.ndarray | None = None,
        parameters: Sequence | None = None,
        indices: Sequence | None = None,
    ) -> np.ndarray:
        """Evaluate the model.

        Parameters
        ----------
        x
            The coordinates at which to evaluate the model (by default, use ``self.x``).
        params
            A list/array of parameters at which to evaluate the model. Will use the
            instance's parameters if available. If using a subset of the basis
            functions, you can pass a subset of parameters.
        indices
            Sequence of parameters indices to use (other parameters are set to zero).

        Returns
        -------
        model
            The model evaluated at the input ``x``.
        """
        return self.model(
            basis=self.basis if x is None else None,
            x=x,
            parameters=parameters,
            indices=indices,
        )

    def fit(
        self,
        ydata: np.ndarray,
        weights: np.ndarray | float = 1.0,
        xdata: np.ndarray | None = None,
    ):
        """Create a linear-regression fit object."""
        thing = self.at_x(xdata) if xdata is not None else self
        return ModelFit(thing, ydata=ydata, weights=weights,)

    def at_x(self, x: np.ndarray) -> FixedLinearModel:
        """Return a new :class:`FixedLinearModel` at given co-ordinates."""
        return attr.evolve(self, x=x, init_basis=None)

    def with_nterms(
        self, n_terms: int, parameters: Sequence | None = None
    ) -> FixedLinearModel:
        """Return a new :class:`FixedLinearModel` with given nterms and parameters."""
        init_basis = as_readonly(self.basis[: min(self.model.n_terms, n_terms)])
        model = self.model.with_nterms(n_terms=n_terms, parameters=parameters)
        return attr.evolve(self, model=model, init_basis=init_basis)

    def with_params(self, parameters: Sequence) -> FixedLinearModel:
        """Return a new :class:`FixedLinearModel` with givne parameters."""
        n_terms = len(parameters)
        return self.with_nterms(n_terms=n_terms, parameters=parameters)


@attr.s(frozen=True, kw_only=True)
class Model(metaclass=ABCMeta):
    """A base class for a linear model."""

    default_n_terms: int | None = None
    n_terms_min: int = 1
    n_terms_max: int = 1000000

    parameters: Sequence | None = attr.ib(
        default=None, converter=attr.converters.optional(np.asarray)
    )
    n_terms: int = attr.ib(converter=attr.converters.optional(int))

    def __init_subclass__(cls, is_meta=False, **kwargs):
        """Initialize a subclass and add it to the registered models."""
        super().__init_subclass__(**kwargs)
        if not is_meta:
            _MODELS[cls.__name__.lower()] = cls

    @n_terms.default
    def _n_terms_default(self):
        if self.parameters is not None:
            return len(self.parameters)
        else:
            return self.default_n_terms

    @n_terms.validator
    def _n_terms_validator(self, att, val):
        if val is None:
            raise ValueError("Either n_terms or explicit parameters must be given.")

        if not (self.n_terms_min <= val <= self.n_terms_max):
            raise ValueError(
                f"n_terms must be between {self.n_terms_min} and {self.n_terms_max}"
            )

        if self.parameters is not None and val != len(self.parameters):
            raise ValueError(f"Wrong number of parameters! Should be {val}.")

    @abstractmethod
    def get_basis_term(self, indx: int, x: np.ndarray) -> np.ndarray:
        """Define the basis terms for the model."""
        pass

    def get_basis_terms(self, x: np.ndarray) -> np.ndarray:
        """Get a 2D array of all basis terms at ``x``."""
        return np.array([self.get_basis_term(indx, x) for indx in range(self.n_terms)])

    def with_nterms(
        self, n_terms: int | None = None, parameters: Sequence | None = None
    ) -> Model:
        """Return a new :class:`Model` with given nterms and parameters."""
        if parameters is not None:
            n_terms = len(parameters)
        return attr.evolve(self, n_terms=n_terms, parameters=parameters)

    @staticmethod
    def from_str(model: str, **kwargs) -> Model:
        """Obtain a :class:`Model` given a string name."""
        return get_mdl(model)(**kwargs)

    def at(self, **kwargs) -> FixedLinearModel:
        """Get an evaluated linear model."""
        return FixedLinearModel(model=self, **kwargs)

    def __call__(
        self,
        x: np.ndarray | None,
        basis: np.ndarray | None,
        parameters: Sequence | None = None,
        indices: Sequence | None = None,
    ) -> np.ndarray:
        """Evaluate the model.

        Parameters
        ----------
        x : np.ndarray, optional
            The co-ordinates at which to evaluate the model (by default, use
            ``default_x``).
        basis : np.ndarray, optional
            The basis functions at which to evaluate the model. This is useful if
            calling the model multiple times, as the basis itself can be cached and
            re-used.
        params
            A list/array of parameters at which to evaluate the model. Will use the
            instance's parameters if available. If using a subset of the basis
            functions, you can pass a subset of parameters.

        Returns
        -------
        model : np.ndarray
            The model evaluated at the input ``x`` or ``basis``.
        """
        parameters = self.parameters if parameters is None else np.array(parameters)

        if parameters is None:
            raise ValueError("You must supply parameters to evaluate the model!")

        indices = np.arange(len(parameters)) if indices is None else np.array(indices)

        if x is None and basis is None:
            raise ValueError("You must supply either x or basis!")

        if basis is None:
            basis = self.get_basis_terms(x)

        if any(idx >= len(basis) for idx in indices):
            raise ValueError("Cannot use more basis sets than available!")

        if len(parameters) != len(indices):
            parameters = parameters[indices]

        return np.dot(parameters, basis[indices])

    def fit(
        self, xdata: np.ndarray, ydata: np.ndarray, weights: np.ndarray | float = 1.0,
    ) -> ModelFit:
        """Create a linear-regression fit object."""
        return self.at(x=xdata).fit(ydata, weights=weights)


def get_mdl(model: str | Type[Model]) -> Type[Model]:
    """Get a linear model class from a string input."""
    if isinstance(model, str):
        return _MODELS[model]
    elif np.issubclass_(model, Model):
        return model
    else:
        raise ValueError("model needs to be a string or Model subclass")


def get_mdl_inst(model: str | Model | Type[Model], **kwargs) -> Model:
    """Get a model instance from given string input."""
    if isinstance(model, Model):
        if kwargs:
            return attr.evolve(model, **kwargs)
        else:
            return model

    return get_mdl(model)(**kwargs)


@attr.s(frozen=True, kw_only=True)
class Foreground(Model, is_meta=True):
    """
    Base class for Foreground models.

    Parameters
    ----------
    f_center : float
        A "center" or "reference" frequency. Typically models will have their
        co-ordindates divided by this frequency before solving for the
        co-efficients.
    with_cmb : bool
        Whether to add a simply CMB component to the foreground.
    """

    f_center: float = attr.ib(default=F_CENTER, converter=float)
    with_cmb: bool = attr.ib(default=False, converter=bool)


@attr.s(frozen=True, kw_only=True)
class PhysicalLin(Foreground):
    """Foreground model using a linearized physical model of the foregrounds."""

    n_terms_max: int = 5
    default_n_terms: int = 5

    def get_basis_term(self, indx: int, x: np.ndarray) -> np.ndarray:
        """Define the basis functions of the model."""
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


@attr.s(frozen=True, kw_only=True)
class Polynomial(Foreground):
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

    log_x: bool = attr.ib(default=False, converter=bool)
    offset: float = attr.ib(default=0, converter=float)

    def get_basis_term(self, indx: int, x: np.ndarray) -> np.ndarray:
        """Define the basis functions of the model."""
        y = x / self.f_center
        if self.log_x:
            y = np.log(y)

        return y ** (indx + self.offset)


@attr.s(frozen=True, kw_only=True)
class EdgesPoly(Polynomial):
    """
    Polynomial with an offset corresponding to approximate galaxy spectral index.

    Parameters
    ----------
    offset : float
        The offset to use. Default is close to the Galactic spectral index.
    kwargs
        All other arguments are passed through to :class:`Polynomial`.
    """

    offset: float = attr.ib(default=-2.5, converter=float)


@attr.s(frozen=True, kw_only=True)
class LinLog(Foreground):
    beta: float = attr.ib(default=-2.5, converter=float)

    @cached_property
    def _poly(self):
        return Polynomial(
            log_x=True, offset=0, n_terms=self.n_terms, parameters=self.parameters
        )

    def get_basis_term(self, indx: int, x: np.ndarray) -> np.ndarray:
        """Define the basis functions of the model."""
        term = self._poly.get_basis_term(indx, x)
        return term * (x / self.f_center) ** self.beta


@attr.s(frozen=True, kw_only=True)
class Fourier(Model):
    """A Fourier-basis model."""

    period: float = attr.ib(default=2 * np.pi, converter=float)

    @cached_property
    def _period_fac(self):
        return 2 * np.pi / self.period

    def get_basis_term(self, indx: int, x: np.ndarray) -> np.ndarray:
        """Define the basis functions of the model."""
        if indx == 0:
            return np.ones_like(x)
        elif indx % 2:
            return np.cos(self._period_fac * (indx + 1) // 2 * x)
        else:
            return np.sin(self._period_fac * (indx + 1) // 2 * x)


@attr.s(frozen=True, kw_only=True)
class FourierDay(Model):
    """A Fourier-basis model with period of 24 (hours)."""

    @cached_property
    def _fourier(self):
        return Fourier(period=48.0, n_terms=self.n_terms, parameters=self.parameters)

    def get_basis_term(self, indx: int, x: np.ndarray) -> np.ndarray:
        """Define the basis functions of the model."""
        return self._fourier.get_basis_term(indx, x)


class NoiseWaves(Model):
    """A multi-model for fitting noise waves.

    Notes
    -----
    The Linear Model is

    .. math:: T_{NS}Q - K_0 T_{ant} = T_{unc} K_1 + T_{cos} K_2 + T_{sin} K_3 - T_L

    where the LHS is non-deterministic (i.e. contains random measured variables Q and
    T_ANT). Each of the T variables is in fact a polynomial.
    We *cannot*  estimate T_NS as part of the linear model because it multiples the
    non-deterministic Q and therefore scales the variance non-uniformly.

    Parameers
    ---------
    freq
        The frequencies at which the lab measurements were taken.
    gamma_coeffs
        The linear coefficients that are functions of the S11 for the sources (i.e.
        the K_X in the above equation). Formulas for these are given in Monsalve et al.
        (2017) Eq. 7. Each array (there should be three, one for each term) should have
        shape (n_sources, n_freq).
    c_terms
        The number of polynomial terms describing T_L.
    w_terms
        The number of polynomial terms describing the noise-wave temperatures,
        T_unc, T_cos and T_sin.
    model
        The kind of model to use for the unknown models (noise-waves and T_load).
        Typically this is a polynomial, but you can choose another linear model if you
        choose.
    """

    def __init__(
        self,
        freq: np.ndarray,
        gamma_coeffs: Tuple[np.ndarray, np.ndarray, np.ndarray],
        c_terms: int = 6,
        w_terms: int = 6,
        fg_terms: int = 0,
        model: Union[Type[Model], str] = "polynomial",
        fg_model: Optional[Union[Type[Model], str]] = "linlog",
        gamma_coeff_fg: Optional[np.ndarray] = None,
        **kwargs,
    ):
        self.freq = FrequencyRange(freq)
        n_sources = len(gamma_coeffs[0])  # ambient, hot, short, open / other?
        n_freq = self.freq.n

        assert (
            len(gamma_coeffs) == 3
        )  # remember we don't have K_0 here, as it's part of the data vector.
        assert (kk.shape == (n_sources, n_freq) for kk in gamma_coeffs)

        if fg_terms:
            # Assume that the LAST source is the actual antenna.
            self.fg_model = Model.get_mdl(fg_model)(
                default_x=self.freq.freq, n_terms=fg_terms
            )

        # Add in the coefficient of unity for T_L
        self.K = np.array(
            [
                -np.ones((n_sources, n_freq)),
                gamma_coeffs[0],
                gamma_coeffs[1],
                gamma_coeffs[2],
            ]
        )
        self.K = self.K.reshape((4, -1))
        self.gamma_coeff_fg = gamma_coeff_fg

        if gamma_coeff_fg is not None:
            assert self.gamma_coeff_fg.shape == (n_freq,)

            # Make the K[0] for the foreground term zero everywhere except for the
            # field data, which is assumed to be the last dimension!
            self.gamma_coeff_fg = np.concatenate(
                (np.zeros(n_freq),) * (n_sources - 1) + (self.gamma_coeff_fg,)
            )

        # TODO might be good to re-centre the frequencies?
        self.c_terms = c_terms
        self.w_terms = w_terms
        self.model = Model.get_mdl(model)(
            default_x=self.freq.freq_recentred, n_terms=max(self.c_terms, self.w_terms)
        )

        super().__init__(
            parameters=kwargs.get("parameters"),
            n_terms=c_terms + 3 * w_terms + fg_terms,
            default_x=np.concatenate((self.freq.freq_recentred,) * n_sources),
        )

    def _get_basis_term(self, indx: int, x: np.ndarray) -> np.ndarray:
        if indx < self.c_terms:
            return self.K[0] * self.model.get_basis_term(indx, x)
        elif indx < self.c_terms + self.w_terms:
            return self.K[1] * self.model.get_basis_term(indx - self.c_terms, x)
        elif indx < self.c_terms + 2 * self.w_terms:
            return self.K[2] * self.model.get_basis_term(
                indx - self.c_terms - self.w_terms, x
            )
        elif indx < self.c_terms + 3 * self.w_terms:
            return self.K[3] * self.model.get_basis_term(
                indx - self.c_terms - 2 * self.w_terms, x
            )
        elif indx < self.c_terms + 3 * self.w_terms + self.fg_model.n_terms:
            return self.gamma_coeff_fg * self.fg_model.get_basis_term(
                indx - self.c_terms - 3 * self.w_terms, self.freq.denormalize(x)
            )

    def _get_normalized_freq(self, freq):
        if freq is None:
            return self.freq.freq_recentred
        else:
            return self.freq.normalize(freq)

    def t_load(
        self, parameters: np.ndarray = None, freq: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute T_L for given polynomial parameters and frequencies."""
        freq = self._get_normalized_freq(freq)

        if parameters is None:
            parameters = self.parameters

        if len(parameters) == self.c_terms:
            p = parameters
        else:
            p = parameters[: self.c_terms]

        return self.model(parameters=p, indices=np.arange(self.c_terms))

    def t_unc(
        self, parameters: np.ndarray = None, freq: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute T_unc for given polynomial parameters and frequencies."""
        freq = self._get_normalized_freq(freq)
        if parameters is None:
            parameters = self.parameters

        if len(parameters) == self.w_terms:
            p = parameters
        else:
            p = parameters[self.c_terms : self.c_terms + self.w_terms]

        return self.model(parameters=p, indices=np.arange(self.w_terms))

    def t_cos(
        self, parameters: np.ndarray = None, freq: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute T_cos for given polynomial parameters and frequencies."""
        freq = self._get_normalized_freq(freq)

        if parameters is None:
            parameters = self.parameters

        if len(parameters) == self.w_terms:
            p = parameters
        else:
            p = parameters[
                self.c_terms + self.w_terms : self.c_terms + 2 * self.w_terms
            ]

        return self.model(parameters=p, indices=np.arange(self.w_terms))

    def t_sin(
        self, parameters: np.ndarray = None, freq: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute T_sin for given polynomial parameters and frequencies."""
        freq = self._get_normalized_freq(freq)

        if parameters is None:
            parameters = self.parameters

        if len(parameters) == self.w_terms:
            p = parameters
        else:
            p = parameters[
                self.c_terms + 2 * self.w_terms : self.c_terms + 3 * self.w_terms
            ]

        return self.model(parameters=p, indices=np.arange(self.w_terms))

    def t_fg(
        self, parameters: np.ndarray = None, freq: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute foreground temperature for given parameters and frequencies."""
        if parameters is None:
            parameters = self.parameters

        if len(parameters) == self.fg_model.n_terms:
            p = parameters
        else:
            p = parameters[self.c_terms + 3 * self.w_terms :]

        return self.fg_model(parameters=p)


@attr.s(frozen=True)
class ModelFit:
    """A class representing a fit of model to data.

    Parameters
    ----------
    model
        The evaluatable model to fit to the data.
    ydata
        The values of the measured data.
    weights
        The weight of the measured data at each point. This corresponds to the
        *variance* of the measurement (not the standard deviation). This is
        appropriate if the weights represent the number of measurements going into
        each piece of data.

    Raises
    ------
    ValueError
        If model_type is not str, or a subclass of :class:`Model`.
    """

    model: FixedLinearModel = attr.ib()
    ydata: np.ndarray = attr.ib()
    weights: np.ndarray | float = attr.ib(
        default=1.0, validator=attr.validators.instance_of((np.ndarray, float))
    )

    @ydata.validator
    def _ydata_vld(self, att, val):
        assert val.shape == self.model.x.shape

    @weights.validator
    def _weights_vld(self, att, val):
        if isinstance(val, np.ndarray):
            assert val.shape == self.model.x.shape

    @cached_property
    def degrees_of_freedom(self) -> int:
        """The number of degrees of freedom of the fit."""
        return self.model.x.size - self.model.model.n_terms - 1

    @cached_property
    def fit(self) -> FixedLinearModel:
        """A model that has parameters set based on the best fit to this data."""
        if np.isscalar(self.weights):
            pars = self._ls(self.model.basis, self.ydata)
        else:
            pars = self._wls(self.model.basis, self.ydata, w=self.weights)

        # Create a new model with the same parameters but specific parameters and xdata.
        return self.model.with_params(parameters=pars)

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
        return as_readonly(self.fit.model.parameters)

    def evaluate(self, x: Optional[np.ndarray] = None) -> np.ndarray:
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
        return self.fit(x=x)

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


def _model_yaml_constructor(
    loader: yaml.SafeLoader, node: yaml.nodes.MappingNode
) -> Model:
    mapping = loader.construct_mapping(node)
    model = get_mdl(mapping.pop("model"))
    return model(**mapping)


def _model_yaml_representer(
    dumper: yaml.SafeDumper, model: Model
) -> yaml.nodes.MappingNode:
    model_dct = attr.asdict(model)
    model_dct.update(model=model.__class__.__name__.lower())
    return dumper.represent_mapping("!Model", model_dct)


yaml.FullLoader.add_constructor("!Model", _model_yaml_constructor)

# for mdl in _MODELS.values():
yaml.add_multi_representer(Model, _model_yaml_representer)
