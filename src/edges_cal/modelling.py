"""Functions for generating least-squares model fits for linear models."""
from __future__ import annotations

import attr
import numpy as np
import yaml
from abc import ABCMeta, abstractmethod
from cached_property import cached_property
from edges_io.h5 import register_h5type
from typing import Sequence

from . import receiver_calibration_func as rcf
from .simulate import simulate_q_from_calobs
from .tools import as_readonly

F_CENTER = 75.0
_MODELS = {}


@attr.s(frozen=True, kw_only=True)
class FixedLinearModel(yaml.YAMLObject):
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

    yaml_tag = "!Model"

    model: Model = attr.ib()
    x: np.ndarray = attr.ib(converter=np.asarray)
    _init_basis: np.ndarray | None = attr.ib(
        default=None, converter=attr.converters.optional(np.asarray)
    )

    @classmethod
    def to_yaml(cls, dumper, data):
        """Method to convert to YAML format."""
        return _model_yaml_representer(dumper, data.model)

    @model.validator
    def _model_vld(self, att, val):
        assert isinstance(val, (Model, CompositeModel))

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
                out[indx] = self.model.get_basis_term_transformed(indx, self.x)

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
        return ModelFit(
            thing,
            ydata=ydata,
            weights=weights,
        )

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
        assert len(parameters) == self.model.n_terms

        init_basis = as_readonly(self.basis)
        model = self.model.with_params(parameters=parameters)
        return attr.evolve(self, model=model, init_basis=init_basis)

    @property
    def parameters(self) -> np.ndarray | None:
        """The parameters of the model, if set."""
        return self.model.parameters


def _transform_yaml_constructor(
    loader: yaml.SafeLoader, node: yaml.nodes.MappingNode
) -> Model:
    mapping = loader.construct_mapping(node, deep=True)
    return ModelTransform.get(node.tag[1:])(**mapping)


def _transform_yaml_representer(
    dumper: yaml.SafeDumper, tr: ModelTransform
) -> yaml.nodes.MappingNode:
    dct = attr.asdict(tr, recurse=False)
    return dumper.represent_mapping(f"!{tr.__class__.__name__}", dct)


@attr.s(frozen=True, kw_only=True)
class ModelTransform(metaclass=ABCMeta):
    _models = {}

    def __init_subclass__(cls, is_meta=False, **kwargs):
        """Initialize a subclass and add it to the registered models."""
        super().__init_subclass__(**kwargs)

        yaml.add_constructor(f"!{cls.__name__}", _transform_yaml_constructor)

        if not is_meta:
            cls._models[cls.__name__.lower()] = cls

    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform the coordinates."""
        pass

    @classmethod
    def get(cls, model: str) -> type[ModelTransform]:
        """Get a ModelTransform class."""
        return cls._models[model.lower()]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Transform the coordinates."""
        return self.transform(x)


@attr.s(frozen=True, kw_only=True)
class IdentityTransform(ModelTransform):
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform the coordinates."""
        return x


@attr.s(frozen=True, kw_only=True)
class ScaleTransform(ModelTransform):
    scale: float = attr.ib(converter=float)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform the coordinates."""
        return x / self.scale


def tuple_converter(x):
    """Convert input to tuple of floats."""
    return tuple(float(xx) for xx in x)


@attr.s(frozen=True, kw_only=True)
class CentreTransform(ModelTransform):
    range: tuple[float, float] = attr.ib(converter=tuple_converter)
    centre: float = attr.ib(default=0.0, converter=float)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform the coordinates."""
        return x - self.range[0] - (self.range[1] - self.range[0]) / 2 + self.centre


@attr.s(frozen=True, kw_only=True)
class UnitTransform(ModelTransform):
    """A transform that takes the input range down to -1 to 1."""

    range: tuple[float, float] = attr.ib(converter=tuple_converter)

    @cached_property
    def _centre(self):
        return CentreTransform(centre=0, range=self.range)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform the coordinates."""
        if self.range[0] is None:
            self.range[0] = float(x.min())
            self.range[1] = float(x.max())

        return 2 * self._centre.transform(x) / (self.range[1] - self.range[0])


@attr.s(frozen=True, kw_only=True)
class LogTransform(ModelTransform):
    """A transform that takes the logarithm of the input."""

    scale: float = attr.ib(1.0)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform the coordinates."""
        return np.log(x)


@attr.s(frozen=True, kw_only=True)
class ZerotooneTransform(ModelTransform):
    """A transform that takes an input range down to (0,1)."""

    range: tuple[float, float] = attr.ib(converter=tuple_converter)

    @range.default
    def _rng_default(self):
        return [None, None]

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform the coordinates."""
        return (x - self.range[0]) / (self.range[1] - self.range[0])


@register_h5type
@attr.s(frozen=True, kw_only=True)
class Model(metaclass=ABCMeta):
    """A base class for a linear model."""

    default_n_terms: int | None = None
    n_terms_min: int = 1
    n_terms_max: int = 1000000

    parameters: Sequence | None = attr.ib(
        default=None,
        converter=attr.converters.optional(np.asarray),
        eq=attr.cmp_using(eq=np.array_equal),
    )
    n_terms: int = attr.ib(converter=attr.converters.optional(int))
    transform: ModelTransform = attr.ib(default=IdentityTransform())

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

    def get_basis_term_transformed(self, indx: int, x: np.ndarray) -> np.ndarray:
        """Get the basis term after coordinate transformation."""
        return self.get_basis_term(indx=indx, x=self.transform(x))

    def get_basis_terms(self, x: np.ndarray) -> np.ndarray:
        """Get a 2D array of all basis terms at ``x``."""
        x = self.transform(x)
        return np.array([self.get_basis_term(indx, x) for indx in range(self.n_terms)])

    def with_nterms(
        self, n_terms: int | None = None, parameters: Sequence | None = None
    ) -> Model:
        """Return a new :class:`Model` with given nterms and parameters."""
        if parameters is not None:
            n_terms = len(parameters)

        return attr.evolve(self, n_terms=n_terms, parameters=parameters)

    def with_params(self, parameters: Sequence | None):
        """Get new model with different parameters."""
        assert len(parameters) == self.n_terms
        return self.with_nterms(parameters=parameters)

    @staticmethod
    def from_str(model: str, **kwargs) -> Model:
        """Obtain a :class:`Model` given a string name."""
        return get_mdl(model)(**kwargs)

    def at(self, **kwargs) -> FixedLinearModel:
        """Get an evaluated linear model."""
        return FixedLinearModel(model=self, **kwargs)

    def __call__(
        self,
        x: np.ndarray | None = None,
        basis: np.ndarray | None = None,
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
        self,
        xdata: np.ndarray,
        ydata: np.ndarray,
        weights: np.ndarray | float = 1.0,
    ) -> ModelFit:
        """Create a linear-regression fit object."""
        return self.at(x=xdata).fit(ydata, weights=weights)


def get_mdl(model: str | type[Model]) -> type[Model]:
    """Get a linear model class from a string input."""
    if isinstance(model, str):
        return _MODELS[model]
    elif np.issubclass_(model, Model):
        return model
    else:
        raise ValueError("model needs to be a string or Model subclass")


def get_mdl_inst(model: str | Model | type[Model], **kwargs) -> Model:
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
        Whether to add a simple CMB component to the foreground.
    """

    with_cmb: bool = attr.ib(default=False, converter=bool)
    f_center: float = attr.ib(F_CENTER, converter=float)
    transform: ModelTransform = attr.ib()

    @transform.default
    def _tr_default(self):
        return ScaleTransform(scale=self.f_center)


@attr.s(frozen=True, kw_only=True)
class PhysicalLin(Foreground):
    """Foreground model using a linearized physical model of the foregrounds."""

    n_terms_max: int = 5
    default_n_terms: int = 5

    def get_basis_term(self, indx: int, x: np.ndarray) -> np.ndarray:
        """Define the basis functions of the model."""
        if indx < 3:
            logy = np.log(x)
            y25 = x ** -2.5
            return y25 * logy ** indx

        elif indx == 3:
            return x ** -4.5
        elif indx == 4:
            return 1 / (x * x)
        else:
            raise ValueError("too many terms supplied!")


@attr.s(frozen=True, kw_only=True)
class Polynomial(Model):
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

    offset: float = attr.ib(default=0, converter=float)

    def get_basis_term(self, indx: int, x: np.ndarray) -> np.ndarray:
        """Define the basis functions of the model."""
        return x ** (indx + self.offset)


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

    @property
    def _poly(self):
        return Polynomial(
            transform=LogTransform(),
            offset=0,
            n_terms=self.n_terms,
            parameters=self.parameters,
        )

    def get_basis_term(self, indx: int, x: np.ndarray) -> np.ndarray:
        """Define the basis functions of the model."""
        term = self._poly.get_basis_term_transformed(indx, x)
        return term * x ** self.beta


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
            return np.cos(self._period_fac * ((indx + 1) // 2) * x)
        else:
            return np.sin(self._period_fac * ((indx + 1) // 2) * x)


@attr.s(frozen=True, kw_only=True)
class FourierDay(Model):
    """A Fourier-basis model with period of 24 (hours)."""

    @property
    def _fourier(self):
        return Fourier(period=48.0, n_terms=self.n_terms, parameters=self.parameters)

    def get_basis_term(self, indx: int, x: np.ndarray) -> np.ndarray:
        """Define the basis functions of the model."""
        return self._fourier.get_basis_term(indx, x)


@attr.s(frozen=True, kw_only=True)
class CompositeModel:
    models: dict[str, Model] = attr.ib()
    extra_basis: dict[str, np.ndarray] = attr.ib(factory=dict)

    @extra_basis.validator
    def _eb_vld(self, att, val):
        assert all(v in self.models for v in val)

    @cached_property
    def n_terms(self) -> int:
        """The number of terms in the full composite model."""
        return sum(m.n_terms for m in self.models.values())

    @cached_property
    def parameters(self) -> np.ndarray:
        """The read-only list of parameters of all sub-models."""
        return np.concatenate(tuple(m.parameters for m in self.models.values()))

    @cached_property
    def _index_map(self):
        _index_map = {}

        indx = 0
        for name, model in self.models.items():
            for i in range(model.n_terms):
                _index_map[indx] = (name, i)
                indx += 1

        return _index_map

    def __getattr__(self, item):
        """Get sub-models as if they were top-level attributes."""
        if item not in self.models:
            raise AttributeError(f"{item} not one of the models.")

        return self.models[item]

    def _get_model_param_indx(self, model: str):
        indx = list(self.models.keys()).index(model)
        n_before = sum(m.n_terms for m in list(self.models.values())[:indx])
        model = self.models[model]
        return slice(n_before, n_before + model.n_terms, 1)

    def get_extra_basis(self, model: str, x: np.ndarray | None = None):
        """Get the extra model-dependent basis function for a given model."""
        extra = self.extra_basis.get(model, 1)
        if callable(extra):
            extra = extra(x)
        return extra

    @cached_property
    def model_idx(self) -> dict[str, slice]:
        """Dictionary of parameter indices correponding to each model."""
        return {name: self._get_model_param_indx(name) for name in self.models}

    def get_model(
        self,
        model: str,
        parameters: np.ndarray = None,
        x: np.ndarray | None = None,
        with_extra: bool = False,
    ):
        """Calculate a sub-model."""
        indx = self.model_idx[model]

        extra = self.get_extra_basis(model, x) if with_extra else 1
        model = self.models[model]

        if parameters is None:
            parameters = self.parameters

        p = parameters if len(parameters) == model.n_terms else parameters[indx]
        return model(x=x, parameters=p) * extra

    def get_basis_term(self, indx: int, x: np.ndarray) -> np.ndarray:
        """Define the basis terms for the model."""
        model, indx = self._index_map[indx]
        extra = self.get_extra_basis(model, x)
        try:
            mask = extra.astype(bool)
            extra = extra[mask]
        except AttributeError:
            mask = np.ones(len(x), dtype=bool)

        out = np.zeros_like(x)
        out[mask] = getattr(self, model).get_basis_term(indx, x[mask]) * extra
        return out

    def get_basis_term_transformed(self, indx: int, x: np.ndarray) -> np.ndarray:
        """Get the basis function term after coordinate tranformation."""
        model = self._index_map[indx][0]
        return self.get_basis_term(indx, x=getattr(self, model).transform(x))

    def get_basis_terms(self, x: np.ndarray) -> np.ndarray:
        """Get a 2D array of all basis terms at ``x``."""
        return np.array(
            [self.get_basis_term_transformed(indx, x) for indx in range(self.n_terms)]
        )

    def with_nterms(
        self, model: str, n_terms: int | None = None, parameters: Sequence | None = None
    ) -> Model:
        """Return a new :class:`Model` with given nterms and parameters."""
        model_ = self[model]

        if parameters is not None:
            n_terms = len(parameters)

        model_ = model_.with_nterms(n_terms=n_terms, parameters=parameters)

        return attr.evolve(self, models={**self.models, **{model: model_}})

    def with_params(self, parameters: Sequence):
        """Get a new model with specified parameters."""
        assert len(parameters) == self.n_terms
        models = {
            name: model.with_params(
                parameters=parameters[self._get_model_param_indx(name)]
            )
            for name, model in self.models.items()
        }
        return attr.evolve(self, models=models)

    def at(self, **kwargs) -> FixedLinearModel:
        """Get an evaluated linear model."""
        return FixedLinearModel(model=self, **kwargs)

    def __call__(
        self,
        x: np.ndarray | None = None,
        basis: np.ndarray | None = None,
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
        return Model.__call__(
            self, x=x, basis=basis, parameters=parameters, indices=indices
        )

    def fit(
        self,
        xdata: np.ndarray,
        ydata: np.ndarray,
        weights: np.ndarray | float = 1.0,
    ) -> ModelFit:
        """Create a linear-regression fit object."""
        return self.at(x=xdata).fit(ydata, weights=weights)


@attr.s(frozen=True, kw_only=True)
class ComplexRealImagModel(yaml.YAMLObject):
    """A composite model that is specifically for complex functions in real/imag."""

    yaml_tag = "ComplexRealImagModel"

    real: Model | FixedLinearModel = attr.ib()
    imag: Model | FixedLinearModel = attr.ib()

    def at(self, **kwargs) -> FixedLinearModel:
        """Get an evaluated linear model."""
        return attr.evolve(
            self,
            real=self.real.at(**kwargs),
            imag=self.imag.at(**kwargs),
        )

    def __call__(
        self,
        x: np.ndarray | None = None,
        parameters: Sequence | None = None,
    ) -> np.ndarray:
        """Evaluate the model.

        Parameters
        ----------
        x
            The co-ordinates at which to evaluate the model (by default, use
            ``default_x``).
        params
            A list/array of parameters at which to evaluate the model. Will use the
            instance's parameters if available. If using a subset of the basis
            functions, you can pass a subset of parameters.

        Returns
        -------
        model
            The model evaluated at the input ``x`` or ``basis``.
        """
        return self.real(
            x=x,
            parameters=parameters[: self.real.n_terms]
            if parameters is not None
            else None,
        ) + 1j * self.imag(
            x=x,
            parameters=parameters[self.real.n_terms :]
            if parameters is not None
            else None,
        )

    def fit(
        self,
        ydata: np.ndarray,
        weights: np.ndarray | float = 1.0,
        xdata: np.ndarray | None = None,
    ):
        """Create a linear-regression fit object."""
        if isinstance(self.real, FixedLinearModel):
            real = self.real
        else:
            real = self.real.at(x=xdata)

        if isinstance(self.imag, FixedLinearModel):
            imag = self.imag
        else:
            imag = self.imag.at(x=xdata)

        real = real.fit(np.real(ydata), weights=weights).fit
        imag = imag.fit(np.imag(ydata), weights=weights).fit
        return attr.evolve(self, real=real, imag=imag)


@attr.s(frozen=True, kw_only=True)
class ComplexMagPhaseModel(yaml.YAMLObject):
    """A composite model that is specifically for complex functions in mag/phase."""

    yaml_tag = "ComplexMagPhaseModel"

    mag: Model | FixedLinearModel = attr.ib()
    phs: Model | FixedLinearModel = attr.ib()

    def at(self, **kwargs) -> FixedLinearModel:
        """Get an evaluated linear model."""
        return attr.evolve(
            self,
            mag=self.mag.at(**kwargs),
            phs=self.phs.at(**kwargs),
        )

    def __call__(
        self,
        x: np.ndarray | None = None,
        parameters: Sequence | None = None,
    ) -> np.ndarray:
        """Evaluate the model.

        Parameters
        ----------
        x
            The co-ordinates at which to evaluate the model (by default, use
            ``default_x``).
        params
            A list/array of parameters at which to evaluate the model. Will use the
            instance's parameters if available. If using a subset of the basis
            functions, you can pass a subset of parameters.

        Returns
        -------
        model : np.ndarray
            The model evaluated at the input ``x`` or ``basis``.
        """
        return self.mag(
            x=x,
            parameters=parameters[: self.mag.n_terms]
            if parameters is not None
            else None,
        ) * np.exp(
            1j
            * self.phs(
                x=x,
                parameters=parameters[self.mag.n_terms :]
                if parameters is not None
                else None,
            )
        )

    def fit(
        self,
        ydata: np.ndarray,
        weights: np.ndarray | float = 1.0,
        xdata: np.ndarray | None = None,
    ):
        """Create a linear-regression fit object."""
        if isinstance(self.mag, FixedLinearModel):
            mag = self.mag
        else:
            mag = self.mag.at(x=xdata)

        if isinstance(self.phs, FixedLinearModel):
            phs = self.phs
        else:
            phs = self.phs.at(x=xdata)

        mag = mag.fit(np.abs(ydata), weights=weights).fit
        phs = phs.fit(np.unwrap(np.angle(ydata)), weights=weights).fit
        return attr.evolve(self, mag=mag, phs=phs)


@attr.s(frozen=True, kw_only=True)
class NoiseWaves:
    freq: np.ndarray = attr.ib()
    gamma_src: dict[str, np.ndarray] = attr.ib()
    gamma_rec: np.ndarray = attr.ib()
    c_terms: int = attr.ib(default=5)
    w_terms: int = attr.ib(default=6)
    parameters: Sequence | None = attr.ib(default=None)

    @cached_property
    def src_names(self) -> tuple[str]:
        """List of names of inputs sources (eg. ambient, hot_load, open, short)."""
        return tuple(self.gamma_src.keys())

    @cached_property
    def linear_model(self) -> CompositeModel:
        """The actual composite linear model object associated with the noise waves."""
        # K should be a an array of shape (Nsrc Nnu x Nnoisewaveterms)
        K = np.hstack(
            tuple(
                rcf.get_K(gamma_rec=self.gamma_rec, gamma_ant=s11src)
                for s11src in self.gamma_src.values()
            )
        )

        # x is the frequencies repeated for every input source
        x = np.tile(self.freq, len(self.gamma_src))
        tr = UnitTransform(range=(x.min(), x.max()))

        return CompositeModel(
            models={
                "tunc": Polynomial(
                    n_terms=self.w_terms,
                    parameters=self.parameters[: self.w_terms]
                    if self.parameters is not None
                    else None,
                    transform=tr,
                ),
                "tcos": Polynomial(
                    n_terms=self.w_terms,
                    parameters=self.parameters[self.w_terms : 2 * self.w_terms]
                    if self.parameters is not None
                    else None,
                    transform=tr,
                ),
                "tsin": Polynomial(
                    n_terms=self.w_terms,
                    parameters=self.parameters[2 * self.w_terms : 3 * self.w_terms]
                    if self.parameters is not None
                    else None,
                    transform=tr,
                ),
                "tload": Polynomial(
                    n_terms=self.c_terms,
                    parameters=self.parameters[3 * self.w_terms :]
                    if self.parameters is not None
                    else None,
                    transform=tr,
                ),
            },
            extra_basis={"tunc": K[1], "tcos": K[2], "tsin": K[3], "tload": -1},
        ).at(x=x)

    def get_noise_wave(
        self,
        noise_wave: str,
        parameters: Sequence | None = None,
        src: str | None = None,
    ) -> np.ndarray:
        """Get the model for a particular noise-wave term."""
        out = self.linear_model.model.get_model(
            noise_wave,
            parameters=parameters,
            x=self.linear_model.x,
            with_extra=bool(src),
        )
        if src:
            indx = self.src_names.index(src)
            return out[indx * len(self.freq) : (indx + 1) * len(self.freq)]
        else:
            return out[: len(self.freq)]

    def get_full_model(
        self, src: str, parameters: Sequence | None = None
    ) -> np.ndarray:
        """Get the full model (all noise-waves) for a particular input source."""
        out = self.linear_model(parameters=parameters)
        indx = self.src_names.index(src)
        return out[indx * len(self.freq) : (indx + 1) * len(self.freq)]

    def get_fitted(
        self, data: np.ndarray, weights: np.ndarray | None = None
    ) -> NoiseWaves:
        """Get a new noise wave model with fitted parameters."""
        fit = self.linear_model.fit(ydata=data, weights=weights)
        return attr.evolve(self, parameters=fit.model_parameters)

    def with_params_from_calobs(self, calobs) -> NoiseWaves:
        """Get a new noise wave model with parameters fitted using standard methods."""
        c2 = (-calobs.C2_poly.coefficients[::-1]).tolist()
        c2[0] += calobs.t_load

        params = (
            calobs.Tunc_poly.coefficients[::-1].tolist()
            + calobs.Tcos_poly.coefficients[::-1].tolist()
            + calobs.Tsin_poly.coefficients[::-1].tolist()
            + c2
        )

        return attr.evolve(self, parameters=params)

    def get_data_from_calobs(
        self, calobs, tns: Model | None = None, sim: bool = False
    ) -> np.ndarray:
        """Generate input data to fit from a calibration observation."""
        data = []
        for src in self.src_names:
            load = calobs._loads[src]
            if tns is None:
                _tns = calobs.C1() * calobs.t_load_ns
            else:
                _tns = tns(x=calobs.freq.freq)

            q = (
                simulate_q_from_calobs(calobs, load=src)
                if sim
                else load.spectrum.averaged_Q
            )
            c = calobs.get_K()[src][0]
            data.append(_tns * q - c * load.temp_ave)
        return np.concatenate(tuple(data))

    @classmethod
    def from_calobs(cls, calobs, cterms=None, wterms=None, sources=None) -> NoiseWaves:
        """Initialize a noise wave model from a calibration observation."""
        if sources is None:
            sources = tuple(calobs._loads.keys())

        loads = {src: load for src, load in calobs._loads.items() if src in sources}
        freq = calobs.freq.freq

        gamma_src = {name: source.s11_model(freq) for name, source in loads.items()}

        try:
            lna_s11 = calobs.lna.s11_model(freq)
        except AttributeError:
            lna_s11 = calobs.lna_s11(freq)

        nw_model = cls(
            freq=freq,
            gamma_src=gamma_src,
            gamma_rec=lna_s11,
            c_terms=cterms or calobs.cterms,
            w_terms=wterms or calobs.wterms,
        )
        if not cterms and not wterms:
            return nw_model.with_params_from_calobs(calobs)
        else:
            return nw_model

    def __call__(self, **kwargs) -> np.ndarray:
        """Call the underlying linear model."""
        return self.linear_model(**kwargs)


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
        scl = np.sqrt(np.square(van).sum())

        # Solve the least squares problem.
        return np.linalg.lstsq((van.T / scl), y.T, rcond)[0] / scl

    @cached_property
    def model_parameters(self):
        """The best-fit model parameters."""
        # Parameters need to be copied into this object, otherwise a new fit on the
        # parent model will change the model_parameters of this fit!
        return as_readonly(self.fit.model.parameters)

    def evaluate(self, x: np.ndarray | None = None) -> np.ndarray:
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

    @cached_property
    def hessian(self):
        """The Hessian matrix of the linear parameters."""
        return (self.model.basis * self.weights).dot(self.model.basis.T)

    @cached_property
    def parameter_covariance(self) -> np.ndarray:
        """The Covariance matrix of the parameters."""
        return np.linalg.inv(self.hessian)

    def get_sample(self, size: int | tuple[int] = 1):
        """Generate a random sample from the posterior distribution."""
        return np.random.multivariate_normal(
            mean=self.model_parameters, cov=self.parameter_covariance, size=size
        )


def _model_yaml_constructor(
    loader: yaml.SafeLoader, node: yaml.nodes.MappingNode
) -> Model:
    mapping = loader.construct_mapping(node, deep=True)
    model = get_mdl(mapping.pop("model"))
    return model(**mapping)


def _model_yaml_representer(
    dumper: yaml.SafeDumper, model: Model
) -> yaml.nodes.MappingNode:
    model_dct = attr.asdict(model, recurse=False)
    model_dct.update(model=model.__class__.__name__.lower())
    if model_dct["parameters"] is not None:
        model_dct["parameters"] = model_dct["parameters"].tolist()

    return dumper.represent_mapping("!Model", model_dct)


yaml.FullLoader.add_constructor("!Model", _model_yaml_constructor)

yaml.add_multi_representer(Model, _model_yaml_representer)
yaml.add_multi_representer(ModelTransform, _transform_yaml_representer)
