"""Module defining composite models."""
from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property

import attrs
import numpy as np
import yaml
from hickleable import hickleable

from .. import noise_waves as rcf
from ..simulate import simulate_q_from_calobs
from . import fitting
from . import xtransforms as xt
from .core import FixedLinearModel, Model
from .models import Polynomial


@hickleable()
@attrs.define(frozen=True, kw_only=True, slots=False)
class CompositeModel:
    """
    Define a composite model from a set of sub-models.

    In totality, the resulting model is still
    """

    models: dict[str, Model] = attrs.field()

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

    def __getitem__(self, item):
        """Get sub-models as if they were top-level attributes."""
        if item not in self.models:
            raise KeyError(f"{item} not one of the models.")

        return self.models[item]

    def __getattr__(self, item):
        """Get sub-models as if they were top-level attributes."""
        if item not in self.models:
            raise AttributeError(f"{item} is not one of the models.")

        return self[item]

    def _get_model_param_indx(self, model: str):
        indx = list(self.models.keys()).index(model)
        n_before = sum(m.n_terms for m in list(self.models.values())[:indx])
        model = self.models[model]
        return slice(n_before, n_before + model.n_terms, 1)

    @cached_property
    def model_idx(self) -> dict[str, slice]:
        """Dictionary of parameter indices correponding to each model."""
        return {name: self._get_model_param_indx(name) for name in self.models}

    def get_model(
        self,
        model: str,
        parameters: np.ndarray = None,
        x: np.ndarray | None = None,
        with_scaler: bool = True,
    ):
        """Calculate a sub-model."""
        indx = self.model_idx[model]

        model = self.models[model]

        if parameters is None:
            parameters = self.parameters

        p = parameters if len(parameters) == model.n_terms else parameters[indx]
        return model(x=x, parameters=p, with_scaler=with_scaler)

    def get_basis_term(self, indx: int, x: np.ndarray) -> np.ndarray:
        """Define the basis terms for the model."""
        model, indx = self._index_map[indx]

        return self[model].get_basis_term(indx, x)

    def get_basis_term_transformed(self, indx: int, x: np.ndarray) -> np.ndarray:
        """Get the basis function term after coordinate tranformation."""
        model = self._index_map[indx][0]
        return self.get_basis_term(indx, x=self[model].transform(x))

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

        return attrs.evolve(self, models={**self.models, model: model_})

    def with_params(self, parameters: Sequence):
        """Get a new model with specified parameters."""
        assert len(parameters) == self.n_terms
        models = {
            name: model.with_params(
                parameters=parameters[self._get_model_param_indx(name)]
            )
            for name, model in self.models.items()
        }
        return attrs.evolve(self, models=models)

    def at(self, **kwargs) -> FixedLinearModel:
        """Get an evaluated linear model."""
        return FixedLinearModel(model=self, **kwargs)

    def __call__(
        self,
        x: np.ndarray | None = None,
        basis: np.ndarray | None = None,
        parameters: Sequence | None = None,
        indices: Sequence | None = None,
        with_scaler: bool = True,
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
            self,
            x=x,
            basis=basis,
            parameters=parameters,
            indices=indices,
            with_scaler=with_scaler,
        )

    def fit(
        self,
        xdata: np.ndarray,
        ydata: np.ndarray,
        weights: np.ndarray | float = 1.0,
        **kwargs,
    ) -> fitting.ModelFit:
        """Create a linear-regression fit object."""
        return self.at(x=xdata).fit(ydata, weights=weights, **kwargs)


@hickleable()
@attrs.define(frozen=True, slots=False)
class ComplexRealImagModel(yaml.YAMLObject):
    """A composite model that is specifically for complex functions in real/imag."""

    yaml_tag = "ComplexRealImagModel"

    real: Model | FixedLinearModel = attrs.field()
    imag: Model | FixedLinearModel = attrs.field()

    def at(self, **kwargs) -> FixedLinearModel:
        """Get an evaluated linear model."""
        return attrs.evolve(
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
        **kwargs,
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

        real = real.fit(np.real(ydata), weights=weights, **kwargs).fit
        imag = imag.fit(np.imag(ydata), weights=weights, **kwargs).fit
        return attrs.evolve(self, real=real, imag=imag)


@hickleable()
@attrs.define(frozen=True, slots=False)
class ComplexMagPhaseModel(yaml.YAMLObject):
    """A composite model that is specifically for complex functions in mag/phase."""

    yaml_tag = "ComplexMagPhaseModel"

    mag: Model | FixedLinearModel = attrs.field()
    phs: Model | FixedLinearModel = attrs.field()

    def at(self, **kwargs) -> FixedLinearModel:
        """Get an evaluated linear model."""
        return attrs.evolve(
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
        **kwargs,
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

        mag = mag.fit(np.abs(ydata), weights=weights, **kwargs).fit
        phs = phs.fit(np.unwrap(np.angle(ydata)), weights=weights, **kwargs).fit
        return attrs.evolve(self, mag=mag, phs=phs)


@hickleable()
@attrs.define(frozen=True, kw_only=True, slots=False)
class NoiseWaves:
    freq: np.ndarray = attrs.field()
    gamma_src: dict[str, np.ndarray] = attrs.field()
    gamma_rec: np.ndarray = attrs.field()
    c_terms: int = attrs.field(default=5)
    w_terms: int = attrs.field(default=6)
    parameters: Sequence | None = attrs.field(default=None)
    with_tload: bool = attrs.field(default=True)

    @cached_property
    def src_names(self) -> tuple[str]:
        """List of names of inputs sources (eg. ambient, hot_load, open, short)."""
        return tuple(self.gamma_src.keys())

    def get_linear_model(self, with_k: bool = True) -> CompositeModel:
        """Define and return a Model.

        Parameters
        ----------
        with_k
            Whether to use the K matrix as an "extra basis" in the linear model.
        """
        if with_k:
            # K should be a an array of shape (Nsrc Nnu x Nnoisewaveterms)
            K = np.hstack(
                tuple(
                    rcf.get_K(gamma_rec=self.gamma_rec, gamma_ant=s11src)
                    for s11src in self.gamma_src.values()
                )
            )

        # x is the frequencies repeated for every input source
        x = np.tile(self.freq, len(self.gamma_src))
        tr = xt.UnitTransform(range=(x.min(), x.max()))

        models = {
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
        }

        if with_k:
            extra_basis = {"tunc": K[1], "tcos": K[2], "tsin": K[3]}

        if self.with_tload:
            models["tload"] = Polynomial(
                n_terms=self.c_terms,
                parameters=self.parameters[3 * self.w_terms :]
                if self.parameters is not None
                else None,
                transform=tr,
            )

            if with_k:
                extra_basis["tload"] = -1 * np.ones(len(x))

        if with_k:
            return CompositeModel(models=models, extra_basis=extra_basis).at(x=x)
        return CompositeModel(models=models).at(x=x)

    @cached_property
    def linear_model(self) -> CompositeModel:
        """The actual composite linear model object associated with the noise waves."""
        return self.get_linear_model()

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
        return out[: len(self.freq)]

    def get_full_model(
        self, src: str, parameters: Sequence | None = None
    ) -> np.ndarray:
        """Get the full model (all noise-waves) for a particular input source."""
        out = self.linear_model(parameters=parameters)
        indx = self.src_names.index(src)
        return out[indx * len(self.freq) : (indx + 1) * len(self.freq)]

    def get_fitted(
        self, data: np.ndarray, weights: np.ndarray | None = None, **kwargs
    ) -> NoiseWaves:
        """Get a new noise wave model with fitted parameters."""
        fit = self.linear_model.fit(ydata=data, weights=weights, **kwargs)
        return attrs.evolve(self, parameters=fit.model_parameters)

    def with_params_from_calobs(self, calobs, cterms=None, wterms=None) -> NoiseWaves:
        """Get a new noise wave model with parameters fitted using standard methods."""
        cterms = cterms or calobs.cterms
        wterms = wterms or calobs.wterms

        def modify(thing, n):
            if isinstance(thing, np.ndarray):
                thing = thing.tolist()
            elif isinstance(thing, tuple):
                thing = list(thing)

            if len(thing) < n:
                return thing + [0] * (n - len(thing))
            if len(thing) > n:
                return thing[:n]
            return thing

        tu = modify(calobs.cal_coefficient_models["Tunc"].parameters, wterms)
        tc = modify(calobs.cal_coefficient_models["Tcos"].parameters, wterms)
        ts = modify(calobs.cal_coefficient_models["Tsin"].parameters, wterms)

        if self.with_tload:
            c2 = -np.asarray(calobs.cal_coefficient_models["C2"].parameters)
            c2[0] += calobs.t_load
            c2 = modify(c2, cterms)

        return attrs.evolve(self, parameters=tu + tc + ts + c2)

    def get_data_from_calobs(
        self,
        calobs,
        tns: Model | None = None,
        sim: bool = False,
        loads: dict | None = None,
    ) -> np.ndarray:
        """Generate input data to fit from a calibration observation."""
        if loads is None:
            loads = calobs.loads

        data = []
        for src in self.src_names:
            load = loads[src]
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
    def from_calobs(
        cls,
        calobs,
        cterms=None,
        wterms=None,
        sources=None,
        with_tload: bool = True,
        loads: dict | None = None,
    ) -> NoiseWaves:
        """Initialize a noise wave model from a calibration observation."""
        if loads is None:
            if sources is None:
                sources = calobs.load_names

            loads = {src: load for src, load in calobs.loads.items() if src in sources}

        freq = calobs.freq.freq.to_value("MHz")

        gamma_src = {name: source.s11_model(freq) for name, source in loads.items()}

        try:
            lna_s11 = calobs.receiver.s11_model(freq)
        except AttributeError:
            lna_s11 = calobs.receiver_s11(freq)

        nw_model = cls(
            freq=freq,
            gamma_src=gamma_src,
            gamma_rec=lna_s11,
            c_terms=cterms or calobs.cterms,
            w_terms=wterms or calobs.wterms,
            with_tload=with_tload,
        )
        return nw_model.with_params_from_calobs(calobs, cterms=cterms, wterms=wterms)

    def __call__(self, **kwargs) -> np.ndarray:
        """Call the underlying linear model."""
        return self.linear_model(**kwargs)
