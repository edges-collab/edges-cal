# -*- coding: utf-8 -*-
"""Functions dealing with correcting S11's."""

import attr
import numpy as np
from cached_property import cached_property
from edges_io import io
from typing import Callable, Tuple, Union

from . import reflection_coefficient as rc
from .modelling import Model, Polynomial


def _read_data_and_corrections(switching_state: io.SwitchingState):

    # Standards assumed at the switch
    sw = {
        "open": 1 * np.ones_like(switching_state.freq),
        "short": -1 * np.ones_like(switching_state.freq),
        "match": np.zeros_like(switching_state.freq),
    }
    # Correction at the switch
    corrections = {
        kind: rc.de_embed(
            sw["open"],
            sw["short"],
            sw["match"],
            getattr(switching_state, "open").s11,
            getattr(switching_state, "short").s11,
            getattr(switching_state, "match").s11,
            getattr(switching_state, "external%s" % kind).s11,
        )[0]
        for kind in sw
    }
    return corrections, sw


def _tuplify(x):
    if not hasattr(x, "__len__"):
        return (int(x), int(x), int(x))
    else:
        return tuple(int(xx) for xx in x)


@attr.s
class InternalSwitch:
    data: io.SwitchingState = attr.ib()
    resistance: float = attr.ib(default=50.0)
    model: Model = attr.ib(default=Polynomial(n_terms=7))
    n_terms: Union[Tuple[int, int, int], int] = attr.ib(
        default=(7, 7, 7), converter=_tuplify
    )

    @cached_property
    def fixed_model(self):
        """The input model fixed to evaluate at the given frequencies."""
        return self.model.at(x=self.data.freq)

    @n_terms.validator
    def _n_terms_val(self, att, val):
        if len(val) != 3:
            raise TypeError(
                f"n_terms must be an integer or tuple of three integers "
                f"(for s11, s12, s22). Got {val}."
            )
        if any(not isinstance(v, (np.int, int)) for v in val):
            raise TypeError(f"n_terms must be integer, got {val}.")

    @cached_property
    def s11_data(self):
        """The measured S11."""
        return self._de_embedded_reflections[0]

    @cached_property
    def s12_data(self):
        """The measured S12."""
        return self._de_embedded_reflections[1]

    @cached_property
    def s22_data(self):
        """The measured S22."""
        return self._de_embedded_reflections[2]

    @cached_property
    def _s11_model(self):
        """The input unfit S11 model."""
        return self.fixed_model.with_nterms(n_terms=self.n_terms[0])

    @cached_property
    def _s12_model(self):
        """The input unfit S12 model."""
        return self.fixed_model.with_nterms(n_terms=self.n_terms[1])

    @cached_property
    def _s22_model(self):
        """The input unfit S22 model."""
        return self.fixed_model.with_nterms(n_terms=self.n_terms[2])

    @cached_property
    def s11_model(self) -> Callable:
        """The fitted S11 model."""
        return self._get_reflection_model("s11")

    @cached_property
    def s12_model(self) -> Callable:
        """The fitted S12 model."""
        return self._get_reflection_model("s12")

    @cached_property
    def s22_model(self) -> Callable:
        """The fitted S22 model."""
        return self._get_reflection_model("s22")

    @cached_property
    def _de_embedded_reflections(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get de-embedded reflection parameters for the internal switch."""
        corrections = _read_data_and_corrections(self.data)[0]

        # Computation of S-parameters to the receiver input
        oa, sa, la = rc.agilent_85033E(
            f=self.data.freq * 1e6, resistance_of_match=self.resistance
        )

        _, s11, s12s21, s22 = rc.de_embed(
            oa,
            sa,
            la,
            corrections["open"],
            corrections["short"],
            corrections["match"],
            corrections["open"],
        )

        return s11, s12s21, s22

    def _get_reflection_model(self, kind: str) -> Model:
        # 'kind' should be 's11', 's12' or 's22'
        data = getattr(self, f"{kind}_data")

        rl = getattr(self, f"_{kind}_model").fit(np.real(data))
        im = getattr(self, f"_{kind}_model").fit(np.imag(data))

        return lambda freq: rl.evaluate(freq) + 1j * im.evaluate(freq)
