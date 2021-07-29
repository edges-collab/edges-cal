"""Tools to use in other modules."""
import numpy as np
import warnings
from itertools import product
from typing import List, Optional

from .cached_property import cached_property


def as_readonly(x: np.ndarray) -> np.ndarray:
    """Get a read-only view into an array without copying."""
    result = x.view()
    result.flags.writeable = False
    return result


def dct_of_list_to_list_of_dct(dct: dict) -> List:
    """Take a dict of key: list pairs and turn it into a list of all combinations of dicts.

    Parameters
    ----------
    dct : dict
        A dictionary for which each value is an iterable.

    Returns
    -------
    list :
        A list of dictionaries, each having the same keys as the input ``dct``, but
        in which the values are the elements of the original iterables.

    Examples
    --------
    >>> dct_of_list_to_list_of_dct(
    >>>    { 'a': [1, 2], 'b': [3, 4]}
    [
        {"a": 1, "b": 3},
        {"a": 1, "b": 4},
        {"a": 2, "b": 3},
        {"a": 2, "b": 4},
    ]
    """
    lists = dct.values()

    prod = product(*lists)

    return [{k: v for k, v in zip(dct.keys(), p)} for p in prod]


class FrequencyRange:
    def __init__(
        self,
        f: np.ndarray,
        f_low: Optional[float] = None,
        f_high: Optional[float] = None,
    ):
        """
        Class defining a set of frequencies.

        A given frequency range can be cut on either end.

        Parameters
        ----------
        f : array_like
            An array of frequencies defining a given spectrum.
        f_low : float
            A minimum frequency to keep in the array. Default is min(f).
        f_high : float
            A minimum frequency to keep in the array. Default is min(f).
        """
        self.freq_full = f
        self._f_high = f_high or f.max()
        self._f_low = f_low or f.min()

        if self._f_low >= self._f_high:
            raise ValueError("Cannot create frequency range: f_low >= f_high")

    @cached_property
    def n(self) -> int:
        """The number of frequencies in the (masked) array."""
        return len(self.freq)

    @cached_property
    def df(self) -> float:
        """Resolution of the frequencies."""
        if not np.allclose(np.diff(self.freq, 2), 0):
            warnings.warn(
                "Not all frequency intervals are even, so using df is ill-advised!"
            )
        return self.freq[1] - self.freq[0]

    @cached_property
    def min(self):  # noqa
        """Minimum frequency in the array."""
        return self.freq.min()

    @cached_property
    def max(self):  # noqa
        """Maximum frequency in the array."""
        return self.freq.max()

    @cached_property
    def mask(self):
        """Mask used to take input frequencies to output frequencies."""
        return np.logical_and(
            self.freq_full >= self._f_low, self.freq_full <= self._f_high
        )

    @cached_property
    def freq(self):
        """The frequency array."""
        return self.freq_full[self.mask]

    @cached_property
    def range(self):
        """Full range of the frequencies."""
        return self.max - self.min

    @cached_property
    def center(self):
        """The center of the frequency array."""
        return self.min + self.range / 2.0

    @cached_property
    def freq_recentred(self):
        """The frequency array re-centred so that it extends from -1 to 1."""
        return self.normalize(self.freq)

    def normalize(self, f):
        """
        Normalise a set of frequencies.

        Normalizes such that -1 aligns with ``min`` and +1 aligns with ``max``.

        Parameters
        ----------
        f : array_like
            Frequencies to normalize

        Returns
        -------
        array_like, shape [f,]
            The normalized frequencies.
        """
        return 2 * (f - self.center) / self.range

    def denormalize(self, f):
        """
        De-normalise a set of frequencies.

        Normalizes such that -1 aligns with ``min`` and +1 aligns with ``max``.

        Parameters
        ----------
        f : array_like
            Frequencies to de-normalize

        Returns
        -------
        array_like, shape [f,]
            The de-normalized frequencies.
        """
        return f * self.range / 2 + self.center


class EdgesFrequencyRange(FrequencyRange):
    def __init__(self, n_channels=16384 * 2, max_freq=200.0, **kwargs):
        """Subclass of :class:`FrequencyRange` specifying the default EDGES frequencies.

        Parameters
        ----------
        n_channels : int
            Number of channels
        max_freq : float
            Maximum frequency in original measurement.
        kwargs
            All other arguments passed through to :class:`FrequencyRange`.
        """
        f = self.get_edges_freqs(n_channels, max_freq)
        super().__init__(f, **kwargs)

    @staticmethod
    def get_edges_freqs(
        n_channels: int = 16384 * 2, max_freq: float = 200.0
    ) -> np.ndarray:
        """
        Return the raw EDGES frequency array, in MHz.

        Parameters
        ----------
        n_channels : int
            Number of channels in the EDGES spectrum
        max_freq : float
            Maximum frequency in the spectrum.

        Returns
        -------
        freqs: 1D-array
            full frequency array from 0 to 200 MHz, at raw resolution
        """
        df = max_freq / n_channels

        # This is correct. The channel width is the important thing.
        # The channel width is given by the FFT. We actually take
        # 32678*2 samples of data at 400 Mega-samples per second.
        # We only use the first half of the samples (since it's real input).
        # Regardless, the frequency channel width is thus
        # 400 MHz / (32678*2) == 200 MHz / 32678 ~ 6.103 kHz

        # The final frequency here will be slightly less than 200 MHz. 200 MHz
        # corresponds to the centre of the N+1 bin, which doesn't actually exist.
        return np.arange(0, max_freq, df)
