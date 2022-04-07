"""Module dealing with calibration spectra and thermistor measurements."""
from __future__ import annotations

import attr
import h5py
import hickle
import inspect
import numpy as np
from astropy import units as un
from datetime import datetime, timedelta
from edges_io import h5, io
from edges_io import utils as iou
from edges_io.logging import logger
from functools import partial
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from . import receiver_calibration_func as rcf
from . import tools
from . import types as tp
from . import xrfi
from .cached_property import cached_property
from .config import config
from .tools import FrequencyRange


def read_spectrum(
    spec_obj: Sequence[io.Spectrum],
    freq: FrequencyRange | None = None,
    ignore_times: float | int = 0,
) -> dict[str, np.ndarray]:
    """
    Read the contents of the spectrum files into memory.

    Removes a starting percentage of times, and masks out certain frequencies.

    Returns
    -------
    dict :
        A dictionary of the contents of the file. Usually p0, p1, p2 (un-normalised
        powers of source, load, and load+noise respectively), and Q (the
        uncalibrated ratio).
    """
    if freq is None:
        freq = FrequencyRange.from_edges()

    data = [o.data for o in spec_obj]

    n_times = sum(len(d["time_ancillary"]["times"]) for d in data)
    nfreq = np.sum(freq.mask)
    out = {
        "p0": np.empty((nfreq, n_times)),
        "p1": np.empty((nfreq, n_times)),
        "p2": np.empty((nfreq, n_times)),
        "Q": np.empty((nfreq, n_times)),
    }

    if ignore_times < 1:
        index_start_spectra = int(ignore_times * n_times)
    else:
        assert isinstance(ignore_times, int)
        index_start_spectra = ignore_times

    for key, val in out.items():
        nn = 0
        for d in data:
            n = len(d["time_ancillary"]["times"])
            val[:, nn : (nn + n)] = d["spectra"][key][freq.mask]
            nn += n

        out[key] = val[:, index_start_spectra:]

    return out


def get_spectrum_ancillary(
    spec_obj: Sequence[io.Spectrum], ignore_times_percent: float = 0
) -> dict[str, np.ndarray]:
    """Ancillary data from the spectrum measurements."""
    anc = [s.data["time_ancillary"] for s in spec_obj]

    n_times = sum(len(a["times"]) for a in anc)

    index_start_spectra = int((ignore_times_percent / 100) * n_times)

    return {
        key: np.hstack(tuple(a[key].T for a in anc)).T[index_start_spectra:]
        for key in anc[0]
    }


@h5.hickleable()
@attr.s
class ThermistorReadings:
    """
    Object containing thermistor readings.

    Parameters
    ----------
    data
        The data array containing the readings.
    ignore_times_percent
        The fraction of readings to ignore at the start of the observation. If greater
        than 100, will be interpreted as being a number of seconds to ignore.
    """

    _data: np.ndarray = attr.ib()
    ignore_times_percent: float = attr.ib(0.0, validator=attr.validators.ge(0.0))

    @_data.validator
    def _data_vld(self, att, val):
        if "start_time" not in val.dtype.names:
            for key in ["time", "date", "load_resistance"]:
                if key not in val.dtype.names:
                    raise ValueError(
                        f"{key} must be in the data for ThermistorReadings"
                    )

    @cached_property
    def ignore_ntimes(self) -> int:
        """Number of time integrations to ignore from the start of the observation."""
        if self.ignore_times_percent <= 100.0:
            return int(len(self._data) * self.ignore_times_percent / 100)
        else:
            ts = self.get_timestamps()
            for i, t in enumerate(ts):
                if (t - ts[0]).seconds > self.ignore_times_percent:
                    break
            return i

    @property
    def data(self):
        """The associated data, without initial ignored times."""
        return self._data[self.ignore_ntimes :]

    @classmethod
    def from_io(cls, resistance_obj: io.Resistance, **kwargs) -> ThermistorReadings:
        """Generate the object from an io.Resistance object."""
        return cls(data=resistance_obj.read()[0], **kwargs)

    def get_timestamps(self) -> list[datetime]:
        """Timestamps of all the thermistor measurements."""
        if "time" in self._data.dtype.names:
            times = self.data["time"]
            dates = self.data["date"]
            times = [
                datetime.strptime(d + ":" + t, "%m/%d/%Y:%H:%M:%S")
                for d, t in zip(dates.astype(str), times.astype(str))
            ]
        else:
            times = [
                datetime.strptime(d.split(".")[0], "%m/%d/%Y %H:%M:%S")
                for d in self._data["start_time"].astype(str)
            ]

        return times

    def get_physical_temperature(self) -> np.ndarray:
        """The associated thermistor temperature in K."""
        return rcf.temperature_thermistor(self.data["load_resistance"])

    def get_thermistor_indices(self, timestamps) -> list[int | np.nan]:
        """Get the index of the closest therm measurement for each spectrum."""
        closest = []
        indx = 0
        thermistor_timestamps = self.get_timestamps()

        deltat = thermistor_timestamps[1] - thermistor_timestamps[0]

        for d in timestamps:
            if indx >= len(thermistor_timestamps):
                closest.append(np.nan)
                continue

            for i, td in enumerate(thermistor_timestamps[indx:], start=indx):

                if d - td > timedelta(0) and d - td <= deltat:
                    closest.append(i)
                    break
                if d - td > timedelta(0):
                    indx += 1

            else:
                closest.append(np.nan)

        return closest


def get_ave_and_var_spec(
    spec_obj,
    load_name,
    freq,
    ignore_times_percent,
    freq_bin_size,
    rfi_threshold,
    rfi_kernel_width_freq,
    temperature_range,
    thermistor,
    frequency_smoothing: str,
    time_coordinate_swpos: int | tuple[int, int] = 0,
) -> tuple[dict, dict, int]:
    """Get the mean and variance of the spectra.

    Parameters
    ----------
    freqeuncy_smoothing
        How to average frequency bins together. Default is to merely bin them
        directly. Other options are 'gauss' to do Gaussian filtering (this is the
        same as Alan's C pipeline).
    """
    logger.info(f"Reducing {load_name} spectra...")
    spec_anc = get_spectrum_ancillary(spec_obj, 0)

    try:
        base_time, time_coordinate_swpos = time_coordinate_swpos
    except Exception:
        base_time = time_coordinate_swpos

    spec_timestamps = spec_obj[0].data.get_times(
        str_times=spec_anc["times"], swpos=time_coordinate_swpos
    )

    if ignore_times_percent > 100.0:
        # Interpret as a number of seconds.

        # The first time could be measured from a different swpos than the one we are
        # measuring it to.
        if base_time == time_coordinate_swpos:
            t0 = spec_timestamps[0]
        else:
            t0 = spec_obj[0].data.get_times(
                str_times=spec_anc["times"][:1], swpos=base_time
            )[0]

        for i, t in enumerate(spec_timestamps):
            if (t - t0).seconds > ignore_times_percent:
                break
        ignore_times_percent = 100 * i / len(spec_timestamps)
        ignore_ninteg = i
    else:
        ignore_ninteg = int(len(spec_timestamps) * ignore_times_percent / 100.0)

    spectra = read_spectrum(spec_obj=spec_obj, freq=freq, ignore_times=ignore_ninteg)
    spec_anc = {k: v[ignore_ninteg:] for k, v in spec_anc.items()}
    spec_timestamps = spec_timestamps[ignore_ninteg:]
    thermistor_temp = thermistor.get_physical_temperature()
    thermistor_times = thermistor.get_timestamps()

    means = {}
    variances = {}

    if temperature_range is not None:
        # Cut on temperature.
        if not hasattr(temperature_range, "__len__"):
            median = np.median(thermistor_temp)
            temp_range = (
                median - temperature_range / 2,
                median + temperature_range / 2,
            )
        else:
            temp_range = temperature_range

        temp_mask = np.zeros(spectra["Q"].shape[1], dtype=bool)
        for i, c in enumerate(thermistor.get_thermistor_indices(spec_timestamps)):
            if np.isnan(c):
                temp_mask[i] = False
            else:
                temp_mask[i] = (thermistor_temp[c] >= temp_range[0]) & (
                    thermistor_temp[c] < temp_range[1]
                )

        if not np.any(temp_mask):
            raise RuntimeError(
                "The temperature range has masked all spectra!"
                f"Temperature Range Desired: {temp_range}.\n"
                "Temperature Range of Data: "
                f"{(thermistor_temp.min(), thermistor_temp.max())}\n"
                f"Time Range of Spectra: "
                f"{(spec_timestamps[0], spec_timestamps[-1])}\n"
                f"Time Range of Thermistor: "
                f"{(thermistor_times[0], thermistor_times[-1])}"
            )

    else:
        temp_mask = np.ones(spectra["Q"].shape[1], dtype=bool)

    for key, spec in spectra.items():
        # Weird thing where there are zeros in the spectra.
        # For the Q-ratio, zero values are perfectly fine.
        if key.lower() != "q":
            spec[spec == 0] = np.nan

        if freq_bin_size > 1:
            if frequency_smoothing == "bin":
                spec = tools.bin_array(spec.T, size=freq_bin_size).T
            elif frequency_smoothing == "gauss":
                # We only really allow Gaussian smoothing so that we can match Alan's
                # pipeline. In that case, the frequencies actually kept start from the
                # 0th index, instead of taking the centre of each new bin. Thus we
                # set decimate_at = 0.
                spec = tools.gauss_smooth(spec.T, size=freq_bin_size, decimate_at=0).T
            else:
                raise ValueError("frequency_smoothing must be one of ('bin', 'gauss').")
        spec[:, ~temp_mask] = np.nan

        mean = np.nanmean(spec, axis=1)
        var = np.nanvar(spec, axis=1)
        n_intg = spec.shape[1]

        nsample = np.sum(~np.isnan(spec), axis=1)

        width = max(1, rfi_kernel_width_freq // freq_bin_size)

        varfilt = xrfi.flagged_filter(var, size=2 * width + 1)
        resid = mean - xrfi.flagged_filter(mean, size=2 * width + 1)
        flags = np.logical_or(
            resid > rfi_threshold * np.sqrt(varfilt / nsample),
            var - varfilt > rfi_threshold * np.sqrt(2 * varfilt**2 / (nsample - 1)),
        )

        mean[flags] = np.nan
        var[flags] = np.nan

        means[key] = mean
        variances[key] = var

    return means, variances, n_intg


@h5.hickleable()
@attr.s(kw_only=True, frozen=True)
class LoadSpectrum:
    """A class representing a measured spectrum from some Load averaged over time.

    Parameters
    ----------
    freq
        The frequencies associated with the spectrum.
    q
        The measured power-ratios of the three-position switch averaged over time.
    variance
        The variance of *a single* time-integration as a function of frequency.
    n_integrations
        The number of integrations averaged over.
    temp_ave
        The average measured physical temperature of the load while taking spectra.
    t_load_ns
        The "assumed" temperature of the load + noise source
    t_load
        The "assumed" temperature of the load
    _metadata
        A dictionary of metadata items associated with the spectrum.
    """

    freq: FrequencyRange = attr.ib()
    q: np.ndarray = attr.ib(
        eq=attr.cmp_using(eq=partial(np.array_equal, equal_nan=True))
    )
    variance: np.ndarray | None = attr.ib(
        eq=attr.cmp_using(eq=partial(np.array_equal, equal_nan=True))
    )
    n_integrations: int = attr.ib()
    temp_ave: float = attr.ib()
    t_load_ns: float = attr.ib(300, 0)
    t_load: float = attr.ib(400.0)
    _metadata: dict[str, Any] = attr.ib(default=attr.Factory(dict))

    @q.validator
    @variance.validator
    def _q_vld(self, att, val):
        if val.shape != (self.freq.n,):
            raise ValueError(
                f"{att.name} must be one-dimensional with same length as un-masked "
                f"frequency. Got {val.shape}, needed ({self.freq.n},)"
            )

    @property
    def metadata(self) -> dict[str, Any]:
        """Metadata associated with the object."""
        return {
            **self._metadata,
            **{
                "n_integrations": self.n_integrations,
                "f_low": self.freq.min,
                "f_high": self.freq.max,
            },
        }

    @classmethod
    def from_h5(cls, path):
        """Read the contents of a .h5 file into a LoadSpectrum."""

        def read_group(grp):
            return cls(
                freq=FrequencyRange(grp["frequency"][...] * un.MHz),
                q=grp["Q_mean"][...],
                variance=grp["Q_var"],
                n_integrations=grp["n_integrations"],
                temp_ave=grp["temp_ave"],
                t_load_ns=grp["t_load_ns"],
                t_load=grp["t_load"],
                metadata=dict(grp.attrs),
            )

        if isinstance(path, (str, Path)):
            with h5py.File(path, "r") as fl:
                return read_group(fl)
        else:
            return read_group(path)

    @classmethod
    def from_io(
        cls,
        io_obs: io.CalibrationObservation,
        load_name: str,
        f_low=40.0 * un.MHz,
        f_high=np.inf * un.MHz,
        f_range_keep: tuple[tp.FreqType, tp.Freqtype] | None = None,
        freq_bin_size=1,
        ignore_times_percent: float = 5.0,
        rfi_threshold: float = 6.0,
        rfi_kernel_width_freq: int = 16,
        temperature_range: float | tuple[float, float] | None = None,
        frequency_smoothing: str = "bin",
        temperature: float | None = None,
        time_coordinate_swpos: int = 0,
        **kwargs,
    ):
        """Instantiate the class from a given load name and directory.

        Parameters
        ----------
        load_name : str
            The load name (one of 'ambient', 'hot_load', 'open' or 'short').
        direc : str or Path
            The top-level calibration observation directory.
        run_num : int
            The run number to use for the spectra.
        filetype : str
            The filetype to look for (acq or h5).
        freqeuncy_smoothing
            How to average frequency bins together. Default is to merely bin them
            directly. Other options are 'gauss' to do Gaussian filtering (this is the
            same as Alan's C pipeline).
        ignore_times_percent
            The fraction of readings to ignore at the start of the observation. If
            greater than 100, will be interpreted as being a number of seconds to
            ignore.
        kwargs :
            All other arguments to :class:`LoadSpectrum`.

        Returns
        -------
        :class:`LoadSpectrum`.
        """
        spec = getattr(io_obs.spectra, load_name)
        res = getattr(io_obs.resistance, load_name)

        freq = FrequencyRange.from_edges(
            f_low=f_low,
            f_high=f_high,
            bin_size=freq_bin_size,
            alan_mode=frequency_smoothing == "gauss",
        )

        sig = inspect.signature(cls.from_io)
        lc = locals()
        defining_dict = {p: lc[p] for p in sig.parameters if p not in ["cls", "io_obs"]}
        defining_dict["spec"] = spec
        defining_dict["res"] = res

        hsh = iou.stable_hash(
            tuple(defining_dict.values()) + (__version__.split(".")[0],)
        )

        cache_dir = config["cal"]["cache-dir"]
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            fname = cache_dir / f"{load_name}_{hsh}.h5"

            if fname.exists():
                logger.info(
                    f"Reading in previously-created integrated {load_name} spectra..."
                )
                return hickle.load(fname)

        thermistor = ThermistorReadings.from_io(
            res, ignore_times_percent=ignore_times_percent
        )
        means, variances, n_integ = get_ave_and_var_spec(
            spec_obj=spec,
            load_name=load_name,
            freq=freq,
            ignore_times_percent=ignore_times_percent,
            freq_bin_size=freq_bin_size,
            rfi_threshold=rfi_threshold,
            rfi_kernel_width_freq=rfi_kernel_width_freq,
            temperature_range=temperature_range,
            thermistor=thermistor,
            frequency_smoothing=frequency_smoothing,
            time_coordinate_swpos=time_coordinate_swpos,
        )

        if temperature is None:
            temperature = np.nanmean(thermistor.get_physical_temperature())

        out = cls(
            freq=freq,
            q=means["Q"],
            variance=variances["Q"],
            n_integrations=n_integ,
            temp_ave=temperature,
            metadata={
                "spectra_path": spec[0].path,
                "resistance_path": res.path,
                "freq_bin_size": freq_bin_size,
                "ignore_times_percent": ignore_times_percent,
                "rfi_threshold": rfi_threshold,
                "rfi_kernel_width_freq": rfi_kernel_width_freq,
                "temperature_range": temperature_range,
                "hash": hsh,
                "frequency_smoothing": frequency_smoothing,
            },
            **kwargs,
        )

        if f_range_keep is not None:
            out = out.between_freqs(*f_range_keep)

        if cache_dir is not None:
            if not cache_dir.exists():
                cache_dir.mkdir(parents=True)
            hickle.dump(out, fname)

        return out

    def between_freqs(self, f_low: tp.FreqType, f_high: tp.FreqType = np.inf * un.MHz):
        """Return a new LoadSpectrum that is masked between new frequencies."""
        mask = (self.freq.freq >= f_low) & (self.freq.freq <= f_high)
        return attr.evolve(
            self,
            freq=self.freq.with_new_mask(post_bin_f_low=f_low, post_bin_f_high=f_high),
            q=self.q[mask],
            variance=self.variance[mask],
        )

    @property
    def averaged_Q(self) -> np.ndarray:
        """Ratio of powers averaged over time.

        Notes
        -----
        The formula is

        .. math:: Q = (P_source - P_load)/(P_noise - P_load)
        """
        return self.q

    @property
    def variance_Q(self) -> np.ndarray:
        """Variance of Q across time (see averaged_Q)."""
        return self.variance

    @property
    def averaged_spectrum(self) -> np.ndarray:
        """T* = T_noise * Q  + T_load."""
        return self.averaged_Q * self.t_load_ns + self.t_load

    @property
    def variance_spectrum(self) -> np.ndarray:
        """Variance of uncalibrated spectrum across time (see averaged_spectrum)."""
        return self.variance_Q * self.t_load_ns**2
