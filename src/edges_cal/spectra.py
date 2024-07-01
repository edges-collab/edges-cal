"""Module dealing with calibration spectra and thermistor measurements."""

from __future__ import annotations

import inspect
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any

import attr
import h5py
import hickle
import numpy as np
from astropy import units as un
from astropy.time import Time
from edges_io import io, io3
from edges_io import types as tp
from edges_io import utils as iou
from edges_io.logging import logger
from hickleable import hickleable
from pygsdata import GSData
from pygsdata.concat import concat
from pygsdata.select import select_freqs, select_times

from . import __version__, tools
from .cached_property import cached_property
from .config import config
from .tools import FrequencyRange, dicke_calibration, temperature_thermistor


@hickleable()
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
        ts = self.get_timestamps()
        return next(
            (
                i
                for i, t in enumerate(ts)
                if (t - ts[0]).seconds > self.ignore_times_percent
            ),
            len(ts),
        )

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
            times = self._data["time"]
            dates = self._data["date"]
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
        return temperature_thermistor(self.data["load_resistance"])

    def get_thermistor_indices(self, timestamps: Time) -> list[int | np.nan]:
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
                if d.datetime - td > timedelta(0) and d.datetime - td <= deltat:
                    closest.append(i)
                    break
                if d.datetime - td > timedelta(0):
                    indx += 1

            else:
                closest.append(np.nan)

        return closest


def flag_data_outside_temperature_range(
    temperature_range: float | tuple[float, float],
    spec_times: np.ndarray,
    thermistor: ThermistorReadings,
) -> np.ndarray:
    """Get a mask that flags data outside a temperature range."""
    thermistor_temp = thermistor.get_physical_temperature()
    thermistor_times = thermistor.get_timestamps()

    # Cut on temperature.
    if not hasattr(temperature_range, "__len__"):
        median = np.median(thermistor_temp)
        temp_range = (
            median - temperature_range / 2,
            median + temperature_range / 2,
        )
    else:
        temp_range = temperature_range

    temp_mask = np.zeros(len(spec_times), dtype=bool)
    for i, c in enumerate(thermistor.get_thermistor_indices(spec_times)):
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
            f"{(spec_times[0], spec_times[-1])}\n"
            f"Time Range of Thermistor: "
            f"{(thermistor_times[0], thermistor_times[-1])}"
        )

    return temp_mask


def get_ave_and_var_spec(
    data: GSData,
    load_name: str,
    thermistor: ThermistorReadings,
    frequency_smoothing: str,
    freq: FrequencyRange | None = None,
    ignore_times_percent: int | float = 0,
    freq_bin_size: int = 1,
    temperature_range: tuple[float, float] | None = None,
    time_coordinate_swpos: int | tuple[int, int] = 0,
) -> tuple[dict, dict, int]:
    """Get the mean and variance of the spectra.

    Parameters
    ----------
    frequency_smoothing
        How to average frequency bins together. Default is to merely bin them
        directly. Other options are 'gauss' to do Gaussian filtering (this is the
        same as Alan's C pipeline).
    """
    logger.info(f"Reducing {load_name} spectra...")

    if freq is not None:
        data = select_freqs(data, freq_range=(freq._f_low, freq._f_high))

    spec_timestamps = data.times[:, time_coordinate_swpos]  # jd

    try:
        base_time, time_coordinate_swpos = time_coordinate_swpos
    except Exception:
        base_time = time_coordinate_swpos

    if ignore_times_percent > 100.0:
        # Interpret as a number of seconds.

        # The first time could be measured from a different swpos than the one we are
        # measuring it to.
        t0 = data.times[0, base_time]  # what is base_time?

        t_elapsed = (spec_timestamps - t0) * 24 * 3600  # seconds

        if np.all(t_elapsed < ignore_times_percent):
            raise ValueError(
                "You would be ignoring all times! Check your ignore_times_percent value"
            )

        ignore_ninteg = np.argwhere(t_elapsed > ignore_times_percent)[0][0]
        ignore_times_percent = 100 * ignore_ninteg / len(spec_timestamps)
    else:
        ignore_ninteg = int(len(spec_timestamps) * ignore_times_percent / 100.0)

    data = select_times(
        data,
        time_range=(spec_timestamps[ignore_ninteg], spec_timestamps[-1]),
        load=data.loads[time_coordinate_swpos],
    )

    spec_timestamps = spec_timestamps[ignore_ninteg:]

    if temperature_range is not None:
        temp_mask = flag_data_outside_temperature_range(
            temperature_range, spec_timestamps, thermistor
        )
    else:
        temp_mask = np.ones(len(spec_timestamps), dtype=bool)

    q = dicke_calibration(data)

    if freq_bin_size > 1:
        if frequency_smoothing == "bin":
            spec = tools.bin_array(q.data, size=freq_bin_size)
        elif frequency_smoothing == "gauss":
            # We only really allow Gaussian smoothing so that we can match Alan's
            # pipeline. In that case, the frequencies actually kept start from the
            # 0th index, instead of taking the centre of each new bin. Thus we
            # set decimate_at = 0.
            spec = tools.gauss_smooth(q.data, size=freq_bin_size, decimate_at=0)
        else:
            raise ValueError("frequency_smoothing must be one of ('bin', 'gauss').")
    else:
        spec = q.data

    spec = spec[0, 0, temp_mask]

    mean = np.nanmean(spec, axis=0)
    variance = np.nanvar(spec, axis=0)
    n_intg = np.sum(temp_mask)

    return mean, variance, n_intg


@hickleable()
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
            "n_integrations": self.n_integrations,
            "f_low": self.freq.min,
            "f_high": self.freq.max,
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
        io_obs: io.CalibrationObservation | io3.CalibrationObservation,
        load_name: str,
        f_low=40.0 * un.MHz,
        f_high=np.inf * un.MHz,
        f_range_keep: tuple[tp.FreqType, tp.Freqtype] | None = None,
        freq_bin_size=1,
        ignore_times_percent: float = 5.0,
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
        if isinstance(io_obs, io.CalibrationObservation):
            spec = getattr(io_obs.spectra, load_name)
            res = getattr(io_obs.resistance, load_name)
        else:
            spec = io_obs.spectra[load_name]
            res = io3.get_mean_temperature(
                io_obs.temperature_table,
                load=load_name,
                start_time=spec["ancillary"]["time"][0],
                end_time=spec["ancillary"]["time"][-1],
            )

        freq = FrequencyRange.from_edges(
            f_low=f_low,
            f_high=f_high,
        )

        sig = inspect.signature(cls.from_io)
        lc = locals()
        defining_dict = {p: lc[p] for p in sig.parameters if p not in ["cls", "io_obs"]}
        defining_dict["spec"] = spec
        defining_dict["res"] = res

        hsh = iou.stable_hash(
            (*tuple(defining_dict.values()), __version__.split(".")[0])
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
        data: GSData = concat([s.get_data() for s in spec], axis="time")

        meanq, varq, n_integ = get_ave_and_var_spec(
            data=data,
            load_name=load_name,
            freq=freq,
            ignore_times_percent=ignore_times_percent,
            freq_bin_size=freq_bin_size,
            temperature_range=temperature_range,
            thermistor=thermistor,
            frequency_smoothing=frequency_smoothing,
            time_coordinate_swpos=time_coordinate_swpos,
        )

        if freq_bin_size > 0:
            freq = freq.decimate(
                bin_size=freq_bin_size,
                decimate_at=0 if frequency_smoothing == "gauss" else "centre",
                embed_mask=True,
            )

        if temperature is None:
            temperature = np.nanmean(thermistor.get_physical_temperature())

        out = cls(
            freq=freq,
            q=meanq,
            variance=varq,
            n_integrations=n_integ,
            temp_ave=temperature,
            metadata={
                "spectra_path": spec[0].path,
                "resistance_path": res.path,
                "freq_bin_size": freq_bin_size,
                "pre_smooth_freq_range": (f_low, f_high),
                "ignore_times_percent": ignore_times_percent,
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

    @classmethod
    def from_edges3(
        cls,
        io_obs: io3.CalibrationObservation,
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
        allow_closest_time: bool = False,
        cache_dir: str | Path | None = None,
        invalidate_cache: bool = False,
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
        allow_closest_time
            If True, allow the closest time in the temperature table that corresponds
            to the range of times in the spectra to be used if none are within the
            range.
        kwargs :
            All other arguments to :class:`LoadSpectrum`.

        Returns
        -------
        :class:`LoadSpectrum`.
        """
        if not invalidate_cache:
            sig = inspect.signature(cls.from_edges3)
            lc = locals()
            defining_dict = {
                p: lc[p] for p in sig.parameters if p not in ["cls", "invalidate_cache"]
            }
            hsh = iou.stable_hash(
                (*tuple(defining_dict.values()), __version__.split(".")[0])
            )

            cache_dir = cache_dir or config["cal"]["cache-dir"]
            if cache_dir is not None:
                cache_dir = Path(cache_dir)
                fname = cache_dir / f"{load_name}_{hsh}.h5"

                if fname.exists():
                    logger.info(f"Reading in cached integrated {load_name} spectra...")
                    return hickle.load(fname)

        spec: GSData = io_obs.get_spectra(load_name).get_data()
        if temperature is None:
            start = spec.times.min()
            end = spec.times.max()
            table = io_obs.get_temperature_table()

            if (
                not np.any((table["time"] >= start) & (table["time"] <= end))
                and allow_closest_time
            ):
                start = table["time"][np.argmin(np.abs(table["time"] - start))]
                end = table["time"][np.argmin(np.abs(table["time"] - end))]

            temperature = io3.get_mean_temperature(
                table,
                load=load_name,
                start_time=start,
                end_time=end,
            ).to_value("K")

        freq = FrequencyRange.from_edges(f_low=f_low, f_high=f_high)
        q = dicke_calibration(spec).data[0, 0, :, freq.mask]

        freq = freq.decimate(
            bin_size=freq_bin_size,
            decimate_at=0 if frequency_smoothing == "gauss" else "centre",
            embed_mask=True,
        )

        if freq_bin_size > 1:
            if frequency_smoothing == "bin":
                q = tools.bin_array(q, size=freq_bin_size)
            elif frequency_smoothing == "gauss":
                q = tools.gauss_smooth(q, size=freq_bin_size, decimate_at=0)
            else:
                raise ValueError("frequency_smoothing must be one of ('bin', 'gauss').")

        out = cls(
            freq=freq,
            q=q.mean(axis=0),
            variance=np.var(q, axis=0),
            n_integrations=q.shape[0],
            temp_ave=temperature,
            metadata={
                "spectra_path": io_obs.acq_files[load_name],
                "s11_paths": io_obs.s11_files[load_name],
                "freq_bin_size": freq_bin_size,
                "pre_smooth_freq_range": (f_low, f_high),
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
        freq = self.freq.clone(f_low=f_low, f_high=f_high)
        return attr.evolve(
            self,
            freq=freq,
            q=self.q[freq.mask],
            variance=self.variance[freq.mask],
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
