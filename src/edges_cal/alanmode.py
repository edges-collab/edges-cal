"""Functions that run the calibration in a style similar to the C-code."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy import units as un
from read_acq.gsdata import read_acq_to_gsdata

from . import reflection_coefficient as rc
from .s11 import StandardsReadings, VNAReading
from .tools import FrequencyRange, dicke_calibration, gauss_smooth


def reads1p1(
    load_name: str,
    res: float,
    Tfopen: str,
    Tfshort: str,
    Tfload: str,
    Tfant: str,
    loadps: float = 33.0,
    openps: float = 33.0,
    shortps: float = 33.0,
):
    """Reads the s1p1 file and returns the data."""
    standards = StandardsReadings(
        open=VNAReading.from_s1p(Tfopen),
        short=VNAReading.from_s1p(Tfshort),
        match=VNAReading.from_s1p(Tfload),
    )
    load = VNAReading.from_s1p(Tfant)
    freq = standards.freq

    calkit = rc.get_calkit(rc.AGILENT_ALAN, resistance_of_match=res * un.ohms)

    calkit = calkit.clone(
        short={"offset_delay": shortps * un.ps},
        open={"offset_delay": openps * un.ps},
        match={"offset_delay": loadps * un.ps},
    )

    calibrated = rc.de_embed(
        calkit.open.reflection_coefficient(freq.freq),
        calkit.short.reflection_coefficient(freq.freq),
        calkit.match.reflection_coefficient(freq.freq),
        standards.open.s11,
        standards.short.s11,
        standards.match.s11,
        load.s11,
    )[0]
    return freq, calibrated


def acqplot7amoon(
    acqfile: str | Path,
    fstart: float,
    fstop: float,
    pfit: int = 27,
    smooth: int = 8,
    rfi: float = 0,
    peakpwr: float = 10.0,
    minpwr: float = 1,
    pkpwrm: float = 40.0,
    maxrmsf: float = 400.0,
    maxfm: float = 200.0,
    tload: float = 300.0,
    tcal: float = 1000.0,
):
    """A function that does what the acqplot7amoon C-code does."""
    data = read_acq_to_gsdata(acqfile, telescope="edges-low")

    freq = FrequencyRange.from_edges(f_low=fstart * un.MHz, f_high=fstop * un.MHz)
    q = dicke_calibration(data).data[0, 0, :, freq.mask]

    freq = freq.decimate(
        bin_size=smooth,
        decimate_at=0,
        embed_mask=True,
    )

    if smooth > 0:
        q = gauss_smooth(q, size=smooth, decimate_at=0)

    return freq, tload * np.mean(q, axis=0) + tcal
