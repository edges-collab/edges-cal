"""Functions that run the calibration in a style similar to the C-code."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy import units as un
from read_acq.gsdata import read_acq_to_gsdata

from . import modelling as mdl
from . import reflection_coefficient as rc
from .cal_coefficients import Load
from .loss import get_cable_loss_model
from .s11 import LoadS11, StandardsReadings, VNAReading
from .spectra import LoadSpectrum
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


def edges3cal(
    spfreq: np.ndarray,
    spcold: np.ndarray,
    sphot: np.ndarray,
    spopen: np.ndarray,
    spshort: np.ndarray,
    s11freq: np.ndarray,
    s11hot: np.ndarray,
    s11cold: np.ndarray,
    s11lna: np.ndarray,
    s11open: np.ndarray,
    s11short: np.ndarray,
    Lh: int = -1,
    mfit: int = -1,
    smooth: int = 8,
    wfstart: float = 50,
    wfstop: float = 190,
    tcold: float = 306.5,
    thot: float = 393.22,
    lmode: int = -1,
    tant: float = 306.5,
    tcab: float = 306.5,
    cfit: int = 7,
    wfit: int = 7,
    nfit3: int = 10,
    ldb: float = 0.0,
    adb: float = 0.0,
    delaylna: float = 0e-12,
    nfit4: int = 27,
    nfit2: int = 27,
    tload: float = 300,
    tcal: float = 1000.0,
):
    # First set up the S11 models
    sources = ["ambient", "hot_load", "open", "short"]
    s11_models = {}
    s11freq_mask = np.logical_and((s11freq >= wfstart), (s11freq <= wfstop))

    for name, s11 in zip(sources, [s11cold, s11hot, s11open, s11short]):
        s11_models[name] = LoadS11(
            raw_s11=s11[s11freq_mask],
            freq=s11freq[s11freq_mask],
            n_terms=nfit2,
            model_type=mdl.Fourier if nfit2 > 16 else mdl.Polynomial,
            complex_model_type=mdl.ComplexRealImagModel,
            model_transform=mdl.ZerotooneTransform(range=(1, 2))
            if nfit2 > 16
            else mdl.Log10Transform(scale=1),
            set_transform_range=True,
            fit_kwargs={"method": "alan-qrd"},
        ).with_model_delay()

    receiver = LoadS11(
        raw_s11=s11lna[s11freq_mask],
        freq=s11freq[s11freq_mask],
        n_terms=nfit3,
        model_type=mdl.Fourier if nfit3 > 16 else mdl.Polynomial,
        complex_model_type=mdl.ComplexRealImagModel,
        model_transform=mdl.ZerotooneTransform(range=(1, 2))
        if nfit3 > 16
        else mdl.Log10Transform(scale=1),
        set_transform_range=True,
        fit_kwargs={"method": "alan-qrd"},
    ).with_model_delay()

    specs = {}

    for name, spec, temp in zip(
        specs,
        [spcold, sphot, spopen, spshort],
        [tcold, thot, tcab, tcab],
    ):
        specs[name] = LoadSpectrum(
            freq=spfreq,
            q=(spec - tcal) / tload,
            variance=np.ones_like(spec),  # note: unused here
            n_integrations=1,  # unused
            temp_ave=tcab,
            t_load_ns=tload,
            t_load=tcal,
        ).between_freqs(wfstart, wfstop)

    if Lh == -1:
        hot_loss_model = get_cable_loss_model(
            "UT-141C-SP", cable_length=4 * un.imperial.inch
        )
    else:
        hot_loss_model = None

    loads = {}
    for name in sources:
        loads[name] = Load(
            spectrum=specs[name],
            reflections=s11_models[name],
            loss_model=hot_loss_model,
            ambient_temperature=tcold,
        )
