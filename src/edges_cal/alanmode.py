"""Functions that run the calibration in a style similar to the C-code."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from astropy import units as un
from astropy.constants import c as speed_of_light
from read_acq.gsdata import read_acq_to_gsdata

from . import modelling as mdl
from . import reflection_coefficient as rc
from .calobs import CalibrationObservation, Load
from .loss import get_cable_loss_model
from .s11 import LoadS11, Receiver, StandardsReadings, VNAReading
from .spectra import LoadSpectrum
from .tools import FrequencyRange, dicke_calibration, gauss_smooth


def reads1p1(
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
    freq = standards.freq.freq

    calkit = rc.get_calkit(rc.AGILENT_ALAN, resistance_of_match=res * un.ohm)

    calkit = calkit.clone(
        short={"offset_delay": shortps * un.ps},
        open={"offset_delay": openps * un.ps},
        match={"offset_delay": loadps * un.ps},
    )

    calibrated = rc.de_embed(
        calkit.open.reflection_coefficient(freq),
        calkit.short.reflection_coefficient(freq),
        calkit.match.reflection_coefficient(freq),
        standards.open.s11,
        standards.short.s11,
        standards.match.s11,
        load.s11,
    )[0]
    return freq, calibrated


def corrcsv(
    freq: np.ndarray, s11: np.ndarray, cablen: float, cabdiel: float, cabloss: float
):
    """Corrects the S11 data (LNA) for cable effects.

    This function is a direct translation of the C-code function corrcsv.

    Parameters
    ----------
    freq : np.ndarray
        The frequency array.
    s11 : np.ndarray
        The S11 data.
    cablen : float
        The cable length, in inches.
    cabdiel : float
        The cable dielectric constant, as a percent.
    cabloss : float
        The cable loss, as a percent.
    """
    cable_length = (cablen * un.imperial.inch).to("m")

    _, cable_s11, cable_s12 = rc.path_length_correction_edges3(
        freq=freq,
        delay=cable_length / speed_of_light,
        gamma_in=0,
        lossf=1 + cabloss * 0.01,
        dielf=1 + cabdiel * 0.01,
    )

    if cable_length > 0.0:
        return cable_s11 + (cable_s12**2 * s11) / (1 - cable_s11 * s11)
    return (s11 - cable_s11) / (cable_s12**2 - cable_s11**2 + cable_s11 * s11)


def acqplot7amoon(
    acqfile: str | Path,
    fstart: float,
    fstop: float,
    smooth: int = 8,
    tload: float = 300.0,
    tcal: float = 1000.0,
    pfit: int | None = None,
    rfi: float | None = None,
    peakpwr: float | None = None,
    minpwr: float | None = None,
    pkpwrm: float | None = None,
    maxrmsf: float | None = None,
    maxfm: float | None = None,
):
    """A function that does what the acqplot7amoon C-code does."""
    # We raise/warn when non-implemented parameters are passed. Serves as a reminder
    # to implement them in the future as necessary
    if any(p is not None for p in (pfit, rfi, peakpwr, minpwr, pkpwrm, maxrmsf, maxfm)):
        warnings.warn(
            "pfit, rfi, peakpwr, minpwr, pkpwrm, maxrmsf, and maxfm are not yet "
            "implemented. This is almost certainly OK for calibration purposes, as no "
            "calibration load data is typically filtered out by these parameters.",
            stacklevel=2,
        )

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

    return freq.freq, len(q), tcal * np.mean(q, axis=0) + tload


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
    wfstart: float = 50,
    wfstop: float = 190,
    tcold: float = 306.5,
    thot: float = 393.22,
    tcab: float = 306.5,
    cfit: int = 7,
    wfit: int = 7,
    nfit3: int = 10,
    nfit2: int = 27,
    tload: float = 300,
    tcal: float = 1000.0,
    nter: int = 8,
    mfit: int | None = None,
    smooth: int | None = None,
    lmode: int | None = None,
    tant: float | None = None,
    ldb: float | None = None,
    adb: float | None = None,
    delaylna: float | None = None,
    nfit4: int | None = None,
):
    """A function that does what the edges3 C-code does."""
    # Some of the parameters are defined, but not yet implemented,
    # so we warn/error here. We do this explicitly because it serves as a
    # reminder to implement them in the future as necessary
    if mfit is not None or smooth is not None or tant is not None:
        warnings.warn(
            "mfit, smooth and tant are not used in this function, because "
            "they are only used for making output plots in the C-code."
            "They can be used in higher-level scripts instead. Continuing...",
            stacklevel=2,
        )

    if any(p is not None for p in (lmode, ldb, adb, delaylna, nfit4)):
        raise NotImplementedError(
            "lmode, ldb, adb, delaylna, and nfit4 are not yet implemented."
        )

    if not isinstance(spfreq, FrequencyRange):
        spfreq = FrequencyRange(spfreq)

    # First set up the S11 models
    sources = ["ambient", "hot_load", "open", "short"]
    s11_models = {}
    if not isinstance(s11freq, FrequencyRange):
        s11freq = FrequencyRange(
            s11freq, f_low=wfstart * un.MHz, f_high=wfstop * un.MHz
        )

    for name, s11 in zip(sources, [s11cold, s11hot, s11open, s11short]):
        s11_models[name] = LoadS11(
            raw_s11=s11[s11freq.mask],
            freq=s11freq,
            n_terms=nfit2,
            model_type=mdl.Fourier if nfit2 > 16 else mdl.Polynomial,
            complex_model_type=mdl.ComplexRealImagModel,
            model_transform=mdl.ZerotooneTransform(range=(1, 2))
            if nfit2 > 16
            else mdl.Log10Transform(scale=1),
            set_transform_range=True,
            fit_kwargs={"method": "alan-qrd"},
            internal_switch=None,
            model_kwargs={"period": 1.5},
        ).with_model_delay()

    receiver = Receiver(
        raw_s11=s11lna[s11freq.mask],
        freq=s11freq,
        n_terms=nfit3,
        model_type=mdl.Fourier if nfit3 > 16 else mdl.Polynomial,
        complex_model_type=mdl.ComplexRealImagModel,
        model_transform=mdl.ZerotooneTransform(range=(1, 2))
        if nfit3 > 16
        else mdl.Log10Transform(scale=120),
        set_transform_range=True,
        fit_kwargs={"method": "alan-qrd"},
    ).with_model_delay()

    specs = {}

    for name, spec, temp in zip(
        sources,
        [spcold, sphot, spopen, spshort],
        [tcold, thot, tcab, tcab],
    ):
        specs[name] = LoadSpectrum(
            freq=spfreq,
            q=(spec - tload) / tcal,
            variance=np.ones_like(spec),  # note: unused here
            n_integrations=1,  # unused
            temp_ave=temp,
            t_load_ns=tcal,
            t_load=tload,
        ).between_freqs(wfstart * un.MHz, wfstop * un.MHz)

    if Lh == -1:
        hot_loss_model = get_cable_loss_model(
            "UT-141C-SP", cable_length=4 * un.imperial.inch
        )
    else:
        hot_loss_model = None

    loads = {
        name: Load(
            spectrum=specs[name],
            reflections=s11_models[name],
            loss_model=hot_loss_model,
            ambient_temperature=tcold,
        )
        for name in specs
    }
    return CalibrationObservation(
        loads=loads,
        receiver=receiver,
        cterms=cfit,
        wterms=wfit,
        apply_loss_to_true_temp=False,
        smooth_scale_offset_within_loop=False,
        ncal_iter=nter,
        cable_delay_sweep=np.arange(0, -1e-8, -1e-9),  # hard-coded in the C code.
        fit_method="alan-qrd",
        scale_offset_poly_spacing=0.5,
    )


def read_s11_csv(fname) -> tuple[np.ndarray, np.ndarray]:
    """Read a CSV file containing S11 data in Alan's output format."""
    with open(fname) as fl:
        data = np.genfromtxt(fl, delimiter=",", skip_header=1, skip_footer=1)
        freq = data[:, 0]
        s11 = data[:, 1] + data[:, 2] * 1j
    return freq, s11


def read_spec_txt(fname):
    """Read an averaged-spectrum file, like the ones output by acqplot7amoon."""
    return np.genfromtxt(
        fname,
        names=["freq", "spectra", "weight"],
        comments="/",
        usecols=[0, 1, 2],
    )


def read_specal(fname):
    """Read a specal file, like the ones output by edges3(k)."""
    return np.genfromtxt(
        fname,
        names=[
            "freq",
            "s11lna_real",
            "s11lna_imag",
            "C1",
            "C2",
            "Tunc",
            "Tcos",
            "Tsin",
            "weight",
        ],
        usecols=(1, 3, 4, 6, 8, 10, 12, 14, 16),
    )
