"""Functions that run the calibration in a style similar to the C-code."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from astropy import units as un
from astropy.constants import c as speed_of_light
from pygsdata.select import select_times
from read_acq.gsdata import read_acq_to_gsdata

from . import modelling as mdl
from . import reflection_coefficient as rc
from .calobs import CalibrationObservation, Calibrator, Load
from .loss import HotLoadCorrection, get_cable_loss_model, get_loss_model_from_file
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
    smatrix = rc.SMatrix.from_calkit_and_vna(calkit, standards)
    calibrated = rc.gamma_de_embed(load.s11, smatrix)
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
    nrfi: int = 0,
    tstart: int = 0,
    tstop: int = 23,
    delaystart: int = 0,
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

    if tstart > 0 or tstop < 23:
        # Note that tstop=23 includes all possible hours since we have <=
        hours = np.array([x.hour for x in data.times[:, 0].datetime])
        data = select_times(data, indx=(hours >= tstart) & (hours <= tstop))

    if delaystart > 0:
        secs = (data.times - data.times.min()).sec
        idx = np.all(secs > delaystart, axis=1)
        data = select_times(data, indx=idx)

    freq = FrequencyRange.from_edges(f_low=fstart * un.MHz, f_high=fstop * un.MHz)
    q = dicke_calibration(data).data[0, 0, :, freq.mask]

    if smooth > 0:
        freq = freq.decimate(
            bin_size=smooth,
            decimate_at=0,
            embed_mask=True,
        )

        q = gauss_smooth(q, size=smooth, decimate_at=0)

    return freq.freq, len(q), tcal * np.mean(q, axis=0) + tload


def edges(
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
    tcab: float | None = None,
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
    s11rig: np.ndarray | None = None,
    s12rig: np.ndarray | None = None,
    s22rig: np.ndarray | None = None,
    lna_poly: int = -1,
):
    """A function that does what the edges3.c and edges2k.c C-code do.

    The primary purpose of this function is to model the input S11's, and then
    determine the noise-wave parameters.
    """
    # Some of the parameters are defined, but not yet implemented,
    # so we warn/error here. We do this explicitly because it serves as a
    # reminder to implement them in the future as necessary
    if tcab is None:
        tcab = tcold

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

    for name, s11 in zip(sources, [s11cold, s11hot, s11open, s11short], strict=False):
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

    mt = mdl.Fourier if (nfit3 > 16 or lna_poly == 0) else mdl.Polynomial

    receiver = Receiver(
        raw_s11=s11lna[s11freq.mask],
        freq=s11freq,
        n_terms=nfit3,
        model_type=mt,
        complex_model_type=mdl.ComplexRealImagModel,
        model_transform=mdl.ZerotooneTransform(range=(1, 2))
        if mt == mdl.Fourier
        else mdl.Log10Transform(scale=120),
        set_transform_range=True,
        fit_kwargs={"method": "alan-qrd"},
        model_kwargs={"period": 1.5} if mt == mdl.Fourier else {},
    ).with_model_delay()

    specs = {}

    for name, spec, temp in zip(
        sources,
        [spcold, sphot, spopen, spshort],
        [tcold, thot, tcab, tcab],
        strict=False,
    ):
        specs[name] = LoadSpectrum(
            freq=spfreq,
            q=(spec - tload) / tcal,
            variance=np.ones_like(spec),  # note: unused here
            n_integrations=1,  # unused
            temp_ave=temp,
            t_load_ns=tcal,
            t_load=tload,
        )
        # if not edges2kmode:
        specs[name] = specs[name].between_freqs(wfstart * un.MHz, wfstop * un.MHz)

    if Lh == -1:
        hot_loss_model = get_cable_loss_model("UT-141C-SP")
    elif Lh == -2:
        if s11rig is None or s12rig is None or s22rig is None:
            raise ValueError("must provide rigid cable s11/s12/s22 if Lh=-2")
        mdlopts = {
            "transform": (
                mdl.ZerotooneTransform(
                    range=(s11freq.min.to_value("MHz"), s11freq.max.to_value("MHz"))
                )
                if nfit2 > 16
                else mdl.Log10Transform(scale=1)
            ),
            "n_terms": nfit2,
        }
        if nfit2 > 16:
            mdlopts["period"] = 1.5

        hot_loss_model = HotLoadCorrection(
            freq=s11freq,
            raw_s11=s11rig,
            raw_s12s21=s12rig,
            raw_s22=s22rig,
            model=mdl.Fourier(**mdlopts) if nfit2 > 16 else mdl.Polynomial(**mdlopts),
            complex_model=mdl.ComplexRealImagModel,
        )
    elif isinstance(Lh, Path):
        hot_loss_model = get_loss_model_from_file(Lh)
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


def read_raul_s11_format(fname):
    """
    Read files containing S11's for all loads, LNA, and rigid cable, for EDGES-2.

    These files are outputs from Raul's pipeline, and were used as inputs for the C-code
    in EDGES-2 in some cases.
    """
    s11 = np.genfromtxt(
        fname,
        names=[
            "freq",
            "lna_rl",
            "lna_im",
            "amb_rl",
            "amb_im",
            "hot_rl",
            "hot_im",
            "open_rl",
            "open_im",
            "short_rl",
            "short_im",
            "s11rig_rl",
            "s11rig_im",
            "s12rig_rl",
            "s12rig_im",
            "s22rig_rl",
            "s22rig_im",
        ],
    )

    out = {
        "freq": s11["freq"],
    }
    for name in s11.dtype.names:
        if "_" not in name:
            continue

        if "_rl" in name:
            out[name.split("_")[0]] = s11[name] + 0j
        else:
            out[name.split("_")[0]] += 1j * s11[name]
    return out


def read_s11_csv(fname) -> tuple[np.ndarray, np.ndarray]:
    """Read a CSV file containing S11 data in Alan's output format."""
    with open(fname) as fl:
        data = np.genfromtxt(fl, delimiter=",", skip_header=1, skip_footer=1)
        freq = data[:, 0]
        s11 = data[:, 1] + data[:, 2] * 1j
    return freq, s11


def read_spec_txt(fname):
    """Read an averaged-spectrum file, like the ones output by acqplot7amoon."""
    out = np.genfromtxt(
        fname,
        names=["freq", "spectra", "weight"],
        comments="/",
        usecols=[0, 1, 2],
    )
    with open(fname) as fl:
        n = int(fl.readline()[31:].split(" ")[0])

    return out, n


def write_spec_txt(freq, n, spec, fname):
    """Write an averaged-spectrum file, like spe_<load>r.txt files from edges2k.c."""
    if hasattr(freq, "unit"):
        freq = freq.to_value("MHz")

    with open(fname, "w") as fl:
        for i, (f, sp) in enumerate(zip(freq, spec, strict=False)):
            if i == 0:
                fl.write(f"{f:12.6f} {sp:12.6f} {1:4.0f} {n} // temp.acq\n")
            else:
                fl.write(f"{f:12.6f} {sp:12.6f} {1:4.0f}\n")


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


def read_specal_as_calibrator(
    fname: str | Path, nfit1: int = 27, t_load: float = 300, t_load_ns: float = 1000
):
    """Read a specal.txt file format as a Calibrator object.

    Parameters
    ----------
    fname
        The path to the specal file.
    nfit1
        The number of terms in the model to fit through the points.
    t_load
        The load temperature assumed in the calibration.
    t_load_ns
        The load+noise-source temperature assumed in the calibration.
    """
    data = read_specal(fname)

    model_type = mdl.Fourier if nfit1 > 16 else mdl.Polynomial
    complex_model_type = mdl.ComplexRealImagModel
    mask = data["weight"] > 0
    data = data[mask]

    model_transform = (
        mdl.ZerotooneTransform(range=(data["freq"].min(), data["freq"].max()))
        if nfit1 > 16
        else mdl.Log10Transform(scale=data["freq"][len(data["freq"]) // 2])
    )
    fit_kwargs = {"method": "alan-qrd"}
    model_kwargs = {"period": 1.5} if nfit1 > 16 else {}

    model = model_type(transform=model_transform, n_terms=nfit1, **model_kwargs).at(
        x=data["freq"]
    )
    return Calibrator(
        freq=FrequencyRange(data["freq"] * un.MHz),
        C1=model.fit(ydata=data["C1"], weights=data["weight"], **fit_kwargs).fit,
        C2=model.fit(ydata=data["C2"], weights=data["weight"], **fit_kwargs).fit,
        Tunc=model.fit(ydata=data["Tunc"], weights=data["weight"], **fit_kwargs).fit,
        Tcos=model.fit(ydata=data["Tcos"], weights=data["weight"], **fit_kwargs).fit,
        Tsin=model.fit(ydata=data["Tsin"], weights=data["weight"], **fit_kwargs).fit,
        receiver_s11=complex_model_type(
            real=model.fit(
                ydata=data["s11lna_real"], weights=data["weight"], **fit_kwargs
            ).fit,
            imag=model.fit(
                ydata=data["s11lna_imag"], weights=data["weight"], **fit_kwargs
            ).fit,
        ),
        t_load=t_load,
        t_load_ns=t_load_ns,
    )


def write_specal(calobs, outfile, precision="10.6f"):
    """Write a specal file in the same format as those output by the C-code edges3.c."""
    with open(outfile, "w") as fl:
        for i in range(calobs.freq.n):
            sca = calobs.C1()
            ofs = calobs.C2()
            tlnau = calobs.Tunc()
            tlnac = calobs.Tcos()
            tlnas = calobs.Tsin()
            lna = calobs.receiver_s11
            fl.write(
                f"freq {calobs.freq.freq[i].to_value('MHz'):{precision}} "
                f"s11lna {lna[i].real:{precision}} {lna[i].imag:{precision}} "
                f"sca {sca[i]:{precision}} ofs {ofs[i]:{precision}} "
                f"tlnau {tlnau[i]:{precision}} tlnac {tlnac[i]:{precision}} "
                f"tlnas {tlnas[i]:{precision}} wtcal 1 cal_data\n"
            )


def write_modelled_s11s(calobs, fname):
    """Write all modelled S11's in a calobs object to file, in the same format as C.

    If a HotLoadCorrection exists, also write the rigid cable S-parameters, as
    edges2k.c does, otherwise assume the edges3.c format.
    """
    s11m = {
        name: load.s11_model(calobs.freq.freq) for name, load in calobs.loads.items()
    }
    lna = calobs.receiver_s11
    if isinstance(calobs.hot_load._loss_model, HotLoadCorrection):
        f = calobs.freq.freq.to_value("MHz")
        s11m |= {
            "rig_s11": calobs.hot_load._loss_model.s11_model(f),
            "rig_s12": calobs.hot_load._loss_model.s12s21_model(f),
            "rig_s22": calobs.hot_load._loss_model.s22_model(f),
        }

    with open(fname, "w") as fl:
        if "rig_s11" in s11m:
            fl.write(
                "# freq, amb_real amb_imag hot_real hot_imag open_real open_imag "
                "short_real short_imag lna_real lna_imag rig_s11_real rig_s11_imag "
                "rig_s12_real rig_s12_imag rig_s22_real rig_s22_imag\n"
            )
            for i, (f, amb, hot, op, sh, rigs11, rigs12, rigs22) in enumerate(
                zip(
                    calobs.freq.freq.to_value("MHz"),
                    s11m["ambient"],
                    s11m["hot_load"],
                    s11m["open"],
                    s11m["short"],
                    s11m["rig_s11"],
                    s11m["rig_s12"],
                    s11m["rig_s22"],
                    strict=False,
                )
            ):
                fl.write(
                    f"{f} {amb.real} {amb.imag} "
                    f"{hot.real} {hot.imag} "
                    f"{op.real} {op.imag} "
                    f"{sh.real} {sh.imag} "
                    f"{lna[i].real} {lna[i].imag} "
                    f"{rigs11.real} {rigs11.imag} "
                    f"{rigs12.real} {rigs12.imag} "
                    f"{rigs22.real} {rigs22.imag}\n"
                )

        else:
            fl.write(
                "# freq, amb_real amb_imag hot_real hot_imag open_real open_imag "
                "short_real short_imag lna_real lna_imag\n"
            )
            for i, (f, amb, hot, op, sh) in enumerate(
                zip(
                    calobs.freq.freq.to_value("MHz"),
                    s11m["ambient"],
                    s11m["hot_load"],
                    s11m["open"],
                    s11m["short"],
                    strict=False,
                )
            ):
                fl.write(
                    f"{f} {amb.real} {amb.imag} "
                    f"{hot.real} {hot.imag} "
                    f"{op.real} {op.imag} "
                    f"{sh.real} {sh.imag} "
                    f"{lna[i].real} {lna[i].imag}\n"
                )


def read_spe_file(filename):
    """Read Alan's spectrum files with formats like those of spe0.txt."""
    return np.genfromtxt(
        filename,
        usecols=(1, 3, 6, 9, 12),
        names=("freq", "tant", "model", "resid", "weight"),
    )
