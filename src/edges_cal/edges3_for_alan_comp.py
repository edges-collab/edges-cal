"""
Scripts for comparing alan's output with EDGES-3 from edges-cal
"""

from pathlib import Path

import astropy.units as un
import numpy as np
import pandas as pd

from edges_cal.alanmode import corrcsv, edges, read_s11_csv, reads1p1
from edges_cal.cli import _average_spectra, _inject_s11s, _make_plots


def get_calobs_edges3(
    cterms=6, wterms=5, fstart=48, fstop=198, wfstart=50, wfstop=190, nter=4, out="."
):
    print(f"Starting calibration for cterms = {cterms} and wterms ={wterms}")

    calobs = get_calobs_from_alancal(
        s11date="2022_319_14",
        specyear=2022,
        specday=316,
        datadir="/data5/edges/data/EDGES3_data/MRO/",
        out=out,
        redo_s11=True,
        redo_spectra=False,
        redo_cal=True,
        match_resistance=49.8,
        calkit_delays=33,
        load_delay=None,
        open_delay=None,
        short_delay=None,
        lna_cable_length=4.26,
        lna_cable_loss=-91.5,
        lna_cable_dielectric=-1.24,
        fstart=fstart,
        fstop=fstop,
        smooth=8,
        tload=300,
        tcal=1000,
        nter=nter,
        Lh=-1,
        wfstart=wfstart,
        wfstop=wfstop,
        tcold=306.5,
        thot=393.22,
        tcab=306.5,
        cfit=cterms,
        wfit=wterms,
        nfit3=10,
        nfit2=27,
        plot=False,
        avg_spectra_path=None,
        modelled_s11_path=None,
        inject_lna_s11=True,
        inject_source_s11s=True,
        tstart=0,
        tstop=24,
        delaystart=0,
        write_h5=True,
    )

    return calobs


def get_calobs_from_alancal(
    s11date,
    specyear,
    specday,
    datadir,
    out,
    redo_s11,
    redo_spectra,
    redo_cal,
    match_resistance,
    calkit_delays,
    load_delay,
    open_delay,
    short_delay,
    lna_cable_length,
    lna_cable_loss,
    lna_cable_dielectric,
    fstart,
    fstop,
    smooth,
    tload,
    tcal,
    nter,
    Lh,  # noqa: N803
    wfstart,
    wfstop,
    tcold,
    thot,
    tcab,
    cfit,
    wfit,
    nfit3,
    nfit2,
    plot,
    avg_spectra_path,
    modelled_s11_path,
    inject_lna_s11,
    inject_source_s11s,
    tstart,
    tstop,
    delaystart,
    write_h5,
):
    parameters = locals()
    output_file = "calobs_parameters.txt"
    with open(output_file, "w") as file:
        for key, value in parameters.items():
            file.write(f"{key}: {value}\n")

    """Run a calibration in as close a manner to Alan's code as possible.

    This exists mostly for being able to compare to Alan's memos etc in an easy way. It
    is much less flexible than using the library directly, and is not recommended for
    general use.

    This is supposed to emulate one of Alan's C-shell scripts, usually called "docal",
    and thus it runs a complete calibration, not just a single part. However, you can
    turn off parts of the calibration by setting the appropriate flags to False.

    Parameters
    ----------
    s11date
        A date-string of the form 2022_319_04 (if doing EDGES-3 cal) or a full path
        to a file containing all calibrated S11s (if doing EDGES-2 cal).
    specyear
        The year the spectra were taken in, if doing EDGES-3 cal. Otherwise, zero.
    specday
        The day the spectra were taken on, if doing EDGES-3 cal. Otherwise, zero.
    """
    loads = ("amb", "hot", "open", "short")
    datadir = Path(datadir)
    out = Path(out)

    if load_delay is None:
        load_delay = calkit_delays
    if open_delay is None:
        open_delay = calkit_delays
    if short_delay is None:
        short_delay = calkit_delays

    raws11s = {}
    for load in (*loads, "lna"):
        outfile = out / f"s11{load}.csv"
        if redo_s11 or not outfile.exists():
            print(f"Calibrating {load} S11")

            fstem = f"{s11date}_lna" if load == "lna" else s11date
            s11freq, raws11s[load] = reads1p1(
                Tfopen=Path(datadir) / f"{fstem}_O.s1p",
                Tfshort=Path(datadir) / f"{fstem}_S.s1p",
                Tfload=Path(datadir) / f"{fstem}_L.s1p",
                Tfant=Path(datadir) / f"{s11date}_{load}.s1p",
                res=match_resistance,
                loadps=load_delay,
                openps=open_delay,
                shortps=short_delay,
            )

            if load == "lna":
                # Correction for path length
                raws11s[load] = corrcsv(
                    s11freq,
                    raws11s[load],
                    lna_cable_length,
                    lna_cable_dielectric,
                    lna_cable_loss,
                )

            # write out the CSV file
            with open(out / f"s11{load}.csv", "w") as fl:
                fl.write("BEGIN\n")
                for freq, s11 in zip(s11freq, raws11s[load], strict=False):
                    fl.write(
                        f"{freq.to_value('MHz'):1.16e},{s11.real:1.16e},{s11.imag:1.16e}\n"
                    )
                fl.write("END")

        # Always re-read the S11's to match the precision of the C-code.
        print(f"Reading calibrated {load} S11")
        s11freq, raws11s[load] = read_s11_csv(outfile)
        s11freq <<= un.MHz

    lna = raws11s.pop("lna")

    # Now average the spectra
    spectra = {}
    specdate = f"{specyear:04}_{specday:03}"
    specfiles = {
        load: sorted(
            Path(f"{datadir}/mro/{load}/{specyear:04}").glob(f"{specdate}*{load}.acq")
        )
        for load in loads
    }
    spfreq, spectra = _average_spectra(
        specfiles,
        out,
        redo_spectra,
        avg_spectra_path,
        fstart=fstart,
        fstop=fstop,
        smooth=smooth,
        tload=tload,
        tcal=tcal,
        tstart=tstart,
        tstop=tstop,
        delaystart=delaystart,
    )

    # Now do the calibration
    outfile = out / "specal.txt"
    if not redo_cal and outfile.exists():
        return None

    print("Performing calibration")
    calobs = edges(
        spfreq=spfreq,
        spcold=spectra["amb"],
        sphot=spectra["hot"],
        spopen=spectra["open"],
        spshort=spectra["short"],
        s11freq=s11freq,
        s11cold=raws11s["amb"],
        s11hot=raws11s["hot"],
        s11open=raws11s["open"],
        s11short=raws11s["short"],
        s11lna=lna,
        Lh=Lh,
        nter=nter,
        wfstart=wfstart,
        wfstop=wfstop,
        tcold=tcold,
        thot=thot,
        tcab=tcab,
        cfit=cfit,
        wfit=wfit,
        nfit3=nfit3,
        nfit2=nfit2,
        tload=tload,
        tcal=tcal,
    )

    if modelled_s11_path:
        calobs = _inject_s11s(
            calobs, modelled_s11_path, loads, inject_lna_s11, inject_source_s11s
        )
    else:
        for name, load in calobs.loads.items():
            print(f"Using delay={load.reflections.model_delay} for load {name}")

    _make_plots(out, calobs, plot)

    if write_h5:
        h5file = out / "specal.h5"
        print(f"Writing calibration results to {h5file}")
        calobs.write(h5file)

    return calobs


def read_s11m(pth):
    _s11m = np.genfromtxt(pth, comments="#", names=True)
    s11m = {}
    freq = _s11m["freq"]
    for name in _s11m.dtype.names:
        if name == "freq":
            continue

        bits = name.split("_")
        cmp = bits[-1]
        load = "_".join(bits[:-1])

        if load not in s11m:
            s11m[load] = np.zeros(len(_s11m), dtype=complex)
        if cmp == "real":
            s11m[load] += _s11m[name]
        else:
            s11m[load] += _s11m[name] * 1j

    return freq, pd.DataFrame(s11m)


def calculate_rms(array, digits=3):
    """
    returns RMS of the array
    """
    rms = np.sqrt(np.mean(array**2))
    return round(rms, digits)


def get_log_liklihood(residuals, sigma):
    """
    Returns log-liklihood given the residuals data and model; and noise distribution
    """
    Nchan = len(residuals)

    term1 = -0.5 * Nchan * np.log(2 * np.pi)
    term2 = -Nchan * np.log(np.sum(sigma))
    term3 = -np.sum(residuals**2 / (2 * sigma**2))

    log_likelihood = term1 + term2 + term3

    return log_likelihood


def get_BIC(n, k, log_L):
    """
    computes BIC

    BIC = ln(n)*k -2*ln(L)

    n: No. of data points (Freq channels)
    k: Free parameters (C and W terms)
    L: Liklihood
    """
    # return np.log(n)*k - 2*np.log(np.abs(L))
    return np.log(n) * k - 2 * (log_L)


if __name__ == "__main__":
    get_calobs_edges3(
        cterms=7,
        wterms=7,
        fstart=48,
        fstop=198,
        nter=8,
        wfstart=50,
        wfstop=190,
        out="/data4/vydula/edges/edges3-data-analysis/scripts/frequency_tests/cw77/",
    )
