"""CLI functions for edges-cal."""

import json
import os
from datetime import datetime
from datetime import timezone as tz
from importlib.util import find_spec
from pathlib import Path

import click
import numpy as np
import papermill as pm
import yaml
from astropy import units as un
from nbconvert import PDFExporter
from rich.console import Console
from traitlets.config import Config

from edges_cal import calobs as cc
from edges_cal.alanmode import (
    acqplot7amoon,
    corrcsv,
    edges3cal,
    read_s11_csv,
    read_spec_txt,
    reads1p1,
)
from edges_cal.config import config

console = Console()

main = click.Group()


@main.command()
@click.argument(
    "settings", type=click.Path(dir_okay=False, file_okay=True, exists=True)
)
@click.argument("path", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.option(
    "-o",
    "--out",
    type=click.Path(dir_okay=True, file_okay=False, exists=True),
    default=".",
    help="output directory",
)
@click.option(
    "-g",
    "--global-config",
    type=str,
    default=None,
    help="json string representing global configuration options",
)
@click.option(
    "-p/-P",
    "--plot/--no-plot",
    default=True,
    help="whether to make diagnostic plots of calibration solutions.",
)
@click.option(
    "-s",
    "--simulators",
    multiple=True,
    default=[],
    help="antenna simulators to create diagnostic plots for.",
)
def run(settings, path, out, global_config, plot, simulators):
    """Calibrate using lab measurements in PATH, and make all relevant plots."""
    out = Path(out)

    if global_config:
        config.update(json.loads(global_config))

    obs = cc.CalibrationObservation.from_yaml(settings, obs_path=path)
    io_obs = obs.metadata["io"]
    if plot:
        # Plot Calibrator properties
        fig = obs.plot_raw_spectra()
        fig.savefig(out / "raw_spectra.png")

        ax = obs.plot_s11_models()
        fig = ax.flatten()[0].get_figure()
        fig.savefig(out / "s11_models.png")

        fig = obs.plot_calibrated_temps(bins=256)
        fig.savefig(out / "calibrated_temps.png")

        fig = obs.plot_coefficients()
        fig.savefig(out / "calibration_coefficients.png")

        # Calibrate and plot antsim
        for name in simulators:
            antsim = obs.new_load(load_name=name, io_obj=obs.metadata["io"])
            fig = obs.plot_calibrated_temp(antsim, bins=256)
            fig.savefig(out / f"{name}_calibrated_temp.png")

    # Write out data
    obs.write(out / io_obs.path.parent.name)


@main.command()
@click.argument("config", type=click.Path(dir_okay=False, file_okay=True, exists=True))
@click.argument("path", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.option(
    "-w", "--max-wterms", type=int, default=20, help="maximum number of wterms"
)
@click.option(
    "-r/-R",
    "--repeats/--no-repeats",
    default=False,
    help="explore repeats of switch and receiver s11",
)
@click.option(
    "-n/-N", "--runs/--no-runs", default=False, help="explore runs of s11 measurements"
)
@click.option(
    "-c", "--max-cterms", type=int, default=20, help="maximum number of cterms"
)
@click.option(
    "-w", "--max-wterms", type=int, default=20, help="maximum number of wterms"
)
@click.option(
    "-r/-R",
    "--repeats/--no-repeats",
    default=False,
    help="explore repeats of switch and receiver s11",
)
@click.option(
    "-n/-N", "--runs/--no-runs", default=False, help="explore runs of s11 measurements"
)
@click.option(
    "-t",
    "--delta-rms-thresh",
    type=float,
    default=0,
    help="threshold marking rms convergence",
)
@click.option(
    "-o",
    "--out",
    type=click.Path(dir_okay=True, file_okay=False, exists=True),
    default=".",
    help="output directory",
)
@click.option(
    "-c",
    "--cache-dir",
    type=click.Path(dir_okay=True, file_okay=False),
    default=".",
    help="directory in which to keep/search for the cache",
)
def sweep(
    config,
    path,
    max_cterms,
    max_wterms,
    repeats,
    runs,
    delta_rms_thresh,
    out,
    cache_dir,
):
    """Perform a sweep of number of terms to obtain the best parameter set."""
    with open(config) as fl:
        settings = yaml.load(fl, Loader=yaml.FullLoader)

    if cache_dir != ".":
        settings.update(cache_dir=cache_dir)

    obs = cc.CalibrationObservation(path=path, **settings)

    cc.perform_term_sweep(
        obs,
        direc=out,
        verbose=True,
        max_cterms=max_cterms,
        max_wterms=max_wterms,
        explore_repeat_nums=repeats,
        explore_run_nums=runs,
        delta_rms_thresh=delta_rms_thresh,
    )


@main.command()
@click.argument(
    "cal-settings",
    type=click.Path(dir_okay=False, file_okay=True, exists=True),
)
@click.argument("path", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.option(
    "-o",
    "--out",
    type=click.Path(dir_okay=True, file_okay=False, exists=True),
    default=None,
    help="output directory",
)
@click.option(
    "-g",
    "--global-config",
    type=str,
    default=None,
    help="json string representing global configuration options",
)
@click.option("-r/-R", "--report/--no-report", default=True)
@click.option("-u/-U", "--upload/--no-upload", default=False, help="auto-upload file")
@click.option("-t", "--title", type=str, help="title of the memo", default=None)
@click.option(
    "-a",
    "--author",
    type=str,
    help="adds an author to the author list",
    default=None,
    multiple=True,
)
@click.option("-n", "--memo", type=int, help="which memo number to use", default=None)
@click.option("-q/-Q", "--quiet/--loud", default=False)
@click.option("-p/-P", "--pdf/--no-pdf", default=True)
def report(
    cal_settings,
    path,
    out,
    global_config,
    report,
    upload,
    title,
    author,
    memo,
    quiet,
    pdf,
):
    """Make a full notebook report on a given calibration."""
    single_notebook = Path(__file__).parent / "notebooks/calibrate-observation.ipynb"

    console.print(f"Creating report for '{path}'...")

    path = Path(path)

    out = path / "outputs" if out is None else Path(out)

    if not out.exists():
        out.mkdir()

    # Describe the filename...
    fname = Path(
        f"calibration_{datetime.now(tz=tz.utc).strftime('%Y-%m-%d-%H.%M.%S')}.ipynb"
    )

    global_config = json.loads(global_config) if global_config else {}

    settings = {
        "observation": str(path),
        "settings": cal_settings,
        "global_config": global_config,
    }

    console.print("Settings:")
    with open(cal_settings) as fl:
        console.print(fl.read())

    # This actually runs the notebook itself.
    pm.execute_notebook(
        str(single_notebook),
        out / fname,
        parameters=settings,
        kernel_name="edges",
        log_output=True,
    )

    console.print(f"Saved interactive notebook to '{out / fname}'")

    if pdf:  # pragma: no cover
        make_pdf(out / fname)
        if upload:
            upload_memo(out / fname.with_suffix(".pdf"), title, memo, quiet)


@main.command()
@click.argument(
    "cal-settings",
    type=click.Path(dir_okay=False, file_okay=True, exists=True),
)
@click.argument("path", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.argument(
    "cmp-settings",
    type=click.Path(dir_okay=False, file_okay=True, exists=True),
)
@click.argument("cmppath", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.option(
    "-o",
    "--out",
    type=click.Path(dir_okay=True, file_okay=False, exists=True),
    default=None,
    help="output directory",
)
@click.option(
    "-g",
    "--global-config",
    type=str,
    default=".",
    help="global configuration options as json",
)
@click.option("-r/-R", "--report/--no-report", default=True)
@click.option("-u/-U", "--upload/--no-upload", default=False, help="auto-upload file")
@click.option("-t", "--title", type=str, help="title of the memo", default=None)
@click.option(
    "-a",
    "--author",
    type=str,
    help="adds an author to the author list",
    default=None,
    multiple=True,
)
@click.option("-n", "--memo", type=int, help="which memo number to use", default=None)
@click.option("-q/-Q", "--quiet/--loud", default=False)
@click.option("-p/-P", "--pdf/--no-pdf", default=True)
def compare(
    cal_settings,
    path,
    cmp_settings,
    cmppath,
    out,
    global_config,
    report,
    upload,
    title,
    author,
    memo,
    quiet,
    pdf,
):
    """Make a full notebook comparison report between two observations."""
    single_notebook = Path(__file__).parent / "notebooks/compare-observation.ipynb"

    console.print(f"Creating comparison report for '{path}' compared to '{cmppath}'")

    path = Path(path)
    cmppath = Path(cmppath)

    out = path / "outputs" if out is None else Path(out)

    if not out.exists():
        out.mkdir()

    # Describe the filename...
    fname = Path(
        f"calibration-compare-{cmppath.name}_"
        f"{datetime.now(tz=tz.utc).strftime('%Y-%m-%d-%H.%M.%S')}.ipynb"
    )

    global_config = json.loads(global_config) if global_config else {}

    console.print("Settings for Primary:")
    with open(cal_settings) as fl:
        console.print(fl.read())

    console.print("Settings for Comparison:")
    with open(cmp_settings) as fl:
        console.print(fl.read())

    # This actually runs the notebook itself.
    pm.execute_notebook(
        str(single_notebook),
        out / fname,
        parameters={
            "observation": str(path),
            "cmp_observation": str(cmppath),
            "settings": cal_settings,
            "cmp_settings": cmp_settings,
            "global_config": global_config,
        },
        kernel_name="edges",
    )
    console.print(f"Saved interactive notebook to '{out / fname}'")

    # Now output the notebook to pdf
    if pdf:  # pragma: no cover
        pdf = make_pdf(out / fname)
        if upload:
            upload_memo(pdf, title, memo, quiet)


def make_pdf(ipy_fname) -> Path:
    """Make a PDF out of an ipynb."""
    # Now output the notebook to pdf
    c = Config()
    c.TemplateExporter.exclude_input_prompt = True
    c.TemplateExporter.exclude_output_prompt = True
    c.TemplateExporter.exclude_input = True

    exporter = PDFExporter(config=c)
    body, _resources = exporter.from_filename(ipy_fname)
    with open(ipy_fname.with_suffix(".pdf"), "wb") as fl:
        fl.write(body)

    out = ipy_fname.with_suffix(".pdf")
    console.print(f"Saved PDF to '{out}'")
    return out


def upload_memo(fname, title, memo, quiet):  # pragma: no cover
    """Upload as memo to loco.lab.asu.edu."""
    try:
        find_spec("upload_memo")
    except ImportError as e:
        raise ImportError(
            "You need to manually install upload-memo to use this option."
        ) from e

    opts = ["memo", "upload", "-f", str(fname)]
    if title:
        opts.extend(["-t", title])

    if memo:
        opts.extend(["-n", memo])
    if quiet:
        opts.append("-q")

    run(opts)


@main.command()
@click.argument("s11date", type=str)
@click.argument("specyear", type=int)
@click.argument("specday", type=int)
@click.option(
    "-d",
    "--datadir",
    type=click.Path(dir_okay=True, file_okay=False, exists=True),
    default="/data5/edges/data/EDGES3_data/MRO/",
)
@click.option(
    "-o",
    "--out",
    type=click.Path(dir_okay=True, file_okay=False, exists=True),
    default=".",
    help="output directory",
)
@click.option("--redo-s11/--no-s11", default=None)
@click.option("--redo-spectra/--no-spectra", default=None)
@click.option("--redo-cal/--no-cal", default=None)
@click.option("-res", "--match-resistance", type=float, default=50.0)
@click.option("-ps", "--calkit-delays", type=float, default=33.0, help="in nanoseconds")
@click.option(
    "-loadps",
    "--load-delay",
    type=float,
    default=None,
    help="in nanoseconds. Overrides -ps.",
)
@click.option(
    "-openps",
    "--open-delay",
    type=float,
    default=None,
    help="in nanoseconds. Overrides -ps.",
)
@click.option(
    "-shortps",
    "--short-delay",
    type=float,
    default=None,
    help="in nanoseconds. Overrides -ps.",
)
@click.option(
    "-cablen",
    "--lna-cable-length",
    type=float,
    default=4.26,
    help="in inches",
)
@click.option(
    "-cabloss",
    "--lna-cable-loss",
    type=float,
    default=-1.24,
    help="as percent",
)
@click.option(
    "-cabdiel",
    "--lna-cable-dielectric",
    type=float,
    default=-91.5,
    help="as percent",
)
@click.option(
    "-fstart",
    type=float,
    default=48.0,
    help="in mhz",
)
@click.option(
    "-fstop",
    type=float,
    default=198.0,
    help="in mhz",
)
@click.option(
    "-smooth",
    type=int,
    default=8,
)
@click.option(
    "-tload",
    type=float,
    default=300.0,
    help="guess at the load temp",
)
@click.option(
    "-tcal",
    type=float,
    default=1000.0,
    help="guess at the load+noise source temp",
)
@click.option("-Lh", "Lh", type=int, default=-1)
@click.option("-wfstart", type=float, default=50.0)
@click.option("-wfstop", type=float, default=190.0)
@click.option("-tcold", type=float, default=306.5)
@click.option("-thot", type=float, default=393.22)
@click.option("-tcab", type=float, default=306.5)
@click.option("-cfit", type=int, default=7)
@click.option(
    "-wfit",
    type=int,
    default=7,
)
@click.option(
    "-nfit3",
    type=int,
    default=10,
)
@click.option(
    "-nfit2",
    type=int,
    default=27,
)
@click.option(
    "--plot/--no-plot",
    default=True,
)
@click.option(
    "--avg-spectra-path",
    type=click.Path(dir_okay=False, file_okay=True, exists=True),
    help=(
        "Path to a file containing averaged spectra in the format output by this "
        "script (or the C code)"
    ),
)
@click.option(
    "--modelled-s11-path",
    type=click.Path(dir_okay=False, file_okay=True, exists=True),
    help=(
        "path to a file containing modelled S11s in the format output by this "
        "script (or the C code)"
    ),
)
@click.option(
    "--inject-lna-s11/--no-inject-lna-s11",
    default=True,
    help="inject LNA s11 form modelled_s11_path (if given)",
)
@click.option(
    "--inject-source-s11s/--no-inject-source-s11s",
    default=True,
    help="inject source s11s from modelled_s11_path (if given)",
)
def alancal(
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
):
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
        A date-string of the form 2022_319_04
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
            console.print(f"Calibrating {load} S11")

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
                for freq, s11 in zip(s11freq, raws11s[load]):
                    fl.write(
                        f"{freq.to_value('MHz'):1.16e},{s11.real:1.16e},{s11.imag:1.16e}\n"
                    )
                fl.write("END")
        else:
            console.print(f"Reading calibrated {load} S11")

            s11freq, raws11s[load] = read_s11_csv(outfile)
            s11freq <<= un.MHz

    lna = raws11s.pop("lna")

    # Now average the spectra
    spectra = {}
    for load in loads:
        outfile = out / f"sp{load}.txt"
        if (redo_spectra or not outfile.exists()) and not avg_spectra_path:
            console.print(f"Averaging {load} spectra")

            specdate = f"{specyear:04}_{specday:03}"
            d = f"{datadir}/mro/{load}/{specyear:04}/{specdate}*{load}.acq"
            os.system(f"cat {d} > {out}/temp.acq")
            spfreq, n, spectra[load] = acqplot7amoon(
                acqfile=out / "temp.acq",
                fstart=fstart,
                fstop=fstop,
                smooth=smooth,
                tload=tload,
                tcal=tcal,
            )

            with open(outfile, "w") as fl:
                for i, (freq, spec) in enumerate(zip(spfreq, spectra[load])):
                    f = freq.to_value("MHz")
                    if i == 0:
                        fl.write(f"{f:12.6f} {spec:12.6f} {1:4.0f} {n} // temp.acq\n")
                    else:
                        fl.write(f"{f:12.6f} {spec:12.6f} {1:4.0f}\n")

        else:
            console.print(f"Reading averaged {load} spectra")

            if outfile.exists():
                spec = read_spec_txt(outfile)
            elif avg_spectra_path:
                spec = read_spec_txt(avg_spectra_path)

            spfreq = spec["freq"] * un.MHz
            spectra[load] = spec["spectra"]

    # Now do the calibration
    outfile = out / "specal.txt"
    if not redo_cal and outfile.exists():
        return

    console.print("Performing calibration")
    calobs = edges3cal(
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
        _alans11m = np.genfromtxt(
            modelled_s11_path,
            comments="#",
            names=True,
        )  # np.genfromtxt("alans-code/s11_modelled.txt", comments="#", names=True)

        alans11m = {}
        for load in [*loads, "lna"]:
            alans11m[load] = _alans11m[f"{load}_real"] + 1j * _alans11m[f"{load}_imag"]

        calobs = calobs.inject(
            lna_s11=alans11m["lna"] if inject_lna_s11 else None,
            source_s11s={
                "ambient": alans11m["amb"],
                "hot_load": alans11m["hot"],
                "short": alans11m["short"],
                "open": alans11m["open"],
            }
            if inject_source_s11s
            else None,
        )
    else:
        for name, load in calobs.loads.items():
            console.print(f"Using delay={load.reflections.model_delay} for load {name}")

    with open(outfile, "w") as fl:
        for i in range(calobs.freq.n):
            sca = calobs.C1()
            ofs = calobs.C2()
            tlnau = calobs.Tunc()
            tlnac = calobs.Tcos()
            tlnas = calobs.Tsin()
            lna = calobs.receiver_s11
            fl.write(
                f"freq {calobs.freq.freq[i].to_value('MHz'):10.6f} "
                f"s11lna {lna[i].real:10.6f} {lna[i].imag:10.6f} "
                f"sca {sca[i]:10.6f} ofs {ofs[i]:10.6f} tlnau {tlnau[i]:10.6f} "
                f"tlnac {tlnac[i]:10.6f} tlnas {tlnas[i]:10.6f} wtcal 1 cal_data\n"
            )

    console.print("BEST DELAY: ", calobs.cal_coefficient_models["NW"].delay)

    # Also save the modelled S11s
    console.print("Saving modelled S11s")
    s11m = {
        name: load.s11_model(calobs.freq.freq) for name, load in calobs.loads.items()
    }
    with open(out / "s11_modelled.txt", "w") as fl:
        fl.write(
            "# freq, amb_real amb_imag hot_real hot_imag open_real open_imag short_real"
            " short_imag lna_real lna_imag\n"
        )
        for i, (f, amb, hot, op, sh) in enumerate(
            zip(
                calobs.freq.freq.to_value("MHz"),
                s11m["ambient"],
                s11m["hot_load"],
                s11m["open"],
                s11m["short"],
            )
        ):
            fl.write(
                f"{f} {amb.real} {amb.imag} "
                f"{hot.real} {hot.imag} "
                f"{op.real} {op.imag} "
                f"{sh.real} {sh.imag} "
                f"{lna[i].real} {lna[i].imag}\n"
            )

    if plot:
        # Make plots...
        console.print("Plotting S11 models...")
        ax = calobs.plot_s11_models()
        ax.flatten()[0].figure.savefig(out / "s11_models.png")

        console.print("Plotting raw spectra...")
        fig = calobs.plot_raw_spectra()
        fig.savefig(out / "raw_spectra.png")

        console.print("Plotting calibration coefficients...")
        fig = calobs.plot_coefficients()
        fig.savefig(out / "calibration_coefficients.png")
