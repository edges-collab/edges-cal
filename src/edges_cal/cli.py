import logging
from os import path
from os.path import join

import click
from edges_cal import cal_coefficients as cc

from . import io
from .logging import logger

main = click.Group()


@main.command()
@click.argument("root", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.option(
    "-c",
    "--correction-root",
    type=click.Path(dir_okay=True, file_okay=False, exists=True),
    default=None,
    help="base root to correction data",
)
@click.option(
    "-f", "--f-low", type=float, default=None, help="minimum frequency to calibrate"
)
@click.option(
    "-F", "--f-high", type=float, default=None, help="maximum frequency to calibrate"
)
@click.option("-n", "--run-num", type=int, default=2, help="run number to read")
@click.option(
    "-p",
    "--ignore_times_percent",
    type=float,
    default=5,
    help="percentage of data at start of files to ignore",
)
@click.option(
    "-r",
    "--resistance-f",
    type=float,
    default=50.0002,
    help="female resistance standard",
)
@click.option(
    "-R", "--resistance-m", type=float, default=50.166, help="male resistance standard"
)
@click.option(
    "-C", "--c-terms", type=int, default=11, help="number of terms to fit for C1 and C2"
)
@click.option(
    "-W",
    "--w-terms",
    type=int,
    default=12,
    help="number of terms to fit for TC, TS and TU",
)
@click.option(
    "-o",
    "--out",
    type=click.Path(dir_okay=True, file_okay=False, exists=True),
    default=".",
    help="output directory",
)
def run(
    path,
    correction_path,
    f_low,
    f_high,
    run_num,
    percent,
    resistance_f,
    resistance_m,
    c_terms,
    w_terms,
    out,
):
    """
    Calibrate using lab measurements in PATH, and make all relevant plots.
    """
    #    dataIn = "/data5/edges/data/data_dmlewis/Receiver01_2019_06_24_040_to_200_MHz/25C/"
    #    dataOut = root.expanduser("~/output/")

    obs = cc.CalibrationObservation(
        path=path,
        correction_path=correction_path or path,  # "/data5/edges/data",
        f_low=f_low,
        f_high=f_high,
        run_num=run_num,
        ignore_times_percent=percent,
        resistance_f=resistance_f,
        resistance_m=resistance_m,
        cterms=c_terms,
        wterms=w_terms,
    )

    # Plot Calibrator properties
    fig = obs.plot_raw_spectra()
    fig.savefig(join(out, "raw_spectra.png"))

    figs = obs.plot_s11_models()
    for kind, fig in figs.items():
        fig.savefig(join(out, f"{kind}_s11_model.png"))

    fig = obs.plot_calibrated_temps(bins=256)
    fig.savefig(join(out, "calibrated_temps.png"))

    fig = obs.plot_coefficients()
    fig.savefig(join(out, "calibration_coefficients.png"))

    # Calibrate and plot antsim
    antsim = cc.LoadSpectrum(
        "antsim",
        path,
        correction_path=correction_path or path,
        f_low=f_low,
        f_high=f_high,
        run_num=run_num,
        ignore_times_percent=percent,
    )
    fig = obs.plot_calibrated_temp(antsim, bins=256)
    fig.savefig(join(out, "antsim_calibrated_temp.png"))

    # Write out data
    obs.write_coefficients()
    obs.write(out)


@main.command()
@click.argument("root")
@click.option("--temp", default=25, type=click.Choice([15, 25, 35]))
@click.option("-v", "--verbosity", count=True, help="increase output verbosity")
@click.option("-V", "--less-verbose", count=True, help="decrease output verbosity")
@click.option("--fix/--no-fix", default=False, help="apply common fixes")
@click.option(
    "--remove-cruft/--leave-cruft",
    default=False,
    help="(interactively) remove unnecessary files",
)
def check(root, temp, verbosity, less_verbose, fix, remove_cruft):
    root = path.abspath(root)

    v0 = verbosity or 0
    v1 = less_verbose or 0

    v = 4 + v0 - v1
    if v < 0:
        v = 0
    if v > 4:
        v = 4

    logger.setLevel(
        [
            logging.CRITICAL,
            logging.ERROR,
            logging.STRUCTURE,
            logging.WARNING,
            logging.INFO,
            logging.DEBUG,
        ][v]
    )

    actual_root = join(root, "{}C".format(temp))
    io.CalibrationObservation.check_self(actual_root, fix)
    io.CalibrationObservation.check_contents(actual_root, fix)

    if not logger.errored:
        logger.success("All checks passed successfully!")
    else:
        logger.error(
            "There were {} errors in the checks... please fix them!".format(
                logger.errored
            )
        )
