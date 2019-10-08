from os.path import join

import click
from edges_cal import cal_coefficients as cc

main = click.Group()


@main.command()
@click.argument("path", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.option(
    "-c",
    "--correction-path",
    type=click.Path(dir_okay=True, file_okay=False, exists=True),
    default=None,
    help="base path to correction data",
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
    "--percent",
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
    #    dataOut = path.expanduser("~/output/")

    obs = cc.CalibrationObservation(
        path=path,
        correction_path=correction_path or path,  # "/data5/edges/data",
        f_low=f_low,
        f_high=f_high,
        run_num=run_num,
        percent=percent,
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
        percent=percent,
    )
    fig = obs.plot_calibrated_temp(antsim, bins=256)
    fig.savefig(join(out, "antsim_calibrated_temp.png"))

    # Write out data
    obs.write_coefficients()
    obs.write(out)
