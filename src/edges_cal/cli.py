from os.path import join

import click
import yaml

from edges_cal import cal_coefficients as cc

main = click.Group()


@main.command()
@click.argument("config", type=click.Path(dir_okay=False, file_okay=True, exists=True))
@click.argument("path", type=click.Path(dir_okay=True, file_okay=False, exists=True))
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
def run(config, path, out, cache_dir, plot, simulators):
    """
    Calibrate using lab measurements in PATH, and make all relevant plots.
    """
    with open(config, "r") as fl:
        settings = yaml.load(fl, Loader=yaml.FullLoader)

    if cache_dir != ".":
        settings.update(cache_dir=cache_dir)

    obs = cc.CalibrationObservation(path=path, **settings)

    if plot:
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
        for name in simulators:
            antsim = obs.new_load(load_name=name)
            fig = obs.plot_calibrated_temp(antsim, bins=256)
            fig.savefig(join(out, f"{name}_calibrated_temp.png"))

    # Write out data
    obs.write(join(out, str(obs.path.parent.name)))


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
    with open(config, "r") as fl:
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
