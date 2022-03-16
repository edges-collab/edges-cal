"""CLI functions for edges-cal."""
import click
import json
import papermill as pm
import yaml
from datetime import datetime
from nbconvert import PDFExporter
from pathlib import Path
from rich.console import Console
from traitlets.config import Config

from edges_cal import cal_coefficients as cc
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

        figs = obs.plot_s11_models()
        for kind, fig in figs.items():
            fig.savefig(out / f"{kind}_s11_model.png")

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

    if out is None:
        out = path / "outputs"
    else:
        out = Path(out)

    if not out.exists():
        out.mkdir()

    # Describe the filename...
    fname = Path(f"calibration_{datetime.now().strftime('%Y-%m-%d-%H.%M.%S')}.ipynb")

    if global_config:
        global_config = json.loads(global_config)
    else:
        global_config = {}

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

    console.print(f"Saved interactive notebook to '{out/fname}'")

    if pdf:  # pragma: no cover
        make_pdf(out, fname)
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

    if out is None:
        out = path / "outputs"
    else:
        out = Path(out)

    if not out.exists():
        out.mkdir()

    # Describe the filename...
    fname = Path(
        f"calibration-compare-{cmppath.name}_"
        f"{datetime.now().strftime('%Y-%m-%d-%H.%M.%S')}.ipynb"
    )

    if global_config:
        global_config = json.loads(global_config)
    else:
        global_config = {}

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
    console.print(f"Saved interactive notebook to '{out/fname}'")

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
    body, resources = exporter.from_filename(ipy_fname)
    with open(ipy_fname.with_suffix(".pdf"), "wb") as fl:
        fl.write(body)

    out = ipy_fname.with_suffix(".pdf")
    console.print(f"Saved PDF to '{out}'")
    return out


def upload_memo(fname, title, memo, quiet):  # pragma: no cover
    """Upload as memo to loco.lab.asu.edu."""
    try:
        import upload_memo  # noqa
    except ImportError:
        raise ImportError(
            "You need to manually install upload-memo to use this option."
        )

    opts = ["memo", "upload", "-f", str(fname)]
    if title:
        opts.extend(["-t", title])

    if memo:
        opts.extend(["-n", memo])
    if quiet:
        opts.append("-q")

    run(opts)
