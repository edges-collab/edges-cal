"""CLI functions for edges-cal."""
import click
import papermill as pm
import yaml
from datetime import datetime
from nbconvert import PDFExporter
from pathlib import Path
from rich.console import Console
from traitlets.config import Config

from edges_cal import cal_coefficients as cc

console = Console()

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
    """Calibrate using lab measurements in PATH, and make all relevant plots."""
    out = Path(out)
    with open(config) as fl:
        settings = yaml.load(fl, Loader=yaml.FullLoader)

    if cache_dir != ".":
        settings.update(load_kwargs={"cache_dir": cache_dir})

    obs = cc.CalibrationObservation(path=path, **settings)

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
            antsim = obs.new_load(load_name=name)
            fig = obs.plot_calibrated_temp(antsim, bins=256)
            fig.savefig(out / f"{name}_calibrated_temp.png")

    # Write out data
    obs.write(out / obs.path.parent.name)


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
@click.argument("path", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.option(
    "-c",
    "--config",
    default=None,
    type=click.Path(dir_okay=False, file_okay=True, exists=True),
    help="a YAML config file specifying parameters of the calibration",
)
@click.option(
    "-o",
    "--out",
    type=click.Path(dir_okay=True, file_okay=False, exists=True),
    default=None,
    help="output directory",
)
@click.option(
    "-d",
    "--cache-dir",
    type=click.Path(dir_okay=True, file_okay=False),
    default=".",
    help="directory in which to keep/search for the cache",
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
@click.option("--cterms", type=int, default=8)
@click.option("--wterms", type=int, default=10)
def report(
    config,
    path,
    out,
    cache_dir,
    report,
    upload,
    title,
    author,
    memo,
    quiet,
    pdf,
    cterms,
    wterms,
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

    if config is not None:
        with open(config) as fl:
            settings = yaml.load(fl, Loader=yaml.FullLoader)
    else:
        settings = {}

    if "cterms" not in settings:
        settings["cterms"] = cterms
    if "wterms" not in settings:
        settings["wterms"] = wterms

    console.print("Settings:")
    for k, v in settings.items():
        console.print(f"\t{k}: {v}")

    settings.update(observation=str(path))

    if cache_dir != ".":
        settings.update(cache_dir=cache_dir)

    # This actually runs the notebook itself.
    pm.execute_notebook(
        str(single_notebook),
        out / fname,
        parameters=settings,
        kernel_name="edges",
    )
    console.print(f"Saved interactive notebook to '{out/fname}'")

    if pdf:  # pragma: nocover
        make_pdf(out, fname)
        if upload:
            upload_memo(out / fname.with_suffix(".pdf"), title, memo, quiet)


@main.command()
@click.argument("path", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.argument("cmppath", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.option(
    "-c",
    "--config",
    default=None,
    type=click.Path(dir_okay=False, file_okay=True, exists=True),
    help="a YAML config file specifying parameters of the calibration",
)
@click.option(
    "-C",
    "--config-cmp",
    default=None,
    type=click.Path(dir_okay=False, file_okay=True, exists=True),
    help="a YAML config file specifying parameters of the comparison calibration",
)
@click.option(
    "-o",
    "--out",
    type=click.Path(dir_okay=True, file_okay=False, exists=True),
    default=None,
    help="output directory",
)
@click.option(
    "-d",
    "--cache-dir",
    type=click.Path(dir_okay=True, file_okay=False),
    default=".",
    help="directory in which to keep/search for the cache",
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
@click.option("--cterms", type=int, default=8)
@click.option("--wterms", type=int, default=10)
@click.option("--cterms-comp", type=int, default=8)
@click.option("--wterms-comp", type=int, default=10)
def compare(
    path,
    cmppath,
    config,
    config_cmp,
    out,
    cache_dir,
    report,
    upload,
    title,
    author,
    memo,
    quiet,
    pdf,
    cterms,
    wterms,
    cterms_comp,
    wterms_comp,
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

    if config is not None:
        with open(config) as fl:
            settings = yaml.load(fl, Loader=yaml.FullLoader)
    else:
        settings = {}

    if "cterms" not in settings:
        settings["cterms"] = cterms
    if "wterms" not in settings:
        settings["wterms"] = wterms

    if config_cmp is not None:
        with open(config_cmp) as fl:
            settings_cmp = yaml.load(fl, Loader=yaml.FullLoader)
    else:
        settings_cmp = {}

    if "cterms" not in settings_cmp:
        settings_cmp["cterms"] = cterms_comp
    if "wterms" not in settings_cmp:
        settings_cmp["wterms"] = wterms_comp

    console.print("Settings for Primary:")
    for k, v in settings.items():
        console.print(f"\t{k}: {v}")

    console.print("Settings for Comparison:")
    for k, v in settings_cmp.items():
        console.print(f"\t{k}: {v}")

    if cache_dir != ".":
        lk = settings.get("load_kwargs", {})
        lk.update(cache_dir=cache_dir)
        settings.update(load_kwargs=lk)

        lk = settings_cmp.get("load_kwargs", {})
        lk.update(cache_dir=cache_dir)
        settings_cmp.update(load_kwargs=lk)

    # This actually runs the notebook itself.
    pm.execute_notebook(
        str(single_notebook),
        out / fname,
        parameters={
            "observation": str(path),
            "cmp_observation": str(cmppath),
            "obs_config_": settings,
            "cmp_config_": settings_cmp,
        },
        kernel_name="edges",
    )
    console.print(f"Saved interactive notebook to '{out/fname}'")

    # Now output the notebook to pdf
    if pdf:  # pragma: nocover
        make_pdf(out, fname)
        if upload:
            upload_memo(out / fname.with_suffix(".pdf"), title, memo, quiet)


def make_pdf(out, fname):
    """Make a PDF out of an ipynb."""
    # Now output the notebook to pdf
    if report:

        c = Config()
        c.TemplateExporter.exclude_input_prompt = True
        c.TemplateExporter.exclude_output_prompt = True
        c.TemplateExporter.exclude_input = True

        exporter = PDFExporter(config=c)
        body, resources = exporter.from_filename(out / fname)
        with open(out / fname.with_suffix(".pdf"), "wb") as fl:
            fl.write(body)

        console.print(f"Saved PDF to '{out / fname.with_suffix('.pdf')}'")


def upload_memo(fname, title, memo, quiet):  # pragma: nocover
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
