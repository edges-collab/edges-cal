"""Global Configuration options."""
import warnings
from edges_io.config import config, default_config
from pathlib import Path

_new_defaults = {"cal": {"cache-dir": str(Path("~/.edges-cal-cache").expanduser())}}

config._add_to_schema(_new_defaults)
default_config._add_to_schema(_new_defaults)

if not Path(config["cal"]["cache-dir"]).exists():
    try:
        Path(config["cal"]["cache-dir"]).mkdir()
    except Exception:
        warnings.warn(
            f"Could not create edges-cal cache directory: {config['cal']['cache-dir']}"
        )
