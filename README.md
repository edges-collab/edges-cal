# edges_cal

[![Build Status](https://travis-ci.org/edges-collab/cal_coefficients.svg?branch=master)](https://travis-ci.org/edges-collab/cal_coefficients)
[![codecov](https://codecov.io/gh/edges-collab/cal_coefficients/branch/master/graph/badge.svg)](https://codecov.io/gh/edges-collab/cal_coefficients)

**Calculate calibration coefficients for EDGES Calibration Observations**

## Installation

Download/clone the repo and do

```
pip install .
```

in the top-level directory (optionally add an `-e` for develop-mode).
Preferably, do this in an isolated python/conda environment.

## Usage

### CLI
There is a very basic CLI set up for running a full calibration pipeline over a set of
data. To use it, do

```
$ edges-cal run --help
```

Multiple options exist, but the only ones required are `CONFIG` and `PATH`. The first
should point to a YAML configuration for the run, and the second should point to
a directory in which exists `S11`, `Resistance` and `Spectra` folders. Thus:

```
$ edges-cal run ~/config.yaml .
```

will work if you are in such a directory.

The `config.yaml` consists of a set of parameters passed to `edges_cal.CalibrationObservation`.
See its docstring for more details.

In addition, you can run a "term sweep" over a given calibration, iterating over number
of Cterms and Wterms until some threshold is met. This uses the same configuration as
`edges-cal run`, but you can pass a maximum number of C and W-terms, along with a threshold
at which to stop the sweep (this is a threshold in absolute RMS over degrees of freedom).
This will write out a `Calibration` file for the "best" set of parameters.

### Using the Library
To import:

```
import edges_cal as ec
```

Most of the functionality is highly object-oriented, and objects exist for each kind
of data/measurement. However, there is a container object for all of these, which
manages them. Thus you will typically use

```
>>> calobs = ec.CalibrationObservation(path="path/to/top/level")
```

Several other options exist, and they have documentation that you can access interactively
by using

```
>>> help(ec.CalibrationObservation)
```

The most relevant attributes are the (lazily-evaluated) calibration coefficient models:

```
>>> plt.plot(calobs.freq.freq, calobs.C1())
```

the various plotting routines, eg.

```
>>> calobs.plot_coefficients()
```

and the calibrate/decalibrate methods:

```
>>> calibrated_temp = calobs.calibrate("ambient")
```

Note that this final method can be applied to any `LoadSpectrum` -- i.e. you can pass
in field observations, or an antenna simulator.
