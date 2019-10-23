# edges_cal

This is the code to calculate the calibration coefficients of EDGES LoadSpectrum.

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

Multiple options exist, but the only one required is `PATH`, which should point to
a directory in which exists `S11`, `Resistance` and `Spectra` folders. Thus:

```
$ edges-cal run .
```

will work if you are in such a directory.

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
