# edges_cal

This is the code to calculate the calibration coefficients of EDGES Spectra.

## Installation

Download/clone the repo and do

```
pip install .
```

in the top-level directory (optionally add an `-e` for develop-mode).
Preferably, do this in an isolated python/conda environment.

## Usage

### Imports and Reading Data
Most of the top-level functionality exists in the `cal_coefficients` module. To import:

```
from edges_cal import cal_coefficients as cc
```

To begin, we create an object that will encapsulate the Spectra we are creating.
This step defines the data locations for input and output, low and high frequencies,
the percentage of initial time ignored, and the run number.

```
Spectra = cc.Spectra(dataOut, dataIn, freqlow, freqhigh, percent, runNum)
```

We can then make an initial reading of our Spectra. It will go through the the four
calibration loads (ambient, hot, open, short) as well as the antenna simulator.
It will try to find these files by default, but the user can pass in a list of `.mat`
and `.txt` files to use (still with matching load names) instead of the ones in the
data folder:

```
cc.spec_read(Spectra, specFiles, resFiles)
```

### Initial plotting

Generate plots of the initials uncalibrated data read in by the previous step:
```
cc.spec_plot(Spectra)
```

### S11 Modeling

This will model the S11 after reading in the receiver parameters measured previously
(`s11_path`). Resistances for the male and female standards can be specified, if not
then default values will be used:
```
cc.s11_model(Spectra, s11_path, resistance_f, resistance_m)
```

### S11 Calibration

Use the S11 models to calculate the calibration coefficients for the Spectra.
Specify the number of polynomial terms in both parts, if not specified then
`cterms = 5` and `wterms = 7`:
```
cc.s11_cal(Spectra, cterms, wterms)
```

### Final plotting

Generate the calibrated plots for the Spectra, displaying the S11 and calibrated
temperatures as well as the coefficients.
```
cc.s11_plot(Spectra)
```
