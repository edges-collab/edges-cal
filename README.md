# cal_coefficients

This is the code to calculate the calibration coefficients of EDGES spectra.


### Prerequisites

* Numpy
* SciPy


### Usage

Import

```
from calibrate.cal_coefficients import *
```

Spectra object

To begin, we create an object that will encapsulate the spectra we are creating. This step defines the data locations for input and output,low and high frequencies, the percentage of initial time ignored, and the run number.

```
spectraName=spectra(dataOut, dataIn, freqlow, freqhigh, percent, runNum)
```

Reading

We can then make an initial reading of our spectra. It will go through the the four calibration loads(Ambient, Hot, Open, Short) as well as the antenna simulator. It will try to find these files by default, but the user can pass in a list of .mat and .txt files to use (still with matching load names) instead of the ones in the data folder.

```
spec_read(spectraName, specFiles, resFiles)
```

#### Initial plotting

Generates plots of the initials uncalibrated data read in by the previous step.
```
spec_plot(spectraName)
```

#### S11 Modeling

This will model the S11 after reading in the receiver parameters measured previously (s11_path). Resistances for the male and female standards can be specified, if not then default values will be used.
```
s11_model(spectraName, s11_path, resistance_f, resistance_m)
```

#### S11 Calibration

Uses the S11 models to calculate the calibration coefficients for the spectra. Specify the number of polynomial terms in both parts, if not specified then cterms = 5 and wterms = 7

```
s11_cal(spectraName, cterms, wterms)
```

#### Final plotting

Generates the calibrated plots for the spectra, displaying the S11 and calibrated temperatures as well as the coefficients.
```
s11_plot(spectraName)
```

## Authors

* **Nivedita Mahesh**
* **David Lewis** 
* **Steven Murray** 
