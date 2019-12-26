from os import path

from edges_cal import cal_coefficients as cc

dataIn = "ExampleData/25C"
dataOut = "output"

obs = cc.CalibrationObservation(
    path=dataIn,
    correction_path=dataIn,
    f_low=50,
    f_high=190,
    run_num=2,
    ignore_times_percent=5,
    resistance_f=50.0002,
    resistance_m=50.166,
    cterms=11,
    wterms=12,
)

obs.plot_calibrated_temps(bins=40)

antsim = cc.LoadSpectrum(
    "antsim", dataIn, f_low=50, f_high=190, run_num=2, ignore_times_percent=5
)
obs.plot_calibrated_temp(antsim)

obs.write_coefficients()
obs.plot_coefficients()
obs.write()
