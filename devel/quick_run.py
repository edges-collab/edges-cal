from os import path

from edges_cal import cal_coefficients as cc

dataIn = "/data5/edges/data/data_dmlewis/Receiver01_2019_06_24_040_to_200_MHz/25C/"
dataOut = path.expanduser("~/output/")

obs = cc.CalibrationObservation(
    path=dataIn,
    f_low=50,
    f_high=190,
    run_num=2,
    percent=5,
    resistance_f=50.0002,
    resistance_m=50.166,
    cterms=11,
    wterms=12,
)

obs.plot_calibrated_temps("ambient")
obs.plot_calibrated_temps("hot_load")
obs.plot_calibrated_temps("open")
obs.plot_calibrated_temps("short")

antsim = cc.LoadSpectrum("antsim", dataIn, f_low=50, f_high=190, run_num=2, percent=5)
obs.plot_calibrated_temps(antsim)

obs.write_coefficients()
obs.plot_coefficients()
obs.write()
#
# run1 = cc.spectra(dataOut, dataIn, 50, 190, 5.0, 2)
#
# cc.spec_read(run1)
# run1.write()
# fig = cc.spec_plot(run1)
# fig.savefig("spec_plot.png")
#
# cc.s11_model(run1, "/data5/edges/data/", resistance_f=50.0002, resistance_m=50.166)
# cc.s11_cal(run1, 11, 12)
# figs = cc.s11_plot(run1)
#
# for i, fig in enumerate(figs):
#     fig.savefig(f"fig{i}.png")
#
# cc.s11_write(run1)
