from calibrate.cal_coefficients import *
from os import path

dataIn = '/data5/edges/data/data_dmlewis/Receiver01_2019_06_24_040_to_200_MHz/25C/'
dataOut = path.expanduser('~/output/')

run1 = spectra(dataOut, dataIn, 50, 190, 5.0, 2)

spec_read(run1)
run1.save()
fig = spec_plot(run1)
fig.savefig("spec_plot.png")

s11_model(run1, '/data5/edges/data/', resistance_f=50.0002, resistance_m=50.166)
s11_cal(run1, 11, 12)
s11_plot(run1)
s11_write(run1)