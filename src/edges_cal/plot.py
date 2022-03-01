"""Various plotting functions."""
import matplotlib.pyplot as plt

from . import cal_coefficients as cc


def plot_raw_spectrum(
    spectrum, freq=None, fig=None, ax=None, xlabel=True, ylabel=True, **kwargs
):
    """
    Make a plot of the averaged uncalibrated spectrum associated with this load.

    Parameters
    ----------
    thermistor : bool
        Whether to plot the thermistor temperature on the same axis.
    fig : Figure
        Optionally, pass a matplotlib figure handle which will be used to plot.
    ax : Axis
        Optional, pass a matplotlib Axis handle which will be added to.
    xlabel : bool
        Whether to make an x-axis label.
    ylabel : bool
        Whether to plot the y-axis label
    kwargs :
        All other arguments are passed to `plt.subplots()`.
    """
    if isinstance(spectrum, cc.LoadSpectrum):
        freq = spectrum.freq.freq
        spectrum = spectrum.averaged_spectrum
    else:
        assert freq is not None

    if fig is None:
        fig, ax = plt.subplots(1, 1, **kwargs)

    ax.plot(freq, spectrum)
    if ylabel:
        ax.set_ylabel("$T^*$ [K]")

    ax.grid(True)
    if xlabel:
        ax.set_xlabel("Frequency [MHz]")
