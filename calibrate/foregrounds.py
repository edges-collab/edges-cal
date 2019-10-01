import numpy as np

F_CENTER = 75.0


def explog(x, a, b, c, d, e, f_center=F_CENTER):
    return (
        a
        * (x / f_center)
        ** (
            b
            + c * np.log(x / f_center)
            + d * np.log(x / f_center) ** 2
            + e * np.log(x / f_center) ** 3
        )
        + 2.725
    )


def physical5(x, a, b, c, d, e, f_center=F_CENTER):
    return np.log(
        a
        * (x / f_center) ** (-2.5 + b + c * np.log(x / f_center))
        * np.exp(-d * (x / f_center) ** -2)
        + e * (x / f_center) ** -2
    )


def physicallin5(x, a, b, c, d, e, f_center=F_CENTER):
    return (
        a * (x / f_center) ** -2.5
        + b * (x / f_center) ** -2.5 * np.log(x / f_center)
        + c * (x / f_center) ** -2.5 * (np.log(x / f_center)) ** 2
        + d * (x / f_center) ** -4.5
        + e * (x / f_center) ** -2
    )


def loglog(x, p, f_center=F_CENTER):
    """
    Log-Log foreground model, with arbitrary number of parameters

    Parameters
    ----------
    x : array_like
        Frequencies.
    p : iterable
        Co-efficients of the polynomial, from p0 to pn.
    f_center : float
        Central / reference frequency.
    """
    return sum([pp * x ** i for i, pp in enumerate(p)])


def linlog(x, p, f_center=F_CENTER):
    """
    Lin-Log foreground model, with arbitrary number of parameters

    Parameters
    ----------
    x : array_like
        Frequencies.
    p : iterable
        Co-efficients of the polynomial, from p0 to pn.
    f_center : float
        Central / reference frequency.
    """
    return sum([pp * np.log(x / f_center) ** i for i, pp in enumerate(p)])
