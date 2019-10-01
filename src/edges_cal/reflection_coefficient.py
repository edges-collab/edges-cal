import numpy as np


def impedance2gamma(Z, Z0):
    return (Z - Z0) / (Z + Z0)


def gamma2impedance(r, Z0):
    return Z0 * (1 + r) / (1 - r)


def gamma_de_embed(s11, s12s21, s22, rp):
    return (rp - s11) / (s22 * (rp - s11) + s12s21)


def gamma_shifted(s11, s12s21, s22, r):
    return s11 + (s12s21 * r / (1 - s22 * r))


def _get_kind(path_filename):
    # identifying the format
    with open(path_filename, "r") as d:
        comment_rows = 0
        for line in d.readlines():
            # checking settings line
            if line.startswith("#"):
                if "DB" in line or "dB" in line:
                    flag = "DB"
                if "MA" in line:
                    flag = "MA"
                if "RI" in line:
                    flag = "RI"

                comment_rows += 1
            elif line.startswith("!"):
                comment_rows += 1
            elif flag is not None:
                break

    #  loading data
    d = np.genfromtxt(path_filename, skip_header=comment_rows)

    return d, flag


def s1p_read(path_filename):

    d, flag = _get_kind(path_filename)
    f = d[:, 0]

    if flag == "DB":
        r = 10 ** (d[:, 1] / 20) * (
            np.cos((np.pi / 180) * d[:, 2]) + 1j * np.sin((np.pi / 180) * d[:, 2])
        )
    elif flag == "MA":
        r = d[:, 1] * (
            np.cos((np.pi / 180) * d[:, 2]) + 1j * np.sin((np.pi / 180) * d[:, 2])
        )
    elif flag == "RI":
        r = d[:, 1] + 1j * d[:, 2]
    else:
        raise Exception("file had no flags set!")

    return r, f


def s2p_read(path_filename):

    # loading data
    d, flag = _get_kind(path_filename)
    f = d[:, 0]

    if flag == "DB":
        r = 10 ** (d[:, 1] / 20) * (
            np.cos((np.pi / 180) * d[:, 2]) + 1j * np.sin((np.pi / 180) * d[:, 2])
        )
    if flag == "MA":
        r = d[:, 1] * (
            np.cos((np.pi / 180) * d[:, 2]) + 1j * np.sin((np.pi / 180) * d[:, 2])
        )
    if flag == "RI":
        r = d[:, 1] + 1j * d[:, 2]
        r1 = d[:, 3] + 1j * d[:, 4]
        r2 = d[:, 5] + 1j * d[:, 6]
        r3 = d[:, 7] + 1j * d[:, 8]
    return r, r1, r2, r3, f


def de_embed(r1a, r2a, r3a, r1m, r2m, r3m, rp):
    # This only works with 1D arrays, where each point in the array is
    # a value at a given frequency

    # The output is also a 1D array

    s11 = np.zeros(len(r1a)) + 0j  # 0j added to make array complex
    s12s21 = np.zeros(len(r1a)) + 0j
    s22 = np.zeros(len(r1a)) + 0j

    for i in range(len(r1a)):
        b = np.array([r1m[i], r2m[i], r3m[i]])  # .reshape(-1,1)
        A = np.array(
            [
                [1, r1a[i], r1a[i] * r1m[i]],
                [1, r2a[i], r2a[i] * r2m[i]],
                [1, r3a[i], r3a[i] * r3m[i]],
            ]
        )
        x = np.linalg.lstsq(A, b)[0]
        s11[i] = x[0]
        s12s21[i] = x[1] + x[0] * x[2]
        s22[i] = x[2]

    r = gamma_de_embed(s11, s12s21, s22, rp)

    return r, s11, s12s21, s22


def fiducial_parameters_85033E(R, md=1, md_value_ps=38):
    # Parameters of open
    open_off_Zo = 50
    open_off_delay = 29.243e-12
    open_off_loss = 2.2 * 1e9
    open_C0 = 49.43e-15
    open_C1 = -310.1e-27
    open_C2 = 23.17e-36
    open_C3 = -0.1597e-45

    op = np.array(
        [open_off_Zo, open_off_delay, open_off_loss, open_C0, open_C1, open_C2, open_C3]
    )

    # Parameters of short
    short_off_Zo = 50
    short_off_delay = 31.785e-12
    short_off_loss = 2.36 * 1e9
    short_L0 = 2.077e-12
    short_L1 = -108.5e-24
    short_L2 = 2.171e-33
    short_L3 = -0.01e-42

    sp = np.array(
        [
            short_off_Zo,
            short_off_delay,
            short_off_loss,
            short_L0,
            short_L1,
            short_L2,
            short_L3,
        ]
    )

    # Parameters of match
    match_off_Zo = 50

    if md == 0:
        match_off_delay = 0
    elif md == 1:
        match_off_delay = md_value_ps * 1e-12  # 38 ps, from Monsalve et al. (2016)
    match_off_loss = 2.3 * 1e9
    match_R = R

    mp = np.array([match_off_Zo, match_off_delay, match_off_loss, match_R])

    return (op, sp, mp)


def standard_open(f, par):
    """
    frequency in Hz
    """

    offset_Zo = par[0]
    offset_delay = par[1]
    offset_loss = par[2]
    C0 = par[3]
    C1 = par[4]
    C2 = par[5]
    C3 = par[6]

    # Termination
    Ct_open = C0 + C1 * f + C2 * f ** 2 + C3 * f ** 3
    Zt_open = 0 - 1j / (2 * np.pi * f * Ct_open)
    Rt_open = impedance2gamma(Zt_open, 50)

    # Transmission line
    Zc_open = (
        offset_Zo + (offset_loss / (2 * 2 * np.pi * f)) * np.sqrt(f / 1e9)
    ) - 1j * (offset_loss / (2 * 2 * np.pi * f)) * np.sqrt(f / 1e9)
    temp = ((offset_loss * offset_delay) / (2 * offset_Zo)) * np.sqrt(f / 1e9)
    gl_open = temp + 1j * ((2 * np.pi * f) * offset_delay + temp)

    # Combined reflection coefficient
    R1 = impedance2gamma(Zc_open, 50)
    ex = np.exp(-2 * gl_open)
    Rt = Rt_open
    Ri_open = (R1 * (1 - ex - R1 * Rt) + ex * Rt) / (1 - R1 * (ex * R1 + Rt * (1 - ex)))

    return Ri_open


def standard_short(f, par):
    """
    frequency in Hz
    """

    offset_Zo = par[0]
    offset_delay = par[1]
    offset_loss = par[2]
    L0 = par[3]
    L1 = par[4]
    L2 = par[5]
    L3 = par[6]

    # Termination
    Lt_short = L0 + L1 * f + L2 * f ** 2 + L3 * f ** 3
    Zt_short = 0 + 1j * 2 * np.pi * f * Lt_short
    Rt_short = impedance2gamma(Zt_short, 50)

    # Transmission line %%%%
    Zc_short = (
        offset_Zo + (offset_loss / (2 * 2 * np.pi * f)) * np.sqrt(f / 1e9)
    ) - 1j * (offset_loss / (2 * 2 * np.pi * f)) * np.sqrt(f / 1e9)
    temp = ((offset_loss * offset_delay) / (2 * offset_Zo)) * np.sqrt(f / 1e9)
    gl_short = temp + 1j * ((2 * np.pi * f) * offset_delay + temp)

    # Combined reflection coefficient %%%%
    R1 = impedance2gamma(Zc_short, 50)
    ex = np.exp(-2 * gl_short)
    Rt = Rt_short
    Ri_short = (R1 * (1 - ex - R1 * Rt) + ex * Rt) / (
        1 - R1 * (ex * R1 + Rt * (1 - ex))
    )

    return Ri_short


def standard_match(f, par):
    """
    frequency in Hz
    """

    offset_Zo = par[0]
    offset_delay = par[1]
    offset_loss = par[2]
    Resistance = par[3]

    # Termination
    Zt_match = Resistance
    Rt_match = impedance2gamma(Zt_match, 50)

    # Transmission line
    Zc_match = (
        offset_Zo + (offset_loss / (2 * 2 * np.pi * f)) * np.sqrt(f / 1e9)
    ) - 1j * (offset_loss / (2 * 2 * np.pi * f)) * np.sqrt(f / 1e9)
    temp = ((offset_loss * offset_delay) / (2 * offset_Zo)) * np.sqrt(f / 1e9)
    gl_match = temp + 1j * ((2 * np.pi * f) * offset_delay + temp)

    # combined reflection coefficient %%%%
    R1 = impedance2gamma(Zc_match, 50)
    ex = np.exp(-2 * gl_match)
    Rt = Rt_match
    Ri_match = (R1 * (1 - ex - R1 * Rt) + ex * Rt) / (
        1 - R1 * (ex * R1 + Rt * (1 - ex))
    )

    return Ri_match


def agilent_85033E(f, resistance_of_match, m=1, md_value_ps=38):
    """
    frequency in Hz
    """

    op, sp, mp = fiducial_parameters_85033E(
        resistance_of_match, md=m, md_value_ps=md_value_ps
    )
    o = standard_open(f, op)
    s = standard_short(f, sp)
    m = standard_match(f, mp)

    return (o, s, m)


def input_impedance_transmission_line(Z0, gamma, length, Zload):
    """
    Z0:     complex characteristic impedance
    gamma:  propagation constant
    length: length of transmission line
    Zload:  impedance of termination
    """

    Zin = (
        Z0
        * (Zload + Z0 * np.tanh(gamma * length))
        / (Zload * np.tanh(gamma * length) + Z0)
    )

    return Zin
