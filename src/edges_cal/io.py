from scipy import io as sio


def load_level1_MAT(file_name, kind="temp"):
    """
    This function loads the antenna temperature and date/time from MAT files produced by
    the MATLAB function acq2level1.m

    Parameters
    ----------
    file_name: str
        path and name of MAT file

    Returns
    -------
    2D Uncalibrated Temperature array

    Examples
    --------
    >>> ds, dd = load_level1_MAT('/file.MAT', plot='yes')
    """
    # loading data and extracting main array
    d = sio.loadmat(file_name)
    if kind == "temp":
        if "ta" in d.keys():
            return d["ta"]
        elif "ant_temp" in d.keys():
            return d["ant_temp"]
    else:
        return d[kind]
