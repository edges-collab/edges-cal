from scipy import io as sio


def load_level1_MAT(file_name, kind=None):
    """
    This function loads the antenna temperature and date/time from MAT files produced by
    the MATLAB function acq2level1.m

    Parameters
    ----------
    file_name: str
        path and name of MAT file
    kind : str, optional
        The kind of thing to return. If a string, must be a key in the dict, and return
        value will be a 2D array. If None, return the whole dict, but ensuring keys
        are normalised.

    Returns
    -------
    2D Uncalibrated Temperature array, or dict of such.

    Examples
    --------
    >>> ds, dd = load_level1_MAT('/file.MAT', plot='yes')
    """
    # loading data and extracting main array
    d = sio.loadmat(file_name)

    if kind is None:
        # Return dict of all things
        if "ta" in d:
            d["ant_temp"] = d["ta"]
            del d["ta"]
        return d
    else:
        if kind == "temp":
            if "ta" in d.keys():
                return d["ta"]
            elif "ant_temp" in d.keys():
                return d["ant_temp"]
        else:
            return d[kind]
