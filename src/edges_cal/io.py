"""
This module defines the overall file structure and internal contents of the
calibration observations. It does *not* implement any algorithms/methods on that data,
making it easier to separate the algorithms from the data checking/reading.
"""

import datetime
import glob
import os
import re
import shutil
from abc import ABC, abstractmethod
from collections import defaultdict

import h5py
import numpy as np
import read_acq
from cached_property import cached_property
from scipy import io as sio

from .logging import logger

LOAD_ALIASES = {
    "ambient": "Ambient",
    "hot_load": "HotLoad",
    "open": "LongCableOpen",
    "short": "LongCableShorted",
}


def get_active_files(path):
    if not os.path.isdir(path):
        raise ValueError("{} is not a directory!".format(path))
    fls = glob.glob(os.path.join(path, "*"))
    return [
        fl
        for fl in fls
        if not fl.endswith(".old") and not os.path.basename(fl) == "Notes.txt"
    ]


def ymd_to_jd(y, m, d):
    return (
        datetime.date(int(y), int(m), int(d)) - datetime.date(int(y), 1, 1)
    ).days + 1


def _ask_to_rm(fl):
    while True:
        reply = (
            str(input("Would you like to (recursively) remove {} (y/N)?: ".format(fl)))
            .lower()
            .strip()
        )
        if reply.startswith("y"):
            rm = True
            break
        elif reply.startswith("n") or not reply:
            rm = False
            break
        else:
            print("please select (y/n) only")

    if rm:
        if os.path.isdir(fl):
            shutil.rmtree(fl)
        else:
            os.remove(fl)
        return True
    else:
        return False


class FileStructureError(Exception):
    pass


class _DataFile(ABC):
    def __init__(self, path):
        self.path, self._re_match = self.check_self(path)

    @staticmethod
    @abstractmethod
    def check_self(path, fix=False):
        pass


class _DataContainer(ABC):
    _content_type = None

    def __init__(self, path, fix=False):
        self.path, self._re_match = self.check_self(path, fix)
        self.check_contents(self.path, fix)

        # For a container, if the checks failed, then we ought to bow out now
        if logger.errored:
            raise FileStructureError()

    @classmethod
    @abstractmethod
    def check_self(cls, path, fix=False):
        """Abstract method for checking whether the path is the correct format for the DB"""
        pass

    @classmethod
    def check_contents(cls, path, fix=False):
        """Abstract method for checking whether the contents of this container are in
         the correct format for the DB"""
        cls._check_contents_selves(
            path, fix=fix
        )  # Check that everything that *is* there has correct format.
        cls._check_all_files_there(path)  # Check that all necessary files are there.
        # Check that the files that are there have consistent properties, and are also
        # consistent with outside parameters (eg. if year appears on them, they should
        # be consistent with outer years).
        cls._check_file_consistency(path)

    @classmethod
    @abstractmethod
    def _check_all_files_there(cls, path):
        pass

    @classmethod
    @abstractmethod
    def _check_file_consistency(cls, path):
        pass

    @classmethod
    def _check_contents_selves(cls, path, fix=False):
        fls = get_active_files(path)
        for fl in fls:
            if type(cls._content_type) == dict:
                for key, ct in cls._content_type.items():
                    if os.path.basename(fl).startswith(key):
                        content_type = ct
                        break
                else:
                    logger.error(
                        "{} is an extraneous file/folder".format(os.path.basename(fl))
                    )

                    if fix:
                        fixed = _ask_to_rm(fl)
                        if fixed:
                            logger.success("Successfully removed.")

                    break
            else:
                content_type = cls._content_type

            fl, _ = content_type.check_self(fl, fix=fix)

            # Recursively check the contents of the contents.

            try:
                content_type.check_contents(fl, fix=fix)
            except AttributeError:
                # It's a DataFile, not a DataContainer
                pass


class _SpectrumOrResistance(_DataFile):
    file_pattern = (
        r"(?P<load_name>%s|AntSim\d)" % ("|".join(LOAD_ALIASES.values()))
        + r"_(?P<run_num>\d{2})_(?P<year>\d{4})_(?P<day>\d{3})_("
        r"?P<hour>\d{2})_(?P<minute>\d{2})_(?P<second>\d{2})_lab.(?P<file_format>\w{2,"
        r"3})$"
    )
    supported_formats = []

    def __init__(self, path, fix=False):
        super().__init__(path, fix)

        # Get out metadata
        self._groups = [match.groupdict() for match in self._re_match]

    @classmethod
    def _fix(cls, root, basename):
        if "AmbientLoad_" in basename:
            newname = basename.replace("AmbientLoad_", "Ambient_")
        elif "LongCableShort_" in basename:
            newname = basename.replace("LongCableShort_", "LongCableShorted_")
        else:
            newname = basename

        match = re.search(cls.file_pattern, newname)

        if match is not None:
            shutil.move(os.path.join(root, basename), os.path.join(root, newname))
            logger.success("Successfully converted to {}".format(newname))
            return os.path.join(root, newname), match

        # Try fixing from standard old format with redundant 25C in it.
        old_pattern = (
            r"^(?P<load_name>%s|AntSim\d)" % ("|".join(LOAD_ALIASES.values()))
            + r"_25C_(?P<month>\d{1,2})_(?P<day>\d{1,2})_("
            r"?P<year>\d\d\d\d)_(?P<hour>\d{1,2})_(?P<minute>\d{"
            r"1,2})_(?P<second>\d{1,2}).(?P<file_format>\w{2,3})$"
        )

        match = re.search(old_pattern, newname)

        if match is None:
            # Try a pattern where the name is followed by a number directly
            loads = "|".join(LOAD_ALIASES.values())
            old_pattern = (
                "^(?P<load_name>{})".format(loads)
                + r"(?P<run_num>\d{1,2})_25C_(?P<month>\d{1,"
                r"2})_(?P<day>\d{1,2})_(?P<year>\d\d\d\d)_("
                r"?P<hour>\d{1,2})_(?P<minute>\d{1,"
                r"2})_(?P<second>\d{1,2}).(?P<file_format>\w{2,3})$"
            )
            match = re.search(old_pattern, newname)

        if match is None:
            # Try a pattern where there is no run_num
            old_pattern = (
                r"(?P<load_name>%s|AntSim\d)" % ("|".join(LOAD_ALIASES.values()))
                + r"_(?P<year>\d{4})_(?P<day>\d{3})_"
                r"(?P<hour>\d{2})_(?P<minute>\d{2})_(?P<second>\d{2})_lab."
                r"(?P<file_format>\w{2,3})$"
            )
            match = re.search(old_pattern, newname)

        if match is None:
            logger.warning("\tCould not auto-fix it.")

            fixed = _ask_to_rm(os.path.join(root, newname))
            if fixed:
                logger.success("Successfully removed.")
            return None, None
        else:
            dct = match.groupdict()

            if "month" in dct:
                jd = ymd_to_jd(
                    match.group("year"), match.group("month"), match.group("day"),
                )
            else:
                jd = dct["day"]

            if "run_num" not in dct:
                dct["run_num"] = "01"

            newname = (
                "{load_name}_{run_num:0>2}_{year:0>4}_{jd:0>3}_{hour:0>2}_{minute:0>2}_{"
                "second:0>2}_lab.{file_format}".format(jd=jd, **dct)
            )
            newpath = os.path.join(root, newname)

            match = re.search(cls.file_pattern, newname)

            if match is not None:
                logger.success("Successfully converted to {}".format(newname))
                shutil.move(os.path.join(root, basename), newpath)
                return newpath, match
            else:
                return None, None

    @classmethod
    def check_self(cls, path, fix=False):
        if type(path) == str:
            path = [path]

        base_fnames = [os.path.basename(fname) for fname in path]
        root = os.path.dirname(os.path.normpath(path[0]))

        matches = []
        for i, basename in enumerate(base_fnames):
            match = re.search(cls.file_pattern, basename)
            if match is None:
                logger.error(
                    "The file {} does not have the correct format for a {}".format(
                        basename, cls.__name__
                    )
                )

                if fix:
                    newname, match = cls._fix(root, basename)

            if match is not None:
                path[i] = newname

                groups = match.groupdict()
                if int(groups["run_num"]) < 1:
                    logger.error("The run_num for {} is less than one!".format(newname))
                if not (2010 <= int(groups["year"]) <= 2030):
                    logger.error("The year for {} is a bit strange!".format(newname))
                if not (0 <= int(groups["day"]) <= 366):
                    logger.error(
                        "The day for {} is outside the number of days in a year!".format(
                            newname
                        )
                    )
                if not (0 <= int(groups["hour"]) <= 24):
                    logger.error("The hour for {} is outside 0-24!".format(newname))
                if not (0 <= int(groups["minute"]) <= 60):
                    logger.error("The minute for {} is outside 0-60!".format(newname))
                if not (0 <= int(groups["second"]) <= 60):
                    logger.error("The second for {} is outside 0-60!".format(newname))
                if not groups["file_format"] in cls.supported_formats:
                    logger.error(
                        "The file {} is not of a supported format ({}). Got format {}".format(
                            newname, cls.supported_formats, groups["file_format"]
                        )
                    )
                matches.append(match)
        return path, matches

    @classmethod
    def from_load(cls, load, direc, run_num=None, filetype=None):
        """
        Initialize the object in a simple way.

        Parameters
        ----------
        load : str
            The load name (eg. 'Ambient', 'HotLoad') or its alias (eg. 'ambient', 'hot_load').
        direc : str
            The directory in which to search for relevant data
        run_num : int, optional
            The run number of the data to use. Default, the last run. Each run is
            independent and different run_nums may be used for different loads.
        filetype : str, optional
            The filetype of the data. Must be one of the supported formats. Defaults
            to `_default_filetype`.
        """
        filetype = filetype or cls.supported_formats[0]

        files = glob.glob(
            os.path.join(
                direc,
                "{load}_??_????_???_??_??_??_lab.{filetype}".format(
                    load=load, filetype=filetype
                ),
            )
        )

        # Restrict to the given run_num (default last run)
        run_nums = [int(fl[len(load) : len(load) + 2]) for fl in files]
        if run_num is None:
            run_num = max(run_nums)

        files = [fl for fl, num in zip(files, run_nums) if num == run_num]

        if not files:
            raise ValueError(
                "No files exist for that load ({}) in that path ({}) with that filetype "
                "({})".format(load, direc, filetype)
            )

        return cls(files)

    @cached_property
    def run_num(self):
        """The run number of the data. All run_nums must be the same for all files in the data.

        Every observation may have several runs. Note that different runs may be mixed for
        different loads.
        """
        # Ensure all load names are the same
        if any(
            group["run_num"] != self._groups[0]["run_num"] for group in self._groups
        ):
            raise IOError("Two files given with incompatible run_nums")
        return self._groups[0]["run_num"]

    @cached_property
    def year(self):
        """Year on which data acquisition began"""
        # Ensure all load names are the same
        if any(group["year"] != self._groups[0]["year"] for group in self._groups):
            raise IOError("Two files given with incompatible years")
        return int(self._groups[0]["year"])

    @cached_property
    def days(self):
        """List of integer days (one per file) at which data acquisition was begun"""
        days = [int(group["day"]) for group in self._groups]
        if max(days) - min(days) > 30:
            logger.warning(
                "Spectra taken suspiciously far apart [{} days]".format(
                    max(days) - min(days)
                )
            )

        return days

    @cached_property
    def hours(self):
        """List of integer hours (one per file) at which data acquisition was begun"""
        return [int(group["day"]) for group in self._groups]

    @cached_property
    def minutes(self):
        """List of integer minutes (one per file) at which data acquisition was begun"""
        return [int(group["minute"]) for group in self._groups]

    @cached_property
    def seconds(self):
        """List of integer seconds (one per file) at which data acquisition was begun"""
        return [int(group["second"]) for group in self._groups]


class Spectrum(_SpectrumOrResistance):
    """
    Class representing an observed spectrum.

    Standard initialization takes a filename which will be read directly (as long as it
    is in one of the supported formats). Initialization via :func:`from_load` will
    attempt to find a file with the default naming scheme of the database.

    Supported formats: h5, acq, mat

    Examples
    --------
    >>> spec = Spectrum.from_load("Ambient", ".")
    >>> spec.file_format
    h5
    >>> spec.read()
    >>> spec.p0
    """

    supported_formats = ["h5", "acq", "mat"]

    @cached_property
    def file_format(self):
        """The file format of the data to be read."""
        formats = [os.path.splitext(fl)[1][1:] for fl in self.fnames]
        if any(
            format != formats[0] or format not in self.supported_formats
            for format in formats
        ):
            raise ValueError("not all file formats are the same!")
        return formats[0]

    def read(self):
        """
        Read the files of the object, and concatenate their data.

        Adds the attributes 'p0', 'p1', 'p2' and 'uncalibrated_spectrum'.
        """
        out = {}
        keys = ["p0", "p1", "p2", "ant_temp"]
        for fl in self.fnames:
            this_spec = getattr(self, "_read_" + self.file_format)(fl)

            for key in keys:
                if key not in out:
                    out[key] = this_spec[key]
                else:
                    out[key] = np.concatenate((out[key], this_spec[key]), axis=1)

        self.p0 = out["p0"]
        self.p1 = out["p1"]
        self.p2 = out["p2"]
        self.uncalibrated_spectrum = out["ant_temp"]

    @staticmethod
    def _read_mat(file_name):
        """
        This function loads the antenna temperature and date/time from MAT files.

        Parameters
        ----------
        file_name: str
            The file path to the MAT file.

        Returns
        -------
        2D Uncalibrated Temperature array, or dict of such.
        """
        # loading data and extracting main array
        d = sio.loadmat(file_name)

        # Return dict of all things
        if "ta" in d:
            d["ant_temp"] = d["ta"]
            del d["ta"]
        return d

    @staticmethod
    def _read_acq(file_name):
        ant_temp, px = read_acq.decode_file(file_name, progress=False, write_formats=[])
        return {"ant_temp": ant_temp, "p0": px[0], "p1": px[1], "p2": px[2]}

    @staticmethod
    def _read_h5(file_name):
        out = {}
        with h5py.File(file_name, "r") as fl:
            out["ant_temp"] = fl["ant_temp"][...]
            out["p0"] = fl["p0"][...]
            out["p1"] = fl["p1"][...]
            out["p2"] = fl["p2"][...]

        return out


class Resistance(_SpectrumOrResistance):
    """
    An object representing a resistance measurement (and its structure).
    """

    supported_formats = ("csv",)

    @cached_property
    def file_format(self):
        """The file format of the data to be read."""
        return "csv"

    def read(self):
        fnames = sorted(self.fnames)

        resistance = np.genfromtxt(fnames[0], skip_header=1, delimiter=",")[:, -3]
        for fl in fnames[1:]:
            resistance = np.concatenate((resistance, np.genfromtxt(fl)), axis=0)

        self.resistance = resistance
        return self.resistance


class _SpectraOrResistanceFolder(_DataContainer):
    folder_pattern = None

    def __init__(self, path, run_num=None, filetype=None, fix=False):
        """Collection of spectra in an observation"""
        super().__init__(path, fix)

        if type(run_num) is int or run_num is None:
            run_nums = {load: run_num for load in LOAD_ALIASES.values()}
        else:
            run_nums = run_num

        for name, load in LOAD_ALIASES.items():
            setattr(
                self,
                name,
                self._content_type.from_load(
                    load, path, run_nums.get(load, None), filetype
                ),
            )

    @classmethod
    def check_self(cls, path, fix=False):
        logger.structure("Checking {} folder contents at {}".format(cls.__name__, path))

        match = re.search(cls.folder_pattern, os.path.basename(path))
        if match is None:
            logger.error(
                "{} directory should be called {}".format(
                    cls.__name__, cls.folder_pattern
                )
            )

        return path, match

    @classmethod
    def _check_all_files_there(cls, path):
        # Just need to check for the loads.
        for name, load in LOAD_ALIASES.items():
            if not glob.glob(os.path.join(path, load + "_*")):
                logger.error(
                    "{} does not contain any files for load {}".format(
                        cls.__name__, load
                    )
                )

    @classmethod
    def _check_file_consistency(cls, path):
        fls = get_active_files(path)

        # logger.info("Found the following Antenna Simulators in {}: {}".format(
        #     cls.__name__, [os.path.basename(fl) for fl in fls if os.path.basename(fl).startswith("AntSim")])
        # )
        groups = [
            re.search(cls._content_type.file_pattern, fl).groupdict() for fl in fls
        ]

        # Ensure all years are the same
        for fl, group in zip(fls, groups):
            if group["year"] != groups[0]["year"]:
                logger.error(
                    "All years must be the same in a Spectra folder, but {} was not".format(
                        fl
                    )
                )

        # Ensure days are close-ish
        days = [int(group["day"]) for group in groups]
        if max(days) - min(days) > 30:
            logger.error(
                "Observation days are suspiciously far apart for {}".format(path)
            )

    def read_all(self):
        """Read all spectra"""
        for name in LOAD_ALIASES:
            getattr(self, name).read()


class Spectra(_SpectraOrResistanceFolder):
    folder_pattern = "Spectra"
    _content_type = Spectrum


class Resistances(_SpectraOrResistanceFolder):
    folder_pattern = "Resistance"
    _content_type = Resistance


class S1P(_DataFile):
    file_pattern = r"(?P<kind>\w+)(?P<run_num>\d{2}).s1p$"

    POSSIBLE_KINDS = [
        "Match",
        "Short",
        "Open",
        "ExternalMatch",
        "ExternalShort",
        "ExternalOpen",
        "External",
        "ReceiverReading",
        "ExternalLoad",
    ]

    def __init__(self, path, fix):
        super().__init__(path, fix)

        self.kind = self._re_match.groupdict()["kind"]
        self.run_num = int(self._re_match.groupdict()["run_num"])

        self.s11, self.freq = self.read(self.path)

    @classmethod
    def check_self(cls, path, fix=False):
        basename = os.path.basename(path)
        match = re.search(cls.file_pattern, basename)
        if match is None:
            logger.error(
                "The file {} has the wrong filename format for an S11 file".format(path)
            )
        else:
            groups = match.groupdict()
            if groups["kind"] not in cls.POSSIBLE_KINDS:
                logger.error(
                    "The file {} has a kind ({}) that is not supported. Possible: {}".format(
                        path, groups["kind"], cls.POSSIBLE_KINDS
                    )
                )

            if int(groups["run_num"]) < 1:
                logger.error(
                    "The file {} has a run_num ({}) less than one".format(
                        path, groups["run_num"]
                    )
                )
        return path, match

    @classmethod
    def read(cls, path_filename):
        d, flag = cls._get_kind(path_filename)
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

        return r, f / 1e6

    @staticmethod
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


class _S11SubDir(_DataContainer):
    STANDARD_NAMES = S1P.POSSIBLE_KINDS
    _content_type = S1P

    def __init__(self, path, run_num=None, fix=False):
        super().__init__(path, fix)

        self.run_num = run_num or self._get_max_run_num()

        for name in self.STANDARD_NAMES:
            setattr(
                self,
                name.lower(),
                S1P(os.path.join(path, name + "{:<02}.s1p".format(self.run_num))),
            )

        self.filenames = [
            getattr(self, thing.lower()).fname for thing in self.STANDARD_NAMES
        ]

    @property
    def active_contents(self):
        return get_active_files(self.path)

    @classmethod
    def check_self(cls, path, fix=False):
        if not os.path.exists(path):
            logger.error("The path {} does not exist!".format(path))
            return path, None

        match = re.search(cls.folder_pattern, os.path.basename(path))

        if match is None:
            logger.error(
                "The folder {} did not match any of the correct folder name criteria".format(
                    path
                )
            )

            if fix:
                if "AmbientLoad_" in path:
                    newpath = path.replace("AmbientLoad", "Ambient")
                elif "LongCableShort_" in path:
                    newpath = path.replace("LongCableShort", "LongCableShorted")
                elif "InternalSwitch" in path:
                    newpath = path.replace("InternalSwitch", "SwitchingState")
                else:
                    newpath = path

                if newpath != path:
                    shutil.move(path, newpath)
                    path = newpath

                match = re.search(cls.folder_pattern, os.path.basename(path))

                if match is not None:
                    logger.success("Successfully converted to {}".format(path))

        return path, match

    @classmethod
    def _check_all_files_there(cls, path):
        for name in cls.STANDARD_NAMES:
            if not glob.glob(os.path.join(path, name + "??.s1p")):
                logger.error("No {} standard found in {}".format(name, path))

    @classmethod
    def _check_file_consistency(cls, path):
        pass

        # TODO: The following checks if all subfiles have the same run-num.
        # I'm not sure if this *has* to be the case, so I've commented it out.

        # fls = get_active_files(path)
        #
        # out = defaultdict(list)
        #
        # for fl in fls:
        #     match = re.match(S1P.file_pattern, os.path.basename(fl))
        #     groups = match.groupdict()
        #     out[groups['kind']].append(int(groups['run_num']))
        #
        # for kind, val in out.items():
        #     if len(set(val)) != len(set(out['Short'])):
        #         logger.error(
        #             "Number of run_rums for {} is different in {}".format(kind, path)
        #         )

    def _get_max_run_num(self):
        return max(
            int(re.match(S1P.file_pattern, os.path.basename(fl)).groups("run_num"))
            for fl in self.active_contents
        )


class LoadS11(_S11SubDir):
    STANDARD_NAMES = ["Open", "Short", "Match", "External"]
    folder_pattern = "(?P<load_name>{})$".format("|".join(LOAD_ALIASES.values()))

    def __init__(self, direc, run_num=None, fix=False):
        super().__init__(direc, run_num, fix)
        self.load_name = self._re_match.groupdict("load_name")


class AntSimS11(LoadS11):
    folder_pattern = r"(?P<load_name>AntSim\d)$"


class _RepeatNumberableS11SubDir(_S11SubDir):
    def __init__(self, direc, run_num=None, fix=False):
        super().__init__(direc, run_num, fix)
        self.repeat_num = int(self._re_match.groupdict()["repeat_num"])


class SwitchingState(_RepeatNumberableS11SubDir):
    folder_pattern = r"SwitchingState(?P<repeat_num>\d{2})$"
    STANDARD_NAMES = [
        "Open",
        "Short",
        "Match",
        "ExternalOpen",
        "ExternalShort",
        "ExternalMatch",
    ]


class ReceiverReading(_RepeatNumberableS11SubDir):
    folder_pattern = r"ReceiverReading(?P<repeat_num>\d{2})$"
    STANDARD_NAMES = ["Open", "Short", "Match", "ReceiverReading"]


class S11Dir(_DataContainer):
    _content_type = {
        **{load: LoadS11 for load in LOAD_ALIASES.values()},
        **{
            "AntSim": AntSimS11,
            "SwitchingState": SwitchingState,
            "ReceiverReading": ReceiverReading,
            "InternalSwitch": SwitchingState,  # To catch the old way so it can be fixed.
        },
    }

    def __init__(self, path, repeat_num=None, run_num=None, fix=False):
        """Class representing the entire S11 subdirectory of an observation

        Parameters
        ----------
        path : str
            Top-level directory of the S11 measurements.
        repeat_num : int or dict, optional
            If int, the repeat num of any applicable sub-directories to use.
            If dict, each key specifies either SwitchingState or ReceiverReading
            and the repeat_num to use for that.
            By default, will find the last repeat.
        run_num : int or dict, optional
            If int, the run num of any applicable sub-directories to use.
            Any given sub-directory uses the same run_num for all files, but each
            sub-directory can use different run_nums
            If dict, each key specifies any of the sub-dir names
            and the repeat_num to use for that.
            By default, will find the last repeat.
        """
        super().__init__(path, fix)

        if type(repeat_num) == int or repeat_num is None:
            rep_nums = {"SwitchingState": repeat_num, "ReceiverReading": repeat_num}
        else:
            rep_nums = repeat_num

        if type(run_num) == int or run_num is None:
            run_nums = {"SwitchingState": run_num, "ReceiverReading": run_num}.update(
                {name: run_num for name in LOAD_ALIASES.values()}
            )
        else:
            run_nums = run_num

        self.switching_state = SwitchingState(
            os.path.join(path, "SwitchingState" + rep_nums.get("SwitchingState", None)),
            run_num=run_nums.get("SwitchingState", None),
        )
        self.receiver_reading = ReceiverReading(
            os.path.join(
                path, "ReceiverReading" + rep_nums.get("ReceiverReading", None)
            ),
            run_num=run_nums.get("ReceiverReading", None),
        )

        for name, load in LOAD_ALIASES.items():
            setattr(
                self,
                name,
                LoadS11(os.path.join(path, load), run_num=run_nums.get(load, None)),
            )

    @classmethod
    def check_self(cls, path, fix=False):
        logger.structure("Checking S11 folder contents at {}".format(path))

        if not os.path.exists(path):
            logger.error("This path does not exist: {}".format(path))

        if not os.path.basename(path) == "S11":
            logger.error("The S11 folder should be called S11")

        return path, True

    @classmethod
    def _check_all_files_there(cls, path):
        for load in LOAD_ALIASES.values():
            if not glob.glob(os.path.join(path, load)):
                logger.error("No {} S11 directory found!".format(load))

        for other in ["SwitchingState", "ReceiverReading"]:
            if not glob.glob(os.path.join(path, other + "??")):
                logger.error("No {} S11 directory found!".format(other))

    @classmethod
    def _check_file_consistency(cls, path):
        fls = get_active_files(path)
        logger.info(
            "Found the following Antenna Simulators in S11: {}".format(
                [
                    os.path.basename(fl)
                    for fl in fls
                    if os.path.basename(fl).startswith("AntSim")
                ]
            )
        )


class CalibrationObservation(_DataContainer):
    file_pattern = (
        r"Receiver(?P<rcv_num>\d{2})_(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})_("
        r"?P<freq_low>\d{3})_to_(?P<freq_hi>\d{3})MHz"
    )
    _content_type = {"S11": S11Dir, "Spectra": Spectra, "Resistance": Resistances}

    def __init__(self, path, ambient_temp=25, run_num=None, repeat_num=None, fix=False):
        """
        A class defining a full calibration observation, with all Spectra, Resistance
        and S11 files necessary to do a single analysis.
        """
        if ambient_temp not in [15, 25, 35]:
            raise ValueError("ambient temp must be one of 15, 25, 35!")

        path = os.path.join(path, "{}C".format(ambient_temp))

        super().__init__(path, fix)

        self.ambient_temp = ambient_temp
        self._groups = self._re_match.groupdict()
        self.receiver_num = int(self._groups["rcv_num"])
        self.year = int(self._groups["year"])
        self.month = int(self._groups["month"])
        self.day = int(self._groups["day"])
        self.freq_low = int(self._groups["freq_low"])
        self.freq_high = int(self._groups["freq_hi"])

        if type(run_num) == int or run_num is None:
            run_nums = {"Spectra": run_num, "Resistance": run_num, "S11": run_num}
        else:
            run_nums = run_num

        self.spectra = Spectra(
            os.path.join(self.base_path, "Spectra"),
            run_num=run_nums.get("Spectra", None),
            fix=fix,
        )
        self.resistance = Resistances(
            os.path.join(self.base_path, "Resistance"),
            run_num=run_nums.get("Resistance", None),
            fix=fix,
        )
        self.s11 = S11Dir(
            os.path.join(self.base_path, "S11"),
            run_num=run_nums.get("S11", None),
            repeat_num=repeat_num,
            fix=fix,
        )

    @classmethod
    def check_self(cls, path, fix=False):
        logger.structure("Checking root folder: {}".format(path))

        if not os.path.exists(path):
            logger.error("The path {} does not exist!".format(path))

        match = re.search(cls.file_pattern, path)

        if match is None:
            logger.error(
                "Calibration Observation directory name is in the wrong format!"
            )

            if fix:
                bad_pattern = (
                    r"^Receiver(\d{1,2})_(\d{4})_(\d{1,2})_(\d{1,2})_(\d{2,3})_to_(\d{"
                    r"2,3})MHz$"
                )
                base = os.path.dirname(os.path.normpath(path))
                name = os.path.basename(os.path.normpath(path))

                match = re.search(bad_pattern, name)

                if match is not None:
                    newname = "Receiver{:0>2}_{}_{:0>2}_{:0>2}_{:0>3}_to_{:0>3}MHz".format(
                        *match.groups()
                    )
                    shutil.move(
                        os.path.normpath(path) + "/", os.path.join(base, newname)
                    )
                    name = newname
                    path = os.path.join(base, name)
                    logger.success("Successfully renamed to {}".format(newname))
                else:
                    logger.warning("Failed to fix the name scheme")

        if match is not None:
            groups = match.groupdict()
            if int(groups["rcv_num"]) < 1:
                logger.error("Unknown receiver number: {}".format(groups["rcv_num"]))
            if not (2010 <= int(groups["year"]) <= 2030):
                logger.error("Unknown year: {}".format(groups["year"]))
            if not (1 <= int(groups["month"]) <= 12):
                logger.error("Unknown month: {}".format(groups["month"]))
            if not (1 <= int(groups["day"]) <= 31):
                logger.error("Unknown day: {}".format(groups["day"]))
            if not (1 <= int(groups["freq_low"]) <= 300):
                logger.error("Low frequency is weird: {}".format(groups["freq_low"]))
            if not (1 <= int(groups["freq_hi"]) <= 300):
                logger.error("High frequency is weird: {}".format(groups["high_hi"]))
            if not int(groups["freq_low"]) < int(groups["freq_hi"]):
                logger.error(
                    "Low frequency > High Frequency: {} > {}".format(
                        groups["freq_low"]
                    ),
                    groups["freq_hi"],
                )

            logger.info("Calibration Observation Metadata: {}".format(groups))

        return path, match

    @classmethod
    def _check_all_files_there(cls, path):
        for folder in ["S11", "Spectra", "Resistance"]:
            if not os.path.exists(os.path.join(path, folder)):
                logger.error("No {} folder in observation!".format(folder))

    @classmethod
    def _check_file_consistency(cls, path):
        pass

    def read_all(self):
        """Read all spectra and resistance files."""
        self.spectra.read_all()
        self.resistance.read_all()
