import os
import copy
import uuid
import glob
import json
import numpy as np
import datetime
import requests
import contextlib
import importlib
from functools import cached_property
from abc import ABC
from typing import (
    Optional, Union, List, Any, Tuple, ClassVar, Iterator, Callable,
    Mapping
)
from . import logger
from .utils import (
    cfg, NoDefault, create_registry_metaclass, readonly_cached_property,
    CachedPropertyMixin, promptuser_boolean
)


FileMeta = create_registry_metaclass(
    ("CATEGORY", "NAME"), ABC
)


class BaseFile(CachedPropertyMixin, metaclass=FileMeta):
    r"""Base class for managing files."""

    CATEGORY: ClassVar[str] = None
    NAME: ClassVar[str] = None
    DESC: ClassVar[str] = None
    CACHED: ClassVar[bool] = False
    EXAMPLE: ClassVar[str] = None
    _EXPECTS_DIRECTORY: ClassVar[bool] = False
    _default_ext: ClassVar[str] = ".json"
    DONT_TEST: ClassVar[bool] = False

    def __init__(self, fname: str, generated: Optional[bool] = False,
                 contents: Optional[dict] = None,
                 fname_orig: Optional[str] = None) -> None:
        r"""Initialize a file wrapper.

        Args:
            fname: Path to a file.
            generated: If True, this file was generated.
            contents: Contents to initialize the file with.
            fname_orig: Original file that this one was generated from.

        """
        self.fname = fname
        self.fname_orig = fname_orig or fname
        self.generated = generated
        if contents is not None:
            self.contents = contents
        super().__init__()

    def __del__(self) -> None:
        r"""Cleanup any generated file."""
        self.cleanup()

    def cleanup(self) -> None:
        r"""Cleanup any generated file."""
        if self.generated and self.exists and not self.CACHED:
            os.remove(self.fname)
            self.generated = False
            self._clear_cached_properties()

    @classmethod
    def from_file(
            cls, fname: Union[str, "BaseFile"]) -> "BaseFile":
        r"""Create an instance by loading it from a file.

        Args:
            fname: File or file instance to create an instance from. If
                an instance of this class is provided, it will be
                returned.

        Returns:
            File instance.

        """
        if isinstance(fname, cls):
            return fname
        return cls(fname)

    @classmethod
    def _read(cls, fname: str):
        r"""Read a file.

        Args:
            fname: Path to file to read.

        Returns:
            object: File contents.

        """
        if cls._default_ext.endswith(".json"):
            with open(fname, "r") as fd:
                return json.load(fd)
        elif cls._default_ext.endswith(".csv"):
            import pandas as pd
            return pd.read_csv(fname)
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def _write(cls, fname: str, contents: Any):
        r"""Write a file.

        Args:
            fname: Path to file to write.
            contents: File contents to write.

        """
        if cls._default_ext.endswith(".json"):
            with open(fname, "w") as fd:
                json.dump(contents, fd, indent=4)
                fd.write("\n")
            return
        elif cls._default_ext.endswith(".csv"):
            import pandas as pd
            assert isinstance(contents, pd.DataFrame)
            fd.to_csv(index=False)
            return
        raise NotImplementedError  # pragma: no cover

    def _get(self, name: str):
        r"""Get a parameter from the file.

        Args:
            name: Parameter name.

        Returns:
            Parameter value.

        Raises:
            KeyError: If name is not a valid parameter name.

        """
        raise KeyError(name)

    def _set(self, name: str, value: Any) -> Any:
        r"""Set a parameter in the file.

        Args:
            name: Parameter name.
            value: Parameter value.

        Raises:
            KeyError: If name is not a valid parameter name.

        """
        raise KeyError(name)

    @staticmethod
    def parameter_property(method: Callable) -> property:
        r"""Decorator for a BaseFile method that produces the default
        value that should be used if a KeyError is not raised by
        BaseFile.get(<property name>).

        Args:
            method: BaseFile method being wrapped.

        """

        name = method.__qualname__.rsplit('.', 1)[-1]

        @property
        def _parameter_property(self):
            r"""Get the parameter value, computing it if missing."""
            try:
                return self.get(name)
            except KeyError:
                self._cached_properties[name] = method(self)
                return self._cached_properties[name]

        return _parameter_property

    @cached_property
    def contents(self) -> Any:
        r"""object: File contents."""
        return self._read(self.fname)

    @contextlib.contextmanager
    def prevent_overwrite(self, suffix: Optional[str] = "-Modified"
                          ) -> Iterator[None]:
        r"""Context to ensure that a duplicate is made if the context
        exits successfully during modification of the file contents.

        Args:
            suffix: File suffix to add if a new file name is generated.

        """
        assert self.contents  # Ensure contents loaded
        yield
        if self.exists:
            self.move(suffix=suffix)
        self.generated = False
        self._clear_cached_properties()

    @property
    def exists(self) -> bool:
        r"""bool: True if the file exists."""
        return self.path_exists(self.fname)

    @classmethod
    def path_exists(cls, fpath: str) -> bool:
        r"""Check if a file/directory exists according to the path
        expected by this file type.

        Args:
            fpath: File or directory.

        Returns:
            bool: True if the file or directory exists.

        """
        if cls._EXPECTS_DIRECTORY:
            return os.path.isdir(fpath)
        return os.path.isfile(fpath)

    def get(self, name: str, default: Any = NoDefault) -> Any:
        r"""Get a parameter from the file.

        Args:
            name: Parameter name.
            default: Value to return if the parameter can't be found.

        Returns:
            Parameter value.

        """
        if name in self._cached_properties:
            return self._cached_properties[name]
        try:
            out = self._get(name)
            self._cached_properties[name] = out
            return out
        except KeyError:
            if default is not NoDefault:
                return default
            raise

    def set(self, name: str, value: Any) -> None:
        r"""Set a parameter in the file.

        Args:
            name: Parameter name.
            value: Parameter value.

        Raises:
            KeyError: If name is not a valid parameter name.

        """
        with self.prevent_overwrite():
            self._set(name, value)

    def write(self, new_contents: Optional[dict] = None,
              overwrite: Optional[bool] = False) -> None:
        r"""Write a new set of contents to the file.

        Args:
            new_contents: New contents to write.
            overwrite: If True, overwrite the existing file.

        """
        if (not overwrite) and self.exists:
            raise RuntimeError(f"File already exists: "
                               f"\"{self.fname}\"")
        if new_contents is not None:
            self.contents = new_contents
            self._clear_cached_properties()
        parent_dir = os.path.dirname(self.fname)
        if parent_dir and not os.path.isdir(parent_dir):
            os.mkdir(parent_dir)
        self._write(self.fname, self.contents)
        self.generated = True

    def move(self, dst: Optional[str] = None,
             suffix: Optional[str] = None,
             directory: Optional[str] = None) -> str:
        r"""Change the path to the file the contents will be written to
        when write is called.

        Args:
            dst: Path to the new location where the should be
                saved when write is called.
            suffix: Suffix to add to the current filename if dst is
                not provided.
            directory: Path to the directory that the should be
                written to when write is called.

        Returns:
            str: The new file path.

        """
        assert self.contents
        if dst is None:
            if suffix:
                dst = suffix.join(os.path.splitext(self.fname))
            else:
                dst = self.fname
        if directory is not None:
            dst = os.path.join(directory, os.path.basename(dst))
        while self.path_exists(dst):
            dst = str(uuid.uuid4()).join(os.path.splitext(dst))
        if dst != self.fname:
            self.generated = False
        self.fname = dst
        if self.exists:
            raise ValueError(f"Cannot move to a file that already exists: "
                             f"\"{self.fname}\"")
        return self.fname

    def copy(self, **kwargs: Any) -> "BaseFile":
        r"""Create a copy of this file.

        Args:
            **kwargs: Addiitonal keyword arguments are passed to move.

        Returns:
            BaseFile: Copied file.

        """
        out = type(self)(self.fname, generated=self.generated,
                         fname_orig=self.fname_orig)
        out.contents = copy.deepcopy(self.contents)
        if kwargs:
            out.move(**kwargs)
        return out


class BaseDataFile(BaseFile):
    r"""Base class for external data files."""

    CACHED: ClassVar[bool] = True
    URL: ClassVar[str] = None
    DEFAULT_EXTERNAL_TYPE: ClassVar[type] = None
    DEFAULT_DOWNLOAD_PARAMETERS: ClassVar[List[str]] = []
    REQUIRED_EXTERNAL_PARAMETERS: ClassVar[dict] = {}
    DEFAULT_DATE_RANGE: ClassVar[Tuple[datetime.date, datetime.date]] = None
    DEFAULT_CACHE_DIR: ClassVar[str] = None
    STATIC_DATE_LIMITS: ClassVar[Tuple[datetime.date, datetime.date]] = None
    STATIC_LOCATION_LIMITS: ClassVar[
        # Min latitude, max latitude, min longitude, max longitude
        Tuple[float, float, float, float]] = None
    DATE_LIMITS: ClassVar[Tuple[datetime.date, datetime.date]] = None
    LOCATION_LIMITS: ClassVar[
        # Min latitude, max latitude, min longitude, max longitude
        Tuple[float, float, float, float]] = (-90.0, 90.0, -180.0, 180.0)
    PYTEST_MARKS: ClassVar[List[str]] = []
    REQUIRED_OPTIONAL_PACKAGES: ClassVar[List[str]] = []

    @staticmethod
    def _on_registration(cls):
        if cls.NAME is not None:
            if cls.DESC is None:
                cls.DESC = cls.NAME
            if cls.DEFAULT_CACHE_DIR is None:
                if cls.NAME not in cfg["directories"]:
                    cfg["directories"][cls.NAME] = os.path.join(
                        os.getcwd(), cls.NAME)
                cls.DEFAULT_CACHE_DIR = cfg["directories"][cls.NAME]
            if ((cls.STATIC_DATE_LIMITS is not None
                 and cls.DATE_LIMITS is None)):
                cls.DATE_LIMITS = cls.STATIC_DATE_LIMITS
            if ((cls.STATIC_LOCATION_LIMITS is not None
                 and cls.LOCATION_LIMITS is None)):
                cls.LOCATION_LIMITS = cls.STATIC_LOCATION_LIMITS
            if ((cls.DEFAULT_DATE_RANGE is None
                 and cls.DEFAULT_EXTERNAL_TYPE is not None)):
                cls.DEFAULT_DATE_RANGE = (
                    cls.DEFAULT_EXTERNAL_TYPE.DEFAULT_DATE_RANGE
                )

    @classmethod
    def tools_installed(cls) -> bool:
        r"""bool: Check if the required Python packages are installed."""
        for x in cls.REQUIRED_OPTIONAL_PACKAGES:
            try:
                importlib.import_module(x)
            except ImportError:
                return False
        return True

    @classmethod
    def confirm_download(cls, prefix: Optional[str] = "",
                         dont_raise: Optional[bool] = False) -> bool:
        r"""Ask the user to confirm if download should proceed."""
        out = promptuser_boolean(
            f"{prefix}Are you sure you want to download {cls.DESC} data? "
            f"This dataset can require a large amount of resources "
            f"(time/memory).",
            _gha_default=False
        )
        if (not dont_raise) and (not out):
            raise RuntimeError("User declined download")
        return out

    @classmethod
    def time_specific(cls) -> bool:
        r"""bool: True if the class tracks time, False otherwise."""
        return (cls.STATIC_DATE_LIMITS is None)

    @classmethod
    def location_specific(cls) -> bool:
        r"""bool: True if the class has one instance per location,
        False otherwise."""
        return (cls.STATIC_LOCATION_LIMITS is None)

    @property
    def parameters(self) -> list:
        r"""list: Set of parameters contained by this file."""
        raise NotImplementedError  # pragma: no cover

    @readonly_cached_property
    def latitude(self) -> float:
        r"""float: Latitude (degrees)."""
        if self.STATIC_LOCATION_LIMITS is not None:
            return self.STATIC_LOCATION_LIMITS[:2]
        elif isinstance(self.contents, dict) and "latitude" in self.contents:
            return self.contents["latitude"]
        raise NotImplementedError  # pragma: no cover

    @readonly_cached_property
    def longitude(self) -> float:
        r"""float: Longitude (degrees)."""
        if self.STATIC_LOCATION_LIMITS is not None:
            return self.STATIC_LOCATION_LIMITS[2:]
        elif isinstance(self.contents, dict) and "longitude" in self.contents:
            return self.contents["longitude"]
        raise NotImplementedError  # pragma: no cover

    @readonly_cached_property
    def start_date(self) -> datetime.date:
        r"""datetime.date: Start of range covered by the file."""
        if self.STATIC_DATE_LIMITS is not None:
            return self.STATIC_DATE_LIMITS[0]
        elif isinstance(self.contents, dict) and "start_date" in self.contents:
            out = self.contents["start_date"]
            if isinstance(out, str):
                out = datetime.date.fromisoformat(out)
            return out
        raise NotImplementedError  # pragma: no cover

    @readonly_cached_property
    def end_date(self) -> datetime.date:
        r"""datetime.date: End of range covered by the file."""
        if self.STATIC_DATE_LIMITS is not None:
            return self.STATIC_DATE_LIMITS[1]
        elif isinstance(self.contents, dict) and "end_date" in self.contents:
            out = self.contents["end_date"]
            if isinstance(out, str):
                out = datetime.date.fromisoformat(out)
            return out
        raise NotImplementedError  # pragma: no cover

    @readonly_cached_property
    def timezone(self):
        r"""datetime.tzinfo: Timezone information for the location."""
        import timezonefinder
        import pytz
        tf = timezonefinder.TimezoneFinder()
        return pytz.timezone(tf.certain_timezone_at(
            lat=np.mean(self.latitude), lng=np.mean(self.longitude)))

    @classmethod
    def _f2str(cls, x: Any) -> str:
        r"""Convert a value to a string for use in a file name."""
        if isinstance(x, datetime.datetime):
            return x.date().isoformat()
        elif isinstance(x, datetime.date):
            return x.isoformat()
        elif isinstance(x, float):
            return str(x).replace(".", "p").replace("-", "n")
        elif isinstance(x, tuple):
            return "-".join([cls._f2str(xx) for xx in x])
        return x

    @classmethod
    def _round_location(cls, loc: float | Tuple[float, float]):
        if isinstance(loc, tuple):
            return (cls._round_location(loc[0]),
                    cls._round_location(loc[1]))
        return loc

    @classmethod
    def _format_location(cls, latitude: float | Tuple[float, float],
                         longitude: float | Tuple[float, float]):
        latitude, longitude = cls._round_location((
            latitude, longitude))
        return f"{cls._f2str(latitude)}_{cls._f2str(longitude)}"

    @classmethod
    def _format_time(cls, start_date: Optional[datetime.date],
                     end_date: Optional[datetime.date]):
        return f"_{cls._f2str(start_date)}_to_{cls._f2str(end_date)}"

    @classmethod
    def _file_prefix(cls) -> str:
        return f"{cls.NAME}_"

    @classmethod
    def format_filename(cls, latitude: float | Tuple[float, float],
                        longitude: float | Tuple[float, float],
                        start_date: Optional[datetime.date] = None,
                        end_date: Optional[datetime.date] = None,
                        cache_dir: Optional[str] = None,
                        prefix: Optional[str] = "",
                        suffix: Optional[str] = "") -> str:
        r"""Construct the file name for the cached data file containing
        the requested data.

        Args:
            latitude: Location latitude (degrees).
            longitude: Location longitude (degrees).
            start_date: Starting date for data.
            end_date: Ending date for data.
            cache_dir: Directory where the data should be cached.
            prefix: Additional prefix to add to the file name.
            suffix: Additional suffix to add to the file name.

        Returns:
            str: File name.

        """
        cache_dir = cache_dir or cls.DEFAULT_CACHE_DIR
        out = cls._file_prefix() + prefix
        if cls.location_specific():
            out += cls._format_location(latitude, longitude)
        if cls.time_specific():
            if cls.DEFAULT_DATE_RANGE is not None:
                start_date = start_date or cls.DEFAULT_DATE_RANGE[0]
                end_date = end_date or cls.DEFAULT_DATE_RANGE[1]
            else:
                assert start_date is not None
                assert end_date is not None
            out += cls._format_time(start_date, end_date)
        if suffix:
            out += suffix
        out += cls._default_ext
        if cache_dir:
            out = os.path.join(cache_dir, out) if out else cache_dir
        return out

    @classmethod
    def fetch_data(cls, *args: Any, **kwargs: Any) -> str:
        r"""Look for an existing file that contains the data for the
        requested location and dates. If one does not exist, create it
        by downloading data and converting it to the correct format.

        Args:
            *args, \*\*kwargs: Arguments are passed along to
                from_location.

        Returns:
            str: File name.

        """
        instance = cls.from_location(*args, **kwargs)
        if not instance.exists:
            instance.write()
        return instance.fname

    @classmethod
    def check_file_coverage(
            cls, fname: str,
            latitude: float,
            longitude: float,
            start_date: Union[datetime.date, datetime.datetime] = None,
            end_date: Union[datetime.date, datetime.datetime] = None
    ) -> bool:
        r"""Check if a file is valid for the given location & time.

        Args:
            fname: Path to file to check. If not provided and any of
                the location or time parameters are, this method will
                return False.
            latitude: Location latitude (degrees).
            longitude: Location longitude (degrees).
            start_date: Starting date for data.
            end_date: Ending date for data.

        Returns:
            bool: True if the file is valid, False otherwise.

        """
        if not any(x is not None for x in [fname, latitude, longitude]):
            return True
        src = (cls if cls.URL else cls.DEFAULT_EXTERNAL_TYPE)
        if fname and os.path.isfile(fname):
            instance = cls(fname)
            if instance.covers_location(latitude, longitude,
                                        start=start_date,
                                        end=end_date):
                return True
            logger.info(
                f"The provided {cls.CATEGORY} file (valid for "
                f"time={(instance.start_date, instance.end_date)}, "
                f"latitude={instance.latitude}, "
                f"longitude={instance.longitude}) does not "
                f"cover the required time/location ("
                f"time={(start_date, end_date)}, "
                f"latitude={latitude}, "
                f"longitude={longitude}). "
                f"{cls.CATEGORY.title()} data will be downloaded from "
                f"{src.DESC}"
            )
        elif fname:
            logger.info(
                f"The specified {cls.CATEGORY} file (\"{fname}\") "
                f"does not exist. {cls.CATEGORY.title()} data will be "
                f"downloaded from {src.DESC}"
            )
        return False

    @classmethod
    def from_location(
            cls, latitude: float, longitude: float,
            start_date: Union[datetime.date, datetime.datetime] = None,
            end_date: Union[datetime.date, datetime.datetime] = None,
            parameters: Optional[List[str]] = None,
            cache_dir: Optional[str] = None,
            src: Optional[str | FileMeta] = None,
    ) -> "BaseDataFile":
        r"""Look for an existing file that contains the data for the
        requested location and dates. If one does not exist, create it
        by downloading data from the API specified by the class
        attribute DEFAULT_EXTERNAL_TYPE.

        Args:
            latitude: Location latitude (degrees).
            longitude: Location longitude (degrees).
            start_date: Starting date for data.
            end_date: Ending date for data.
            parameters: Set of parameters that should be included in
                the data.
            cache_dir: Directory where the data should be cached.
            src: Name/class for the file type that the data should be
                collected from.

        Returns:
            BaseDataFile: File instance.

        """
        if isinstance(src, str):
            src = FileMeta.get_class(cls.CATEGORY, src)
        elif src is None:
            src = cls.DEFAULT_EXTERNAL_TYPE
        if src is not None and src != cls:
            if parameters is None:
                parameters = cls.REQUIRED_EXTERNAL_PARAMETERS.get(
                    src.NAME, None)
            fbase = src.from_location(
                latitude, longitude, start_date, end_date,
                parameters=parameters)
            return cls.from_compatible(fbase, cache_dir=cache_dir)
        cache_dir = cache_dir or cls.DEFAULT_CACHE_DIR
        latitude, longitude = cls._round_location((
            latitude, longitude))
        if cls.time_specific():
            if cls.DEFAULT_DATE_RANGE is not None:
                start_date = start_date or cls.DEFAULT_DATE_RANGE[0]
                end_date = end_date or cls.DEFAULT_DATE_RANGE[1]
            else:
                assert start_date is not None
                assert end_date is not None
            if isinstance(start_date, datetime.datetime):
                start_date = start_date.date()
            if isinstance(end_date, datetime.datetime):
                end_date = end_date.date()
            fglob = cls.format_filename(latitude, longitude, "*", "*",
                                        cache_dir=cache_dir)
            for f in glob.glob(fglob):
                fparts = os.path.splitext(os.path.basename(f))[0].split("_")
                fstart = datetime.date.fromisoformat(fparts[-3])
                fend = datetime.date.fromisoformat(fparts[-1])
                # TODO: Check location?
                if fstart <= start_date and fend >= end_date:
                    out = cls(f)
                    if parameters:
                        out.add_missing_param(parameters)
                    return out
            if cls.DEFAULT_DATE_RANGE is not None:
                start_date = min(start_date, cls.DEFAULT_DATE_RANGE[0])
                end_date = max(end_date, cls.DEFAULT_DATE_RANGE[1])
            assert end_date > start_date
        if cls.DATE_LIMITS is not None:
            if not cls._check_inside_range((start_date, end_date),
                                           cls.DATE_LIMITS):
                raise ValueError(
                    f"The requested date range {(start_date, end_date)} "
                    f"is not withing the date limits for {cls.DESC} "
                    f"data files {cls.DATE_LIMITS}"
                )
        if cls.LOCATION_LIMITS is not None:
            if not cls._check_inside_range(latitude, cls.LOCATION_LIMITS[:2]):
                raise ValueError(
                    f"The requested latitude {latitude} is not within "
                    f"the limits for {cls.DESC} data files "
                    f"{cls.LOCATION_LIMITS[:2]}"
                )
            if not cls._check_inside_range(longitude, cls.LOCATION_LIMITS[:2]):
                raise ValueError(
                    f"The requested longitude {longitude} is not within "
                    f"the limits for {cls.DESC} data files "
                    f"{cls.LOCATION_LIMITS[2:]}"
                )
        fname = cls.format_filename(latitude, longitude,
                                    start_date, end_date,
                                    cache_dir=cache_dir)
        if cls.path_exists(fname):
            assert not cls.time_specific()
            out = cls(fname)
            if parameters:
                out.add_missing_param(parameters)
            return out
        return cls.download_and_save_data(
            fname, latitude, longitude,
            start_date, end_date,
            parameters=parameters,
        )

    @classmethod
    def download_and_save_data(
            cls, fname: str,
            latitude: float, longitude: float,
            start_date: Union[datetime.date, datetime.datetime] = None,
            end_date: Union[datetime.date, datetime.datetime] = None,
            parameters: Optional[List[str]] = None,
    ) -> "BaseDataFile":
        r"""Download and save data for a location.

        Args:
            fname: Path where the data should be saved.
            latitude: Location latitude (degrees).
            longitude: Location longitude (degrees).
            start_date: Starting date for data.
            end_date: Ending date for data.
            parameters: Set of parameters that should be included in
                the data.

        Returns:
            BaseDataFile: File instance.

        """
        if "slow" in cls.PYTEST_MARKS:
            cls.confirm_download()
        data = cls.download_data(latitude, longitude,
                                 start_date, end_date,
                                 parameters=parameters)
        out = cls(fname, contents=data)
        out.write()
        return out

    def calculate_missing(self):
        r"""Calculate missing parameters that are required."""
        pass

    @classmethod
    def from_file(cls, fname: str | BaseFile) -> BaseFile:
        r"""Create an instance by loading it from a file.

        Args:
            fname: File or file instance to create an instance from. If
                an instance of this class is provided, it will be
                returned.

        Returns:
            File instance.

        """
        if ((isinstance(fname, BaseFile)
             and fname.CATEGORY == cls.CATEGORY)):
            return cls.from_compatible(fname)
        return super().from_file(fname)

    @classmethod
    def from_compatible(
            cls, fbase: "BaseDataFile",
            fname: Optional[str] = None,
            cache_dir: Optional[str] = None,
    ) -> "BaseDataFile":
        r"""Create a data file from another file of the same category.

        Args:
            fbase: External file to create an version of this file from.
            fname: File name where the data should be written to.
            cache_dir: Directory where the data should be cached. Only
                used if fname is not provided.

        """
        if fname is None:
            suffix = ""
            if fbase.NAME != cls.NAME:
                suffix = f"_from_{fbase.NAME}"
            fname = cls.format_filename(
                fbase.latitude,
                fbase.longitude,
                fbase.start_date,
                fbase.end_date,
                suffix=suffix,
                cache_dir=cache_dir,
            )
        if cls.path_exists(fname):
            return cls(fname)
        if fbase.NAME == cls.NAME:
            contents = fbase.contents
            # contents = copy.deepcopy(fbase.contents)
        else:
            if fbase.NAME in cls.REQUIRED_EXTERNAL_PARAMETERS:
                fbase.add_missing_param(
                    cls.REQUIRED_EXTERNAL_PARAMETERS[fbase.NAME])
            if hasattr(cls, f"_from_{fbase.NAME}"):
                contents = getattr(cls, f"_from_{fbase.NAME}")(fbase)
            elif hasattr(fbase, f"_to_{cls.NAME}"):
                contents = getattr(fbase, f"_to_{cls.NAME}")()
            else:
                contents = cls._from_compatible(fbase)
        out = cls(fname, contents=contents)
        out.calculate_missing()
        out.write()
        return out

    @classmethod
    def _from_compatible(cls, src: "BaseDataFile"):
        r"""Convert data from another file format into the correct
        format for this file.

        Args:
            src: Base class data.

        Returns:
            Converted data.

        """
        if isinstance(src, cls):
            return src.contents
        raise NotImplementedError(
            f"Conversion from {type(src)} to {cls}")  # pragma: no cover

    @classmethod
    def _check_inside_range(
            cls,
            a: float | datetime.date | None | Tuple[Any, Any],
            b: float | datetime.date | None | Tuple[Any, Any],
            inclusive_left: Optional[bool] = True,
            inclusive_right: Optional[bool] = False,
    ) -> bool:
        if isinstance(a, tuple):
            if a == (None, None):
                return True
            if isinstance(b, tuple):
                if not cls._check_inside_range(a[0], b):
                    return False
                return cls._check_inside_range(
                    a[1], b,
                    inclusive_left=False,
                    inclusive_right=True,
                )
            return ((a[0] is None or a[0] == b)
                    and (a[1] is None or a[1] == b))
        elif not isinstance(b, tuple):
            return (a is None or b is None or np.allclose(a, b))
        if a is None or b == (None, None):
            return True
        if ((b[0] is not None
             and (a < b[0] or ((not inclusive_left) and a == b[0])))):
            return False
        if ((b[1] is not None
             and (a > b[1] or ((not inclusive_right) and a == b[1])))):
            return False
        return True

    def covers_location(
            self, latitude: Optional[float | Tuple[float, float]] = None,
            longitude: Optional[float | Tuple[float, float]] = None,
            start: Union[datetime.date, datetime.datetime] = None,
            end: Union[datetime.date, datetime.datetime] = None) -> bool:
        r"""Check if the file contains data for the specified location.

        Args:
            latitude: Latitude that data should cover.
            longitude: Longitude that data should cover.
            start: Start date/time that data should cover.
            end: End date/time that data should cover.

        Returns:
            bool: True if the location is covered, False otherwise.

        """
        latitude, longitude = self._round_location((
            latitude, longitude))
        if isinstance(start, datetime.datetime):
            start = start.date()
        if isinstance(end, datetime.datetime):
            end = end.date()
        if not self._check_inside_range(latitude, self.latitude):
            return False
        if not self._check_inside_range(longitude, self.longitude):
            return False
        if not self._check_inside_range(
                (start, end), (self.start_date, self.end_date)):
            return False
        return True

    def add_missing_param(self, parameters: List[str]) -> None:
        r"""Fill in any missing parameters.

        Args:
            parameters: Set of parameters that must be present.

        """
        missing = [k for k in parameters if k not in self.parameters]
        if not missing:
            return
        data = self.download_data(self.latitude, self.longitude,
                                  self.start_date, self.end_date,
                                  parameters=missing)
        self.update_param(data)
        self.write(overwrite=True)

    def update_param(self, contents: Any) -> None:
        r"""Merge downloaded parameters into the current data.

        Args:
            contents: New data to incorporate.

        """
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def download_data(cls, latitude: float, longitude: float,
                      start_date: Optional[datetime.date] = None,
                      end_date: Optional[datetime.date] = None,
                      parameters: Optional[List[str]] = None) -> dict:
        r"""Use REST API (or other method) to download external data for
        a location.

        Args:
            latitude: Location latitude (degrees).
            longitude: Location longitude (degrees).
            start_date: Starting date for data.
            end_date: Ending date for data.
            parameters: Set of parameters to request.

        Returns:
            dict: JSON result.

        """
        raise NotImplementedError  # pragma: no cover


class BaseWeatherFile(BaseDataFile):
    r"""Base class for weather files."""

    CATEGORY: ClassVar[str] = "weather"
    NAME: ClassVar[str] = None

    @property
    def dates(self) -> np.ndarray:
        r"""np.ndarray: Dates covered by this file."""
        raise NotImplementedError  # pragma: no cover

    @property
    def start_date(self) -> datetime.date:
        r"""datetime.date: Minimum date covered by this file."""
        return min(self.dates).astype(datetime.date)

    @property
    def end_date(self) -> datetime.date:
        r"""datetime.date: Maximum date covered by this file."""
        return max(self.dates).astype(datetime.date)


class NASAPOWERWeatherFile(BaseWeatherFile):
    r"""Wrapper for loading NASA POWER data."""

    NAME: ClassVar[str] = "nasa_power_weather_data"
    DESC: ClassVar[str] = "NASA POWER"
    URL: ClassVar[str] = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
    )
    # Daily meterology data from 1981-7-1
    # Daily solar data from 1984-7-1
    DEFAULT_DATE_RANGE: ClassVar[Tuple[datetime.date, datetime.date]] = (
        datetime.date(1984, 7, 1), datetime.date(2026, 5, 8))
    DATE_LIMITS: ClassVar[Tuple[datetime.date, datetime.date]] = (
        datetime.date(1981, 7, 1), datetime.date.today())
    DEFAULT_DOWNLOAD_PARAMETERS: ClassVar[List[str]] = [
        "TOA_SW_DWN",
        "ALLSKY_SFC_SW_DWN",  # MJ
        "T2M", "T2M_MIN", "T2M_MAX",  # C
        "T2MDEW",  # C
        "WS2M",  # wind
        "PRECTOTCORR",  # mm
    ]

    @readonly_cached_property
    def parameters(self) -> list:
        r"""list: Set of power parameters contained by this file."""
        return list(self.contents["properties"]["parameter"].keys())

    @readonly_cached_property
    def dates(self) -> np.ndarray:
        r"""np.ndarray: Dates covered by this file."""
        import pandas as pd
        dates = pd.Series(
            self.contents["properties"]["parameter"][self.parameters[0]])
        return pd.to_datetime(dates.index, format="%Y%m%d")

    @readonly_cached_property
    def latitude(self) -> float:
        r"""float: Latitude (degrees)."""
        return float(self.contents["geometry"]["coordinates"][1])

    @readonly_cached_property
    def longitude(self) -> float:
        r"""float: Longitude (degrees)."""
        return float(self.contents["geometry"]["coordinates"][0])

    @readonly_cached_property
    def start_date(self) -> datetime.date:
        r"""datetime.date: Start of range covered by the file."""
        return datetime.datetime.strptime(
            self.contents["header"]["start"], "%Y%m%d").date()

    @readonly_cached_property
    def end_date(self) -> datetime.date:
        r"""datetime.date: End of range covered by the file."""
        return datetime.datetime.strptime(
            self.contents["header"]["end"], "%Y%m%d").date()

    def update_param(self, contents: Any) -> None:
        r"""Merge downloaded parameters into the current data.

        Args:
            contents: New data to incorporate.

        """
        self.contents["properties"]["parameter"].update(
            contents["properties"]["parameter"])

    @classmethod
    def download_data(cls, latitude: float, longitude: float,
                      start_date: Optional[datetime.date],
                      end_date: Optional[datetime.date],
                      parameters: Optional[List[str]] = None) -> dict:
        r"""Use REST API to get NASA POWER data for a location.

        Args:
            latitude: Location latitude (degrees).
            longitude: Location longitude (degrees).
            start_date: Starting date for data.
            end_date: Ending date for data.
            parameters: Set of parameters to request.

        Returns:
            dict: JSON result.

        """
        # Based on PCSE _query_NASAPower_server
        if parameters is None:
            parameters = cls.DEFAULT_DOWNLOAD_PARAMETERS.copy()
        # Build request for retrieving data, using new NASA POWER api
        payload = {
            "request": "execute",
            "parameters": ",".join(parameters),
            "latitude": latitude,
            "longitude": longitude,
            "start": start_date.strftime("%Y%m%d"),
            "end": end_date.strftime("%Y%m%d"),
            "community": "AG",
            "format": "JSON",
        }
        logger.debug("Starting retrieval from NASA POWER")
        req = requests.get(cls.URL, params=payload)
        req.raise_for_status()
        logger.debug("Successfully retrieved data from NASA POWER")
        return req.json()


class BaseSoilFile(BaseDataFile):
    r"""Base class for soil files."""

    CATEGORY: ClassVar[str] = "soil"
    NAME: ClassVar[str] = None
    # Set date range to widest possible since most soil data is not
    #   temporal
    STATIC_DATE_LIMITS: ClassVar[Tuple[datetime.date, datetime.date]] = (
        datetime.date(1, 1, 1), datetime.date.today())

    @property
    def depths(self) -> list:
        r"""list: List of (start, end) depth pairs covered by this
        file."""
        raise NotImplementedError  # pragma: no cover


class ISRICSoilGridsFile(BaseSoilFile):
    r"""Wrapper for loading ISRIC SoilGrids data from REST API."""

    NAME: ClassVar[str] = "isric_soil_data"
    DESC: ClassVar[str] = "ISRIC SoilGrids"
    URL: ClassVar[str] = (
        "https://rest.isric.org/soilgrids/v2.0/properties/query"
    )
    DEFAULT_DOWNLOAD_PARAMETERS: ClassVar[List[str]] = [
        "bdod",  # Bulk density of the fine earth fraction
        "cec",  # Cation exchange capacity
        "cfvo",  # Coarse fragment content
        "clay",
        # "landmask",
        "nitrogen",
        # "ocd",
        # "ocs",
        "phh2o",  # Soil pH
        "sand",
        "silt",
        "soc",  # Soil organic carbon
        # "wrb",
        "wv0010",  # Volumetric water content at 10 kPa (mm/mm)
        "wv0033",  # Volumetric water content at 33 kPa (mm/mm)
        "wv1500",  # Volumetric water content at 1500 kPa (mm/mm)
    ]
    SOIL_GRIDS_DEPTHS: ClassVar[dict] = {
        f"{start}-{end}cm": (start, end) for start, end in
        [(0, 5), (5, 15), (15, 30), (30, 60), (60, 100), (100, 200)]
    }
    # TODO: Tests for this class skipped by default until the API is
    #   stable
    PYTEST_MARKS: ClassVar[List[str]] = ["slow"]

    @classmethod
    def _round_location(cls, loc: float | Tuple[float, float]):
        if isinstance(loc, float):
            loc = np.round(loc)
        return super()._round_location(loc)

    @readonly_cached_property
    def parameters(self) -> list:
        r"""list: Set of soil parameters contained by this file."""
        return [
            layer["name"]
            for layer in self.contents["properties"]["layers"]
        ]

    @readonly_cached_property
    def depths(self) -> list:
        r"""list: List of (start, end) depth pairs covered by this
        file."""
        out = []
        for layer in self.contents["properties"]["layers"]:
            for depth in layer["depths"]:
                depth_range = (
                    depth["range"]["top_depth"],
                    depth["range"]["bottom_depth"]
                )
                if depth_range not in out:
                    out.append(depth_range)
        return sorted(out)

    @readonly_cached_property
    def latitude(self) -> float:
        r"""float: Latitude (degrees)."""
        return float(
            self.contents["geometry"]["coordinates"][1])

    @readonly_cached_property
    def longitude(self) -> float:
        r"""float: Longitude (degrees)."""
        return float(
            self.contents["geometry"]["coordinates"][0])

    def update_param(self, contents: Any) -> None:
        r"""Merge downloaded parameters into the current data.

        Args:
            contents: New data to incorporate.

        """
        self.contents["properties"]["layers"].extend(
            contents["properties"]["layers"])

    @classmethod
    def download_data(
            cls, latitude: float, longitude: float,
            start_date: Optional[datetime.date] = None,
            end_date: Optional[datetime.date] = None,
            parameters: Optional[List[str]] = None,
            depths: Optional[List[Union[str, Tuple[int, int]]]] = None,
            quantiles: Optional[List[str]] = ["mean"],
    ) -> dict:
        r"""Use REST API to get ISRIC SoilGrids data for a location.

        Args:
            latitude: Location latitude (degrees).
            longitude: Location longitude (degrees).
            parameters: Set of parameters to request.
            start_date: Starting date for data [UNUSED].
            end_date: Ending date for data [UNUSED].
            depths: List of depth strings or (start, end) depth pairs
                to request.
            quantiles: Names of the quantiles to return.

        Returns:
            dict: JSON result.

        """
        if parameters is None:
            parameters = cls.DEFAULT_DOWNLOAD_PARAMETERS.copy()
        if depths:
            depths = [
                f"{k[0]}-{k[1]}cm" if isinstance(k, tuple) else k
                for k in depths
            ]
            invalid_depths = [
                k for k in depths if k not in cls.SOIL_GRIDS_DEPTHS]
            if invalid_depths:
                raise ValueError(f"One or more of the provided depths are "
                                 f"not supported by ISRIC SoilGrids: "
                                 f"{depths} (supported = "
                                 f"{cls.SOIL_GRIDS_DEPTHS})")
        payload = {
            "lon": longitude,
            "lat": latitude,
        }
        if parameters:
            payload["property"] = parameters
        if depths:
            payload["depth"] = depths
        if quantiles:
            payload["value"] = quantiles
        logger.debug("Starting retrieval from ISRIC SoilGrids")
        req = requests.get(cls.URL, params=payload)
        req.raise_for_status()
        logger.debug("Successfully retrieved data from ISRIC SoilGrids")
        return req.json()


class SSURGOSoilFile(BaseSoilFile):
    r"""Wrapper for loading soil data from SSURGO."""

    NAME: ClassVar[str] = "ssurgo_soil_data"
    DESC: ClassVar[str] = "SSURGO"
    URL: ClassVar[str] = (
        "https://SDMDataAccess.nrcs.usda.gov/Tabular/"
        "SDMTabularService.asmx"
    )
    _default_ext: ClassVar[str] = ".xml"
    REQUIRED_DOWNLOAD_PARAMTERS: ClassVar[List[str]] = [
        "co.cokey",       # cokey
        "ch.chkey",       # chkey
        "comppct_r",      # prcent
        "hzdept_r",       # topdepth,
        "hzdepb_r",       # bottomdepth,
        "muname",         # muname,
    ]
    STRING_FIELDS: ClassVar[List[str]] = [
        "co.cokey",       # cokey
        "ch.chkey",       # chkey
        "compkind",
        "compname",
        "muname",
        "musym",
        "hzname",
    ]
    DEFAULT_DOWNLOAD_PARAMETERS: ClassVar[List[str]] = [
        "co.cokey",       # cokey
        "ch.chkey",       # chkey
        "comppct_r",      # prcent
        "compkind",       # compkind_series
        "wsatiated_r",    # wat_r,
        "partdensity",    # pd,
        "dbthirdbar_h",   # bb,
        "musym",          # musymbol,
        "compname",       # componentname,
        "muname",         # muname,
        "slope_r",
        "slope_h",        # slope,
        "hzname",
        "hzdept_r",       # topdepth,
        "hzdepb_r",       # bottomdepth,
        "awc_r",          # PAW,
        "ksat_l",         # KSAT,
        "claytotal_r",    # clay,
        "silttotal_r",    # silt,
        "sandtotal_r",    # sand,
        "om_r",           # OM,
        "iacornsr",       # CSR,
        "dbthirdbar_r",   # BD,
        "wfifteenbar_r",  # L15,
        "wthirdbar_h",    # DUL,
        "ph1to1h2o_r",    # pH,
        "ksat_r",         # sat_hidric_cond,
        # (dbthirdbar_r-wthirdbar_r)/100 as bd
    ]

    @classmethod
    def _read(cls, fname: str):
        r"""Read a model input file.

        Args:
            fname: Path to file to read.

        Returns:
            object: File contents.

        """
        import xml.etree.ElementTree as ET
        return ET.parse(fname)

    @classmethod
    def _write(cls, fname: str, contents: Any):
        r"""Read a model input file.

        Args:
            fname: Path to file to read.
            contents: File contents to write.

        """
        import xml.etree.ElementTree as ET
        assert isinstance(contents, ET.ElementTree)
        contents.write(fname)

    @classmethod
    def download_data(
            cls, latitude: float, longitude: float,
            start_date: Optional[datetime.date] = None,
            end_date: Optional[datetime.date] = None,
            parameters: Optional[List[str]] = None,
    ) -> dict:
        r"""Use SDMTabularService to get SSURGO data via query.

        Args:
            latitude: Location latitude (degrees).
            longitude: Location longitude (degrees).
            start_date: Starting date for data [UNUSED].
            end_date: Ending date for data [UNUSED].
            parameters: Set of parameters to request.

        Returns:
            dict: JSON result.

        """
        if parameters is None:
            parameters = cls.DEFAULT_DOWNLOAD_PARAMETERS.copy()
        parameters = [
            x for x in cls.REQUIRED_DOWNLOAD_PARAMTERS
            if x not in parameters
        ] + parameters
        lonLat = f"{longitude} {latitude}"
        headers = {'Content-Type': 'application/soap+xml; charset=utf-8'}
        body = """<?xml version="1.0" encoding="utf-8"?>
        <soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope"
        xmlns:sdm="http://SDMDataAccess.nrcs.usda.gov/Tabular/SDMTabularService.asmx">
        <soap:Header/>
        <soap:Body>
           <sdm:RunQuery>
              <sdm:Query>SELECT """
        body += ", ".join(parameters) + """ FROM sacatalog sc
        FULL OUTER JOIN legend lg  ON sc.areasymbol=lg.areasymbol
        FULL OUTER JOIN mapunit mu ON lg.lkey=mu.lkey
        FULL OUTER JOIN component co ON mu.mukey=co.mukey
        FULL OUTER JOIN chorizon ch ON co.cokey=ch.cokey
        FULL OUTER JOIN chtexturegrp ctg ON ch.chkey=ctg.chkey
        FULL OUTER JOIN chtexture ct ON ctg.chtgkey=ct.chtgkey
        FULL OUTER JOIN copmgrp pmg ON co.cokey=pmg.cokey
        FULL OUTER JOIN corestrictions rt ON co.cokey=rt.cokey
        WHERE mu.mukey IN (SELECT * from
        SDA_Get_Mukey_from_intersection_with_WktWgs84('point("""
        body += lonLat + """)'))
        AND sc.areasymbol != 'US'
        order by co.cokey, ch.chkey, comppct_r, hzdept_r, hzdepb_r, muname
        </sdm:Query>
        </sdm:RunQuery>
        </soap:Body>
        </soap:Envelope>"""
        req = requests.post(
            cls.URL, data=body, headers=headers, timeout=140)
        req.raise_for_status()
        logger.debug("Successfully retrieved data from SSURGO")
        import xml.etree.ElementTree as ET
        import io
        fd = io.BytesIO(req.content)
        out = ET.parse(fd)
        root = out.getroot()
        ET.SubElement(root, "location", attrib={
            "latitude": str(latitude),
            "longitude": str(longitude),
        })
        return out

    @property
    def parameters(self) -> list:
        r"""list: Set of parameters contained by this file."""
        return list(self.df.columns)

    @readonly_cached_property
    def df(self):
        r"""pandas.DataFrame: Data frame containing the data."""
        root = self.contents.getroot()
        data = {x.tag: [] for x in root[0][0][0][1][0][0]}
        for row in root[0][0][0][1][0]:
            for x in row:
                if x.tag in self.STRING_FIELDS or x.text is None:
                    data[x.tag].append(x.text)
                else:
                    data[x.tag].append(float(x.text))
        import pandas as pd
        out = pd.DataFrame.from_dict(data)
        out = out.drop_duplicates(subset=["hzdept_r"]).sort_values(
            'hzdept_r', ascending=True)
        return out

    @property
    def latitude(self) -> float:
        r"""float: Latitude (degrees)."""
        root = self.contents.getroot()
        return float(root[1].attrib["latitude"])

    @property
    def longitude(self) -> float:
        r"""float: Longitude (degrees)."""
        root = self.contents.getroot()
        return float(root[1].attrib["longitude"])

    @property
    def depths(self) -> list:
        r"""list: List of (start, end) depth pairs covered by this
        file."""
        out = sorted([
            (top, bottom) for top, bottom in
            zip(self.df["hzdept_r"], self.df["hzdepb_r"])
            if not (np.isnan(bottom) or np.isnan(top))
        ])
        return out


class HUMERISSoilData(BaseSoilFile):
    r"""Wrapper for loading soil data from the HUMERIS data set.

    Dalle Vaglie, Matteo; Francini, Saverio; Chirici, Gherardo;
        martellozzo, federico (2026), “HUMERIS Global Soil Dataset”,
        Mendeley Data, V2, doi: 10.17632/z8v8m579z4.2

    M. Dalle Vaglie, S. Francini, G. Chirici, & F. Martellozzo, A
        large-scale framework for estimating soil carbon, nitrogen,
        pH, and salinity dynamics for 1985–2023, Proc. Natl. Acad. Sci.
        U.S.A. 123 (22) e2534913123,
        https://doi.org/10.1073/pnas.2534913123 (2026).

    Documentation: https://matteodallevaglie.com/works/humeris/

    """
    NAME: ClassVar[str] = "HUMERIS"
    URL: ClassVar[str] = (
        "https://data.mendeley.com/public-api/zip/z8v8m579z4/download/2"
    )
    _default_ext: ClassVar[str] = ""
    _EXPECTS_DIRECTORY: ClassVar[bool] = True
    STATIC_DATE_LIMITS: ClassVar[Tuple[datetime.date, datetime.date]] = (
        datetime.date(1980, 1, 1), datetime.date(2025, 12, 31))
    STATIC_LOCATION_LIMITS: ClassVar[
        # Min latitude, max latitude, min longitude, max longitude
        Tuple[float, float, float, float]] = (-90.0, 90.0, -180.0, 180.0)
    PYTEST_MARKS: ClassVar[List[str]] = ["slow"]

    @classmethod
    def _file_prefix(cls) -> str:
        return ""

    @classmethod
    def download_and_save_data(
            cls, fname: str, *args: Any, **kwargs: Any
    ) -> "HUMERISSoilData":
        r"""Download and save data for a location.

        Args:
            fname: Path where the data should be saved.
            \*args, \*\*kwargs: Additional arguments are ignored.

        Returns:
            BaseDataFile: File instance.

        """
        import urllib.request
        import zipfile
        cls.confirm_download()
        fname_zip = f"{fname}.zip"
        pre_existing = os.path.isfile(fname_zip)
        if not pre_existing:
            urllib.request.urlretrieve(cls.URL, fname_zip)
        with zipfile.ZipFile(fname_zip, 'r') as zip_ref:
            zip_ref.extractall(fname)
        if not pre_existing:
            os.remove(fname_zip)
        return cls(fname)

    @classmethod
    def _read(cls, fname: str):
        r"""Read a model input file.

        Args:
            fname: Path to file to read.

        Returns:
            object: File contents.

        """
        import rasterio
        files = sorted(glob.glob(os.path.join(fname, "Mean", "*.tif")))
        assert files
        return {
            os.path.splitext(os.path.basename(x))[0]: rasterio.open(x)
            for x in files
        }

    @readonly_cached_property
    def parameters(self) -> list:
        r"""list: Set of soil parameters contained by this file."""
        return list(self.contents.keys())

    @readonly_cached_property
    def latitude(self) -> Tuple[float, float]:
        r"""tuple: Latitude range (degrees)."""
        first = self.parameters[0]
        return (self.contents[first].bounds.bottom,
                self.contents[first].bounds.top)

    @readonly_cached_property
    def longitude(self) -> Tuple[float, float]:
        r"""tuple: Longitude (degrees)."""
        first = self.parameters[0]
        return (self.contents[first].bounds.left,
                self.contents[first].bounds.right)

    @property
    def depths(self) -> list:
        r"""list: List of (start, end) depth pairs covered by this
        file."""
        return [(0, 30)]


class EarthDataAuth(CachedPropertyMixin):
    r"""Wrapper class for handling authentication for NASA data products
    through earthaccess."""

    _global_auth = None

    def __init__(self, *args, **kwargs):
        import earthaccess
        self.earthaccess = earthaccess
        super().__init__(*args, **kwargs)
        self.ea_auth  # Ensure its created via login

    @classmethod
    def get_global_earthaccess(cls):
        r"""Get the global earthaccess import."""
        return cls.get_global_auth().earthaccess

    @classmethod
    def get_global_auth(cls) -> "EarthDataAuth":
        r"""Get the global authentication instance."""
        if cls._global_auth is None:
            cls._global_auth = cls()
        return cls._global_auth

    @readonly_cached_property
    def ea_auth(self):
        r"""earthaccess.auth.Auth: Authentication instance."""
        return self.earthaccess.login()

    @property
    def username(self) -> str:
        r"""str: User name."""
        return self.ea_auth.username

    @property
    def password(self) -> str:
        r"""str: Password."""
        return self.ea_auth.password

    @property
    def token(self) -> str:
        r"""str: Bearer authentication token."""
        return self.ea_auth.token

    @property
    def auth(self):
        r"""tuple: Username and password."""
        if self.username and self.password:
            return (self.username, self.password)
        return None

    def get_session(self) -> requests.Session:
        r"""Get a requests session with the authentication in effect."""
        out = requests.Session()
        if self.auth:
            out.auth = self.auth
        if self.token:
            out.headers["Authorization"] = (
                f"Bearer {self.token['access_token']}")
        return out


class EarthDataAppEEARSToken(CachedPropertyMixin):
    r"""Wrapper class for handling NASA AppEEARS authentication that
    caches a bearer authentication token and renews it once it expires."""

    URL: ClassVar[str] = (
        "https://appeears.earthdatacloud.nasa.gov/api/login"
    )
    _global_auth = None

    def __init__(self, fname: str = None) -> None:
        r"""Create a new instance.

        Args:
            fname: File where authentication token should be loaded from
                or saved to for reuse.

        """
        if fname is None:
            fname = os.path.join(os.getcwd(),
                                 "earthdata_appeears_token.json")
        self.fname: str = fname
        self._token: Mapping[str, str] | None = None

    @classmethod
    def get_global_auth(cls) -> "EarthDataAppEEARSToken":
        r"""Get the global authentication instance."""
        if cls._global_auth is None:
            cls._global_auth = cls()
        return cls._global_auth

    @classmethod
    def get_global_session(cls, session: Optional[requests.Session] = None):
        r"""Get a global requests session with the global authentication
        in effect.

        Args:
            session: Existing session to use if provided.

        """
        if session is not None:
            return session
        return cls.get_global_auth().get_session()

    @property
    def token(self):
        r"""dict: Bearer token."""
        if self._token is None and os.path.isfile(self.fname):
            with open(self.fname, 'r') as fd:
                self._token = json.load(fd)
        if self._token is not None:
            texp = datetime.datetime.fromisoformat(
                self._token["expiration"])
            if texp > datetime.datetime.now(texp.tzinfo):
                return self._token
        logger.info("Creating a new AppEEARS bearer token")
        ea_auth = EarthDataAuth.get_global_auth()
        kws = {
            "headers": {"Content-Length": 0},
        }
        with ea_auth.get_session() as s:
            req = s.post(
                "https://appeears.earthdatacloud.nasa.gov/api/login",
                **kws
            )
        req.raise_for_status()
        self._token = req.json()
        with open(self.fname, 'w') as fd:
            json.dump(self._token, fd)
        return self._token

    def get_session(self) -> requests.Session:
        r"""Get a requests session with the authentication in effect."""
        out = requests.Session()
        out.headers["Authorization"] = (
            f"{self.token['token_type']} {self.token['token']}")
        return out


class NASAAppEEARSData(BaseSoilFile):
    r"""Wrapper for requesting data from the NASA AppEEARS REST API."""

    NAME: ClassVar[str] = "AppEEARS"
    DESC: ClassVar[str] = "NASA AppEEARS"
    VERSION: ClassVar[str] = None
    URL: ClassVar[str] = (
        "https://appeears.earthdatacloud.nasa.gov/api")
    STATIC_DATE_LIMITS: ClassVar[Tuple[datetime.date, datetime.date]] = None
    DONT_TEST: ClassVar[bool] = True
    PYTEST_MARKS: ClassVar[List[str]] = ["slow"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.exists:
            self.complete_files()

    @classmethod
    def _file_prefix(cls) -> str:
        return f"AppEEARS_{cls.NAME}_{cls.VERSION}_"

    @property
    def bundle_dir(self) -> str:
        r"""str: Path to the directory containing downloaded results."""
        return os.path.splitext(self.fname)[0] + "_results"

    @readonly_cached_property
    def df(self):
        r"""pandas.DataFrame: Data frame containing the data."""
        import pandas as pd
        return pd.read_csv(os.path.join(
            self.bundle_dir,
            f"{self.NAME}-{self.VERSION}-results.csv"))

    @readonly_cached_property
    def metadata(self):
        r"""xml.etree.ElementTree: Metadata associated with the results."""
        import xml.etree.ElementTree as ET
        fname = os.path.join(
            self.bundle_dir,
            f"{self.NAME}-{self.VERSION}-metadata.xml")
        with open(fname, 'r') as fd:
            return ET.parse(fd)

    @property
    def parameters(self) -> list:
        r"""list: Set of parameters contained by this file."""
        return list(self.df.columns)

    @property
    def status(self) -> str:
        r"""str: Status of the task request."""
        out = self.contents["status"]
        if out in ["pending", "processing"]:
            update = self.check_status(self.contents["task_id"])
            if update != out:
                self.contents["status"] = update
                self.write(overwrite=True)
                out = update
        return out

    @classmethod
    def product(cls) -> str:
        r"""str: Name of the product this class uses."""
        return f"{cls.NAME}.{cls.VERSION}"

    @classmethod
    def products(cls) -> list:
        r"""Get a list of products available via NASA AppEEARS"""
        with EarthDataAppEEARSToken.get_global_session() as s:
            return s.get(cls.URL + "/product").json()

    @classmethod
    def layers(cls, product: Optional[str] = None) -> dict:
        r"""Get a list of layers available for a product on NASA AppEEARS"""
        if product is None:
            product = cls.product()
        with EarthDataAppEEARSToken.get_global_session() as s:
            return s.get(cls.URL + f"/product/{product}").json()

    @classmethod
    def download_data(
            cls, latitude: float, longitude: float,
            start_date: Optional[datetime.date] = None,
            end_date: Optional[datetime.date] = None,
            parameters: Optional[List[str]] = None,
    ) -> dict:
        r"""Use SDMTabularService to get SSURGO data via query.

        Args:
            fname: Path where the data should be saved.
            latitude: Location latitude (degrees).
            longitude: Location longitude (degrees).
            start_date: Starting date for data [UNUSED].
            end_date: Ending date for data [UNUSED].
            parameters: Set of parameters to request.

        Returns:
            dict: JSON result.

        """
        task_name = os.path.splitext(
            os.path.basename(cls.format_filename(
                latitude, longitude,
                start_date, end_date)))[0]
        if parameters is None:
            parameters = cls.DEFAULT_DOWNLOAD_PARAMETERS.copy()
        task = {
            "task_name": task_name,
            "task_type": "point",
            "params": {
                "dates": [
                    {
                        "startDate": start_date.strftime("%m-%d-%Y"),
                        "endDate": end_date.strftime("%m-%d-%Y"),
                    },
                ],
                "layers": [],
                "coordinates": [
                    {
                        "latitude": latitude,
                        "longitude": longitude,
                    }
                ],
            },
        }
        for parameter in parameters:
            if isinstance(parameter, tuple):
                task["params"]["layers"].append({
                    "product": parameter[0],
                    "layer": parameter[1],
                })
            else:
                task["params"]["layers"].append({
                    "product": cls.product(),
                    "layer": parameter,
                })
        cls.confirm_download(
            prefix=(
                "AppEEARS tasks can take hours to complete so this "
                "data will not be available immediately. "
            )
        )
        with EarthDataAppEEARSToken.get_global_session() as s:
            req = s.post(cls.URL + "/task", json=task)
            req.raise_for_status()
            out = req.json()
        # This is the task_id for the test task
        # out = {
        #     "task_id": "0a249513-42aa-402a-80b8-2a0dbd34cde1",
        #     "status": "pending",
        # }
        out.update(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
        return out

    @classmethod
    def check_status(cls, task_id: str,
                     session: Optional[requests.Session] = None) -> str:
        r"""Check the status of an AppEEARS task.

        Args:
            task_id: Task ID string.
            session: Requests session to reuse.

        Returns:
            str: Status string.

        """
        with EarthDataAppEEARSToken.get_global_session(session) as s:
            req = s.get(cls.URL + f"/task/{task_id}")
            contents = req.json()
            return contents["status"]

    @classmethod
    def bundle_files(cls, task_id: str,
                     session: Optional[requests.Session] = None) -> dict:
        r"""Get a record of the files generated by a task.

        Args:
            task_id: Task ID string.
            session: Requests session to reuse.

        Returns:
            dict: Task file summary.

        """
        with EarthDataAppEEARSToken.get_global_session(session) as s:
            req = s.get(cls.URL + f"/bundle/{task_id}")
            req.raise_for_status()
            return req.json()

    @classmethod
    def download_files(cls, task_id: str, dst: str,
                       bundle: Optional[dict] = None,
                       session: Optional[requests.Session] = None) -> None:
        r"""Download files produced by an AppEEARS task.

        Args:
            task_id: Task ID string.
            dst: Directory where files should be downloaded to.
            bundle: Pre-requested information about the files resulting
                from an AppEEARS task.
            session: Requests session to reuse.

        """
        cls.confirm_download()
        os.makedirs(dst, exist_ok=True)
        with EarthDataAppEEARSToken.get_global_session(session) as s:
            if bundle is None:
                bundle = cls.bundle_files(task_id, session=s)
            else:
                assert bundle["task_id"] == task_id
            for finfo in bundle["files"]:
                file_id = finfo["file_id"]
                fpath = os.path.join(dst, finfo["file_name"])
                if os.path.isfile(fpath):
                    continue
                req = s.get(cls.URL + f"/bundle/{task_id}/{file_id}",
                            allow_redirects=True,
                            stream=True)
                with open(fpath, 'wb') as fd:
                    for data in req.iter_content(chunk_size=8192):
                        fd.write(data)

    def complete_files(self) -> None:
        r"""Download the individual files for the required times/dates
        if they are available and have not been downloaded."""
        if self.status in ["pending", "processing"]:
            raise RuntimeError("Task has not finished running so "
                               "the file are not available")
        if self.status == "downloaded":
            return
        task_id = self.contents["task_id"]
        if "files" not in self.contents:
            self.contents["files"] = self.bundle_files(task_id)
            self.write(overwrite=True)
        self.download_files(task_id, self.bundle_dir,
                            bundle=self.contents["files"])
        self.contents["status"] = "downloaded"
        self.write(overwrite=True)


class SPL4SMGPData(NASAAppEEARSData):
    r"""Wrapper for requesting SMAP L4 data from the NASA AppEEARS REST
    API."""

    NAME: ClassVar[str] = "SPL4SMGP"
    DESC: ClassVar[str] = "SMAP L4 Soil Moisture Geophysical"
    VERSION: ClassVar[str] = "008"
    DEFAULT_DOWNLOAD_PARAMETERS: ClassVar[List[str]] = [
        "Geophysical_Data_land_evapotranspiration_flux",
        "Geophysical_Data_land_fraction_saturated",
        "Geophysical_Data_land_fraction_wilting",
        "Geophysical_Data_leaf_area_index",
        "Geophysical_Data_mwrtm_vegopacity",
        "Geophysical_Data_sm_profile",
        "Geophysical_Data_sm_profile_pctl",
        "Geophysical_Data_sm_profile_wetness",
        "Geophysical_Data_sm_rootzone",
        "Geophysical_Data_sm_rootzone_pctl",
        "Geophysical_Data_sm_rootzone_wetness",
        "Geophysical_Data_sm_surface",
        "Geophysical_Data_sm_surface_wetness",
        "Geophysical_Data_soil_water_infiltration_flux",
        "Geophysical_Data_vegetation_greenness_fraction",
        "Geophysical_Data_surface_temp",
        # "Geophysical_Data_soil_temp_layer1",
        # ...
        # "Geophysical_Data_soil_temp_layer6",
    ]

    @property
    def depths(self) -> list:
        r"""list: List of (start, end) depth pairs covered by this
        file."""
        # return [(0, 5)]  # cm, Surface measurements
        return [(0, 100)]  # cm, Root zone measurements


class EarthaccessData(BaseSoilFile):
    r"""Wrapper for requesting data via earthaccess."""

    NAME: ClassVar[str] = "earthaccess"
    VERSION: ClassVar[str] = None
    URL: ClassVar[str] = None
    STATIC_DATE_LIMITS: ClassVar[Tuple[datetime.date, datetime.date]] = None
    DONT_TEST: ClassVar[bool] = True
    PYTEST_MARKS: ClassVar[List[str]] = ["slow"]
    _default_ext: ClassVar[str] = ".json"
    REQUIRED_OPTIONAL_PACKAGES: ClassVar[List[str]] = ["earthaccess"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.exists:
            self.download_missing()
            self.collect_parameters()

    def collect_parameters(self):
        r"""Collect parameters from each map for the selected location."""
        if "parameters" in self.contents:
            return
        import h5py
        self.contents["parameters"] = {}
        for i, fname in enumerate(self.contents["files"]):
            time = self.contents["granules"][i]["time"]
            with h5py.File(fname, "r") as f:
                lat_grid = f["cell_lat"][:]
                lon_grid = f["cell_lon"][:]
                # TODO: Handle lon/lat ranges?
                dist = (
                    (lat_grid - self.latitude) ** 2
                    + (lon_grid - self.longitude) ** 2
                )
                x, y = np.unravel_index(np.argmin(dist), dist.shape)
                self.contents["parameters"].setdefault("time", [])
                self.contents["parameters"]["time"].append(time)
                for k in f["Geophysical_Data"].keys():
                    self.contents["parameters"].setdefault(k, [])
                    self.contents["parameters"][k].append(
                        float(f["Geophysical_Data"][k][x, y]))
        assert self.contents["parameters"]
        self.write(overwrite=True)

    def download_missing(self):
        r"""Download any missing files"""
        if "parameters" in self.contents:
            return
        files = [
            os.path.join(self.product_dir(), x["native-id"])
            for x in self.contents["granules"]
        ]
        missing = [x for x in files if not os.path.isfile(x)]
        # if "parameters" in self.contents:
        #     assert not missing
        if not missing:
            return
        self.confirm_download(
            prefix=(
                f"{len(missing)}/{len(self.contents['granules'])} "
                f"granules need to be downloaded. "
            ))
        for x in missing:
            self.download_granule(x)
        self.contents["files"] = files
        self.write(overwrite=True)

    @classmethod
    def product_name(cls) -> str:
        r"""str: Underlying product name."""
        # EA_ prefix is added to avoid conflict with AppEEARS classes
        assert cls.NAME.startswith("EA_")
        return cls.NAME[3:]

    @classmethod
    def product_dir(cls) -> str:
        r"""str: Directory where downloaded products will be stored."""
        return os.path.join(EarthaccessData.DEFAULT_CACHE_DIR,
                            cls.product_name())

    @readonly_cached_property
    def df(self):
        r"""pandas.DataFrame: Data frame containing the data."""
        import pandas as pd
        return pd.DataFrame.from_dict(self.contents["parameters"])

    @property
    def parameters(self) -> list:
        r"""list: Set of parameters contained by this file."""
        return list(self.df.columns)

    @classmethod
    def get_granule_time(
            cls, x, timezone: Optional[datetime.tzinfo] = None
    ) -> datetime.datetime:
        times = x["umm"]["TemporalExtent"]["RangeDateTime"]
        start = datetime.datetime.fromisoformat(
            times["BeginningDateTime"])
        end = datetime.datetime.fromisoformat(
            times["EndingDateTime"])
        if timezone is not None:
            start = start.astimezone(timezone)
            end = end.astimezone(timezone)
        return start + (end - start) / 2.0

    @classmethod
    def download_data(
            cls, latitude: float, longitude: float,
            start_date: Optional[datetime.date] = None,
            end_date: Optional[datetime.date] = None,
            parameters: Optional[List[str]] = None,
            temporal_resolution: Optional[int | datetime.timedelta] = 10,
    ) -> dict:
        r"""Use earthaccess to get SMAP L4 data. This requires that
        EARTHDATA_USERNAME and EARTHDATA_PASSWORD provide the necessary
        credentials.

        Args:
            latitude: Location latitude (degrees).
            longitude: Location longitude (degrees).
            parameters: Set of parameters to request.
            start_date: Starting date for data [UNUSED].
            end_date: Ending date for data [UNUSED].

        Returns:
            dict: JSON result.

        """
        if parameters is None:
            parameters = cls.DEFAULT_DOWNLOAD_PARAMETERS.copy()
        if isinstance(temporal_resolution, int):
            temporal_resolution = datetime.timedelta(temporal_resolution)
        assert temporal_resolution >= datetime.timedelta(1)
        kws = {}
        if isinstance(latitude, tuple) and isinstance(longitude, tuple):
            kws["bounding_box"] = (
                longitude[0], latitude[0],
                longitude[1], latitude[1]
            )
        elif isinstance(latitude, tuple):
            assert isinstance(longitude, float)
            kws["line"] = [
                (longitude, latitude[0]),
                (longitude, latitude[1]),
            ]
        elif isinstance(longitude, tuple):
            assert isinstance(latitude, float)
            kws["line"] = [
                (longitude[0], latitude),
                (longitude[1], latitude),
            ]
        elif isinstance(latitude, list) and isinstance(longitude, list):
            assert len(latitude) == len(longitude)
            kws["line"] = list(zip(longitude, latitude))
        elif isinstance(latitude, float) and isinstance(longitude, float):
            kws["point"] = (longitude, latitude)
        else:
            raise NotImplementedError(f"latitude = {latitude}, "
                                      f"longitude = {longitude}")
        if start_date is not None or end_date is not None:
            assert start_date is not None
            assert end_date is not None
            kws["temporal"] = (
                start_date.isoformat(), end_date.isoformat()
            )
        EarthDataAuth.get_global_auth()
        earthaccess = EarthDataAuth.get_global_earthaccess()
        results = earthaccess.search_data(
            short_name=cls.product_name(),
            downloadable=True,
            version=cls.VERSION,
            **kws
        )
        # Downsample based on temporal_resolution to reduce the number of
        #   files downloaded. The sample with the time closest to noon
        #   on the downsampled date at the selected location will be
        #   used.
        import timezonefinder
        import pytz
        tf = timezonefinder.TimezoneFinder()
        timezone = pytz.timezone(tf.certain_timezone_at(
            lat=np.mean(latitude), lng=np.mean(longitude)))
        start_noon = datetime.datetime(
            start_date.year, start_date.month, start_date.day, 12, 0, 0,
            tzinfo=timezone,
        )
        nstep = int((end_date - start_date) / temporal_resolution)
        dates = (start_date + temporal_resolution * np.arange(nstep)).tolist()
        dates.append(end_date)
        results_grouped = {}
        for x in results:
            time = cls.get_granule_time(x, timezone)
            offset = time - start_noon
            idx = int(np.round(offset / temporal_resolution))
            rem = abs(offset - (idx * temporal_resolution)).total_seconds()
            date = dates[idx]
            results_grouped.setdefault(date, [])
            results_grouped[date].append((rem, time, x))
        results_selected = [
            min(results_grouped[k], key=lambda x: x[0])
            for k in sorted(results_grouped.keys())
        ]
        assert len(results_selected) == len(dates)
        out = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "granules": [
                dict(x[2]["meta"], time=x[1].isoformat())
                for x in results_selected
            ],
        }
        return out

    @classmethod
    def download_granule(cls, granule: dict) -> str:
        r"""Download a data granule from NASA systems using earthaccess.

        Args:
            granule: Granule metadata.

        Returns:
            str: The file where the granule was saved.

        """
        if not os.path.isdir(cls.product_dir()):
            os.mkdir(cls.product_dir())
        fname = os.path.join(cls.product_dir(), granule["native-id"])
        if os.path.isfile(fname):
            return fname
        logger.info(f"Downloading granule {fname}")
        EarthDataAuth.get_global_auth()
        earthaccess = EarthDataAuth.get_global_earthaccess()
        results = earthaccess.search_data(
            short_name=cls.product_name(),
            granule_name=granule["native-id"],
            version=cls.VERSION,
            downloadable=True,
        )
        assert len(results) == 1
        files = earthaccess.download(results, cls.product_dir())
        assert len(files) == 1
        assert str(files[0]) == fname
        return fname


class SMAPL4SoilData(EarthaccessData):
    r"""Wrapper for loading soil data from SMAP L4 using earthaccess."""

    NAME: ClassVar[str] = "EA_SPL4SMGP"
    DESC: ClassVar[str] = "SMAP L4 Soil Moisture Geophysical"
    VERSION: ClassVar[str] = "008"
    URL: ClassVar[str] = "https://nsidc.org/data/spl4smgp/versions/8"
    # STATIC_LOCATION_LIMITS: ClassVar[
    #     # Min latitude, max latitude, min longitude, max longitude
    #     Tuple[float, float, float, float]] = (
    #         -85.044, 85.044, -180.0, 180.0
    #     )
    # DEFAULT_DATE_RANGE: ClassVar[Tuple[datetime.date, datetime.date]] = (
    #     datetime.date(2015, 3, 31), datetime.date(2026, 5, 8))
    DATE_LIMITS: ClassVar[Tuple[datetime.date, datetime.date]] = (
        datetime.date(2015, 3, 31), datetime.date.today())

    @property
    def depths(self) -> list:
        r"""list: List of (start, end) depth pairs covered by this
        file."""
        # return [(0, 5)]  # cm, Surface measurements
        return [(0, 100)]  # cm, Root zone measurements
