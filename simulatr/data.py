import os
import glob
import numpy as np
import datetime
import requests
from abc import abstractmethod
from typing import Optional, Union, List, Any, Tuple, ClassVar
from .base import (
    readonly_cached_property,
    _ModelFileMeta, BaseModelFile,
)
from . import logger
from .utils import cfg


class BaseDataFile(BaseModelFile):
    r"""Base class for external data files."""

    CACHED: ClassVar[bool] = True
    DEFAULT_EXTERNAL_TYPE: ClassVar[str] = None
    DEFAULT_DOWNLOAD_PARAMTERS: ClassVar[List[str]] = []
    REQUIRED_EXTERNAL_PARAMETERS: ClassVar[dict] = {}
    DEFAULT_DATE_RANGE: ClassVar[Tuple[datetime.date, datetime.date]] = None
    DEFAULT_CACHE_DIR: ClassVar[str] = None
    DATE_LIMITS: ClassVar[Tuple[datetime.date, datetime.date]] = None
    _EXPECTS_DIRECTORY: ClassVar[bool] = False
    _make_interactive = None

    @classmethod
    def time_specific(cls) -> bool:
        r"""bool: True if the class tracks time, False otherwise."""
        return (cls.DEFAULT_DATE_RANGE is not None)

    @property
    @abstractmethod
    def parameters(self) -> list:
        r"""list: Set of parameters contained by this file."""
        raise NotImplementedError  # pragma: no cover

    @property
    @abstractmethod
    def latitude(self) -> float:
        r"""float: Latitude (degrees)."""
        raise NotImplementedError  # pragma: no cover

    @property
    @abstractmethod
    def longitude(self) -> float:
        r"""float: Longitude (degrees)."""
        raise NotImplementedError  # pragma: no cover

    @property
    def start_date(self) -> datetime.date:
        r"""datetime.date: Start of range covered by the file."""
        if not self.time_specific():
            return None
        raise NotImplementedError  # pragma: no cover

    @property
    def end_date(self) -> datetime.date:
        r"""datetime.date: End of range covered by the file."""
        if not self.time_specific():
            return None
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def _f2str(cls, x: Any) -> str:
        r"""Convert a value to a string for use in a file name."""
        if isinstance(x, datetime.date):
            return x.isoformat()
        elif isinstance(x, float):
            return str(x).replace(".", "p").replace("-", "n")
        elif isinstance(x, tuple):
            return "-".join([cls._f2str(xx) for xx in x])
        return x

    @classmethod
    def format_filename(cls, latitude: float | Tuple[float, float],
                        longitude: float | Tuple[float, float],
                        start_date: Optional[datetime.date] = None,
                        end_date: Optional[datetime.date] = None,
                        cache_dir: Optional[str] = None) -> str:
        r"""Construct the file name for the cached data file containing
        the requested data.

        Args:
            latitude: Location latitude (degrees).
            longitude: Location longitude (degrees).
            start_date: Starting date for data.
            end_date: Ending date for data.
            cache_dir: Directory where the data should be cached.

        Returns:
            str: File name.

        """
        cache_dir = cache_dir or cls.DEFAULT_CACHE_DIR

        out = f"{cls._f2str(latitude)}_{cls._f2str(longitude)}"
        if cls.time_specific():
            start_date = start_date or cls.DEFAULT_DATE_RANGE[0]
            end_date = end_date or cls.DEFAULT_DATE_RANGE[1]
            out += f"_{cls._f2str(start_date)}_to_{cls._f2str(end_date)}"
        out += cls._default_ext
        if cache_dir:
            out = os.path.join(cache_dir, out)
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
    def base_cls(cls):
        r"""type: Default base class that should be used to fetch data
        for this class."""
        # TODO: Use a ranking instead?
        return _ModelFileMeta.get_filetype(
            cls.CATEGORY, cls.DEFAULT_EXTERNAL_TYPE)

    @classmethod
    def from_location(
            cls, latitude: float, longitude: float,
            start_date: Union[datetime.date, datetime.datetime] = None,
            end_date: Union[datetime.date, datetime.datetime] = None,
            parameters: Optional[List[str]] = None,
            cache_dir: Optional[str] = None
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

        Returns:
            BaseDataFile: File instance.

        """
        base_cls = cls.base_cls()
        if base_cls != cls:
            if parameters is None:
                parameters = cls.REQUIRED_EXTERNAL_PARAMETERS.get(
                    base_cls.NAME, None)
            fbase = base_cls.from_location(
                latitude, longitude, start_date, end_date,
                parameters=parameters)
            return cls.from_base(fbase)
        cache_dir = cache_dir or cls.DEFAULT_CACHE_DIR
        if cls.time_specific():
            start_date = start_date or cls.DEFAULT_DATE_RANGE[0]
            end_date = end_date or cls.DEFAULT_DATE_RANGE[1]
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
                if fstart <= start_date and fend >= end_date:
                    out = cls(f)
                    if parameters:
                        out.add_missing_param(parameters)
                    return out
            start_date = min(start_date, cls.DEFAULT_DATE_RANGE[0])
            end_date = max(end_date, cls.DEFAULT_DATE_RANGE[1])
            assert end_date > start_date
            if cls.DATE_LIMITS is not None:
                if start_date < cls.DATE_LIMITS[0]:
                    raise ValueError(
                        f"The requested start date ({start_date}) "
                        f"predates the minimum ({cls.DATE_LIMITS[0]})")
                if end_date > cls.DATE_LIMITS[1]:
                    raise ValueError(
                        f"The requested end date ({end_date}) excedes "
                        f"the maximum ({cls.DATE_LIMITS[1]})")
        fname = cls.format_filename(latitude, longitude,
                                    start_date, end_date,
                                    cache_dir=cache_dir)
        if cls.path_exists(fname):
            assert not cls.time_specific()
            out = cls(fname)
            if parameters:
                out.add_missing_param(parameters)
            return out
        data = cls.download_data(latitude, longitude,
                                 start_date, end_date,
                                 parameters=parameters)
        if not os.path.isdir(os.path.dirname(fname)):
            os.mkdir(os.path.dirname(fname))
        out = cls(fname, contents=data)
        out.write()
        return out

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

    def calculate_missing(self):
        r"""Calculate missing parameters that are required."""
        pass

    @classmethod
    def from_file(cls, fname: str | BaseModelFile) -> BaseModelFile:
        r"""Create an instance by loading it from a file.

        Args:
            fname: File or file instance to create an instance from. If
                an instance of this class is provided, it will be
                returned.

        Returns:
            File instance.

        """
        if ((isinstance(fname, BaseModelFile)
             and fname.CATEGORY == cls.CATEGORY)):
            return cls.from_base(fname)
        return super().from_file(fname)

    @classmethod
    def from_base(cls, fbase: Union[str, "BaseDataFile"],
                  fname: Optional[str] = None) -> "BaseDataFile":
        r"""Create a data file from another file of the same category.

        Args:
            src: JSON file containing data.
            fname: File name where the data should be written to.

        """
        if isinstance(fbase, str):
            fbase = cls.base_cls()(fbase)
        if fname is None:
            fname = os.path.splitext(fbase.fname)[0] + cls._default_ext
        if cls.path_exists(fname):
            return cls(fname)
        if isinstance(fbase, cls):
            contents = fbase.contents
        else:
            if fbase.NAME in cls.REQUIRED_EXTERNAL_PARAMETERS:
                fbase.add_missing_param(
                    cls.REQUIRED_EXTERNAL_PARAMETERS[fbase.NAME])
            contents = cls._from_base(fbase)
        out = cls(fname, contents=contents)
        out.calculate_missing()
        out.write()
        return out

    @classmethod
    def _from_base(cls, src: "BaseDataFile"):
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
            return (a is None or b is None or a == b)
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
        if isinstance(start, datetime.datetime):
            start = start.date()
        if isinstance(end, datetime.datetime):
            end = end.date()
        if not self._check_inside_range(latitude, self.latitude):
            return False
        if not self._check_inside_range(longitude, self.longitude):
            return False
        if ((self.time_specific()
             and not self._check_inside_range(
                 (start, end), (self.start_date, self.end_date)))):
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

    @abstractmethod
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
    CACHED: ClassVar[bool] = True
    DEFAULT_EXTERNAL_TYPE: ClassVar[str] = "NASA POWER"
    REQUIRED_EXTERNAL_PARAMETERS: ClassVar[dict] = {
        "NASA POWER": [
            "TOA_SW_DWN",
            "ALLSKY_SFC_SW_DWN",  # MJ
            "T2M", "T2M_MIN", "T2M_MAX",  # C
            "T2MDEW",  # C
            "WS2M",  # wind
            "PRECTOTCORR",  # mm
        ],
    }
    DEFAULT_DATE_RANGE: ClassVar[Tuple[datetime.date, datetime.date]] = (
        datetime.date(1981, 1, 1), datetime.date(2026, 5, 8))
    DATE_LIMITS: ClassVar[Tuple[datetime.date, datetime.date]] = (
        datetime.date(1981, 1, 1), datetime.date.today())

    @property
    def parameters(self) -> list:
        r"""list: Set of power parameters contained by this file."""
        return self.REQUIRED_EXTERNAL_PARAMETERS[self.DEFAULT_EXTERNAL_TYPE]

    @property
    @abstractmethod
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

    NAME: ClassVar[str] = "NASA POWER"
    CACHED: ClassVar[bool] = True
    DEFAULT_CACHE_DIR: ClassVar[str] = (
        cfg["directories"]["nasa_power_weather_data"])

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
            parameters = cls.REQUIRED_EXTERNAL_PARAMETERS["NASA POWER"]
        # build URL for retrieving data, using new NASA POWER api
        server = "https://power.larc.nasa.gov/api/temporal/daily/point"
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
        req = requests.get(server, params=payload)
        req.raise_for_status()
        # if req.status_code != 200:
        #     raise RuntimeError(
        #         f"Failed retrieving POWER data, server returned HTTP "
        #         f"code: {req.status_code} on following URL {req.url}"
        #     )
        logger.debug("Successfully retrieved data from NASA POWER")
        return req.json()


class BaseSoilFile(BaseDataFile):
    r"""Base class for soil files."""

    CATEGORY: ClassVar[str] = "soil"
    NAME: ClassVar[str] = None
    CACHED: ClassVar[bool] = True
    DEFAULT_EXTERNAL_TYPE: ClassVar[str] = "ISRIC SoilGrids"
    REQUIRED_EXTERNAL_PARAMETERS: ClassVar[dict] = {
        "ISRIC SoilGrids": [
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
    }

    @property
    @abstractmethod
    def depths(self) -> list:
        r"""list: List of (start, end) depth pairs covered by this
        file."""
        raise NotImplementedError  # pragma: no cover


class ISRICSoilGridsFile(BaseSoilFile):
    r"""Wrapper for loading ISRIC SoilGrids data from REST API."""

    NAME: ClassVar[str] = "ISRIC SoilGrids"
    CACHED: ClassVar[bool] = True
    SOIL_GRIDS_DEPTHS: ClassVar[dict] = {
        f"{start}-{end}cm": (start, end) for start, end in
        [(0, 5), (5, 15), (15, 30), (30, 60), (60, 100), (100, 200)]
    }
    DEFAULT_CACHE_DIR: ClassVar[str] = (
        cfg["directories"]["isric_soil_data"])

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
        server = ("https://rest.isric.org/soilgrids/v2.0/"
                  "properties/query")
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
        req = requests.get(server, params=payload)
        req.raise_for_status()
        # if req.status_code != 200:
        #     raise RuntimeError(
        #         f"Failed retrieving SoilGrids data, server returned "
        #         f"HTTP code: {req.status_code} on following URL "
        #         f"{req.url}"
        #     )
        logger.debug("Successfully retrieved data from ISRIC SoilGrids")
        return req.json()


class SSURGOSoilFile(BaseSoilFile):
    r"""Wrapper for loading soil data from SSURGO.

    """

    NAME: ClassVar[str] = "SSURGO"
    CACHED: ClassVar[bool] = True
    _default_ext: ClassVar[str] = ".xml"
    DEFAULT_CACHE_DIR: ClassVar[str] = (
        cfg["directories"]["ssurgo_soil_data"])
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
    DEFAULT_DOWNLOAD_PARAMTERS: ClassVar[List[str]] = [
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
    def base_cls(cls):
        r"""type: Default base class that should be used to fetch data
        for this class."""
        return cls

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
            parameters = cls.DEFAULT_DOWNLOAD_PARAMTERS.copy()
        parameters = [
            x for x in cls.REQUIRED_DOWNLOAD_PARAMTERS
            if x not in parameters
        ] + parameters
        lonLat = f"{longitude} {latitude}"
        url = (
            "https://SDMDataAccess.nrcs.usda.gov/Tabular/"
            "SDMTabularService.asmx"
        )
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
            url, data=body, headers=headers, timeout=140)
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
        # out = ET.fromstring(req.content)
        # out[0][0][0][1][0]
        # Array of arrays of fields?
        # import pdb; pdb.set_trace()
        return out

    @property
    def parameters(self) -> list:
        r"""list: Set of parameters contained by this file."""
        root = self.contents.getroot()
        return [x.tag for x in root[0][0][0][1][0][0]]

    @property
    def data_frame(self):
        root = self.contents.getroot()
        data = {k: [] for k in self.parameters}
        for row in root[0][0][0][1][0]:
            for x in row:
                if x.tag in self.STRING_FIELDS or x.text is None:
                    data[x.tag].append(x.text)
                else:
                    data[x.tag].append(float(x.text))
        import pandas as pd
        out = pd.DataFrame.from_dict(data)
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
        raise NotImplementedError  # pragma: no cover

    def update_param(self, contents: Any) -> None:
        r"""Merge downloaded parameters into the current data.

        Args:
            contents: New data to incorporate.

        """
        raise NotImplementedError  # pragma: no cover


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
    CACHED: ClassVar[bool] = True
    _default_ext: ClassVar[str] = ""
    _EXPECTS_DIRECTORY: ClassVar[bool] = True
    DEFAULT_CACHE_DIR: ClassVar[str] = (
        cfg["directories"]["humeris_soil_data"])

    @classmethod
    def base_cls(cls):
        r"""type: Default base class that should be used to fetch data
        for this class."""
        return cls

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
