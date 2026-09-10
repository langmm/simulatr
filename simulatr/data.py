import os
import copy
import uuid
import glob
import numpy as np
import datetime
import requests
import contextlib
import importlib
import json
# import yggdrasil_rapidjson as json
from yggdrasil_rapidjson import units as rj_units
from functools import cached_property
from abc import ABC
from typing import (
    Optional, Union, List, Any, Tuple, ClassVar, Iterator, Callable,
    Mapping
)
from . import logger, utils
from .utils import (
    cfg, NoDefault, create_registry_metaclass, readonly_cached_property,
    CachedPropertyMixin, promptuser_boolean
)


FileMeta = create_registry_metaclass(
    ("CATEGORY", "NAME"), ABC,
    noninheritable_attr={
        "DONT_TEST": False,
        "IS_BASE_CLASS": False,
    },
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
    IS_BASE_CLASS: ClassVar[bool] = False

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
        if fname.endswith((".json", ".apsimx")):
            with open(fname, "r") as fd:
                return json.load(fd)
        elif fname.endswith(".csv"):
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
        if fname.endswith(".json"):
            with open(fname, "w") as fd:
                json.dump(contents, fd, indent=4)
                fd.write("\n")
            return
        elif fname.endswith(".csv"):
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
        value that should be used if a KeyError is raised by
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
        if fpath is None:
            return False
        if cls._EXPECTS_DIRECTORY:
            return os.path.isdir(fpath)
        return os.path.isfile(fpath)

    def get(self, name: str, default: Any = NoDefault,
            **kwargs: Any) -> Any:
        r"""Get a parameter from the file.

        Args:
            name: Parameter name.
            default: Value to return if the parameter can't be found.
            \*\*kwargs: Additional keyword arguments are passed to _get.

        Returns:
            Parameter value.

        """
        if name in self._cached_properties:
            return self._cached_properties[name]
        try:
            out = self._get(name, **kwargs)
            self._cached_properties[name] = out
            return out
        except KeyError:
            if default is not NoDefault:
                return default
            raise

    def set(self, name: str, value: Any, **kwargs: Any) -> None:
        r"""Set a parameter in the file.

        Args:
            name: Parameter name.
            value: Parameter value.
            \*\*kwargs: Additional keyword arguments are passed to _set.

        Raises:
            KeyError: If name is not a valid parameter name.

        """
        with self.prevent_overwrite():
            self._set(name, value, **kwargs)

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
    UNIVERSAL_PARAMETERS: ClassVar[List[str]] = []
    SUPPORTED_PARAMETERS: ClassVar[List[str]] = []
    SUPPORTED_UNIVERSAL_PARAMETERS: ClassVar[List[str]] = []
    DEFAULT_EXTERNAL_TYPE: ClassVar[type] = None
    DEFAULT_DOWNLOAD_PARAMETERS: ClassVar[List[str]] = []
    REQUIRED_DOWNLOAD_PARAMETERS: ClassVar[List[str]] = []
    REQUIRED_EXTERNAL_PARAMETERS: ClassVar[dict] = {}
    REQUIRED_PARAMETERS: ClassVar[List[str]] = []
    HEADER_PARAMETERS: ClassVar[List[str]] = []
    STATIC_PARAMETERS: ClassVar[dict] = {}
    PARAMETER_DEFAULTS: ClassVar[dict] = {}
    PARAMETER_LIMITS: ClassVar[dict] = {
        "latitude": (-90.0, 90.0),
        "longitude": (-180.0, 180.0),
    }
    DEFAULT_DATE_RANGE: ClassVar[Tuple[datetime.date, datetime.date]] = None
    DEFAULT_CACHE_DIR: ClassVar[str] = None
    PYTEST_MARKS: ClassVar[List[str]] = []
    REQUIRED_OPTIONAL_PACKAGES: ClassVar[List[str]] = []
    PARAMETER_ALIASES: ClassVar[Mapping[str, str]] = {}
    PARAMETER_ALIASES_REV: ClassVar[Mapping[str, str]] = {}
    PARAMETER_UNITS: ClassVar[Mapping[str, str]] = {}
    PARAMETER_CALCULATORS: ClassVar[dict] = {}
    PARAMETER_FILL_VALUE: ClassVar[float | dict | None] = None
    UNIVERSAL_LOCATION_PARAMETERS: ClassVar[List[str]] = [
        "latitude", "longitude",
    ]
    UNIT_ALIASES: ClassVar[Mapping[str, str]] = {}
    UNIT_ALIASES_REV: ClassVar[Mapping[str, str]] = {}
    IS_BASE_CLASS: ClassVar[bool] = True

    @staticmethod
    def _on_registration(cls):
        if cls.IS_BASE_CLASS:
            cls.DONT_TEST = True
        else:
            assert cls.NAME is not None
            cls.PARAMETER_CALCULATORS = copy.deepcopy(
                cls.PARAMETER_CALCULATORS)
            if cls.DESC is None:
                cls.DESC = cls.NAME
            if cls.DEFAULT_CACHE_DIR is None:
                if cls.NAME not in cfg["directories"]:
                    cfg["directories"][cls.NAME] = os.path.join(
                        os.getcwd(), cls.NAME)
                cls.DEFAULT_CACHE_DIR = cfg["directories"][cls.NAME]
            cls.PARAMETER_LIMITS.setdefault("latitude", (-90.0, 90.0))
            cls.PARAMETER_LIMITS.setdefault("longitude", (-180.0, 180.0))
            if (("start_date" in cls.STATIC_PARAMETERS
                 and "start_date" not in cls.PARAMETER_LIMITS)):
                date_limits = (
                    cls.STATIC_PARAMETERS["start_date"],
                    cls.STATIC_PARAMETERS["end_date"]
                )
                cls.PARAMETER_LIMITS["start_date"] = date_limits
                cls.PARAMETER_LIMITS["end_date"] = date_limits
            if "start_date" in cls.PARAMETER_LIMITS:
                cls.PARAMETER_LIMITS["end_date"] = cls.PARAMETER_LIMITS[
                    "start_date"]
            for k, v in cls.STATIC_PARAMETERS.items():
                if k not in cls.PARAMETER_LIMITS and isinstance(v, tuple):
                    cls.PARAMETER_LIMITS[k] = v
            if ((cls.DEFAULT_DATE_RANGE is None
                 and cls.DEFAULT_EXTERNAL_TYPE is not None)):
                cls.DEFAULT_DATE_RANGE = (
                    cls.DEFAULT_EXTERNAL_TYPE.DEFAULT_DATE_RANGE
                )
            cls.PARAMETER_ALIASES_REV = {
                v: k for k, v in cls.PARAMETER_ALIASES.items()
            }
            cls.UNIT_ALIASES_REV = {
                v: k for k, v in cls.UNIT_ALIASES.items()
            }
            cls.SUPPORTED_PARAMETERS = sorted(list(set(
                cls.HEADER_PARAMETERS
                + cls.REQUIRED_PARAMETERS
                + list(cls.STATIC_PARAMETERS.keys())
                + list(cls.PARAMETER_ALIASES.keys())
                + list(cls.PARAMETER_DEFAULTS.keys())
                + list(cls.PARAMETER_LIMITS.keys())
                + list(cls.PARAMETER_UNITS.keys())
            )))
            cls.SUPPORTED_UNIVERSAL_PARAMETERS = []
            for k in cls.SUPPORTED_PARAMETERS:
                kuni = cls.PARAMETER_ALIASES.get(k, k)
                if kuni in cls.UNIVERSAL_PARAMETERS:
                    cls.SUPPORTED_UNIVERSAL_PARAMETERS.append(kuni)
            for k in ["latitude", "longitude"]:
                cls.PARAMETER_UNITS.setdefault(k, "degrees")
            if not cls.DEFAULT_DOWNLOAD_PARAMETERS:
                cls.DEFAULT_DOWNLOAD_PARAMETERS = [
                    cls.PARAMETER_ALIASES_REV.get(k, k)
                    for k in cls.UNIVERSAL_PARAMETERS
                    if ((k in cls.PARAMETER_ALIASES_REV
                         or k in cls.PARAMETER_UNITS)
                        and k not in cls.HEADER_PARAMETERS)
                ]

    def __init__(self, *args, **kwargs):
        self._CALCULATING = []
        super().__init__(*args, **kwargs)

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
    def parameter_calculation(cls, name: str, *args: str) -> Callable:
        r"""Decorator for a BaseFile method that calculates a parameter
        from other parameters.

        Args:
            name: Name of the parameter that the method calculates.
            \*args: Additional keyword arguments are the names of
                 parameters that are required by the calculation.

        Returns:
            Decorator callable.

        """

        def _parameter_calculation(method):
            method_name = method.__qualname__.rsplit('.', 1)[-1]
            cls.PARAMETER_CALCULATORS.setdefault(name, [])
            cls.PARAMETER_CALCULATORS[name].append((method_name, args))
            return method

        return _parameter_calculation

    # def _can_calculate(self, name: str) -> bool:

    def calculate_parameter(self, name: str) -> Any:
        r"""Try calculating a parameter using registered parameter
        calculators.

        Args:
            name: Name of the parameter to calculate.

        Returns:
            Result of the calculation if successful.

        Raises:
            KeyError: If the callculation fails.

        """
        name_int, name_ext = self.param_name_aliases(name)
        for iname in set([name_int, name_ext]):
            for method, required in self.PARAMETER_CALCULATORS.get(iname, []):
                if ((name_ext in self._CALCULATING
                     or any(x in self._CALCULATING for x in required))):
                    raise KeyError(f"Recursive {iname}")
                self._CALCULATING.append(iname)
                try:
                    if not hasattr(self, method):
                        continue
                    try:
                        args = tuple([self.get(x) for x in required])
                        return getattr(self, method)(*args)
                    except KeyError:
                        pass
                finally:
                    self._CALCULATING.pop()
        raise KeyError(name)

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
        return ("start_date" not in cls.STATIC_PARAMETERS)

    @classmethod
    def location_specific(cls) -> bool:
        r"""bool: True if the class has one instance per location,
        False otherwise."""
        return ("latitude" not in cls.STATIC_PARAMETERS)

    @classmethod
    def get_units(cls, value: Any) -> str:
        r"""Get value units if it has any.

        Args:
            value: Parameter value.

        Returns:
            str: Units string if the value has any units, None if it
                does not.

        """
        if isinstance(value, (rj_units.Quantity, rj_units.QuantityArray)):
            return str(value.units)
        elif value is None or isinstance(value, (float, int, np.int_,
                                                 np.ndarray,
                                                 datetime.date, str)):
            return None
        elif isinstance(value, (list, tuple)):
            if len(value) == 0:
                return None
            out = cls.get_units(value[0])
            assert all(cls.get_units(v) == out for v in value)
            return out
        elif isinstance(value, GeoTIFFWrapper):
            return value.units
        else:
            raise TypeError(type(value))
        return value

    @classmethod
    def remove_units(cls, value: Any) -> Any:
        r"""Remove units from a parameter value.

        Args:
            value: Parameter value.

        Returns:
            Value without units.

        """
        if isinstance(value, (rj_units.Quantity, rj_units.QuantityArray)):
            return value.value
        elif value is None or isinstance(value, (float, int, np.int_,
                                                 np.ndarray,
                                                 datetime.date, str)):
            return value
        elif isinstance(value, list):
            return [cls.remove_units(v) for v in value]
        elif isinstance(value, tuple):
            return tuple([cls.remove_units(v) for v in value])
        elif isinstance(value, GeoTIFFWrapper):
            value = copy.deepcopy(value)
            value.units = None
            return value
        else:
            raise TypeError(type(value))
        return value

    @classmethod
    def add_units(cls, value: Any, value_units: str,
                  return_scalar: Optional[bool] = False) -> Any:
        r"""Add units to a parameter value. If it already has units,
        the value will be converted.

        Args:
            value: Parameter value.
            value_units: Units to add/convert value to.
            return_scalar: If True, return the value scalar without
                units after performing any conversion.

        Returns:
            Value with units.

        """
        if value_units is None:
            if return_scalar:
                return cls.remove_units(value)
            return value
        if isinstance(value, (rj_units.Quantity, rj_units.QuantityArray)):
            out = value.to(value_units)
            if return_scalar:
                return out.value
            return out
        if value is None or isinstance(value, (datetime.date, str)):
            return value
        elif isinstance(value, (float, int, np.int_)):
            if return_scalar:
                return value
            return rj_units.Quantity(float(value), value_units)
        elif isinstance(value, list):
            out = [
                cls.add_units(v, value_units,
                              return_scalar=return_scalar)
                for v in value
            ]
            if all(isinstance(x, rj_units.Quantity) for x in out):
                out_units = out[0].units
                out_value = [x.value for x in out]
                return rj_units.QuantityArray(
                    np.array(out_value), out_units)
            elif all(isinstance(x, float) for x in out):
                return np.array(out)
            return out
        elif isinstance(value, tuple):
            return tuple([
                cls.add_units(v, value_units,
                              return_scalar=return_scalar)
                for v in value
            ])
        elif isinstance(value, np.ndarray):
            if return_scalar:
                return value
            return rj_units.QuantityArray(value, value_units)
        elif isinstance(value, GeoTIFFWrapper):
            out = copy.deepcopy(value)
            out.units = value_units
            if return_scalar:
                out.units = None
            return out
        raise TypeError(type(value))

    @classmethod
    def add_param_units(cls, name: str, value: Any,
                        return_scalar: Optional[bool] = False) -> Any:
        r"""Add units to a parameter value.

        Args:
            name: Parameter name.
            value: Parameter value.
            return_scalar: If True, return the value scalar without
                units after performing any conversion.

        Returns:
            Value with units.

        """
        name_int, name_ext = cls.param_name_aliases(name)
        if name_int not in cls.PARAMETER_UNITS:
            if return_scalar:
                return cls.remove_units(value)
            return value
        value_units = cls.PARAMETER_UNITS[name_int]
        value_units = cls.UNIT_ALIASES.get(value_units, value_units)
        return cls.add_units(value, value_units,
                             return_scalar=return_scalar)

    @classmethod
    def get_from_other(cls, src: "BaseDataFile", name: str,
                       default: Optional[Any] = NoDefault,
                       with_units: Optional[bool] = False) -> Any:
        r"""Get a parameter from another file, converting it into the
        correct units if necessary.

        Args:
            src: Data file that the parameter should be retrieved from.
            name: Parameter name.
            default: Value to return if the parameter can't be found.
            with_units: If True, the returned value will be of a type
                with units if there are units for the parameter.

        Returns:
            Parameter value.

        """
        name_int, name_ext = cls.param_name_aliases(name)
        out = src.get(name_ext, default)
        out = cls.add_param_units(name_int, out,
                                  return_scalar=(not with_units))
        return out

    def get(self, name: str, default: Optional[Any] = NoDefault,
            **kwargs: Any) -> Any:
        r"""Get a parameter from the file.

        Args:
            name: Parameter name.
            default: Value to return if the parameter can't be found.
            \*\*kwargs: Additional keyword arguments are passed to _get.

        Returns:
            Parameter value.

        """
        out = None
        name_int, name_ext = self.param_name_aliases(name)
        if name_ext in self.STATIC_PARAMETERS:
            out = self.STATIC_PARAMETERS[name_ext]
        else:
            parameters = self.internal_parameters(
                include_header=True)
            parameters_calculated = self.internal_parameters(
                include_header=True, include_calculated=True)
            try:
                if name_int in parameters:
                    out = super().get(name_int, **kwargs)
                    if isinstance(out, np.ndarray):
                        fill_value = self.param_fill_value(name_int)
                        if fill_value is not None:
                            out[out == fill_value] = np.nan
                elif name_int in parameters_calculated:
                    out = self.calculate_parameter(name_int)
            except KeyError:
                pass
            if out is None:
                if default is NoDefault:
                    raise KeyError(
                        f"{name_int} not supported. Valid parameters:"
                        f" {parameters + parameters_calculated}")
                out = default
        out = self.add_param_units(name_int, out)
        if name_ext in ["start_date", "end_date"] and isinstance(out, str):
            out = datetime.date.fromisoformat(out)
        return out

    def set(self, name: str, value: Any, **kwargs: Any) -> None:
        r"""Set a parameter in the file.

        Args:
            name: Parameter name.
            value: Parameter value.
            \*\*kwargs: Additional keyword arguments are passed to _set.

        Raises:
            KeyError: If name is not a valid parameter name.

        """
        name_int, name_ext = self.param_name_aliases(name)
        value = self.add_param_units(name_int, value, return_scalar=True)
        value_units = self.get_units(value)
        assert value_units is None
        if name_int not in self.SUPPORTED_PARAMETERS:
            raise KeyError(f"Unsupported parameter \"{name}\"")
        if name_ext in self.STATIC_PARAMETERS:
            if value == self.STATIC_PARAMETERS[name_ext]:
                return
            raise RuntimeError(f"Cannot set static parameter \"{name}\"")
        return super().set(name_int, value, **kwargs)

    @classmethod
    def param_name_aliases(cls, name: str) -> Tuple[str, str]:
        r"""Get the internal & external aliases for a given parameter
        name.

        Args:
            name: Parameter name.

        Returns:
            tuple: (internal, external) parameter aliases.

        """
        name_ext = name
        name_int = name
        if name in cls.PARAMETER_ALIASES:
            name_ext = cls.PARAMETER_ALIASES[name_int]
        elif name in cls.PARAMETER_ALIASES_REV:
            name_int = cls.PARAMETER_ALIASES_REV[name_ext]
        return (name_int, name_ext)

    def param_fill_value(self, name: str) -> float:
        r"""Get the value indicating missing data for a parameter.

        Args:
            name: Parameter name.

        Returns:
            float: Fill value.

        """
        if ((self.PARAMETER_FILL_VALUE is None
             or isinstance(self.PARAMETER_FILL_VALUE, float))):
            return self.PARAMETER_FILL_VALUE
        name_int = self.internal_param_name(name)
        return self.PARAMETER_FILL_VALUE.get(name_int, None)

    @classmethod
    def internal_param_name(cls, name: str) -> str:
        r"""Get the aliased internal parameter name if there is one.

        Args:
            name: Parameter name.

        Returns:
            str: Aliased parameter name.

        """
        if isinstance(name, list):
            return [cls.internal_param_name(x) for x in name]
        return cls.PARAMETER_ALIASES_REV.get(name, name)

    @classmethod
    def external_param_name(cls, name: str) -> str:
        r"""Get the external parameter alias name if there is one.

        Args:
            name: Parameter name.

        Returns:
            str: Aliased parameter name.

        """
        if isinstance(name, list):
            return [cls.external_param_name(x) for x in name]
        return cls.PARAMETER_ALIASES.get(name, name)

    @classmethod
    def supports_parameters(cls, *args: str):
        r"""Check if the file supports one or more parameters."""
        for name in args:
            name_int = cls.internal_param_name(name)
            if name_int not in cls.SUPPORTED_PARAMETERS:
                return False
        return True

    def has_parameters(self, *args: str):
        r"""Check if the file has one or more parameters."""
        parameters = self.internal_parameters(
            include_header=True, include_calculated=True)
        for name in args:
            name_int = self.internal_param_name(name)
            if name_int not in parameters:
                return False
        return True

    @property
    def _internal_parameters(self) -> list:
        r"""list: Set of parameters contained by this file."""
        raise NotImplementedError  # pragma: no cover

    def internal_parameters(
            self, include_header: Optional[bool] = False,
            include_calculated: Optional[bool] = False,
    ) -> list:
        r"""list: Set of parameters contained by this file."""
        out = self._internal_parameters + list(self.STATIC_PARAMETERS.keys())
        if include_header:
            out = out + [
                k for k in self.HEADER_PARAMETERS if k not in out
            ]
        if include_calculated:
            out_external = self.external_param_name(out)
            additional = [
                k for k, v in self.PARAMETER_CALCULATORS.items()
                if bool([x for x in v if (
                        hasattr(self, x[0])
                        and all(xx in out_external or xx in
                                self.PARAMETER_CALCULATORS
                                for xx in x[1]))])
            ]
            out = out + [
                k for k in self.internal_param_name(additional)
                if k not in out
            ]
        return out

    @readonly_cached_property
    def _external_parameters(self) -> list:
        r"""list: Aliased paraemters."""
        return self.external_param_name(self.internal_parameters())

    def external_parameters(
            self, include_header: Optional[bool] = False,
            include_calculated: Optional[bool] = False,
    ) -> list:
        r"""list: Aliased paraemters."""
        if include_header or include_calculated:
            return self.external_param_name(self.internal_parameters(
                include_header=include_header,
                include_calculated=include_calculated,
            ))
        return self._external_parameters

    @readonly_cached_property
    def _internal_parameter_map(self) -> dict:
        r"""dict: Map of parameters using internal names."""
        parameters = self.internal_parameters(
            include_header=True, include_calculated=True)
        out = {}
        for k in parameters:
            try:
                out[k] = self.get(k)
            except KeyError:
                pass
        return out

    def internal_parameter_map(self, **kwargs) -> dict:
        r"""dict: Map of parameters using internal names."""
        parameters = self.internal_parameters(**kwargs)
        parameter_map = self._internal_parameter_map
        return {k: parameter_map[k] for k in parameters}

    @readonly_cached_property
    def _universal_parameter_map(self) -> dict:
        r"""dict: Map of parameters using universal names."""
        parameters = self.external_parameters(
            include_header=True, include_calculated=True)
        out = {}
        for k in self.UNIVERSAL_PARAMETERS:
            if k not in parameters:
                continue
            try:
                out[k] = self.get(k)
            except KeyError:
                pass
        return out

    def universal_parameter_map(self, **kwargs) -> dict:
        r"""dict: Map of parameters using universal names."""
        parameters = self.external_parameters(**kwargs)
        parameter_map = self._universal_parameter_map
        return {k: parameter_map[k] for k in parameters
                if k in parameter_map}

    def get_universal_parameter_map(
            self, dst: type,
            keep_units: Optional[bool] = False,
            latitude: Optional[float] = None,
            longitude: Optional[float] = None,
    ) -> dict:
        r"""Get the universal parameter map with the names & units
        expected for another file type of the same category.

        Args:
            dst: File type that the universal parameters should be
                returned for.
            keep_units: If True, don't strip units from the returned
                parameters after conversion.
            latitude: The latitude to sample.
            longitude: The longitude to sample.

        Returns:
            dict: Map of parameters using parameter names and units for
                dst file type.

        """
        parameters = self.universal_parameter_map(
            include_header=True, include_calculated=True)
        out = {
            dst.internal_param_name(k): dst.add_param_units(
                k, v, return_scalar=(not keep_units))
            for k, v in parameters.items()
            if dst.PARAMETER_ALIASES_REV.get(k, k) in
            dst.SUPPORTED_PARAMETERS
        }
        if "latitude" in dst.STATIC_PARAMETERS:
            assert isinstance(dst.STATIC_PARAMETERS["latitude"], tuple)
            if ((self.STATIC_PARAMETERS.get("latitude", None)
                 != dst.STATIC_PARAMETERS["latitude"])):
                raise NotImplementedError
        elif "latitude" in self.STATIC_PARAMETERS:
            assert isinstance(self.STATIC_PARAMETERS["latitude"], tuple)
            if latitude is None or longitude is None:
                raise NotImplementedError
            out = dst.resample_parameter_map_location(
                out, latitude=latitude, longitude=longitude)
        return out

    @classmethod
    def resample_parameter_map_location(
            cls, parameters: dict,
            latitude: Optional[float] = None,
            longitude: Optional[float] = None,
    ):
        r"""Resample parameters for a location.

        Args:
            latitude: The latitude to sample.
            longitude: The longitude to sample.

        Returns:
            dict: Resampled parameter map.

        """
        if latitude is None and longitude is None:
            return parameters
        assert latitude is not None
        assert longitude is not None
        out = {
            "latitude": latitude,
            "longitude": longitude,
        }
        for k, v in parameters.items():
            if k in out:
                continue
            if k in cls.HEADER_PARAMETERS:
                out[k] = v
            else:
                out[k] = cls.interp_location(
                    parameters[k],
                    parameters["latitude"], parameters["longitude"],
                    latitude, longitude
                )
        return out

    @readonly_cached_property
    def latitude(self) -> float:
        r"""float: Latitude (degrees)."""
        return self.get("latitude")

    @readonly_cached_property
    def longitude(self) -> float:
        r"""float: Longitude (degrees)."""
        return self.get("longitude")

    @readonly_cached_property
    def start_date(self) -> datetime.date:
        r"""datetime.date: Start of range covered by the file."""
        return self.get("start_date")

    @readonly_cached_property
    def end_date(self) -> datetime.date:
        r"""datetime.date: End of range covered by the file."""
        return self.get("end_date")

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
        if isinstance(x, (rj_units.Quantity, rj_units.QuantityArray)):
            return cls._f2str(x.value)
        elif isinstance(x, datetime.datetime):
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
        if "start_date" in cls.PARAMETER_LIMITS:
            if not cls._check_inside_range(
                    (start_date, end_date),
                    cls.PARAMETER_LIMITS["start_date"]):
                raise ValueError(
                    f"The requested date range {(start_date, end_date)} "
                    f"is not withing the date limits for {cls.DESC} "
                    f"data files {cls.DATE_LIMITS}"
                )
        if "latitude" in cls.PARAMETER_LIMITS:
            if not cls._check_inside_range(
                    latitude, cls.PARAMETER_LIMITS["latitude"]):
                raise ValueError(
                    f"The requested latitude {latitude} is not within "
                    f"the limits for {cls.DESC} data files "
                    f"{cls.PARAMETER_LIMITS['latitude']}"
                )
            if not cls._check_inside_range(
                    longitude, cls.PARAMETER_LIMITS["longitude"]):
                raise ValueError(
                    f"The requested longitude {longitude} is not within "
                    f"the limits for {cls.DESC} data files "
                    f"{cls.PARAMETER_LIMITS['longitude']}"
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
            **kwargs: Any
    ) -> "BaseDataFile":
        r"""Create a data file from another file of the same category.

        Args:
            fbase: External file to create an version of this file from.
            fname: File name where the data should be written to.
            cache_dir: Directory where the data should be cached. Only
                used if fname is not provided.
            \*\*kwargs: Additional keyword arguments are passed to
                calculate_missing.

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
            assert not kwargs
            return cls(fname)
        if fbase.NAME == cls.NAME:
            contents = fbase.contents
            # contents = copy.deepcopy(fbase.contents)
        else:
            # TODO: Get this based on required universal parameters
            if fbase.NAME in cls.REQUIRED_EXTERNAL_PARAMETERS:
                fbase.add_missing_param(
                    cls.REQUIRED_EXTERNAL_PARAMETERS[fbase.NAME])
            if hasattr(cls, f"_from_{fbase.NAME}"):
                contents = getattr(cls, f"_from_{fbase.NAME}")(fbase)
            elif hasattr(fbase, f"_to_{cls.NAME}"):
                contents = getattr(fbase, f"_to_{cls.NAME}")()
            else:
                contents = cls._from_compatible(fbase)
        if isinstance(contents, cls):
            contents = contents.contents
        out = cls(fname, contents=contents)
        out.calculate_missing(**kwargs)
        out.write()
        return out

    @classmethod
    def empty_contents(cls):
        r"""Create empty contents for a file wrapper instance."""
        raise NotImplementedError

    @classmethod
    def _from_compatible(cls, src: "BaseDataFile", **kwargs: Any):
        r"""Convert data from another file format into the correct
        format for this file.

        Args:
            src: Base class data.
            \*\*kwargs: Additional keyword arguments are passed to
                get_universal_parameter_map.

        Returns:
            Converted data.

        """
        if isinstance(src, cls):
            return src.contents
        parameters = src.get_universal_parameter_map(cls, **kwargs)
        # print(f"{src.NAME} -> {cls.NAME}")
        # import pprint; pprint.pprint(parameters)
        # import pdb; pdb.set_trace()
        out = cls(None, contents=cls.empty_contents())
        for k, v in parameters.items():
            out.set(k, v, allow_add=True)
        # raise NotImplementedError(
        #     f"Conversion from {type(src)} to {cls}")  # pragma: no cover
        return out

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
        r"""Fill in any missing parameters by downloading them.

        Args:
            parameters: Set of parameters that must be present.

        """
        parameters = self.internal_param_name(parameters)
        missing = [
            k for k in parameters if k not in self.internal_parameters()
        ]
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
    def parameters_for_download(
            cls, parameters: Optional[List[str]] = None) -> list:
        r"""Parse a list of parameters for downloading, adding missing
        required parameters and setting the default if None is passed.

        Args:
            parameters: List of parameters to parse.

        Returns:
            list: Parsed parameters.

        """
        if parameters is None:
            parameters = cls.DEFAULT_DOWNLOAD_PARAMETERS.copy()
        else:
            parameters = cls.internal_param_name(parameters)
        parameters = [
            x for x in cls.REQUIRED_DOWNLOAD_PARAMETERS
            if x not in parameters
        ] + parameters
        for k in ["latitude", "longitude", "start_time", "end_time",
                  "depths"] + cls.HEADER_PARAMETERS:
            assert k not in parameters
        return parameters

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
    UNIVERSAL_PARAMETERS: ClassVar[List[str]] = [
        "start_date",
        "end_date",
        "latitude",
        "longitude",
        "elevation",
        "radiation_atmosphere",
        "radiation_surface",
        "vapor_pressure",
        "precipitation",
        "temperature",
        "temperature_avg",
        "temperature_amp",
        "temperature_min",
        "temperature_max",
        "temperature_dew",
        "wind_speed",
    ]
    IS_BASE_CLASS: ClassVar[bool] = True

    @property
    def dates(self) -> np.ndarray:
        r"""np.ndarray: Dates covered by this file."""
        raise NotImplementedError  # pragma: no cover

    @BaseDataFile.parameter_calculation("vapor_pressure", "temperature_dew")
    def vaporpressure_from_dewpoint(self, Tdew):
        r"""Calculate the vapor pressure from the dew point. Based on the
        calculation in PCSE from

        Allen, R.G., Pereira, L.S., Raes, D. and Smith, M. (1998) Crop
            evapotranspiration. Guidelines for computing crop water
            requirements, FAO irrigation and drainage paper 56)

        """
        a = rj_units.Quantity(6.108, "hPa")
        b = rj_units.Quantity(237.3, "degC")
        return a * utils.exp((17.27 * Tdew) / (Tdew + b))

    @BaseDataFile.parameter_calculation("temperature_dew", "vapor_pressure")
    def dewpoint_from_vaporpressure(self, vp):
        r"""Inverse of vaporpressure_from_dewpoint."""
        a = rj_units.Quantity(6.108, "hPa")
        b = rj_units.Quantity(237.3, "degC")
        return (-b * utils.log(vp / a)
                / (utils.log(vp / a) - 17.27))

    @BaseDataFile.parameter_calculation("temperature_avg", "temperature")
    def average_temperature(self, temp):
        return utils.mean(temp)


class NASAPOWERWeatherFile(BaseWeatherFile):
    r"""Wrapper for loading NASA POWER data."""

    NAME: ClassVar[str] = "nasa_power_weather_data"
    DESC: ClassVar[str] = "NASA POWER"
    URL: ClassVar[str] = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
    )
    # Daily meterology data from 1981-7-1
    # Daily solar data from 1984-7-1
    PARAMETER_LIMITS: ClassVar[dict] = {
        "start_date": (
            datetime.date(1981, 7, 1), datetime.date.today()),
    }
    # TODO move DEFAULT_DATE_RANGE in DEFAULT_PARAMETERS
    DEFAULT_DATE_RANGE: ClassVar[Tuple[datetime.date, datetime.date]] = (
        datetime.date(1984, 7, 1), datetime.date(2026, 5, 8))
    REQUIRED_PARAMETERS: ClassVar[List[str]] = [
        "latitude", "longitude", "elevation",
        "start_date", "end_date",
    ]
    HEADER_PARAMETERS: ClassVar[List[str]] = [
        "latitude", "longitude", "elevation",
        "start_date", "end_date"
    ]
    PARAMETER_ALIASES: ClassVar[Mapping[str, str]] = {
        "TOA_SW_DWN": "radiation_atmosphere",
        "ALLSKY_SFC_SW_DWN": "radiation_surface",
        "T2M": "temperature",
        "T2M_MIN": "temperature_min",
        "T2M_MAX": "temperature_max",
        "T2MDEW": "temperature_dew",
        "WS2M": "wind_speed",
        "PRECTOTCORR": "precipitation",
    }
    PARAMETER_UNITS: ClassVar[Mapping[str, str]] = {
        # Units are for the Agroclimatology (AG) community (daily)
        "latitude": "degrees",
        "longitude": "degrees",
        "elevation": "m",
        "TOA_SW_DWN": "MJ/m^2/day",
        "ALLSKY_SFC_SW_DWN": "MJ/m^2/day",
        "T2M": "degC",
        "T2M_MIN": "degC",
        "T2M_MAX": "degC",
        "T2MDEW": "degC",
        "WS2M": "m/s",
        "PRECTOTCORR": "mm",
    }
    _coord_params: ClassVar[List[str]] = [
        "longitude", "latitude", "elevation",
    ]

    @classmethod
    def empty_contents(cls):
        r"""Create empty contents for a file wrapper instance."""
        return {
            "header": {
                "fill_value": "SOMETHING",
            },
            "geometry": {
                "coordinates": [],
            },
            "properties": {
                "parameter": {},
            },
        }

    def param_fill_value(self, name: str) -> float:
        r"""Get the value indicating missing data for a parameter.

        Args:
            name: Parameter name.

        Returns:
            float: Fill value.

        """
        return float(self.contents["header"]["fill_value"])

    @readonly_cached_property
    def _internal_parameters(self) -> list:
        r"""list: Set of power parameters contained by this file."""
        return list(self.contents["properties"]["parameter"].keys())

    def _get(self, name: str):
        r"""Get a parameter from the file.

        Args:
            name: Parameter name.

        Returns:
            Parameter value.

        Raises:
            KeyError: If name is not a valid parameter name.

        """
        if name in self._coord_params:
            return float(self.contents["geometry"]["coordinates"][
                self._coord_params.index(name)
            ])
        elif name in ["start_date", "end_date"]:
            return datetime.datetime.strptime(
                self.contents["header"][name.split("_")[0]],
                "%Y%m%d").date()
        out = self.contents["properties"]["parameter"][name]
        return np.array([out[k] for k in sorted(out.keys())])

    def _set(self, name: str, value: Any,
             allow_add: Optional[bool] = False) -> Any:
        r"""Set a parameter in the file.

        Args:
            name: Parameter name.
            value: Parameter value.
            allow_add: Allow for a new parameter value to be added.

        Raises:
            KeyError: If name is not a valid parameter name.

        """
        if name in self._coord_params:
            assert isinstance(value, float)
            idx = self._coord_params.index(name)
            self.contents["geometry"]["coordinates"][idx] = str(value)
            return
        elif name in ["start_date", "end_date"]:
            assert isinstance(value, datetime.date)
            self.contents["header"][name.split("_")[0]] = value.strftime(
                "%Y%m%d")
            return
        assert len(value) == len(self.dates)
        if (((not allow_add)
             and name not in self.contents["properties"]["parameter"])):
            raise KeyError(name)
        self.contents["properties"]["parameter"][name] = {
            date.strftime("%Y%m%d"): v
            for date, v in zip(self.dates, value)
        }

    @readonly_cached_property
    def dates(self) -> np.ndarray:
        r"""np.ndarray: Dates covered by this file."""
        import pandas as pd
        dates = pd.Series(
            self.contents["properties"]["parameter"][
                self.internal_parameters()[0]])
        return pd.to_datetime(dates.index, format="%Y%m%d")

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
        parameters = cls.parameters_for_download(parameters)
        # Build request for retrieving data, using new NASA POWER api
        # Based on PCSE _query_NASAPower_server
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

    @classmethod
    def _from_compatible(cls, src: BaseWeatherFile):
        r"""Convert weather data from another file format into the
        correct format for this file.

        Args:
            src: Universal weather parameters.

        Returns:
            Converted data.

        """
        parameters = src.get_universal_parameter_map(cls)
        out = {
            "header": {
                "start": src.start_date.strftime("%Y%m%d"),
                "end": src.end_date.strftime("%Y%m%d"),
            },
            "geometry": {
                "coordinates": [
                    parameters[k] for k in cls._coord_params
                ]
            },
            "properties": {
                "parameter": {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in parameters.items()
                    if k not in cls.HEADER_PARAMETERS
                },
            },
        }
        return out


class BaseSoilFile(BaseDataFile):
    r"""Base class for soil files."""

    CATEGORY: ClassVar[str] = "soil"
    NAME: ClassVar[str] = None
    # Set date range to widest possible since most soil data is not
    #   temporal
    STATIC_PARAMETERS: ClassVar[dict] = {
        "start_date": datetime.date(1, 1, 1),
        "end_date": datetime.date.today(),
    }
    UNIVERSAL_PARAMETERS: ClassVar[List[str]] = [
        "latitude",
        "longitude",
        "top_depth",
        "bottom_depth",
        "thickness",
        "pH",
        "clay",
        "silt",
        "sand",
        "rocks",
        "bulk_density",     # Bulk density
        "organic_carbon",   # Organic carbon content
        "nitrogen",         # Nitrogen content
        "no3",
        "nh4",
        "urea",
        "ksat",             # Saturated hydraulic conductivity
        "dul",              # Drained upper limit
        # Wilting point or LL15 -> volumetric water content of soil
        #     at 15 bar (1500 kPa) suction
        "wilting_point",    # Wilting point
        "sat",              # Saturated water content
        "cec",              # Cation exchange capacity
    ]
    # UNIVERSAL_LAYER_PARAMETERS: ClassVar[List[str]] = []
    UNIVERSAL_LAYER_DEPTH_PARAMETERS: ClassVar[List[str]] = [
        "top_depth", "bottom_depth", "thickness", "depths"
    ]
    LAYER_DEPTH_PARAMETERS: ClassVar[List[str]] = []
    LAYER_PARAMETERS: ClassVar[List[str]] = []
    IS_BASE_CLASS: ClassVar[bool] = True

    @staticmethod
    def _on_registration(cls):
        BaseDataFile._on_registration(cls)
        if cls.NAME is not None:
            cls.LAYER_DEPTH_PARAMETERS = []
            for k in cls.UNIVERSAL_LAYER_DEPTH_PARAMETERS:
                kint = cls.PARAMETER_ALIASES_REV.get(k, k)
                if kint in cls.SUPPORTED_PARAMETERS:
                    cls.LAYER_DEPTH_PARAMETERS.append(kint)
            cls.LAYER_PARAMETERS = [
                k for k in cls.SUPPORTED_PARAMETERS
                if k not in (cls.HEADER_PARAMETERS
                             + cls.LAYER_DEPTH_PARAMETERS)
            ]

    def set(self, name: str, value: Any,
            allow_resize: Optional[bool] = False,
            **kwargs: Any) -> None:
        r"""Set a parameter in the file.

        Args:
            name: Parameter name.
            value: Parameter value.
            allow_resize: If True, allow the parameter to be resized.
            \*\*kwargs: Additional keyword arguments are passed to _set.

        Raises:
            KeyError: If name is not a valid parameter name.
            ValueError: If name is setting a layer parameter and does
                not have a size matching the expected number of layers.

        """
        if allow_resize or self.nlayers == 0:
            name_int = self.internal_param_name(name)
            if name_int in self.LAYER_DEPTH_PARAMETERS:
                nlayers = self.get_param_nlayers(value)
                if nlayers != self.nlayers:
                    self.resize_layer_parameters(nlayers)
        else:
            name_int = self.internal_param_name(name)
            if name_int in self.LAYER_PARAMETERS:
                nlayers = self.get_param_nlayers(value)
                if nlayers != self.nlayers:
                    raise ValueError(
                        f"New value for \"{name}\" contains data for "
                        f"{nlayers} layers, but the data set currently "
                        f"includes {self.nlayers} layers."
                    )
        return super().set(name, value, **kwargs)

    @classmethod
    def get_param_nlayers(cls, value: Any) -> int:
        r"""Calculate the number of layers described by layer parameter
        data.

        Args:
            value: Layer parameter data.

        Returns:
            int: Number of layers that value describes.

        """
        return len(value)

    @readonly_cached_property
    def nlayers(self) -> int:
        r"""int: Number of soil layers covered by this file."""
        return len(self.depths)

    @readonly_cached_property
    def depths(self) -> list:
        r"""list: List of (start, end) depth pairs covered by this
        file."""
        return self.get("depths")

    # TODO: Scale factors

    # @BaseDataFile.parameter_calculation("no3", "nitrogen")
    # def no3_from_total_nitrogen(cls, nitrogen):
    #     return nitrogen

    # @BaseDataFile.parameter_calculation("nh4", "nitrogen")
    # def nh4_from_total_nitrogen(cls, nitrogen):
    #     return 0.1 * nitrogen

    # @BaseDataFile.parameter_calculation("urea", "nitrogen")
    # def urea_from_total_nitrogen(cls, nitrogen):
    #     return 0.0 * nitrogen

    @BaseDataFile.parameter_calculation("depths", "top_depth",
                                        "bottom_depth")
    @classmethod
    def depth_from_split(cls, top_depth, bottom_depth):
        if isinstance(top_depth, (float, rj_units.Quantity)):
            top_depth = [top_depth]
        if isinstance(bottom_depth, (float, rj_units.Quantity)):
            bottom_depth = [bottom_depth]
        assert len(top_depth) == len(bottom_depth)
        return [
            (start, stop) for start, stop in zip(top_depth, bottom_depth)
        ]

    @BaseDataFile.parameter_calculation("depths", "thickness")
    @classmethod
    def depth_from_thickness(cls, thickness):
        prev_units = cls.get_units(thickness)
        out, top = [], cls.add_units(0.0, prev_units)
        # TODO: Can be reverted to basic iteration once iteration
        #   fixed in yggdrasil_python_rapidjson
        # for x in thickness:
        for i in range(len(thickness)):
            x = thickness[i]
            bottom = top + x
            out.append((top, bottom))
            top = bottom
        return out

    @BaseDataFile.parameter_calculation("thickness", "top_depth",
                                        "bottom_depth")
    @classmethod
    def thickness_from_split(cls, top_depth, bottom_depth):
        return bottom_depth - top_depth

    @BaseDataFile.parameter_calculation("thickness", "depths")
    @classmethod
    def thickness_from_depths(cls, depths):
        prev_units = cls.get_units(depths)
        if prev_units is not None:
            return cls.add_units(
                cls.thickness_from_depths(cls.remove_units(depths)),
                prev_units)
        if len(depths) == 0:
            return np.array([], dtype=np.float64)
        return np.array([
            stop - start for start, stop in depths
        ])

    @BaseDataFile.parameter_calculation("top_depth", "depths")
    @classmethod
    def top_depth_from_depths(cls, depths):
        if len(depths) == 0:
            return np.array([], dtype=np.float64)
        prev_units = cls.get_units(depths)
        out = [x[0] for x in depths]
        if prev_units is not None:
            out = cls.add_units(out, prev_units)
        return out

    @BaseDataFile.parameter_calculation("bottom_depth", "depths")
    @classmethod
    def bottom_depth_from_depths(cls, depths):
        if len(depths) == 0:
            return np.array([], dtype=np.float64)
        prev_units = cls.get_units(depths)
        out = [x[1] for x in depths]
        if prev_units is not None:
            out = cls.add_units(out, prev_units)
        return out

    @BaseDataFile.parameter_calculation("top_depth", "thickness")
    @classmethod
    def top_depth_from_thickness(cls, thickness):
        prev_units = cls.get_units(thickness)
        if prev_units is not None:
            return rj_units.QuantityArray(
                cls.top_depth_from_thickness(
                    cls.remove_units(thickness)),
                prev_units)
        if len(thickness) == 0:
            return np.array([], dtype=np.float64)
        return np.array([0.0] + np.cumsum(thickness[:-1]).tolist())

    @BaseDataFile.parameter_calculation("bottom_depth", "thickness")
    @classmethod
    def bottom_depth_from_thickness(cls, thickness):
        prev_units = cls.get_units(thickness)
        if prev_units is not None:
            return rj_units.QuantityArray(
                cls.bottom_depth_from_thickness(
                    cls.remove_units(thickness)),
                prev_units)
        return np.cumsum(thickness)

    @readonly_cached_property
    def thickness(self) -> list:
        r"""list: Thickness of each soil layer covered by this file."""
        return self.get("thickness")

    @BaseDataFile.parameter_calculation("silt", "clay", "sand")
    @classmethod
    def silt_from_clay_and_sand(cls, clay: float, sand: float) -> float:
        # TODO: Clay & sand derived?
        total = rj_units.Quantity(100, "%")
        return total - (clay + sand)

    @BaseDataFile.parameter_calculation("dul", "clay", "sand",
                                        "organic_carbon")
    @classmethod
    def dul_from_texture(cls, clay: float, sand: float,
                         om: float) -> float:
        r"""Estimate drained upper limit (field capacity) from soil
        texture and organic matter using the Saxton and Rawls (2006)
        pedotransfer function Eq. 2 (Theta_33).

        Args:
            clay: Clay content.
            sand: Sand content.
            om: Organic matter content.

        Returns:
            float: Drained upper limit (m3/m3).

        """
        ans0 = (-0.251 * sand + 0.195 * clay + 0.011 * om
                + 0.006 * sand * om - 0.027 * clay * om
                + 0.452 * sand * clay + 0.299)
        return ans0 + (1.283 * ans0**2 - 0.374 * ans0 - 0.015)

    @BaseDataFile.parameter_calculation("wilting_point",
                                        "clay", "sand",
                                        "organic_carbon")
    @classmethod
    def wilting_point_from_texture(cls, clay: float, sand: float,
                                   om: float) -> float:
        r"""Estimate lower limit (wilting point) from soil texture and
        organic matter using the Saxton and Rawls (2006) pedotransfer
        function Eq. 1 (Theta_1500).

        Args:
            clay: Clay content.
            sand: Sand content.
            om: Organic matter content.

        Returns:
            float: Lower limit (m3/m3).

        """
        ans0 = (-0.024 * sand + 0.487 * clay + 0.006 * om
                + 0.005 * sand * om - 0.013 * clay * om
                + 0.068 * sand * clay + 0.031)
        return ans0 + (0.14 * ans0 - 0.02)

    @classmethod
    def dul_s_from_texture(cls, clay: float, sand: float,
                           om: float) -> float:
        r"""Estimate saturated water content from soil texture and
        organic matter using the Saxton and Rawls (2006) pedotransfer
        function Eq. 3 (Theta_{S-33}).

        Args:
            clay: Clay content.
            sand: Sand content.
            om: Organic matter content.

        Returns:
            float: Saturated water content (m3/m3).

        """
        ans0 = (0.278 * sand + 0.034 * clay + 0.022 * om
                - 0.018 * sand * om - 0.027 * clay * om
                - 0.584 * sand * clay + 0.078)
        return ans0 + (0.636 * ans0 - 0.107)

    @BaseDataFile.parameter_calculation("sat", "clay", "sand",
                                        "organic_carbon")
    @classmethod
    def sat_from_texture(cls, clay: float, sand: float,
                         om: float) -> float:
        r"""Estimate saturated water content from the drained upper
        limit using the Saxton and Rawls (2006) pedotransfer function
        Eq. 5 (Theta_S).

        Args:
            clay: Clay content.
            sand: Sand content.
            om: Organic matter content.

        Returns:
            float: Saturated water content (m3/m3).

        """
        dul = cls.dul_from_texture(clay, sand, om)
        dul_s = cls.dul_s_from_texture(clay, sand, om)
        return dul + dul_s - 0.097 * sand + 0.043

    @BaseDataFile.parameter_calculation("ksat", "clay", "sand",
                                        "organic_carbon")
    @classmethod
    def ksat_from_texture(cls, clay: float, sand: float,
                          om: float) -> float:
        r"""Estimate saturated hydraulic conductivity from soil texture
        and organic matter using the Saxton and Rawls (2006)
        pedotransfer function (Eq. 16 K_s).

        Args:
            clay: Clay content.
            sand: Sand content.
            om: Organic matter content.

        Returns:
            float: Saturated hydraulic conductivity (mm/day).

        """
        dul = cls.dul_from_texture(clay, sand, om)
        ll15 = cls.wilting_point_from_texture(clay, sand, om)
        sat = cls.sat_from_texture(clay, sand, om)
        a = rj_units.Quantity(1930.0, "mm/hr")
        b = ((utils.log(1500) - utils.log(33))
             / (utils.log(dul) - utils.log(ll15)))
        return a * utils.pow(sat - dul, 3 - (1.0 / b))

    @classmethod
    def texture_class(cls, clay_pct: float, sand_pct: float) -> str:
        r"""Get the soil texture class from the USDA clay and sand
        fractions converted to the international system (Minasny et
        al., 2001).

        Args:
            clay_pct: Clay content (%).
            sand_pct: Sand content (%).

        Returns:
            str: Texture class name.

        """
        clay = clay_pct / 100
        silt = (100 - clay_pct - sand_pct) / 100
        intl_clay = clay
        intl_silt = max(
            0.0,
            -0.0041 - 0.127 * clay + 0.553 * silt
            + 0.17 * clay**2 - 0.19 * silt**2 + 0.59 * clay * silt)
        intl_sand = 1 - intl_clay - intl_silt
        if intl_sand < 0.75 - intl_clay and intl_clay >= 0.40:
            return "silty clay"
        if intl_sand < 0.75 - intl_clay and intl_clay >= 0.26:
            return "silty clay loam"
        if intl_sand < 0.75 - intl_clay:
            return "silty loam"
        if (intl_clay >= 0.40 + (0.305 - 0.40) / (0.635 - 0.35)
                * (intl_sand - 0.35)
                and intl_clay < 0.50 + (0.305 - 0.50) / (0.635 - 0.50)
                * (intl_sand - 0.50)):
            return "clay"
        if (intl_clay >= 0.26 + (0.305 - 0.26) / (0.635 - 0.74)
                * (intl_sand - 0.74)):
            return "sandy clay"
        if (intl_clay >= 0.26 + (0.17 - 0.26) / (0.83 - 0.49)
                * (intl_sand - 0.49)
                and intl_clay < 0.10 + (0.305 - 0.10) / (0.635 - 0.775)
                * (intl_sand - 0.775)):
            return "clay loam"
        if (intl_clay >= 0.26 + (0.17 - 0.26) / (0.83 - 0.49)
                * (intl_sand - 0.49)):
            return "sandy clay loam"
        if (intl_clay >= 0.10 + (0.12 - 0.10) / (0.63 - 0.775)
                * (intl_sand - 0.775)
                and intl_clay < 0.10 + (0.305 - 0.10) / (0.635 - 0.775)
                * (intl_sand - 0.775)):
            return "loam"
        if (intl_clay >= 0.10 + (0.12 - 0.10) / (0.63 - 0.775)
                * (intl_sand - 0.775)):
            return "sandy loam"
        if (intl_clay < 0.0 + (0.08 - 0.0) / (0.88 - 0.93)
                * (intl_sand - 0.93)):
            return "loamy sand"
        return "sand"

    @classmethod
    def interp_location(cls, y: Any, latitude_old, longitude_old,
                        latitude_new, longitude_new,
                        kind: Optional[str] = "linear"):
        r"""Interpolate a map to a specific location.

        Args:
            y: Initial dependent map values.
            latitude_old: Initial latitude values or range.
            longitude_old: Initial longitude values or range.
            latitude_new: New latitude value.
            longitude_new: New longitude value.

        Returns:
            np.ndarray, float: Interpolated values.

        """
        if isinstance(latitude_old, tuple):
            assert isinstance(longitude_old, tuple)
            assert y.ndim == 2
            # TODO: Verify order of dimensions
            latitude_old = np.linspace(
                latitude_old[0], latitude_old[1], y.shape[0])
            longitude_old = np.linspace(
                longitude_old[0], longitude_old[1], y.shape[1])
        if y.ndim == 2:
            if latitude_old.ndim == 2:
                assert y.shape == latitude_old.shape
                assert y.shape == longitude_old.shape
            else:
                assert y.shape[0] == len(latitude_old)
                assert y.shape[1] == len(longitude_old)
        if isinstance(latitude_new, (list, np.ndarray)):
            assert len(latitude_new) == len(longitude_new)
            raise NotImplementedError
        dist = (
            (latitude_old - latitude_new) ** 2
            + (longitude_old - longitude_new) ** 2
        )
        if dist.ndim == 2:
            idx = np.unravel_index(np.argmin(dist), dist.shape)
        else:
            idx = np.argmin(dist)
        return y[*idx]

    @classmethod
    def interp_profile(cls, y: Any, xold, xnew,
                       kind: Optional[str] = "linear") -> np.ndarray:
        r"""Interpolate a profile to a new set of independent values.

        Args:
            y: Initial dependent values.
            xold: Initial independent values.
            xnew: New independent values.
            kind: Type of interpolation to use.

        Returns:
            np.ndarray: Interpolated profile.

        """
        from scipy import interpolate
        # xranges = xold.max() - xold.min()
        yinterp = interpolate.interp1d(
            xold, y, kind=kind, assume_sorted=False,
            fill_value='extrapolate')
        return yinterp(xnew)

    @classmethod
    def create_profile(cls, layers: int | np.ndarray,
                       a: Optional[float] = 0.5,
                       b: Optional[float] = 0.5) -> np.ndarray:
        r"""Create a variable profile that following an exponential
        dependency on layer depth via a * x * e^(-b * x) if a > 0 and
        .

        This was adapted from apsimNGpy.

        Args:
            layers: Array of layer depths or number of even layers.
            a: Scale factor for the profile.
            b: Exponential scale factor for the profile.

        Returns:
            np.ndarray: Array of profile values at each layer.

        """
        if isinstance(layers, list):
            layers = np.array(layers)
        if isinstance(layers, int):
            depthn = np.arange(1, layers + 1, 1)
        elif isinstance(layers, np.ndarray):
            depthn = 1 + (
                len(layers)
                * (layers - layers.min())
                / (layers.max() - layers.min())
            )
        if a < 0:
            raise RuntimeError(f"a cannot be negative (a = {a})")
        elif (a > 0 and b != 0):
            ep = -b * depthn
            result = (a * depthn) * np.exp(ep)
            return result / result.max()
        elif (a == 0 and b != 0):
            ep = -b * depthn
            result = np.exp(ep) / np.exp(-b)
            return result
        elif (a == 0 or b == 0):
            out = depthn.copy()
            out[1:] -= out[:-1]
            return out
        raise RuntimeError(f"Invalid parameters a = {a}, b = {b}")

    def calculate_missing(self, depths: Optional[list] = None,
                          thickness: Optional[list] = None,
                          **kwargs: Any):
        r"""Calculate missing parameters that are required.

        Args:
            depths: List of top & bottom depths defining each layer in
                the soil profile (in cm).
            thickness: The thickness of each layer (in cm). Only used if
                depths is not provided.
            \*\*kwargs: Additional keyword arguments are passed to the
                parent class's method.

        """
        super().calculate_missing(**kwargs)
        if depths is not None or thickness is not None:
            self.resample_depths(depths=depths, thickness=thickness)

    def resize_layer_parameters(self, nlayers: int) -> None:
        r"""Resize the layer parameters to match a new number of layers.

        Args:
            nlayers: Updated number of layers.

        """
        pass

    def get_universal_parameter_map(self, dst: type, **kwargs) -> dict:
        r"""Get the universal parameter map with the names & units
        expected for another file type of the same category.

        Args:
            dst: File type that the universal parameters should be
                returned for.
            \*\*kwargs: Additional keyword arguments are passed to the
                parent class.

        Returns:
            dict: Map of parameters using parameter names and units for
                dst file type.

        """
        depth_param = {
            k: dst.STATIC_PARAMETERS[k]
            for k in dst.UNIVERSAL_LAYER_DEPTH_PARAMETERS
            if k in dst.STATIC_PARAMETERS
        }
        out_unsorted = super().get_universal_parameter_map(dst, **kwargs)
        out = {
            k: out_unsorted.pop(k) for k in
            dst.LAYER_DEPTH_PARAMETERS
            if k in out_unsorted
        }
        out.update(**out_unsorted)
        if not depth_param:
            return out
        return dst.resample_parameter_map_layers(out, **depth_param)

    @classmethod
    def resample_parameter_map_layers(
            cls, parameters: dict,
            top_depth: Optional[list] = None,
            bottom_depth: Optional[list] = None,
            depths: Optional[list] = None,
            thickness: Optional[list] = None,
    ) -> dict:
        r"""Resample layer parameters in a parameter map for a new set
        of depths.

        Args:
            parameters: Parameter map that should be resampled. Only
                layer parameters will be resampled.
            top_depth: Top depths of each layer.
            bottom_depth: Bottom depths of each layer.
            depths: List of top & bottom depths defining each layer in
                the soil profile.
            thickness: The thickness of each layer. Only used if
                depths is not provided.

        Returns:
            dict: Resampled parameter map.

        """
        # TODO: Check if there is only a single layer present and then
        #   use create_profile?
        # TODO: Units
        # TODO: Raise error if STATIC & params don't match
        depth_param = {
            cls.external_param_name(k): parameters[k]
            for k in cls.LAYER_DEPTH_PARAMETERS
        }
        if depths is None and thickness is None and bottom_depth is None:
            raise ValueError("Either bottom_depth, depths, or thickness "
                             "must be provided")
        if bottom_depth is not None:
            assert top_depth is not None
        else:
            if depths is None:
                bottom_depth = cls.bottom_depth_from_thickness(thickness)
                top_depth = cls.top_depth_from_thickness(thickness)
            else:
                bottom_depth = cls.bottom_depth_from_depths(depths)
                top_depth = cls.top_depth_from_depths(depths)
        out = {}
        if "bottom_depth" in depth_param:
            old_depths = depth_param["bottom_depth"]
            out.update(
                bottom_depth=bottom_depth,
                top_depth=top_depth,
            )
        elif "thickness" in depth_param:
            old_depths = [
                x[1] for x in
                cls.depth_from_thickness(depth_param["thickness"])
            ]
            out.update(
                thickness=cls.thickness_from_split(top_depth, bottom_depth),
            )
        elif "depths" in depth_param:
            old_depths = [x[1] for x in depth_param["depths"]]
            out.update(
                depths=cls.depth_from_split(top_depth, bottom_depth),
            )
        else:
            raise ValueError(
                f"No depth information available: {depth_param}")
        out = {k: cls.add_param_units(k, v) for k, v in out.items()}
        for k, v in parameters.items():
            if k in cls.LAYER_DEPTH_PARAMETERS:
                assert k in out
                continue
            if k in cls.HEADER_PARAMETERS:
                out[k] = v
            else:
                out[k] = cls.interp_profile(
                    parameters[k], old_depths, bottom_depth
                )
        return out

    def resample_depths(self, **kwargs):
        r"""Resample the soil profile for the specified depths/thicknesses.

        Args:
            \*\*kwargs: Keyword arguments are passed to
                resample_parameter_map_layers.

        """
        old_parameters = self.internal_parameter_map(include_header=True)
        new_parameters = self.resample_parameter_map_layers(
            old_parameters, **kwargs)
        # Set depths/thicknesses first
        for k in self.LAYER_DEPTH_PARAMETERS:
            assert k in new_parameters
            self.set(k, new_parameters[k], allow_resize=True)
        # Set remainder parameters, interpolating for new bottom depths
        for k, v in new_parameters.items():
            if k in self.HEADER_PARAMETERS + self.LAYER_DEPTH_PARAMETERS:
                continue
            self.set(k, v)


class BaseSoilMapFile(BaseSoilFile):
    r"""Generic wrapper for soil files that are based on maps."""

    NAME: ClassVar[str] = "generic_soil_map"
    DESC: ClassVar[str] = "Generic Soil Map"
    DONT_TEST: ClassVar[bool] = True
    STATIC_PARAMETERS: ClassVar[dict] = {
        "latitude": (-90.0, 90.0),
        "longitude": (-180.0, 180.0),
    }
    HEADER_PARAMETERS: ClassVar[List[str]] = [
        "latitude", "longitude",
    ]
    IS_BASE_CLASS: ClassVar[bool] = True

    # TODO: Method for sampling a location genericly
    @classmethod
    def get_param_nlayers(cls, value: Any) -> int:
        r"""Calculate the number of layers described by layer parameter
        data.

        Args:
            value: Layer parameter data.

        Returns:
            int: Number of layers that value describes.

        """
        assert isinstance(value, np.ndarray)
        assert value.ndim >= 2
        if value.ndim == 2:
            return 1
        return value.shape[-1]

    def get_universal_parameter_map(self, dst: type, **kwargs) -> dict:
        r"""Get the universal parameter map with the names & units
        expected for another file type of the same category.

        Args:
            dst: File type that the universal parameters should be
                returned for.
            \*\*kwargs: Additional keyword arguments are passed to the
                parent class.

        Returns:
            dict: Map of parameters using parameter names and units for
                dst file type.

        """
        out = super().get_universal_parameter_map(dst, **kwargs)
        return out


class ISRICSoilGridsFile(BaseSoilFile):
    r"""Wrapper for loading ISRIC SoilGrids data from REST API."""

    NAME: ClassVar[str] = "isric_soilgrids_data"
    DESC: ClassVar[str] = "ISRIC SoilGrids"
    URL: ClassVar[str] = (
        "https://rest.isric.org/soilgrids/v2.0/properties/query"
    )
    HEADER_PARAMETERS: ClassVar[List[str]] = [
        "longitude", "latitude",
        "top_depth", "bottom_depth",
    ]
    PARAMETER_ALIASES: ClassVar[Mapping[str, str]] = {
        "bdod": "bulk_density",
        "cfvo": "rocks",
        "soc": "organic_carbon",
        "phh2o": "pH",
        "wv0010": "sat",
        "wv0033": "dul",
        "wv1500": "wilting_point",
    }
    PARAMETER_UNITS: ClassVar[Mapping[str, str]] = {
        "top_depth": "cm",
        "bottom_depth": "cm",
        "bdod": "cg/cm^3",    # Bulk density of the fine earth fraction
        "cec": "mmol/kg",     # Cation exchange capacity, CEC buffered at pH7
        "cfvo": "cm^3/dm^3",  # Coarse fragment content
        "clay": "g/kg",
        "nitrogen": "cg/kg",
        # "ocd": "hg/m3",       # Organic carbon density
        # "ocs": "t/ha",        # Organic carbon stocks
        "soc": "dg/kg",       # Soil organic carbon
        "phh2o": "10",        # 10 * pH
        "sand": "g/kg",
        "silt": "g/kg",
        "wv0010": "mm/m",     # Volumetric water content at 10 kPa
        "wv0033": "mm/m",     # Volumetric water content at 33 kPa
        "wv1500": "mm/m",     # Volumetric water content at 1500 kPa
    }
    SOIL_GRIDS_DEPTHS: ClassVar[dict] = {
        f"{start}-{end}cm": (start, end) for start, end in
        [(0, 5), (5, 15), (15, 30), (30, 60), (60, 100), (100, 200)]
    }
    # TODO: Tests for this class skipped by default until the API is
    #   stable
    PYTEST_MARKS: ClassVar[List[str]] = ["slow"]
    _coord_params: ClassVar[List[str]] = [
        "longitude", "latitude",
    ]

    @classmethod
    def empty_contents(cls):
        r"""Create empty contents for a file wrapper instance."""
        return {
            "geometry": {
                "coordinates": [0.0, 0.0, 0.0],
            },
            "properties": {
                "layers": [
                    {
                        "name": "clay",  # standin, could be anythingxo
                        "depths": [],
                        #     {
                        #         "range": {
                        #             "top_depth": 0.0,
                        #             "bottom_depth": 0.0,
                        #         },
                        #         "values": {
                        #             "mean": 0.0,
                        #         },
                        #     }
                        #     for x in []  #TODO
                        # ]
                    }
                ],
            },
        }

    # TODO: Need to prime layers for use in calculate_missing or
    #   _from_compatible

    def _get(self, name: str):
        r"""Get a parameter from the file.

        Args:
            name: Parameter name.

        Returns:
            Parameter value.

        Raises:
            KeyError: If name is not a valid parameter name.

        """
        if name in self._coord_params:
            return float(self.contents["geometry"]["coordinates"][
                self._coord_params.index(name)])
        elif name in ["top_depth", "bottom_depth"]:
            if len(self.contents["properties"]["layers"]) == 0:
                return np.array([], dtype=np.float64)
            return np.array([
                depth["range"][name] for depth in
                self.contents["properties"]["layers"][0]["depths"]
            ])
        for layer in self.contents["properties"]["layers"]:
            if layer["name"] == name:
                return np.array([
                    depth["values"]["mean"] for depth in layer["depths"]
                ], dtype=np.float64)
        raise KeyError(name)

    def resize_layer_parameters(self, nlayers: int) -> None:
        r"""Resize the layer parameters to match a new number of layers.

        Args:
            nlayers: Updated number of layers.

        """
        for layer in self.contents["properties"]["layers"]:
            if len(layer["depths"]) == nlayers:
                continue
            elif len(layer["depths"]) > nlayers:
                layer["depths"] = layer["depths"][:nlayers]
                continue
            for i in range(nlayers - len(layer["depths"])):
                layer["depths"].append(
                    {
                        "range": {
                            "top_depth": 0.0,
                            "bottom_depth": 0.0,
                        },
                        "values": {
                            "mean": 0.0,
                        },
                    }
                )

    def _set(self, name: str, value: Any,
             allow_add: Optional[bool] = False) -> Any:
        r"""Set a parameter in the file.

        Args:
            name: Parameter name.
            value: Parameter value.
            allow_add: Allow for a new parameter value to be added.

        Raises:
            KeyError: If name is not a valid parameter name.

        """
        if name in self._coord_params:
            assert isinstance(value, float)
            self.contents["geometry"]["coordinates"][
                self._coord_params.index(name)] = str(value)
            return
        if isinstance(value, np.ndarray):
            value = value.tolist()
        assert isinstance(value, list)
        if name in ["top_depth", "bottom_depth"]:
            nlayers = self.get_param_nlayers(value)
            # if nlayers and not self.contents["properties"]["layers"]:
            #     self.contents["properties"]["layers"].append({
            #         "name": "clay",  # standin
            #         "depths": [],
            #     })
            if nlayers != self.nlayers:
                self.resize_layer_parameters(nlayers)
            for layer in self.contents["properties"]["layers"]:
                for v, depth in zip(value, layer["depths"]):
                    depth["range"][name] = v
            return
        assert len(value) == len(self.depths)
        for layer in self.contents["properties"]["layers"]:
            if layer["name"] == name:
                assert len(value) == len(layer["depths"])
                for v, depth in zip(value, layer["depths"]):
                    depth["values"]["mean"] = v
                return
        if not allow_add:
            raise KeyError(name)
        self.contents["properties"]["layers"].append(
            {
                "name": name,
                "depths": [
                    {
                        "range": {
                            "top_depth": start,
                            "bottom_depth": end,
                        },
                        "values": {
                            "mean": v,
                        },
                    }
                    for v, (start, end) in zip(value, self.depths)
                ]
            }
        )

    @classmethod
    def _round_location(cls, loc: float | Tuple[float, float]):
        if isinstance(loc, float):
            loc = np.round(loc)
        return super()._round_location(loc)

    @readonly_cached_property
    def _internal_parameters(self) -> list:
        r"""list: Set of soil parameters contained by this file."""
        return [
            layer["name"]
            for layer in self.contents["properties"]["layers"]
        ]

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
        parameters = cls.parameters_for_download(parameters)
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
    REQUIRED_DOWNLOAD_PARAMETERS: ClassVar[List[str]] = [
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
    HEADER_PARAMETERS: ClassVar[List[str]] = [
        "latitude", "longitude",
    ]
    # TODO: These parameters were used by apsimNGpy, but are currently
    #   not used
    # DEFAULT_DOWNLOAD_PARAMETERS: ClassVar[List[str]] = [
    #     "partdensity",    # pd,
    #     "dbthirdbar_h",   # bb,
    #     "musym",          # musymbol,
    #     "compname",       # componentname,
    #     "slope_r",
    #     "slope_h",        # slope,
    #     "hzname",
    #     "awc_r",          # PAW,
    #     "iacornsr",       # CSR,
    #     "ksat_r",         # sat_hidric_cond,
    #     # (dbthirdbar_r-wthirdbar_r)/100 as bd
    # ]
    PARAMETER_ALIASES: ClassVar[Mapping[str, str]] = {
        "hzdept_r": "top_depth",
        "hzdepb_r": "bottom_depth",
        "elev_r": "elevation",
        "ph1to1h2o_r": "pH",
        "claytotal_r": "clay",
        "silttotal_r": "silt",
        "sandtotal_r": "sand",
        # rocks?
        "dbthirdbar_r": "bulk_density",
        "om_r": "organic_carbon",
        # No nitrogen
        "ksat_l": "ksat",
        "wthirdbar_h": "dul",
        "wfifteenbar_r": "wilting_point",
        "wsatiated_r": "sat",
        "cec7_r": "cec",
    }
    PARAMETER_UNITS: ClassVar[Mapping[str, str]] = {
        "hzdept_r": "cm",
        "hzdepb_r": "cm",
        "elev_r": "m",
        "ph1to1h2o_r": "",
        "claytotal_r": "%",
        "silttotal_r": "%",
        "sandtotal_r": "%",
        # rocks?
        "dbthirdbar_r": "g/cm3",
        "om_r": "%",
        # No nitrogen
        "ksat_l": "um/s",
        "wthirdbar_h": "%",
        "wfifteenbar_r": "%",
        "wsatiated_r": "%",
        # really "meq/(100*g(", but assumed equivalent
        "cec7_r": "mmol/(100*g)",
    }

    @classmethod
    def _read(cls, fname: str):
        r"""Read a data file.

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

    def _get(self, name: str):
        r"""Get a parameter from the file.

        Args:
            name: Parameter name.

        Returns:
            Parameter value.

        Raises:
            KeyError: If name is not a valid parameter name.

        """
        root = self.contents.getroot()
        if name in ["latitude", "longitude"]:
            assert root[1].tag == "location"
            return float(root[1].attrib[name])
        # This version extracts an individual field
        # for idx, x in enumerate(root[0][0][0][1][0][0]):
        #     if x.tag == name:
        #         break
        # else:
        #     raise KeyError(name)
        # out = []
        # for row in root[0][0][0][1][0]:
        #     if name in self.STRING_FIELDS or row[idx].text is None:
        #         out.append(row[idx].text)
        #     else:
        #         out.append(float(row[idx].text))
        # return np.array(out)
        return np.array(self.df[name])

    def _set(self, name: str, value: Any,
             allow_add: Optional[bool] = False) -> Any:
        r"""Set a parameter in the file.

        Args:
            name: Parameter name.
            value: Parameter value.
            allow_add: Allow for a new parameter value to be added.

        Raises:
            KeyError: If name is not a valid parameter name.

        """
        root = self.contents.getroot()
        if name in ["latitude", "longitude"]:
            assert root[1].tag == "location"
            root[1].attrib[name] = str(value)
            return
        for idx, x in enumerate(root[0][0][0][1][0][0]):
            if x.tag == name:
                break
        else:
            if not allow_add:
                raise KeyError(name)
            import xml.etree.ElementTree as ET
            for row in root[0][0][0][1][0]:
                ET.SubElement(row, name)
        assert len(value) == len(root[0][0][0][1][0])
        for row, v in zip(root[0][0][0][1][0], value):
            if v is None or np.isnan(v):
                row[idx].text = None
            else:
                row[idx].text = str(v)

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
        parameters = cls.parameters_for_download(parameters)
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

    @readonly_cached_property
    def _internal_parameters(self) -> list:
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
        out = out.drop_duplicates(subset=["hzdepb_r"]).sort_values(
            'hzdepb_r', ascending=True)
        ix = out["hzdepb_r"].isnull()
        out = out[~ix]
        return out

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


class GeoTIFFWrapper:

    def __init__(self, fname: str, index: int,
                 scale: Optional[float] = 1.0,
                 units: Optional[str] = ""):
        self.fname = fname
        self.index = index
        self.scale = scale
        self._units = [units]

    @property
    def units(self) -> Union[str, None]:
        return self._units[-1]

    @units.setter
    def units(self, value: str | None) -> None:
        self._units.append(value)

    @property
    def data(self) -> Union[np.ndarray, rj_units.QuantityArray]:
        import rasterio
        fd = rasterio.open(self.fname)
        out = rj_units.QuantityArray(
            self.scale * self.fd.read(self.index).astype(np.float64),
            self._units[0]
        )
        fd.close()
        return_scalar = False
        for x in self._units[::-1]:
            if x is None:
                return_scalar = True
                continue
            out = out.to(x)
            break
        if return_scalar:
            return out.value
        return out


class HUMERISSoilData(BaseSoilMapFile):
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
    REQUIRED_OPTIONAL_PACKAGES: ClassVar[List[str]] = ["rasterio"]
    _default_ext: ClassVar[str] = ""
    _EXPECTS_DIRECTORY: ClassVar[bool] = True
    STATIC_PARAMETERS: ClassVar[dict] = {
        "start_date": datetime.date(1980, 1, 1),
        "end_date": datetime.date(2025, 12, 31),
        "latitude": (-90.0, 90.0),
        "longitude": (-180.0, 180.0),
        "top_depth": 0.0,
        "bottom_depth": 30.0,
    }
    PYTEST_MARKS: ClassVar[List[str]] = ["slow"]
    HEADER_PARAMETERS: ClassVar[List[str]] = [
        "latitude", "longitude",
        "top_depth", "bottom_depth",
    ]
    PARAMETER_ALIASES: ClassVar[Mapping[str, str]] = {
        "N": "nitrogen",
        "ECe": "salinity",
        "OC": "organic_carbon",
        "pH": "pH",
    }
    # TODO: Determine the units for HUMERIS
    PARAMETER_UNITS: ClassVar[Mapping[str, str]] = {
        "top_depth": "cm",
        "bottom_depth": "cm",
        "N": "g/kg",
        "ECe": "dS/m",
        "OC": "g/kg",
        "pH": "",
    }

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
        r"""Read a data file.

        Args:
            fname: Path to file to read.

        Returns:
            object: File contents.

        """
        files = sorted(glob.glob(os.path.join(fname, "Mean", "*.tif")))
        assert files
        out = {}
        for x in files:
            k = os.path.splitext(os.path.basename(x))[0]
            out[k] = GeoTIFFWrapper(
                x, 1, scale=0.01, units=cls.PARAMETER_UNITS[k],
            )
        return out

    @readonly_cached_property
    def _internal_parameters(self) -> list:
        r"""list: Set of soil parameters contained by this file."""
        return list(self.contents.keys())

    def _get(self, name: str):
        r"""Get a parameter from the file.

        Args:
            name: Parameter name.

        Returns:
            Parameter value.

        Raises:
            KeyError: If name is not a valid parameter name.

        """
        # TODO: Return latitude grid?
        if name == "latitude":
            first = self.internal_parameters()[0]
            return (self.contents[first].bounds.bottom,
                    self.contents[first].bounds.top)
        elif name == "longitude":
            first = self.internal_parameters()[0]
            return (self.contents[first].bounds.left,
                    self.contents[first].bounds.right)
        out = self.contents[name]
        # TODO: Figure out how to get data out of rasterio
        # print(out)
        # import pdb; pdb.set_trace()
        return out

    def _set(self, name: str, value: Any,
             allow_add: Optional[bool] = False) -> Any:
        r"""Set a parameter in the file.

        Args:
            name: Parameter name.
            value: Parameter value.
            allow_add: Allow for a new parameter value to be added.

        Raises:
            KeyError: If name is not a valid parameter name.

        """
        if name == "latitude":
            assert len(value) == 2
            for v in self.contents.values():
                v.bounds.bottom = value[0]
                v.bounds.top = value[1]
            return
        elif name == "longitude":
            assert len(value) == 2
            for v in self.contents.values():
                v.bounds.left = value[0]
                v.bounds.right = value[1]
            return
        elif name == "depths":
            assert len(value) == 1
            return
        dst = self.contents[name]
        print(dst)
        raise NotImplementedError(
            "Figure out how to set channels w/ rasterio")


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
    DONT_TEST: ClassVar[bool] = True
    PYTEST_MARKS: ClassVar[List[str]] = ["slow"]
    PARAMETER_PREFIX: ClassVar[str] = None
    HEADER_PARAMETERS: ClassVar[List[str]] = [
        "latitude", "longitude",
        "start_date", "end_date",
    ]
    STATIC_PARAMETERS: ClassVar[dict] = {}

    def __init__(self, *args, **kwargs):
        self._results = None
        super().__init__(*args, **kwargs)
        if self.exists:
            self.complete_files()

    def write(self, *args, **kwargs) -> None:
        r"""Write a new set of contents to the file."""
        super().write(*args, **kwargs)
        if self.results is not None:
            self._write(self.results_file, self.results)

    @classmethod
    def _file_prefix(cls) -> str:
        return f"AppEEARS_{cls.NAME}_{cls.VERSION}_"

    @property
    def bundle_dir(self) -> str:
        r"""str: Path to the directory containing downloaded results."""
        return os.path.splitext(self.fname)[0] + "_results"

    @readonly_cached_property
    def results_file(self) -> str:
        r"""str: Path to the CSV containing the results."""
        return os.path.join(
            self.bundle_dir,
            f"{self.NAME}-{self.VERSION}-results.csv")

    @cached_property
    def results(self):
        r"""pandas.DataFrame: Results table."""
        if not os.path.isfile(self.results_file):
            return None
        return self._read(self.results_file)

    @readonly_cached_property
    def metadata_file(self):
        r"""str: Path to the XML file containing metadata."""
        return os.path.join(
            self.bundle_dir,
            f"{self.NAME}-{self.VERSION}-metadata.xml")

    @readonly_cached_property
    def metadata(self):
        r"""xml.etree.ElementTree: Metadata associated with the results."""
        import xml.etree.ElementTree as ET
        with open(self.metadata_file, 'r') as fd:
            return ET.parse(fd)

    @readonly_cached_property
    def _internal_parameters(self) -> list:
        r"""list: Set of parameters contained by this file."""
        return [
            k.split(self.PARAMETER_PREFIX)[-1] for k in
            self.results.columns
            if k not in ["Latitude", "Longitude"]
        ]

    def _get(self, name: str):
        r"""Get a parameter from the file.

        Args:
            name: Parameter name.

        Returns:
            Parameter value.

        Raises:
            KeyError: If name is not a valid parameter name.

        """
        if name in self.HEADER_PARAMETERS:
            return self.contents[name]
        if name in self.results.columns:
            return np.array(self.results[name])
        prefix = f"{self.NAME}_{self.VERSION}_{self.PARAMETER_PREFIX}"
        return np.array(self.results[prefix + name])

    def _set(self, name: str, value: Any,
             allow_add: Optional[bool] = False) -> Any:
        r"""Set a parameter in the file.

        Args:
            name: Parameter name.
            value: Parameter value.
            allow_add: Allow for a new parameter value to be added.

        Raises:
            KeyError: If name is not a valid parameter name.

        """
        if name in self.HEADER_PARAMETERS:
            self.contents[name] = value
            return
        if (((not allow_add)
             and self.PARAMETER_PREFIX + name not in self.results)):
            raise KeyError(name)
        self.results[self.PARAMETER_PREFIX + name] = value

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
        parameters = cls.parameters_for_download(parameters)
        task_name = os.path.splitext(
            os.path.basename(cls.format_filename(
                latitude, longitude,
                start_date, end_date)))[0]
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
    PARAMETER_PREFIX: ClassVar[str] = "Geophysical_Data_"
    PARAMETER_LIMITS: ClassVar[dict] = {
        "start_date": (
            datetime.date(2015, 3, 31), datetime.date.today()),
    }
    STATIC_PARAMETERS: ClassVar[dict] = {
        "top_depth": 0.0,
        "bottom_depth": 100.0,  # cm, Root zone
        # "bottom_depth":5.0,  # cm, Surface
    }
    # PARAMETER_DEFAULTS: ClassVar[dict] = {
    #     "start_date": datetime.date(2015, 3, 31),
    #     "end_date": datetime.date(2026, 5, 8),
    # }
    # TODO: Remove this after setting aliases
    DEFAULT_DOWNLOAD_PARAMETERS: ClassVar[List[str]] = [
        "land_evapotranspiration_flux",
        "land_fraction_saturated",
        "land_fraction_wilting",
        "leaf_area_index",
        "mwrtm_vegopacity",
        "sm_profile",
        "sm_profile_pctl",
        "sm_profile_wetness",
        "sm_rootzone",
        "sm_rootzone_pctl",
        "sm_rootzone_wetness",
        "sm_surface",
        "sm_surface_wetness",
        "soil_water_infiltration_flux",
        "vegetation_greenness_fraction",
        "surface_temp",
        # "soil_temp_layer1",
        # ...
        # "soil_temp_layer6",
    ]
    PARAMETER_ALIASES: ClassVar[Mapping[str, str]] = {
        # TODO
        # "sm_rootzone": "",
    }
    PARAMETER_UNITS: ClassVar[Mapping[str, str]] = {
        "top_depth": "cm",
        "bottom_depth": "cm",
        "land_evapotranspiration_flux": "kg/m^2/s",
        "land_fraction_saturated": "fraction",
        "land_fraction_wilting": "fraction",
        "leaf_area_index": "fraction",
        "mwrtm_vegopacity": "",
        "sm_profile": "fraction",
        "sm_profile_pctl": "%",
        "sm_profile_wetness": "fraction",
        "sm_rootzone": "fraction",
        "sm_rootzone_pctl": "%",
        "sm_rootzone_wetness": "fraction",
        "sm_surface": "fraction",
        "sm_surface_pctl": "%",
        "sm_surface_wetness": "fraction",
        "soil_water_infiltration_flux": "kg/m^2/s",
        "vegetation_greenness_fraction": "fraction",
        "surface_temp": "K",
    }


class EarthaccessData(BaseSoilFile):
    r"""Wrapper for requesting data via earthaccess."""

    NAME: ClassVar[str] = "earthaccess"
    PRODUCT: ClassVar[str] = None
    VERSION: ClassVar[str] = None
    URL: ClassVar[str] = None
    DONT_TEST: ClassVar[bool] = True
    PYTEST_MARKS: ClassVar[List[str]] = ["slow"]
    _default_ext: ClassVar[str] = ".json"
    REQUIRED_OPTIONAL_PACKAGES: ClassVar[List[str]] = ["earthaccess"]
    STATIC_PARAMETERS: ClassVar[dict] = {}
    PARAMETER_DEFAULTS: ClassVar[dict] = {}
    PARAMETER_LIMITS: ClassVar[dict] = {}
    PARAMETER_ALIASES: ClassVar[Mapping[str, str]] = {}
    PARAMETER_ALIASES_REV: ClassVar[Mapping[str, str]] = {}
    PARAMETER_UNITS: ClassVar[Mapping[str, str]] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.exists:
            self.download_missing()
            self.collect_parameters()

    @staticmethod
    def _on_registration(cls):
        if cls.NAME != "earthaccess":
            assert cls.NAME.startswith("EA_")
            cls.PRODUCT = cls.NAME[3:]
            appeears_cls = FileMeta.get_class(cls.CATEGORY, cls.PRODUCT)
            for k in ["PARAMETER_ALIASES", "PARAMETER_UNITS",
                      "HEADER_PARAMETERS", "PARAMETER_LIMITS",
                      "STATIC_PARAMETERS"]:
                if getattr(cls, k):
                    continue
                setattr(cls, k, copy.deepcopy(getattr(appeears_cls, k)))
        BaseSoilFile._on_registration(cls)

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
    def product_dir(cls) -> str:
        r"""str: Directory where downloaded products will be stored."""
        return os.path.join(EarthaccessData.DEFAULT_CACHE_DIR,
                            cls.PRODUCT)

    @readonly_cached_property
    def df(self):
        r"""pandas.DataFrame: Data frame containing the data."""
        import pandas as pd
        return pd.DataFrame.from_dict(self.contents["parameters"])

    @readonly_cached_property
    def _internal_parameters(self) -> list:
        r"""list: Set of parameters contained by this file."""
        return list(self.df.columns)

    def _get(self, name: str):
        r"""Get a parameter from the file.

        Args:
            name: Parameter name.

        Returns:
            Parameter value.

        Raises:
            KeyError: If name is not a valid parameter name.

        """
        if name in self.HEADER_PARAMETERS:
            return self.contents[name]
        return np.array(self.contents["parameters"][name])

    def _set(self, name: str, value: Any,
             allow_add: Optional[bool] = False) -> Any:
        r"""Set a parameter in the file.

        Args:
            name: Parameter name.
            value: Parameter value.
            allow_add: Allow for a new parameter value to be added.

        Raises:
            KeyError: If name is not a valid parameter name.

        """
        if name in self.HEADER_PARAMETERS:
            self.contents[name] = value
            return
        if isinstance(value, np.ndarray):
            value = value.tolist()
        assert isinstance(value, list)
        if (not allow_add) and name not in self.contents["parameters"]:
            raise KeyError(name)
        self.contents["parameters"][name] = value

    @classmethod
    def get_granule_time(
            cls, x, timezone: Optional[datetime.tzinfo] = None
    ) -> datetime.datetime:
        r"""Extract the midpoint time when the granule data was collected

        Args:
            x: Data granule.
            timezone: Timezone that the time should be returned in.

        Returns:
            datetime.datetime: Midpoint time.

        """
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
        parameters = cls.parameters_for_download(parameters)
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
            short_name=cls.PRODUCT,
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
            short_name=cls.PRODUCT,
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
    # STATIC_PARAMETERS: ClassVar[dict] = {
    #     "latitude": (-85.044, 85.044),
    #     "longitude": (-180.0, 180.0),
    # }
    PARAMETER_ALIASES: ClassVar[Mapping[str, str]] = {}
    PARAMETER_UNITS: ClassVar[Mapping[str, str]] = {}
