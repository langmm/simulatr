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
from functools import cached_property, partial
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


class ParameterCalculator:
    r"""Class for handling derived parameters.

    Args:
        name: Parameter name.
        method: Callable for performing the calculation.
        required: Names of parameters required for the calculation.
        optional: Names of parameters that are optional for the
            calculation.

    """

    REGISTRY: ClassVar[Mapping[str, "ParameterCalculator"]] = {}

    def __init__(self, name: str, method: str, required: List[str],
                 optional: List[str] = None):
        self.name = name
        self.method = method
        self.required = required
        self.optional = optional

    @classmethod
    def get_from_dict(cls, source, params, name, **kwargs):
        return source.get_from_dict(params, name, **kwargs)

    def __call__(self, source, prev=None, params=None, stack=None):
        if not hasattr(source, self.method):
            raise KeyError(f"{self.name}: No method \"{self.method}\" "
                           f"on source {source}")
        fget = (
            partial(self.get_from_dict, source, params, stack=stack)
            if params is not None
            else source.get
        )
        args = tuple([
            prev if (prev is not None and x == self.name) else fget(x)
            for x in self.required
        ])
        kwargs = {}
        if self.optional:
            for k in self.optional:
                try:
                    kwargs[k] = fget(k)
                except KeyError:
                    pass
        return getattr(source, self.method)(*args, **kwargs)

    @classmethod
    def valid_entries(cls, source, external: Optional[list] = None):
        r"""Get the set of parameters that can be calculated for the
        provided source.

        Args:
            source: Data file providing parameters.
            external: List of parameters that can be used to run the
                calculation.

        Returns:
            list: Parameters that can be calculated.

        """
        out = []
        if external is None:
            assert not isinstance(source, type)
            external = source.external_parameters(include_header=True)
        for k in cls.REGISTRY.keys():
            if cls != ParameterNormalizer and k in external + out:
                continue
            if cls.has_valid_entry(source, k, previous=(external + out)):
                out.append(k)
        return out

    @classmethod
    def has_valid_entry(cls, source, name: str,
                        previous: Optional[list] = None,
                        excluded: Optional[list] = None):
        r"""Check if a parameter has a registered calculation that can
        be called from the existing parameters in the data file.

        Args:
            name: Parameter name to check.
            previous: Names of parameters that have already been verified
                to have values or be calculatable.
            excluded: Names of parameters that cannot be used in the
                parameter calculation.

        Returns:
            bool: True if the parameter calculation can be called.

        """
        if previous is None:
            assert not isinstance(source, type)
            previous = source.external_parameters(include_header=True)
        if excluded is None:
            excluded = []
        if name not in cls.REGISTRY:
            return False
        excluded_next = (
            excluded if cls == ParameterNormalizer
            else excluded + [name]
        )
        for x in cls.REGISTRY[name]:
            if not hasattr(source, x.method):
                continue
            missing = [xx for xx in x.required if xx not in previous]
            if not missing:
                return True
            if any(xx in excluded for xx in missing):
                continue
            if all(ParameterCalculator.has_valid_entry(
                    source, xx, previous=previous,
                    excluded=excluded_next)
                   for xx in missing):
                return True
        return False

    @classmethod
    def register_method(cls, name: str, *args: str,
                        optional: List[str] = None) -> Callable:
        r"""Decorator for a method that normalizes a parameter
        using other parameters.

        Args:
            name: Name of the parameter that the method normalizes.
            \*args: Additional keyword arguments are the names of
                 parameters that are required by the calculation.
            optional: Names of optional parameters for the calculation.

        Returns:
            Decorator callable.

        """

        def _register_method(method):
            method_name = method.__qualname__.rsplit('.', 1)[-1]
            cls.REGISTRY.setdefault(name, [])
            cls.REGISTRY[name].append(
                cls(name, method_name, args,
                    optional=optional))
            return method

        return _register_method

    @classmethod
    def try_registered(cls, source, name: str,
                       stack: Optional[list] = None, **kwargs) -> Any:
        r"""Try calculating a parameter using registered parameter
        calculators.

        Args:
            source: Data file providing parameters.
            name: Name of the parameter to calculate.
            \*\*kwargs: Additional keyword arguments are passed to the
                calculators.

        Returns:
            Result of the calculation if successful.

        Raises:
            KeyError: If the calculation fails.

        """
        if stack is None:
            stack = (
                [] if isinstance(source, type)
                else source._CALCULATING
            )
        name_int, name_ext = source.param_name_aliases(name)
        for iname in set([name_int, name_ext]):
            for calculator in cls.REGISTRY.get(iname, []):
                if ((name_ext in stack
                     or any(x in stack for x in calculator.required))):
                    raise KeyError(f"Recursive {iname}")
                stack.append(iname)
                try:
                    return calculator(source, stack=stack, **kwargs)
                except KeyError:
                    pass
                finally:
                    stack.pop()
        raise KeyError(name)


class ParameterNormalizer(ParameterCalculator):
    r"""Class for handling parameter normalization.

    Args:
        name: Parameter name.
        method: Callable for performing the calculation.
        required: Names of parameters required for the normalization,
            not including the parameter itself.
        optional: Names of parameters that are optional for the
            calculation.

    """

    REGISTRY: ClassVar[Mapping[str, "ParameterNormalizer"]] = {}

    def __init__(self, name: str, method: str, required: List[str],
                 optional: List[str] = None):
        if name not in required:
            required = [name] + list(required)
        super().__init__(name, method, required, optional=optional)

    @classmethod
    def try_registered(cls, source, name: str,
                       prev: Optional[Any] = None, **kwargs) -> Any:
        r"""Try normalizing a parameter using registered parameter
        normalizers.

        Args:
            source: Data file providing parameters.
            name: Name of the parameter to normalize.
            prev: Value to normalize.
            \*\*kwargs: Additional keyword arguments are passed to the
                normalizers.

        Returns:
            Result of the normalization if successful.

        Raises:
            KeyError: If the normalization fails.

        """
        name_int, name_ext = source.param_name_aliases(name)
        if prev is None:
            if isinstance(source, type):
                assert "params" in kwargs
                prev = kwargs["params"][name_int]
            else:
                prev = source.get(name_int)
        for iname in set([name_int, name_ext]):
            for normalizer in cls.REGISTRY.get(iname, []):
                try:
                    prev = normalizer(source, prev=prev, **kwargs)
                except KeyError:
                    pass
        return prev


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
    REQUEST_PARAMETERS: ClassVar[List[str]] = []
    INDEPENDENT_PARAMETERS: ClassVar[List[str]] = []
    DEPENDENT_PARAMETERS: ClassVar[List[str]] = []
    SUPPORTED_PARAMETERS: ClassVar[List[str]] = []
    SUPPORTED_UNIVERSAL_PARAMETERS: ClassVar[List[str]] = []
    DEFAULT_EXTERNAL_TYPE: ClassVar[type] = None
    DEFAULT_DOWNLOAD_PARAMETERS: ClassVar[List[str]] = []
    REQUIRED_DOWNLOAD_PARAMETERS: ClassVar[List[str]] = []
    REQUIRED_EXTERNAL_PARAMETERS: ClassVar[dict] = {}
    REQUIRED_PARAMETERS: ClassVar[List[str]] = []
    SORTED_PARAMETERS: ClassVar[List[str]] = []
    SCALAR_PARAMETERS: ClassVar[List[str]] = []
    STRING_PARAMETERS: ClassVar[List[str]] = []
    HEADER_PARAMETERS: ClassVar[List[str]] = []
    STATIC_PARAMETERS: ClassVar[dict] = {}
    DEFAULT_DATE_RANGE: ClassVar[Tuple[datetime.date, datetime.date]] = None
    DEFAULT_CACHE_DIR: ClassVar[str] = None
    PYTEST_MARKS: ClassVar[List[str]] = []
    REQUIRED_OPTIONAL_PACKAGES: ClassVar[List[str]] = []
    PARAMETER_ALIASES: ClassVar[Mapping[str, str]] = {}
    PARAMETER_ALIASES_REV: ClassVar[Mapping[str, str]] = {}
    PARAMETER_ALIASES_REV_ALT: ClassVar[Mapping[str, List[str]]] = {}
    PARAMETER_UNITS: ClassVar[Mapping[str, str]] = {}
    PARAMETER_LIMITS: ClassVar[dict] = {}
    PARAMETER_DEFAULTS: ClassVar[Mapping[str, Any]] = {}
    PARAMETER_CALCULATORS: ClassVar[dict] = {}
    PARAMETER_NORMALIZERS: ClassVar[dict] = {}
    PARAMETER_PRECISION: ClassVar[Mapping[str, int]] = {}
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
            for k in ["PARAMETER_CALCULATORS", "PARAMETER_NORMALIZERS",
                      "PARAMETER_LIMITS"]:
                setattr(cls, k, copy.deepcopy(getattr(cls, k)))
            if cls.DESC is None:
                cls.DESC = cls.NAME
            if cls.DEFAULT_CACHE_DIR is None:
                if cls.NAME not in cfg["directories"]:
                    cfg["directories"][cls.NAME] = os.path.join(
                        os.getcwd(), cls.NAME)
                cls.DEFAULT_CACHE_DIR = cfg["directories"][cls.NAME]
            cls.PARAMETER_ALIASES_REV = {}
            cls.PARAMETER_ALIASES_REV_ALT = {}
            for k, v in cls.PARAMETER_ALIASES.items():
                if v in cls.PARAMETER_ALIASES_REV:
                    cls.PARAMETER_ALIASES_REV_ALT.setdefault(v, [])
                    cls.PARAMETER_ALIASES_REV_ALT[v].append(k)
                else:
                    cls.PARAMETER_ALIASES_REV[v] = k
            cls.STATIC_PARAMETERS = {
                cls.PARAMETER_ALIASES_REV.get(k, k): v
                for k, v in cls.STATIC_PARAMETERS.items()
            }
            cls.UNIT_ALIASES_REV = {
                v: k for k, v in cls.UNIT_ALIASES.items()
            }
            cls.SUPPORTED_PARAMETERS = sorted(list(set(
                cls.INDEPENDENT_PARAMETERS
                + cls.SCALAR_PARAMETERS
                + cls.HEADER_PARAMETERS
                + cls.REQUIRED_PARAMETERS
                + list(cls.STATIC_PARAMETERS.keys())
                + list(cls.PARAMETER_ALIASES.keys())
                + list(cls.PARAMETER_LIMITS.keys())
                + list(cls.PARAMETER_UNITS.keys())
                + list(cls.PARAMETER_DEFAULTS.keys())
                + list(cls.PARAMETER_PRECISION.keys())
            )))
            cls.DEPENDENT_PARAMETERS = [
                k for k in cls.SUPPORTED_PARAMETERS
                if k not in (cls.INDEPENDENT_PARAMETERS
                             + cls.HEADER_PARAMETERS
                             + cls.SCALAR_PARAMETERS
                             + cls.REQUEST_PARAMETERS
                             + list(cls.STATIC_PARAMETERS.keys()))
            ]
            cls.SUPPORTED_UNIVERSAL_PARAMETERS = []
            for k in cls.SUPPORTED_PARAMETERS:
                kuni = cls.PARAMETER_ALIASES.get(k, k)
                if kuni in cls.UNIVERSAL_PARAMETERS:
                    cls.SUPPORTED_UNIVERSAL_PARAMETERS.append(kuni)
            for k in ["latitude", "longitude"]:
                kint = cls.PARAMETER_ALIASES.get(k, k)
                if kint not in cls.SUPPORTED_PARAMETERS:
                    continue
                cls.PARAMETER_UNITS.setdefault(kint, "degrees")
                if k == "latitude":
                    cls.PARAMETER_LIMITS.setdefault(kint, (-90.0, 90.0))
                elif k == "longitude":
                    cls.PARAMETER_LIMITS.setdefault(kint, (-180.0, 180.0))
            if not cls.DEFAULT_DOWNLOAD_PARAMETERS:
                cls.DEFAULT_DOWNLOAD_PARAMETERS = [
                    k for k in cls.SUPPORTED_PARAMETERS
                    if k not in (cls.INDEPENDENT_PARAMETERS
                                 + cls.HEADER_PARAMETERS
                                 + cls.REQUEST_PARAMETERS
                                 + list(cls.STATIC_PARAMETERS.keys()))
                ]
            kstart_date = cls.PARAMETER_ALIASES.get("start_date",
                                                    "start_date")
            kend_date = cls.PARAMETER_ALIASES.get("end_date", "end_date")
            if ((kstart_date in cls.STATIC_PARAMETERS
                 and kstart_date not in cls.PARAMETER_LIMITS)):
                date_limits = (
                    cls.STATIC_PARAMETERS[kstart_date],
                    cls.STATIC_PARAMETERS[kend_date]
                )
                cls.PARAMETER_LIMITS[kstart_date] = date_limits
                cls.PARAMETER_LIMITS[kend_date] = date_limits
            if kstart_date in cls.PARAMETER_LIMITS:
                cls.PARAMETER_LIMITS[kend_date] = cls.PARAMETER_LIMITS[
                    kstart_date]
            for k, v in cls.STATIC_PARAMETERS.items():
                if k not in cls.PARAMETER_LIMITS and isinstance(v, tuple):
                    cls.PARAMETER_LIMITS[k] = v
            if cls.DEFAULT_EXTERNAL_TYPE is not None:

                def _pconv(x):
                    x = cls.DEFAULT_EXTERNAL_TYPE.PARAMETER_ALIASES.get(
                        x, x)
                    return cls.PARAMETER_ALIASES_REV.get(x, x)

                for k in ["DEFAULT_DATE_RANGE", "REQUEST_PARAMETERS"]:
                    if getattr(cls, k, None):
                        continue
                    v = copy.deepcopy(getattr(cls.DEFAULT_EXTERNAL_TYPE, k))
                    if k in ["REQUEST_PARAMETERS"]:
                        v = [_pconv(x) for x in v]
                        v = [
                            x for x in v if x in cls.SUPPORTED_PARAMETERS
                        ]
                    setattr(cls, k, v)

    @staticmethod
    def _ensure_internal_names(cls, dst, dont_include_alt=False):
        for k, v in cls.PARAMETER_ALIASES.items():
            if v in dst:
                dst[dst.index(v)] = k
            if dont_include_alt or k not in dst:
                continue
            if v not in cls.PARAMETER_ALIASES_REV:
                continue
            dst += [
                kalt for kalt in cls.PARAMETER_ALIASES_REV[v]
                if kalt not in dst
            ]

    @staticmethod
    def _add_alternate_aliases(cls, dst):
        for k, v in cls.PARAMETER_ALIASES.items():
            if k not in dst:
                continue
            if v not in cls.PARAMETER_ALIASES_REV:
                continue
            dst += [
                kalt for kalt in cls.PARAMETER_ALIASES_REV[v]
                if kalt not in dst
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
            # except BaseException as e:
            #     logger.warning(
            #         f"Non-ImportError raised during attempt to import "
            #         f"\"{x}\":\n{e}")
            #     return False
        return True

    @classmethod
    def parameter_calculation(cls, name: str, *args: str,
                              optional: List[str] = None) -> Callable:
        r"""Decorator for a BaseFile method that calculates a parameter
        from other parameters.

        Args:
            name: Name of the parameter that the method calculates.
            \*args: Additional keyword arguments are the names of
                 parameters that are required by the calculation.
            optional: Names of optional parameters for the calculation.

        Returns:
            Decorator callable.

        """

        def _parameter_calculation(method):
            method_name = method.__qualname__.rsplit('.', 1)[-1]
            cls.PARAMETER_CALCULATORS.setdefault(name, [])
            cls.PARAMETER_CALCULATORS[name].append(
                ParameterCalculator(name, method_name, args,
                                    optional=optional))
            return method

        return _parameter_calculation

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
        return ParameterCalculator.try_registered(self, name)

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
    def is_parameter_specific(cls, name: str) -> bool:
        r"""Check if different files will be generated for different
        values of the provided parameter.

        Args:
            name: Parameter name.

        Returns:
            bool: True if different files will be generated, False
                otherwise.

        """
        if cls.DEFAULT_EXTERNAL_TYPE is not None:
            return cls.DEFAULT_EXTERNAL_TYPE.is_parameter_specific(name)
        name_int = cls.internal_param_name(name)
        return (name_int in cls.REQUEST_PARAMETERS
                and name_int not in cls.STATIC_PARAMETERS)

    @classmethod
    def covers_multiple_parameter_values(cls, name: str) -> bool:
        r"""Check if the file contains parameters for multiple values
        of the provided parameter.

        Args:
            name: Parameter name.

        Returns:
            bool: True if there are multiple values, False otherwise.

        """
        name_int = cls.internal_param_name(name)
        return (name_int in cls.INDEPENDENT_PARAMETERS)

    @classmethod
    def time_specific(cls) -> bool:
        r"""bool: True if the class tracks time, False otherwise."""
        return cls.is_parameter_specific("start_date")

    @classmethod
    def location_specific(cls) -> bool:
        r"""bool: True if the class has one instance per location,
        False otherwise."""
        return cls.is_parameter_specific("latitude")

    @classmethod
    def multiple_times(cls) -> bool:
        r"""bool: True if the class has multiple times."""
        return cls.covers_multiple_parameter_values("time")

    @classmethod
    def multiple_locations(cls) -> bool:
        r"""bool: True if the class has multiple locations."""
        return cls.covers_multiple_parameter_values("latitude")

    @classmethod
    def get_units(cls, value: Any) -> str:
        r"""Get value units if it has any.

        Args:
            value: Parameter value.

        Returns:
            str: Units string if the value has any units, None if it
                does not.

        """
        if utils.has_units(value):
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
        if utils.has_units(value):
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
        if utils.has_units(value):
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
            if len(out) == 0:
                return out
            if all(utils.has_units(x) for x in out):
                out_units = out[0].units
                assert all(x.units == out_units for x in out[1:])
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
            if name_ext in ["start_date", "end_date", "date"]:
                value = utils.string_to_datetime(value, only_date=True)
            elif name_ext in ["time"]:
                value = utils.string_to_datetime(value)
            if return_scalar:
                return cls.remove_units(value)
            return value
        value_units = cls.PARAMETER_UNITS[name_int]
        value_units = cls.UNIT_ALIASES.get(value_units, value_units)
        return cls.add_units(value, value_units,
                             return_scalar=return_scalar)

    @classmethod
    def _get_post_process(
            cls, name_int: str, out: Any,
            default: Optional[Any] = NoDefault,
            return_scalar: Optional[bool] = False,
            valid_parameters: Optional[List[str]] = [],
    ) -> Any:
        if out is None:
            if default is NoDefault:
                raise KeyError(
                    f"{name_int} not supported. Valid parameters:"
                    f" {valid_parameters}")
            out = default
        out = cls.add_param_units(
            name_int, out, return_scalar=return_scalar)
        # if name_ext in self.normalizable_parameters:
        #     out = ParameterNormalizer.try_registered(
        #         self, name_ext, out)
        return out

    @classmethod
    def _set_pre_process(cls, name_int: str, value: Any) -> Any:
        value = cls.add_param_units(name_int, value, return_scalar=True)
        if name_int in cls.PARAMETER_PRECISION:
            value = np.round(
                value, decimals=cls.PARAMETER_PRECISION[name_int])
        return value

    @classmethod
    def get_from_dict(cls, params: dict, name: str,
                      default: Optional[Any] = NoDefault,
                      return_scalar: Optional[bool] = False,
                      **kwargs) -> Any:
        r"""Get a parameter from a dictionary of parameters,
        converting it to the correct units if necessary or calling
        calculators.

        Args:
            params: Parameter dictionary.
            name: Parameter name.
            default: Value to return if the parameter can't be found.
            return_scalar: If True, return the value scalar without
                units after performing any conversion.
            \*\*kwargs: Additional keyword arguments are passed to
                the calculators.

        Returns:
            Parameter value.

        """
        name_int, name_ext = cls.param_name_aliases(name)
        out = None
        valid_parameters = list(params.keys())
        valid_parameters += [
            k for k in cls.STATIC_PARAMETERS.keys()
            if k not in valid_parameters
        ]
        try:
            if name_int in cls.STATIC_PARAMETERS:
                out = cls.STATIC_PARAMETERS[name_int]
            elif name_int in params:
                out = params[name_int]
            elif name_ext in params:
                out = params[name_ext]
            elif name_int in cls.PARAMETER_DEFAULTS:
                out = cls.PARAMETER_DEFAULTS[name_int]
            else:
                out = ParameterCalculator.try_registered(
                    cls, name_int, params=params, **kwargs)
        except KeyError:
            pass
        return cls._get_post_process(name_int, out, default=default,
                                     return_scalar=return_scalar,
                                     valid_parameters=valid_parameters)

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
        out = src.get(name_ext, default=default)
        return cls._get_post_process(name_int, out,
                                     return_scalar=(not with_units))

    def get(self, name: str, default: Optional[Any] = NoDefault,
            return_scalar: Optional[bool] = False, **kwargs: Any) -> Any:
        r"""Get a parameter from the file.

        Args:
            name: Parameter name.
            default: Value to return if the parameter can't be found.
            return_scalar: If True, return the scalar value without
                units, but scaled to the units of the parameter.
            \*\*kwargs: Additional keyword arguments are passed to _get.

        Returns:
            Parameter value.

        """
        out = None
        name_int, name_ext = self.param_name_aliases(name)
        valid_parameters = []
        if name_int in self.STATIC_PARAMETERS:
            out = self.STATIC_PARAMETERS[name_int]
        else:
            parameters = self.internal_parameters(
                include_header=True)
            parameters_calculated = self.calculatable_parameters
            valid_parameters = parameters + parameters_calculated
            try:
                if name_int in parameters:
                    out = super().get(name_int, **kwargs)
                    if ((isinstance(out, np.ndarray)
                         and out.ndim > 0
                         and np.issubdtype(out.dtype, (float, int)))):
                        fill_value = self.param_fill_value(name_int)
                        if fill_value is not None:
                            out[out == fill_value] = np.nan
                elif name_ext in parameters_calculated:
                    out = self.calculate_parameter(name_int)
            except KeyError:
                pass
            if out is None and name_int in self.PARAMETER_DEFAULTS:
                out = self.PARAMETER_DEFAULTS[name_int]
        return self._get_post_process(name_int, out, default=default,
                                      return_scalar=return_scalar,
                                      valid_parameters=valid_parameters)

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

        """
        name_int, name_ext = self.param_name_aliases(name)
        value = self._set_pre_process(name_int, value)
        # if name_ext in self.normalizable_parameters:
        #     value = ParameterNormalizer.try_registered(
        #         self, name_ext, value)
        value_units = self.get_units(value)
        assert value_units is None
        if name_int not in self.SUPPORTED_PARAMETERS:
            raise KeyError(f"Unsupported parameter \"{name}\"")
        if name_int in self.STATIC_PARAMETERS:
            if value == self.STATIC_PARAMETERS[name_int]:
                return
            raise RuntimeError(f"Cannot set static parameter \"{name}\"")
        old_size = 0
        new_size = 0
        is_empty = False
        if name_int in self.INDEPENDENT_PARAMETERS:
            old_size = self.independent_parameter_dimensions[name_int]
            new_size = self.independent_parameter_size(name_int, value)
            is_empty = (old_size == 0)
        elif name_int in self.DEPENDENT_PARAMETERS:
            old_size = self.dimensions
            new_size = self.dependent_parameter_size(name_int, value)
            is_empty = (0 in old_size)
        if allow_resize or is_empty:
            if ((name_int in self.INDEPENDENT_PARAMETERS
                 and new_size != old_size)):
                self.resize_dependent_parameters(name_int, new_size)
        elif ((name_int in self.DEPENDENT_PARAMETERS
               and new_size != old_size)):
            raise ValueError(
                f"New value for \"{name}\" contains data with the "
                f"shape {new_size}, but the data set currently has "
                f"the shape {old_size}."
            )
        out = super().set(name_int, value, **kwargs)
        for name_alt in self.PARAMETER_ALIASES_REV_ALT.get(name_ext, []):
            super().set(name_alt, value, **kwargs)
        return out

    @classmethod
    def independent_parameters(cls):
        r"""list: Independent parameters."""
        return cls.INDEPENDENT_PARAMETERS

    @classmethod
    def independent_parameter_axes(cls) -> dict:
        r"""dict: Mapping between independent parameter names and the
        axis in dependent variables that represents dependence on that
        independent parameter."""
        return {k: i for i, k in enumerate(cls.independent_parameters())}

    def independent_parameter_size(self, name: str,
                                   value: Optional[Any] = None) -> int:
        r"""Get the size of an independent parameter space.

        Args:
            name: Independent parameter name.
            value: Value to determine size from if different from the
                value in the file.

        Returns:
            int: Size of the independent parameter space.

        """
        assert name in self.INDEPENDENT_PARAMETERS
        try:
            if value is None:
                value = self.get(name)
            return len(value)
        except KeyError:
            return 0

    def dependent_parameter_size(self, name: str,
                                 value: Optional[Any] = None) -> tuple:
        r"""Get the size of a dependent parameter in each of the
        independent parameter dimensions.

        Args:
            name: Dependent parameter name.
            value: Value to determine size from if different from the
                value in the file.

        Returns:
            tuple: Size of the dependent parameter in each of the
                independent parameter dimensions.

        """
        try:
            if value is None:
                value = self.get(name)
            out = value.shape
            assert len(out) == len(self.independent_parameters())
            return out
        except KeyError:
            return tuple(len(self.independent_parameters()) * [0])

    @readonly_cached_property
    def independent_parameter_dimensions(self) -> dict:
        r"""dict: Mapping between independent parameter dimensions and
        their size."""
        out = {
            k: self.independent_parameter_size(k)
            for k in self.INDEPENDENT_PARAMETERS
        }
        return out

    @readonly_cached_property
    def dimensions(self) -> tuple:
        r"""tuple: Shape of the data in each independent dimension."""
        out = [
            self.independent_parameter_dimensions[k]
            for k in self.independent_parameters()
        ]
        return tuple(out)

    def resize_dependent_parameters(self, name: str, size: int) -> None:
        r"""Resize the dependent parameters to match a new size in one
        dimension for an independent parameter.

        Args:
            name: Independent parameter.
            size: New size of the specified independent parameter.

        """
        pass

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
    def internal_param_name(cls, name: str,
                            include_alt: Optional[bool] = False) -> str:
        r"""Get the aliased internal parameter name if there is one.

        Args:
            name: Parameter name.
            include_alt: If True, return a list of all possible parameter
                names.

        Returns:
            str: Aliased parameter name.

        """
        if isinstance(name, list):
            return [cls.internal_param_name(x) for x in name]
        out = cls.PARAMETER_ALIASES_REV.get(name, name)
        if include_alt:
            out = [out] + cls.PARAMETER_ALIASES_REV_ALT.get(name, [])
        return out

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

    def has_parameters(self, *args: str,
                       ignore_calculated: Optional[bool] = False):
        r"""Check if the file has one or more parameters.

        Args:
            \*args: Parameter name(s) to check for.
            ignore_calculated: If True, don't return True if the
                parameter can be calculated, but does not exist
                explicitly in the file.

        Returns:
            bool: True if all of the parameters exist.

        """
        parameters = self.internal_parameters(
            include_header=True,
            include_calculated=(not ignore_calculated))
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
        out = self._internal_parameters
        if out and self.INDEPENDENT_PARAMETERS:
            out = (
                [k for k in self.INDEPENDENT_PARAMETERS if k not in out]
                + out
            )
        out = out + list(self.STATIC_PARAMETERS.keys())
        if include_header:
            out = out + [
                k for k in self.HEADER_PARAMETERS if k not in out
            ]
        if include_calculated:
            out = out + [
                k for k in self.internal_param_name(
                    self.calculatable_parameters)
                if k not in out
            ]
        return out

    @readonly_cached_property
    def calculatable_parameters(self) -> list:
        r"""list: Parameters that can be calculated (universal names)."""
        return ParameterCalculator.valid_entries(self)

    @readonly_cached_property
    def normalizable_parameters(self) -> list:
        r"""list: Parameters that can be normalized (universal names)."""
        return ParameterNormalizer.valid_entries(self)

    def normalize_parameters(self):
        r"""Normalize the parameters in the file."""
        for k in self.normalizable_parameters:
            value = self.get(k)
            value = ParameterNormalizer.try_registered(
                self, k, value)
            self.set(k, value)

    @readonly_cached_property
    def _external_parameters(self) -> list:
        r"""list: Aliased parameters."""
        return self.external_param_name(self.internal_parameters())

    def external_parameters(
            self, include_header: Optional[bool] = False,
            include_calculated: Optional[bool] = False,
    ) -> list:
        r"""list: Aliased parameters."""
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
            **kwargs: Any,
    ) -> dict:
        r"""Get the universal parameter map with the names & units
        expected for another file type of the same category.

        Args:
            dst: File type that the universal parameters should be
                returned for.
            keep_units: If True, don't strip units from the returned
                parameters after conversion.
            \*\*kwargs: Additional keyword arguments are treated as
                independent parameter values to resample for.

        Returns:
            dict: Map of parameters using parameter names and units for
                dst file type.

        """
        parameters = self.universal_parameter_map(
            include_header=True, include_calculated=True)
        external = list(parameters.keys())
        for k in ParameterNormalizer.valid_entries(dst, external=external):
            parameters[k] = ParameterNormalizer.try_registered(
                dst, k, prev=parameters[k], params=parameters)
        for k in ParameterCalculator.valid_entries(dst, external=external):
            try:
                parameters[k] = ParameterCalculator.try_registered(
                    dst, k, params=parameters)
            except KeyError:
                continue
        out = {}
        for k, v in parameters.items():
            for kint in dst.internal_param_name(k, include_alt=True):
                if kint in dst.SUPPORTED_PARAMETERS:
                    out[kint] = dst.add_param_units(
                        kint, v, return_scalar=(not keep_units))
        for k, v in dst.PARAMETER_DEFAULTS.items():
            if k in dst.REQUIRED_PARAMETERS and k not in out:
                out[k] = dst.add_param_units(
                    k, v, return_scalar=(not keep_units))
        # TODO: Assert that dst.INDEPENDENT_PARAMETERS satisfied?
        for k0 in self.INDEPENDENT_PARAMETERS:
            k = dst.internal_param_name(self.external_param_name(k0))
            if k in kwargs or k not in dst.STATIC_PARAMETERS:
                continue
            v = dst.STATIC_PARAMETERS[k]
            if k in dst.STATIC_PARAMETERS and v is not out.get(k, None):
                kwargs[k] = v
        parameter_axes = {
            dst.internal_param_name(self.external_param_name(k)): v
            for k, v in self.independent_parameter_axes().items()
        }
        out = dst.resample_parameter_map(
            out, parameter_axes=parameter_axes, **kwargs)
        out_unsorted = out
        out = {
            k: out_unsorted.pop(k) for k in
            dst.INDEPENDENT_PARAMETERS + dst.SORTED_PARAMETERS
            if k in out_unsorted
        }
        out.update(**out_unsorted)
        return out

    @classmethod
    def resample_parameter_map(cls, parameters: dict,
                               parameter_axes: Optional[dict] = None,
                               **kwargs):
        r"""Resample dependent parameters for new independent parameter
        values.

        Args:
            parameters: Parameters to sample.
            parameter_axes: Mapping between independent parameter names
                and the axis in dependent variables that represents
                dependence on that independent parameter.
            \*\*kwargs: Additional keyword arguments are treated as
                independent parameter values to resample for.

        Returns:
            dict: Resampled parameter map.

        """
        out = parameters  # TODO: Keyword to force copy?
        if parameter_axes is None:
            parameter_axes = cls.independent_parameter_axes()
        klatitude = cls.internal_param_name("latitude")
        klongitude = cls.internal_param_name("longitude")
        if klatitude in kwargs or klongitude in kwargs:
            assert klatitude in kwargs
            assert klongitude in kwargs
            order = sorted([klatitude, klongitude],
                           key=lambda x: parameter_axes[x])
            xold = tuple([out[k] for k in order])
            xnew = tuple([kwargs[k] for k in order])
            axis = tuple([parameter_axes[k] for k in order])
            for k in cls.DEPENDENT_PARAMETERS:
                if k not in out:
                    continue
                # TODO: Pass index as axes?
                # TODO: Handle case of preceeding dimensions
                assert parameter_axes[order[0]] == 0
                out[k] = cls.interp_location(out[k], *xold, *xnew)
                # out[k] = cls.interp_parameter(
                #     out[k], xold, xnew,
                #     axis=axis,
                # )
            out[klatitude] = kwargs.pop(klatitude)
            out[klongitude] = kwargs.pop(klongitude)
            if utils.is_scalar(out[klatitude]):
                parameter_axes.pop(klatitude)
                parameter_axes.pop(klongitude)
        for knew, vnew in kwargs.items():
            kint = cls.internal_param_name(knew)
            assert ((kint in cls.INDEPENDENT_PARAMETERS
                     or kint in cls.STATIC_PARAMETERS))
            vold = out[kint]
            axis = parameter_axes[knew]
            for k in cls.DEPENDENT_PARAMETERS:
                if k not in out:
                    continue
                out[k] = cls.interp_parameter(
                    out[k], vold, vnew, axis=axis,
                )
            if utils.is_scalar(vnew):
                parameter_axes.pop(knew)
            out[kint] = vnew
        return out

    def resample(self, **kwargs):
        r"""Resample the dependent parameters for the specified new
        independent parameters values.

        Args:
            \*\*kwargs: Additional keyword arguments are treated as
                independent parameter values to resample for.

        """
        old_parameters = self.internal_parameter_map(include_header=True)
        new_parameters = self.resample_parameter_map(
            old_parameters, **kwargs)
        # Set independent parameters first
        for k in self.INDEPENDENT_PARAMETERS:
            if k not in new_parameters:
                continue
            self.set(k, new_parameters.pop(k), allow_resize=True)
        # Set remainder
        for k, v in new_parameters.items():
            self.set(k, v)

    @readonly_cached_property
    def latitude(self) -> float:
        r"""float: Latitude (degrees)."""
        out = self.get("latitude")
        if isinstance(out, np.ndarray) and out.ndim > 0:
            out = (np.min(out), np.max(out))
        return out

    @readonly_cached_property
    def longitude(self) -> float:
        r"""float: Longitude (degrees)."""
        out = self.get("longitude")
        if isinstance(out, np.ndarray) and out.ndim > 0:
            out = (np.min(out), np.max(out))
        return out

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

    @ParameterCalculator.register_method("start_date", "time")
    @classmethod
    def start_date_from_time(cls, time: float) -> datetime.date:
        out = min(time)
        if not isinstance(out, datetime.date):
            out = out.astype(datetime.date)
        return out

    @ParameterCalculator.register_method("end_date", "time")
    @classmethod
    def end_date_from_time(cls, time: float) -> datetime.date:
        out = max(time)
        if not isinstance(out, datetime.date):
            out = out.astype(datetime.date)
        return out

    @classmethod
    def _f2str(cls, x: Any) -> str:
        r"""Convert a value to a string for use in a file name."""
        if utils.has_units(x):
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
    def _round_location(cls, loc: float | Tuple[float, float],
                        src: Optional[type] = None):
        if src is None:
            src = cls.DEFAULT_EXTERNAL_TYPE
        if src is not None and src != cls:
            return src._round_location(loc)
        if isinstance(loc, tuple):
            return (cls._round_location(loc[0]),
                    cls._round_location(loc[1]))
        if loc is None:
            return loc
        # This is really the NASA Power limit
        return utils.round(loc, decimals=3)

    @classmethod
    def _format_location(cls, latitude: float | Tuple[float, float],
                         longitude: float | Tuple[float, float], **kwargs):
        assert latitude is not None
        assert longitude is not None
        latitude, longitude = cls._round_location((
            latitude, longitude), **kwargs)
        return f"{cls._f2str(latitude)}_{cls._f2str(longitude)}"

    @classmethod
    def _format_time(cls, start_date: Optional[datetime.date],
                     end_date: Optional[datetime.date]):
        return f"{cls._f2str(start_date)}_to_{cls._f2str(end_date)}"

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
                        suffix: Optional[str] = "",
                        src: Optional[type] = None) -> str:
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
            src: Name/class for the file type that will be used to
                generate the file contents.

        Returns:
            str: File name.

        """
        cache_dir = cache_dir or cls.DEFAULT_CACHE_DIR
        out = cls._file_prefix() + prefix
        if isinstance(src, str):
            src = FileMeta.get_class(cls.CATEGORY, src)
        elif src is None:
            src = cls.DEFAULT_EXTERNAL_TYPE
        if src == cls:
            src = None
        if cls.location_specific():
            if not out.endswith("_"):
                out += "_"
            out += cls._format_location(latitude, longitude,
                                        src=src)
        if cls.time_specific():
            start_date = start_date or cls.PARAMETER_DEFAULTS.get(
                cls.internal_param_name("start_date"), None)
            end_date = end_date or cls.PARAMETER_DEFAULTS.get(
                cls.internal_param_name("end_date"), None)
            assert start_date is not None
            assert end_date is not None
            if not out.endswith("_"):
                out += "_"
            out += cls._format_time(start_date, end_date)
        if src is not None:
            if not out.endswith("_"):
                out += "_"
            out += f"from_{src.NAME}"
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
            **kwargs: Any
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
            \*\*kwargs: Additional keyword arguments are passed to
                from_compatible for files created from compatible base
                files and the class constructor for files created via
                download.

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
            return cls.from_compatible(fbase, cache_dir=cache_dir,
                                       **kwargs)
        cache_dir = cache_dir or cls.DEFAULT_CACHE_DIR
        latitude, longitude = cls._round_location((
            latitude, longitude))
        if isinstance(start_date, datetime.datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime.datetime):
            end_date = end_date.date()
        if cls.time_specific():
            if cls.DEFAULT_DATE_RANGE is not None:
                start_date = start_date or cls.DEFAULT_DATE_RANGE[0]
                end_date = end_date or cls.DEFAULT_DATE_RANGE[1]
            else:
                assert start_date is not None
                assert end_date is not None
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
                    f"data files {cls.PARAMETER_LIMITS['start_date']}"
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
            out = cls(fname, **kwargs)
            if parameters:
                out.add_missing_param(parameters)
            return out
        return cls.download_and_save_data(
            fname, latitude, longitude,
            start_date, end_date,
            parameters=parameters,
            **kwargs
        )

    @classmethod
    def download_and_save_data(
            cls, fname: str,
            latitude: float, longitude: float,
            start_date: Union[datetime.date, datetime.datetime] = None,
            end_date: Union[datetime.date, datetime.datetime] = None,
            parameters: Optional[List[str]] = None,
            **kwargs: Any
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
            \*\*kwargs: Additional keyword arguments are passed to the
                class constructor.

        Returns:
            BaseDataFile: File instance.

        """
        if "slow" in cls.PYTEST_MARKS:
            cls.confirm_download()
        data = cls.download_data(latitude, longitude,
                                 start_date, end_date,
                                 parameters=parameters)
        out = cls(fname, contents=data, **kwargs)
        out.write()
        return out

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
                the class constructor.

        """
        if fname is None:
            fname = cls.format_filename(
                fbase.latitude,
                fbase.longitude,
                fbase.start_date,
                fbase.end_date,
                cache_dir=cache_dir,
                src=type(fbase),
                **kwargs
            )
        if cls.path_exists(fname):
            return cls(fname, **kwargs)
        if fbase.NAME == cls.NAME:
            contents = fbase.contents
            # contents = copy.deepcopy(fbase.contents)
        else:
            # TODO: Get this based on required universal parameters
            if fbase.NAME in cls.REQUIRED_EXTERNAL_PARAMETERS:
                fbase.add_missing_param(
                    cls.REQUIRED_EXTERNAL_PARAMETERS[fbase.NAME])
            contents = cls._from_compatible(fbase)
        if isinstance(contents, cls):
            contents = contents.contents
        out = cls(fname, contents=contents, **kwargs)
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
        out = cls(None)
        out.contents = cls.empty_contents()
        for k, v in parameters.items():
            out.set(k, v, allow_add=True)
        out.normalize_parameters()
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
        for k in cls.HEADER_PARAMETERS + cls.REQUEST_PARAMETERS:
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


class BaseDataMapMixin:
    r"""Generic mixin for data files that cover a map."""

    STATIC_PARAMETERS: ClassVar[dict] = {
        "latitude": (-90.0, 90.0),
        "longitude": (-180.0, 180.0),
    }
    INDEPENDENT_PARAMETERS: ClassVar[List[str]] = [
        # Verify that this is the order of the dimensions in every map
        "latitude", "longitude",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def multiple_locations(cls) -> bool:
        r"""bool: True if the class has multiple locations."""
        return True


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
        "time",
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
    def times(self) -> np.ndarray:
        r"""np.ndarray: Times/dates covered by this file."""
        return self.get("time")

    @ParameterCalculator.register_method("vapor_pressure", "temperature_dew")
    @classmethod
    def vaporpressure_from_dewpoint(cls, Tdew):
        r"""Calculate the vapor pressure from the dew point. Based on the
        calculation in PCSE from

        Allen, R.G., Pereira, L.S., Raes, D. and Smith, M. (1998) Crop
            evapotranspiration. Guidelines for computing crop water
            requirements, FAO irrigation and drainage paper 56)

        """
        a = utils.match_unit_type(6.108, "hPa", Tdew)
        b = utils.match_unit_type(237.3, "degC", Tdew)
        return a * utils.exp((17.27 * Tdew) / (Tdew + b))

    @ParameterCalculator.register_method("temperature_dew", "vapor_pressure")
    @classmethod
    def dewpoint_from_vaporpressure(cls, vp):
        r"""Inverse of vaporpressure_from_dewpoint."""
        a = utils.match_unit_type(6.108, "hPa", vp)
        b = utils.match_unit_type(237.3, "degC", vp)
        return (-b * utils.log(vp / a)
                / (utils.log(vp / a) - 17.27))

    @ParameterCalculator.register_method("temperature_avg", "temperature")
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
    REQUEST_PARAMETERS: ClassVar[List[str]] = [
        "longitude", "latitude",
        "start_date", "end_date",
    ]
    INDEPENDENT_PARAMETERS: ClassVar[List[str]] = [
        "time",
    ]
    REQUIRED_PARAMETERS: ClassVar[List[str]] = [
        "latitude", "longitude", "elevation",
        "start_date", "end_date", "time",
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
    PARAMETER_DEFAULTS: ClassVar[Mapping[str, Any]] = {
        "start_date": datetime.date(1984, 7, 1),
        "end_date": datetime.date(2026, 5, 8),
    }
    _coord_params: ClassVar[List[str]] = [
        "longitude", "latitude", "elevation",
    ]

    @classmethod
    def empty_contents(cls):
        r"""Create empty contents for a file wrapper instance."""
        return {
            "header": {
                "fill_value": -999.0,
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
        out = list(self.contents["properties"]["parameter"].keys())
        if out or hasattr(self, "_times"):
            out = ["time"] + out
        return out

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
        elif name == "time":
            if not hasattr(self, "_times"):
                import pandas as pd
                dates = pd.Series(
                    self.contents["properties"]["parameter"][
                        self.internal_parameters()[1]])
                self._times = pd.to_datetime(
                    dates.index, format="%Y%m%d").to_numpy()
            return self._times
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
        import pandas as pd
        if name in self._coord_params:
            assert isinstance(value, float)
            idx = self._coord_params.index(name)
            coords = self.contents["geometry"]["coordinates"]
            if allow_add:
                while (len(coords) - 1) < idx:
                    coords.append(None)
            self.contents["geometry"]["coordinates"][idx] = str(value)
            return
        elif name in ["start_date", "end_date"]:
            assert isinstance(value, datetime.date)
            self.contents["header"][name.split("_")[0]] = value.strftime(
                "%Y%m%d")
            return
        elif name == "time":
            self._times = value
            return
        assert len(value) == len(self.times)
        if (((not allow_add)
             and name not in self.contents["properties"]["parameter"])):
            raise KeyError(name)
        self.contents["properties"]["parameter"][name] = {
            date.strftime("%Y%m%d"): v
            for date, v in zip(pd.Series(self.times), value)
        }

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
        "time",
        "texture_class",
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
        "organic_matter",   # Organic matter content
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
        "cn_ratio",         # Ratio of carbon to nitrogen
        "moisture",         # Current soil moisture
        "temperature",      # Current soil temperature
        "lai",              # Leaf area index
        "albedo",
        "caco3",
    ]
    INDEPENDENT_PARAMETERS: ClassVar[List[str]] = [
        "top_depth", "bottom_depth", "thickness", "depths"
    ]
    UNIVERSAL_LAYER_DEPTH_PARAMETERS: ClassVar[List[str]] = [
        "top_depth", "bottom_depth", "thickness", "depths"
    ]
    IS_BASE_CLASS: ClassVar[bool] = True
    N_LAYERS_MIN: ClassVar[int] = 1

    def __init__(self, fname: str, generated: Optional[bool] = False,
                 contents: Optional[dict] = None,
                 fname_orig: Optional[str] = None,
                 depths: Optional[list] = None,
                 thickness: Optional[list] = None) -> None:
        r"""Initialize a file wrapper.

        Args:
            fname: Path to a file.
            generated: If True, this file was generated.
            contents: Contents to initialize the file with.
            fname_orig: Original file that this one was generated from.
            depths: List of top & bottom depths defining each layer in
                the soil profile (in cm).
            thickness: The thickness of each layer (in cm). Only used if
                depths is not provided.

        """
        super().__init__(fname, generated=generated, contents=contents,
                         fname_orig=fname_orig)
        if depths is not None or thickness is not None:
            self.resample(depths=depths, thickness=thickness)

    @classmethod
    def independent_parameters(cls):
        r"""list: Independent parameters."""
        param = cls.INDEPENDENT_PARAMETERS.copy()
        param_depths = (
            cls.internal_param_name("top_depth"),
            cls.internal_param_name("bottom_depth")
        )
        # TODO: This removes top_depth so that bottom_depth is used for
        #   the creation of soil profiles, but maybe a mean should be
        #   used?
        if all(x in param for x in param_depths):
            param.remove(param_depths[0])
        return param

    @classmethod
    def independent_parameter_axes(cls) -> dict:
        r"""dict: Mapping between independent parameter names and the
        axis in dependent variables that represents dependence on that
        independent parameter."""
        out = super().independent_parameter_axes()
        param_depths = [
            cls.internal_param_name(k)
            for k in cls.UNIVERSAL_LAYER_DEPTH_PARAMETERS
        ]
        existing = [k for k in param_depths if k in out]
        if existing:
            assert len(existing) == 1
            axis = out[existing[0]]
            for k in param_depths:
                if k not in out:
                    out[k] = axis
        return out

    @readonly_cached_property
    def nlayers(self) -> int:
        r"""int: Number of soil layers covered by this file."""
        try:
            return len(self.depths)
        except KeyError:
            return 0

    @readonly_cached_property
    def depths(self) -> list:
        r"""list: List of (start, end) depth pairs covered by this
        file."""
        return self.get("depths")

    # TODO: Scale factors

    # @ParameterCalculator.register_method("no3", "nitrogen")
    # def no3_from_total_nitrogen(cls, nitrogen):
    #     return nitrogen

    # @ParameterCalculator.register_method("nh4", "nitrogen")
    # def nh4_from_total_nitrogen(cls, nitrogen):
    #     return 0.1 * nitrogen

    # @ParameterCalculator.register_method("urea", "nitrogen")
    # def urea_from_total_nitrogen(cls, nitrogen):
    #     return 0.0 * nitrogen

    @ParameterCalculator.register_method("top_depth", "bottom_depth")
    @classmethod
    def top_depth_from_bottom_depth(cls, bottom_depth):
        depth_units = cls.get_units(bottom_depth)
        if isinstance(bottom_depth, (float, rj_units.Quantity)):
            bottom_depth = [bottom_depth]
        out = copy.deepcopy(bottom_depth)
        top = 0.0
        if depth_units is not None:
            top = cls.add_units(top, depth_units)
        out[1:] = out[:-1]
        out[0] = top
        return out

    @ParameterCalculator.register_method("depths", "top_depth",
                                         "bottom_depth")
    @classmethod
    def depths_from_split(cls, top_depth, bottom_depth):
        depth_units = None
        if utils.has_units(top_depth):
            depth_units = str(top_depth.units)
        if isinstance(top_depth, (float, rj_units.Quantity)):
            top_depth = [top_depth]
        if isinstance(bottom_depth, (float, rj_units.Quantity)):
            bottom_depth = [bottom_depth]
        assert len(top_depth) == len(bottom_depth)
        out = [
            (start, stop) for start, stop in zip(top_depth, bottom_depth)
        ]
        if depth_units is not None:
            out = cls.add_units(out, depth_units)
        return out

    @ParameterCalculator.register_method("depths", "thickness")
    @classmethod
    def depths_from_thickness(cls, thickness):
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

    @ParameterCalculator.register_method("thickness", "top_depth",
                                         "bottom_depth")
    @classmethod
    def thickness_from_split(cls, top_depth, bottom_depth):
        return bottom_depth - top_depth

    @ParameterCalculator.register_method("thickness", "depths")
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

    @ParameterCalculator.register_method("top_depth", "depths")
    @classmethod
    def top_depth_from_depths(cls, depths):
        if len(depths) == 0:
            return np.array([], dtype=np.float64)
        prev_units = cls.get_units(depths)
        out = [x[0] for x in depths]
        if prev_units is not None:
            out = cls.add_units(out, prev_units)
        return out

    @ParameterCalculator.register_method("bottom_depth", "depths")
    @classmethod
    def bottom_depth_from_depths(cls, depths):
        if len(depths) == 0:
            return np.array([], dtype=np.float64)
        prev_units = cls.get_units(depths)
        out = [x[1] for x in depths]
        if prev_units is not None:
            out = cls.add_units(out, prev_units)
        return out

    @ParameterCalculator.register_method("top_depth", "thickness")
    @classmethod
    def top_depth_from_thickness(cls, thickness):
        prev_units = cls.get_units(thickness)
        if prev_units is not None:
            return cls.add_units(
                cls.top_depth_from_thickness(
                    cls.remove_units(thickness)),
                prev_units)
        try:
            if len(thickness) == 0:
                return np.array([], dtype=np.float64)
        except TypeError:
            return 0.0
        return np.array([0.0] + np.cumsum(thickness[:-1]).tolist())

    @ParameterCalculator.register_method("bottom_depth", "thickness")
    @classmethod
    def bottom_depth_from_thickness(cls, thickness):
        prev_units = cls.get_units(thickness)
        if prev_units is not None:
            return cls.add_units(
                cls.bottom_depth_from_thickness(
                    cls.remove_units(thickness)),
                prev_units)
        try:
            if len(thickness) == 0:
                return np.array([], dtype=np.float64)
        except TypeError:
            return thickness
        return np.cumsum(thickness)

    @readonly_cached_property
    def thickness(self) -> list:
        r"""list: Thickness of each soil layer covered by this file."""
        return self.get("thickness")

    @ParameterCalculator.register_method("organic_matter",
                                         "organic_carbon")
    @classmethod
    def organic_matter_from_carbon(cls, oc: float) -> float:
        r"""Calculate organic matter from carbon using Van Bemmelen
        factor (Brady & Weil, 2016).

        Args:
            oc: Organic carbon content.

        Returns:
            float: Organic matter content.

        """
        return oc * 1.724

    @ParameterCalculator.register_method("organic_carbon",
                                         "organic_matter")
    @classmethod
    def organic_carbon_from_matter(cls, om: float) -> float:
        r"""Calculate organic carbon from matter using Van Bemmelen
        factor (Brady & Weil, 2016).

        Args:
            om: Organic matter content.

        Returns:
            float: Organic carbon content.

        """
        return om / 1.724

    @ParameterCalculator.register_method("silt", "clay", "sand")
    @classmethod
    def silt_from_clay_and_sand(cls, clay: float, sand: float) -> float:
        r"""Calculate silt content from clay & sand content.

        Args:
            clay: Clay content.
            sand: Sand content.

        Returns:
            float: Silt content.

        """
        # TODO: Clay & sand derived?
        clay = cls.add_units(clay, "%", return_scalar=True)
        sand = cls.add_units(sand, "%", return_scalar=True)
        return cls.add_units(100 - (clay + sand), "%").to('')

    @classmethod
    def _prepare_saxton_rawls(cls, clay: float, sand: float,
                              om: float) -> Tuple[float, float, float]:
        r"""The equations from Saxton and Rawls (2006) are stated to take
        inputs in units of percent (clay, sand, om), but Table 3 values
        require clay & sand to be fractions and om to be %. This method
        ensures that the inputs are in the correct units for those
        equations. If the inputs do not have explicit units, they are
        assumed to be in fractions.

        Args:
            clay: Clay content.
            sand: Sand content.
            om: Organic matter content.

        Returns:
            tuple: clay, sand, & om values in the correct units.

        """
        if clay is not None:
            clay = cls.add_units(clay, "", return_scalar=True)
        if sand is not None:
            sand = cls.add_units(sand, "", return_scalar=True)
        if om is not None:
            om = cls.add_units(om, "").to("%").value
        return (clay, sand, om)

    @ParameterCalculator.register_method("dul", "clay", "sand",
                                         "organic_matter",
                                         optional=["sat"])
    @classmethod
    def dul_from_texture(cls, clay: float, sand: float, om: float,
                         sat: Optional[float] = None) -> float:
        r"""Estimate drained upper limit (field capacity) from soil
        texture and organic matter using the Saxton and Rawls (2006)
        pedotransfer function Eq. 2 (Theta_33).

        Args:
            clay: Clay content.
            sand: Sand content.
            om: Organic matter content.
            sat: 0 kPa moisture, used to set the upper limit
                for the drained upper limit.

        Returns:
            float: Drained upper limit (m3/m3).

        """
        clay, sand, om = cls._prepare_saxton_rawls(clay, sand, om)
        ans0 = (-0.251 * sand + 0.195 * clay + 0.011 * om
                + 0.006 * sand * om - 0.027 * clay * om
                + 0.452 * sand * clay + 0.299)
        out = ans0 + (1.283 * ans0**2 - 0.374 * ans0 - 0.015)
        out = cls.add_units(out, '')
        if sat is not None:
            out_buf = cls.add_units(0.02, "")
            if isinstance(out, rj_units.QuantityArray):
                for i in range(len(out)):
                    if out[i] > sat[i]:
                        out[i] = sat[i] - out_buf
            elif out > sat:
                out = sat - out_buf
        return out

    @ParameterCalculator.register_method("wilting_point",
                                         "clay", "sand",
                                         "organic_matter")
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
        clay, sand, om = cls._prepare_saxton_rawls(clay, sand, om)
        ans0 = (-0.024 * sand + 0.487 * clay + 0.006 * om
                + 0.005 * sand * om - 0.013 * clay * om
                + 0.068 * sand * clay + 0.031)
        out = ans0 + (0.14 * ans0 - 0.02)
        out = cls.add_units(out, '')
        return out

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
        clay, sand, om = cls._prepare_saxton_rawls(clay, sand, om)
        ans0 = (0.278 * sand + 0.034 * clay + 0.022 * om
                - 0.018 * sand * om - 0.027 * clay * om
                - 0.584 * sand * clay + 0.078)
        out = ans0 + (0.636 * ans0 - 0.107)
        out = cls.add_units(out, '')
        return out

    @ParameterCalculator.register_method("sat", "bulk_density",
                                         optional=["particle_density"])
    @classmethod
    def sat_from_bulk_density(
            cls, bd: float,
            particle_density: Optional[float] = 2.65) -> float:
        r"""Estimate saturated water content from the bulk density and
        particle density (adapted from apsimNGpy; Saxon and Rawls (2006)
        Eq. 6).

        Args:
            bd: Bulk density.
            particle_density: Particle density.

        Returns:
            float: Saturated water content (m3/m3).

        """
        bd = cls.add_units(bd, "g/cm3")
        particle_density = cls.add_units(particle_density, "g/cm3")
        a = utils.match_unit_type(1.0, "", bd, particle_density)
        b = utils.match_unit_type(0.02, "", bd, particle_density)
        return a - (bd / particle_density) - b

    @ParameterCalculator.register_method("sat", "clay", "sand",
                                         "organic_matter")
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
        # Must do this after so that values in unexpected units are not
        # passed to dul_from_texture and dul_s_from_textur (e.g. om in %
        # but provided as a scalar without excluded units will be assumed
        # to be a fraction)
        clay, sand, om = cls._prepare_saxton_rawls(clay, sand, om)
        a = 0.097 * sand
        b = cls.add_units(0.043, '')
        return dul + dul_s - a + b

    @ParameterCalculator.register_method("ksat",
                                         "sat", "dul", "wilting_point")
    @classmethod
    def ksat_from_moisture(cls, theta_s: float, theta_33: float,
                           theta_1500: float) -> float:
        r"""Estimate saturated hydraulic conductivity from soil moisture
        properties using the Saxton and Rawls (2006) pedotransfer
        function (Eq. 16 K_s).

        Args:
            theta_s: 0 kPa moisture, saturated moisture.
            theta_33: 33 kPa moisture, drained upper limit.
            theta_1500: 1500 kPa moisture, wilting point.

        Returns:
            float: Saturated hydraulic conductivity (mm/day).

        """
        a = utils.match_unit_type(1930.0, "mm/hr",
                                  theta_s, theta_33, theta_1500)
        theta_s = cls.add_units(theta_s, '', return_scalar=True)
        theta_33 = cls.add_units(theta_33, '', return_scalar=True)
        theta_1500 = cls.add_units(theta_1500, '', return_scalar=True)
        b = ((utils.log(1500.0) - utils.log(33.0))
             / (utils.log(theta_33) - utils.log(theta_1500)))
        diff = theta_s - theta_33
        diff[diff < 0] = 0
        out = a * utils.pow(diff, 3.0 - (1.0 / b))
        return out

    @ParameterCalculator.register_method("ksat", "clay", "sand",
                                         "organic_matter")
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
        a = utils.match_unit_type(1930.0, "mm/hr", clay, sand, om)
        theta_33 = cls.dul_from_texture(clay, sand, om)
        theta_1500 = cls.wilting_point_from_texture(clay, sand, om)
        theta_s = cls.sat_from_texture(clay, sand, om)
        theta_33 = theta_33.to('').value
        theta_1500 = theta_1500.to('').value
        theta_s = theta_s.to('').value
        b = ((utils.log(1500) - utils.log(33))
             / (utils.log(theta_33) - utils.log(theta_1500)))
        out = a * utils.pow(theta_s - theta_33, 3 - (1.0 / b))
        return out

    @ParameterCalculator.register_method("ksat", "clay", "sand")
    @classmethod
    def ksat_from_texture_soilmat_v1(cls, clay: float, sand: float):
        r"""Estimate saturated hydraulic conductivity from soil texture
        using AI suggested calculation used by Samarth in V1 of SoilMat.

        Args:
            clay: Clay content.
            sand: Sand content.

        Returns:
            float: Saturated hydraulic conductivity (mm/day).

        """
        unit_type = rj_units.Quantity
        if any(isinstance(x, rj_units.QuantityArray)
               for x in [clay, sand]):
            unit_type = rj_units.QuantityArray
        clay = cls.add_units(clay, "", return_scalar=True)
        sand = cls.add_units(sand, "", return_scalar=True)
        a = unit_type(5.0, "mm/day")
        b = unit_type(2.0, "mm/day")
        c = unit_type(1.5, "mm/day")
        vmax = unit_type(500.0, "mm/day")
        vmin = unit_type(1.0, "mm/day")
        out = a + b * sand - c * clay
        if isinstance(out, np.ndarray):
            out[out > vmax] = vmax
            out[out < vmin] = vmin
        else:
            out = max(vmin, min(vmax, out))
        return out

    @ParameterCalculator.register_method("nitrogen", "organic_carbon",
                                         "cn_ratio")
    @classmethod
    def nitrogen_from_cn_ratio(cls, organic_carbon: Any,
                               cn_ratio: Any) -> Any:
        r"""Estimate the nitrogen content from the organic carbon
        content and the soil C:N ratio.

        Args:
            organic_carbon: Organic carbon content.
            cn_ratio: Soil C:N ratio.

        Returns:
            float: Nitrogen content.

        """
        # TODO: There is a bug in division for '%' in yggdrasil_rapidjson
        organic_carbon = cls.add_units(organic_carbon, "")
        return organic_carbon / cn_ratio

    @ParameterCalculator.register_method("cn_ratio", "nitrogen",
                                         "organic_carbon")
    @classmethod
    def cn_ratio_from_nitrogen_and_carbon(cls, nitrogen: Any,
                                          organic_carbon: Any) -> Any:
        r"""Estimate the soil C:N ratio from the nitrogen content and
        the organic carbon content.

        Args:
            nitrogen: Nitrogen content.
            organic_carbon: Organic carbon content.

        Returns:
            float: Soil C:N ratio.

        """
        nitrogen = cls.add_units(nitrogen, '')
        organic_carbon = cls.add_units(organic_carbon, "")
        return organic_carbon / nitrogen

    # These methods allow conversion between USDA & International
    # standards for soil classification where USDA uses 50 µm as the
    # line between silt and sand while International uses 20 µm.
    # Adapted from Minasny et al. (2001) Eq. 3
    #             International        USDA/FAO
    #    clay       <2 µm                <2 µm
    #    silt       2-20 µm              2-50 µm
    #    sand       20-2000 µm           50-2000 µm
    @classmethod
    def intl_clay(cls, clay: float, silt: float):
        return clay

    @classmethod
    def intl_silt(cls, clay: float, silt: float):
        clay_units = cls.get_units(clay)
        clay_units_type = utils.matching_unit_type(clay, silt)
        clay = cls.add_units(clay, "", return_scalar=True)
        silt = cls.add_units(silt, "", return_scalar=True)
        out = (
            -0.004070 - 0.1271 * clay + 0.5527 * silt
            + 0.17 * (clay ** 2) - 0.19 * (silt ** 2)
            + 0.59 * clay * silt
        )
        if isinstance(out, np.ndarray):
            idx = (out < 0)
            out[idx] = 0.1154 * clay[idx] + 0.2212 * silt[idx]
        elif out < 0:
            out = 0.1154 * clay + 0.2212 * silt
        if clay_units is not None:
            out = clay_units_type(out, clay_units)
        return out

    @ParameterCalculator.register_method("texture_class", "clay", "sand")
    @classmethod
    def texture_class_minasny(cls, clay: float, sand: float) -> str:
        r"""Get the soil texture class from the USDA clay and sand
        fractions converted to the international system (Minasny et
        al., 2001).

        Args:
            clay: Clay content.
            sand: Sand content.

        Returns:
            str: Texture class name.

        """
        clay = cls.add_units(clay, "", return_scalar=True)
        sand = cls.add_units(sand, "", return_scalar=True)
        if isinstance(clay, (np.ndarray, list)):
            assert len(clay) == len(sand)
            clay = clay[0]
            sand = sand[0]
        silt = 1.0 - (clay + sand)
        intl_clay = np.round(cls.intl_clay(clay, silt), decimals=5)
        intl_silt = np.round(cls.intl_silt(clay, silt), decimals=5)
        intl_sand = 1 - intl_clay - intl_silt
        clay = intl_clay
        silt = intl_silt
        sand = intl_sand
        if sand < (0.75 - clay) and clay >= 0.40:
            return "silty clay"
        if sand < (0.75 - clay) and clay >= 0.26:
            return "silty clay loam"
        if sand < (0.75 - clay):
            return "silty loam"
        if (clay >= (0.40 + ((0.305 - 0.40) / (0.635 - 0.35)
                             * (sand - 0.35)))
            and clay < (0.50 + ((0.305 - 0.50) / (0.635 - 0.50)
                                * (sand - 0.50)))):
            return "clay"
        if (clay >= (0.26 + ((0.305 - 0.26) / (0.635 - 0.74)
                             * (sand - 0.74)))):
            return "sandy clay"
        if (clay >= (0.26 + ((0.17 - 0.26) / (0.83 - 0.49)
                             * (sand - 0.49)))
            and clay < (0.10 + ((0.305 - 0.10) / (0.635 - 0.775)
                                * (sand - 0.775)))):
            return "clay loam"
        if (clay >= (0.26 + ((0.17 - 0.26) / (0.83 - 0.49)
                             * (sand - 0.49)))):
            return "sandy clay loam"
        if (clay >= (0.10 + ((0.12 - 0.10) / (0.63 - 0.775)
                             * (sand - 0.775)))
            and clay < (0.10 + ((0.305 - 0.10) / (0.635 - 0.775)
                                * (sand - 0.775)))):
            return "loam"
        if (clay >= (0.10 + ((0.12 - 0.10) / (0.63 - 0.775)
                             * (sand - 0.775)))):
            return "sandy loam"
        if (clay < (0.0 + ((0.08 - 0.0) / (0.88 - 0.93)
                           * (sand - 0.93)))):
            return "loamy sand"
        return "sand"

    @ParameterCalculator.register_method("texture_class", "clay", "sand")
    @classmethod
    def texture_class_usda(cls, clay: float, sand: float) -> str:
        r"""Get the soil texture class from the USDA clay and sand
        fractions (from USDA Soil Survey Manual, Handbook No. 18, 2017).

        Args:
            clay: Clay content.
            sand: Sand content.

        Returns:
            str: Texture class name.

        """
        clay = cls.add_units(clay, "", return_scalar=True)
        sand = cls.add_units(sand, "", return_scalar=True)
        if isinstance(clay, (np.ndarray, list)):
            assert len(clay) == len(sand)
            clay = clay[0]
            sand = sand[0]
        clay = np.round(clay, decimals=5)
        sand = np.round(sand, decimals=5)
        silt = 1.0 - (clay + sand)
        if (sand > 0.85 and (silt + 1.5 * clay) < 0.15):
            return "sand"
        elif (0.70 < sand < 0.90 and (silt + 1.5 * clay) >= 0.15
              and (silt + 2.0 * clay) < 0.30):
            return "loamy sand"
        elif ((silt + 2.0 * clay) >= 0.3
              and ((sand > 0.52 and 0.07 <= clay < 0.20)
                   or (clay < 0.07 and silt < 0.5))):
            return "sandy loam"
        elif ((0.07 <= clay < 0.27) and (0.28 <= silt < 0.50)
              and sand <= 0.52):
            return "loam"
        elif ((silt >= 0.50 and 0.12 <= clay < 0.27)
              or (0.50 <= silt < 0.80 and clay < 0.12)):
            return "silty loam"
        elif (silt >= 0.80 and clay < 0.12):
            return "silt"
        elif (0.20 <= clay < 0.35 and silt < 0.28 and sand > 0.45):
            return "sandy clay loam"
        elif (0.27 <= clay < 0.40 and 0.20 < sand <= 0.45):
            return "clay loam"
        elif (0.27 <= clay < 0.40 and sand <= 0.20):
            return "silty clay loam"
        elif (clay >= 0.35 and sand > 0.45):
            return "sandy clay"
        elif (clay >= 0.40 and silt >= 0.40):
            return "silty clay"
        elif (clay >= 0.40 and sand <= 0.45 and silt < 0.40):
            return "clay"
        raise Exception(f"Unclassified:\n"
                        f"    clay = {clay}\n"
                        f"    sand = {sand}\n"
                        f"    silt = {silt}")

    @classmethod
    def interp_parameter(cls, y: Any, xold: Any, xnew: Any,
                         kind: Optional[str] = "linear",
                         fill_value: Optional[str] = "extrapolate",
                         assume_sorted: Optional[bool] = False,
                         axis: Optional[int] = -1,
                         **kwargs) -> np.ndarray:
        r"""Interpolate a parameter to a new set of independent values.

        Args:
            y: Initial dependent values.
            xold: Initial independent values.
            xnew: New independent values.
            kind: Type of interpolation to use.
            \*\*kwargs: Additional keyword arguments are passed to
                scipy.interpolate.interp1d.

        Returns:
            np.ndarray: Interpolated profile.

        """
        from scipy import interpolate
        if not isinstance(axis, tuple):
            yinterp = interpolate.interp1d(
                xold, y,
                kind=kind,
                assume_sorted=assume_sorted,
                fill_value=fill_value,
                axis=axis,
                **kwargs
            )
            return yinterp(xnew)
        assert isinstance(xold, tuple) and len(xold) == len(axis)
        assert y.ndim >= len(axis) and y.ndim >= max(axis)
        order = list(axis)
        order += [i for i in range(y.ndim) if i not in order]
        old_shape = tuple([y.shape[i] for i in order])
        assert tuple([len(x) for x in xold]) == old_shape
        y_transposed = np.transpose(y, order)
        if isinstance(xnew, tuple):
            assert isinstance(xnew, tuple) and len(xnew) == len(axis)
            xnew_grid = np.vstack([
                x.flatten() for x in np.meshgrid(*xnew, indexing='ij')
            ]).T
            new_shape = tuple([len(x) for x in xnew])
            new_order = np.argsort(order)
        else:
            xnew_grid = xnew
            new_shape = tuple([xnew.shape[0]])
            raise NotImplementedError(
                "Handle new order in case where array of points is "
                "sampled"
            )
        new_shape = tuple(
            list(new_shape)
            + [y.shape[i] for i in range(y.ndim) if i not in axis]
        )
        assert isinstance(xnew_grid, np.ndarray)
        assert xnew_grid.shape[-1] == len(axis)
        y_out = interpolate.interpn(
            xold, y_transposed, xnew_grid,
            method=kind,
            fill_value=fill_value,
        )
        if isinstance(xnew, tuple):
            y_out = y_out.reshape(new_shape)
        return np.transpose(y_out, new_order)

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
                       kind: Optional[str] = "linear",
                       fill_value: Optional[str] = "extrapolate",
                       assume_sorted: Optional[bool] = False,
                       **kwargs: Any) -> np.ndarray:
        r"""Interpolate a profile to a new set of independent values.

        Args:
            y: Initial dependent values.
            xold: Initial independent values.
            xnew: New independent values.
            kind: Type of interpolation to use.
            \*\*kwargs: Additional keyword arguments are passed to
                scipy.interpolate.interp1d.

        Returns:
            np.ndarray: Interpolated profile.

        """
        return cls.interp_parameter(
            y, xold, xnew,
            kind=kind,
            fill_value=fill_value,
            assume_sorted=assume_sorted,
            **kwargs
        )

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
            out = out / out.max()
            return out
        raise RuntimeError(f"Invalid parameters a = {a}, b = {b}")

    @classmethod
    def resample_parameter_map(cls, parameters: dict,
                               parameter_axes: Optional[dict] = None,
                               **kwargs):
        r"""Resample dependent parameters for new independent parameter
        values.

        Args:
            parameters: Parameters to sample.
            parameter_axes: Mapping between independent parameter names
                and the axis in dependent variables that represents
                dependence on that independent parameter.
            \*\*kwargs: Additional keyword arguments are treated as
                independent parameter values to resample for.

        Returns:
            dict: Resampled parameter map.

        """
        if parameter_axes is None:
            parameter_axes = cls.independent_parameter_axes()
        new_depth_param = {}
        old_depth_param = {}
        for kext in cls.UNIVERSAL_LAYER_DEPTH_PARAMETERS:
            k = cls.internal_param_name(kext)
            if k in kwargs:
                new_depth_param[k] = kwargs.pop(k)
            elif kext in kwargs:
                new_depth_param[k] = kwargs.pop(kext)
            if k in parameters:
                old_depth_param[k] = parameters[k]
            elif kext in parameters:
                old_depth_param[k] = parameters[kext]
        if new_depth_param:
            assert old_depth_param
            old_depth = cls.interp_depth_from_param(**old_depth_param)
            new_depth = cls.interp_depth_from_param(**new_depth_param)
            depth_axis = parameter_axes[list(old_depth_param.keys())[0]]
            for k in cls.DEPENDENT_PARAMETERS:
                if k not in parameters:
                    continue
                parameters[k] = cls.interp_profile(
                    parameters[k], old_depth, new_depth,
                    axis=depth_axis,
                )
            for k in old_depth_param.keys():
                parameters.pop(k)
                parameter_axes.pop(k)
            for k in new_depth_param.keys():
                parameter_axes[k] = depth_axis
            parameters.update(**new_depth_param)
        return super().resample_parameter_map(
            parameters, parameter_axes=parameter_axes, **kwargs)

    @classmethod
    def interp_depth_from_param(cls, **kwargs) -> float:
        r"""Convert depth parameters into a single depth used for
        interpolation.

        Args:
            top_depth: Top depths of each layer.
            bottom_depth: Bottom depths of each layer.
            depths: List of top & bottom depths defining each layer in
                the soil profile.
            thickness: The thickness of each layer. Only used if
                depths is not provided.

        Returns:
            np.ndarray: Depths to use in interpolation.

        """
        # TODO: Use create profile if there is a single parameter
        for k in ["top_depth", "bottom_depth", "depths", "thickness"]:
            if k in kwargs:
                continue
            kint = cls.internal_param_name(k)
            if kint in kwargs:
                kwargs[k] = kwargs.pop(kint)
        # top_depth =
        kwargs.pop("top_depth", None)
        bottom_depth = kwargs.pop("bottom_depth", None)
        depths = kwargs.pop("depths", None)
        thickness = kwargs.pop("thickness", None)
        assert not kwargs
        if depths is None and thickness is None and bottom_depth is None:
            raise ValueError("Either bottom_depth, depths, or thickness "
                             "must be provided")
        if bottom_depth is not None:
            # if top_depth is None:
            #     top_depth = cls.top_depth_from_bottom_depth(bottom_depth)
            pass
        elif depths is not None:
            bottom_depth = cls.bottom_depth_from_depths(depths)
            # top_depth = cls.top_depth_from_depths(depths)
        else:
            bottom_depth = cls.bottom_depth_from_thickness(thickness)
            # top_depth = cls.top_depth_from_thickness(thickness)
        # TODO: Use midpoint instead?
        # return (bottom_depth + top_depth) / 2.0
        return bottom_depth


class BaseSoilMapFile(BaseDataMapMixin, BaseSoilFile):
    r"""Generic wrapper for soil files that are based on maps."""

    NAME: ClassVar[str] = "generic_soil_map"
    DESC: ClassVar[str] = "Generic Soil Map"
    DONT_TEST: ClassVar[bool] = True
    IS_BASE_CLASS: ClassVar[bool] = True


class ISRICSoilGridsFile(BaseSoilFile):
    r"""Wrapper for loading ISRIC SoilGrids data from REST API."""

    NAME: ClassVar[str] = "isric_soilgrids_data"
    DESC: ClassVar[str] = "ISRIC SoilGrids"
    URL: ClassVar[str] = (
        "https://rest.isric.org/soilgrids/v2.0/properties/query"
    )
    REQUEST_PARAMETERS: ClassVar[List[str]] = [
        "longitude", "latitude", "quantiles",
    ]
    INDEPENDENT_PARAMETERS: ClassVar[List[str]] = [
        "top_depth", "bottom_depth",
    ]
    HEADER_PARAMETERS: ClassVar[List[str]] = [
        "longitude", "latitude", "quantile", "quantiles",
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
        "phh2o": "0.1",        # 10 * pH
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
    DEFAULT_QUANTILE: ClassVar[str] = "mean"
    DUMMY_FIELD: ClassVar[str] = "DUMMY_FIELD"

    @classmethod
    def empty_contents(cls):
        r"""Create empty contents for a file wrapper instance."""
        return {
            "geometry": {
                "coordinates": [0.0, 0.0, 0.0],
            },
            "properties": {
                "layers": [],
            },
        }

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
        elif name == "quantile":
            quantiles = self.get(
                "quantiles", default=[self.DEFAULT_QUANTILE])
            if self.DEFAULT_QUANTILE in quantiles:
                default_quantile = self.DEFAULT_QUANTILE
            else:
                default_quantile = quantiles[0]
            out = getattr(self, "_quantile", default_quantile)
            assert out in quantiles
            return out
        elif name == "quantiles":
            if not self.contents["properties"]["layers"]:
                raise KeyError(name)
            if not self.contents["properties"]["layers"][0]["depths"]:
                raise KeyError(name)
            return sorted(list(self.contents[
                "properties"]["layers"][0]["depths"][0]["values"].keys()))
        elif name in ["top_depth", "bottom_depth"]:
            if len(self.contents["properties"]["layers"]) == 0:
                return np.array([], dtype=np.float64)
            return np.array([
                float(depth["range"][name]) for depth in
                self.contents["properties"]["layers"][0]["depths"]
            ])
        quantile = self.get("quantile")
        layer = self._find_layer(name, self.contents)
        if layer is not None:
            return np.array([
                depth["values"][quantile] for depth in layer["depths"]
            ], dtype=np.float64)
        raise KeyError(name)

    def resize_dependent_parameters(self, name: str, size: int) -> None:
        r"""Resize the dependent parameters to match a new size in one
        dimension for an independent parameter.

        Args:
            name: Independent parameter.
            size: New size of the specified independent parameter.

        """
        assert name in ["bottom_depth", "top_depth"]
        quantiles = self.get(
            "quantiles", default=[self.DEFAULT_QUANTILE])
        if not self.contents["properties"]["layers"]:
            self.contents["properties"]["layers"].append({
                "name": self.DUMMY_FIELD,
                "depths": [],
            })
        for layer in self.contents["properties"]["layers"]:
            if len(layer["depths"]) == size:
                continue
            elif len(layer["depths"]) > size:
                layer["depths"] = layer["depths"][:size]
                continue
            for i in range(size - len(layer["depths"])):
                layer["depths"].append(
                    {
                        "range": {
                            "top_depth": 0.0,
                            "bottom_depth": 0.0,
                        },
                        "values": {
                            k: 0.0 for k in quantiles
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
        elif name == "quantile":
            quantiles = self.get(
                "quantiles", default=[self.DEFAULT_QUANTILE])
            assert value in quantiles
            setattr(self, "_quantile", value)
            return
        elif name == "quantiles":
            # TODO: Allow set?
            raise KeyError(name)
        if isinstance(value, np.ndarray):
            value = value.tolist()
        assert isinstance(value, list)
        if name in ["top_depth", "bottom_depth"]:
            nlayers = self.independent_parameter_size(name, value)
            # if nlayers and not self.contents["properties"]["layers"]:
            #     self.contents["properties"]["layers"].append({
            #         "name": "clay",  # standin
            #         "depths": [],
            #     })
            if nlayers != self.nlayers:
                self.resize_dependent_parameters("top_depth", nlayers)
            for layer in self.contents["properties"]["layers"]:
                for v, depth in zip(value, layer["depths"]):
                    depth["range"][name] = float(v)
            return
        quantile = self.get("quantile")
        layer = self._find_layer(name, self.contents)
        if layer is not None:
            assert len(value) == len(layer["depths"])
            for v, depth in zip(value, layer["depths"]):
                depth["values"][quantile] = float(v)
            return
        if not allow_add:
            raise KeyError(name)
        layers = self.contents["properties"]["layers"]
        layers.append(
            {
                "name": name,
                "depths": [
                    {
                        "range": {
                            "top_depth": float(start),
                            "bottom_depth": float(end),
                        },
                        "values": {
                            quantile: v,
                        },
                    }
                    for v, (start, end) in zip(value, self.depths)
                ]
            }
        )
        if layers[0]["name"] == self.DUMMY_FIELD:
            layers.pop(0)

    @classmethod
    def _round_location(cls, loc: float | Tuple[float, float], **kwargs):
        loc = super()._round_location(loc, **kwargs)
        if isinstance(loc, float):
            loc = np.round(loc)
        return loc

    @readonly_cached_property
    def _internal_parameters(self) -> list:
        r"""list: Set of soil parameters contained by this file."""
        out = [
            layer["name"]
            for layer in self.contents["properties"]["layers"]
        ]
        if self.DUMMY_FIELD in out:
            out.remove(self.DUMMY_FIELD)
            if not out:
                out += self.INDEPENDENT_PARAMETERS
        return out

    @classmethod
    def _find_layer(cls, name, contents, return_index=False):
        for i, layer in enumerate(contents["properties"]["layers"]):
            if layer["name"] == name:
                if return_index:
                    return i
                return layer
        return None

    @classmethod
    def _find_top_depth(cls, depth, layer, return_index=False):
        for i, depth in enumerate(layer["depths"]):
            if depth["range"]["top_depth"] == depth:
                if return_index:
                    return i
                return depth
        return None

    def update_param(self, contents: Any) -> None:
        r"""Merge downloaded parameters into the current data.

        Args:
            contents: New data to incorporate.

        """
        self.contents["geometry"]["coordinates"] = copy.deepcopy(
            contents["geometry"]["coordinates"])
        for new_layer in contents["properties"]["layers"]:
            iold = self._find_layer(
                new_layer["name"], self.contents,
                return_index=True)
            if iold is None:
                self.contents["properties"]["layers"].append(new_layer)
                continue
            old_layer = self.contents["properties"]["layers"][iold]
            max_depth = max(
                x["range"]["bottom_depth"] for x in old_layer["depths"]
            )
            for new_depth in new_layer["depths"]:
                iold = self._find_top_depth(
                    new_depth["range"]["top_depth"], old_layer,
                    return_index=True)
                if iold is None:
                    assert new_depth["range"]["top_depth"] >= max_depth
                    old_layer["depths"].append(new_depth)
                    continue
                old_layer["depths"][iold]["values"].update(
                    new_depth["values"])

    @classmethod
    def download_data(
            cls, latitude: float, longitude: float,
            start_date: Optional[datetime.date] = None,
            end_date: Optional[datetime.date] = None,
            parameters: Optional[List[str]] = None,
            depths: Optional[List[Union[str, Tuple[int, int]]]] = None,
            quantiles: Optional[List[str]] = ["mean", "Q0.5"],
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
        out = req.json()
        logger.debug("Successfully retrieved data from ISRIC SoilGrids")
        return out


class SoilMatFile(BaseSoilFile):
    r"""Wrapper for loading SoilMat data from the SoilMat REST API."""

    NAME: ClassVar[str] = "soilmat_data"
    DESC: ClassVar[str] = "SoilMat"
    URL: ClassVar[str] = os.environ.get("SOILMAT_URL", None)
    REQUEST_PARAMETERS: ClassVar[List[str]] = [
        "longitude", "latitude",
    ]
    INDEPENDENT_PARAMETERS: ClassVar[List[str]] = [
        "top_depth", "bottom_depth",
    ]
    HEADER_PARAMETERS: ClassVar[List[str]] = [
        "longitude", "latitude",
        "address", "match_type",  # "neighbors_used",
    ]
    PARAMETER_ALIASES: ClassVar[Mapping[str, str]] = {
        "BD": "bulk_density",
        "PH": "pH",
        "Sand_percent": "sand",
        "Silt_percent": "silt",
        "Clay_percent": "clay",
        "OC_percent": "organic_carbon",
        "SoilCNRatio": "cn_ratio",
        "KS_mm_day": "ksat",
        "DUL": "dul",
        "LL15": "wilting_point",
        "SAT": "sat",
    }
    PARAMETER_UNITS: ClassVar[Mapping[str, str]] = {
        "top_depth": "cm",
        "bottom_depth": "cm",
        "Thickness_mm": "mm",
        "Sand_percent": "%",
        "Silt_percent": "%",
        "Clay_percent": "%",
        "OC_percent": "%",
        "BD": "g/cm3",
        "AirDry": "%",
        "LL15": "%",
        "DUL": "%",
        "SAT": "%",
        "KS_mm_day": "mm/day",
        "LL": "mm/mm",
        "SoilCNRatio": "",
        "PH": "",  # 1:5 water
    }
    _using_apsimx_names: ClassVar[bool] = False
    _apsimx_names: ClassVar[List[str]] = []

    @staticmethod
    def _on_registration(cls):
        cls._apsimx_names = [
            k for k in cls.PARAMETER_UNITS
            if k not in ISRICSoilGridsFile.PARAMETER_UNITS
        ]
        if cls._using_apsimx_names:
            cls.NAME += "_apsimx_calc"
        else:
            for k in ["PARAMETER_ALIASES", "PARAMETER_UNITS"]:
                setattr(cls, k, copy.deepcopy(getattr(
                    ISRICSoilGridsFile, k)))
        BaseSoilFile._on_registration(cls)

    @classmethod
    def empty_contents(cls):
        r"""Create empty contents for a file wrapper instance."""
        return {
            "latitude": 0.0,
            "longitude": 0.0,
            "layers": [],
        }

    @classmethod
    def _round_location(cls, loc: float | Tuple[float, float], **kwargs):
        # TODO: SoilMat is coarser than SoilGrids
        return ISRICSoilGridsFile._round_location(loc, **kwargs)

    def _get(self, name: str) -> Any:
        r"""Get a parameter from the file.

        Args:
            name: Parameter name.

        Returns:
            Parameter value.

        Raises:
            KeyError: If name is not a valid parameter name.

        """
        if name in self.HEADER_PARAMETERS + self.SCALAR_PARAMETERS:
            if name in ["address", "match_type", "neighbors_used"]:
                return self.contents[name]
            return float(self.contents[name])
        elif name in ["top_depth", "bottom_depth"]:
            return np.array([
                float(layer[name]) for layer in self.contents["layers"]
            ], dtype=np.float64)
        out = [
            layer["values"][name] for layer in self.contents["layers"]
            if name in layer["values"]
        ]
        if not out:
            raise KeyError(name)
        return np.array(out, dtype=np.float64)

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
        if name in self.HEADER_PARAMETERS + self.SCALAR_PARAMETERS:
            if name in ["address", "match_type"]:
                assert isinstance(value, str)
                self.contents[name] = value
            elif name in ["neighbors_used"]:
                assert isinstance(value, int)
                self.contents[name] = int(value)
            else:
                assert isinstance(value, float)
                self.contents[name] = float(value)
            return
        if isinstance(value, np.ndarray):
            value = value.tolist()
        assert isinstance(value, list)
        if name in ["top_depth", "bottom_depth"]:
            nlayers = self.independent_parameter_size(name, value)
            if nlayers != self.nlayers:
                self.resize_dependent_parameters("bottom_depth", nlayers)
            for v, layer in zip(value, self.contents["layers"]):
                layer[name] = float(v)
            return
        for layer in self.contents["layers"]:
            if name in layer["values"]:
                assert len(value) == len(self.contents["layers"])
                for v, lyr in zip(value, self.contents["layers"]):
                    lyr["values"][name] = float(v)
                return
        if not allow_add:
            raise KeyError(name)
        for v, layer in zip(value, self.contents["layers"]):
            layer["values"][name] = float(v)

    def resize_dependent_parameters(self, name: str, size: int) -> None:
        r"""Resize the dependent parameters to match a new size in one
        dimension for an independent parameter.

        Args:
            name: Independent parameter.
            size: New size of the specified independent parameter.

        """
        assert name in ["bottom_depth", "top_depth"]
        layers = self.contents["layers"]
        if len(layers) > size:
            self.contents["layers"] = layers[:size]
            return
        while len(layers) < size:
            layers.append({
                "top_depth": 0.0,
                "bottom_depth": 0.0,
                "values": {},
            })

    @readonly_cached_property
    def _internal_parameters(self) -> list:
        r"""list: Set of soil parameters contained by this file."""
        out = []
        for layer in self.contents["layers"]:
            for k in layer["values"]:
                if ((k in out
                     or ((not self._using_apsimx_names)
                         and k in self._apsimx_names)
                     or (self._using_apsimx_names
                         and k not in self._apsimx_names))):
                    continue
                out.append(k)
        return out

    @readonly_cached_property
    def stddev(self) -> dict:
        r"""dict: Standard deviation of the values from SoilGrids that
        soilmat used to get the data for the queried point."""
        # TODO: Include estimate of error propagation for calculated
        #   variables. Make this more generic on the base data class,
        #   maybe via get
        out = {}
        for k in self.contents["layers"][0].get("stddev", {}).keys():
            kext = self.external_param_name(k)
            out[kext] = np.array(
                [layer["stddev"][k] for layer in
                 self.contents["layers"]],
                dtype=np.float64)
        return out

    def update_param(self, contents: Any) -> None:
        r"""Merge downloaded parameters into the current data.

        Args:
            contents: New data to incorporate.

        """
        for k in self.HEADER_PARAMETERS + self.SCALAR_PARAMETERS:
            if k in contents:
                self.contents[k] = contents[k]
        for new_layer in contents["layers"]:
            for layer in self.contents["layers"]:
                if ((layer["top_depth"] == new_layer["top_depth"])
                        and (layer["bottom_depth"]
                             == new_layer["bottom_depth"])):
                    layer["values"].update(new_layer["values"])
                    break
            else:
                self.contents["layers"].append(new_layer)

    @classmethod
    def download_data(
            cls, latitude: float, longitude: float,
            start_date: Optional[datetime.date] = None,
            end_date: Optional[datetime.date] = None,
            parameters: Optional[List[str]] = None,
    ) -> dict:
        r"""Use REST API to get SoilMat data for a location.

        Args:
            latitude: Location latitude (degrees).
            longitude: Location longitude (degrees).
            start_date: Starting date for data [UNUSED].
            end_date: Ending date for data [UNUSED].
            parameters: Set of parameters to include.

        Returns:
            dict: Downloaded data, with parameter values converted to
            the units used by ISRICSoilGridsFile.

        """
        parameters = cls.parameters_for_download(parameters)
        latitude = cls.remove_units(latitude)
        longitude = cls.remove_units(longitude)
        logger.debug("Starting retrieval from SoilMat")
        req = requests.get(cls.URL, params={"lat": latitude,
                                            "lon": longitude})
        req.raise_for_status()
        logger.debug("Successfully retrieved data from SoilMat")
        raw = req.json()
        out = {
            "latitude": float(raw.get("latitude", latitude)),
            "longitude": float(raw.get("longitude", longitude)),
            "layers": [
                {
                    "top_depth": float(top),
                    "bottom_depth": float(bottom),
                    "values": {},
                    "stddev": {},
                }
                for top, bottom in
                ISRICSoilGridsFile.SOIL_GRIDS_DEPTHS.values()
            ],
        }
        depth_strings = tuple(ISRICSoilGridsFile.SOIL_GRIDS_DEPTHS.keys())
        for k, v in raw.items():
            if k in out:
                continue
            is_stddev = k.endswith("_SD")
            if is_stddev:
                k = k.rsplit("_SD")[0]
            if not k.endswith(depth_strings):
                out[k] = v
                continue
            kp, depth_str = k.rsplit('_', 1)
            idx = depth_strings.index(depth_str)
            if is_stddev:
                out["layers"][idx]["stddev"][kp] = v
            else:
                out["layers"][idx]["values"][kp] = v
        # print(raw.keys())
        # print(out["layers"][0]["values"].keys())
        # print(out["layers"][0]["stddev"].keys())
        # import pdb; pdb.set_trace()
        return out


# TODO: There are often multiple samples for the same location on
#   different dates. Currently the most recent one is used, but this
#   data could be exposed as "time"
# TODO: The returned values for bulk density and sat conflict in apsimx
#   normalization resulting in low values of sat & dul after normalization
class SSURGOSoilFile(BaseSoilFile):
    r"""Wrapper for loading soil data from SSURGO."""

    NAME: ClassVar[str] = "ssurgo_soil_data"
    DESC: ClassVar[str] = "SSURGO"
    URL: ClassVar[str] = (
        "https://SDMDataAccess.nrcs.usda.gov/Tabular/"
        "SDMTabularService.asmx"
    )
    _default_ext: ClassVar[str] = ".xml"
    REQUEST_PARAMETERS: ClassVar[List[str]] = [
        "longitude", "latitude",
    ]
    INDEPENDENT_PARAMETERS: ClassVar[List[str]] = [
        "hzdept_r",       # topdepth,
        "hzdepb_r",       # bottomdepth,
    ]
    REQUIRED_DOWNLOAD_PARAMETERS: ClassVar[List[str]] = [
        "co.cokey",       # cokey, date of sample
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
        # "cokey": "time",
        "hzdept_r": "top_depth",
        "hzdepb_r": "bottom_depth",
        "elev_r": "elevation",
        "ph1to1h2o_r": "pH",
        "claytotal_r": "clay",
        "silttotal_r": "silt",
        "sandtotal_r": "sand",
        # rocks?
        # "dbthirdbar_r": "bulk_density",
        "dbthirdbar_h": "bulk_density",
        "om_r": "organic_matter",
        # No nitrogen
        "ksat_l": "ksat",
        "wthirdbar_r": "dul",
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
        "dbthirdbar_h": "g/cm3",
        "dbthirdbar_r": "g/cm3",
        "om_r": "%",
        # No nitrogen
        "ksat_l": "um/s",
        "wthirdbar_r": "%",
        "wfifteenbar_r": "%",
        "wsatiated_r": "%",
        # really "meq/(100*g(", but assumed equivalent
        "cec7_r": "mmol/(100*g)",
    }

    # @ParameterCalculator.register_method("bulk_density",
    #                                      "dbthirdbar_r",
    #                                      "dul")  # -> wthirdbar_r)
    # @classmethod
    # def bulk_density_from_dbthirdbar_r(cls, dbthirdbar_r: float,
    #                                    dul: float):
    #     dbthirdbar_r = cls.add_units(dbthirdbar_r, "g/cm3",
    #                                  return_scalar=True)
    #     dul = cls.add_units(dul, "%", return_scalar=True)
    #     out = cls.add_units((dbthirdbar_r - dul) / 100.0, "g/cm3")
    #     return out

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
        subset = ["cokey", "hzdepb_r"]  # cokey has date
        out = out.drop_duplicates(subset=subset).sort_values(
            subset, ascending=True)
        ix = out["hzdepb_r"].isnull()
        out = out[~ix]
        out = out[out["cokey"] == out["cokey"].values[-1]]
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

    def __add__(self, solf):
        if isinstance(solf, GeoTIFFWrapper):
            solf = solf.data
        return self.data + solf

    def __sub__(self, solf):
        if isinstance(solf, GeoTIFFWrapper):
            solf = solf.data
        return self.data - solf

    def __mul__(self, solf):
        if isinstance(solf, GeoTIFFWrapper):
            solf = solf.data
        return self.data * solf

    def __truediv__(self, solf):
        if isinstance(solf, GeoTIFFWrapper):
            solf = solf.data
        return self.data / solf

    @property
    def data(self) -> Union[np.ndarray, rj_units.QuantityArray]:
        import rasterio
        fd = rasterio.open(self.fname)
        out = rj_units.QuantityArray(
            self.scale * fd.read(self.index).astype(np.float64),
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


# class HUMERISSoilData(BaseSoilMapFile):
#     r"""Wrapper for loading soil data from the HUMERIS data set.

#     Dalle Vaglie, Matteo; Francini, Saverio; Chirici, Gherardo;
#         martellozzo, federico (2026), “HUMERIS Global Soil Dataset”,
#         Mendeley Data, V2, doi: 10.17632/z8v8m579z4.2

#     M. Dalle Vaglie, S. Francini, G. Chirici, & F. Martellozzo, A
#         large-scale framework for estimating soil carbon, nitrogen,
#         pH, and salinity dynamics for 1985–2023, Proc. Natl. Acad. Sci.
#         U.S.A. 123 (22) e2534913123,
#         https://doi.org/10.1073/pnas.2534913123 (2026).

#     Documentation: https://matteodallevaglie.com/works/humeris/

#     """
#     NAME: ClassVar[str] = "HUMERIS"
#     URL: ClassVar[str] = (
#         "https://data.mendeley.com/public-api/zip/z8v8m579z4/download/2"
#     )
#     REQUIRED_OPTIONAL_PACKAGES: ClassVar[List[str]] = ["rasterio"]
#     _default_ext: ClassVar[str] = ""
#     _EXPECTS_DIRECTORY: ClassVar[bool] = True
#     STATIC_PARAMETERS: ClassVar[dict] = {
#         # The publically released snapshot is an average over all time
#         "start_date": datetime.date(1980, 1, 1),
#         "end_date": datetime.date(2025, 12, 31),
#         "latitude": (-90.0, 90.0),
#         "longitude": (-180.0, 180.0),
#         "top_depth": np.array([0.0]),
#         "bottom_depth": np.array([30.0]),
#     }
#     PYTEST_MARKS: ClassVar[List[str]] = ["slow"]
#     HEADER_PARAMETERS: ClassVar[List[str]] = [
#         "latitude", "longitude",
#         "top_depth", "bottom_depth",
#     ]
#     PARAMETER_ALIASES: ClassVar[Mapping[str, str]] = {
#         "N": "nitrogen",
#         "ECe": "salinity",
#         "OC": "organic_carbon",
#         "pH": "pH",
#     }
#     PARAMETER_UNITS: ClassVar[Mapping[str, str]] = {
#         "top_depth": "cm",
#         "bottom_depth": "cm",
#         "N": "g/kg",
#         "ECe": "dS/m",
#         "OC": "g/kg",
#         "pH": "",
#     }

#     @classmethod
#     def _file_prefix(cls) -> str:
#         return ""

#     @readonly_cached_property
#     def calculatable_parameters(self) -> list:
#         r"""list: Parameters that can be calculated (universal names)."""
#         # This is a hack to prevent a costly calculation in tests, but
#         #   should be disabled if HUMERIS performance is improved for use
#         #   in production
#         return ["depths", "thickness"]

#     @classmethod
#     def download_and_save_data(
#             cls, fname: str, *args: Any, **kwargs: Any
#     ) -> "HUMERISSoilData":
#         r"""Download and save data for a location.

#         Args:
#             fname: Path where the data should be saved.
#             \*args, \*\*kwargs: Additional arguments are ignored.

#         Returns:
#             BaseDataFile: File instance.

#         """
#         import urllib.request
#         import zipfile
#         cls.confirm_download()
#         fname_zip = f"{fname}.zip"
#         pre_existing = os.path.isfile(fname_zip)
#         if not pre_existing:
#             urllib.request.urlretrieve(cls.URL, fname_zip)
#         with zipfile.ZipFile(fname_zip, 'r') as zip_ref:
#             zip_ref.extractall(fname)
#         if not pre_existing:
#             os.remove(fname_zip)
#         return cls(fname)

#     @classmethod
#     def _read(cls, fname: str):
#         r"""Read a data file.

#         Args:
#             fname: Path to file to read.

#         Returns:
#             object: File contents.

#         """
#         files = sorted(glob.glob(os.path.join(fname, "Mean", "*.tif")))
#         assert files
#         out = {}
#         for x in files:
#             k = os.path.splitext(os.path.basename(x))[0]
#             out[k] = GeoTIFFWrapper(
#                 x, 1, scale=0.01, units=cls.PARAMETER_UNITS[k],
#             )
#         return out

#     @readonly_cached_property
#     def _internal_parameters(self) -> list:
#         r"""list: Set of soil parameters contained by this file."""
#         return list(self.contents.keys())

#     def _get(self, name: str):
#         r"""Get a parameter from the file.

#         Args:
#             name: Parameter name.

#         Returns:
#             Parameter value.

#         Raises:
#             KeyError: If name is not a valid parameter name.

#         """
#         # TODO: Return latitude grid?
#         if name == "latitude":
#             first = self.internal_parameters()[0]
#             size = None
#             import pdb; pdb.set_trace()
#             return np.linspace(self.contents[first].bounds.bottom,
#                                self.contents[first].bounds.top,
#                                size + 1)
#             # return (self.contents[first].bounds.bottom,
#             #         self.contents[first].bounds.top)
#         elif name == "longitude":
#             first = self.internal_parameters()[0]
#             size = None
#             import pdb; pdb.set_trace()
#             return np.linspace(self.contents[first].bounds.left,
#                                self.contents[first].bounds.right,
#                                size + 1)
#             # return (self.contents[first].bounds.left,
#             #         self.contents[first].bounds.right)
#         out = self.contents[name]
#         # TODO: Figure out how to get data out of rasterio efficiently
#         # print(out)
#         # import pdb; pdb.set_trace()
#         return out

#     def _set(self, name: str, value: Any,
#              allow_add: Optional[bool] = False) -> Any:
#         r"""Set a parameter in the file.

#         Args:
#             name: Parameter name.
#             value: Parameter value.
#             allow_add: Allow for a new parameter value to be added.

#         Raises:
#             KeyError: If name is not a valid parameter name.

#         """
#         if name == "latitude":
#             if isinstance(value, np.ndarray):
#                 value = (np.min(value), np.max(value))
#             assert len(value) == 2
#             for v in self.contents.values():
#                 v.bounds.bottom = value[0]
#                 v.bounds.top = value[1]
#             return
#         elif name == "longitude":
#             if isinstance(value, np.ndarray):
#                 value = (np.min(value), np.max(value))
#             assert len(value) == 2
#             for v in self.contents.values():
#                 v.bounds.left = value[0]
#                 v.bounds.right = value[1]
#             return
#         elif name == "depths":
#             assert len(value) == 1
#             return
#         dst = self.contents[name]
#         print(dst)
#         raise NotImplementedError(
#             "Figure out how to set channels w/ rasterio")


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
    REQUEST_PARAMETERS: ClassVar[List[str]] = [
        "longitude", "latitude",
        "start_date", "end_date",
    ]
    INDEPENDENT_PARAMETERS: ClassVar[List[str]] = []
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
        if name in self.HEADER_PARAMETERS + self.SCALAR_PARAMETERS:
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
        if name in self.HEADER_PARAMETERS + self.SCALAR_PARAMETERS:
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
    INDEPENDENT_PARAMETERS: ClassVar[List[str]] = [
        "Date",
    ]
    STATIC_PARAMETERS: ClassVar[dict] = {
        "top_depth": np.array([0.0]),
        "bottom_depth": np.array([100.0]),  # cm, Root zone
        # "bottom_depth": np.array([5.0]),  # cm, Surface
    }
    PARAMETER_ALIASES: ClassVar[Mapping[str, str]] = {
        "Date": "time",
        "sm_rootzone": "moisture",
        "surface_temp": "temperature",
        "leaf_area_index": "lai",
    }
    PARAMETER_UNITS: ClassVar[Mapping[str, str]] = {
        "top_depth": "cm",
        "bottom_depth": "cm",
        ##
        "land_evapotranspiration_flux": "kg/m^2/s",
        "land_fraction_saturated": "fraction",
        "land_fraction_wilting": "fraction",
        "leaf_area_index": "m**2/m**2",
        "mwrtm_vegopacity": "",
        "sm_profile": "m**3/m**3",
        "sm_profile_pctl": "%",
        "sm_profile_wetness": "m**3/m**3",
        ##
        "sm_rootzone": "m**3/m**3",
        ##
        "sm_rootzone_pctl": "%",
        "sm_rootzone_wetness": "m**3/m**3",
        "sm_surface": "m**3/m**3",
        "sm_surface_pctl": "%",
        "sm_surface_wetness": "fraction",
        ##
        "soil_water_infiltration_flux": "kg/m^2/s",
        "vegetation_greenness_fraction": "fraction",
        "surface_temp": "K",
        # "soil_temp_layer1": "K",
        # "soil_temp_layer2": "K",
        # "soil_temp_layer3": "K",
        # "soil_temp_layer4": "K",
        # "soil_temp_layer5": "K",
        # "soil_temp_layer6": "K",
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
    REQUEST_PARAMETERS: ClassVar[List[str]] = []
    INDEPENDENT_PARAMETERS: ClassVar[List[str]] = []
    STATIC_PARAMETERS: ClassVar[dict] = {}
    PARAMETER_LIMITS: ClassVar[dict] = {}
    PARAMETER_ALIASES: ClassVar[Mapping[str, str]] = {}
    PARAMETER_ALIASES_REV: ClassVar[Mapping[str, str]] = {}
    PARAMETER_ALIASES_REV_ALT: ClassVar[Mapping[str, List[str]]] = {}
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
            for k in ["REQUEST_PARAMETERS", "INDEPENDENT_PARAMETERS",
                      "PARAMETER_ALIASES", "PARAMETER_UNITS",
                      "PARAMETER_LIMITS", "HEADER_PARAMETERS",
                      "SCALAR_PARAMETERS", "STATIC_PARAMETERS"]:
                if getattr(cls, k):
                    continue
                setattr(cls, k, copy.deepcopy(getattr(appeears_cls, k)))
        BaseSoilFile._on_registration(cls)

    def collect_parameters(self):
        r"""Collect parameters from each map for the selected location."""
        if "parameters" in self.contents:
            if "time" in self.contents["parameters"]:
                self.contents["parameters"]["Date"] = (
                    self.contents["parameters"].pop("time"))
                self.write(overwrite=True)
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
                    (lat_grid - self.latitude.value) ** 2
                    + (lon_grid - self.longitude.value) ** 2
                )
                x, y = np.unravel_index(np.argmin(dist), dist.shape)
                self.contents["parameters"].setdefault("Date", [])
                self.contents["parameters"]["Date"].append(time)
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
        if name in self.HEADER_PARAMETERS + self.SCALAR_PARAMETERS:
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
        if name in self.HEADER_PARAMETERS + self.SCALAR_PARAMETERS:
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
