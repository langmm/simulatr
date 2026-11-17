import os
import re
import glob
import copy
import time
import zmq
import platform
import msgpack
import subprocess
import contextlib
import datetime
import json
import pprint
from functools import cached_property
from collections import OrderedDict
import numpy as np
from typing import (
    Optional, Union, Any, List, Tuple, Callable, Iterator, ClassVar,
    Mapping
)
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema
import yggdrasil_rapidjson as yggrj
from . import logger, utils
from .utils import cfg, LogPipe, readonly_cached_property
from .base import (
    RecoverableError, ModelEngineError, InvalidActionError,
    RecoverableModelEngineError,
    SimulatorNotInstalled,
    ModelOperation,
)
from .data import (
    ParameterCalculator,
    ParameterNormalizer,
    BaseWeatherFile, BaseSoilFile,
    NASAPOWERWeatherFile,
    # SoilMatFile,
    ISRICSoilGridsFile,
)
from .crop import (
    CropModelFile,
    CropModelEngine, CropModelLLMPromptGenerator, CropModelEnv
)


def _read_resource(name: str) -> dict:
    r"""Read a resource file from the APSIMX resources directory.

    Args:
        name: Name of the resource to read.

    Returns:
        dict: Resource node matching the specified name.

    """
    try:
        resource = ApsimXFileNode.from_resource(name, required=True)
        if resource is not None:
            return resource.contents
    except SimulatorNotInstalled:
        pass
    return {}


def _replace_root(x: str) -> str:
    r"""Replace the %root% placeholder in a file path.

    Args:
        x: Path to make replacement in.

    Returns:
        str: Modified path.

    """
    root = cfg['directories']['apsimx']
    if platform.system() == "Windows":
        from pathlib import PureWindowsPath
        root = str(PureWindowsPath(root).as_posix())
    return x.replace("%root%", root)


class ApsimXFileNode:
    r"""Container for node in ApsimXFile.

    Args:
        contents: Contents of the node.
        parent: Parent node.
        **kwargs: Additional keywords are added directly to the node.

    """

    def __init__(self, contents: dict,
                 parent: Optional["ApsimXFileNode"] = None,
                 **kwargs: Any) -> None:
        r"""Initialize a new node.

        Args:
            contents: Contents of the node.
            parent: Parent node.
            **kwargs: Additional keywords are added directly to the
                node.

        """
        self.contents = contents
        self.contents.update(**kwargs)
        self.parent = parent
        if "Children" in self.contents:
            for i, x in enumerate(self.contents["Children"]):
                if isinstance(x, ApsimXFileNode):
                    self.contents["Children"][i] = x.contents

    @classmethod
    def from_param(cls, node_type: str, **kwargs: Any) -> "ApsimXFileNode":
        r"""Create a new node from the provided parameters.

        Args:
            node_type: Node type.
            **kwargs: Additional keyword arguments are passed to the
                class constructor.

        Returns:
            ApsimXFileNode: New node.

        """
        contents = {
            "$type": node_type,
            "Name": kwargs.pop(
                "Name",
                node_type.split(",", 1)[0].rsplit(".", 1)[-1].strip()
            ),
            "ResourceName": None,
            "Children": [],
            "Enabled": True,
            "ReadOnly": False,
        }
        return cls(contents, **kwargs)

    @classmethod
    def from_file(cls, fname: str, **kwargs: Any) -> "ApsimXFileNode":
        r"""Create a new node by loading code from the provided JSON
        file.

        Args:
            fname: Full path to file.
            **kwargs: Additional keyword arguments are passed to the
                class constructor.

        Returns:
            ApsimXFileNode: New node.

        """
        with open(fname, 'r') as fd:
            contents = json.load(fd)
        return cls(contents, **kwargs)

    @classmethod
    def from_data(cls, name: str, crop_name: Optional[str] = None,
                  crop_root: Optional["ApsimXFileNode"] = None,
                  **kwargs: Any) -> "ApsimXFileNode":
        r"""Create a new node by loading code from a data file.

        Args:
            name: Name of the data file resource.
            crop_root: Example crop root node that should be used to
                populate default values for the created node if there
                is an equivalent node for the crop.
            crop_name: Crop name. If provided, the examples directory
                will be checked and if an example exists, the
                corresponding module will be used from that file.
            **kwargs: Additional keyword arguments are passed to the
                class constructor.

        Returns:
            ApsimXFileNode: New node.

        """
        fname = os.path.join(ApsimXEngine.data_dir(), f"{name}.json")
        out = cls.from_file(fname, **kwargs)
        if crop_name is not None:
            try:
                crop_root = cls.from_example(crop_name.title())
            except BaseException:
                crop_root = None
        if crop_root is None:
            return out
        out.copy_contents(crop_root)
        return out

    def copy_contents(self, other, excluded=None):
        if excluded is None:
            excluded = []
        node = other.find(self["Name"])
        parameters = list(self.parameter_keys())
        if node is None:
            requirements = {"$type": self["$type"]}
            checked = {}
            nodes = [x for x in other.findall(
                requirements=requirements)]
            if len(nodes) == 1:
                node = nodes[0]
            elif parameters:
                node = max(
                    nodes, key=lambda x: len([k for k in parameters
                                              if x.has_parameter(k)]))
            else:
                checked = {x["Name"]: x["$type"] for x in nodes}
            if node is None:
                # TODO: Special case of operations
                logger.warning(f"Could not find node that matches "
                               f"{requirements}. "
                               f"Checked:\n"
                               f"{pprint.pformat(checked)}")
        if node is None:
            return
        excluded_base = ["Parameters", "Name", "$type", "CodeArray",
                         "Children"]
        for k, v in node.contents.items():
            if k in excluded + excluded_base:
                continue
            self.contents[k] = copy.deepcopy(v)
        if "Parameters" not in excluded and parameters:
            for k in parameters:
                if not node.has_parameter(k):
                    continue
                self.set_parameter(k, copy.deepcopy(node.get_parameter(k)))
        if "Children" not in excluded:
            for child in self.children:
                child.copy_contents(other, excluded=excluded)

    @classmethod
    def from_resource(cls, name: str, **kwargs: Any) -> "ApsimXFileNode":
        r"""Create a new node by loading code from an ApsimX resource
        file.

        Args:
            name: Name of the resource file.
            **kwargs: Additional keyword arguments are passed to the
                class constructor.

        Returns:
            ApsimXFileNode: New node.

        """
        if not os.path.isdir(cfg['directories']['apsimx']):
            raise SimulatorNotInstalled(cfg['directories']['apsimx'])
        fname = os.path.join(
            cfg['directories']['apsimx'],
            "Models", "Resources",
            f"{name}.json")
        return cls.from_file(fname, **kwargs)

    @classmethod
    def from_example(cls, name: str, **kwargs: Any) -> "ApsimXFileNode":
        r"""Create a new node by loading code from an ApsimX example.

        Args:
            name: Name of the example file.
            **kwargs: Additional keyword arguments are passed to the
                class constructor.

        Returns:
            ApsimXFileNode: New node.

        """
        if not os.path.isdir(cfg['directories']['apsimx']):
            raise SimulatorNotInstalled(cfg['directories']['apsimx'])
        fname = os.path.join(
            cfg['directories']['apsimx'], "Examples",
            f"{name}.apsimx")
        return cls.from_file(fname, **kwargs)

    def __str__(self) -> str:
        r"""str: A string representation of this node."""
        return f"ApsimXFileNode({self.absolute_path})"

    # @cached_property
    # def state_variables(self) -> List[str]:
    #     r"""list: State variable names."""
    #     _regex_model = r'^\[(?P<model>\w+)\].*'
    #     out = []
    #     for x in self.children:
    #         for sv in x.state_variables:
    #             if sv.startswith("["):
    #                 out.append(sv)
    #             else:
    #                 out.append(
    #         out += x.state_variables
    #     return out

    @cached_property
    def root(self) -> "ApsimXFileNode":
        r"""Root node"""
        if self.parent is None:
            return self
        return self.parent.root

    @cached_property
    def absolute_path(self) -> str:
        r"""str: Absolute path to the node from the root node."""
        if self.parent is None:
            return "[" + self["Name"] + "]"
        return self.parent.absolute_path + "." + self["Name"]

    @property
    def children(self) -> Iterator["ApsimXFileNode"]:
        r"""list: Child nodes."""
        for x in self.contents.get("Children", []):
            yield ApsimXFileNode(x, parent=self)

    def __contains__(self, name: str) -> bool:
        r"""Check if the node contains the named element."""
        return name in self.contents

    def __getitem__(self, name: str) -> Any:
        r"""Get the value of the named element."""
        return self.contents[name]

    def __setitem__(self, name: str, value: Any) -> None:
        r"""Set the value of the named element."""
        self.contents[name] = value

    def __delitem__(self, name: str) -> None:
        r"""Remove the named element from the node."""
        del self.contents[name]

    def get(self, name: str, default: Any) -> Any:
        r"""Get the value of the named element from the node.

        Args:
            name: Name of the element.
            default: Value returned if the element is not present.

        Returns:
            object: Value of the named element.

        """
        return self.contents.get(name, default)

    def keys(self) -> Iterator[str]:
        r"""Get the keys in the node."""
        for x in self.contents.keys():
            yield x

    def values(self) -> Iterator[Any]:
        r"""Get the values in the node."""
        for x in self.contents.values():
            yield x

    def items(self) -> Iterator[Any]:
        r"""Get the items in the node."""
        for x in self.contents.items():
            yield x

    def parameters(self) -> Iterator[Any]:
        r"""Get the parameters in the node."""
        for x in self.contents.get("Parameters", []):
            yield (x["Key"], x["Value"])

    def parameter_keys(self) -> Iterator[Any]:
        r"""Get the parameter names in the node."""
        for k, v in self.parameters():
            yield k

    def parameter_values(self) -> Iterator[Any]:
        r"""Get the parameter values in the node."""
        for k, v in self.parameters():
            yield v

    def specialize_crop(self, crop_name: str,
                        parameter_name: str = "Crop",
                        placeholder: str = "CROP") -> None:
        r"""Specialize the crop referenced by the node.

        Args:
            crop_name: Name of crop to specialize.
            parameter_name: Parameter name where the crop name is
                stored.
            placeholder: Placeholder to replace.

        """
        if self.has_parameter(parameter_name):
            if placeholder not in self["Name"]:
                prev = self.get_parameter(parameter_name)
                self["Name"] = self["Name"].replace(prev, crop_name)
            self.set_parameter(parameter_name, crop_name)
        if placeholder in self["Name"]:
            self["Name"] = self["Name"].replace(placeholder, crop_name)
        for x in self.children:
            x.specialize_crop(crop_name, parameter_name=parameter_name,
                              placeholder=placeholder)

    def has_parameter(self, name: str | set) -> bool:
        r"""Check if the node has a parameter of a given name.

        Args:
            name: Parameter name.

        Returns:
            bool: True if the parameter is present, False otherwise.

        """
        if "Parameters" not in self:
            return False
        for x in self["Parameters"]:
            if isinstance(name, set) and x["Key"] in name:
                return True
            elif x["Key"] == name:
                return True
        return False

    def get_parameter(self, name: str) -> Any:
        r"""Get a node parameter value.

        Args:
            name: Parameter name.

        Returns:
            Parameter value.

        Raises:
            KeyError: If name is not a valid parameter name.


        """
        if "Parameters" not in self:
            raise KeyError(f"No parameters in {self}")
        for x in self["Parameters"]:
            if isinstance(name, set) and x["Key"] in name:
                return x["Value"]
            elif x["Key"] == name:
                return x["Value"]
        raise KeyError(f"No parameter named \"{name}\" in {self}")

    def set_parameter(self, name: str, value: Any) -> None:
        r"""Set a node parameter.

        Args:
            name: Parameter name.
            value: Parameter value.

        Raises:
            KeyError: If name is not a valid parameter name.

        """
        if "Parameters" not in self:
            raise KeyError(f"No parameters in {self}")
        for x in self["Parameters"]:
            if isinstance(name, set) and x["Key"] in name:
                x["Value"] = value
                return
            elif x["Key"] == name:
                x["Value"] = value
                return
        raise KeyError(f"No parameter named \"{name}\" in {self}")

    def findall(self, name: Optional[str] = None,
                requirements: Optional[dict] = None,
                ) -> Iterator["ApsimXFileNode"]:
        r"""Find a node in the file.

        Args:
            name: Name of the node to find.
            requirements: Set of requirements that the node must
                satisfy (see node_matches for a description of the
                available options).

        Yields:
            ApsimXFileNode: All nodes matching the specified name.

        """
        requirements = requirements or {}
        assert name is not None or requirements
        if self.matches(name=name, **requirements):
            yield self
        for x in self.children:
            for out in x.findall(name=name, requirements=requirements):
                yield out

    def find(self, name: Optional[str] = None,
             required: Optional[bool] = False,
             requirements: Optional[dict] = None) -> "ApsimXFileNode":
        r"""Find a node in the file.

        Args:
            name: Name of the node to find.
            required: If True, an error will be raised if the node
                cannot be located.
            requirements: Set of requirements that the node must
                satisfy (see node_matches for a description of the
                available options).

        Returns:
            ApsimXFileNode: The node matching the specified name.
                Empty if no node can be found.

        Raises:
            KeyError: If required is True and the node cannot be located.

        """
        requirements = requirements or {}
        assert name is not None or requirements
        if self.matches(name=name, **requirements):
            return self
        for x in self.children:
            out = x.find(name=name, requirements=requirements)
            if out is not None:
                return out
        if required:
            msg = ""
            if name is not None:
                msg += f" with \"Name\" {name}"
            if requirements:
                msg += f" matching requirements {requirements}"
            raise KeyError(f"Could not locate a node{msg}")
        return None

    def matches(self,
                errors: Optional[list] = None,
                name: Optional[str] = None,
                field: Optional[str] = None,
                parameter: Optional[str] = None,
                internal: Optional[str] = None,
                contains: Optional[Union[list, set, dict]] = None,
                equals: Optional[Any] = None,
                fvalid: Optional[Callable] = None,
                calls: Optional[str] = None,
                anyOf: Optional[list] = None,
                nested: Optional[dict] = None,
                **kwargs: Any) -> bool:
        r"""Check if a node matches the specified requirements.

        Args:
            errors: If a list is provided, errors will be added to this
                list.
            name: Name that the node must have.
            field: Name of a field that must be present.
            parameter: Name of a parameter that must be present.
            contains: Fields/elements that the node must contain. If a
                set is provided, only one of the elements must be
                present. If a dict is provided, the values in the node
                must match the values in the provided dict.
            equals: Value that the node must be equivalent to.
            fvalid: Function that returns True if the node is valid, and
                False otherwise.
            calls: Name of a function called in the node code block.
            anyOf: List of kwargs for node_matches that should be
                checked. If the node satisfies any of these requirements,
                True will be returned.
            nested: Set of requirements for individual fields.
            $type: Type that the node should have.
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            bool: True if the node matches, False otherwise.

        """
        ntype = kwargs.pop("$type", None)

        def add_error(msg: str) -> bool:
            r"""Add an error message to the errors list.

            Args:
                msg: Error message to add.

            Returns:
                bool: True if the error was added, False otherwise.

            """
            if not isinstance(errors, list):
                return False
            errors.append(msg)
            return True

        if anyOf:
            if isinstance(errors, list):
                xerrors = []
                if not any(self.matches(errors=xerrors, **x)
                           for x in anyOf):
                    errors += xerrors
            else:
                if not any(self.matches(**x) for x in anyOf):
                    return False
        if name is not None and self.get("Name", None) != name:
            if not add_error(f"{self} name is not {name}"):
                return False
        if ntype is not None and self.get("$type", None) != ntype:
            if not add_error(f"{self} $type is not {ntype}"):
                return False
        if field is not None and field not in self:
            if not add_error(f"{self} is missing field \"{field}\""):
                return False
        if ((parameter is not None
             and not self.has_parameter(parameter))):
            if not add_error(f"{self} is missing parameter \"{parameter}\""):
                return False
        if internal:
            if not self.matches(
                    **ApsimXEngine.get_field_metadata(internal)):
                return False
        if contains:
            if isinstance(contains, str):
                contains = [contains]
            if isinstance(contains, set):
                missing = (
                    list(contains)
                    if not any(k in self for k in contains)
                    else []
                )
            elif isinstance(contains, dict):
                missing = [k for k in contains.keys() if k not in self]
            else:
                assert isinstance(contains, list)
                missing = [k for k in contains if k not in self]
            if missing and not add_error(f"Missing {missing}"):
                return False
            if isinstance(contains, dict):
                for k, v in contains.items():
                    if k not in self:
                        continue
                    if self[k] != v and not add_error(
                            f"{k}: {self[k]} != {v}"):
                        return False
        if equals and self.contents != equals:
            if not add_error(f"{self.contents} != {equals}"):
                return False
        if fvalid and not fvalid(self.contents):
            if not add_error(f"{self} fails function {fvalid}"):
                return False
        if ((calls and not (
                isinstance(self.contents, dict)
                and any(f"{calls}(" in x
                        for x in self.get("CodeArray", []))))):
            if not add_error(f"{self} does not call \"{calls}\""):
                return False
        if nested:
            missing = [k for k in nested.keys() if k not in self]
            if missing and not add_error(f"Missing {missing}"):
                return False
            for k, v in nested.items():
                if k not in self:
                    continue
                knode = ApsimXFileNode(self[k])
                if isinstance(errors, list):
                    verrors = []
                    knode.matches(errors=verrors, **v)
                    errors += [f"{k}: {x}" for x in verrors]
                elif not knode.matches(**v):
                    return False
        if errors:
            return False
        return True


class ApsimXWeatherFile(BaseWeatherFile):
    r"""Container for ApsimX weather data."""

    NAME: ClassVar[str] = "apsimx_met"
    DESC: ClassVar[str] = "ApsimX Weather"
    DEFAULT_EXTERNAL_TYPE: ClassVar[type] = NASAPOWERWeatherFile
    _default_ext: ClassVar[str] = ".met"
    REQUIRED_PARAMETERS: ClassVar[List[str]] = [
        "longitude", "latitude", "elevation", "tav",
        "year",
        "day",
        "radn",
        "maxt",
        "mint",
        "rain",
        "vp",
    ]
    SORTED_PARAMETERS: ClassVar[List[str]] = [
        "year",
        "day",
        "radn",
        "maxt",
        "mint",
        "rain",
        "vp",
    ]
    INDEPENDENT_PARAMETERS: ClassVar[List[str]] = [
        "year", "day",
    ]
    HEADER_PARAMETERS: ClassVar[List[str]] = [
        "latitude", "longitude", "elevation", "tav",
    ]
    PARAMETER_UNITS: ClassVar[Mapping[str, str]] = {
        "radn": "MJ/m^2/day",
        "maxt": "oC",
        "mint": "oC",
        "rain": "mm",
        "vp": "hPa",
        "tav": "oC",
        "amp": "oC",
        "latitude": "decimal degrees",
        "longitude": "decimal degrees",
        "elevation": "m",
    }
    PARAMETER_ALIASES: ClassVar[Mapping[str, str]] = {
        "radn": "radiation_surface",
        "vp": "vapor_pressure",
        "rain": "precipitation",
        "tav": "temperature_avg",
        "amp": "temperature_amp",
        "maxt": "temperature_max",
        "mint": "temperature_min",
    }
    UNIT_ALIASES: ClassVar[Mapping[str, str]] = {
        "oC": "degC",
        "decimal degrees": "degrees",
    }

    @classmethod
    def empty_contents(cls):
        r"""Create empty contents for a file wrapper instance."""
        import pandas as pd
        return {
            "constants": {},
            "columns": pd.DataFrame(),
            "units": {},
        }

    @ParameterCalculator.register_method("time", "year", "day")
    @classmethod
    def time_from_year_and_day(cls, year: float, day: float) -> float:
        return ((year - 1970).astype("datetime64[Y]")
                + (day - 1).astype("timedelta64[D]"))

    @ParameterCalculator.register_method("year", "time")
    @classmethod
    def year_from_time(cls, time: float) -> float:
        # return time.astype("datetime64[Y]").astype(int) + 1970
        import pandas as pd
        out = pd.DatetimeIndex(time).year.to_numpy()
        return out.astype('int64')

    @ParameterCalculator.register_method("day", "time")
    @classmethod
    def day_from_time(cls, time: float) -> float:
        import pandas as pd
        out = pd.DatetimeIndex(time).day_of_year.to_numpy()
        return out.astype('int64')

    @classmethod
    def independent_parameters(cls):
        r"""list: Independent parameters."""
        # TODO: Verify that time interpolation is handled correctly
        param = cls.INDEPENDENT_PARAMETERS.copy()
        param.remove("day")
        return param

    @readonly_cached_property
    def _internal_parameters(self) -> list:
        r"""list: Set of power parameters contained by this file."""
        return (
            list(self.contents["constants"].keys())
            + list(self.contents["columns"].columns)
        )

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
            return self.contents["constants"][name]
        return np.array(self.contents["columns"][name])

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
            if name not in self.contents["constants"]:
                if not allow_add:
                    raise KeyError(name)
                if name in self.PARAMETER_UNITS:
                    self.contents["units"][name] = (
                        self.PARAMETER_UNITS[name])
            self.contents["constants"][name] = value
            return
        if name not in self.contents["columns"]:
            if not allow_add:
                raise KeyError(name)
            if name in self.PARAMETER_UNITS:
                self.contents["units"][name] = self.PARAMETER_UNITS[name]
        self.contents["columns"][name] = value

    @classmethod
    def _read(cls, fname: str):
        r"""Read a model input file.

        Args:
            fname: Path to file to read.

        Returns:
            object: File contents.

        """
        import pandas as pd
        out = {
            "constants": {},
            "units": {},
        }
        with open(fname, "r") as fd:
            for line in fd:
                if line.startswith("[weather.met.weather]"):
                    break
            for line in fd:
                if line.startswith("!"):
                    continue
                elif line.lstrip().startswith("year"):
                    names = line.strip().split()
                    for k, x in zip(names, fd.readline().strip().split()):
                        # out["units"][k] = cls.UNIT_ALIASES.get(
                        #     x.strip("()"))
                        x_units = x.strip("()")
                        if x_units:
                            out["units"][k] = x.strip("()")
                    out["columns"] = pd.read_csv(
                        fd, sep=r"\s+", names=names,
                    )
                else:
                    pattern = (
                        r"(?P<name>\w+)\s+\=\s+(?P<value>[+-]?\d+(\.\d+)?)\s+"
                        r"\((?P<units>(\w[\w\/\^ ]*)?)\)"
                    )
                    match = re.search(pattern, line)
                    if not match:
                        raise ValueError(f"Failed to parse .met line: "
                                         f"\"{line}\"")
                    match = match.groupdict()
                    if match["units"]:
                        out["units"][match["name"]] = match["units"]
                    out["constants"][match["name"]] = float(match["value"])
        return out

    @classmethod
    def _write(cls, fname: str, contents):
        r"""Read a model input file.

        Args:
            fname: Path to file to read.
            contents: File contents to write.

        """
        out = ["[weather.met.weather]"]
        if "constants" in contents:
            for k in cls.HEADER_PARAMETERS:
                if k not in contents["constants"]:
                    continue
                v = contents["constants"][k]
                out.append(f"{k} = {v} ({contents['units'].get(k, '')})")
        column_order = ["year", "day"]
        column_order += [k for k in contents["columns"].columns
                         if k not in column_order]
        units = {
            k: f"({contents['units'].get(k, '')})"
            for k in column_order
        }
        col_space = {k: max(len(k), len(v)) for k, v in units.items()}
        for k in column_order:
            v = units[k]
            pad = (col_space[k] - len(v)) * " "
            units[k] += pad
        head, body = contents["columns"].to_string(
            index=False, col_space=col_space, columns=column_order,
        ).split("\n", maxsplit=1)
        out.append(head)
        out.append(" " + " ".join(list(units.values())))
        out.append(body)
        with open(fname, "w") as fd:
            fd.write("\n".join(out))

    def update_param(self, contents: Any) -> None:
        r"""Merge downloaded parameters into the current data.

        Args:
            contents: New data to incorporate.

        """
        # TODO: Currently untested and dataframe in columns is just
        #   replaced rather than updated
        for k in ["units", "contents", "columns"]:
            self.contents[k].update(contents[k])


class ApsimXSoilFile(BaseSoilFile):
    r"""Container for ApsimX soil data.
    (e.g. ``simulatr/apsimx_data/Soil.json``)
    """

    NAME: ClassVar[str] = "apsimx_soil"
    DESC: ClassVar[str] = "ApsimX Soil"
    # TODO: Fix calculated Ks when using SoilMat is invalid (not >0)
    # DEFAULT_EXTERNAL_TYPE: ClassVar[type] = SoilMatFile
    DEFAULT_EXTERNAL_TYPE: ClassVar[type] = ISRICSoilGridsFile
    _default_ext: ClassVar[str] = ".json"
    INDEPENDENT_PARAMETERS: ClassVar[List[str]] = [
        "Thickness",
    ]
    HEADER_PARAMETERS: ClassVar[List[str]] = [
        "Latitude", "Longitude", "SoilType",
    ]
    PARAMETER_ALIASES: ClassVar[Mapping[str, str]] = {
        "Latitude": "latitude",
        "Longitude": "longitude",
        "SoilType": "texture_class",
        "Thickness": "thickness",
        "ParticleSizeSand": "sand",
        "ParticleSizeSilt": "silt",
        "ParticleSizeClay": "clay",
        "Rocks": "rocks",
        "BD": "bulk_density",
        # "AirDry": "",
        "LL15": "wilting_point",
        "DUL": "dul",
        "SAT": "sat",
        "KS": "ksat",
        # "LL": "",
        # "KL": "",
        # "XF": "",
        # "SWCON": "",
        "Carbon": "organic_carbon",
        "SoilCNRatio": "cn_ratio",
        # "FBiom": "",
        # "FInert": "",
        # "FOM": "",
        "PH": "pH",
        "CEC": "cec",
        "NO3": "no3",
        "NH4": "nh4",
        "Urea": "urea",
        "Water": "moisture",  # Current water level
        "Temperature": "temperature",
    }
    PARAMETER_UNITS: ClassVar[Mapping[str, str]] = {
        "Thickness": "mm",
        "ParticleSizeSand": "%",
        "ParticleSizeSilt": "%",
        "ParticleSizeClay": "%",
        "Rocks": "fraction",
        "BD": "g/cm3",
        "AirDry": "",
        "LL15": "mm**3/mm**3",
        "DUL": "mm**3/mm**3",
        "SAT": "mm**3/mm**3",
        "KS": "mm/day",
        "LL": "",
        "KL": "",
        "XF": "",
        "SWCON": "",
        "Carbon": "%",
        "FOMCNRatio": "",
        "SoilCNRatio": "",
        "FBiom": "",
        "FInert": "",
        "FOM": "",
        "PH": "",  # 1:5 water
        "CEC": "cmol/kg",
        "NO3": "0.000001",  # "ppm",
        "NH4": "0.000001",  # "ppm",
        "Urea": "kg/ha",
        "Water": "mm**3/mm**3",
        "Temperature": "decC",
    }
    SCALAR_PARAMETERS: ClassVar[List[str]] = [
        "FOMCNRatio",
    ]
    _layer_nodes: ClassVar[OrderedDict] = OrderedDict([
        # ("SoilPhysical", [
        ("Models.Soils.Physical, Models", [
            "ParticleSizeSand", "ParticleSizeSilt", "ParticleSizeClay",
            "Rocks", "BD", "AirDry", "LL15", "DUL", "SAT", "KS",
            "Thickness",
        ]),
        # ("SoilCrop", [  # No thickness
        ("Models.Soils.SoilCrop, Models", [
            # These are crop specific
            "LL", "KL", "XF",
        ]),
        # ("SoilWaterBalance", [
        ("Models.WaterModel.WaterBalance, Models", [
            "SWCON",
            "Thickness",
        ]),
        # ("SoilOrganic", [
        ("Models.Soils.Organic, Models", [
            "Carbon",
            "FOMCNRatio",
            "SoilCNRatio",
            "FBiom",
            "FInert",
            "FOM",
            "Thickness",
        ]),
        # ("SoilChemical", [
        ("Models.Soils.Chemical, Models", [
            "PH", "CEC",
            "Thickness",
        ]),
        ("Models.Soils.Water, Models", [
            "Water",
            # "InitialValues",
            "Thickness",
        ]),
        ("Models.Soils.Nutrients.Nutrient, Models", []),
        # ("SoilSolute", [
        ("Models.Soils.Solute, Models", [
            "NO3", "NH4", "Urea",
            # "InitialValues",
            "Thickness",
        ]),
        ("Models.Soils.SoilTemp.SoilTemperature, Models", [
            "Temperature",
            "Thickness",
        ]),
        ("Models.Soils.CERESSoilTemperature, Models", []),
        ("Models.Soils.InitialWater, Models", [
            "Water",  # TODO: Determine how to set FractionFull from this
        ]),
        ("Models.Soils.Sample, Models", [
            # TODO: There are others "SW",
            "Thickness", "NO3", "NH4",
        ]),
    ])
    _alternate_nodes: ClassVar[Mapping[str, str]] = {
        "Models.Soils.InitialWater, Models": "Models.Soils.Water, Models",
        "Models.Soils.Sample, Models": "Models.Soils.Solute, Models",
    }
    _initial_values_nodes: ClassVar[List[str]] = [
        "Models.Soils.Solute, Models",
        "Models.Soils.Water, Models",
        "Models.Soils.SoilTemp.SoilTemperature, Models",
    ]
    _curveparam_a: ClassVar[float] = 0.0
    _curveparam_b: ClassVar[float] = 0.2

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
                         fname_orig=fname_orig, depths=depths,
                         thickness=thickness)
        if self.exists or contents:
            self.calculate_missing()

    @classmethod
    def from_file_and_prev(cls, fname, prev):
        out = cls.from_file(fname)
        crop_name = prev.crop_name
        try:
            crop_root = ApsimXFileNode.from_example(crop_name)
            crop_soil = crop_root.find(
                requirements={"$type": "Models.Soils.Soil, Models"})
        except BaseException:
            crop_soil = None
        if crop_soil is None:
            out.root.specialize_crop(crop_name)
            return out.contents
        crop_soil = cls(None, contents=crop_soil.contents)
        params = out.get_universal_parameter_map(
            type(out), keep_units=True)
        crop_soil.resample(thickness=params["Thickness"])
        for k, v in params.items():
            crop_soil.set(k, v, allow_add=True)
        crop_soil.normalize_parameters()
        return crop_soil.contents

    @cached_property
    def root(self) -> ApsimXFileNode:
        r"""Root soil node."""
        return ApsimXFileNode(self.contents)

    @property
    def crop_name(self) -> str:
        r"""Crop name."""
        return self.crop_soil["Name"].rsplit("Soil", 1)[0]

    @cached_property
    def physical(self) -> ApsimXFileNode:
        return self._child("Models.Soils.Physical, Models")

    @cached_property
    def crop_soil(self) -> ApsimXFileNode:
        return self._child("Models.Soils.SoilCrop, Models",
                           root=self.physical)

    @cached_property
    def water_balance(self) -> ApsimXFileNode:
        return self._child("Models.WaterModel.WaterBalance, Models")

    @cached_property
    def organic(self) -> ApsimXFileNode:
        return self._child("Models.Soils.Organic, Models")

    @cached_property
    def chemical(self) -> ApsimXFileNode:
        return self._child("Models.Soils.Chemical, Models")

    @cached_property
    def water(self) -> ApsimXFileNode:
        if self.uses_initial_water:
            return self._child("Models.Soils.InitialWater, Models")
        return self._child("Models.Soils.Water, Models")

    @cached_property
    def nutrient(self) -> ApsimXFileNode:
        return self._child("Models.Soils.Nutrients.Nutrient, Models")

    @cached_property
    def solutes(self) -> ApsimXFileNode:
        if self.uses_soil_samples:
            return self.root._child("Models.Soils.Sample, Models")
        # This one will match multiple
        return {x.contents["Name"]: x for x in self.root.findall(
            requirements={"$type": "Models.Soils.Solute, Models"},
        )}

    @cached_property
    def uses_soil_samples(self) -> bool:
        r"""bool: True if the file uses soil samples instead of solutes."""
        return self.has_child("Models.Soils.Sample, Models")

    @cached_property
    def uses_initial_water(self) -> bool:
        r"""bool: True if the file uses InitialWater instead of Water."""
        return self.has_child("Models.Soils.InitialWater, Models")

    @cached_property
    def temperature(self) -> ApsimXFileNode:
        if self.has_child("Models.Soils.CERESSoilTemperature, Models"):
            return self._child(
                "Models.Soils.CERESSoilTemperature, Models")
        return self._child("Models.Soils.SoilTemp.SoilTemperature, Models")

    def has_child(self, node_type: str,
                  root: Optional[ApsimXFileNode] = None, **kwargs) -> bool:
        try:
            self._child(node_type, root=root, **kwargs)
            return True
        except KeyError:
            return False

    def _child(self, node_type: str,
               root: Optional[ApsimXFileNode] = None, **kwargs) -> dict:
        r"""Get the child node of the specified type.

        Args:
            node_type: Node type name.
            root: Node to start looking from.
            \*\*kwargs: Additional keyword arguments are treated as
                additional node requirements.

        Returns:
            dict: Child node contents.

        """
        root = (root or self.root)
        kwargs["$type"] = node_type
        name = kwargs.pop("Name", None)
        return root.find(
            name=name,
            required=True,
            requirements=kwargs,
        )

    @classmethod
    def empty_contents(cls):
        r"""Create empty contents for a file wrapper instance."""
        return ApsimXFileNode.from_data(
            "Soil",
            DataSource=None,
            Comments=None,
            Children=[],
            # Children=[
            #     ApsimXFileNode.from_data(
            #         "SoilPhysical",
            #         Children=[
            #             ApsimXFileNode.from_data(
            #                 "SoilCrop",
            #             ).contents,
            #         ],
            #     ).contents,
            #     ApsimXFileNode.from_data("SoilWaterBalance").contents,
            #     ApsimXFileNode.from_data("SoilOrganic").contents,
            #     ApsimXFileNode.from_data("SoilChemical").contents,
            #     ApsimXFileNode.from_data("SoilWater").contents,
            #     ApsimXFileNode.from_param(
            #         "Models.Soils.Nutrients.Nutrient, Models",
            #         ResourceName="Nutrient",
            #     ).contents,
            #     cls.create_solute("NO3", [], [], "ppm"),
            #     cls.create_solute("NH4", [], [], "ppm"),
            #     cls.create_solute("Urea", [], [], "kg/ha"),
            #     ApsimXFileNode.from_data("SoilTemperature").contents,
            # ],
        ).contents

    def add_empty_node(self, node: str, **kwargs):
        parent = self.root
        if node == "Models.Soils.SoilCrop, Models":
            try:
                parent = self._child("Models.Soils.Physical, Models")
            except KeyError:
                self.add_empty_node("Models.Soils.Physical, Models")
                parent = self._child("Models.Soils.Physical, Models")
        elif node == "Models.Soils.Solute, Models":
            if not self.has_child("Models.Soils.Nutrients.Nutrient, Models"):
                self.add_empty_node("Models.Soils.Nutrients.Nutrient, Models",
                                    ResourceName="Nutrient")
        data_name = node.split(",", 1)[0].rsplit(".", 1)[-1].strip()
        if not data_name.startswith("Soil"):
            data_name = f"Soil{data_name}"
        parent.contents.setdefault("Children", [])
        try:
            child = ApsimXFileNode.from_data(data_name, **kwargs).contents
            param = self._layer_nodes.get(node, [])
            if node in self._initial_values_nodes:
                param = param + ["InitialValues"]
            for k in param:
                if k not in kwargs:
                    child.pop(k, None)
        except FileNotFoundError:
            child = ApsimXFileNode.from_param(node, **kwargs).contents
        parent.contents["Children"].append(child)

    def nodes_with_param(self, name: str) -> Iterator[dict]:
        for node, param in self._layer_nodes.items():
            if name not in param:
                continue
            for x in self.root.findall(requirements={"$type": node}):
                if ((node in self._initial_values_nodes
                     and name != "Thickness"
                     and x["Name"] != name)):
                    continue
                yield x

    @readonly_cached_property
    def _internal_parameters(self) -> list:
        r"""list: Set of soil parameters contained by this file."""
        out = []
        for node, param in self._layer_nodes.items():
            try:
                child = self._child(node)
            except KeyError:
                continue
            out += [
                k for k in param
                if child.get(k, None) is not None
            ]
            if node == "Models.Soils.InitialWater, Models":
                if child.get("FractionFull", None) is not None:
                    out.append("Water")
            elif node in self._initial_values_nodes:
                if child.get("InitialValues", None) is not None:
                    out.append(child["Name"])
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
        if name in self.HEADER_PARAMETERS:
            return self.contents[name]
        for x in self.nodes_with_param(name):
            if ((x["$type"] in self._initial_values_nodes
                 and name != "Thickness")):
                out = x["InitialValues"]
            elif x["$type"] == "Models.Soils.InitialWater, Models":
                out = x["FractionFull"]
            else:
                out = x[name]
            return np.array(out)
        raise KeyError(name)

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
        if allow_add and not self.has_parameters(name, ignore_calculated=True):
            added_nodes = []
            for node, param in self._layer_nodes.items():
                if name not in param:
                    continue
                kws = {}
                if ((node in self._initial_values_nodes
                     and name != "Thickness")):
                    kws["Name"] = name
                if not self.has_child(node, **kws):
                    if ((name == "Thickness"
                         and (node in self._initial_values_nodes
                              or node in self._alternate_nodes))):
                        continue
                    elif (node in self._alternate_nodes
                          and self.has_child(
                              self._alternate_nodes[node],
                              Name=name)):
                        continue
                    self.add_empty_node(node, **kws)
                    added_nodes.append(node)
        match = False
        if isinstance(value, np.ndarray):
            value = value.tolist()
        for x in self.nodes_with_param(name):
            if ((x["$type"] in self._initial_values_nodes
                 and name != "Thickness")):
                x["InitialValues"] = value
                if "Thickness" not in x:
                    thickness = self.get("thickness", default=None)
                    if thickness is not None:
                        x["Thickness"] = thickness.value.tolist()
                if x["$type"] == "Models.Soils.Solute, Models":
                    solute_units = self.PARAMETER_UNITS[name]
                    if solute_units in ["ppm", "0.000001"]:
                        solute_units = 0
                    elif solute_units == "kg/ha":
                        solute_units = 1
                    else:
                        raise RuntimeError(f"Unsupported units: "
                                           f"\"{solute_units}\"")
                    x["InitialValuesUnits"] = solute_units
            elif x["$type"] == "Models.Soils.InitialWater, Models":
                x["FractionFull"] = value[0]
            else:
                x[name] = value
            match = True
        if not match:
            raise KeyError(name)

    def update_param(self, contents: Any) -> None:
        r"""Merge downloaded parameters into the current data.

        Args:
            contents: New data to incorporate.

        """
        for k, v in contents.items():
            # TODO: Original behavior ignored unused parameters
            # try:
            self._set(k, v, allow_add=True)
            # except KeyError:
            #     pass

    @ParameterNormalizer.register_method("texture_class")
    @classmethod
    def normalize_texture_class(cls, texture_class: str):
        return texture_class.title()

    @ParameterCalculator.register_method("AirDry", "wilting_point")
    @classmethod
    def airdry_from_wilting_point(cls, ll15: float):
        ll15_units = cls.get_units(ll15)
        ll15_units_type = utils.matching_unit_type(ll15)
        ll15 = cls.add_units(ll15, "", return_scalar=True)
        out = ll15
        out[0] = 0.5 * ll15[0]
        out[out < 0.005] = 0.005
        # out[3:] = ll15[3:]
        if ll15_units:
            out = ll15_units_type(out, "")
        return out

    @ParameterNormalizer.register_method("sat", "bulk_density")
    @classmethod
    def sat_from_bulk_density_norm(cls, sat: float, bd: float,
                                   particle_density: Optional[float] = 2.65):
        # TODO: Adjust for DUL at the same time?
        sat_units = cls.get_units(sat)
        sat_units_type = utils.matching_unit_type(sat)
        sat = cls.add_units(sat, "", return_scalar=True)
        bd = cls.add_units(bd, "g/cm3", return_scalar=True)
        particle_density = cls.add_units(particle_density, "g/cm3",
                                         return_scalar=True)
        max_sw = 1.0 - (bd / particle_density)
        max_bd = (1.0 - sat) * particle_density
        gap = 0.01
        idx = np.logical_or(sat > (max_sw - gap), bd >= (max_bd - gap))
        sat[idx] = max_sw[idx] - gap
        if sat_units is not None:
            sat = sat_units_type(sat, "")
        return sat

    @ParameterNormalizer.register_method("dul", "sat")
    @classmethod
    def dul_normalization(cls, dul: float, sat: float):
        r"""Normalize DUL to be less than SAT as expected by ApsimX."""
        dul_units = cls.get_units(dul)
        dul_units_type = utils.matching_unit_type(dul)
        dul = cls.add_units(dul, "", return_scalar=True)
        sat = cls.add_units(sat, "", return_scalar=True)
        gap = 0.05  # 0.002
        idx = ((sat - dul) < gap)
        dul[idx] = sat[idx] - gap
        dul[dul < 0] = 0.0
        if dul_units is not None:
            dul = dul_units_type(dul, "")
        return dul

    @ParameterCalculator.register_method("moisture", "dul",
                                         "wilting_point")
    @classmethod
    def default_initial_moisture(cls, dul: float, ll15: float):
        return 0.5 * (dul + ll15)

    @ParameterCalculator.register_method("LL", "wilting_point")
    @classmethod
    def default_LL(cls, wilting_point: float):
        # TODO: LL, KL, & KF should be normalized based on crop values
        return wilting_point

    @ParameterCalculator.register_method("KL", "bottom_depth")
    @classmethod
    def default_KL(cls, layers: float):
        layers = cls.add_units(layers, "mm", return_scalar=True)
        return 0.06 * cls.create_profile(
            layers, a=cls._curveparam_a, b=cls._curveparam_b)

    @ParameterCalculator.register_method("XF", "bottom_depth")
    @classmethod
    def default_XF(cls, layers: float):
        layers = cls.add_units(layers, "mm", return_scalar=True)
        return cls.create_profile(layers, a=cls._curveparam_a, b=0)

    @ParameterCalculator.register_method("FOMCNRatio", "bottom_depth")
    @classmethod
    def default_FOMCNRatio(cls, layers: float):
        return 40.0

    @ParameterCalculator.register_method("cn_ratio", "bottom_depth")
    @classmethod
    def default_SoilCNRatio(cls, layers: float):
        return 12.0 * np.ones(len(layers), dtype=np.float64)

    @ParameterCalculator.register_method("FBiom", "bottom_depth")
    @classmethod
    def default_FBiom(cls, layers: float):
        layers = cls.add_units(layers, "mm", return_scalar=True)
        out = 0.045 * cls.create_profile(
            layers, a=cls._curveparam_a, b=cls._curveparam_b)
        # TODO: Scale?
        out[0] = 0.0395
        out[1] = 0.035
        return out

    @ParameterCalculator.register_method("FInert", "bottom_depth")
    @classmethod
    def default_FInert(cls, layers: float):
        layers = cls.add_units(layers, "mm", return_scalar=True)
        out = 0.83 * cls.create_profile(
            layers, a=cls._curveparam_a, b=-0.01)
        out[0] = 0.65
        out[1] = 0.668
        return out

    @ParameterCalculator.register_method("FOM", "bottom_depth")
    @classmethod
    def default_FOM(cls, layers: float):
        layers = cls.add_units(layers, "mm", return_scalar=True)
        return 160 * cls.create_profile(
            layers, a=cls._curveparam_a, b=cls._curveparam_b)

    @ParameterCalculator.register_method("no3", "bottom_depth")
    @classmethod
    def default_NO3(cls, layers: float):
        layers = cls.add_units(layers, "mm", return_scalar=True)
        return 0.5 * cls.create_profile(
            layers, a=cls._curveparam_a, b=0.01)

    @ParameterCalculator.register_method("nh4", "bottom_depth")
    @classmethod
    def default_NH4(cls, layers: float):
        layers = cls.add_units(layers, "mm", return_scalar=True)
        return 0.05 * cls.create_profile(
            layers, a=cls._curveparam_a, b=0.01)

    @ParameterCalculator.register_method("urea", "bottom_depth")
    @classmethod
    def default_Urea(cls, layers: float):
        return np.zeros(len(layers), dtype=np.float64)

    @ParameterCalculator.register_method("SWCON", "bottom_depth")
    @classmethod
    def default_SWCON(cls, layers: float):
        return 0.3 * np.ones(len(layers), dtype=np.float64)

    def normalize_parameters(self):
        r"""Normalize the parameters in the file."""
        super().normalize_parameters()
        # TODO: Delay this so it is filled in by crop defaults? or using
        #   current soil properties
        if not (self.has_child("Models.Soils.SoilTemp.SoilTemperature, Models")
                or self.has_child(
                    "Models.Soils.CERESSoilTemperature, Models")):
            self.add_empty_node(
                "Models.Soils.SoilTemp.SoilTemperature, Models",
                Thickness=None,
                InitialValues=None,
            )
        ordered_children = []
        for node, param in self._layer_nodes.items():
            if node == "Models.Soils.SoilCrop, Models":
                continue
            for x in self.root.findall(requirements={"$type": node}):
                if node in self._initial_values_nodes:
                    for k in ["Thickness", "InitialValues"]:
                        x.contents.setdefault(k, None)
                else:
                    for k in param:
                        x.contents.setdefault(k, None)
                ordered_children.append(x.contents)
        assert len(ordered_children) == len(self.contents["Children"])
        self.contents["Children"] = ordered_children
        # Refresh thickness to fill in places where it is missing
        # self.set("thickness", self.get("thickness"))
        # TODO: Generic minimum number of layers for other file types?
        if self.nlayers == 1:
            thickness = 200.0 * np.ones((10, ))
            self.resample_depths(thickness=thickness)

    # TODO: Remove this after it is verified to pass (i.e. parameters
    # all set or normalized by calculators/normalizers)
    def calculate_missing(self):
        r"""Calculate missing parameters that are required."""
        # TODO: Check for scalars and interpolate?
        if ((self.physical.contents.get("KS", None)
             and not np.isfinite(self.physical.contents["KS"]).all())):
            raise ValueError(
                "Failed to compute saturated hydraulic "
                "conductivity from clay, sand, and carbon data")
        if (((not self.uses_initial_water)
             and (not self.water.contents.get("InitialValues", None))
             and self.physical.contents.get("LL15", None)
             and self.physical.contents.get("DUL", None))):
            # TODO: Set this from NASA Powers or SMAP L4?
            # self.water.contents["InitialValues"] = [
            #     0.5 * (d + l) for d, l in zip(
            #         self.physical.contents["DUL"],
            #         self.physical.contents["LL15"]
            #     )
            # ]
            raise Exception("THIS SHOULD HAVE BEEN CALCULATED")
        layers = np.array(self.physical.contents["Thickness"]).cumsum()
        nlayers = len(layers)
        # TODO: Generic minimum number of layers
        if nlayers == 1:
            raise Exception("CALCUALTE_MISSING WOULD HAVE RESAMPLED")
        # if nlayers == 1:
        #     thickness = 200.0 * np.ones((10, ))
        #     self.resample_depths(thickness=thickness)

    @classmethod
    def create_solute(cls, name: str,
                      thickness: list,
                      initial: list,
                      units: str | int) -> dict:
        r"""Create a solute node.

        Args:
            name: Solute name.
            thickness: Layer thickness.
            initial: Initial solute concentration at each later.
            units: Solute contentration units. 0 indicates ppm, 1
                indicates kg/ha.

        Returns:
            dict: Solute node.

        """
        # TODO: Use actual units from yggdrasil_rapidjson?
        if isinstance(units, str):
            if units == "ppm":
                units = 0
            elif units == "kg/ha":
                units = 1
            else:
                raise RuntimeError(f"Unsupported units: \"{units}\"")
        assert len(initial) == 0 or len(thickness) == len(initial)
        out = ApsimXFileNode.from_data("SoilSolute")
        out.contents.update(
            Name=name,
            Thickness=thickness,
            InitialValues=initial,
            InitialValuesUnits=units)
        return out.contents

    @classmethod
    def _from_compatible(cls, src: BaseSoilFile, **kwargs):
        r"""Convert data from another file format into the correct
        format for this file.

        Args:
            src: Base class data.
            \*\*kwargs: Additional keyword arguments are passed to
                get_universal_parameter_map.

        Returns:
            Converted data.

        """
        out = super()._from_compatible(src, **kwargs)
        # if out.contents["DataSource"] is None:
        #     now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #     out.contents["DataSource"] = (
        #         f"Original source is {src.DESC} ({src.URL}) "
        #         f"retrieved on {now}"
        #     )
        if out.contents["Comments"] is None:
            out.contents["Comments"] = (
                f"Generated by simulatr from {src.DESC} data"
            )
        return out


class ApsimXFile(CropModelFile):
    r"""Container for manipulating .apsimx model files.

    Args:
        fname: Path to a .apsimx model file.
        generated: If True, this file was generated.
        contents: Contents to initialize the file with.

    """

    NAME: ClassVar[str] = "apsimx"
    EXAMPLE: ClassVar[str] = os.path.join("Examples", "Wheat.apsimx")
    ACTION_NODES: ClassVar[dict] = dict({
        "sow": {
            "conflicts": [
                {
                    "contains": {"$type": "Models.Manager, Models"},
                    "calls": "Sow",
                },
            ],
        },
        "harvest": {
            "conflicts": [
                {
                    "contains": {"$type": "Models.Manager, Models"},
                    "calls": "Harvest",
                },
            ],
        },
        "irrigate": {
            "parent": {"contains": {"Name": "Field"}},
            "contains": {"Name": "Irrigation"},
            "required": True,
            # "default": "Irrigate.json",
            "default": {
                "$type": "Models.Irrigation, Models",
                "Name": "Irrigation",
                "ResourceName": "Irrigation",
                "Children": [],
                "Enabled": True,
                "ReadOnly": False,
            },
        },
    }, ** {
        k: {
            "parent": {"contains": {"Name": "Field"}},
            "contains": {"Name": "Fertiliser"},
            "required": True,
            # "default": "Fertilize.json",
            "default": {
                "$type": "Models.Fertiliser, Models",
                "Name": "Fertiliser",
                "ResourceName": "Fertiliser",
                "Children": [],
                "Enabled": True,
                "ReadOnly": False,
            },
        }
        for k in ["fertilize", "nitrogen", "calcium", "phosphorus"]
    })

    @readonly_cached_property
    def parameter_nodes(self) -> dict:
        r"""dict: Previously loaded parameter nodes."""
        return {}

    @classmethod
    def available_crops(cls, category: Optional[str] = None) -> List[str]:
        r"""Get the crops that can be simulated via this model.

        Args:
            category: Get crops in a certain category (e.g. "oilseed").

        Returns:
            list: Available crop names.

        """
        if isinstance(category, list):
            out = []
            for x in category:
                iout = cls.available_crops(x)
                out += [k for k in iout if k not in out]
            return out
        if category == "tree":
            return ["Gliricidia", "Pinus", "Eucalyptus"]
        elif category == "legume":
            return ["Soybean", "Chickpea", "Peanut", "Mungbean"]
        elif category == "oilseed":
            return ["Canola", "Peanut", "Soybean", "OilPalm"]
        elif category == "cereal":
            return [
                "Wheat", "Maize", "Oats", "Barley", "Sorghum",
            ]
        elif category == "cover":
            return [
                "WhiteClover", "RedClover",
                "Mungbean", "Lucerne",
            ]
        elif category == "forage":
            return [
                "PlantainForage", "WhiteClover", "RedClover",
                "Chicory", "Lucerne",
            ]
        elif category == "root":
            return [
                "FodderBeet",  # Check that root is the part used
                "Potato",
            ]
        elif category is not None:
            raise NotImplementedError(category)
        resources_dir = os.path.join(
            ApsimXEngine.model_dir(), "Models", "Resources")
        files = glob.glob(os.path.join(resources_dir, "*.json"))
        exclude = [
            "MicroClimate",
            "Nutrient",
            "SurfaceOrganicMatter",
            "WaterBalance",
            "Fertiliser",
            # Simplified that could be used
            "CLEM",
            "SCRUM",
            "Slurp",
            "SPRUM",
            "STRUM",
            # Non-PMF (TODO: Exclude by parsing)
            "Sugarcane",
            # Error due to missing parameters
            # "FodderBeet",  # [StorageRoot].Live.MetabolicWt, StorageWt
            "Lucerne",  # Missing .Grain.Size:
            "Grapevine",  # No Total.Wt
            "OilPalm",  # missing mortalityRate
        ]
        out = []
        for x in files:
            name = os.path.splitext(os.path.basename(x))[0]
            if name in exclude or name.startswith("AGP"):
                continue
            out.append(name)
        return out

    @classmethod
    def available_cultivars(cls, crop_name: str) -> List[str]:
        r"""Get the cultivars for a given crop that can be simulated
        via this model.

        Args:
            crop_name: Crop name.

        Returns:
            list: Available crop cultivar names.

        """
        crop_name = cls.validate_crop_name(crop_name)
        resources_file = ApsimXFileNode.from_file(
            os.path.join(
                ApsimXEngine.model_dir(), "Models", "Resources",
                f"{crop_name}.json"))
        out = []
        for x in resources_file.findall(
                requirements={
                    "contains": {
                        "$type": "Models.PMF.Cultivar, Models"}
                }):
            out.append(x["Name"])
        skip = []
        if crop_name == "Chickpea":
            # Requires [Phenology].Vegetative.Progression
            #            .PhotoperiodModifier.CriticalPhotoperiod
            skip = ["Anwar", "Hashem"] + [
                k for k in out
                if k.startswith(("9", "0", "Ghab"))
            ]
        elif crop_name == "Eucalyptus":
            # Requires [Leaf].FRGRFunction.FRGRFunctionTemp.Response.X
            skip = ["nitensLewisham"]
        elif crop_name == "Oats":
            skip = [
                # Error in MathUtilities.LinearInterpReal
                "Drummond_orig",
                # Cannot find property [Leaf].InitialLeaves1.Area
                "PFR_100_05",
                "Coronet",
            ]
        for k in skip:
            out.remove(k)
        return out

    @classmethod
    def find_example(cls, crop_name: str) -> str:
        r"""Locate an example model file for a given crop name.

        Args:
            crop_name: Crop name.

        Returns:
            str: Model input file for the specified crop.

        """
        examples_dir = os.path.join(ApsimXEngine.model_dir(), "Examples")
        for x in [crop_name, crop_name.title()]:
            fname = os.path.join(examples_dir, f"{x}.apsimx")
            if os.path.isfile(fname):
                return fname
        raise ValueError(f"Could not locate a model file for crop "
                         f"\"{crop_name}\".")

    @classmethod
    def from_example(cls, src: Union[str, "ApsimXFile"],
                     dst: Optional[str | None] = None,
                     directory: Optional[str | None] = None,
                     interactive: Optional[bool] = False,
                     actions: Optional[List[str] | None] = None
                     ) -> CropModelFile:
        r"""Create an input model file from an example.

        Args:
            src (str, ApsimXFile): Path to the source .apsimx model.
            dst (str, optional): Path to the location where the generated
                .apsimx model should be saved.
            directory: Directory where the generated file should be
                saved (only used if dst is not provided).
            interactive: If True, make the file interactive.
            actions: Interactive actions that should be added.

        Returns:
            CropModelFile: Constructed model input file.

        """
        if not isinstance(src, ApsimXFile):
            src = ApsimXFile(src)
        out = src.copy(dst=dst, directory=directory)
        if interactive or actions:
            if not actions:
                actions = list(ApsimXEngine.AVAILABLE_ACTION_MAP.keys())
            out.make_interactive(actions)
        return out

    @classmethod
    def from_crop_name(cls, crop_name: str,
                       crop_variety: Optional[str | None] = None,
                       dst: Optional[str | None] = None,
                       directory: Optional[str | None] = None,
                       interactive: Optional[bool] = False,
                       actions: Optional[List[str] | None] = None,
                       dont_fetch_weather: Optional[bool] = False,
                       dont_fetch_soil: Optional[bool] = False,
                       **kwargs: Any) -> CropModelFile:
        r"""Create an input model file for a given crop name.

        Args:
            crop_name: Crop name.
            crop_variety: Crop variety.
            dst: Path to the location where the generated file should
                be saved.
            directory: Directory where the generated file should be
                saved (only used if dst is not provided).
            interactive: If True, make the file interactive.
            actions: Interactive actions that should be added.
            dont_fetch_weather: If True, don't download weather data.
            dont_fetch_soil: If True, don't download soil data.
            \*\*kwargs: Additional keyword arguments are treated as
                parameter key/value pairs.

        Returns:
            CropModelFile: Constructed model input file.

        """
        crop_name = cls.validate_crop_name(crop_name)
        if crop_variety is None:
            if crop_name == "Wheat":
                crop_variety = "Hartog"
            else:
                varieties = cls.available_cultivars(crop_name)
                crop_variety = "" if not varieties else varieties[0]
        if crop_variety:
            kwargs["crop_variety"] = crop_variety
        crop_resource = ApsimXFileNode.from_resource(crop_name)
        if dst is None:
            dst = f"{crop_name}-{crop_variety}-Generated.apsimx"
            if interactive or actions:
                dst = "-Interactive".join(os.path.splitext(dst))
            if directory:
                dst = os.path.join(directory, dst)
        if actions is None:
            if interactive:
                actions = list(ApsimXEngine.AVAILABLE_ACTION_MAP.keys())
            else:
                actions = []
        try:
            crop_root = ApsimXFileNode.from_example(crop_name)
        except BaseException:
            crop_root = None
        output_vars = [
            f"[{crop_name}].LAI",
        ]
        if crop_resource.find("Zadok"):
            output_vars += [
                f"[{crop_name}].Phenology.Zadok.Stage",
            ]
        output_vars += [
            f"[{crop_name}].Phenology.CurrentStageName",
            f"[{crop_name}].AboveGround.Wt",
            f"[{crop_name}].AboveGround.N",
        ]
        if crop_resource.find("Grain"):
            output_vars += [
                f"[{crop_name}].Grain.Total.Wt*10 as Yield",
            ]
            if crop_resource.find("Protein"):
                output_vars += [
                    # Missing from Maize
                    f"[{crop_name}].Grain.Protein",
                ]
            output_vars += [
                f"[{crop_name}].Grain.Size",
                f"[{crop_name}].Grain.Number",
                f"[{crop_name}].Grain.Total.Wt",
                f"[{crop_name}].Grain.Total.N",
            ]
        else:
            output_vars += [
                f"[{crop_name}].Total.Wt*10 as Yield"
            ]
        output_vars += [
            f"[{crop_name}].Total.Wt"
        ]
        sim_children = [
            ApsimXFileNode.from_data("Clock"),
            ApsimXFileNode.from_data("Summary"),
            ApsimXFileNode.from_data("Weather",
                                     crop_root=crop_root),
            ApsimXFileNode.from_param(
                "Models.Soils.Arbitrator.SoilArbitrator, Models"),
            ApsimXFileNode.from_data("MicroClimate",
                                     crop_root=crop_root),
        ]
        zone_children = [
            ApsimXFileNode.from_param(
                "Models.Report, Models",
                VariableNames=["[Clock].Today"] + output_vars,
                EventNames=[
                    "[Clock].EndOfDay",
                ],
                GroupByVariableName=None,
            ),
            ApsimXFileNode.from_param(
                "Models.Fertiliser, Models",
                ResourceName="Fertiliser",
            ),
            ApsimXFileNode.from_param(
                "Models.Irrigation, Models",
                # ResourceName="Irrigation",
            ),
            # TODO: Use a different soil module if the crop uses
            # "Models.Soils.Sample, Models" and
            # "Models.Soils.InitialWater, Models"
            ApsimXFileNode.from_data("Soil", crop_root=crop_root),
            ApsimXFileNode.from_data("SurfaceOrganicMatter",
                                     crop_root=crop_root),
            ApsimXFileNode.from_param(
                "Models.PMF.Plant, Models",
                Name=crop_name,
                ResourceName=crop_name,
            ),
            ApsimXFileNode.from_data(
                "AutoSow",  # TODO: Handle cultivar
                Enabled=("sow" not in actions),
                crop_root=crop_root,
            ),
            # Fertilise @ sow?
            # Scheduled fertilizer/irrigation
            # (see Operations in Examples/Potato.apsimx)
            ApsimXFileNode.from_data(
                "AutoHarvest",
                Enabled=("harvest" not in actions),
                crop_root=crop_root,
            ),
            # Should this be turned on by default?
            # ApsimXFileNode.from_data(
            #     "AutoIrrigate",
            #     Enabled=("irrigate" not in actions),
            #     crop_root=crop_root,
            # ),
        ]
        if "weed" in actions:
            kwargs.setdefault("weed_name", "Oats")
        weed_soil = None
        weed_name = kwargs.get("weed_name", None)
        weed_variety = kwargs.get("weed_variety", None)
        if weed_name is not None:
            weed_name = kwargs["weed_name"]
            weed_variety = kwargs.get("weed_variety", None)
            if weed_variety is None:
                weed_varieties = cls.available_cultivars(weed_name)
                weed_variety = weed_varieties[0]
                kwargs["weed_variety"] = weed_variety
            weed_root = ApsimXFileNode.from_example(weed_name)
            weed_soil = ApsimXFileNode.from_data(
                "SoilCrop", crop_root=weed_root, Name="WEEDSoil")
            zone_children += [
                ApsimXFileNode.from_param(
                    "Models.PMF.Plant, Models",
                    Name=weed_name,
                    ResourceName=weed_name,
                    Description="Weed",
                ),
                ApsimXFileNode.from_data(
                    "WeedCycle",
                )
            ]
        sim_children.append(
            ApsimXFileNode.from_param(
                "Models.Core.Zone, Models",
                Area=1.0,
                Slop=0.0,
                AspectAngle=0.0,
                Altitude=50.0,
                Name="Field",
                Children=zone_children,
            )
        )
        contents = ApsimXFileNode.from_param(
            "Models.Core.Simulations, Models",
            Version=168,
            Children=[
                ApsimXFileNode.from_param(
                    "Models.Core.Simulation, Models",
                    Descriptors=None,
                    Children=sim_children,
                ),
                ApsimXFileNode.from_param(
                    "Models.Storage.DataStore, Models",
                    useFirebird=False,
                    CustomFileName=None,
                ),
            ]
        )
        contents.specialize_crop(crop_name)
        if weed_name is not None:
            soil_node = contents.find(
                requirements={"$type": "Models.Soils.Physical, Models"})
            soil_node["Children"].append(weed_soil)
            contents.specialize_crop(weed_name,
                                     parameter_name="WeedName",
                                     placeholder="WEED")
        assert contents["Name"] == "Simulations"
        out = cls(dst, generated=True, contents=contents.contents)
        if interactive or actions:
            out.make_interactive(actions)
        for k, v in kwargs.items():
            out.set(k, v)
        if ((("latitude" in kwargs and "longitude" in kwargs)
             or ("start_time" in kwargs and "end_time" in kwargs))):
            if "weather_file" not in kwargs and not dont_fetch_weather:
                out.set(
                    "weather_file",
                    ApsimXWeatherFile.fetch_data(
                        out.get("latitude"), out.get("longitude"),
                        out.get("start_time"), out.get("end_time"),
                    )
                )
            if "soil_file" not in kwargs and not dont_fetch_soil:
                out.set(
                    "soil_file",
                    ApsimXSoilFile.fetch_data(
                        out.get("latitude"), out.get("longitude"),
                        out.get("start_time"), out.get("end_time"),
                    )
                )
        return out

    @property
    def formal_crop_name(self) -> str:
        r"""str: Crop name used for resources."""
        return self.validate_crop_name(self.crop_name)

    def _get_external_name(self, name: str) -> str:
        r"""Get the external variable name from the internal variable
        name.

        Args:
            name: Internal parameter name.

        Returns:
            str: Parameter name.

        """
        # TODO: Map field variables in PARAM_NODES?
        if name.startswith(f"[{self.formal_crop_name}]"):
            return name.replace(f"[{self.formal_crop_name}]", "[CROP]")
        return name

    def _get_internal_name(self, name: str) -> str:
        r"""Get the internal variable name from a model parameter name.

        Args:
            name: Parameter name.

        Returns:
            str: Internal parameter name.

        """
        info = ApsimXEngine.get_field_metadata(name, None)
        if info is None:
            if name.startswith("[CROP]"):
                return name.replace("[CROP]",
                                    f"[{self.formal_crop_name}]")
            return name
        node = self.find_parameter(name, required=True)
        if "parameter" in info:
            names = [x["Key"] for x in node["Parameters"]]
            field = info["parameter"]
            if field not in names:
                raise KeyError(
                    f"Parameter \"{field}\" not present in node "
                    f"\"{node['Name']}\" for parameter \"{name}\" "
                    f"(Parameters = {names})"
                )
            idx = names.index(field)
            out = f"[{node['Name']}].Parameters[{idx}].Value"
        else:
            out = f"[{node['Name']}].{info['field']}"
        return out

    @classmethod
    def _get_node_parameter(cls, node: dict, name: str) -> Any:
        r"""Extract a node parameter from a parameter list.

        Args:
            node: Node containing parameters
            name: Parameter name.

        Returns:
            Parameter value.

        Raises:
            KeyError: If name is not a valid parameter name.

        """
        for x in node["Parameters"]:
            if isinstance(name, set) and x["Key"] in name:
                return x["Value"]
            elif x["Key"] == name:
                return x["Value"]
        raise KeyError(f"No parameter named \"{name}\" in {node}")

    @classmethod
    def _set_node_parameter(cls, node: dict, name: str, value: Any):
        r"""Set a node parameter.

        Args:
            node: Node containing parameters
            name: Parameter name.
            value: Parameter value.

        Raises:
            KeyError: If name is not a valid parameter name.

        """
        for x in node["Parameters"]:
            if isinstance(name, set) and x["Key"] in name:
                x["Value"] = value
                return
            elif x["Key"] == name:
                x["Value"] = value
                return
        raise KeyError(f"No parameter named \"{name}\" in {node}")

    @classmethod
    def _get_parameter(cls, node: dict, info: dict):
        r"""Get a parameter value from a node based on the provided
        information.

        Args:
            node: Node containing the parameter.
            info: Information about how to locate the parameter.

        Returns:
            Parameter value.

        Raises:
            ValueError: If the node does not match any of the
                requirements.
            NotImplementedError: If info is invalid.
            KeyError: If the parameter cannot be located.

        """
        out = None
        if "parameter" in info:
            out = cls._get_node_parameter(node, info["parameter"])
        elif "field" in info:
            out = node[info["field"]]
        elif "internal" in info:
            out = cls._get_parameter(
                node, ApsimXEngine.get_field_metadata(info["internal"]))
        elif "anyOf" in info:
            for x in info["anyOf"]:
                if cls.node_matches(node, **x):
                    out = cls._get_parameter(node, x)
                    break
            else:
                errors = []
                for x in info["anyOf"]:
                    cls.node_matches(node, errors=errors, **x)
                raise ValueError(
                    "Node does not match the requirements:\n  "
                    + "\n  ".join(errors)
                )
        elif info.get("full_node"):
            out = node
        else:
            raise NotImplementedError(f"Invalid info {info}")
        if "fget" in info:
            try:
                out = info["fget"](out)
            except ValueError as e:
                raise KeyError(e)
        return out

    @classmethod
    def _set_parameter(cls, node: dict, info: dict, value: Any):
        r"""Set a parameter value in a node based on the provided
        information.

        Args:
            node: Node containing the parameter.
            info: Information about how to locate the parameter.
            value: Value to set the parameter to.

        Raises:
            ValueError: If the node does not match any of the
                requirements.
            NotImplementedError: If info is invalid.
            KeyError: If the parameter cannot be located.

        """
        if isinstance(value, (datetime.datetime, datetime.date)):
            value = value.isoformat()
        if "fset" in info:
            try:
                value = info["fset"](value)
            except ValueError as e:
                raise KeyError(e)
        if "fset_prev" in info:
            prev = cls._get_parameter(node, info)
            value = info["fset_prev"](value, prev)
        if "parameter" in info:
            cls._set_node_parameter(node, info["parameter"], value)
        elif "field" in info:
            node[info["field"]] = value
        elif "internal" in info:
            cls._set_parameter(
                node, ApsimXEngine.get_field_metadata(info["internal"]),
                value)
        elif "anyOf" in info:
            for x in info["anyOf"]:
                if cls.node_matches(node, **x):
                    cls._set_parameter(node, x, value)
                    break
            else:
                errors = []
                for x in info["anyOf"]:
                    cls.node_matches(node, errors=errors, **x)
                raise ValueError(
                    "Node does not match the requirements:\n  "
                    + "\n  ".join(errors)
                )
        elif info.get("full_node", False) is True:
            node.clear()
            node.update(**value)
        else:
            raise NotImplementedError(f"Invalid info {info}")

    def _get(self, name: str):
        r"""Get a parameter from the model file.

        Args:
            name: Parameter name.

        Returns:
            Parameter value.

        Raises:
            KeyError: If name is not a valid parameter name.

        """
        node = self.find_parameter(name, required=True)
        info = ApsimXEngine.get_field_metadata(name)
        try:
            return self._get_parameter(node, info)
        except KeyError as e:
            raise KeyError(f"{name}: {e}")

    def _set(self, name: str, value: Any,
             info: Optional[dict] = None) -> Any:
        r"""Set a parameter in the model file.

        Args:
            name: Parameter name.
            value: Parameter value.
            info: Information about how to locate the parameter.

        Raises:
            KeyError: If name is not a valid parameter name.

        """
        add_missing = (info is None)
        if info is None:
            info = ApsimXEngine.get_field_metadata(name)
        if info is False:
            return
        if name == "output_vars" and isinstance(value, list):
            value = [self._get_internal_name(x) for x in value]
        try:
            anyset = False
            for xnode in self.findall_parameters(name, info=info):
                try:
                    self._set_parameter(xnode, info, value)
                    anyset = True
                except KeyError:
                    continue
            if not anyset:
                node = self.find_parameter(
                    name, required=True,
                    add_missing=add_missing,
                    info=info,
                )
                self._set_parameter(node, info, value)
        except KeyError as e:
            raise KeyError(f"{name}: {e}")

    @classmethod
    def _write(cls, fname: str, contents):
        r"""Read a model input file.

        Args:
            fname: Path to file to read.
            contents: File contents to write.

        """
        if isinstance(contents, ApsimXFileNode):
            contents = contents.contents
        with open(fname, "w") as fd:
            json.dump(contents, fd, indent="  ")

    @readonly_cached_property
    def is_interactive(self) -> bool:
        r"""bool: True if the .apsimx model is interactive."""
        return bool(self.find("Synchroniser"))

    def disable_parameter_conflicts(self, name: str,
                                    info: Optional[dict] = None,
                                    node: Optional[dict] = None) -> None:
        r"""Disable nodes that conflict with a parameter/action.

        Args:
            name: Action name.
            info: Information about the parameter.
            node: Parameter/action node to avoid disabling.

        """
        if info is None:
            info = (
                self.ACTION_NODES[name] if name in self.ACTION_NODES
                else ApsimXEngine.get_field_metadata(name)
            )
        if node is None and self.includes_constraints(info):
            node = self.find_parameter(name, info=info)
        for vconflict in info.get("conflicts", []):
            for x in self.findall(requirements=vconflict):
                if x == node:
                    continue
                logger.info(
                    f"Disabling node \"{x['Name']}\" which "
                    f"conflicts with parameter/action \"{name}\""
                )
                x["Enabled"] = False
                self.generated = False

    def add_parameter(self, name: str, info: Optional[dict] = None,
                      parent: Optional[dict] = None) -> dict:
        r"""Add a node to facilitate use of a parameter/action if it
        is missing.

        Args:
            name: Action name.
            info: Information about how to add the parameter.
            parent: Parent node that the action node should be added to
                if it is missing.

        Returns:
            dict: Action node.

        """
        if info is None:
            info = (
                self.ACTION_NODES[name] if name in self.ACTION_NODES
                else ApsimXEngine.get_field_metadata(name)
            )
        # Do conflicts first before adding the default so that the
        # default is not disabled by mistake
        self.disable_parameter_conflicts(name, info=info, node={})
        default = info.get("default", None)
        if isinstance(default, str):
            if not os.path.isabs(default):
                default = os.path.join(ApsimXEngine.data_dir, default)
            default = ApsimXFile(default).contents
        if not default:
            logger.warning(
                f"No default node registered for parameter/action "
                f"\"{name}\""
            )
        if parent is None and "parent" in info:
            parent = self.find(
                parent=True, requirements=info["parent"]
            )
        if not parent:
            logger.warning(
                f"No parent node registered for parameter/action "
                f"\"{name}\""
            )
        node = None
        if parent and default:
            parent["Children"].append(default.copy())
            node = parent["Children"][-1]
            node["Enabled"] = True
            self.generated = False
        if node:
            for k, v in info.get("parameter_properties", {}).items():
                value = getattr(self, v, None)
                if value is not None:
                    self._set_node_parameter(node, k, value)
                    self.generated = False
        return node

    def disable_action(self, name: str, **kwargs: Any) -> None:
        r"""Disable any nodes that automatically control an action.

        Args:
            name: Action name.
            **kwargs: Additional keyword arguments are passed to
                find_parameter.

        """
        node = self.find_parameter(name, **kwargs)
        if node is not None:
            node["Enabled"] = False
            self.generated = False

    def enable_action(self, name: str, parent: Optional[dict] = None,
                      **kwargs: Any) -> Optional[dict]:
        r"""Enable any nodes that automatically control an action.

        Args:
            name: Action name.
            parent: Parent node that the action node should be added to
                if it is missing.
            **kwargs: Additional keyword arguments are passed to
                find_parameter.

        Returns:
            dict: Action node.

        """
        info = self.ACTION_NODES[name]
        if not self.includes_constraints(info):
            self.disable_parameter_conflicts(name, info=info, node={})
            return
        return self.find_parameter(
            name, info=info,
            add_missing=(parent if parent else True),
            current=parent, **kwargs
        )

    def disable(self, name: str, **kwargs: Any) -> None:
        r"""Disable a node in the file if it exists.

        Args:
            name: Name of the node to disable.
            **kwargs: Additional keyword arguments are passed to find.

        """
        node = self.find(name, **kwargs)
        if node is not None:
            node["Enabled"] = False
            self.generated = False

    @classmethod
    def includes_constraints(cls, info: dict) -> bool:
        r"""Check if a set of node requirements constrain the node.

        Args:
            info: Node requirements.

        Returns:
            bool: True if info constrains the node, False otherwise.

        """
        return any(k in info for k in [
            "name", "field", "parameter", "internal", "contains",
            "equals", "fvalid", "calls", "anyOf", "nested"])

    @classmethod
    def node_matches(cls, node: Any,
                     errors: Optional[list] = None,
                     name: Optional[str] = None,
                     field: Optional[str] = None,
                     parameter: Optional[str] = None,
                     internal: Optional[str] = None,
                     contains: Optional[Union[list, set, dict]] = None,
                     equals: Optional[Any] = None,
                     fvalid: Optional[Callable] = None,
                     calls: Optional[str] = None,
                     anyOf: Optional[list] = None,
                     nested: Optional[dict] = None,
                     **kwargs: Any) -> bool:
        r"""Check if a node matches the specified requirements.

        Args:
            node: Node to check.
            errors: If a list is provided, errors will be added to this
                list.
            name: Name that the node must have.
            field: Name of a field that must be present.
            parameter: Name of a parameter that must be present.
            contains: Fields/elements that the node must contain. If a
                set is provided, only one of the elements must be
                present. If a dict is provided, the values in the node
                must match the values in the provided dict.
            equals: Value that the node must be equivalent to.
            fvalid: Function that returns True if the node is valid, and
                False otherwise.
            calls: Name of a function called in the node code block.
            anyOf: List of kwargs for node_matches that should be
                checked. If the node satisfies any of these requirements,
                True will be returned.
            nested: Set of requirements for individual fields.
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            bool: True if the node matches, False otherwise.

        """

        def add_error(msg: str) -> bool:
            r"""Add an error message to the errors list.

            Args:
                msg: Error message to add.

            Returns:
                bool: True if the error was added, False otherwise.

            """
            if not isinstance(errors, list):
                return False
            errors.append(msg)
            return True

        if anyOf:
            if isinstance(errors, list):
                xerrors = []
                if not any(cls.node_matches(node, errors=xerrors, **x)
                           for x in anyOf):
                    errors += xerrors
            else:
                if not any(cls.node_matches(node, **x) for x in anyOf):
                    return False
        if name is not None and node.get("Name", None) != name:
            if not add_error(f"{node} name is not {name}"):
                return False
        if field is not None and field not in node:
            if not add_error(f"{node} is missing field \"{field}\""):
                return False
        if ((parameter is not None
             and not any(x["Key"] == parameter
                         for x in node.get("Parameters", [])))):
            if not add_error(f"{node} is missing parameter \"{parameter}\""):
                return False
        if internal:
            if not cls.node_matches(
                    node, **ApsimXEngine.get_field_metadata(internal)):
                return False
        if contains:
            if isinstance(contains, str):
                contains = [contains]
            if isinstance(contains, set):
                missing = (
                    list(contains)
                    if not any(k in node for k in contains)
                    else []
                )
            elif isinstance(contains, dict):
                missing = [k for k in contains.keys() if k not in node]
            else:
                assert isinstance(contains, list)
                missing = [k for k in contains if k not in node]
            if missing and not add_error(f"Missing {missing}"):
                return False
            if isinstance(contains, dict):
                for k, v in contains.items():
                    if k not in node:
                        continue
                    if node[k] != v and not add_error(
                            f"{k}: {node[k]} != {v}"):
                        return False
        if equals and node != equals:
            if not add_error(f"{node} != {equals}"):
                return False
        if fvalid and not fvalid(node):
            if not add_error(f"{node} fails function {fvalid}"):
                return False
        if ((calls and not (
                isinstance(node, dict)
                and any(f"{calls}(" in x
                        for x in node.get("CodeArray", []))))):
            if not add_error(f"{node} does not call \"{calls}\""):
                return False
        if nested:
            missing = [k for k in nested.keys() if k not in node]
            if missing and not add_error(f"Missing {missing}"):
                return False
            for k, v in nested.items():
                if k not in node:
                    continue
                if isinstance(errors, list):
                    verrors = []
                    cls.node_matches(node[k], errors=verrors, **v)
                    errors += [f"{k}: {x}" for x in verrors]
                elif not cls.node_matches(node[k], **v):
                    return False
        # if kwargs:
        #     missing = [k for k in kwargs.keys() if k not in node]
        #     if missing and not add_error(f"Missing {missing}"):
        #         return False
        #     for k, v in kwargs.items():
        #         if k not in node:
        #             continue
        #         if node[k] != v and not add_error(f"{k}: {node[k]} != {v}"):
        #             return False
        if errors:
            return False
        return True

    def findall_parameters(self, name: str, info: Optional[dict] = None,
                           **kwargs: Any) -> Iterator[dict]:
        r"""Find all parameters nodes in this file matching the
        parameter info.

        Args:
            name: Parameter name.
            info: Information about how to locate the parameter.
            **kwargs: Additional keyword arguments are passed to findall.

        Yields:
            dict: The nodes matching the parameter info.

        Raises:
            KeyError: If info not provided and name is not a valid
                parameter/action.

        """
        if info is None:
            info = ApsimXEngine.get_field_metadata(name, None)
            if info is None:
                if name in self.ACTION_NODES:
                    info = self.ACTION_NODES[name]
                else:
                    raise KeyError(f"No node registered for parameter "
                                   f"\"{name}\"")
        for node in self.findall(requirements=info, **kwargs):
            yield node

    def find_parameter(self, name: str,
                       add_missing: Optional[Union[bool, dict]] = False,
                       info: Optional[dict] = None,
                       **kwargs: Any) -> dict:
        r"""Find a parameter node in the file.

        Args:
            name: Parameter name.
            add_missing: If True or dict, the default for the parameter
                will be added if it cannot be located. If a dict is
                provided, the parameter default will be added to this if
                the parameter cannot be located.
            info: Information about how to locate the parameter.
            **kwargs: Additional keyword arguments are passed to find.

        Returns:
            dict: The node matching the specified name. Empty if no
                node can be found.

        Raises:
            KeyError: If required is True and the node cannot be located.

        """
        if info is None:
            info = ApsimXEngine.get_field_metadata(name, None)
            if info is None:
                if name not in self.ACTION_NODES:
                    if not kwargs.get("required", False):
                        return {}
                    raise KeyError(f"No node registered for parameter "
                                   f"\"{name}\"")
                info = self.ACTION_NODES[name]
        if info is False:
            raise KeyError(f"Ignored parameter \"{name}\"")
        if ((name in self.parameter_nodes
             and not kwargs.get("parent", False))):
            return self.parameter_nodes[name]
        node = None
        try:
            node = self.find(requirements=info, **kwargs)
            if add_missing and not node:
                assert not kwargs.get("parent", False)  # corner case
                node = self.add_parameter(
                    name, info=info, parent=(
                        add_missing if isinstance(add_missing, dict)
                        else None
                    ),
                )
            if node and not kwargs.get("parent", False):
                self.parameter_nodes[name] = node
            return node
        except KeyError as e:
            if add_missing and not node:
                assert not kwargs.get("parent", False)  # corner case
                node = self.add_parameter(name)
                if node:
                    return node
            raise KeyError(f"{name}: {e}")

    def findall(self, name: Optional[str] = None,
                current: Optional[dict] = None,
                parent: Optional[bool] = False,
                requirements: Optional[dict] = None) -> Iterator[dict]:
        r"""Find a node in the file.

        Args:
            name: Name of the node to find.
            current: The current node being searched.
            parent: If True, the parent node will be returned.
            requirements: Set of requirements that the node must
                satisfy (see node_matches for a description of the
                available options).

        Yields:
            dict: All nodes matching the specified name.

        """
        assert not isinstance(parent, dict)
        if current is None:
            current = self.contents
        current_node = (
            current if isinstance(current, ApsimXFileNode)
            else ApsimXFileNode(current)
        )
        for node in current_node.findall(
                name=name, requirements=requirements):
            if parent:
                yield node.parent.contents
            else:
                yield node.contents

    def find(self, name: Optional[str] = None,
             current: Optional[dict] = None,
             parent: Optional[bool] = False,
             required: Optional[bool] = False,
             requirements: Optional[dict] = None) -> dict:
        r"""Find a node in the file.

        Args:
            name: Name of the node to find.
            current: The current node being searched.
            parent: If True, the parent node will be returned.
            required: If True, an error will be raised if the node
                cannot be located.
            requirements: Set of requirements that the node must
                satisfy (see node_matches for a description of the
                available options).

        Returns:
            dict: The node matching the specified name. Empty if no
                node can be found.

        Raises:
            KeyError: If required is True and the node cannot be located.

        """
        assert not isinstance(parent, dict)
        if current is None:
            current = self.contents
        current_node = (
            current if isinstance(current, ApsimXFileNode)
            else ApsimXFileNode(current)
        )
        node = current_node.find(
            name=name, required=required,
            requirements=requirements)
        if node is None:
            return {}
        if parent and node.parent:
            node = node.parent
        return node.contents

    def _make_interactive(self, actions: list):
        r"""Modify this file to make it interactive.

        Args:
            actions: List of actions that should be enabled.

        """
        serialization = cfg["apsimx"].get("serialization", "json")
        sync = ApsimXFileNode.from_data(f"Synchroniser_{serialization}")
        field = self.find("Field", required=True)
        for k, v in self.ACTION_NODES.items():
            parent = None
            if "parent" in v and self.node_matches(field, **v["parent"]):
                parent = field
            if k not in actions:
                if v.get("required", False):
                    self.find_parameter(
                        k, current=parent, required=True,
                        add_missing=(parent if parent else True),
                    )
                else:
                    self.disable_action(k)
                continue
            self.enable_action(k, parent=parent)
        # Do sync last to avoid it being disabled
        field["Children"].append(copy.deepcopy(sync.contents))


_fertilizer_node = _read_resource("Fertiliser")


def _fertilizer_action(name: Optional[str] = None,
                       solutes: Optional[List[str]] = None,
                       **kwargs):
    r"""Construct an action specification for applying fertilizer.

    Args:
        name: Name of the fertilizer to apply.
        solutes: Solutes that the fertilizer must contain.
        **kwargs: Additional keywords are added to the action.

    Returns:
        dict: Action specification.

    """
    if name is None:
        fullname = "fertilizer"
    else:
        fullname = f"{name} fertilizer"
    if solutes:
        types = []
        for x in _fertilizer_node.get("Children", []):
            for i in range(1, 10):
                field = f"Solute{i}Name"
                if field not in x:
                    break
                if x[field] in solutes:
                    types.append(x["Name"])
    else:
        types = [x["Name"] for x in _fertilizer_node.get("Children", [])]
    out = {
        "description": (
            "Apply {amount} {amount_units} "
            + fullname + " in the form of {type}"
        ),
        "action_param": "amount",
        "param_desc": {
            "amount": {
                "type": "number",
                "units": "kg/ha",
                "min": 0.0,
                "max": 8.0,
            },
            "type": {
                "type": "string",
                "default": types[0] if types else name,
                "enum": types,
            },
            # "depth": {
            #     "type": "number",
            #     "units": "mm",
            # },
            # "depthBottom": {
            #     "type": "number",
            #     "units": "mm",
            # },
        },
    }
    out.update(**kwargs)
    return out


# connect -> ok
# paused -> resume/get/set
# finished -> ok
class ApsimXEngine(CropModelEngine):
    r"""Class for managing communication with an APSIMX server running
    in another process."""

    _MODEL_NAME: ClassVar[str] = "apsimx"
    _allow_bulk_set: ClassVar[bool] = True
    _allow_bulk_get: ClassVar[bool] = True
    _global_zmq_context: ClassVar[zmq.Context] = None
    MINIMUM_TIMESTEP: ClassVar[datetime.timedelta] = datetime.timedelta(
        days=1)
    STATUS_MESSAGES: ClassVar[list] = [
        "connect", "finished", "error", "recoverable_error",
    ]
    ERROR_MESSAGES: ClassVar[list] = [
        "error", "recoverable_error",
    ]
    INPUT_FILE_TYPE: ClassVar[Any] = ApsimXFile
    WEATHER_FILE_TYPE: ClassVar[Any] = ApsimXWeatherFile
    SOIL_FILE_TYPE: ClassVar[Any] = ApsimXSoilFile
    SERIALIZER: ClassVar[Callable] = staticmethod(
        msgpack.packb
        if cfg["apsimx"].get("serialization", "json") == "msgpack"
        else lambda x: yggrj.dumps(x).encode('utf8')
    )
    DESERIALIZER: ClassVar[Callable] = staticmethod(
        msgpack.unpackb
        if cfg["apsimx"].get("serialization", "json") == "msgpack"
        else lambda x: yggrj.loads(x.decode('utf8'),
                                   datetime_mode=yggrj.DM_ISO8601)
    )
    AVAILABLE_ACTION_MAP: ClassVar[dict] = {
        "sow": {
            "description": (
                "Sow a {crop_variety} {crop_name} crop at a density "
                "of {population} {population_units} with a sowing "
                "depth of {sowingDepth} {sowingDepth_units} and a row "
                "spacing of {rowSpacing} {rowSpacing_units}"
            ),
            "action_param": None,  # Boolean
            "param_desc": {
                "crop_name": {
                    "type": "string",
                },
                "crop_variety": {
                    "type": "string",
                    "default": "Hartog",
                },
                "population": {
                    "type": "number",
                    "units": "seeds per square meter",  # seeds/m²
                    "default": 5.0,
                },
                "sowingDepth": {
                    "type": "number",
                    "units": "mm",
                    "default": 50.0,
                },
                "rowSpacing": {
                    "type": "number",
                    "units": "mm",
                    "default": 1000.0,
                },
            },
            "num_levels": -1,  # Boolean
        },
        "harvest": {
            "description": "Harvest the {crop_name} crop",
            "action_param": None,  # Boolean
            "param_desc": {
                "crop_name": {
                    "type": "string",
                    "default": "",
                },
            },
            "num_levels": -1,  # Boolean
        },
        "tillage": {
            "description": "Till the field using {type} tillage",
            "action_param": "type",
            "param_desc": {
                "type": {
                    "type": "string",
                    "enum": ["chisel", "disc", "planter", "burn"],
                },
            },
        },
        "fertilize": _fertilizer_action(),
        "nitrogen": _fertilizer_action(
            "nitrogen", ["NO3", "NH4", "Urea"],
            alias="N",  # cost=0.46,  # $/kg
        ),
        "calcium": _fertilizer_action(
            "calcium", ["Ca"],
            alias="Ca",
        ),
        "phosphorus": _fertilizer_action(
            "phosphorus", ["RockP", "LabileP", "BandedP"],
            alias="P",
        ),
        "irrigate": {
            "alias": "water",
            "description": (
                "Irrigate with {amount} {amount_units} of water."
            ),
            "action_param": "amount",
            # 1 mm over 1 ha == 10000 L, $20 per 10000 L
            # "cost": 20.0,  # $/mm/ha
            "param_desc": {
                "amount": {
                    "type": "number",
                    "units": "mm",
                    "min": 0.0,
                    "max": 20.0,
                },
            },
        },
        "weed": {
            "description": (
                "Reduce a {weed_name} weed by removing "
                "{killFractionPlants} of the weed population and "
                "{killFractionSeeds} of the weed seeds"
            ),
            "action_param": "killFractionPlants",
            "param_desc": {
                "killFractionPlants": {
                    "type": "number",
                    "min": 0.0,
                    "max": 1.0,
                    "default": 0.9,
                },
                "weed_name": {
                    "type": "string",
                },
                "killFractionSeeds": {
                    "type": "number",
                    "min": 0.0,
                    "max": 1.0,
                    "default": 0.0,
                },
            },
        },
    }
    EXAMPLE_STATE: ClassVar[Tuple[str, float]] = (
        "[Grain].MaximumPotentialGrainSize.FixedValue",
        0.043,
    )
    EXAMPLE_ACTION: ClassVar[Tuple[str, dict]] = (
        "nitrogen", {"amount": 160.0},
    )
    _SIMULATOR_FIELD_ANNOTATIONS: ClassVar[dict] = {
        "duration": False,
        "season_length": False,
        "output_vars": {
            "contains": {
                "$type": "Models.Report, Models",
            },
            # "nested": {
            #     "EventNames": {"contains": "EndOfDay"},
            # },
            "field": "VariableNames",
            "fget": lambda x: [xx for xx in x if " as " not in xx],
        },
        "crop_name": {
            "anyOf": [
                {
                    "contains": {
                        "$type": "Models.PMF.Plant, Models"
                    },
                    "field": "Name",
                    "fvalid": lambda x: x.get("Description", None) != "Weed",
                },
                {
                    "contains": {"Name": "SowOrHarvestByDate"},
                    "parameter": "CropName",
                },
                {
                    "calls": "Sow",
                    "parameter": "Crop",
                },
                {
                    "calls": "Harvest",
                    "parameter": "Crop",
                },
            ],
            "fget": lambda x: x.lower(),
        },
        "crop_variety": {
            "contains": {
                "$type": "Models.Manager, Models",
            },
            "calls": "Sow",
            "parameter": "CultivarName",
            "fvalid": lambda x: (not x["Name"].endswith(
                "Weed Lifecycle")),
        },
        "weed_name": {
            "anyOf": [
                {
                    "contains": {
                        "$type": "Models.PMF.Plant, Models"
                    },
                    "field": "Name",
                    "fvalid": lambda x: x.get("Description", None) == "Weed",
                },
                {
                    "contains": {
                        "$type": "Models.Manager, Models"
                    },
                    "parameter": "WeedName",
                },
            ],
            "fget": lambda x: x.lower(),
        },
        "weed_variety": {
            "contains": {
                "$type": "Models.Manager, Models",
            },
            "parameter": "WeedCultivarName",
        },
        "year": {
            "fget": lambda x: x.year,
            "anyOf": [
                {"internal": "start_time",
                 "fset_prev": lambda x, prev: prev.replace(year=x)},
                {"internal": "end_time",
                 "fset_prev": lambda x, prev: prev.replace(year=x)},
                {"internal": "sow_date",
                 "fset_prev": lambda x, prev: prev.replace(year=x)},
                {"internal": "harvest_date",
                 "fset_prev": lambda x, prev: prev.replace(year=x)},
            ],
        },
        "start_time": {
            "$type": "Models.Clock, Models",
            "field": "Start",
            "fget": datetime.datetime.fromisoformat,
        },
        "end_time": {
            "$type": "Models.Clock, Models",
            "field": "End",
            "fget": datetime.datetime.fromisoformat,
        },
        "weather_file": {
            "$type": "Models.Climate.Weather, Models",
            "field": "FileName",
            "fget": _replace_root,
        },
        "soil_file": {
            "$type": "Models.Soils.Soil, Models",
            "fset_prev": ApsimXSoilFile.from_file_and_prev,
            "fget": lambda x: (
                ApsimXSoilFile.from_file(x)
                if isinstance(x, str)
                else ApsimXSoilFile(None, contents=x)
            ),
            "full_node": True,
        },
        "latitude": {
            "$type": "Models.Soils.Soil, Models",
            "field": "Latitude"
        },
        "longitude": {
            "$type": "Models.Soils.Soil, Models",
            "field": "Longitude",
        },
        "field_area": {
            "$type": "Models.Core.Zone, Models",
            "field": "Area",
        },
        "sow_date": {
            "parent": {"contains": {"Name": "Field"}},
            "contains": {"Name": "SowOrHarvestByDate"},
            "parameter": "SowingDate",
            "default": "SowOrHarvestByDate.json",
            "fget": datetime.datetime.fromisoformat,
            "conflicts": [
                {
                    "$type": "Models.Manager, Models",
                    "calls": "Sow",
                },
            ],
            "parameter_properties": {
                "CropName": "formal_crop_name",
                "CultivarName": "crop_variety",
            },
        },
        "harvest_date": {
            "parent": {"contains": {"Name": "Field"}},
            "contains": {"Name": "SowOrHarvestByDate"},
            "parameter": "HarvestDate",
            "default": "SowOrHarvestByDate.json",
            "fget": datetime.datetime.fromisoformat,
            "conflicts": [
                {
                    "$type": "Models.Manager, Models",
                    "calls": "Harvest",
                },
            ],
            "parameter_properties": {
                "CropName": "formal_crop_name",
                "CultivarName": "crop_variety",
            },
        },
    }

    from_example: Optional[bool | str | SkipJsonSchema[None]] = Field(
        default=False,
        description="If True, copy the bundled example for the crop to "
                    "use as the model file. If a string, the path to "
                    "the example file to copy.",
        json_schema_extra={"hidden_for_server": True},
    )

    def model_post_init(self, __context: Any) -> None:
        r"""Initialize the engine.

        Args:
            model_file: Path to a .apsimx model input file.
            **kwargs: Additional keyword arguments are passed to the
                CropModelEngine constructor.

        """
        self.context = None
        self.socket = None
        self.host = None
        self.port = None
        self.process = None
        self.stdout_pipe = None
        self.stderr_pipe = None
        self._status = None
        self._current_time = None
        super().model_post_init(__context)
        if self.output_vars:
            self.output_vars = [
                self.model._get_external_name(x) for x in
                self.output_vars
            ]
        if not self.output_dir:
            # ApsimX saves output to the directory containing the
            # model input file
            self.output_dir = os.path.dirname(self.model.fname)
        self.products += [
            self.output_file,
            f"{self.output_file}-shm",
            f"{self.output_file}-wal",
            os.path.splitext(self.output_file)[0] + ".Report.csv",
        ]

    @classmethod
    def library_directory(cls, dotnet_version: Optional[str] = None) -> str:
        r"""Path to the directory containing apsimx executables."""
        if dotnet_version is None:
            dotnet_version = cfg["apsimx"].get("dotnet_version", "10.0")
        # if platform.system() == "Windows":
        #     return os.path.join(
        #         cls.model_dir(), "bin", "x64", "Debug",
        #         f"net{dotnet_version}")
        return os.path.join(
            cls.model_dir(), "bin", "Debug", f"net{dotnet_version}")

    @classmethod
    def apsim_direct(cls, **kwargs) -> str:
        r"""Path to the apsimx models executable."""
        return os.path.join(cls.library_directory(**kwargs),
                            "Models.dll")

    @classmethod
    def apsim_srv(cls, **kwargs) -> str:
        r"""Path to the apsimx server executable."""
        return os.path.join(cls.library_directory(**kwargs),
                            "ApsimZMQServer.dll")

    @classmethod
    def is_installed(cls) -> bool:
        r"""Check if the model is installed in the specified directory.

        Returns:
            bool: True if the model is installed, False otherwise.

        """
        if not super().is_installed():
            return False
        if not os.path.isfile(cls.apsim_srv()):
            for dotnet_version in ["8.0", "10.0"]:
                fsrv = cls.apsim_srv(dotnet_version=dotnet_version)
                if os.path.isfile(fsrv):
                    logger.warning(
                        f"{cls._MODEL_NAME} exists for dotnet version "
                        f"{dotnet_version}, but that version is not "
                        f"specified in the configuration "
                        f"({cfg['apsimx']['dotnet_version']}). "
                        f"Update the config if you would like to use it."
                    )
                    break
        return os.path.isfile(cls.apsim_srv())

    @classmethod
    def download(cls, model_dir: Optional[str] = None,
                 **kwargs: Any) -> None:
        r"""Clone the simulator from a git repository.

        Args:
            model_dir: Path to the root directory where the model should
                be downloaded. Defaults to the auto-detected directory.
            \*\*kwargs: Additional keyword arguments are passed to
                utils.partialclone for .git URLs.

        """
        kwargs.setdefault(
            "patterns", [
                "APSIM.*",
                "CONTRIBUTING.md",
                "README.md",
                "LICENSE.md",
                "Models/",
                "Examples/*.apsimx",
                "Examples/WeatherFiles/",
                "DeepCloner.Core",
                # The following are required if other projects built
                # "ApsimNG",
                # "ApsimX.sln",
                "!Tests/",
                "!Tools/",
                "!Gtk.Sheet/",
            ]
        )
        super().download(model_dir=model_dir, **kwargs)

    @classmethod
    def _install(cls, model_dir: str) -> None:
        r"""Install APSIMX by building it from source.

        Args:
            model_dir: Path to the root directory of the APSIMX source
                checkout. Defaults to the auto-detected directory.

        """
        sln_file = os.path.join(
            model_dir, "APSIM.Server", "ZMQ+msgpack",
            "APSIM.ZMQServer.csproj")
        # sln_file = os.path.join(
        #     model_dir, "Models", "Models.csproj")
        # sln_file = os.path.join(model_dir, "ApsimX.sln")
        if not os.path.isfile(sln_file):
            raise RuntimeError(f"APSIMX solution does not "
                               f"exist: \"{sln_file}\"")
        logger.info(f"Building APSIMX from \"{sln_file}\"")
        if platform.system() == "Windows":
            noplatform = "/p:AppendPlatformToOutputPath=false"
        else:
            noplatform = "-p:AppendPlatformToOutputPath=false"
        subprocess.run(
            ["dotnet", "build", noplatform, sln_file],
            check=True)

    @classmethod
    def default_server_fields(cls) -> dict:
        r"""dict: The default fields that should be used for a server."""
        out = super().default_server_fields()
        out["actions"] = [
            k for k in out["actions"] if k not in ["sow", "harvest"]
        ]
        return out

    def create_model_file(self) -> CropModelFile:
        r"""Create a model input file.

        Returns:
            CropModelFile: Constructed model input file.

        """
        if self.from_example:
            if isinstance(self.from_example, str):
                src = self.from_example
            else:
                if not self.crop_name:
                    raise ValueError(
                        "Either model_file or crop_name must "
                        "be provided"
                    )
                src = self.INPUT_FILE_TYPE.find_example(self.crop_name)
            return self.INPUT_FILE_TYPE.from_example(
                src, dst=self.model_file,
                directory=self.output_dir,
                interactive=(not self.non_interactive),
                actions=list(self.actions.keys()),
            )
        return super().create_model_file()

    @property
    def is_running(self) -> bool:
        r"""bool: True if the model engine is still running."""
        return (self.process is not None
                and self.process.poll() is None)

    @property
    def is_operable(self) -> bool:
        r"""bool: True if the model engine is running and functioning."""
        if not super().is_operable:
            return False
        if self.non_interactive:
            return True
        return (self._status not in ["finished", "error", "terminated",
                                     "never connected"])

    @property
    def current_time(self) -> datetime.datetime:
        r"""datetime.datetime: Current simulation time."""
        if self.non_interactive:
            if self.process is not None and self.process.poll() == 0:
                self._current_time = self.end_time
        if self._current_time is None:
            if not self.is_operable:
                self._current_time = self.start_time
            else:
                self._current_time = self.get("[Clock].Today")
        return self._current_time

    @property
    def status(self) -> Optional[str]:
        r"""str: Current simulation status."""
        if self.non_interactive and self.process is not None:
            if self.process.poll() is None:
                self._status = "running"
            elif self.process.poll() == 0:
                self._status = "finished"
            else:
                self._status = "error"
        if self._status is None and self.socket is not None:
            self._status = self.socket.recv_string()
            if self._status == "paused":
                prev = self._current_time
                self._current_time = None
                self.current_time
                if self.is_operable:
                    logger.debug(
                        f"Simulation waiting at {self.current_time}")
                else:
                    self._current_time = prev
            elif self._status in self.STATUS_MESSAGES:
                out = self._status
                self.send_command("ok")
                return out
            else:
                raise NotImplementedError(f"Unsupported status message: "
                                          f"\"{self._status}\"")
        return self._status

    @classmethod
    def get_output_file(cls, model_file: str,
                        ext: Optional[str] = ".db") -> str:
        r"""Get the expected output file path based on the input
        model file path.

        Args:
            model_file: Input model file path.
            ext: File extension.

        Returns:
            str: The expected output file path.

        """
        return os.path.join(os.path.splitext(model_file)[0] + ext)

    @property
    def output_file(self) -> str:
        r"""str: Path to the .db output file that will be produced."""
        return self.get_output_file(self.model.fname)

    def get_results(self, return_dataframe: Optional[bool] = False) -> Any:
        r"""Get the simulation results.

        Args:
            return_dataframe: If True, return the results in a pandas
                dataframe. Otherwise, a JSON object will be returned.

        Returns:
            Simulation results.

        """
        if self.non_interactive:
            fname = self.get_output_file(self.model.fname, ".Report.csv")
        else:
            fname = self.output_file
        if not os.path.isfile(fname):
            return None
        import pandas as pd
        if fname.endswith(".csv"):
            df = pd.read_csv(fname)
        else:
            import sqlite3
            # This currently errors due to missing report
            conn = sqlite3.connect(self.output_file)
            df = pd.read_sql_query("SELECT * FROM Report", conn)
            conn.close()
        if return_dataframe:
            return df
        return df.to_json()

    def plot_output(self, axes, label: Optional[str] = None,
                    **kwargs: Any):
        r"""Plot the output from a simulation run.

        Args:
            axes: Matplotlib axes that the data should be plot on.
            label: Line label.
            \*\*kwargs: Additional keyword arguments are passed to
                the plot method.

        """
        import pandas as pd
        df = self.get_results(return_dataframe=True)
        t = pd.to_datetime(df["Clock.Today"])
        try_vars = ["Yield", f"[{self.crop_name.title()}].Total.Wt"]
        var = None
        for var in try_vars:
            if var in df:
                v = df[var]
                if var.endswith("].Total.Wt"):
                    v = 10 * v
                break
        else:
            raise RuntimeError("Could not determine a yield variable")
        if not axes.get_ylabel():
            axes.set_ylabel(var)
        if not label:
            label = f"{self.crop_name.title()} [{self.crop_variety}]"
        axes.plot(t, v, label=label, **kwargs)

    @classmethod
    def global_zmq_context(cls) -> zmq.Context:
        r"""Get a global zeromq context"""
        if cls._global_zmq_context is None:
            cls._global_zmq_context = zmq.Context()
            cls._global_zmq_context.set(zmq.MAX_SOCKETS, 8000)
            cls._global_zmq_context.setsockopt(zmq.LINGER, 0)
            # cls._global_zmq_context.setsockopt(zmq.IMMEDIATE, 0)
        return cls._global_zmq_context

    @classmethod
    def start_direct_subprocess(cls, model_file: str,
                                verbose: Optional[bool] = False,
                                ncpu: Optional[int] = None,
                                csv: Optional[bool] = False,
                                **kwargs) -> subprocess.Popen:
        r"""Start an apsim model in a subprocess.

        Args:
            model_file: Path to model input file.
            verbose: If True, the model should be run with verbose
                output.
            ncpu: Number of CPUs that the server should use.
            csv: Output to a CSV.
            \*\*kwargs: Additional keyword arguments are used to create
                the subprocess.

        Returns:
            subprocess.Popen: Subprocess with the model running.

        """
        cmd = [
            "dotnet", cls.apsim_direct(),
            model_file,
        ]
        if verbose:
            cmd += ["--verbose"]
        if ncpu:
            cmd += ["--cpu-count", str(ncpu)]
        if csv:
            cmd += ["--csv"]
        return utils.start_subprocess(cmd, **kwargs)

    @classmethod
    def start_server_subprocess(cls, model_file: str,
                                protocol: Optional[str] = "interactive",
                                host: Optional[str] = "127.0.0.1",
                                port: Optional[str | int] = None,
                                verbose: Optional[bool] = False,
                                ncpu: Optional[int] = None,
                                **kwargs) -> subprocess.Popen:
        r"""Start the apsim server in a subprocess.

        Args:
            model_file: Path to model input file.
            protocol: How the server should be run.
            host: ZeroMQ host.
            port: ZeroMQ port (required if portocol is "interactive").
            verbose: If the server should be run with verbose output.
            ncpu: Number of CPUs that the server should use.
            \*\*kwargs: Additional keyword arguments are used to create
                the subprocess.

        Returns:
            subprocess.Popen: Subprocess with the server running.

        """
        assert host and port
        cmd = [
            "dotnet", cls.apsim_srv(),
            "-f", model_file,
            "-P", protocol,
            "-a", host,
            "-p", str(port),
        ]
        assert protocol in ["oneshot", "interactive"]
        if verbose:
            cmd += ["-v"]
        if ncpu:
            cmd += ["-c", str(ncpu)]
        return utils.start_subprocess(cmd, **kwargs)

    def _start(self):
        r"""Start a listening server on a random port."""
        # if self.non_interactive:
        #     self.process = self.start_direct_subprocess(
        #         self.model.fname,
        self._current_time = None
        self._status = None
        logger.info(f"Running model \"{self.model.fname}\"")
        if not self.non_interactive:
            self.context = self.global_zmq_context()
            self.host = "127.0.0.1"
            if self.socket is None:
                self.socket = self.context.socket(zmq.REP)
                self.socket.bind(f"tcp://{self.host}:0")
                self.port = self.socket.getsockopt(
                    zmq.LAST_ENDPOINT).decode().split(":")[-1]
            logger.info(
                f"Listening on: {self.socket.getsockopt(zmq.LAST_ENDPOINT)}")
        use_pipes = (
            platform.system() != 'Windows' and not self.non_interactive
        )
        kws = {}
        if use_pipes:
            kws.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if self.non_interactive:
            self._current_time = self.start_time
            self.process = self.start_direct_subprocess(
                self.model.fname, csv=True,
                **kws)
        else:
            self.process = self.start_server_subprocess(
                self.model.fname,
                host=self.host,
                port=self.port,
                **kws)
        if use_pipes:
            self.stdout_pipe = LogPipe(
                self.process.stdout, prefix="APSIMX: ",
                level=self.model_log_level)
            self.stderr_pipe = LogPipe(
                self.process.stderr, prefix="APSIMX", level="ERROR")
        logger.info(f"Started APSIMX process id: {self.process.pid}")
        if self.non_interactive:
            logger.debug("APSIMX Start complete")
            return
        timeout = 10
        timewait = 0.01
        if platform.system() == 'Windows':
            timeout = 20
            timewait = 0.1
        tstart = time.time()
        while time.time() - tstart < timeout and self.is_running:
            try:
                self._status = self.socket.recv_string(
                    flags=zmq.NOBLOCK)
                break
            except zmq.ZMQError as e:
                if e.errno != zmq.EAGAIN:
                    raise
                time.sleep(timewait)
        if self._status != "connect":
            logger.error(f"Failed to connect after {timeout} seconds "
                         f"(status = {self._status})")
            self._status = "never connected"
            self.stop(cleanup=True)
            raise ModelEngineError("Failed to connect with the "
                                   "ApsimX ZMQ server")
        self.send_command("ok")
        if self.status != "paused":
            raise AssertionError(f"Server is not awaiting instructions: "
                                 f"status = \"{self.status}\"")
        if self.start_time is None:
            self.start_time = self.get("[Clock].Start")
        else:
            assert self.get("[Clock].Start") == self.start_time
        if self.end_time is None:
            self.end_time = self.get("[Clock].End")
        else:
            assert self.get("[Clock].End") == self.end_time
        logger.debug("APSIMX Start complete")

    def _stop(self):
        r"""Stop the listening server and close the communication port."""
        logger.debug(f"ApsimX _stop (is_operable = {self.is_operable}, "
                     f"is_running = {self.is_running}, "
                     f"is_complete = {self.is_complete}, "
                     f"current_time = {self._current_time})")
        if self.is_running and self.is_complete and self._status == "paused":
            self._resume(wait=True)
        if self.is_operable:
            try:
                record = ModelOperation(method="act", args=("terminate", ))
                with self.stop_on_error(record=record):
                    self._act("terminate", {})
                self._resume(wait=True)
                if self.status != "finished":
                    raise ValueError(
                        f"Status after terminate is \"{self.status}\" "
                        f"(is_complete = {self.is_complete})")
                self._status = "terminated"
            except ModelEngineError:
                pass
        logger.debug("Closing socket")
        if self.socket is not None:
            self.socket.close()
        logger.debug("Terminating process")
        try:
            if self.process is not None and self.process.poll() is None:
                timeout = (10 if platform.system() == 'Windows' else 1)
                logger.debug("Calling kill")
                utils.kill_subprocess(self.process, timeout=timeout)
                logger.debug("Kill returned")
                logger.debug(f"Poll = {self.process.poll()}")
                assert self.process.poll() is not None
            logger.debug("Process closed")
        finally:
            if self.stderr_pipe is not None:
                self.stderr_pipe.close()
            if self.stderr_pipe is not None:
                self.stderr_pipe.close()
            # if self.context is not None:
            #     self.context.destroy()
        if self._status != "error":
            self.context = None
            self.socket = None
            self.port = None
            self.process = None
            self.stdout_pipe = None
            self.stderr_pipe = None
        logger.debug("ApsimX _stop finished")

    def send_command(self, command: str, args: Optional[list] = None) -> None:
        r"""Send a command to the server process, e.g. resume/set/get.

        Args:
            command: Command to send.
            args: Additional arguments to send with the commaned.

        """
        assert self.status is not None
        self.socket.send_string(command, zmq.SNDMORE if args else 0)
        if args:
            for i, arg in enumerate(args):
                self.socket.send(self.SERIALIZER(arg),
                                 zmq.SNDMORE if i < len(args) - 1 else 0)
        if self.status != 'finished':
            self._status = None

    def recv_reply(self, unpack: Optional[bool] = False) -> Any:
        r"""Receive a reply from the server process.

        Args:
            unpack: If True, the message will be unpacked using msgpack.

        Returns:
            object: Received message.

        """
        if unpack:
            outb = self.socket.recv()
            try:
                out = self.DESERIALIZER(outb)
            except (msgpack.exceptions.ExtraData, yggrj.JSONDecodeError):
                out = outb.decode()
                if out not in self.STATUS_MESSAGES:
                    raise
        else:
            out = self.socket.recv_string()
        if out in self.STATUS_MESSAGES:
            self._status = out
            self.send_command("ok")
            if out == "recoverable_error":
                self._status = self.recv_reply()
                assert self._status == "paused"
        return out

    def check_paused(self) -> None:
        r"""Check that the simulation server is paused."""
        if self.status != "paused":
            raise ModelEngineError(
                f"Simulation is not paused (status = {self.status})"
            )

    @contextlib.contextmanager
    def stop_on_error(self, record: Optional[tuple] = None,
                      allow_error: Optional[bool] = False) -> Iterator[None]:
        r"""Context manager that stops the simulation on an error.

        Args:
            record: Action to log when successful.
            allow_error: If True, a RecoverableError error will not
                result in the simulation being stopped.

        """
        self.check_paused()
        with super().stop_on_error(record=record, allow_error=allow_error):
            status = self.status
            try:
                yield
            except RecoverableError:
                raise
            except BaseException:
                self._status = "error"
                raise
            self._status = status

    def _reply_error(self, reply):
        r"""Get the error class for the given error reply.

        Args:
            reply: Reply message received from the server.

        Returns:
            type: Error class to raise for the reply.

        """
        if reply == "recoverable_error":
            error_cls = RecoverableModelEngineError
        else:
            error_cls = ModelEngineError
        return error_cls

    def _get(self, name: str | list):
        r"""Send a request to get the current value of a simulation state
        variable.

        Args:
            name: Name(s) of variable to get the value of.

        Returns:
            object: Current variable value.

        """
        reply = None
        expect_list = isinstance(name, list)
        orig_names = (name if isinstance(name, list) else [name])
        names = []
        try:
            for name in orig_names:
                name = self.model._get_internal_name(name)
                names.append(name)
        except KeyError as e:
            raise InvalidActionError(e)
        self.send_command("get", names)
        reply = self.recv_reply(unpack=True)
        if reply in self.ERROR_MESSAGES:
            raise self._reply_error(reply)(
                f"get for \"{name}\" received error reply "
                f"\"{reply}\""
            )
        assert isinstance(reply, list) and len(reply) == len(names)
        out = {}
        for k, v in zip(orig_names, reply):
            if isinstance(v, msgpack.ext.Timestamp):
                v = v.to_datetime().replace(tzinfo=None)
            out[k] = v
        if not expect_list:
            return out[orig_names[0]]
        return out

    def _set(self, name: str | dict, value: Any = None) -> None:
        r"""Send a request to set simulation state variable(s).

        Args:
            name: Name of the variable to update.
            value: New value for the named variable.

        """
        if isinstance(name, dict):
            values = name
            assert value is None
        else:
            values = {name: value}
        args = []
        try:
            for name, value in values.items():
                name = self.model._get_internal_name(name)
                if isinstance(value, (datetime.datetime, datetime.date)):
                    value = value.isoformat()
                args += [name, value]
        except KeyError as e:
            raise InvalidActionError(e)
        self.send_command("set", args)
        reply = self.recv_reply()
        if reply != "ok":
            raise self._reply_error(reply)(
                f"set for {values} received non-ok reply "
                f"\"{reply}\""
            )

    def _act(self, action: str | dict, param: dict = None) -> None:
        r"""Perform an action.

        Args:
            name: Name of the action to perform.
            param: Action parameters.

        """
        if isinstance(action, dict):
            values = action
            assert param is None
        else:
            values = {action: param}
        args = []
        for action, param in values.items():
            args_flat = []
            for k, v in param.items():
                args_flat += [k, v]
            args += [action] + args_flat
        logger.debug(f"_act: {args}")
        self.send_command("act", args)
        logger.debug("_act: recv_reply")
        reply = self.recv_reply()
        logger.debug(f"_act: recv_reply returned {reply}")
        if reply != "ok":
            raise self._reply_error(reply)(
               f"act for {values} received non-ok reply "
               f"\"{reply}\""
            )

    def _resume(self, wait: Optional[bool] = False) -> None:
        r"""Resume the simulation.

        Args:
            wait: If True, wait for the simulation to pause.

        """
        self.check_paused()
        self.send_command("resume")
        if wait:
            self.status


class ApsimXLLMPromptGenerator(CropModelLLMPromptGenerator):
    r"""ApsimX LLM prompt generator."""

    # TODO: Verify units
    DEFAULT_DESC_MAP: ClassVar[dict] = {
        "[Clock].Today": (
            "Timeline", "Date/time"),
        "[CROP].LAI": (
            "Crop status", "Leaf area index"),
        "[CROP].Phenology.Zadok.Stage": (
            "Crop status", "Zadok developmental stage"),
        "[CROP].Phenology.CurrentStageName": (
            "Crop status", "Developmental stage name"),
        "[CROP].Total.Wt": (
            "Crop status", "Total crop mass (kg/ha)"),
        "[CROP].Grain.Total.Wt": (
            "Crop status", "Crop yield (g/m²)"),
        "[CROP].Grain.Protein": (
            "Crop status", "Crop grain protein content"),
        "[CROP].Grain.Size": (
            "Crop status", "Crop grain size"),
        "[CROP].Grain.Number": (
            "Crop status", "Crop grain number"),
        "[CROP].Grain.Total.N": (
            "Crop status", "Crop grain nitrogen content"),
        # "[CROP].Biomass.StorageWt": (
        #     "Crop status", "Storage organ dry matter (g/m²)"),
        "[CROP].AboveGround.Wt": (
            "Crop status", "Total above-ground crop mass"),
        "[CROP].AboveGround.N": (
            "Crop status", "Total above-ground crop nitrogen content"),
        "[Nutrient].TotalN.kgha": (
            "Soil nutrients", "Available soil nitrogen (kg/ha)"),
        "[Soil].Water.PAW": (
            "Soil & water",
            "Plant-available root-zone soil moisture (fraction)"),
        "[Fertiliser].NitrogenApplied": (
            "Cumulative actions",
            "Cumulative nitrogen applied so far (kg/ha)"),
        "[Irrigation].IrrigationApplied": (
            "Cumulative actions",
            "Cumulative irrigation depth applied so far (mm)"),
        "[Weather].Radn": (
            "Weather", "Daily solar radiation (MJ/m²/day)"),
        "[Weather].Tav": (
            "Weather", "Mean air temperature (°C)"),
        "[Weather].Rain": (
            "Weather", "Daily rainfall (mm)"),
        "[CROP].DaysAfterSowing": ("Timeline", "Days since sowing"),
    }


class ApsimXEnv(CropModelEnv):
    r"""ApsimX environment."""

    MODEL_ENGINE_CLASS: ClassVar[Any] = ApsimXEngine
    LLM_PROMPT_GENERATOR_CLASS: ClassVar[Any] = ApsimXLLMPromptGenerator
    DEFAULT_ACTIONS: ClassVar[list] = ["nitrogen", "irrigate"]
    DEFAULT_REVENUE_VAR: ClassVar[dict] = {
        "name": "[CROP].Grain.Total.Wt",
        # "name": "Yield",  # Only includes harvested weight
        # "cost": ??,  # $/kg/ha
    }
