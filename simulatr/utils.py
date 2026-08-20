import os
import threading
import subprocess
import signal
import logging
import typing
import argparse
import datetime
from functools import cached_property
from typing import Optional, Union, Any, List, ClassVar, Tuple, Dict
from pydantic import BaseModel, PrivateAttr, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema
from collections import OrderedDict
from io import BufferedReader
from . import logger
from .config import PackageConfig, _pkgdir


cfg = PackageConfig(
    'simulatr',
    defaults={
        'directories': {
            'output': os.path.join(os.getcwd(), 'output'),
            'models': os.path.join(os.getcwd(), 'models'),
            'apsimx': os.path.join(os.getcwd(), 'models', 'apsimx'),
            'scratch': os.path.join(os.getcwd(), 'scratch'),
            'nasa_power_weather_data': os.path.join(
                os.getcwd(), 'nasa_power_weather_data'),
            'isric_soil_data': os.path.join(
                os.getcwd(), 'isric_soil_data'),
        },
        'urls': {
            'n8n_api': "https://tools.uiuc.chat/api/v1",
        },
    },
)
cfg.setdefaults(
    directories={
        'source': _pkgdir,
        'apsimx_data': os.path.join(_pkgdir, 'apsimx_data'),
    },
)


def start_subprocess(*args, **kwargs) -> subprocess.Popen:
    r"""Start a subprocess, ensuring the correct flags are set so the
    process can be managed.

    Args:
        \*args, \*\*kwargs: All arguments and keyword arguments are
            passed to subprocess.Popen.

    Returns:
        subprocess.Popen: Subprocess.

    """
    # import platform
    # if platform.system() == 'Windows':
    #     if "creationflags" not in kwargs:
    #         kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    #     else:
    #         kwargs["creationflags"] |= subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(*args, **kwargs)


def kill_subprocess(process: subprocess.Popen, timeout: int = 1):
    r"""Kill a subprocess instance, first trying kill method, then
    falling back on SIGINT.

    Args:
        process: Subprocess instance.
        timeout: Number of seconds to wait after calling kill.

    """
    timeout_try = 0.1
    try:
        process.kill()
        process.wait(timeout=timeout_try)
        logger.info("Success via process.kill()")
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        process.send_signal(signal.CTRL_C_EVENT)
        process.wait(timeout=timeout_try)
        logger.info("Success via CTRL_C_EVENT")
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        process.send_signal(signal.CTRL_BREAK_EVENT)
        process.wait(timeout=timeout_try)
        logger.info("Success via CTRL_BREAK_EVENT")
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        import psutil
        psutil_process = psutil.Process(process.pid)
        for proc in psutil_process.children(recursive=True):
            proc.kill()
        psutil_process.kill()
        process.wait(timeout=timeout_try)
        logger.info("Success via psutil.kill()")
        return
    except subprocess.TimeoutExpired:
        pass
    process.wait(timeout=timeout)


def partialclone(repourl: str, dst: str = None,
                 patterns: List[str] = []):
    r"""Clone a git repository, only including certain files/directories.

    Args:
        repourl: Repository URL.
        dst: Directory that the repository should be cloned into.
        patterns: One or more patterns specifying which files/directories
            to include in the cloned repository.

    """
    import tempfile
    import subprocess
    if dst is None:
        dst = os.path.splitext(repourl.rsplit('/')[-1])[0]
    if not patterns:
        if not os.path.isdir(dst):
            subprocess.run(
                ["git", "clone", repourl, dst], check=True)
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        fpattern = os.path.join(tmpdir, "patterns.txt")
        with open(fpattern, "w") as fd:
            fd.write("\n".join(patterns))
        if not os.path.isdir(dst):
            subprocess.run(
                f"git clone --filter=blob:none --no-checkout {repourl} {dst}",
                shell=True, check=True)
            subprocess.run(
                "git sparse-checkout init --no-cone",
                shell=True, check=True, cwd=dst)
        cmds = [
            f"git sparse-checkout set --stdin < {fpattern}",
            "git read-tree -mu HEAD",
        ]
        for cmd in cmds:
            subprocess.run(cmd, shell=True, check=True,
                           cwd=dst)
    # import pprint
    # import glob
    # pprint.pprint(sorted(glob.glob(os.path.join(dst, "*"))))


def promptuser(prompt: str, _gha_default: str = ""):
    r"""Prompt for input from the user. Set to default if GITHUB_ACTIONS
    environment variable is set.

    Args:
        prompt: Prompt to provide the user with.
        _gha_default: Default when GITHUB_ACTIONS set.

    Returns:
        str: User response.

    """
    if os.environ.get("GITHUB_ACTIONS", None):
        return _gha_default
    return input(prompt)


class LogPipe(threading.Thread):
    r"""Thread to move output from a process PIPE to the logger.

    Args:
        pipe: Pipe that output should be streamed from.
        level: Integer logging level or the name of the logging level.
        prefix: Prefix to add to log messages.
        daemon: True if thread should be daemon.
        **kwargs: Additional keyword arguments are passed to the
            threading.Thread constructor.

    """

    def __init__(self, pipe: BufferedReader,
                 level: Optional[Union[str, int]] = "INFO",
                 prefix: Optional[str] = "",
                 daemon: Optional[bool] = True,
                 **kwargs: Any) -> None:
        r"""Initialize the LogPipe thread.

        Args:
            pipe: Pipe that output should be streamed from.
            level: Integer logging level or the name of the logging
                level.
            prefix: Prefix to add to log messages.
            daemon: True if thread should be daemon.
            **kwargs: Additional keyword arguments are passed to the
                threading.Thread constructor.

        """
        self.level = level
        self.prefix = prefix
        if isinstance(level, str):
            self.level = getattr(logging, level)
        self.pipe = pipe
        self.terminated = threading.Event()
        super(LogPipe, self).__init__(daemon=daemon, **kwargs)
        self.start()

    def close(self) -> None:
        r"""Close the pipe."""
        self.terminated.set()
        self.pipe.close()
        self.join()

    def run(self) -> None:
        r"""Run the thread, moving messages from the pipe to the
        logger."""
        for line0 in iter(self.pipe.readline, ''):
            line = line0.decode().strip('\n')
            if line:
                logger.log(self.level, self.prefix + line)
            if self.terminated.is_set():
                break
        self.terminated.set()


class SkipFieldType(BaseException):
    r"""Error to raise for fields that are skipped by a field handler."""
    pass


class FieldHandler(BaseModel):
    r"""Base class for performing operations for each field in a
    pydantic model subclass."""

    field_name: str = Field(description="Field name")
    field_info: Any = Field(description="Field information")
    skip_fields: Optional[List[str]] = Field(
        None,
        description="Names of fields to skip")
    only_fields: Optional[List[str]] = Field(
        None,
        description="Names of fields to include (others will be skipped)")
    skip_annotation_values: Optional[list] = Field(
        [argparse.SUPPRESS],
        description="Annotation values that should be skipped")
    skip_annotation_types: Optional[List[type]] = Field(
        [SkipJsonSchema],
        description="Annotation types that should be skipped")

    def skip_field(self, reason: str):
        r"""Raise a SkipFieldType error to skip this field.

        Args:
            reason: Reason field is skipped.

        Raises:
            SkipFieldType

        """
        raise SkipFieldType(f"{self.field_name}: {reason}")

    def model_post_init(self, __context: Any) -> None:
        r"""Initialize the handler."""
        self.skip_fields = (self.skip_fields or [])
        if self.field_name in self.skip_fields:
            self.skip_field("in skip_fields")
        if self.only_fields and self.field_name not in self.only_fields:
            self.skip_field("not in only_fields")
        if any(x in self.field_info.metadata for x in
               self.skip_annotation_values):
            self.skip_field("annotation in skip_annotation_values")
        return super().model_post_init(__context)

    def __call__(self, *args, **kwargs):
        r"""Perform operation for the field."""
        raise NotImplementedError

    @classmethod
    def handle_model(cls, model: Any, *args: Any, **kwargs: Any):
        r"""Call this handler for each fiedl on a model.

        Args:
            model: Pydantic model with fields that should be handled.
            \*args: Arguments to pass to the FieldSource handler
                __call__ method.
            \*\*kwargs: Additional keyword arguments are used to
                create a new FieldSource instance.

        Returns:
            The result of calling the FieldSource handler.

        """
        kwargs.setdefault("field_handler", cls)
        src = FieldSource(model=model, **kwargs)
        return src(*args)

    @cached_property
    def annotation_types(self) -> typing.Union[list, type]:
        r"""Type(s) indicated by the annotation after stripping
        skipped annotations and merging nested unions."""
        return self.extract_type(self.field_info.annotation)

    @cached_property
    def is_array(self) -> bool:
        r"""True if the type indicates the field expects a list."""
        return (typing.get_origin(self.annotation_types) == list)

    @cached_property
    def flattened_annotation(self) -> type:
        r"""Type annotation for the field after stripping skipped
        annotations and merging nested unions."""
        out = self.annotation_types
        if isinstance(out, list):
            return typing.Union[tuple(out)]
        return out

    @cached_property
    def enum(self) -> list:
        r"""list: Enumerated values allowed by the field."""
        if not self.field_info.json_schema_extra:
            return None
        if "enum" in self.field_info.json_schema_extra:
            return self.field_info.json_schema_extra["enum"]
        elif "enum" in self.field_info.json_schema_extra.get("items", {}):
            return self.field_info.json_schema_extra["items"]["enum"]
        return None

    def extract_type_list(self, args: list) -> list:
        r"""Extract type information from a list of type hints by
        extracting skipped annotations and merging nested unions.

        Args:
            args: Set of annotations to get types from.

        Returns:
            list: Flattened type hints.

        """
        out = []
        for x in args:
            try:
                xout = self.extract_type(x)
                if isinstance(xout, list):
                    out += [xx for xx in xout if xx not in out]
                else:
                    out.append(xout)
            except SkipFieldType:
                continue
        if not out:
            self.skip_field("Empty Union")
        return out

    def extract_type(self, annotation: type) -> type:
        r"""Extract type information from a type hint by
        extracting skipped annotations and merging nested unions.

        Args:
            annotaion: Type hint to extract a type from.

        Returns:
            list: Flattened type hints.

        """
        if annotation is type(None):
            self.skip_field("None")
        origin = typing.get_origin(annotation)
        if origin is None:
            return annotation
        args = typing.get_args(annotation)
        if origin == typing.Annotated:
            if isinstance(args[1], tuple(self.skip_annotation_types)):
                self.skip_field(f"{args[1]} in skip_annotation_types")
            if args[1] in self.skip_annotation_values:
                self.skip_field(f"{args[1]} in skip_annotation_values")
            return self.extract_type(args[0])
        elif origin == typing.Union:
            out = self.extract_type_list(args)
            if datetime.timedelta in out:
                for x in [int, float]:
                    if x in out:
                        out.remove(x)
            if len(out) == 1:
                return out[0]
            if len(out) == 2:
                if ((typing.get_origin(out[0]) == list
                     and out[0] == List[out[1]])):
                    return out[0]
                if ((typing.get_origin(out[1]) == list
                     and out[1] == List[out[0]])):
                    return out[1]
            return out
        elif origin == list:
            args = self.extract_type_list(args)
            return typing.List[tuple(args)]
        raise NotImplementedError(
            f"Extraction of type from annotation {annotation} "
            f"for field {self.field_name} (origin = {origin}, "
            f"args = {args})")


class FieldSource(BaseModel):
    r"""Wrapper for model to perform operations over fields on a
    pydantic model subclass."""

    model_config = ConfigDict(extra='allow')

    model: Any = Field(description="Model providing fields")
    field_handler: Optional[type] = Field(
        FieldHandler,
        description="Field handler")
    field_specific_kwargs: Optional[dict] = Field(
        None,
        description="Map of field specific keyword arguments that "
                    "should be passed to the field handler when the "
                    "field is handled")
    _args: PrivateAttr(default_factory=OrderedDict)

    def fields(self) -> typing.Iterator:
        r"""Field iterator."""
        for v in self._args.values():
            yield v

    def model_post_init(self, __context: Any) -> None:
        r"""Initialize the class by creating handlers for each field."""
        self._args = OrderedDict()
        field_order = getattr(self.model, "FORM_FIELD_ORDER", []).copy()
        field_order += [k for k in self.model.model_fields.keys()
                        if k not in field_order]
        for field_name in field_order:
            field_info = self.model.model_fields[field_name]
            try:
                field_inst = self.field_handler(
                    field_name=field_name,
                    field_info=field_info,
                    **self.model_extra,
                )
                self._args[field_name] = field_inst
            except SkipFieldType:
                continue
        return super().model_post_init(__context)

    def __call__(self, *args, **kwargs):
        r"""Call handler for each field."""
        for field in self.fields():
            field_kwargs = kwargs
            if ((self.field_specific_kwargs
                 and field.field_name in self.field_specific_kwargs)):
                field_kwargs = dict(
                    kwargs,
                    **self.field_specific_kwargs[field.field_name])
            try:
                field(*args, **field_kwargs)
            except SkipFieldType:
                continue


def create_registry_metaclass(key_attr: str | tuple = "_NAME",
                              base_type: type = None):
    r"""Class factor for creating a metaclass for registering
    classes.

    Args:
        key_attr: Attribute(s) that should be used to register classes.
        base_type: Type that classes using this metaclass will inherit
            from.

    Returns:
        type: New registr metaclass.

    """

    if base_type is None:
        meta_type = type
    else:
        meta_type = type(base_type)

    class RegistryMetaclass(meta_type):
        r"""Metaclass that registers subclasses."""

        _registry_attr: ClassVar[Optional[str | tuple]] = key_attr
        _registry: ClassVar[OrderedDict] = OrderedDict()

        def __new__(mcs, name: str, bases: Tuple[type, ...],
                    namespace: Dict[str, Any], **kwargs: Any):
            cls = super().__new__(mcs, name, bases, namespace, **kwargs)
            mcs._register(cls)
            return cls

        @classmethod
        def _get_key(mcs, cls):
            if isinstance(mcs._registry_attr, tuple):
                return tuple([getattr(cls, k, None)
                              for k in mcs._registry_attr])
            return (getattr(cls, mcs._registry_attr, None), )

        @classmethod
        def _register(mcs, cls):
            key = mcs._get_key(cls)
            if None in key:
                return
            dst = mcs._registry
            for k in key[:-1]:
                dst.setdefault(k, OrderedDict())
                dst = dst[k]
            assert key[-1] not in dst
            dst[key[-1]] = cls

        @classmethod
        def registered_classes(mcs) -> List[str]:
            r"""Get the names of all registered classes.

            Returns:
                List[str]: Names of the registered classes.

            """
            return sorted(mcs._registry)

        @classmethod
        def get_class(mcs, *key: tuple) -> type:
            r"""Get the registered class associated with a registry
            key.

            Args:
                key: Registry key.

            Returns:
                type: Registered class.

            """
            out = mcs._registry
            for k in key:
                out = out[k]
            return out

    return RegistryMetaclass
