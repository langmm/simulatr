# python -m simulatr run apsimx --crop-name wheat --from-example
import os
import json
import argparse
import logging
import typing
import datetime
from functools import cached_property
from typing import Any, Optional, List
from . import (
    logger, registered_simulators, get_simulator_class,
)
from .utils import cfg, FieldHandler, create_registry_metaclass


class OverrideExtendAction(argparse.Action):
    r"""Action class to prevent extending default values."""

    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(self, '_overwritten', False):
            current_values = getattr(namespace, self.dest)
            current_values.extend(values)
        else:
            setattr(namespace, self.dest, list(values))
            setattr(self, '_overwritten', True)


class CliArgHandler(FieldHandler):

    @cached_property
    def names(self) -> list:
        r"""list: Argument names for the field"""
        out = [self.field_name]
        if self.field_info.alias:
            out.append(self.field_info.alias)
        out = [x.replace("_", "-") for x in out]
        if not self.field_info.is_required():
            out = ["--" + x for x in out]
        return tuple(out)

    def type_kwargs(self, annotation: type) -> dict:
        r"""Get the kwargs defining the type for a command line argument
        that should accept the provided annotation.

        Args:
            annotation: Type hint.

        Returns:
            dict: Keyword arguments for add_argument defining the type.

        """
        types = self.extract_type(annotation)
        if isinstance(types, list):
            if len(types) == 2 and bool in types:
                other = types[1] if types[0] == bool else types[0]
                out = self.type_kwargs(other)
                assert ("type" in out and "nargs" not in out
                        and "action" not in out)
                out.update(
                    nargs="?",
                    const=True,
                )
                if not self.field_info.default:
                    out["default"] = False
                return out
            elif datetime.timedelta in types:
                for x in [int, float]:
                    if x in types:
                        types.remove(x)
            if isinstance(types, list) and len(types) == 1:
                types = types[0]
            if isinstance(types, list):
                raise RuntimeError(
                    f"More than one type for {self.field_name}: "
                    f"{types}")
        if types == bool:
            return {"action": "store_true"}
        elif types in [str, int, float]:
            return {"type": types}
        elif types in [datetime.datetime, datetime.date]:
            return {"type": str}
        elif types in [datetime.timedelta]:
            return {"type": float}
        elif typing.get_origin(types) is list:
            args = typing.get_args(types)
            if len(args) != 1:
                raise RuntimeError(
                    f"More than one type in list for "
                    f"{self.field_name}: {args}")
            out = self.type_kwargs(args[0])
            assert "type" in out and "nargs" not in out and "action" not in out
            out.update(
                nargs="+",
                action=OverrideExtendAction,
            )
            return out
        raise NotImplementedError(
            f"Handling of type {types} for field {self.field_name}")

    def __call__(self, parser: argparse.ArgumentParser, **kws):
        r"""Add an argument to the parser for this field.

        Args:
            parser: Parse to add the argument to.
            \*\*kwargs: Additional keyword arguments are passed to
                add_argument.

        """
        kwargs = (
            {} if "type" in kws
            else self.type_kwargs(self.field_info.annotation)
        )
        if self.field_info.description:
            kwargs["help"] = self.field_info.description
        if self.field_info.default is not None:
            kwargs["default"] = self.field_info.default
        if self.enum is not None:
            kwargs["choices"] = self.enum
        kwargs.update(**kws)
        names = kwargs.pop("names", self.names)
        parser.add_argument(
            *names,
            **kwargs
        )

    @classmethod
    def add_subparser(cls, root_parser: argparse.ArgumentParser,
                      name: str, model: Any,
                      skip_fields: Optional[List[str]] = None,
                      only_fields: Optional[List[str]] = None,
                      field_specific_kwargs: Optional[dict] = None,
                      **kwargs) -> argparse.ArgumentParser:
        r"""Add a subparser with arguments based on a pydantic model's
        fields.

        Args:
            root_parser: Parser that the subparser should be added to.
            name: Name for the subparser.
            model: Pydantic models with fields that should be added to
                the subparser as arguments.
            skip_fields: Set of fields that should not be added.
            only_fields: Set of fields that should be added.
            field_specific_kwargs: Mapping of argument keyword args
                that should be overridden for each field.
            \*\*kwargs: Additional keyword arguments are passed to the
                call to add_parser.

        Returns:
            argparse.ArgumentParser: Subparser.

        """
        parser = root_parser.add_parser(name, **kwargs)
        cls.handle_model(
            model, parser,
            skip_fields=skip_fields,
            only_fields=only_fields,
            field_specific_kwargs=field_specific_kwargs,
        )
        return parser


def _add_args_from_engine(simulator: str,
                          root_parser: argparse.ArgumentParser,
                          include_all_crop_name: Optional[bool] = False,
                          **kwargs: Any) -> argparse.ArgumentParser:
    engine = get_simulator_class(simulator)
    if "crop_name" not in kwargs.get("field_specific_kwargs", {}):
        kwargs.setdefault("field_specific_kwargs", {})
        kwargs["field_specific_kwargs"]["crop_name"] = {
            "type": str.lower,
            "choices": [
                x.lower() for x in
                engine.INPUT_FILE_TYPE.available_crops()
            ],
        }
        if include_all_crop_name:
            kwargs["field_specific_kwargs"]["crop_name"]["choices"].insert(
                0, "all")
    return CliArgHandler.add_subparser(
        root_parser, simulator, engine, **kwargs)


CliRegistry = create_registry_metaclass("NAME")


class SimulatrCliSubparser(metaclass=CliRegistry):

    NAME = None
    HELP = None
    KWARGS = {}

    def __init__(self, subparsers):
        self.subparsers = {}
        self.parser = subparsers.add_parser(
            self.NAME, help=self.HELP, **self.KWARGS)
        self.add_arguments(self.parser)
        if self.subparsers:
            for v in self.subparsers.values():
                self.add_global_arguments(v)
        else:
            self.add_global_arguments(self.parser)
        self.parser.set_defaults(func=self)

    def __call__(self, args):
        raise NotImplementedError

    def add_global_arguments(self, parser):
        parser.add_argument(
            "--pdb", action="store_true",
            help="Run with Python debugger",
        )

    def add_arguments(self, parser):
        pass


class ConfigCli(SimulatrCliSubparser):

    NAME = "config"
    HELP = "Set a configuration option"

    def add_arguments(self, parser):
        parser.add_argument(
            "section", type=str,
            help="Name of the section the configuration option is in",
        )
        parser.add_argument(
            "name", type=str,
            help="Name of the option to set",
        )
        parser.add_argument(
            "value", type=str,
            help="Value that the option should be set to",
        )
        parser.add_argument(
            "--level", type=str, choices=["env", "local", "user"],
            help=(
                "Level of the config file that the option should be "
                "set in"
            ),
        )
        super().add_arguments(parser)

    def __call__(self, args):
        if args.section in ["directories", "files"]:
            args.value = os.path.abspath(os.path.expanduser(args.value))
        cfg.set(args.section, args.name, args.value)
        dst = cfg.write(level=args.level)
        logger.info(f"Set {args.name} in the \"{args.section}\" section"
                    f" of \"{dst}\" to \"{args.value}\" ")


class InstallCli(SimulatrCliSubparser):

    NAME = "install"
    HELP = "Install a simulator"

    def __init__(self, *args, **kwargs):
        self.simulators = registered_simulators()
        super().__init__(*args, **kwargs)

    def add_arguments(self, parser):
        parser.add_argument(
            "--simulator", type=str,  nargs="+", action="extend",
            choices=self.simulators,
            help="Name(s) of simulators to install. If not provided, "
                 "all of the registered simulators will be installed.",
        )
        parser.add_argument(
            "--directory", type=str,
            help="Directory where the simulator should be installed. "
                 "Cannot be used if more than one simulator is specified.",
        )
        parser.add_argument(
            "--always-yes", action="store_true",
            help="Don't ask the user to approve the install",
        )
        parser.add_argument(
            "--force", action="store_true",
            help="Force reinstallation of the simulator even if it is "
                 "already installed",
        )
        super().add_arguments(parser)

    def __call__(self, args):
        if not args.simulator:
            args.simulator = self.simulators
        if args.directory:
            if len(args.simulator) > 1:
                raise RuntimeError(
                    f"Cannot specify an install directory for more "
                    f"than one simulator ({len(args.simulator)} "
                    f"specified, {args.simulator})")
            cfg.set("directories", args.simulator[0], args.directory)
        for simulator in args.simulator:
            logger.info(f"Installing {simulator} simulator...")
            engine_cls = get_simulator_class(simulator)
            engine_cls.install(always_yes=args.always_yes,
                               force=args.force)


class CreateModelFileCli(SimulatrCliSubparser):

    NAME = "create"
    HELP = "Create a simulator input file"

    def __init__(self, *args, **kwargs):
        self.simulators = registered_simulators()
        super().__init__(*args, **kwargs)

    def add_arguments(self, parser):
        parser_create_sim = parser.add_subparsers(
            dest="simulator", required=True,
            help="Name of the simulator to create an input file for",
        )
        for k in self.simulators:
            self.subparsers[k] = _add_args_from_engine(
                k, parser_create_sim,
                help=f"Create an {k} input file",
                skip_fields=["overwrite"],
            )
        # self.subparsers["apsimx"].add_argument(
        #     "crop_name", type=str.lower,
        #     choices=[
        #         x.lower() for x in
        #         get_simulator_class("apsimx", "file").available_crops()
        #     ],
        #     help="Crop name to create an input file for",
        # )
        super().add_arguments(parser)

    def add_global_arguments(self, parser):
        parser.add_argument(
            "--dst", type=str,
            help="Path to where the new file should be saved",
        )
        parser.add_argument(
            "--overwrite", action="store_true",
            help="Overwrite any existing file",
        )
        super().add_global_arguments(parser)

    def __call__(self, args):
        from . import get_simulator_class
        engine_cls = get_simulator_class(args.simulator)
        kws = {
            k: v for k, v in vars(args).items()
            if k not in ["action", "simulator", "dst", "overwrite", "pdb"]
        }
        if args.dst and os.path.isfile(args.dst):
            if not args.overwrite:
                raise RuntimeError(f"Model file already exists: "
                                   f"\"{args.dst}\"")
            os.remove(args.dst)
        if args.dst:
            if args.model_file:
                raise RuntimeError("Cannot provide both \"--model-file\" "
                                   "and \"--dst\"")
            kws["model_file"] = args.dst
        engine = engine_cls(**kws)
        engine.model.generated = False  # Prevent cleanup
        logger.info(f"Created input file \"{engine.model.fname}\"")


class RunCli(SimulatrCliSubparser):

    NAME = "run"
    HELP = "Run a simulation"

    def __init__(self, *args, **kwargs):
        self.simulators = registered_simulators()
        super().__init__(*args, **kwargs)

    def add_arguments(self, parser):
        self.subparser = parser.add_subparsers(
            dest="simulator", required=True,
            help="Name of the simulator to run")
        for k in self.simulators:
            self.subparsers[k] = _add_args_from_engine(
                k, self.subparser,
                help=f"Run a {k} simulation",
                include_all_crop_name=True,
                field_specific_kwargs={
                    "timestep": {"default": 0},
                },
            )
        super().add_arguments(parser)

    def add_global_arguments(self, parser):
        parser.add_argument(
            "--plot", type=str, nargs="?", const=True, default=False,
            help="File where results should be plot",
        )
        parser.add_argument(
            "--log-file", type=str, nargs="?", const=True,
            help="File where log message should be written",
        )
        parser.add_argument(
            "--log-level", choices=[
                "NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
            ],
            help="Logging level", default="INFO",
        )
        super().add_global_arguments(parser)

    def __call__(self, args):
        from . import run
        kws = {
            k: v for k, v in vars(args).items()
            if k not in ["action", "simulator", "log_file", "log_level",
                         "pdb"]
        }
        if getattr(args, "log_file", None) is True:
            log_file = args.simulator
            if getattr(args, "model_file", None):
                log_file += "_" + os.path.splitext(
                    os.path.basename(args.model_file))[0]
            elif ((args.simulator == "apsimx"
                   and getattr(args, "crop_name", None))):
                log_file += "_" + args.crop_name
            args.log_file = os.path.join(os.getcwd(), log_file + ".log")
        if getattr(args, "log_file", None):
            logger.info(f"Log being written to \"{args.log_file}\"")
        logging.basicConfig(filename=args.log_file,
                            level=getattr(logging, args.log_level))
        run(args.simulator, **kws)


class ServeCli(SimulatrCliSubparser):

    NAME = "serve"
    HELP = "Launch simulator(s) as fastapi application"

    def __init__(self, *args, **kwargs):
        self.installed_simulators = registered_simulators(
            only_installed=True)
        super().__init__(*args, **kwargs)

    def add_arguments(self, parser):
        parser.add_argument(
            "--simulator", type=str, nargs="+",
            action=OverrideExtendAction,
            choices=self.installed_simulators,
            default=self.installed_simulators,
            help=(
                "Name of the simulator(s) to create application endpoints "
                "for. If not specified, all of the installed simulators "
                "will be included."
            )
        )
        parser.add_argument(
            "--port", type=int, default=5000,
            help="Port that application should be served on",
        )
        parser.add_argument(
            "--host", type=str, default="0.0.0.0",
            help="Host address",
        )
        parser.add_argument(
            "--log-file", type=str,
            help="File where log message should be written",
        )
        parser.add_argument(
            "--log-level", choices=[
                "NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
            ],
            help="Logging level", default="INFO",
        )
        parser.add_argument(
            "--allow-shutdown", action="store_true",
            help="Include an endpoint to allow shutdown from the client",
        )
        super().add_arguments(parser)

    def __call__(self, args):
        from .server import run_server
        run_server(
            args.simulator,
            host=args.host, port=args.port,
            log_file=args.log_file,
            log_level=args.log_level,
            allow_shutdown=args.allow_shutdown,
        )


class N8NCli(SimulatrCliSubparser):

    NAME = "n8n"
    HELP = "Manage n8n tools for supported simulators"

    def __init__(self, *args, **kwargs):
        # from server import EndPointRegistry
        # self.n8n_entry_points = list(
        #     EndPointRegistry._registry.keys())
        self.simulators = registered_simulators()
        self.n8n_entry_points = ["start", "start-interactive"]
        super().__init__(*args, **kwargs)

    def add_arguments(self, parser):
        parser.add_argument(
            "simulator", type=str, choices=self.simulators,
            help="Name of the simulator to manage n8n tools for",
        )
        self.subparser = parser.add_subparsers(
            dest="utility", required=True,
            help="Name of the utility to run.")
        self.subparsers["create"] = self.subparser.add_parser(
            "create", help="Create an n8n tool")
        self.subparsers["update"] = self.subparser.add_parser(
            "update", help="Update an existing tool")
        self.subparsers["remove"] = self.subparser.add_parser(
            "remove", help="Remove an n8n tool")
        self.subparsers["query"] = self.subparser.add_parser(
            "query", help="Query an n8n tool")
        for k, parser_x in self.subparsers.items():
            if k == "query":
                continue
            parser_x.add_argument(
                "--dry-run", action="store_true",
                help="Don't actually do anything, just show requests.",
            )
        for k in ["create", "update"]:
            parser_x = self.subparsers[k]
            parser_x.add_argument(
                "--publish-for-address", type=str,
                help=(
                    "Address for the service that should be used in the "
                    "published tool"
                ),
            )
            parser_x.add_argument(
                "--overwrite", action="store_true",
                help="Overwrite any existing tool",
            )
            parser_x.add_argument(
                "--output-request", type=str, nargs='?', const=True,
                help="Output the tool creation request to a file",
            )
            parser_x.add_argument(
                "--output-form", type=str, nargs='?', const=True,
                help="Output the form for a tool to a file",
            )
        self.subparsers["create"].add_argument(
            "--update", action="store_true",
            help="Update any existing tool",
        )
        super().add_arguments(parser)

    def add_global_arguments(self, parser):
        parser.add_argument(
            "--name", type=str, nargs="+", action="extend",
            choices=self.n8n_entry_points,
            help="Name of the entry point(s)",
        )
        parser.add_argument(
            "--toolname", type=str,
            help="Name of the tool (single name must be provided)",
        )
        parser.add_argument(
            "--output-tool", "--output", type=str, nargs='?', const=True,
            help="Output the tool summary to a file",
        )
        parser.add_argument(
            "--verbose", action="store_true",
            help="Print all REST API responses",
        )
        super().add_global_arguments(parser)

    def __call__(self, args):
        from . import n8n
        if not args.name:
            if args.utility in ["query", "remove"] and args.toolname:
                args.name = [""]  # Won't be used
            else:
                args.name = self.n8n_entry_points
        if len(args.name) > 1:
            assert not args.toolname
            for k in ["output_request", "output_tool", "output_form"]:
                assert not isinstance(getattr(args, k, None), str)
        if args.utility in ["create", "update"]:
            if not args.publish_for_address:
                args.publish_for_address = os.environ.get(
                    "SIMULATR_REMOTE_SERVER_ADDRESS", None)
        if args.utility == "update":
            args.update = "required"
        for name in args.name:
            logger.info(f"{args.utility} {name}")
            if args.utility in ["create", "update"]:
                n8n.publish_n8n_service(
                    args.simulator, name,
                    service_address=args.publish_for_address,
                    toolname=args.toolname,
                    overwrite=args.overwrite,
                    update=args.update,
                    dry_run=args.dry_run,
                    output_request=args.output_request,
                    output_tool=args.output_tool,
                    output_form=args.output_form,
                    verbose=args.verbose,
                )
            elif args.utility == "remove":
                n8n.remove_n8n_service(
                    args.simulator, name,
                    toolname=args.toolname,
                    output=args.output_tool,
                    dry_run=args.dry_run,
                    verbose=args.verbose,
                )
            elif args.utility == "query":
                response = n8n.query_n8n_service(
                    args.simulator, name,
                    toolname=args.toolname,
                    output=args.output_tool,
                    allow_multiple=True,
                    verbose=args.verbose,
                )
                if not args.output_tool:
                    logger.info(json.dumps(response, indent=2))
            else:
                raise NotImplementedError(
                    f"n8n utility = \"{args.utility}\"")


class ProfileCli(SimulatrCliSubparser):

    NAME = "profile"
    HELP = "Profile simulatr components"

    def add_arguments(self, parser):
        from . import profile
        self.subparser = parser.add_subparsers(
            dest="target", required=True,
            help="Component(s) that should be profiled.",
        )
        self.subparsers["all"] = CliArgHandler.add_subparser(
            self.subparser, "all", profile.TargetBaseClass,
            help="Profile all targets")
        for k, v in profile.TargetRegistry._registry.items():
            self.subparsers[k] = CliArgHandler.add_subparser(
                self.subparser, k, v,
                help=v._DESCRIPTION)
        super().add_arguments(parser)

    def __call__(self, args):
        from .profile import TargetRegistry
        if args.target == "all":
            targets = TargetRegistry.registered_classes()
        else:
            targets = [args.target]
        kws = {
            k: v for k, v in vars(args).items()
            if k not in ["action", "target", "pdb"]
        }
        for target in targets:
            profiler = TargetRegistry.get_class(target)(**kws)
            profiler.run()


class ApsimXCli(SimulatrCliSubparser):

    NAME = "apsimx"
    HELP = "Run ApsimX model directly"

    def add_arguments(self, parser):
        from .apsimx import ApsimXEngine
        CliArgHandler.handle_model(
            ApsimXEngine, parser,
            only_fields=[
                "model_file", "crop_name", "crop_variety",
            ],
            field_specific_kwargs={
                "model_file": {
                    "type": str,  # "default": None,
                    # "names": ("model_file", ),
                },
            },
        )
        super().add_arguments(parser)

    def __call__(self, args):
        from .apsimx import ApsimXEngine
        generated = False
        if not args.model_file:
            assert args.crop_name
            generated = True
            fout = ApsimXEngine.INPUT_FILE_TYPE.from_crop_name(
                args.crop_name,  # crop_variety=args.crop_variety,
            )
            fout.write()
            fout.generated = False
            args.model_file = fout.fname
        try:
            process = ApsimXEngine.start_direct_subprocess(
                args.model_file,
                csv=True,
            )
            process.wait(60)
        finally:
            if generated and os.path.isfile(args.model_file):
                os.remove(args.model_file)


def main() -> None:
    r"""Run the command line interface."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="action", required=True,
        help='Action to perform')
    subparsers_map = {}
    for k, v in CliRegistry._registry.items():
        subparsers_map[k] = v(subparsers)
    # Parse
    args = parser.parse_args()
    try:
        args.func(args)
    except BaseException:
        if args.pdb:
            import pdb
            pdb.set_trace()
        raise
