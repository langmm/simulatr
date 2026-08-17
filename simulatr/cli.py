# python -m simulatr run apsimx --crop-name wheat --from-example
import os
import json
import argparse
import logging
from typing import Any, Optional, List
from . import (
    logger, registered_simulators, get_simulator_class, n8n, server,
)
from .utils import cfg
from pydantic_settings import CliSettingsSource


def _add_args_from_engine(simulator: str,
                          root_parser: argparse.ArgumentParser,
                          omit_fields: Optional[List[str]] = None,
                          **kwargs: Any) -> CliSettingsSource:
    # TODO:
    # - Skip fields
    # - Use extend for lists
    # - Allow for choices
    parser = root_parser.add_parser(simulator, **kwargs)
    engine = get_simulator_class(simulator)
    return CliSettingsSource(
        engine,
        root_parser=parser,
        cli_parse_args=False,
        cli_hide_none_type=True,
        cli_kebab_case=True,
    )


def run(simulator: str, timestep: int = 0,
        **kwargs: Any) -> None:
    r"""Run a simulation.

    Args:
        simulator: Name of the simulator to run.
        timestep: Time between actions (in days). 0 for continuous.
        **kwargs: Additional keyword arguments are passed along to
            the environment class constructor.

    """
    if timestep > 0:
        kwargs["intervention_interval"] = timestep
    env_cls = get_simulator_class(simulator, "env")
    env = env_cls.create_interactive_for_human(**kwargs)
    try:
        if timestep > 0:
            # Stop at each timestep to ask the user what action to take
            # using a prompt generated from the current observation.
            logger.info(
                "Simulation will pause at each timestep to ask the "
                "user for an action")
            env.run_interactive_for_human()
        else:
            # Run continuously to completion without intervention.
            logger.info("Running the simulation continuously")
            env.reset()
            env.model.fast_forward()
    finally:
        env.close()
    print(f"Output written to {env.model.output_file}")


def main() -> None:
    r"""Run the command line interface."""
    simulators = registered_simulators()
    installed_simulators = registered_simulators(only_installed=True)
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="action", required=True,
        help='Action to perform')
    # For setting configuration options
    parser_config = subparsers.add_parser(
        "config", help="Set a configuration option",
    )
    parser_config.add_argument(
        "section", type=str,
        help="Name of the section that the configuration option is in",
    )
    parser_config.add_argument(
        "name", type=str,
        help="Name of the option to set",
    )
    parser_config.add_argument(
        "value", type=str,
        help="Value that the option should be set to",
    )
    parser_config.add_argument(
        "--level", type=str, choices=["env", "local", "user"],
        help="Level of the config file that the option should be set in",
    )
    # Installation
    parser_install = subparsers.add_parser(
        "install", help="Install a simulator"
    )
    parser_install.add_argument(
        "simulator", type=str,  # choices=
        help="Name of the simulator to install",
    )
    parser_install.add_argument(
        "--directory", type=str,
        help="Directory where the simulator should be installed.",
    )
    parser_install.add_argument(
        "--always-yes", action="store_true",
        help="Don't ask the user to approve the install",
    )
    # For creating model input files
    parser_create = subparsers.add_parser(
        "create", help="Create a simulator input file"
    )
    parser_create_sim = parser_create.add_subparsers(
        dest="simulator", required=True,
        help="Name of the simulator to create an input file for",
    )
    create_parsers = {}
    for k in simulators:
        create_parsers[k] = _add_args_from_engine(
            k, parser_create_sim,
            # TODO: omit_fields=[],
            help=f"Create an {k} input file",
        )
    # create_parsers["apsimx"].add_argument(
    #     "crop_name", type=str.lower,
    #     choices=[
    #         x.lower() for x in
    #         get_simulator_class("apsimx", "file").available_crops()
    #     ],
    #     help="Crop name to create an input file for",
    # )
    for setting_x in create_parsers.values():
        parser_x = setting_x.root_parser
        parser_x.add_argument(
            "--dst", type=str,
            help="Path to where the new file should be saved",
        )
        parser_x.add_argument(
            "--overwrite", action="store_true",
            help="Overwrite any existing file",
        )
    # For running
    parser_run = subparsers.add_parser(
        "run", help="Run a simulation"
    )
    parser_run_sim = parser_run.add_subparsers(
        dest="simulator", required=True,
        help="Name of the simulator to run")
    run_parsers = {}
    for k in simulators:
        run_parsers[k] = _add_args_from_engine(
            k, parser_run_sim,
            help=f"Run a {k} simulation",
        )
        parser_x = run_parsers[k].root_parser
        parser_x.add_argument(
            "--log-file", type=str, nargs="?", const=True,
            help="File where log message should be written",
        )
        parser_x.add_argument(
            "--log-level", choices=[
                "NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
            ],
            help="Logging level", default="INFO",
        )
    # Simulator servers
    parser_server = subparsers.add_parser(
        "serve", help="Launch simulator(s) as fastapi application")
    parser_server.add_argument(
        "--simulator", type=str, nargs="+", action="extend",
        choices=installed_simulators,
        default=installed_simulators,
        help=(
            "Name of the simulator(s) to create application endpoints "
            "for. If not specified, all of the installed simulators "
            "will be included."
        )
    )
    parser_server.add_argument(
        "--port", type=int, default=5000,
        help="Port that application should be served on",
    )
    parser_server.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host address",
    )
    parser_server.add_argument(
        "--log-file", type=str,
        help="File where log message should be written",
    )
    parser_server.add_argument(
        "--log-level", choices=[
            "NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        ],
        help="Logging level", default="INFO",
    )
    parser_server.add_argument(
        "--allow-shutdown", action="store_true",
        help="Include an endpoint to allow shutdown from the client",
    )
    # n8n tool utilities
    # n8n_entry_points = list(server.EndPointRegistry._registry.keys())
    n8n_entry_points = ["start", "start-interactive"]
    parser_n8n = subparsers.add_parser(
        "n8n", help="Manage n8n tools for models")
    parser_n8n.add_argument(
        "simulator", type=str, choices=simulators,
        help="Name of the simulator to manage n8n tools for",
    )
    parser_n8n_act = parser_n8n.add_subparsers(
        dest="utility", required=True,
        help="Name of the utility to run.")
    n8n_parsers = {}
    n8n_parsers["create"] = parser_n8n_act.add_parser(
        "create", help="Create an n8n tool")
    n8n_parsers["update"] = parser_n8n_act.add_parser(
        "update", help="Update an existing tool")
    n8n_parsers["remove"] = parser_n8n_act.add_parser(
        "remove", help="Remove an n8n tool")
    n8n_parsers["query"] = parser_n8n_act.add_parser(
        "query", help="Query an n8n tool")
    for k, parser_x in n8n_parsers.items():
        parser_x.add_argument(
            "--name", type=str, nargs="+", action="extend",
            choices=n8n_entry_points,
            help="Name of the entry point(s)",
        )
        parser_x.add_argument(
            "--toolname", type=str,
            help="Name of the tool (single name must be provided)",
        )
        parser_x.add_argument(
            "--output-tool", "--output", type=str, nargs='?', const=True,
            help="Output the tool summary to a file",
        )
        parser_x.add_argument(
            "--verbose", action="store_true",
            help="Print all REST API responses",
        )
        if k != 'query':
            parser_x.add_argument(
                "--dry-run", action="store_true",
                help="Don't actually do anything, just show requests.",
            )
    for k in ["create", "update"]:
        parser_x = n8n_parsers[k]
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
    n8n_parsers["create"].add_argument(
        "--update", action="store_true",
        help="Update any existing tool",
    )
    # Parse
    args = parser.parse_args()
    if args.action == "config":
        if args.section in ["directories", "files"]:
            args.value = os.path.abspath(os.path.expanduser(args.value))
        cfg.set(args.section, args.name, args.value)
        dst = cfg.write(level=args.level)
        print(f"Set {args.name} in the \"{args.section}\" section of "
              f"\"{dst}\" to \"{args.value}\" ")
        return
    elif args.action == "install":
        if args.directory:
            cfg.set("directories", args.simulator, args.directory)
        engine_cls = get_simulator_class(args.simulator)
        engine_cls.install(always_yes=args.always_yes)
    elif args.action == "create":
        engine_cls = get_simulator_class(args.simulator)
        kws = {
            k: v for k, v in vars(args).items()
            if k not in ["action", "simulator", "dst", "overwrite"]
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
        print(f"Created input file \"{engine.model.fname}\"")
    elif args.action == "run":
        kws = {
            k: v for k, v in vars(args).items()
            if k not in ["action", "simulator", "log_file", "log_level"]
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
            print(f"Log being written to \"{args.log_file}\"")
        logging.basicConfig(filename=args.log_file,
                            level=getattr(logging, args.log_level))
        run(args.simulator, **kws)
    elif args.action == "serve":
        server.run_server(
            args.simulator,
            host=args.host, port=args.port,
            log_file=args.log_file,
            log_level=args.log_level,
            allow_shutdown=args.allow_shutdown,
        )
    elif args.action == "n8n":
        if not args.name:
            if args.utility in ["query", "remove"] and args.toolname:
                args.name = [""]  # Won't be used
            else:
                args.name = n8n_entry_points
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
            print(f"{args.utility} {name}")
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
                    print(json.dumps(response, indent=2))
            else:
                raise NotImplementedError(
                    f"n8n utility = \"{args.utility}\"")
    else:
        raise NotImplementedError(f"action = \"{args.action}\"")
