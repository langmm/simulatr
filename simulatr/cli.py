# python -m simulatr run apsimx --crop-name wheat --from-example
import os
import argparse
import logging
from typing import Any, Optional, List
from . import logger, registered_simulators, get_simulator_class
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
        state_variables: Optional[List[str]] = None,
        **kwargs: Any) -> None:
    r"""Run a simulation.

    Args:
        simulator: Name of the simulator to run.
        timestep: Time between actions (in days). 0 for continuous.
        state_variables: Set of state variables to request at each
            timestep.
        **kwargs: Additional keyword arguments are passed along to
            the environment class constructor.

    """
    if state_variables:
        kwargs["output_vars"] = state_variables
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
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="action", help='Action to perform')
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
    # For creating model input files
    parser_create = subparsers.add_parser(
        "create", help="Create a simulator input file"
    )
    parser_create_sim = parser_create.add_subparsers(
        dest="simulator",
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
        dest="simulator", help="Name of the simulator to run")
    run_parsers = {}
    for k in simulators:
        run_parsers[k] = _add_args_from_engine(
            k, parser_run_sim,
            help=f"Run an {k} simulation",
        )
    for setting_x in run_parsers.values():
        parser_x = setting_x.root_parser
        parser_x.add_argument(
            "--timestep", type=int, default=0,
            help=(
                "Time between actions (in days). If non-zero, the "
                "simulation pauses at each timestep to ask for user "
                "input. 0 for continuous"
            ),
        )
        parser_x.add_argument(
            "--state-variables", type=str, nargs="+", action="extend",
            help="State variables to request at each time step.",
        )
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
    # Generic arguments
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
        engine_cls.install()
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
        if args.log_file is True:
            log_file = args.simulator
            if getattr(args, "model_file", None):
                log_file += "_" + os.path.splitext(
                    os.path.basename(args.model_file))[0]
            elif ((args.simulator == "apsimx"
                   and getattr(args, "crop_name", None))):
                log_file += "_" + args.crop_name
            args.log_file = os.path.join(os.getcwd(), log_file + ".log")
        if args.log_file:
            print(f"Log being written to \"{args.log_file}\"")
        logging.basicConfig(filename=args.log_file,
                            level=getattr(logging, args.log_level))
        run(args.simulator, **kws)
