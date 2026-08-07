# python -m simulatr run apsimx Examples/Wheat.apsimx
import os
import argparse
import datetime
import logging
from typing import Any, Optional, List
from . import logger, get_simulator_class
from .utils import cfg


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
            the engine class constructor.

    """
    # TODO: Ask for user input on the action that should be taken
    # using the LLM prompt generation
    engine_cls = get_simulator_class(simulator)
    engine = engine_cls(**kwargs)
    engine.start()
    data = []
    try:
        i = 0
        while engine.is_running and not engine.is_complete:
            logger.info(f"Time: {engine.current_time}")
            if state_variables:
                ivars = engine.getvars(state_variables)
                data.append(ivars)
            # TODO: PROMPT
            if timestep <= 0:
                engine.fast_forward()
            else:
                engine.fast_forward(datetime.timedelta(days=timestep))
            i += 1
    finally:
        engine.stop()
    print(f"Output written to {engine.output_file}")


def main() -> None:
    r"""Run the command line interface."""
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
    parser_create_apsimx = parser_create_sim.add_parser(
        "apsimx", help="Create an ApsimX input file")
    parser_create_apsimx.add_argument(
        "crop_name", type=str.lower,
        choices=[
            x.lower() for x in
            get_simulator_class("apsimx", "file").available_crops()
        ],
        help="Crop name to create an input file for",
    )
    parser_create_apsimx.add_argument(
        "--from-example", type=str, nargs="?", const=True, default=False,
        help=(
            "Create a new .apsimx model by copying an example. "
            "The path to the example can be passed."
        ),
    )
    for parser_x in [parser_create_apsimx]:
        parser_x.add_argument(
            "--interactive", action="store_true",
            help="Make the new file interactive",
        )
        parser_x.add_argument(
            "--actions", type=str, nargs="+", action="extend",
            help="Interactive actions that should be supported",
        )
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
    parser_run_apsimx = parser_run_sim.add_parser(
        "apsimx", help="Run an ApsimX simulation")
    parser_run_apsimx.add_argument(
        "--crop-name", type=str,
        choices=[
            x.lower() for x in
            get_simulator_class("apsimx", "file").available_crops()
        ],
        help="Name of crop to simulate",
    )
    run_parsers = [parser_run_apsimx]
    for parser_x in run_parsers:
        parser_x.add_argument(
            "--model-file", type=str,
            help="Path to a model input file",
        )
        parser_x.add_argument(
            "--timestep", type=int, default=0,
            help="Time step between actions (in days). 0 for continuous",
        )
        parser_x.add_argument(
            "--actions", type=str, nargs="+", action="extend",
            help="Actions to allow",
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
        kws = {}
        if args.simulator == "apsimx":
            if args.crop_name:
                kws["crop_name"] = args.crop_name
            kws["from_example"] = args.from_example
        if args.dst and os.path.isfile(args.dst):
            if not args.overwrite:
                raise RuntimeError(f"Model file already exists: "
                                   f"\"{args.dst}\"")
            os.remove(args.dst)
        engine = engine_cls(
            model_file=args.dst,
            # interactive=args.interactive,  # TODO
            actions=args.actions,
            **kws
        )
        engine.model.generated = False  # Prevent cleanup
        print(f"Created input file \"{engine.model.fname}\"")
    elif args.action == "run":
        kws = {"timestep": args.timestep}
        if args.log_file is True:
            log_file = args.simulator
            if args.model_file:
                log_file += "_" + os.path.splitext(
                    os.path.basename(args.model_file))[0]
            elif args.simulator == "apsimx" and args.crop_name:
                log_file += "_" + args.crop_name
            args.log_file = os.path.join(os.getcwd(), log_file + ".log")
        if args.log_file:
            print(f"Log being written to \"{args.log_file}\"")
        logging.basicConfig(filename=args.log_file,
                            level=getattr(logging, args.log_level))
        if args.model_file:
            kws["model_file"] = args.model_file
        if args.simulator == "apsimx":
            if args.crop_name:
                kws["crop_name"] = args.crop_name
        if args.actions:
            kws["actions"] = args.actions
        if args.state_variables:
            kws["state_variables"] = args.state_variables
        run(args.simulator, **kws)
