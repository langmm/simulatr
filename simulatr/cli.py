# python -m simulatr.cli run apsimx Examples/Wheat.apsimx
import os
import sys
import argparse
import datetime
import logging
from typing import Any, Optional, List
from . import logger, get_engine
from .utils import cfg
from .apsimx import ApsimXFile


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
    engine_cls = get_engine(simulator)
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
    # TODO: Split by simulator at the top level?
    # Installation
    parser_install = subparsers.add_parser(
        "install", help="Install a simulator"
    )
    parser_install.add_argument(
        "simulator", type=str,  # choices=
        help="Name of the type of simulation to run",
    )
    parser_install.add_argument(
        "--directory", type=str,
        help="Directory where the simulator should be installed.",
    )
    # For running
    parser_run = subparsers.add_parser(
        "run", help="Run a simulation"
    )
    sim_subparsers = parser_run.add_subparsers(
        dest="simulator", help="Name of the type of simulation to run")
    parser_apsimx = sim_subparsers.add_parser(
        "apsimx", help="Run an ApsimX simulation")
    parser_apsimx.add_argument(
        "--crop-name", type=str,
        help="Name of crop to simulate",
    )
    run_parsers = [parser_apsimx]
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
    # For creating interactive .apsimx
    parser_create_apsimx = subparsers.add_parser(
        "create-apsimx", help="Create a .apsimx model input file"
        # interactive version of a .apsimx model"
    )
    parser_create_apsimx.add_argument(
        "crop_name", type=str,  # choices=
        help="Path to a .apsimx model input file",
    )
    parser_create_apsimx.add_argument(
        "--from-example", type=str, nargs="?", const=True, default=False,
        help=(
            "Create a new .apsimx model by copying an example. "
            "The path to the example can be passed."
        ),
    )
    parser_create_apsimx.add_argument(
        "--interactive", action="store_true",
        help="Make the new file interactive",
    )
    parser_create_apsimx.add_argument(
        "--actions", type=str, nargs="+", action="extend",
        help="Interactive actions that should be supported",
    )
    parser_create_apsimx.add_argument(
        "--dst", type=str,
        help="Path to where the new .apsimx file should be saved",
    )
    parser_create_apsimx.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite any existing file",
    )
    # For setting the apsimx directory
    parser_set = subparsers.add_parser(
        "set-apsimx-dir", help="Set the default APSIMX directory"
    )
    parser_set.add_argument(
        "apsimx_dir", type=str,
        help=(
            "Path to the root directory containing an APSIMX "
            "installation (i.e. the directory that contains "
            "\"bin/Debug/net8.0/ApsimZMQServer.dll\""
        ),
    )
    parser_set.add_argument(
        "--user", action="store_true",
        help=(
            "Set the apsimx directory in the .apsimx configuration file "
            "int the user's home directory"
        )
    )
    # Generic arguments
    args = parser.parse_args()
    if args.action == "install":
        if args.directory:
            cfg.set("directories", args.simulator, args.directory)
        engine_cls = get_engine(args.simulator)
        engine_cls.install()
    elif args.action == "set-apsimx-dir":
        args.apsimx_dir = os.path.abspath(
            os.path.expanduser(args.apsimx_dir))
        cfg.set("directories", "apsimx", args.apsimx_dir)
        if args.user:
            dst = cfg.files['user']
        else:
            dst = cfg.files['local']
        cfg.write(dst)
        print(f"Set apsimx directory to \"{args.apsimx_dir}\" "
              f"in \"{dst}\"")
        return
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
    elif args.action == "create-apsimx":
        # TODO: Generic
        out = ApsimXFile.from_crop_name(
            args.crop_name,
            dst=args.dst,
            from_example=args.from_example,
            interactive=args.interactive,
            actions=args.actions,
        )
        out.write(overwrite=args.overwrite)
        out.generated = False  # Prevent cleanup
        print(f"Created input file \"{out.fname}\"")


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
