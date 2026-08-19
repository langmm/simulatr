import sys
import os
import time
import glob
import pstats
import io
import json
import logging
from typing import Optional, ClassVar, Any, List
from . import registered_simulators, get_simulator_class
from .utils import create_registry_metaclass
from pydantic import BaseModel, Field


TargetRegistry = create_registry_metaclass(base_type=BaseModel)


class TargetBaseClass(BaseModel, metaclass=TargetRegistry):

    _NAME: ClassVar[str] = None
    _DESCRIPTION: ClassVar[str] = None

    nstats: Optional[int] = Field(
        30,
        description="Number of stats to print for cProfile reports")
    niter: Optional[int] = Field(
        100,
        description="Number of iterations to perform when profiling "
                    "via time.perf_counter")
    use_cprofile: Optional[bool] = Field(
        False,
        description="Use cProfile to profile the code.")

    @classmethod
    def _fmt_time(cls, seconds: float) -> str:
        r"""Format a time duration in a human-readable way."""
        if seconds < 0.001:
            return f"{seconds * 1_000_000:.1f} µs"
        elif seconds < 1.0:
            return f"{seconds * 1000:.2f} ms"
        else:
            return f"{seconds:.3f} s"

    @classmethod
    def _print_header(cls, title: str) -> None:
        r"""Print a section header."""
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")

    @classmethod
    def _print_result(cls, label: str, elapsed: float,
                      iterations: Optional[int] = 1,
                      extra: Optional[str] = None) -> None:
        r"""Print a profiling result."""
        per_iter = elapsed / iterations
        msg = f"  {label}: {cls._fmt_time(elapsed)}"
        if iterations > 1:
            msg += (
                f" ({iterations} iterations, "
                f"{cls._fmt_time(per_iter)}/iter)"
            )
        if extra:
            msg += f" -- {extra}"
        print(msg)

    def run(self):
        self._print_header(self._DESCRIPTION)
        self._run()

    def _run(self):
        raise NotImplementedError

    def time_direct(self, func,
                    args: Optional[tuple] = None,
                    kwargs: Optional[dict] = None,
                    iterations: Optional[int] = None,
                    label: Optional[str] = "",
                    extra: Optional[str] = None) -> float:
        r"""Run a function using perf_counter."""
        args = (args or tuple())
        kwargs = (kwargs or {})
        iterations = (iterations or self.niter)
        t0 = time.perf_counter()
        for _ in range(iterations):
            func(*args, **kwargs)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        self._print_result(label, elapsed,
                           iterations=iterations,
                           extra=extra)
        return elapsed

    def time_cprofile(self, func,
                      args: Optional[tuple] = None,
                      kwargs: Optional[dict] = None,
                      nstats: Optional[int] = None,
                      **kws: Any):
        r"""Run a function under cProfile and print the top results."""
        import cProfile
        args = (args or tuple())
        kwargs = (kwargs or {})
        nstats = (nstats or self.nstats)
        pr = cProfile.Profile()
        pr.enable()
        result = self.time_direct(func, args, kwargs, **kws)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(self.nstats)
        print(s.getvalue())
        return result

    def time(self, func,
             args: Optional[tuple] = None,
             kwargs: Optional[dict] = None,
             **kws: Any):
        if self.use_cprofile:
            return self.time_cprofile(func, args, kwargs, **kws)
        return self.time_direct(func, args, kwargs, **kws)

    def time_subprocess(self, code: str | list,
                        label: Optional[str] = ""):
        import subprocess
        if not isinstance(code, list):
            code = [code]
        if self.use_cprofile:
            prefix = [
                "import cProfile, io, pstats",
                "pr = cProfile.Profile()",
                "pr.enable()",
            ]
            suffix = [
                "pr.disable()",
                "s = io.StringIO()",
                "ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')",
                f"ps.print_stats({self.nstats})",
                "print(s.getvalue())",
            ]
        else:
            prefix = [
                "import time",
                "t0 = time.perf_counter()",
            ]
            suffix = [
                "t1 = time.perf_counter()",
                "print(t1 - t0)",
            ]
        fullcode = "; ".join((prefix + code + suffix))
        result = subprocess.run(
            [sys.executable, "-c", fullcode],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            if self.use_cprofile:
                print(result.stdout)
            else:
                elapsed = float(result.stdout.strip())
                self._print_result(label, elapsed)
        else:
            err_msg = result.stderr.strip().split("\n")[0]
            print(f"  {label}: PROFILE FAILED - {err_msg}")


class ProfileImport(TargetBaseClass):
    r"""Profile package import."""

    _NAME: ClassVar[str] = "import"
    _DESCRIPTION: ClassVar[str] = "Import Time Profile"

    def _run(self):
        modules = [
            "simulatr",
            "simulatr.utils",
            "simulatr.config",
            "simulatr.data",
            "simulatr.crop",
            "simulatr.apsimx",
            "simulatr.n8n",
            "simulatr.server",
            "simulatr.cli",
        ]

        # Measure cold import by launching a subprocess
        print("  Measuring cold import times via subprocess...\n")
        for mod in modules:
            self.time_subprocess(f"import {mod}", label=mod)


class ProfileApsimXFile(TargetBaseClass):
    r"""Profile ApsimX tree traversal and node creation operations."""

    _NAME: ClassVar[str] = "apsimx-file-tree"
    _DESCRIPTION: ClassVar[str] = "ApsimX Tree Traversal Profile"

    def _run(self):
        from simulatr.apsimx import ApsimXFile, ApsimXFileNode

        apsimx_dir = None
        try:
            from simulatr.utils import cfg
            apsimx_dir = cfg['directories'].get('apsimx', None)
        except Exception:
            pass

        if apsimx_dir is None or not os.path.isdir(apsimx_dir):
            print("  Skipped: APSIMX directory not configured or not found")
            print("  Set [directories] apsimx in config to enable")
            return

        # Build a representative tree for profiling
        resources_dir = os.path.join(apsimx_dir, "Models", "Resources")
        json_files = glob.glob(os.path.join(resources_dir, "*.json"))

        # Profile ApsimXFileNode creation
        print("\n  ApsimXFileNode construction:")
        for fname in json_files[:3]:
            fpath = os.path.join(resources_dir, fname)
            with open(fpath, "r") as f:
                contents = json.load(f)

            # Count nodes
            def count_nodes(node):
                n = 1
                for child in node.get("Children", []):
                    n += count_nodes(child)
                return n
            total_nodes = count_nodes(contents)

            iterations = 20
            t0 = time.perf_counter()
            for _ in range(iterations):
                ApsimXFileNode(contents.copy())
            t1 = time.perf_counter()
            self._print_result(
                f"Node construction ({fname})",
                t1 - t0, iterations,
                f"{total_nodes} nodes in tree"
            )

        # Profile tree traversal via findall
        print("\n  Tree traversal (findall):")
        for fname in json_files[:3]:
            fpath = os.path.join(resources_dir, fname)
            try:
                afile = ApsimXFile(fpath)
            except Exception:
                continue

            iterations = 10
            # Search for a common node type
            t0 = time.perf_counter()
            for _ in range(iterations):
                results = list(afile.findall({
                    "$type": "ApsimX.Core.Visualisation.Presenter"
                }))
            t1 = time.perf_counter()
            self._print_result(
                f"findall (presenter) in {fname}",
                t1 - t0, iterations,
                f"found {len(results)} matches"
            )

            # Profile node_matches overhead
            t0 = time.perf_counter()
            for _ in range(iterations):
                afile.findall({})
            t1 = time.perf_counter()
            self._print_result(
                f"findall (empty query = all nodes) in {fname}",
                t1 - t0, iterations
            )

        # Profile children property (creating wrappers)
        print("\n  ApsimXFileNode.children property:")
        fpath = os.path.join(resources_dir, json_files[0])
        afile = ApsimXFile(fpath)
        node = ApsimXFileNode(afile.contents)

        iterations = 100
        t0 = time.perf_counter()
        for _ in range(iterations):
            list(node.children)
        t1 = time.perf_counter()
        self._print_result("children iteration", t1 - t0, iterations)

        # Profile parameter access
        print("\n  Parameter operations (linear scan):")

        # Find a node with parameters
        def find_param_node(node):
            if "Parameters" in node and node["Parameters"]:
                return node
            for child in node.get("Children", []):
                result = find_param_node(child)
                if result:
                    return result
            return None

        param_node = find_param_node(afile.contents)
        if param_node:
            wrapper = ApsimXFileNode(param_node)
            n_params = len(param_node.get("Parameters", []))
            iterations = 100
            t0 = time.perf_counter()
            for _ in range(iterations):
                wrapper.has_parameter("Name")
            t1 = time.perf_counter()
            self._print_result(
                "has_parameter (linear scan)", t1 - t0, iterations,
                f"{n_params} parameters"
            )


class ProfileSimulator(TargetBaseClass):
    r"""Profile simulated simulation loop operations."""

    _NAME: ClassVar[str] = "simulator"
    _DESCRIPTION: ClassVar[str] = (
        "Simulation Loop Operations Profile (Simulated)"
    )

    simulator: str = Field(
        description="Simulator to profile",
        json_schema_extra={"enum": registered_simulators()},
    )
    methods: List[str] = Field(
        ["get", "set", "act", "scrub", "run"],
        description="Methods to profile.",
        json_schema_extra={
            "enum": ["get", "set", "act", "scrub", "run"],
        },
    )

    def model_post_init(self, __context: Any) -> None:
        self._simulator_cls = get_simulator_class(
            self.simulator)
        self._reusable_model = self._simulator_cls(
            model_log_level=logging.DEBUG,
            **self._simulator_cls.EXAMPLE_KWARGS)
        self._reusable_model.start()
        self._t_half = (self._reusable_model.end_time
                        - self._reusable_model.start_time) / 2
        return super().model_post_init(__context)

    def _run_run(self):
        model = self._simulator_cls(
            model_log_level=logging.DEBUG,
            **self._simulator_cls.EXAMPLE_KWARGS
        )
        model.run()
        model.stop()

    def _run_get(self):
        self._reusable_model.get(self._simulator_cls.EXAMPLE_STATE[0])

    def _run_set(self):
        self._reusable_model.set(self._simulator_cls.EXAMPLE_STATE[0],
                                 self._simulator_cls.EXAMPLE_STATE[1])

    def _run_act(self):
        self._reusable_model.act(self._simulator_cls.EXAMPLE_ACTION[0],
                                 **self._simulator_cls.EXAMPLE_ACTION[1])

    def _run_scrub(self):
        self._reusable_model.scrub(self._t_half)
        self._reusable_model.scrub(-self._t_half)

    def _run(self):
        for method in self.methods:
            kws = (
                {"iterations": 10} if method in ["scrub", "run"]
                else {}
            )
            self._print_header(
                f"Profiling {self.simulator}.{method}")
            self.time(getattr(self, f"_run_{method}"),
                      label=method, **kws)
