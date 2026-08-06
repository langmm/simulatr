import logging
logger = logging.getLogger(__name__)
from .apsimx import ApsimXFile, ApsimXEngine, ApsimXEnv  # noqa: E402
from ._version import __version__, __version_tuple__  # noqa: F401, E402


def get_engine(name: str) -> type:
    r"""Get the class for a named simulator engine.

    Args:
        name: Name of the simulator to get the engine class for.

    Returns:
        type: Engine class.

    """
    if name == "apsimx":
        return ApsimXEngine
    raise ValueError(f"Unsupported simulator \"{name}\"")


__all__ = ["ApsimXFile", "ApsimXEngine", "ApsimXEnv"]
