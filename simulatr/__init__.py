import logging
from typing import Optional
logger = logging.getLogger(__name__)
from .apsimx import ApsimXFile, ApsimXEngine, ApsimXEnv  # noqa: E402
from ._version import __version__, __version_tuple__  # noqa: F401, E402


def get_simulator_class(simulator: str,
                        file_type: Optional[str] = "engine"):
    r"""Get a simulator class.

    Args:
        simulator: Name of the simulator to get the engine class for.
        file_type: Type of simulator class to return.

    Returns:
        type: Simulator class.

    """
    if file_type == "env":
        if simulator == "apsimx":
            return ApsimXEnv
        raise ValueError(f"Unsupported simulator \"{simulator}\"")
    env_cls = get_simulator_class(simulator, "env")
    if file_type == "engine":
        return env_cls.MODEL_ENGINE_CLASS
    elif file_type == "file":
        return env_cls.MODEL_ENGINE_CLASS.INPUT_FILE_TYPE
    elif file_type == "prompt_generator":
        return env_cls.LLM_PROMPT_GENERATOR_CLASS
    raise ValueError(f"Unsupported file type \"{simulator}\"")


__all__ = ["ApsimXFile", "ApsimXEngine", "ApsimXEnv"]
