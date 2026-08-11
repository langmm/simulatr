import logging
from typing import Optional, List
logger = logging.getLogger(__name__)
from .base import _ModelEnvMeta  # noqa: E402
from .apsimx import ApsimXFile, ApsimXEngine, ApsimXEnv  # noqa: E402
from ._version import __version__, __version_tuple__  # noqa: F401, E402


def registered_simulators(only_installed: Optional[bool] = False) -> List[str]:
    r"""Get the names of all supported simulators.

    Args:
        only_installed: If True, only include installed simulators in the
            returned list.

    Returns:
        list: Names of supported simulators.

    """
    out = _ModelEnvMeta.registered_simulators()
    if only_installed:
        out = [k for k in out if get_simulator_class(k).is_installed()]
    return out


def get_simulator_class(simulator: str,
                        class_type: Optional[str] = "engine"):
    r"""Get a simulator class.

    Args:
        simulator: Name of the simulator to get the engine class for.
        class_type: Type of simulator class to return.

    Returns:
        type: Simulator class.

    """
    if class_type == "env":
        return _ModelEnvMeta.get_simulator_env(simulator)
    env_cls = get_simulator_class(simulator, "env")
    if class_type == "engine":
        return env_cls.MODEL_ENGINE_CLASS
    elif class_type == "file":
        return env_cls.MODEL_ENGINE_CLASS.INPUT_FILE_TYPE
    elif class_type == "prompt_generator":
        return env_cls.LLM_PROMPT_GENERATOR_CLASS
    raise ValueError(f"Unsupported file type \"{simulator}\"")


__all__ = ["ApsimXFile", "ApsimXEngine", "ApsimXEnv"]
