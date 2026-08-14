import os
import json
import uuid
import pprint
import logging
import asyncio
import contextlib
from collections import OrderedDict
import re
from typing import Dict, Tuple, List, Any, ClassVar, Callable, Self
from pydantic import (
    BaseModel, PrivateAttr, field_validator, Field, ConfigDict,
)
from pydantic_settings import CLI_SUPPRESS
from fastapi import HTTPException, FastAPI
import uvicorn
from . import registered_simulators, get_simulator_class


class InteractiveModelRegistry:
    r"""Class for tracking models that are running interactively."""

    _global_instance = None

    def __init__(self):
        r"""Initialize the registry with no running models."""
        self._models = {}
        self._lock = asyncio.Lock()
        self._in_use = []

    def __del__(self):
        r"""Stop and remove all models on destruction."""
        for k in list(self._models.keys()):
            asyncio.run(self._safe_remove(k))

    @classmethod
    def global_instance(cls) -> "InteractiveModelRegistry":
        r"""Global singleton instance of InteractiveModelRegistry."""
        if cls._global_instance is None:
            cls._global_instance = cls()
        return cls._global_instance

    @classmethod
    @contextlib.asynccontextmanager
    async def valid_model(cls, idstr: str, allow_stopped: bool = False):
        r"""Yield the model with the given id if it is valid for use.

        Args:
            idstr: Model ID string.
            allow_stopped: If True, yield the model even if it has
                stopped running.

        Raises:
            HTTPException: If no model exists with the given id, or the
                model is no longer running.

        Yields:
            Model: The model registered under ``idstr``.

        """
        self = cls.global_instance()
        model = None
        async with self._lock:
            if idstr not in self._models:
                raise HTTPException(
                    status_code=404,
                    detail=f"No model with id \"{idstr}\"",
                )
            self._in_use.append(idstr)
            model = self._models[idstr]
        async with model._model_lock:
            try:
                if not (allow_stopped or model.is_running):
                    raise HTTPException(
                        status_code=404,
                        detail=f"Model \"{idstr}\" is no longer running")
                yield model
            finally:
                async with self._lock:
                    self._in_use.remove(idstr)

    @classmethod
    async def add(cls, model: "InteractiveSimulator") -> None:
        r"""Register a model in the registry.

        Args:
            model: The model to register.

        """
        self = cls.global_instance()
        async with self._lock:
            assert model._idstr not in self._models
            self._models[model._idstr] = model

    async def _safe_remove(self, idstr: str,
                           dont_stop: bool = False) -> bool:
        r"""Remove a model from the registry if it is not in use.

        Args:
            idstr: Id of the model to remove.
            dont_stop: If True, keep the model running when removing it.

        Returns:
            bool: True if the model was removed (or not present), False
                if it is currently in use.

        """
        if idstr not in self._models:
            return True
        if idstr in self._in_use:
            return False
        model = self._models[idstr]
        async with model._model_lock:
            if model.is_running and not dont_stop:
                model.stop()
            if not (dont_stop and model.is_running):
                model.cleanup(remove_output=True)
                del self._models[idstr]
        return True

    @classmethod
    async def remove(cls, idstr: str, **kwargs) -> None:
        r"""Remove a model from the registry, waiting until it is free.

        Args:
            idstr: Id of the model to remove.
            \*\*kwargs: Additional options passed to ``_safe_remove``.

        """
        self = cls.global_instance()
        while True:
            async with self._lock:
                if await self._safe_remove(idstr, **kwargs):
                    return

    @classmethod
    async def clear(cls, simulator: str = None, ids: List[str] = None,
                    **kwargs) -> None:
        r"""Remove all (or the given) models from the registry.

        Args:
            simulator: Name of the simulator that models should be
                removed for. Ignored if ids provided. If neither
                simulator nor ids is provided, all of the models will
                be considered for removal.
            ids: Ids of the models to remove.
            \*\*kwargs: Additional options passed to ``_safe_remove``.

        """
        self = cls.global_instance()
        while True:
            async with self._lock:
                if ids is None:
                    if simulator is not None:
                        ids = [k for k, v in self._models.items()
                               if v._simulator_name == simulator]
                    else:
                        ids = list(self._models.keys())
                if not any(idstr in self._models for idstr in ids):
                    return
                ids = [
                    idstr for idstr in ids
                    if not await self._safe_remove(idstr, **kwargs)
                ]

    @classmethod
    async def size(cls, simulator: str) -> int:
        r"""Get the number of models in the registry.

        Args:
            simulator: Name of simulator to count models for.

        Returns:
            int: Number of registered models.

        """
        self = cls.global_instance()
        async with self._lock:
            if simulator is not None:
                return len([k for k, v in self._models.items()
                            if v._simulator_name == simulator])
            return len(self._models)


def _json_schema_extra_server(schema: dict):
    r"""Hide fields marked for the server from the JSON schema.

    Args:
        schema: JSON schema dict to filter in place.

    """
    props = {
        k: v for k, v in schema.get('properties', {}).items()
        if not v.get("hidden_for_server", False)
    }
    schema["properties"] = props
    return schema


class EndPointRegistry(type(BaseModel)):
    r"""Metaclass for registerying and managing simulator endpoint
    classes."""

    _registry: OrderedDict[str, type] = OrderedDict()
    _simulator_registry: OrderedDict[str, Dict[str, type]] = OrderedDict()

    def __new__(mcs, name: str, bases: Tuple[type, ...],
                namespace: Dict[str, Any], **kwargs: Any):
        r"""Register the new endpoint class in the endpoint registry.

        Args:
            name: Name of the new class.
            bases: Base classes of the new class.
            namespace: Class namespace.
            \*\*kwargs: Additional keyword arguments.

        Returns:
            type: The newly created class.

        """
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        if cls._endpoint_name is not None:
            if cls._simulator_name is None:
                dst = mcs._registry
            else:
                mcs._simulator_registry.setdefault(
                    cls._simulator_name, OrderedDict())
                dst = mcs._simulator_registry[cls._simulator_name]
            endpoint_name = cls._endpoint_name.split('/')[-1]
            assert endpoint_name not in dst
            dst[endpoint_name] = cls
        return cls

    @classmethod
    def get_simulator_endpoints(cls, simulator: str = None) -> Dict[str, type]:
        r"""Get the endpoints for a simulator.

        Args:
            simulator: Name of the simulator. If not provided, the base
                class registry will be returned.

        Returns:
            dict: Mapping of classes defining endpoints for the
                simulator.

        """
        if simulator is None:
            return cls._registry
        cls.add_simulator(simulator)
        return cls._simulator_registry[simulator]

    @classmethod
    def add_simulator_endpoints(cls, app: FastAPI, simulator: str) -> None:
        r"""Add simulator endpoints to an application.

        Args:
            app: FastAPI application.
            simulator: Name of the simulator.

        """
        classes = cls.get_simulator_endpoints(simulator)
        import pprint
        pprint.pprint(classes)
        for v in classes.values():
            v.add_endpoint(app)

    @classmethod
    def add_simulator(cls, simulator: str) -> None:
        r"""Add server classes to the registry for a simulator.

        Args:
            simulator: Name of the simulator.

        """
        if simulator in cls._simulator_registry:
            return
        engine = get_simulator_class(simulator)
        for v in cls._registry.values():
            v.create_type(engine)
        assert simulator in cls._simulator_registry
        assert len(cls._simulator_registry[simulator]) == len(cls._registry)


class EndPointBase(BaseModel, metaclass=EndPointRegistry):
    r"""Mixin for an endpoint."""

    _simulator_name: ClassVar[str] = None
    _endpoint_name: ClassVar[str] = None
    _endpoint_type: ClassVar[str] = "post"

    @classmethod
    def add_endpoint(cls, app: FastAPI) -> Callable:
        r"""Add this endpoint to a fastapi application.

        Args:
            app: FastAPI application.

        Returns:
            Callable: Decorated endpoint function.

        """
        return getattr(app, cls._endpoint_type)(cls._endpoint_name)(
            cls.endpoint)

    @classmethod
    def create_type(cls, engine: type,
                    additional_bases: tuple = None,
                    attr: dict = None) -> type:
        r"""Create a server class type for a simulator.

        Args:
            engine: Engine class that the engine point should be
                created for.
            additional_bases: Base classes for the new class.
            attr: Class attributes for the new class.

        Returns:
            type: The newly created class.

        """
        name = f"{engine.__name__}{cls.__name__}"
        if additional_bases:
            bases = (cls, *additional_bases)
        else:
            bases = (cls, )
        if attr is None:
            attr = {}
        attr.setdefault("_simulator_name", engine._MODEL_NAME)
        attr.setdefault("_endpoint_name",
                        f"/{engine._MODEL_NAME}{cls._endpoint_name}")
        out = type(name, bases, attr)
        return out


class EndPointModelMixin(EndPointBase):
    r"""Mixin for an endpoint based on a pydantic model."""

    @classmethod
    def create_type(cls, engine: type,
                    field_updates: dict = {},
                    disable_fields: List[str] = [],
                    **kwargs) -> type:
        r"""Create a server class type for a simulator.

        Args:
            engine: Engine class that the engine point should be
                created for.
            field_updates: Mapping of field names to attributes to
                update on the field.
            disable_fields: Field names to hide from the server schema.
            \*\*kwargs: Additional keyword arguments are passed to
                the base class's method.

        Returns:
            type: The newly created class.

        """
        name = f"{engine.__name__}{cls.__name__}"
        input_type = name
        out = super().create_type(engine, **kwargs)
        if "endpoint" not in kwargs.get("attr", {}):
            if "{idstr}" in cls._endpoint_name:
                async def endpoint(ref, idstr: str, input: input_type):
                    r"""Handle a request for a specific model."""
                    return await cls.endpoint(idstr, input)
            else:
                async def endpoint(ref, input: input_type):
                    r"""Handle a request."""
                    return await cls.endpoint(input)
            out.endpoint = classmethod(endpoint)
        # print(cls.endpoint.__annotations__)
        # print(out.endpoint.__annotations__)
        # import pdb; pdb.set_trace()
        for k, v in field_updates.items():
            for kk, vv in v.items():
                setattr(out.model_fields[k], kk, vv)
        disable_fields = disable_fields + [
            k for k, v in out.model_fields.items()
            if CLI_SUPPRESS in v.metadata
            and k != "test_field"
        ]
        for k in disable_fields:
            out.model_fields[k].json_schema_extra = {
                "hidden_for_server": True,
            }
        for k, v in out.model_fields.items():
            if ((v.default is not None or k in disable_fields
                 or "default" in field_updates.get(k, {})
                 or (v.json_schema_extra
                     and v.json_schema_extra.get(
                         "hidden_for_server", False)))):
                continue
            if v.examples:
                v.default = v.examples[0]
                continue
        out.model_rebuild(force=True)
        globals()[name] = out
        return out

    # @classmethod
    # def add_endpoint(cls, app: FastAPI) -> Callable:
    #     r"""Add this endpoint to a fastapi application.

    #     Args:
    #         app: FastAPI application.

    #     Returns:
    #         Callable: Decorated endpoint function.

    #     """

    #     if "{idstr}" in cls._endpoint_name:
    #         print("ADDED ENDPOINT W/ IDSTR", cls)
    #         async def specialized_endpoint(idstr: str, input: cls):
    #             return cls.endpoint(idstr, input)
    #     else:
    #         async def specialized_endpoint(input: cls):
    #             return cls.endpoint(input)

    #     return getattr(app, cls._endpoint_type)(cls._endpoint_name)(
    #         specialized_endpoint)


class ContinuousSimulator(EndPointModelMixin):
    r"""Mixin for running a simulator to completion on a server."""

    _endpoint_name: ClassVar[str] = "/start"

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        json_schema_extra=_json_schema_extra_server,
    )

    _idstr: PrivateAttr(default=None)
    _trace: PrivateAttr(default=None)
    _results: PrivateAttr(default=None)

    @classmethod
    def create_type(cls, engine: type,
                    additional_bases: tuple = None,
                    field_updates: dict = None,
                    disable_fields: List[str] = None,
                    **kwargs) -> type:
        r"""Create a server class type for a simulator.

        Args:
            engine: Engine class that the engine point should be
                created for.
            additional_bases: Base classes for the new class.
            field_updates: Mapping of field names to attributes to
                update on the field.
            disable_fields: Field names to hide from the server schema.
            \*\*kwargs: Additional keyword arguments are passed to
                the base class's method.

        Returns:
            type: The newly created class.

        """
        if additional_bases is None:
            additional_bases = (engine, )
        if field_updates is None:
            field_updates = {}
        if disable_fields is None:
            disable_fields = ["actions"]
            field_updates.setdefault("actions", {})
            field_updates["actions"].setdefault(
                "default", ["irrigate"],
            )
        field_updates.setdefault("output_vars", {})
        field_updates["output_vars"].setdefault(
            # TODO: Use another variable?
            "examples", [[engine.EXAMPLE_STATE[0]]],
        )
        for k, v in engine.default_server_fields().items():
            field_updates.setdefault(k, {})
            field_updates[k].setdefault("default", v)
        return super().create_type(
            engine, additional_bases=additional_bases,
            field_updates=field_updates,
            disable_fields=disable_fields,
            **kwargs
        )

    def model_post_init(self, context):
        r"""Assign a unique id to the model."""
        self._idstr = str(uuid.uuid4())
        return super().model_post_init(context)

    def run(self) -> Any:
        r"""Run the model to completion. Recording results.

        Returns:
            dict: The trace for the model.

        """
        return super().run(remove_output=True)

    @classmethod
    async def endpoint(cls, input: Self):
        r"""Endpoint.

        Args:
            input: Request parameters.

        Returns:
            dict: Response.

        """
        return input.run()


class InteractiveSimulator(ContinuousSimulator):
    r"""Mixin for running a simulator interactively on a server."""

    _endpoint_name: ClassVar[str] = "/start-interactive"

    _model_lock: PrivateAttr(default=None)
    _model_accessed: PrivateAttr(default=None)
    _shutdown_after_wait: PrivateAttr(default=None)

    wait_time: int = Field(
        300, description=(
            "that the model should be kept alive at each time "
            "step awaiting an interactive command (seconds)"
        ))

    @classmethod
    def create_type(cls, engine: type, **kwargs) -> type:
        r"""Create a server class type for a simulator.

        Args:
            engine: Engine class that the engine point should be
                created for.
            \*\*kwargs: Additional keyword arguments are passed to
                the base class's method.

        Returns:
            type: The newly created class.

        """
        # Prevent actions from being disabled by
        #   ContinuousSimulator.create_type
        kwargs.setdefault("disable_fields", [])
        return super().create_type(engine, **kwargs)

    def model_post_init(self, context):
        r"""Initialize the lock and event for the interactive model."""
        self._model_lock = asyncio.Lock()
        self._model_accessed = asyncio.Event()
        return super().model_post_init(context)

    @field_validator('wait_time')
    @classmethod
    def check_wait_time(cls, v):
        r"""Clamp the wait time to the allowed range."""
        if v < 0 or v > 300:
            return 300
        return v

    def start(self, **kwargs):
        r"""Start the model, scheduling shutdown if it is unused."""
        super().start(**kwargs)
        self._shutdown_after_wait = asyncio.create_task(
            self.shutdown_after_wait())

    async def shutdown_after_wait(self):
        r"""Stop the model if it is not accessed within the wait time."""
        is_running = True
        while is_running:
            if self._model_lock is None:
                return
            async with self._model_lock:
                is_running = self.is_running
            if not is_running:
                break
            try:
                async with asyncio.timeout(self.wait_time):
                    await self._model_accessed.wait()
                self._model_accessed.clear()
            except TimeoutError:
                if is_running:
                    async with self._model_lock:
                        if self.is_running:
                            self.stop()
                        self.cleanup(remove_output=True)
                        assert not self.is_running
                break

    @classmethod
    async def endpoint(cls, input: Self):
        r"""Endpoint.

        Args:
            input: Request parameters.

        Returns:
            dict: Response.

        """
        input.start()
        await InteractiveModelRegistry.add(input)
        return input._idstr


class InteractiveStopAll(EndPointBase):
    r"""Endpoint for stopping all registered interactive models."""

    _endpoint_name: ClassVar[str] = "/stop-interactive"
    _endpoint_type: ClassVar[str] = "post"

    @classmethod
    async def endpoint(cls):
        r"""Endpoint.

        Returns:
            dict: Response.

        """
        await InteractiveModelRegistry.clear(cls._simulator_name)


class InteractivePruneAll(EndPointBase):
    r"""Endpoint for pruning interactive models that are not running."""

    _endpoint_name: ClassVar[str] = "/prune-interactive"
    _endpoint_type: ClassVar[str] = "post"

    @classmethod
    async def endpoint(cls):
        r"""Endpoint.

        Returns:
            dict: Response.

        """
        await InteractiveModelRegistry.clear(cls._simulator_name,
                                             dont_stop=True)


class InteractiveStatus(EndPointBase):
    r"""Endpoint for checking the status of an interactive model."""

    _endpoint_name: ClassVar[str] = "/interactive-model/{idstr}/status"
    _endpoint_type: ClassVar[str] = "get"

    @classmethod
    async def endpoint(cls, idstr: str):
        r"""Endpoint.

        Args:
            idstr: Model id string.

        Returns:
            dict: Response.

        """
        async with InteractiveModelRegistry.valid_model(
                idstr, allow_stopped=True) as model:
            if not model.is_running:
                return {"status": "stopped"}
            return {"status": "running",
                    "time": model.current_time}


class InteractiveTrace(EndPointBase):
    r"""Endpoint for getting the trace of an interactive model."""

    _endpoint_name: ClassVar[str] = "/interactive-model/{idstr}/trace"
    _endpoint_type: ClassVar[str] = "get"

    @classmethod
    async def endpoint(cls, idstr: str):
        r"""Endpoint.

        Args:
            idstr: Model id string.

        Returns:
            dict: Response.

        """
        async with InteractiveModelRegistry.valid_model(
                idstr, allow_stopped=True) as model:
            return model.get_trace()


# class InteractiveResults(EndPointBase):
#     r"""Endpoint for getting the results of an interactive model."""

#     _endpoint_name: ClassVar[str] = "/interactive-model/{idstr}/results"
#     _endpoint_type: ClassVar[str] = "get"

#     @classmethod
#     async def endpoint(cls, idstr: str):
#         r"""Endpoint.

#         Args:
#             idstr: Model id string.

#         Returns:
#             dict: Response.

#         """
#         async with InteractiveModelRegistry.valid_model(
#                 idstr, allow_stopped=True) as model:
#             return model.get_results()


class InteractiveContinue(EndPointBase):
    r"""Endpoint for continuing an interactive model."""

    _endpoint_name: ClassVar[str] = "/interactive-model/{idstr}/continue"
    _endpoint_type: ClassVar[str] = "post"

    @classmethod
    async def endpoint(cls, idstr: str):
        r"""Endpoint.

        Args:
            idstr: Model id string.

        Returns:
            dict: Response.

        """
        async with InteractiveModelRegistry.valid_model(idstr) as model:
            model.resume(wait=True)
            return {"status": "success"}


class InteractiveComplete(EndPointBase):
    r"""Endpoint for completing an interactive model."""

    _endpoint_name: ClassVar[str] = "/interactive-model/{idstr}/complete"
    _endpoint_type: ClassVar[str] = "post"

    @classmethod
    async def endpoint(cls, idstr: str):
        r"""Endpoint.

        Args:
            idstr: Model id string.

        Returns:
            dict: Response.

        """
        async with InteractiveModelRegistry.valid_model(idstr) as model:
            model.fast_forward()
            return {"status": "success"}


class InteractiveRestart(EndPointBase):
    r"""Endpoint for restarting an interactive model."""

    _endpoint_name: ClassVar[str] = "/interactive-model/{idstr}/restart"
    _endpoint_type: ClassVar[str] = "post"

    @classmethod
    async def endpoint(cls, idstr: str):
        r"""Endpoint.

        Args:
            idstr: Model id string.

        Returns:
            dict: Response.

        """
        async with InteractiveModelRegistry.valid_model(idstr) as model:
            model.rewind()
            return {"status": "success"}


class InteractiveStop(EndPointBase):
    r"""Endpoint for stopping an interactive model."""

    _endpoint_name: ClassVar[str] = "/interactive-model/{idstr}/stop"
    _endpoint_type: ClassVar[str] = "post"

    @classmethod
    async def endpoint(cls, idstr: str):
        r"""Endpoint.

        Args:
            idstr: Model id string.

        Returns:
            dict: Response.

        """
        async with InteractiveModelRegistry.valid_model(idstr) as model:
            model.stop()
            return {"status": "success"}


class InteractiveScrub(EndPointBase):
    r"""Endpoint for moving an interactive model forward/backward in
    time."""

    _endpoint_name: ClassVar[str] = "/interactive-model/{idstr}/scrub"
    _endpoint_type: ClassVar[str] = "post"

    @classmethod
    async def endpoint(cls, idstr: str, time: int | str):
        r"""Endpoint.

        Args:
            idstr: Model id string.
            time: Time that the simulation should be moved to or by.

        Returns:
            dict: Response.

        """
        if ((isinstance(time, str)
             and re.fullmatch(r'[-+]?[1-9][0-9]*', time))):
            time = int(time)
        async with InteractiveModelRegistry.valid_model(idstr) as model:
            model.scrub(time)
            return {"status": "success"}


class InteractiveSet(EndPointModelMixin):
    r"""Input for setting state variable values on a running model."""

    _endpoint_name: ClassVar[str] = "/interactive-model/{idstr}/set"
    _endpoint_type: ClassVar[str] = "put"

    values: dict = Field(
        description=(
            "mapping between state variable names and values they "
            "should be set to (json object)"
        ))

    @field_validator('values', mode="before")
    @classmethod
    def check_json(cls, v):
        r"""Parse json string input."""
        if isinstance(v, str):
            return json.loads(v)
        return v

    @classmethod
    async def endpoint(cls, idstr: str, input: Self):
        r"""Endpoint.

        Args:
            idstr: Model id string.
            input: Request parameters.

        Returns:
            dict: Response.

        """
        async with InteractiveModelRegistry.valid_model(idstr) as model:
            model.setvars(input.values)
            return {"status": "success"}

    @classmethod
    def create_type(cls, engine: type, **kwargs) -> type:
        r"""Create a server class type for a simulator.

        Args:
            engine: Engine class that the engine point should be
                created for.
            \*\*kwargs: Additional keyword arguments are passed to
                the base class's method.

        Returns:
            type: The newly created class.

        """
        kwargs["field_updates"] = dict(
            kwargs.get("field_updates", {}),
            values={
                "examples": [
                    {engine.EXAMPLE_STATE[0]: engine.EXAMPLE_STATE[1]},
                ]
            }
        )
        return super().create_type(engine, **kwargs)


class InteractiveGet(EndPointModelMixin):
    r"""Input for getting state variable values from a running model."""

    _endpoint_name: ClassVar[str] = "/interactive-model/{idstr}/get"
    _endpoint_type: ClassVar[str] = "get"

    state_variables: List[str] = Field(
        description="to get values for (comma separated list)"
    )

    @field_validator('state_variables', mode="before")
    @classmethod
    def check_list(cls, v):
        r"""Parse comma separated list."""
        if isinstance(v, str):
            return [vv.strip() for vv in v.split(",")]
        return v

    @classmethod
    def create_type(cls, engine: type, **kwargs) -> type:
        r"""Create a server class type for a simulator.

        Args:
            engine: Engine class that the engine point should be
                created for.
            \*\*kwargs: Additional keyword arguments are passed to
                the base class's method.

        Returns:
            type: The newly created class.

        """
        kwargs["field_updates"] = dict(
            kwargs.get("field_updates", {}),
            state_variables={
                "examples": [[engine.EXAMPLE_STATE[0]]],
            }
        )
        return super().create_type(engine, **kwargs)

    @classmethod
    async def endpoint(cls, idstr: str, input: "InteractiveGet"):
        r"""Endpoint.

        Args:
            idstr: Model id string.
            input: Request parameters.

        Returns:
            dict: Response.

        """
        async with InteractiveModelRegistry.valid_model(idstr) as model:
            return model.getvars(input.state_variables)


class InteractiveAct(EndPointModelMixin):
    r"""Input for performing a management action on a running model."""

    _endpoint_name: ClassVar[str] = "/interactive-model/{idstr}/act"
    _endpoint_type: ClassVar[str] = "post"

    action: str = Field(
        description="to perform",
    )
    parameters: dict = Field(
        {},
        description="describing the management action (json object)"
    )

    @field_validator('parameters', mode="before")
    @classmethod
    def check_json(cls, v):
        r"""Parse json string input."""
        if isinstance(v, str):
            if not v:
                return {}
            return json.loads(v)
        return v

    @classmethod
    def create_type(cls, engine: type, **kwargs) -> type:
        r"""Create a server class type for a simulator.

        Args:
            engine: Engine class that the engine point should be
                created for.
            \*\*kwargs: Additional keyword arguments are passed to
                the base class's method.

        Returns:
            type: The newly created class.

        """
        engine_actions = list(engine.AVAILABLE_ACTION_MAP.keys())
        kwargs["field_updates"] = dict(
            kwargs.get("field_updates", {}),
            action={
                "examples": [engine.EXAMPLE_ACTION[0]],
                "json_schema_extra": {"enum": engine_actions},
            },
            parameters={
                "examples": [engine.EXAMPLE_ACTION[1]],
            },
        )
        return super().create_type(engine, **kwargs)

    @classmethod
    async def endpoint(cls, idstr: str, input: "InteractiveAct"):
        r"""Endpoint.

        Args:
            idstr: Model id string.
            input: Request parameters.

        Returns:
            dict: Response.

        """
        async with InteractiveModelRegistry.valid_model(idstr) as model:
            model.act(input.action, **input.parameters)
            return {"status": "success"}


def add_simulator_endpoints(app: FastAPI, simulators: List[str] = None,
                            allow_shutdown: bool = False,
                            server: uvicorn.Server = None):
    r"""Add simulator endpoints to a fastapi application.

    Args:
        app: FastAPI application to add endpoints to.
        simulators: Names of simulators to add endpoints for. If None,
            all installed simulators are used.
        allow_shutdown: If True, add an endpoint to shut the server
            down.
        server: Uvicorn server to shut down. Used only if
            allow_shutdown is True.

    """
    import signal
    if simulators is None:
        simulators = registered_simulators(only_installed=True)

    if allow_shutdown:
        if server:
            @app.post("/shutdown")
            def shutdown():
                r"""Shut the server down."""
                def stop_server():
                    r"""Set the flag that stops the server."""
                    server.should_exit = True
                asyncio.get_event_loop().call_later(0.1, stop_server)
                return {"status": "Shutdown initiated"}
        else:
            @app.post("/shutdown")
            def shutdown():
                r"""Shut the server down."""
                os.kill(os.getpid(), signal.SIGINT)
                return {"message": "Server shutting down..."}

    @app.get("/")
    async def status():
        r"""Status of all simulators currently being served."""
        counts = {
            k: InteractiveModelRegistry.size(k)
            for k in simulators
        }
        return (
            f"This is a simulatr server providing access to the "
            f"following servers (with the number of interactive "
            f"models of each simulator type that are currently "
            f"running:\n{pprint.pprint(counts)}"
        )

    for x in simulators:
        EndPointRegistry.add_simulator_endpoints(app, x)


def run_server(simulators: List[str] = None,
               host: str = "0.0.0.0", port: int = 5000,
               log_file: str = None,
               log_level: str = "info",
               allow_shutdown: bool = False):
    r"""Run a server for the given simulators.

    Args:
        simulators: Names of simulators to serve. If None, all
            installed simulators are used.
        host: Host to bind the server to.
        port: Port to bind the server to.
        log_file: Path to file to write logs to.
        log_level: Logging level to use.
        allow_shutdown: If True, allow the server to be shut down via
            the API.

    """
    logging.basicConfig(filename=log_file,
                        level=getattr(logging, log_level.upper()))
    app = FastAPI()
    server = None
    # TODO: This version does not actually exit
    # config = uvicorn.Config(app, host=host, port=port,
    #                         log_level=log_level.lower())
    # server = uvicorn.Server(config)
    add_simulator_endpoints(app, simulators, server=server,
                            allow_shutdown=allow_shutdown)
    uvicorn.run(app, host=host, port=port)
    # asyncio.run(server.serve())
