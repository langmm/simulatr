import os
import numpy as np
import datetime
from abc import abstractmethod
from typing import Optional, List, Any, ClassVar
from pydantic import ConfigDict, Field, field_validator
from pydantic.json_schema import SkipJsonSchema
from pydantic_settings import CliSuppress
from .base import (
    SimulatorFieldInfo,
    BaseModelFile, BaseModelEngine,
    BaseModelLLMPromptGenerator, BaseModelEnv,
)
from .data import BaseWeatherFile, BaseSoilFile
from . import logger
from .utils import NoDefault


class CropModelFile(BaseModelFile):
    r"""Base class for managing crop model input files."""

    NAME: ClassVar[str] = "crop"

    @classmethod
    @abstractmethod
    def available_crops(cls) -> List[str]:
        r"""Get the crops that can be simulated via this model.

        Returns:
            list: Available crop names.

        """
        raise NotImplementedError  # pragma: no cover

    @classmethod
    @abstractmethod
    def available_cultivars(cls, crop_name: str) -> List[str]:
        r"""Get the cultivars for a given crop that can be simulated
        via this model.

        Args:
            crop_name: Crop name.

        Returns:
            list: Available crop cultivar names.

        """
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def validate_crop_name(cls, crop_name: str) -> str:
        r"""Ensure the crop name is one of those that can be simulated,
        normalizing it if necessary.

        Args:
            crop_name: Crop name.

        Returns:
            str: Normalized crop name.

        """
        available_crops = cls.available_crops()
        for alias in [crop_name, crop_name.lower(), crop_name.title()]:
            if alias in available_crops:
                return alias
        available_crops_lower = [x.lower() for x in available_crops]
        if crop_name.lower() in available_crops_lower:
            return available_crops[available_crops_lower.index(
                crop_name.lower())]
        raise NotImplementedError(
            f"Invalid crop name \"{crop_name}\". "
            f"Valid crop names are:\n\t"
            + "\n\t".join(available_crops))

    @classmethod
    @abstractmethod
    def from_crop_name(cls, crop_name: str,
                       crop_variety: Optional[str | None] = None,
                       dst: Optional[str | None] = None,
                       directory: Optional[str | None] = None,
                       interactive: Optional[bool] = False,
                       actions: Optional[List[str] | None] = None,
                       **kwargs: Any) -> "CropModelFile":
        r"""Create an input model file for a given crop name.

        Args:
            crop_name: Crop name.
            crop_variety: Crop variety.
            dst: Path to the location where the generated file should
                be saved.
            directory: Directory where the generated file should be
                saved (only used if dst is not provided).
            interactive: If True, make the file interactive.
            actions: Interactive actions that should be added.
            \*\*kwargs: Additional keyword arguments are treated as
                parameter key/value pairs.

        Returns:
            CropModelFile: Constructed model input file.

        """
        raise NotImplementedError  # pragma: no cover

    @BaseModelFile.parameter_property
    def crop_name(self) -> str:
        r"""str: Crop name."""
        return os.path.splitext(
            os.path.basename(self.fname_orig))[0].lower()

    @BaseModelFile.parameter_property
    def crop_variety(self) -> Optional[str]:
        r"""str: Crop cultivar name."""
        return None  # raise KeyError("crop_variety")

    @BaseModelFile.parameter_property
    def location(self) -> str:
        r"""str: Description of the field location."""
        try:
            lat = self.get("latitude")
            lon = self.get("longitude")
            return f"{lat}°N, {lon}°E"
        except KeyError:
            return "the field"

    @BaseModelFile.parameter_property
    def field_area(self) -> float:
        r"""float: Field area"""
        return 1.0


class CropModelEngine(BaseModelEngine):
    r"""Class for managining communication with a crop simulation model."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    WEATHER_FILE_TYPE: ClassVar[Any] = None
    DATE_PARAM: ClassVar[list] = BaseModelEngine.DATE_PARAM + [
        ("sow_date", "harvest_date", "season_length"),
    ]
    DEFAULT_PARAM: ClassVar[dict] = {
        "duration": datetime.timedelta(365),
    }
    FORM_FIELD_ORDER: ClassVar[List[str]] = (
        [
            "crop_name", "crop_variety",
        ] + BaseModelEngine.FORM_FIELD_ORDER
    )
    EXAMPLE_KWARGS: ClassVar[dict] = {"crop_name": "Wheat"}

    crop_name: Optional[str | SkipJsonSchema[None]] = SimulatorFieldInfo(
        default=None, examples=["Wheat"],
        description="Name of the crop that will be simulated")
    crop_variety: Optional[
        str | SkipJsonSchema[None]] = SimulatorFieldInfo(
            default=None, examples=["Hartog"],
            description="Name of the crop variety/cultivar that will be "
                        "simulated")
    sow_date: Optional[
        datetime.date | SkipJsonSchema[None]] = SimulatorFieldInfo(
            default=None,
            examples=[datetime.datetime.fromisoformat("1991-01-01")],
            description="Date that the crop should be sown (ISO 8601 "
                        "format).")
    harvest_date: Optional[
        datetime.date | SkipJsonSchema[None]] = SimulatorFieldInfo(
            default=None,
            examples=[datetime.datetime.fromisoformat("1991-11-05")],
            description="Date that the crop should be harvested "
                        "(ISO 8601 format).")
    season_length: Optional[int | float | datetime.timedelta
                            | SkipJsonSchema[None]] = SimulatorFieldInfo(
        default=None,
        examples=[datetime.timedelta(365)],
        description="Time between sowing and harvest. Only used if only "
                    "one of sow_date or harvest_date are used. If a "
                    "number is provided, it is assumed to be in units "
                    "of days.")
    year: Optional[int | SkipJsonSchema[None]] = SimulatorFieldInfo(
        default=None,
        examples=[1991],
        description="Year to use to get weather data.")
    latitude: Optional[float | SkipJsonSchema[None]] = SimulatorFieldInfo(
        default=None,
        examples=[40.1164],
        description="Field latitude to use to get weather data (degrees).")
    longitude: Optional[float | SkipJsonSchema[None]] = SimulatorFieldInfo(
        default=None,
        examples=[-88.2434],
        description="Field longitude to use to get weather data (degrees).")
    field_area: Optional[float | SkipJsonSchema[None]] = SimulatorFieldInfo(
        default=None,
        examples=[1.0],
        description="Area of the field")
    weather_file: Optional[str | CliSuppress[BaseWeatherFile]
                           | SkipJsonSchema[None]] = SimulatorFieldInfo(
        default=None,
        description="Path to a file containing weather data.")
    soil_file: Optional[str | CliSuppress[BaseSoilFile]
                        | SkipJsonSchema[None]] = SimulatorFieldInfo(
        default=None,
        description="Path to a file containing soil data.")

    @field_validator('sow_date', 'harvest_date', mode="before")
    @classmethod
    def check_date(cls, v):
        r"""Parse date strings in ISO 8601 format."""
        if isinstance(v, str):
            return datetime.date.fromisoformat(v)
        return v

    @field_validator('duration', 'season_length', mode="before")
    @classmethod
    def check_timedelta(cls, v):
        r"""Parse timedelta in days."""
        if isinstance(v, (int, float)):
            if v <= 0:
                return None
            return datetime.timedelta(days=v)
        return v

    @classmethod
    def default_server_fields(cls) -> dict:
        r"""dict: The default fields that should be used for a server."""
        out = super().default_server_fields()
        out.update(
            year=None,
            sow_date=None,
            harvest_date=None,
            weather_file=None,
            soil_file=None,
        )
        return out

    def create_model_file(self) -> CropModelFile:
        r"""Create a model input file.

        Returns:
            CropModelFile: Constructed model input file.

        """
        if not self.crop_name:
            raise ValueError("Either model_file or crop_name must "
                             "be provided")
        return self.INPUT_FILE_TYPE.from_crop_name(
            self.crop_name,
            crop_variety=self.crop_variety,
            dst=self.model_file,
            directory=self.output_dir,
            interactive=(not self.non_interactive),
            actions=list(self.actions.keys()),
        )

    def update_model_file(self) -> None:
        r"""Update the model file to make it interactive and set the
        start/end times."""
        if self.latitude or self.longitude:
            assert self.latitude and self.longitude
        self.sync_param(["duration"], required=True, skip_file=True,
                        dont_update=True)
        self.sync_param(["start_time", "end_time", "year"],
                        dont_update=True, required=True)
        self.sync_param("weather_file", dont_update=True)
        if ((self.weather_file and not os.path.isfile(self.weather_file)
             and os.path.basename(self.weather_file) == "Dalby.met")):
            self.weather_file = self.weather_file.replace(
                "Dalby.met", "AU_Dalby.met")
        sync_called = False
        for category, data_cls in [
                ("weather", self.WEATHER_FILE_TYPE),
                # TODO
                # ("soil", self.SOIL_FILE_TYPE),
        ]:
            param = f"{category}_file"
            fname = getattr(self, param)
            if not (fname
                    or self.has_param("latitude", skip_file=True)
                    or self.has_param("longitude", skip_file=True)):
                continue
            if not sync_called:
                self.sync_param(["latitude", "longitude"],
                                dont_update=True, required=True)
            if data_cls.check_file_coverage(
                    fname,
                    self.latitude, self.longitude,
                    self.start_time, self.end_time,
            ):
                continue
            self.del_param(param)
            new_file = data_cls.fetch_data(
                self.latitude, self.longitude,
                self.start_time, self.end_time,
            )
            setattr(self, param, new_file)
            logger.info(
                f"Downloaded {category} data: \"{new_file}\"")
        if self.sow_date is not None:
            self.actions.pop("sow", None)
        if self.harvest_date is not None:
            self.actions.pop("harvest", None)
        super().update_model_file()

    @classmethod
    def create_and_run(cls, timestep: Optional[int] = 0,
                       plot: Optional[Any] = False,
                       **kwargs: Any) -> str:
        r"""Create and run a simulation using this simulator.

        Args:
            timestep: Time between interactive actions (in days).
                0 for a non-interactive continuous simulation.
            plot: If True or string, the simulation results will be
                plot. If a string is provided, the plot will be saved
                to the specified path. If a matplotlib axes object is
                provided, the simulation results will be added to the
                axes.
            **kwargs: Additional keyword arguments are passed to the
                class constructor.

        Returns:
            str: The path to the simulator output.

        """
        if kwargs.get("crop_name", None) == "all":
            figure, axes = cls._setup_plot(plot)
            kwargs.pop("crop_name")
            out = {}
            crops = cls.INPUT_FILE_TYPE.available_crops()
            for k in crops:
                out[k] = cls.create_and_run(timestep=timestep,
                                            crop_name=k,
                                            plot=axes,
                                            **kwargs)
            cls._finalize_plot(plot, figure, axes, legend=True)
            return out
        if kwargs.get("crop_variety", None) == "all":
            figure, axes = cls._setup_plot(plot)
            assert kwargs.get("crop_name", None)
            kwargs.pop("crop_variety")
            out = {}
            cultivars = cls.INPUT_FILE_TYPE.available_cultivars(
                kwargs["crop_name"])
            for k in cultivars:
                out[k] = cls.create_and_run(timestep=timestep,
                                            crop_variety=k,
                                            plot=axes,
                                            **kwargs)
            cls._finalize_plot(plot, figure, axes, legend=True)
            return out
        return super().create_and_run(timestep=timestep, plot=plot,
                                      **kwargs)

    def calc_param(self, name: str, default: Optional[Any] = NoDefault,
                   **kwargs: Any) -> Any:
        r"""Calculate a parameter from other parameters.

        Args:
            name: Parameter to calculate.
            default: Default to return if the parameter cannot be
                calculated.
            **kwargs: Additional keyword arguments are passed to
                parent class's calc_param.

        Returns:
            Calculated parameter value.

        """
        if name == "year":
            for k, _, _ in self.DATE_PARAM:
                try:
                    v = self.get_param(k, **kwargs)
                    return v.year
                except KeyError:
                    pass
        return super().calc_param(name, default=default, **kwargs)

    @property
    def location(self) -> str:
        r"""str: Description of the field location."""
        return self.model.location


class CropModelLLMPromptGenerator(BaseModelLLMPromptGenerator):
    r"""Crop model LLM prompt generator."""

    DEFAULT_REWARD: ClassVar[str] = (
        "Maximize the end-of-season profit from the crop yield"
    )
    DEFAULT_STATE_DESCRIPTOR: ClassVar[str] = "agronomic"

    crop_name: Optional[str] = Field(
        default="the crop",
        description="Name of the crop being cultivated.")
    crop_variety: Optional[str] = Field(
        default=None,
        description="Name of the crop variant/cultivar being "
                    "cultivated.")
    start_date: Optional[datetime.date] = Field(
        default=None,
        description="Calendar start date of the simulation.")
    season_length: Optional[int] = Field(
        default=241,
        description="Total length of growing season in days.")
    location: Optional[str] = Field(
        default="the field",
        description="Geographic location description.")

    @classmethod
    def from_env(
            cls,
            env: "CropModelEnv",
            **kwargs: Any
    ) -> "CropModelLLMPromptGenerator":
        """Create prompt generator from a model gym environment.

        Args:
            env: model gym environment instance
            **kwargs: Additional keyword arguments are passed to the
                class constructor.

        Returns:
            CropModelLLMPromptGenerator instance configured for the
                environment.

        """
        return super().from_env(
            env,
            crop_name=env.model.crop_name,
            crop_variety=env.model.crop_variety,
            start_date=env.start_time,
            season_length=env.model.season_length,
            location=env.model.location,
            **kwargs
        )

    @property
    def crop_description(self) -> str:
        r"""str: Crop description including provided info."""
        out = self.crop_name
        if self.crop_variety:
            out = f"{self.crop_variety} {out}"
        return out

    def get_system_prompt(self) -> str:
        """Generate the system prompt for the LLM agent.

        Returns:
            System prompt string
        """
        # TODO: Include action costs
        base_prompt = (
            f"You are an agricultural management expert whose goal is "
            f"to {self.reward_inline} "
            f"throughout the entire growing season by selecting the "
            f"appropriate agricultural management action at each "
            f"step."
        )
        return base_prompt

    def turn_context(self, observation: np.ndarray) -> str:
        r"""Generate a string to summarize the current turn at a high
        level with the context of the simulation.

        Args:
            observation: Current state.

        Returns:
            str: Turn summary.

        """
        day_val = None
        for i, var in enumerate(self.output_vars):
            section, description = self.desc_map.get(var, ('Other', var))
            if ((section == "Timeline"
                 and description.upper().startswith("DAYS"))):
                day_val = observation[i]
                break
        day_num = day_val if day_val is not None else 0

        calendar_phrase = ""
        if self.start_date is not None and day_val is not None:
            try:
                calendar_date = (
                    self.start_date
                    + datetime.timedelta(
                        days=max(int(round(float(day_val))) - 1, 0))
                )
                calendar_phrase = (
                    f", corresponding to {calendar_date.strftime('%B')} "
                    f"{calendar_date.day}"
                )
            except Exception:
                pass

        intro = (
            f"We are growing {self.crop_description} from sowing to "
            f"maturity. The planned growing window spans "
            f"{self.season_length} days, actions are taken every "
            f"{self.intervention_interval} days, "
            f"and today is day {day_num:.0f} of the "
            f"season{calendar_phrase}."
        )
        return intro


class CropModelEnv(BaseModelEnv):
    r"""Crop model environment."""

    def get_output_vars(self) -> List[str]:
        r"""Get the output variables specified by the model file.

        Returns:
            list: Output variables

        """
        out = super().get_output_vars()
        return out + ["days_elapsed"]

    def _get_cost(self, action: dict) -> float:
        r"""Calculate the cost of the current action.

        Args:
            action: Current action.

        Returns:
            float: Action cost.

        """
        return self.model.field_area * super()._get_cost(action)

    def _get_revenue(self, obs: dict) -> float:
        r"""Calculate the revenue based on the current observation.

        Args:
            obs: Current observation.

        Returns:
            float: Revenue.

        """
        return self.model.field_area * super()._get_revenue(obs)

    def _process_observation(self, observation: dict):
        r"""Force the observations into the expected format.

        Args:
            observation: Raw observations.

        Returns:
            np.ndarray: Array of observations parameters.

        """
        observation = super()._process_observation(observation)
        days_elapsed = self.model.current_time - self.model.start_time
        return np.concatenate([
            observation,
            [days_elapsed.days],
        ])
