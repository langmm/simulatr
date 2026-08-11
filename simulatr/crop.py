import os
import numpy as np
import datetime
from abc import abstractmethod
from typing import Optional, Union, List, Any, ClassVar
from pydantic import ConfigDict, Field, field_validator
from .base import (
    NoDefault,
    BaseModelFile, BaseModelEngine,
    BaseModelLLMPromptGenerator, BaseModelEnv,
)
from . import logger


class CropModelFile(BaseModelFile):
    r"""Base class for managing crop model input files."""

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
        raise NotImplementedError(
            f"Invalid crop name \"{crop_name}\". "
            f"Valid crop names are:\n\t"
            + "\n\t".join(available_crops))

    @classmethod
    @abstractmethod
    def from_crop_name(cls, crop_name: str, dst: str | None = None,
                       interactive: bool = False,
                       actions: List[str] | None = None) -> "CropModelFile":
        r"""Create an input model file for a given crop name.

        Args:
            crop_name: Crop name.
            dst: Path to the location where the generated file should
                be saved.
            interactive: If True, make the file interactive.
            actions: Interactive actions that should be added.

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
    r"""Class for managining communication with a crop simulation model.

    Args:
        model_file: Path to one or more model input files.
        crop_name: Name of the crop.
        crop_variety: Name of the crop variety/cultivar.
        sow_date: Date that the crop should be sown.
        harvest_date: Date that the crop should be harvested.
        season_length: Time between sowing and harvest. Only used if
            only one of sow_date or harvest_date are used. If an integer
            is provided, it is assumed to be in units of days.
        year: Year to use to get weather data.
        latitude: Field latitude to use to get weather data.
        longitude: Field longitude to use to get weather data.
        weather_file: Path to a file containing NASA power weather data.
        **kwargs: Additional keywords arguments are passed along to
            BaseModelEngine.__init__.

    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    EXPLICIT_PARAM: ClassVar[list] = BaseModelEngine.EXPLICIT_PARAM + [
        "crop_name", "crop_variety",
        "sow_date", "harvest_date", "season_length",
        "latitude", "longitude", "year",
        "weather_file",
    ]
    WEATHER_FILE_TYPE: ClassVar[Any] = None
    DATE_PARAM: ClassVar[list] = BaseModelEngine.DATE_PARAM + [
        ("sow_date", "harvest_date", "season_length"),
    ]
    DEFAULT_PARAM: ClassVar[dict] = {
        "duration": datetime.timedelta(365),
    }

    model_file: Optional[Union[str, List[str], BaseModelFile]] = Field(
        default=None,
        description="Path to one or more model input files.")
    crop_name: Optional[str] = Field(
        default=None, examples=["Wheat"],
        description="Name of the crop that will be simulated")
    crop_variety: Optional[str] = Field(
        default=None, examples=["Herzog"],
        description="Name of the crop variety/cultivar that will be "
                    "simulated")
    sow_date: Optional[datetime.date] = Field(
        default=None,
        examples=[datetime.datetime.fromisoformat("1991-01-01")],
        description="Date that the crop should be sown (ISO 8601 format).")
    harvest_date: Optional[datetime.date] = Field(
        default=None,
        examples=[datetime.datetime.fromisoformat("1991-11-05")],
        description="Date that the crop should be harvested "
                    "(ISO 8601 format).")
    season_length: Optional[Union[int, float, datetime.timedelta]] = Field(
        default=None,
        examples=[datetime.timedelta(365)],
        description="Time between sowing and harvest. Only used if only "
                    "one of sow_date or harvest_date are used. If an "
                    "integer is provided, it is assumed to be in units "
                    "of days.")
    year: Optional[int] = Field(
        default=None,
        examples=[1991],
        description="Year to use to get weather data.")
    latitude: Optional[float] = Field(
        default=None,
        examples=[40.1164],
        description="Field latitude to use to get weather data (degrees).")
    longitude: Optional[float] = Field(
        default=None,
        examples=[-88.2434],
        description="Field longitude to use to get weather data (degrees).")
    weather_file: Optional[str] = Field(
        default=None,
        description="Path to a file containing weather data.")
    soil_file: Optional[str] = Field(
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
            dst=self.model_file,
            interactive=True,
            actions=list(self.action_map.keys()),
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
        download_weather_file = False
        if ((self.weather_file and not os.path.isfile(self.weather_file)
             and os.path.basename(self.weather_file) == "Dalby.met")):
            self.weather_file = self.weather_file.replace(
                "Dalby.met", "AU_Dalby.met")
        if self.weather_file and not os.path.isfile(self.weather_file):
            logger.info(
                f"The specified weather file (\"{self.weather_file}\") "
                f"does not exist. Weather data will be downloaded from "
                f"{self.WEATHER_FILE_TYPE.base_cls().NAME}"
            )
            self.del_param("weather_file")
            download_weather_file = True
        elif self.weather_file:
            weather_file = self.WEATHER_FILE_TYPE(self.weather_file)
            latitude = self.get_param("latitude", None, skip_file=True)
            longitude = self.get_param("longitude", None, skip_file=True)
            if not weather_file.covers_location(
                    latitude=latitude, longitude=longitude,
                    start=self.start_time, end=self.end_time,
            ):
                logger.info(
                    f"The provided weather file (valid for "
                    f"{weather_file.start_date} to "
                    f"{weather_file.end_date}, "
                    f"latitude={weather_file.latitude}, "
                    f"longitude={weather_file.longitude}) does not "
                    f"cover the simulation range ({self.start_time} to "
                    f"{self.end_time}, latitude={latitude}, "
                    f"longitude={longitude}). "
                    f"Weather data will be downloaded from "
                    f"{self.WEATHER_FILE_TYPE.base_cls().NAME}"
                )
                self.del_param("weather_file")
                download_weather_file = True
        elif (self.has_param("latitude", skip_file=True)
              or self.has_param("longitude", skip_file=True)):
            download_weather_file = True
        if download_weather_file:
            self.sync_param(["latitude", "longitude"],
                            dont_update=True, required=True)
            self.weather_file = self.WEATHER_FILE_TYPE.fetch_data(
                self.latitude, self.longitude,
                self.start_time, self.end_time,
            )
            logger.info(
                f"Downloaded weather data: \"{self.weather_file}\"")
        # TODO: Soil file?
        if self.sow_date is not None:
            self.action_map.pop("sow", None)
        if self.harvest_date is not None:
            self.action_map.pop("harvest", None)
        super().update_model_file()

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

    @property
    def field_area(self) -> float:
        r"""float: Field area"""
        return self.model.field_area


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
