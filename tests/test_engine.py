import sys
import numpy as np
import datetime
import logging
import contextlib
import pytest
from simulatr import logger, get_simulator_class
from simulatr.base import (
    InvalidActionError, RecoverableError,
    RecoverableModelEngineError,
)
if sys.platform == 'win32':
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)


##########################################################
# Test base classes
##########################################################


def add_simulator_test(name: str):
    r"""Decorator for adding test class properties for a simulator.

    Args:
        name: Simulator name.

    Returns:
        Callable: Decorator.

    """

    def _add_simulator_test(cls):
        cls._name = name
        cls.__test__ = True
        return cls

    return _add_simulator_test


class TestBase:

    __test__ = False
    _name = None
    _class_type = None

    @pytest.fixture(scope="class")
    @classmethod
    def simulator_cls(cls):
        if cls._name is None or cls._class_type is None:
            pytest.skip("Simulator/file type not defined (base class)")
        out = get_simulator_class(cls._name, class_type=cls._class_type)
        if cls._class_type == "engine":
            is_installed = out.is_installed()
        else:
            is_installed = get_simulator_class(
                cls._name, "engine").is_installed()
        if not is_installed:
            pytest.skip(f"Simulator \"{cls._name}\" is not installed")
        return out

    @pytest.fixture(scope="class")
    @classmethod
    def default_instance_kwargs(cls):
        return {}

    @pytest.fixture(scope="class")
    @classmethod
    def example_model(cls):
        r"""str: Path to example model input file."""
        if cls._name is not None:
            file_cls = get_simulator_class(cls._name, class_type="file")
            if file_cls.EXAMPLE is not None:
                return file_cls.EXAMPLE
        pytest.skip("No example model defined")

    @pytest.fixture(scope="class", params=[])
    @classmethod
    def action_instance_kwargs(cls, request):
        return request.param


class TestModelFile(TestBase):

    _name = None
    _class_type = "file"


class TestModelEngine(TestBase):

    _name = None
    _class_type = "engine"

    @pytest.fixture(scope="class")
    @classmethod
    def new_instance(cls, simulator_cls, example_model,
                     default_instance_kwargs):

        @contextlib.contextmanager
        def _new_instance(model_file=None, **kwargs):
            if (("model_file" not in kwargs
                 and "crop_name" not in kwargs)):
                kwargs["model_file"] = example_model
            for k, v in default_instance_kwargs.items():
                kwargs.setdefault(k, v)
            instance = simulator_cls(**kwargs)
            instance.start()
            try:
                yield instance
            finally:
                instance.stop()
                instance.cleanup(remove_output=True)

        return _new_instance

    @pytest.fixture(scope="class")
    @classmethod
    def instance(cls, new_instance):
        with new_instance(model_suffix="-Prime") as instance:
            yield instance

    def test_attributes(self, instance):
        r"""Test instance attributes."""
        print(instance.get_output_vars())

    @pytest.mark.parametrize(
        "command,error,command_args", [
            ("set", RecoverableModelEngineError, (
                "invalid", 1,
            )),
            ("get", RecoverableModelEngineError, (
                "invalid",
            )),
            ("act", InvalidActionError, (
                "invalid",
            )),
            ("set", RecoverableModelEngineError, (
                "[Clock].Start",
                datetime.datetime(year=1981, month=1, day=1).timestamp()
            )),
            ("act", RecoverableModelEngineError, (
                "nitrogen", "hello",
            )),
        ], ids=[
            "set", "get", "act", "set_value", "act_value",
        ]
    )
    def test_engine_invalid(self, instance, new_instance,
                            command, error, command_args):
        r"""Test error & cleanup on setting a variable that causes
        re-initialization."""
        with new_instance() as instance2:
            with pytest.raises(error):
                getattr(instance2, command)(*command_args)
            if issubclass(error, RecoverableError):
                assert instance2.is_running
                assert not instance2.is_complete
                assert instance2.status == "paused"
            else:
                assert not instance2.is_running
                assert not instance2.is_complete
                assert instance2.status == "error"
        if issubclass(error, RecoverableError):
            getattr(instance, command)(*command_args, allow_error=True)
            assert instance.is_running
            assert not instance.is_complete
            assert instance.status == "paused"


class TestModelEnv(TestBase):

    _class_type = "env"

    @pytest.fixture(scope="class",
                    ids=["discrete", "continuous", "simultaneous"],
                    params=[
                        {},
                        # Continuous
                        {
                            "num_levels": 0
                        },
                        # Simultaneous
                        {
                            "exclusive": False,
                            "num_levels": 0,
                        },
                    ])
    @classmethod
    def action_instance_kwargs(cls, request, default_instance_kwargs):
        return dict(default_instance_kwargs, **request.param)

    @pytest.fixture(scope="class")
    @classmethod
    def continuous(cls, action_instance_kwargs):
        return action_instance_kwargs.get("num_levels", None) == 0

    @pytest.fixture(scope="class")
    @classmethod
    def simultaneous(cls, action_instance_kwargs):
        return action_instance_kwargs.get("exclusive", None) is False

    @pytest.fixture(scope="class")
    @classmethod
    def new_instance(cls, simulator_cls, example_model,
                     action_instance_kwargs):

        @contextlib.contextmanager
        def _new_instance(model_file=None, **kwargs):
            if (("model_file" not in kwargs
                 and "crop_name" not in kwargs)):
                kwargs["model_file"] = example_model
            for k, v in action_instance_kwargs.items():
                kwargs.setdefault(k, v)
            env = simulator_cls(**kwargs)
            try:
                yield env
            finally:
                env.close()
                env.model.cleanup(remove_output=True)

        return _new_instance

    @pytest.fixture(scope="class")
    @classmethod
    def instance(cls, new_instance):
        with new_instance(model_suffix="-Prime") as instance:
            yield instance

    @pytest.fixture(scope="class")
    @classmethod
    def is_valid_action(cls, continuous, simultaneous):

        def _is_valid_action(action):
            action_continuous = (not isinstance(action, int))
            action_simultaneous = isinstance(action, dict)
            if action_continuous != continuous:
                return False
            if action_simultaneous != simultaneous:
                return False
            return True

        return _is_valid_action

    @pytest.fixture(params=[])
    def action_id_base(request, continuous, simultaneous):
        return request.param

    @pytest.fixture
    def action_id(self, action_id_base, is_valid_action):
        r"""int: Action ID."""
        if is_valid_action(action_id_base):
            return action_id_base
        else:
            pytest.skip("Invalid action")

    @pytest.fixture
    def invalid_action_id(self, action_id_base, is_valid_action):
        r"""int: Action ID."""
        if is_valid_action(action_id_base):
            pytest.skip("Valid action")
        else:
            return action_id_base

    @pytest.fixture(scope="class")
    @classmethod
    def sampled_action_id(cls, instance):
        return instance.action_map.space.sample()

    def test_description(self, instance):
        r"""Test description generation."""
        instance.action_map.description

    def test_action_description(self, instance, action_id,
                                sampled_action_id,
                                assert_nested_allclose):
        r"""Test description generation."""
        desc = instance.action_map.action2description(action_id)
        assert instance.action_map.description2action(desc) == action_id
        desc = instance.action_map.action2description(sampled_action_id)
        assert_nested_allclose(
            instance.action_map.description2action(desc),
            sampled_action_id,
            atol=0.1
        )

    def test_invalid_action(self, instance, invalid_action_id):
        with pytest.raises(InvalidActionError):
            instance.action_map.action2description(invalid_action_id)

    def test_step(self, instance, sampled_action_id):
        r"""Test environment step."""
        instance.step(sampled_action_id)
        instance.reset()

    def test_prompt_generator(self, instance, action_id,
                              sampled_action_id, assert_nested_allclose):
        r"""Test creation of prompt generator."""
        prompt = instance.get_llm_prompt_generator()
        prompt.get_system_prompt()
        desc = prompt.describe_action(action_id)
        assert prompt.parse_action_response(desc) == action_id
        desc = prompt.describe_action(sampled_action_id)
        assert_nested_allclose(
            prompt.parse_action_response(desc), sampled_action_id,
            atol=0.1
        )
        assert prompt.parse_action_response("Invalid response") is None


##########################################################
# Tests for ApsimX model
##########################################################

@add_simulator_test("apsimx")
class TestApsimXFile(TestModelFile):

    def test_available_crops(self, simulator_cls):
        assert simulator_cls.available_crops()

    def test_available_cultivars(self, simulator_cls):
        assert simulator_cls.available_cultivars("wheat")


@add_simulator_test("apsimx")
class TestApsimXEngine(TestModelEngine):

    @pytest.fixture(scope="class")
    @classmethod
    def default_instance_kwargs(cls):
        return {
            "crop_name": "Wheat",
            "actions": ["nitrogen", "irrigate"],
            "start_time": datetime.datetime(year=1981, month=1, day=1),
            "end_time": datetime.datetime(year=1981, month=11, day=5),
        }

    def test_attributes(self, instance):
        r"""Test instance attributes."""
        assert instance.crop_name == "Wheat"
        assert instance.crop_variety == "Hartog"
        assert instance.location != "the field"
        assert instance.field_area == 1.0
        print(instance.get_output_vars())

    def test_loop(self, new_instance):
        r"""Test loop to apply fertilizer and irrigate."""
        with new_instance(
                latitude=40.1164, longitude=-88.2434,
                start_time=datetime.datetime(year=1984, month=7, day=1),
                end_time=datetime.datetime(year=1985, month=5, day=5),
        ) as instance:
            i = 0
            orig_value = 0.05
            new_value = 0.043
            reply = instance.get(
                "[Grain].MaximumPotentialGrainSize.FixedValue")
            assert reply == orig_value
            instance.set(
                "[Grain].MaximumPotentialGrainSize.FixedValue",
                new_value)
            reply = instance.get(
                "[Grain].MaximumPotentialGrainSize.FixedValue")
            assert reply == new_value
            while instance.is_running and not instance.is_complete:
                instance.getvars([
                    "[Wheat].Phenology.Zadok.Stage",
                    "[Soil].Water.PAW",
                ])
                # Decision point
                if i % 2 == 0:
                    instance.act("nitrogen", 160.0)  # kg/ha
                else:
                    instance.act("irrigate", 10.0)  # mm
                # reply = instance.get("[Nutrient].NO3.kgha")
                # new_value = [2*ele for ele in reply]
                # instance.set("[Nutrient].NO3.kgha", new_value)
                # reply = instance.get("[Nutrient].NO3.kgha")
                # assert reply == new_value
                reply = instance.getvars([
                    "[Wheat].Phenology.Zadok.Stage",
                    "[Wheat].Grain.Total.Wt",
                    "[Fertiliser].NitrogenApplied",
                    "[Irrigation].IrrigationApplied",
                ])
                reply["[Clock].Today"] = instance.current_time
                assert isinstance(reply, dict)
                # import pprint
                # pprint.pprint(reply)
                instance.fast_forward(datetime.timedelta(days=10))
                i += 1

    def test_scrub(self, instance):
        r"""Test rewind/fast-forward."""
        start_time = instance.start_time
        end_time = instance.end_time
        # start_time = datetime.datetime(year=1981, month=1, day=1)
        # end_time = datetime.datetime(year=1981, month=11, day=5)
        time = start_time.replace(month=5, day=23)
        # Run complete simulation without fertilizing
        assert instance.current_time == start_time
        instance.fast_forward()
        assert instance.current_time == end_time
        assert instance.is_running
        value_none = instance.get("[Wheat].Grain.Total.Wt")
        # print("NONE", instance.current_time, value_none)
        # Rewind and run again with full
        instance.rewind()
        instance.rewind(datetime.timedelta(days=20))
        assert instance.current_time == start_time
        while instance.is_running and not instance.is_complete:
            instance.act("nitrogen", 0.001)
            instance.fast_forward(datetime.timedelta(days=20))
        assert instance.current_time == end_time
        value_full = instance.get("[Wheat].Grain.Total.Wt")
        # print("FULL", instance.current_time, value_full)
        # Rewind halfway and run without from there
        instance.rewind(time)
        assert instance.current_time == time
        instance.fast_forward()
        assert instance.current_time == end_time
        instance.fast_forward(datetime.timedelta(days=20))
        value_half = instance.get("[Wheat].Grain.Total.Wt")
        # print("HALF", instance.current_time, value_half)
        # Compare
        assert value_half > value_none
        assert value_full > value_half
        instance.rewind()
        assert instance.current_time == start_time

    def test_action_param(self, new_instance):
        r"""Test actions with parameters."""
        with new_instance(
                actions=[
                    "nitrogen", "irrigate", "tillage", "sow", "harvest"
                ],
                action_param={
                    "sow": {"population": 10.0, "crop_name": "Wheat"},
                },
        ) as instance:
            instance.act("tillage", "disc")
            instance.act("nitrogen", 160.0, type="UreaN")
            instance.act("sow", sowingDepth=10.0)
            instance.fast_forward(
                datetime.datetime(year=1981, month=5, day=23))
            instance.act("harvest")


@add_simulator_test("apsimx")
class TestApsimXEnv(TestModelEnv):
    r"""Test use of ApsimXEnv for exclusive, discrete actions."""

    @pytest.fixture(scope="class")
    @classmethod
    def default_instance_kwargs(cls):
        return {
            "start_time": datetime.datetime(year=1981, month=1, day=1),
            "end_time": datetime.datetime(year=1981, month=12, day=31),
        }

    @pytest.fixture(params=[
        # Discrete
        0,
        1,
        # Continuous
        (0, 0),
        (1, np.array([0.5])),
        # Simultaneous
        {
            "nitrogen": np.array([0.5]),
            "irrigate": np.array([0.5]),
        },
        # Invalid for all or just simultaneous?
        # np.zeros((5, )),
    ])
    def action_id_base(self, request, continuous, simultaneous):
        return request.param
