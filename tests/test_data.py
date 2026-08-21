import os
import shutil
import pytest
from simulatr.utils import cfg
from simulatr.base import _ModelFileMeta


class TestDataBase:

    __test__ = False
    _category = None
    _name = None

    @pytest.fixture(scope="class")
    @classmethod
    def category(cls):
        return cls._category

    @pytest.fixture(scope="class")
    @classmethod
    def name(cls):
        return cls._name

    @pytest.fixture(scope="class")
    @classmethod
    def data_cls(cls, category, name):
        if name is None or category is None:
            pytest.skip("Data file category or name not defined (base class)")
        out = _ModelFileMeta.get_filetype(category, name)
        return out

    @pytest.fixture(scope="class")
    @classmethod
    def latitude(cls):
        r"""float: Test latitude."""
        return 40.0

    @pytest.fixture(scope="class")
    @classmethod
    def longitude(cls):
        r"""float: Test longitude."""
        return -88.0

    @pytest.fixture(scope="class")
    @classmethod
    def start_date(cls, data_cls):
        r"""datetime.date: Start date."""
        if data_cls.DEFAULT_DATE_RANGE is None:
            return None
        return data_cls.DEFAULT_DATE_RANGE[0]

    @pytest.fixture(scope="class")
    @classmethod
    def end_date(cls, data_cls):
        r"""datetime.date: End date."""
        if data_cls.DEFAULT_DATE_RANGE is None:
            return None
        return data_cls.DEFAULT_DATE_RANGE[1]

    @pytest.fixture(scope="class")
    @classmethod
    def cache_dir(cls, name):
        r"""str: Directory for test data."""
        name_dir = name.replace(" ", "_")
        out = os.path.join(os.getcwd(), f"test_cache_{name_dir}")
        assert not os.path.exists(out)
        try:
            os.mkdir(out)
            yield out
        finally:
            shutil.rmtree(out)

    @pytest.fixture(scope="class")
    @classmethod
    def args(cls, latitude, longitude, start_date, end_date):
        r"""tuple: Instance arguments."""
        return (latitude, longitude, start_date, end_date)

    @pytest.fixture(scope="class")
    @classmethod
    def instance(cls, data_cls, args, cache_dir):
        out = data_cls.from_location(*args, cache_dir=cache_dir)
        assert os.path.isfile(out.fname)
        try:
            yield out
        finally:
            os.remove(out.fname)

    def test_attributes(self, instance, latitude, longitude):
        assert instance.parameters
        assert instance.latitude == latitude
        assert instance.longitude == longitude
        # assert instance.start_date == start_date
        # assert instance.end_date == end_date

    def test_from_location(self, instance, data_cls, args, cache_dir):
        r"""Check caching that a new file is not created."""
        assert os.path.isfile(instance.fname)
        out = data_cls.from_location(*args, cache_dir=cache_dir)
        assert out.fname == instance.fname

    def test_covers_location(self, instance):
        r"""Test covers_location method."""
        valid_args = [
            (instance.latitude, instance.longitude,
             instance.start_date, instance.end_date),
        ]
        invalid_args = [
            (0.0, 0.0, instance.start_date, instance.end_date),
        ]
        if instance.time_specific():
            span = instance.end_date - instance.start_date
            buff = span / 10
            valid_args += [
                (instance.latitude, instance.longitude,
                 instance.start_date + buff,
                 instance.end_date),
                (instance.latitude, instance.longitude,
                 instance.start_date,
                 instance.end_date - buff),
            ]
            invalid_args += [
                (instance.latitude, instance.longitude,
                 instance.start_date - buff,
                 instance.end_date),
                (instance.latitude, instance.longitude,
                 instance.start_date,
                 instance.end_date + buff),
                (instance.latitude, instance.longitude,
                 instance.end_date,
                 instance.end_date + span),
                (instance.latitude, instance.longitude,
                 instance.start_date - span,
                 instance.start_date),
            ]
        for args in valid_args:
            assert instance.covers_location(*args)
        for args in invalid_args:
            assert not instance.covers_location(*args)


##########################################################
# Weather
##########################################################

class TestWeatherBase(TestDataBase):

    __test__ = True
    _category = "weather"

    @pytest.fixture(scope="class", params=list(
        _ModelFileMeta.get_filetype_registry("weather").keys()
    ))
    @classmethod
    def name(cls, request):
        return request.param


##########################################################
# Soil
##########################################################

class TestSoilBase(TestDataBase):

    __test__ = True
    _category = "soil"

    @pytest.fixture(scope="class", params=[
        # TODO: This is skipped by default until the API is stable
        pytest.param(k, marks=pytest.mark.unstable)
        if k == "ISRIC SoilGrids"
        else (pytest.param(k, marks=pytest.mark.slow)
              if k == "HUMERIS" else k)
        for k in _ModelFileMeta.get_filetype_registry("soil").keys()
    ])
    @classmethod
    def name(cls, request):
        if ((request.param == "HUMERIS"
             and not os.path.isdir(cfg["directories"]["humeris_soil_data"]))):
            pytest.skip("HUMERIS is not already installed and download "
                        "takes a long time")
        return request.param

    @pytest.fixture(scope="class")
    @classmethod
    def args(cls, latitude, longitude):
        r"""tuple: Arguments."""
        return (latitude, longitude)
