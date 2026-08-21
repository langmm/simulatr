import os
import uuid
import shutil
from typing import Optional
import pytest
from simulatr.data import FileMeta


def _get_pytest_param(category: str,
                      for_download: Optional[bool] = False):
    return [
        pytest.param(k, marks=[
            getattr(pytest.mark, kmark) for kmark in v.PYTEST_MARKS
            if for_download or kmark != "slow"
        ])
        if v.PYTEST_MARKS else k
        for k, v in FileMeta.get_registry(category).items()
        if k is not None
    ]


def parametrize_data(category: str,
                     for_download: Optional[bool] = False):
    return pytest.mark.parametrize(
        "name", _get_pytest_param(category, for_download=for_download),
        indirect=True)


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
    def name(cls, request):
        return request.param

    @pytest.fixture(scope="class")
    @classmethod
    def get_data_cls(cls, category):

        def _get_data_cls(name):
            if name is None or category is None:
                pytest.skip("Data file category or name not defined "
                            "(base class)")
            return FileMeta.get_class(category, name)

        return _get_data_cls

    @pytest.fixture(scope="class")
    @classmethod
    def get_example_data(cls, get_data_cls, args, data_dir):

        def _get_example_data(name, return_fname=False):
            data_cls = get_data_cls(name)
            # TODO: Arguments for compatible class when this class
            #   has STATIC_LOCATION_LIMITS, but the compatible class
            #   does not. Raise error in from_compatible
            if data_cls.STATIC_LOCATION_LIMITS is None and args[0] is None:
                pytest.skip("Cannot create location specific file "
                            "without a specific location")
            suffix = ""
            if data_cls.DEFAULT_EXTERNAL_TYPE is not None:
                suffix = f"_from_{data_cls.DEFAULT_EXTERNAL_TYPE.NAME}"
            if name == "HUMERIS":
                fname = data_cls.DEFAULT_CACHE_DIR
            else:
                fname = data_cls.format_filename(
                    *args, cache_dir=data_dir,
                    suffix=suffix)
            if not data_cls.path_exists(fname):
                # # data_cls.from_location(*args, cache_dir=data_dir)
                # print(name, fname)
                # import pdb; pdb.set_trace()
                pytest.skip(f"No example data for {name}: {fname}")
            if return_fname:
                return fname
            return data_cls(fname)

        return _get_example_data

    @pytest.fixture(scope="class")
    @classmethod
    def data_cls(cls, get_data_cls, name):
        return get_data_cls(name)

    @pytest.fixture(scope="class")
    @classmethod
    def latitude(cls, data_cls):
        r"""float: Test latitude."""
        if data_cls.STATIC_LOCATION_LIMITS is not None:
            return None  # data_cls.STATIC_LOCATION_LIMITS[:2]
        return 40.116  # Limit of NASA POWER

    @pytest.fixture(scope="class")
    @classmethod
    def longitude(cls, data_cls):
        r"""float: Test longitude."""
        if data_cls.STATIC_LOCATION_LIMITS is not None:
            return None  # data_cls.STATIC_LOCATION_LIMITS[2:]
        return -88.243  # Limit of NASA POWER

    @pytest.fixture(scope="class")
    @classmethod
    def start_date(cls, data_cls):
        r"""datetime.date: Start date."""
        if data_cls.STATIC_DATE_LIMITS is not None:
            return None  # data_cls.STATIC_DATE_LIMITS[0]
        if data_cls.DEFAULT_DATE_RANGE is None:
            return data_cls.DEFAULT_EXTERNAL_TYPE.DEFAULT_DATE_RANGE[0]
        return data_cls.DEFAULT_DATE_RANGE[0]

    @pytest.fixture(scope="class")
    @classmethod
    def end_date(cls, data_cls):
        r"""datetime.date: End date."""
        if data_cls.STATIC_DATE_LIMITS is not None:
            return None  # data_cls.STATIC_DATE_LIMITS[1]
        if data_cls.DEFAULT_DATE_RANGE is None:
            return data_cls.DEFAULT_EXTERNAL_TYPE.DEFAULT_DATE_RANGE[1]
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
    def instance(cls, name, get_example_data):
        return get_example_data(name)


class TestDataDownload(TestDataBase):
    r"""Test base class for download."""

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

    @pytest.mark.download
    def test_download(self, data_cls, args, cache_dir):
        r"""Test download."""
        if data_cls.URL is None:
            pytest.skip(f"Cannot download {data_cls.DESC} "
                        f"{data_cls.CATEGORY} data")
        # print("DOWNLOAD", data_cls.NAME, data_cls.PYTEST_MARKS)
        # import pdb; pdb.set_trace()
        fname = data_cls.format_filename(*args, cache_dir=cache_dir)
        assert not os.path.isfile(fname)
        try:
            out = data_cls.from_location(*args, cache_dir=cache_dir)
            assert out.fname == fname
            assert out.exists
            assert out.generated
        finally:
            if os.path.isfile(fname):
                os.remove(fname)


class TestDataBaseNoDownload(TestDataBase):
    r"""Test base class with downloads disabled."""

    @pytest.fixture
    @classmethod
    def name_compatible(cls, name):
        return name

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def download_disabled(cls, category):

        def fake_download_data(*args, **kwargs):
            raise RuntimeError("Download called")

        with pytest.MonkeyPatch.context() as mp:
            for v in FileMeta.get_registry(category).values():
                mp.setattr(v, "download_and_save_data",
                           fake_download_data)
            yield

    def test_attributes(self, instance, latitude, longitude,
                        start_date, end_date, assert_allclose):
        r"""Test basic attributes."""
        assert instance.parameters
        if instance.location_specific():
            assert instance.latitude == instance._round_location(latitude)
            assert instance.longitude == instance._round_location(longitude)
        if instance.time_specific():
            assert instance.start_date == start_date
            assert instance.end_date == end_date

    def test_from_compatible(self, data_cls, data_dir,
                             name_compatible, get_example_data):
        r"""Test creation of file from a compatible file type."""
        compatible_example = get_example_data(name_compatible)
        fname = os.path.join(
            data_dir, str(uuid.uuid4()) + data_cls._default_ext)
        assert not data_cls.path_exists(fname)
        try:
            out = data_cls.from_compatible(
                compatible_example, fname=fname)
            assert out.fname == fname
            assert out.exists
            assert out.generated
        except NotImplementedError:
            if name_compatible in data_cls.REQUIRED_EXTERNAL_PARAMETERS:
                raise
            pytest.skip(f"Conversion from {name_compatible} to "
                        f"{data_cls.NAME} not implemented")
        finally:
            if os.path.isfile(fname):
                os.remove(fname)

    def test_from_location(self, instance, data_cls, args, data_dir):
        r"""Check caching that a new file is not created."""
        assert data_cls.path_exists(instance.fname)
        assert instance.exists
        assert not instance.generated
        if data_cls.NAME == "HUMERIS":
            out = data_cls.from_location(*args)
        else:
            out = data_cls.from_location(*args, cache_dir=data_dir)
        assert out.fname == instance.fname

    def test_covers_location(self, instance):
        r"""Test covers_location method."""
        valid_args = [
            (instance.latitude, instance.longitude,
             instance.start_date, instance.end_date),
        ]
        invalid_args = []
        if instance.location_specific():
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

@parametrize_data("weather")
class TestWeatherDate(TestDataBaseNoDownload):

    __test__ = True
    _category = "weather"

    @pytest.fixture(params=_get_pytest_param("weather"))
    @classmethod
    def name_compatible(cls, request):
        return request.param


@parametrize_data("weather", for_download=True)
class TestWeatherDateDownload(TestDataDownload):

    __test__ = True
    _category = "weather"


##########################################################
# Soil
##########################################################

@parametrize_data("soil")
class TestSoilBase(TestDataBaseNoDownload):

    __test__ = True
    _category = "soil"

    @pytest.fixture(params=_get_pytest_param("soil"))
    @classmethod
    def name_compatible(cls, request):
        return request.param


@parametrize_data("soil", for_download=True)
class TestSoilBaseDownload(TestDataDownload):

    __test__ = True
    _category = "soil"
