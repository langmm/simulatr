import os
import uuid
import shutil
import datetime
from typing import Optional
import pytest
from simulatr.data import FileMeta
from simulatr.utils import promptuser_boolean


_select_classes = {}
_create_data = False


def _get_pytest_param(category: str,
                      for_download: Optional[bool] = False):
    classes = FileMeta.get_registry(category)
    if _select_classes.get(category, None):
        classes = {k: classes[k] for k in _select_classes[category]}
    return [
        pytest.param(k, marks=[
            getattr(pytest.mark, kmark) for kmark in v.PYTEST_MARKS
            if for_download or kmark != "slow"
        ])
        if v.PYTEST_MARKS else k
        for k, v in classes.items()
        if k is not None and not v.DONT_TEST
    ]


def parametrize_data(category: str,
                     for_download: Optional[bool] = False):
    r"""Decorator for parametrize a test class for each registered
    class of a given data category.

    Args:
        category: Data file category.
        for_download: True if the class being decorated will perform
            downloads.

    Returns:
        pytest.mark.parametrize decoration function.

    """
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
            out = FileMeta.get_class(category, name)
            if not out.tools_installed():
                pytest.skip(
                    f"Not all of the required tools are "
                    f"installed: {out.REQUIRED_OPTIONAL_PACKAGES}")
            return out

        return _get_data_cls

    @pytest.fixture(scope="class")
    @classmethod
    def get_example_data(cls, get_data_cls, args):

        def _get_example_data(name, return_fname=False):
            data_cls = get_data_cls(name)
            # TODO: Arguments for compatible class when this class
            #   has STATIC_PARAMETERS, but the compatible class
            #   does not. Raise error in from_compatible
            if (("latitude" not in data_cls.STATIC_PARAMETERS
                 and args[0] is None)):
                pytest.skip("Cannot create location specific file "
                            "without a specific location")
            if (("start_date" not in data_cls.STATIC_PARAMETERS
                 and args[2] is None)):
                pytest.skip("Cannot create time specific file "
                            "without a specific location")
            suffix = ""
            if data_cls.DEFAULT_EXTERNAL_TYPE is not None:
                suffix = f"_from_{data_cls.DEFAULT_EXTERNAL_TYPE.NAME}"
            fname = data_cls.format_filename(*args, suffix=suffix)
            if _create_data and not data_cls.path_exists(fname):
                if promptuser_boolean(
                        f"Create data for {data_cls.DESC} "
                        f"{data_cls.CATEGORY} file at {fname}?",
                        _gha_default=False):
                    data_cls.from_location(*args)
            if not data_cls.path_exists(fname):
                pytest.skip(f"No example data for {name}: {fname}")
            if return_fname:
                return fname
            return data_cls(fname)

        return _get_example_data

    @pytest.fixture(scope="class")
    @classmethod
    def data_cls(cls, get_data_cls, name):
        r"""type: Class being tested."""
        return get_data_cls(name)

    @pytest.fixture(scope="class")
    @classmethod
    def latitude(cls, data_cls):
        r"""float: Test latitude."""
        if "latitude" in data_cls.STATIC_PARAMETERS:
            return None  # data_cls.STATIC_PARAMETERS["latitude"]
        return 40.116  # Precision limit of NASA POWER

    @pytest.fixture(scope="class")
    @classmethod
    def longitude(cls, data_cls):
        r"""float: Test longitude."""
        if "longitude" in data_cls.STATIC_PARAMETERS:
            return None  # data_cls.STATIC_PARAMETERS["longitude"]
        return -88.243  # Precision limit of NASA POWER

    @pytest.fixture(scope="class")
    @classmethod
    def start_date(cls, data_cls):
        r"""datetime.date: Start date."""
        if "start_date" in data_cls.STATIC_PARAMETERS:
            return None  # data_cls.STATIC_PARAMETERS["start_date"]
        if data_cls.DEFAULT_DATE_RANGE is not None:
            return data_cls.DEFAULT_DATE_RANGE[0]
        if data_cls.DEFAULT_EXTERNAL_TYPE:
            return data_cls.DEFAULT_EXTERNAL_TYPE.DEFAULT_DATE_RANGE[0]
        return datetime.date(year=2020, month=3, day=1)

    @pytest.fixture(scope="class")
    @classmethod
    def end_date(cls, data_cls):
        r"""datetime.date: End date."""
        if "end_date" in data_cls.STATIC_PARAMETERS:
            return None  # data_cls.STATIC_PARAMETERS["end_date"]
        if data_cls.DEFAULT_DATE_RANGE is not None:
            return data_cls.DEFAULT_DATE_RANGE[1]
        if data_cls.DEFAULT_EXTERNAL_TYPE:
            return data_cls.DEFAULT_EXTERNAL_TYPE.DEFAULT_DATE_RANGE[1]
        return datetime.date(year=2020, month=11, day=30)

    @pytest.fixture(scope="class")
    @classmethod
    def args(cls, latitude, longitude, start_date, end_date):
        r"""tuple: Instance arguments."""
        return (latitude, longitude, start_date, end_date)

    @pytest.fixture(scope="class")
    @classmethod
    def instance(cls, name, get_example_data):
        r"""Test instance of the targeted class."""
        return get_example_data(name)


class TestDataDownload(TestDataBase):
    r"""Test base class for download."""

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def using_temp_dir(cls, category, name, data_dir):
        name_dir = name.replace(" ", "_")
        temp_dir = os.path.join(os.getcwd(), f"test_cache_{name_dir}")
        assert not os.path.exists(temp_dir)
        try:
            os.mkdir(temp_dir)
            with pytest.MonkeyPatch.context() as mp:
                for k, v in FileMeta.get_registry(category).items():
                    mp.setattr(v, "DEFAULT_CACHE_DIR", temp_dir)
                yield
        finally:
            shutil.rmtree(temp_dir)

    @pytest.mark.download
    def test_download(self, data_cls, args):
        r"""Test download."""
        if data_cls.URL is None:
            pytest.skip(f"Cannot download {data_cls.DESC} "
                        f"{data_cls.CATEGORY} data")
        # print("DOWNLOAD", data_cls.NAME, data_cls.PYTEST_MARKS)
        # import pdb; pdb.set_trace()
        fname = data_cls.format_filename(*args)
        assert not os.path.isfile(fname)
        try:
            out = data_cls.from_location(*args)
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
    def using_test_data(cls, category, data_dir):
        with pytest.MonkeyPatch.context() as mp:
            for k, v in FileMeta.get_registry(category).items():
                if k != "HUMERIS":  # For local testing
                    mp.setattr(v, "DEFAULT_CACHE_DIR", data_dir)
            yield

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def download_disabled(cls, category):

        def fake_download_data(*args, **kwargs):
            raise RuntimeError("Download called")

        with pytest.MonkeyPatch.context() as mp:
            if not _create_data:
                for k, v in FileMeta.get_registry(category).items():
                    mp.setattr(v, "download_and_save_data",
                               fake_download_data)
            yield

    def test_attributes(self, instance, latitude, longitude,
                        start_date, end_date, assert_allclose):
        r"""Test basic attributes."""
        assert instance.internal_parameters
        assert instance.external_parameters
        if instance.location_specific():
            assert instance.latitude == instance._round_location(latitude)
            assert instance.longitude == instance._round_location(longitude)
        if instance.time_specific():
            assert instance.start_date == start_date
            assert instance.end_date == end_date

    def test_get(self, instance):
        r"""Test get for all listed parameters."""
        parameters = instance.internal_parameters(
            include_header=True, include_calculated=True)
        assert parameters
        print(parameters)
        for k in parameters:
            instance.get(k)

    def test_universal_parameter_map(self, instance):
        r"""Test universal_parameter_map."""
        out = instance.universal_parameter_map
        import pprint
        pprint.pprint(out)
        assert out

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

    def test_from_location(self, instance, data_cls, args):
        r"""Check caching that a new file is not created."""
        assert data_cls.path_exists(instance.fname)
        assert instance.exists
        assert not instance.generated
        out = data_cls.from_location(*args)
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

    def test_soil_attributes(self, instance):
        r"""Test soil specific attributes."""
        assert instance.depths


@parametrize_data("soil", for_download=True)
class TestSoilBaseDownload(TestDataDownload):

    __test__ = True
    _category = "soil"
