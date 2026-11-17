import os
import uuid
import shutil
import datetime
from typing import Optional
import pytest
from simulatr.data import FileMeta, BaseSoilFile
from simulatr.utils import promptuser_boolean
from yggdrasil_rapidjson import units as rj_units


_select_classes = {}


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
    out = pytest.mark.parametrize(
        "name", _get_pytest_param(category, for_download=for_download),
        indirect=True)
    if for_download:

        def _decorator(cls):
            return pytest.mark.download(out(cls))

        return _decorator
    return out


class TestSaxonRawlsSoilCalculations:
    r"""Verify that soil calculations return the expected values."""

    @pytest.fixture(scope="class")
    @classmethod
    def texture_class_map(cls):
        return {
            "Sa": "sand",
            "L": "loam",
            "Si": "silt",
            "C": "clay",
        }

    @pytest.fixture(params=[
        ("Sa", 88, 5, 5, 10, 46, 5, 108.1, 1.43),
        ("LSa", 80, 5, 5, 12, 46, 7, 96.7, 1.43),
        ("SaL", 65, 10, 8, 18, 45, 10, 50.3, 1.46),
        ("L", 40, 20, 14, 28, 46, 14, 15.5, 1.43),
        ("SiL", 20, 15, 11, 31, 48, 20, 16.1, 1.38),
        ("Si", 10, 5, 6, 30, 48, 25, 22.0, 1.38),
        ("SaCL", 60, 25, 17, 27, 43, 10, 11.3, 1.50),
        ("CL", 30, 35, 22, 36, 48, 14, 4.3, 1.39),
        ("SiCL", 10, 35, 22, 38, 51, 17, 5.7, 1.30),
        ("SiC", 10, 45, 27, 41, 52, 14, 3.7, 1.26),
        ("SaC", 50, 40, 25, 36, 44, 11, 1.4, 1.47),
        ("C", 25, 50, 30, 42, 50, 12, 1.1, 1.3),
    ])
    @classmethod
    def table_3_params(cls, request):
        return request.param

    @pytest.fixture
    @classmethod
    def om(cls):
        return rj_units.Quantity(2.5, "%")

    @pytest.fixture
    @classmethod
    def texture_class(cls, table_3_params, texture_class_map):
        orig = table_3_params[0]
        out = []
        while orig:
            for k, v in texture_class_map.items():
                if orig.startswith(k):
                    out.append(v)
                    orig = orig.split(k, 1)[-1]
                    break
        for i in range(len(out) - 1):
            if not out[i].endswith('y'):
                out[i] += 'y'
        return ' '.join(out)

    @pytest.fixture
    @classmethod
    def sand(cls, table_3_params):
        return rj_units.Quantity(float(table_3_params[1]), "%")

    @pytest.fixture
    @classmethod
    def clay(cls, table_3_params):
        return rj_units.Quantity(float(table_3_params[2]), "%")

    @pytest.fixture
    @classmethod
    def wilting_point(cls, table_3_params):
        return rj_units.Quantity(float(table_3_params[3]), "%")

    @pytest.fixture
    @classmethod
    def dul(cls, table_3_params):
        return rj_units.Quantity(float(table_3_params[4]), "%")

    @pytest.fixture
    @classmethod
    def sat(cls, table_3_params):
        return rj_units.Quantity(float(table_3_params[5]), "%")

    @pytest.fixture
    @classmethod
    def available(cls, table_3_params):
        return rj_units.Quantity(float(table_3_params[6]), "%")

    @pytest.fixture
    @classmethod
    def ksat(cls, table_3_params):
        return rj_units.Quantity(float(table_3_params[7]), "mm/hr")

    @pytest.fixture
    @classmethod
    def matric_density(cls, table_3_params):
        return rj_units.Quantity(float(table_3_params[8]), "g/cm**3")

    @pytest.fixture(scope="class")
    @classmethod
    def assert_values_close(cls, assert_nested_allclose):

        def _assert_values_close(a, b, **kwargs):
            kwargs.setdefault('convert_units', True)
            kwargs.setdefault('rtol', 0.05)
            assert_nested_allclose(a, b, **kwargs)

        return _assert_values_close

    def test_dul_from_texture(self, clay, sand, om, dul,
                              assert_values_close):
        assert_values_close(
            BaseSoilFile.dul_from_texture(clay, sand, om),
            dul)

    def test_wilting_point_from_texture(self, clay, sand, om,
                                        wilting_point,
                                        assert_values_close):
        assert_values_close(
            BaseSoilFile.wilting_point_from_texture(clay, sand, om),
            wilting_point)

    def test_sat_from_texture(self, clay, sand, om, sat,
                              assert_values_close):
        assert_values_close(
            BaseSoilFile.sat_from_texture(clay, sand, om),
            sat)

    # To compare we would need to calculate the intermediate values
    # def test_ksat_from_moisture(self, sat, dul, wilting_point, ksat,
    #                             assert_values_close):
    #     assert_values_close(
    #         BaseSoilFile.ksat_from_moisture(sat, dul, wilting_point),
    #         ksat)

    def test_ksat_from_texture(self, clay, sand, om, ksat,
                               assert_values_close):
        assert_values_close(
            BaseSoilFile.ksat_from_texture(clay, sand, om),
            ksat)

    def test_texture_class(self, clay, sand, texture_class):
        assert BaseSoilFile.texture_class_usda(clay, sand) == texture_class


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
    def download_enabled(cls, request):
        for mark in request.node.own_markers:
            if mark.name == "download":
                return True
        return False

    @pytest.fixture(scope="class")
    @classmethod
    def get_example_data(cls, get_data_cls, args, download_enabled,
                         create_missing_data,
                         simulator_data_classes):

        def _get_example_data(name, return_fname=False,
                              compatible_type=None,
                              optional=False,
                              using_args=args,
                              allow_download=download_enabled):
            data_cls = get_data_cls(name)
            # TODO: Arguments for compatible class when this class
            #   has STATIC_PARAMETERS, but the compatible class
            #   does not. Raise error in from_compatible
            if (("latitude" not in data_cls.STATIC_PARAMETERS
                 and using_args[0] is None)):
                pytest.skip("Cannot create location specific file "
                            "without a specific location")
            if (("start_date" not in data_cls.STATIC_PARAMETERS
                 and using_args[2] is None)):
                pytest.skip("Cannot create time specific file "
                            "without a specific location")
            fname = data_cls.format_filename(
                *using_args, src=compatible_type)
            fname_exists = data_cls.path_exists(fname)
            if fname_exists:
                allow_download = False
            elif (compatible_type
                  and name not in simulator_data_classes):
                # Disable conversion to non-simulator formats
                allow_download = False
            elif create_missing_data and (not allow_download):
                allow_download = promptuser_boolean(
                    f"Create data for {data_cls.DESC} "
                    f"{data_cls.CATEGORY} file at {fname}?",
                    _gha_default=False,
                )
            if allow_download and not fname_exists:
                data_cls.from_location(*using_args, src=compatible_type)
            if (not optional) and (not data_cls.path_exists(fname)):
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
        return 40.1164

    @pytest.fixture(scope="class")
    @classmethod
    def longitude(cls, data_cls):
        r"""float: Test longitude."""
        if "longitude" in data_cls.STATIC_PARAMETERS:
            return None  # data_cls.STATIC_PARAMETERS["longitude"]
        return -88.2434

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

    @pytest.fixture(scope="class")
    @classmethod
    def assert_data_allclose(cls, assert_nested_allclose):

        def _assert_data_allclose(a, b, **kwargs):

            a_param = a.universal_parameter_map(include_calculated=True)
            b_param = b.universal_parameter_map(include_calculated=True)
            assert_nested_allclose(a_param, b_param, **kwargs)

        return _assert_data_allclose

    @pytest.fixture
    @classmethod
    def name_compare(cls, name):
        if name == 'soilmat_data':
            return 'isric_soilgrids_data'
        pytest.skip(f"No data type to compare against {name}")

    @pytest.fixture
    @classmethod
    def args_compare(cls, args):
        return args

    def test_compare(self, name, name_compare, args_compare,
                     get_example_data, assert_data_allclose,
                     assert_nested_allclose):
        r"""Test equivalence of values against another data source."""
        local_example = get_example_data(name,
                                         using_args=args_compare)
        other_example = get_example_data(name_compare,
                                         using_args=args_compare)
        kws = {}
        if name == "soilmat_data" and name_compare == "isric_soilgrids_data":
            # a_mean = other_example.universal_parameter_map(
            #     include_calculated=True)
            other_example.set("quantile", "Q0.5")
            # a_median = other_example.universal_parameter_map(
            #     include_calculated=True)
            # assert_nested_allclose(a_mean, a_median)
            # print("SUCCESS")
            # import pdb; pdb.set_trace()
            kws["atol"] = {
                k: v.max() for k, v in local_example.stddev.items()
            }
            # import pprint; pprint.pprint(kws["atol"])
            # import pdb; pdb.set_trace()
        assert_data_allclose(
            local_example, other_example, convert_units=True, **kws
            # rtol=0.4,
        )


class TestDataDownload(TestDataBase):
    r"""Test base class for download."""

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def using_temp_dir(cls, category, name, data_dir,
                       alternate_data_dir):
        name_dir = name.replace(" ", "_")
        temp_dir = os.path.join(os.getcwd(), f"test_cache_{name_dir}")
        assert not os.path.exists(temp_dir)
        try:
            os.mkdir(temp_dir)
            with alternate_data_dir(temp_dir, category):
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

    @pytest.fixture
    @classmethod
    def args_compare(cls, args, sampled_location):
        return (*sampled_location, *args[2:])


class TestDataBaseNoDownload(TestDataBase):
    r"""Test base class with downloads disabled."""

    @pytest.fixture
    @classmethod
    def name_compatible(cls, name):
        return name

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def using_test_data(cls, category, data_dir, alternate_data_dir):
        with alternate_data_dir(data_dir, category, skip_keys=["HUMERIS"]):
            yield

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def download_disabled(cls, category, download_disabled,
                          create_missing_data):
        if create_missing_data:
            yield
        else:
            with download_disabled(category):
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
        out = instance.universal_parameter_map(
            include_header=True, include_calculated=True)
        import pprint
        pprint.pprint(out)
        assert out

    def test_from_compatible(self, data_cls, data_dir,
                             name_compatible, get_example_data,
                             compare_bytes_files,
                             assert_nested_allclose):
        r"""Test creation of file from a compatible file type."""
        compatible_example = get_example_data(name_compatible)
        compatible_example_result = get_example_data(
            data_cls.NAME,  # return_fname=True,
            compatible_type=name_compatible, optional=True)
        fname = os.path.join(
            data_dir, str(uuid.uuid4()) + data_cls._default_ext)
        assert not data_cls.path_exists(fname)
        try:
            out = data_cls.from_compatible(
                compatible_example, fname=fname)
            assert out.fname == fname
            assert out.exists
            assert out.generated
            if compatible_example_result.exists:
                # compare_bytes_files(
                #     out.fname, compatible_example_result.fname)
                assert_nested_allclose(
                    out.contents, compatible_example_result.contents,
                    rtol=1e-06
                )
            elif data_cls.NAME == name_compatible:
                pass
            elif data_cls.NAME != "apsimx_soil":
                pass
            elif promptuser_boolean(
                    f"Create data for conversion from {name_compatible} "
                    f"to {data_cls.NAME}?", _gha_default=False):
                shutil.copy(out.fname, compatible_example_result.fname)
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

    def test_calculations(self, instance,
                          assert_nested_allclose):
        r"""Compare calculations against provided values."""
        external = instance.external_parameters(include_header=True)
        param_int = {}
        param_cal = {}
        param_err = {}
        for k, v in instance.PARAMETER_CALCULATORS.items():
            if not (k in external
                    and instance.can_calculate_parameter(
                        k, previous=external)):
                continue
            result_int = instance.get(k)
            for x in v:
                try:
                    result_cal = x(instance)
                    param_cal.setdefault(k, {})
                    param_int.setdefault(k, {})
                    param_cal[k][x.method] = result_cal
                    param_int[k][x.method] = result_int
                except KeyError as e:
                    param_err.setdefault(k, {})
                    param_err[k][x.method] = e
        assert_nested_allclose(param_cal, param_int, rtol=0.3,
                               convert_units=True)


##########################################################
# Weather
##########################################################

@parametrize_data("weather")
class TestWeatherDate(TestDataBaseNoDownload):

    __test__ = True
    _category = "weather"

    @pytest.fixture(params=_get_pytest_param("weather"))
    @classmethod
    def name_compatible(cls, name, request):
        if request.param == name:
            pytest.skip("Don't test conversion from the same class")
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
    def name_compatible(cls, name, request):
        if request.param == name:
            pytest.skip("Don't test conversion from the same class")
        return request.param

    def test_soil_attributes(self, instance):
        r"""Test soil specific attributes."""
        assert instance.depths


@parametrize_data("soil", for_download=True)
class TestSoilBaseDownload(TestDataDownload):

    __test__ = True
    _category = "soil"
