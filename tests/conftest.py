import numpy as np
import pandas as pd
import pytest
import os
import contextlib
import shutil
from simulatr.data import FileMeta
from simulatr.utils import promptuser_boolean
from yggdrasil_rapidjson import units as rj_units


_markers_disabled_by_default = [
    "slow", "download",
]
# TODO: Use pytest option to control this via the command line
_create_missing = False


def pytest_addoption(parser):
    parser.addoption(
        '--service-location', dest="service_location",
        type=str, choices=["local", "remote", "docker"],
        default="local",
        help="Location where the test server is running or should be run",
    )
    for k in _markers_disabled_by_default + ["slow_download"]:
        parser.addoption(
            f'--run-{k.replace("_", "-")}',
            action='store_true', dest=f"run_{k}",
            default=False,
            help=f"enable tests marked as {k}")


def pytest_configure(config):
    if getattr(config.option, "run_slow_download"):
        config.option.run_download = True
        config.option.run_slow = True
    for k in _markers_disabled_by_default:
        if not getattr(config.option, f"run_{k}"):
            if not config.getoption("markexpr"):
                config.option.markexpr = f"not {k}"
            elif k not in config.getoption("markexpr"):
                markexpr = config.getoption("markexpr")
                config.option.markexpr = f"({markexpr}) and not {k}"


class NestedAssertionError(AssertionError):

    def __init__(self, nested):
        self.nested = nested
        msg = ''
        for k, v in nested.items():
            msg += f'\n\n{k}\n\t' + v.replace('\n', '\n\t')
        super(NestedAssertionError, self).__init__(msg)


@pytest.fixture(scope="session")
def data_dir() -> str:
    r"""Directory containing test data."""
    return os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "data")


@pytest.fixture(scope="session")
def alternate_data_dir() -> str:

    def fake_download_data(*args, **kwargs):
        raise RuntimeError("Download called")

    @contextlib.contextmanager
    def _alternate_data_dir(altdir, categories=None,
                            skip_keys=None, download_disabled=False):
        if skip_keys is None:
            skip_keys = []
        if categories is None:
            categories = list(FileMeta._registry.keys())
        elif isinstance(categories, str):
            categories = [categories]
        with pytest.MonkeyPatch.context() as mp:
            for category in categories:
                for k, v in FileMeta.get_registry(category).items():
                    if ((k in skip_keys
                         or not hasattr(v, "DEFAULT_CACHE_DIR"))):
                        continue
                    mp.setattr(v, "DEFAULT_CACHE_DIR", altdir)
                    if download_disabled:
                        mp.setattr(v, "download_and_save_data",
                                   fake_download_data)
            yield

    return _alternate_data_dir


@pytest.fixture(scope="session")
def create_missing_data():
    if not _create_missing:
        return False

    def _create_missing_data(actual, expected, desc=""):
        if os.path.isfile(expected):
            return
        if promptuser_boolean(
                f"Create missing expected data {expected} "
                f"from test data {actual}{desc}?", _gha_default=False):
            shutil.copy(actual, expected)

    return _create_missing_data


@pytest.fixture(scope="session")
def download_disabled():

    def fake_download_data(*args, **kwargs):
        raise RuntimeError("Download called")

    @contextlib.contextmanager
    def _download_disabled(categories=None, skip_keys=None):
        if skip_keys is None:
            skip_keys = []
        if categories is None:
            categories = [FileMeta._registry.keys()]
        elif isinstance(categories, str):
            categories = [categories]
        with pytest.MonkeyPatch.context() as mp:
            for category in categories:
                for k, v in FileMeta.get_registry(category).items():
                    if k in skip_keys:
                        continue
                    mp.setattr(v, "download_and_save_data",
                               fake_download_data)
            yield

    return _download_disabled


@pytest.fixture
def using_test_data_dir(alternate_data_dir, data_dir):

    with alternate_data_dir(data_dir, skip_keys=["HUMERIS"]):
        yield


@pytest.fixture(scope="session")
def assert_allclose():
    r"""Assert that arrays are close."""

    def _assert_allclose(a, b, rtol=1e-07, atol=1e-15, **kwargs):
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, **kwargs)

    return _assert_allclose


@pytest.fixture(scope="session")
def assert_nested_allclose(assert_allclose):
    from collections import OrderedDict

    def _assert_nested_allclose(a, b, ignore_keys=None,
                                only_keys=None, convert_units=False,
                                **kwargs):
        errors = {}
        if isinstance(b, (list, tuple)):
            kwargs['convert_units'] = convert_units
            assert isinstance(a, type(b))
            assert len(a) == len(b)
            for i, (ia, ib) in enumerate(zip(a, b)):
                try:
                    _assert_nested_allclose(ia, ib, **kwargs)
                except AssertionError as e:
                    if isinstance(e, NestedAssertionError):
                        for kerr, verr in e.nested.items():
                            errors[f'{i}->{kerr}'] = verr
                    else:
                        errors[f'{i}'] = e.args[0]
        elif isinstance(b, (dict, OrderedDict)):
            kwargs['convert_units'] = convert_units
            assert isinstance(a, type(b))
            a_keys = list(sorted(a.keys()))
            b_keys = list(sorted(b.keys()))
            if ignore_keys:
                a_keys = [k for k in a_keys if k not in ignore_keys]
                b_keys = [k for k in b_keys if k not in ignore_keys]
            if only_keys:
                a_keys = [k for k in a_keys if k in only_keys]
                b_keys = [k for k in b_keys if k in only_keys]
            assert a_keys == b_keys
            for k in b_keys:
                try:
                    _assert_nested_allclose(a[k], b[k], **kwargs)
                except AssertionError as e:
                    if isinstance(e, NestedAssertionError):
                        for kerr, verr in e.nested.items():
                            errors[f'{k}->{kerr}'] = verr
                    else:
                        errors[k] = e.args[0]
        elif isinstance(b, (rj_units.Quantity, rj_units.QuantityArray)):
            assert isinstance(a, type(b))
            if convert_units:
                assert a.is_compatible(b.units)
                a = a.to(b.units)
            assert str(a.units) == str(b.units)
            assert_allclose(a.value, b.value, **kwargs)
        elif isinstance(b, (np.ndarray, pd.DataFrame, float)):
            assert_allclose(a, b, **kwargs)
        else:
            assert a == b
        if errors:
            raise NestedAssertionError(errors)

    return _assert_nested_allclose


@pytest.fixture(scope="session")
def compare_bytes():
    r"""Compare bytes in chunks.

    Args:
        actual (bytes): Actual bytes.
        expected (bytes): Expected bytes.

    """

    def _compare_bytes(actual, expected):
        chunk_size = 1000
        len_actual = len(actual)
        len_expected = len(expected)
        pos = 0
        maxpos = max([len_actual, len_expected])
        while pos < maxpos:
            pos_act = min(pos, len_actual)
            pos_exp = min(pos, len_expected)
            chunk_act = actual[
                pos_act:min(pos_act + chunk_size, len_actual)]
            chunk_exp = expected[
                pos_exp:min(pos_exp + chunk_size, len_expected)]
            assert chunk_act == chunk_exp
            pos += chunk_size
        assert len_actual == len_expected
        assert actual == expected

    return _compare_bytes


@pytest.fixture(scope="session")
def compare_bytes_files(compare_bytes):
    r"""Compare files by bytes.

    Args:
        actual (str): Actual file path.
        expected (str): Expected file path.

    """

    def _compare_bytes_files(actual, expected):
        assert os.path.isfile(actual)
        with open(actual, 'rb') as fd:
            actual_bytes = fd.read()
        with open(expected, 'rb') as fd:
            expected_bytes = fd.read()
        compare_bytes(actual_bytes, expected_bytes)

    return _compare_bytes_files


_LAND_BOXES = [
    (30.0, 60.0, -120.0, -75.0),    # North America
    (-35.0, 5.0, -75.0, -45.0),     # South America
    (40.0, 60.0, 0.0, 30.0),        # Europe
    (-30.0, 32.0, 15.0, 40.0),      # Africa
    (20.0, 55.0, 60.0, 100.0),      # Central & South Asia
    (-35.0, -15.0, 120.0, 145.0),   # Australia
]


@pytest.fixture(scope="session")
def land_boxes():
    r"""List: Bounding boxes (latitude_min, latitude_max, longitude_min,
    longitude_max) covering predominantly land areas on several
    continents. Used to draw a random sample of locations over land.
    """
    return _LAND_BOXES


def _sample_locations(n_samples, seed=20261015):
    rng = np.random.default_rng(seed)
    boxes = np.asarray(_LAND_BOXES)
    locations = []
    while len(locations) < n_samples:
        latitude_min, latitude_max, longitude_min, longitude_max = (
            boxes[rng.integers(0, len(boxes))])
        latitude = float(np.round(rng.uniform(
            latitude_min, latitude_max)))
        longitude = float(np.round(rng.uniform(
            longitude_min, longitude_max)))
        if (latitude, longitude) not in locations:
            locations.append((latitude, longitude))
    return locations


@pytest.fixture(params=_sample_locations(3))
def sampled_location(request):
    r"""tuple: Randomly sampled (latitude, longitude) pair."""
    return request.param


@pytest.fixture(scope="session")
def sample_locations():
    r"""Randomly sample (latitude, longitude) pairs over land.

    Locations are rounded to whole degrees since the ISRIC
    SoilGrids REST API only returns data for integer coordinates.

    Args:
        n_samples (int): Number of samples to return.
        seed (int, optional): Seed to use for the random number generator.

    Returns:
        list: List of (latitude, longitude) pairs.

    """
    return _sample_locations
