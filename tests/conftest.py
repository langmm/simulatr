import numpy as np
import pytest


_markers_disabled_by_default = ["slow", "unstable"]


def pytest_addoption(parser):
    parser.addoption(
        '--service-location', dest="service_location",
        type=str, choices=["local", "remote", "docker"],
        default="local",
        help="Location where the test server is running or should be run",
    )
    for k in _markers_disabled_by_default:
        parser.addoption(
            f'--run-{k}', action='store_true', dest=f"run_{k}",
            default=False,
            help=f"enable tests marked as {k}")


def pytest_configure(config):
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
def assert_allclose():
    r"""Assert that arrays are close."""

    def _assert_allclose(a, b, rtol=1e-07, atol=1e-15, **kwargs):
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, **kwargs)

    return _assert_allclose


@pytest.fixture(scope="session")
def assert_nested_allclose(assert_allclose):
    from collections import OrderedDict

    def _assert_nested_allclose(a, b, ignore_keys=None,
                                only_keys=None, **kwargs):
        errors = {}
        if isinstance(b, (list, tuple)):
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
        elif isinstance(b, (np.ndarray, float)):
            assert_allclose(a, b, **kwargs)
        else:
            assert a == b
        if errors:
            raise NestedAssertionError(errors)

    return _assert_nested_allclose
