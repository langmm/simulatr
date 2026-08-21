import os
import datetime
import pytest
from simulatr.apsimx import ApsimXEngine


class TestApsimX:
    r"""Test class for ensuring that apsimx is producing consistent
    results."""

    @pytest.fixture(scope="class", params=[
        ("SimpleWheat.apsimx", None),
        ("SimpleWheatUnmanaged.apsimx", None),
        ({"crop_name": "Wheat"},
         "SimpleWheatUnmanaged.Report.csv.expected"),
        ("SimpleWheatUnmanagedChampaign.apsimx", None),
        ({
            "crop_name": "Wheat",
            "start_time": datetime.datetime(year=1991, month=1, day=1),
            "end_time": datetime.datetime(year=1991, month=11, day=5),
            "latitude": 40.1164,
            "longitude": -88.2434,
          },
         "SimpleWheatUnmanagedChampaign.Report.csv.expected"),
    ], ids=[
        "wheat", "wheat-unmanaged", "wheat-generated",
        "wheat-unmanaged-champaign",
        "wheat-generated-champaign",
    ])
    @classmethod
    def parameters(cls, request):
        return request.param

    @pytest.fixture(scope="class")
    @classmethod
    def input_generated(cls, parameters):
        return isinstance(parameters[0], dict)

    @pytest.fixture(scope="class")
    @classmethod
    def input_file(cls, data_dir, parameters):
        if isinstance(parameters[0], str):
            out = os.path.join(data_dir, parameters[0])
            assert os.path.isfile(out)
            return out
        assert isinstance(parameters[0], dict)
        crop_name = parameters[0]['crop_name']
        kws = {k: v for k, v in parameters[0].items() if k != "crop_name"}
        out = os.path.join(data_dir, f"{crop_name}Generated.apsimx")
        assert not os.path.isfile(out)
        fout = ApsimXEngine.INPUT_FILE_TYPE.from_crop_name(
            crop_name, dst=out, **kws)
        fout.write()
        fout.generated = False  # Prevent cleanup on fout deletion
        assert os.path.isfile(out)
        return out

    @pytest.fixture(scope="class")
    @classmethod
    def output_file(cls, input_file):
        return os.path.splitext(input_file)[0] + ".Report.csv"

    @pytest.fixture(scope="class")
    @classmethod
    def expected_file(cls, data_dir, parameters, output_file):
        if parameters[1]:
            out = os.path.join(data_dir, parameters[1])
        else:
            out = output_file + ".expected"
        assert os.path.isfile(out)
        return out

    @pytest.fixture(scope="class")
    @classmethod
    def products(cls, input_file, output_file, input_generated):
        out = [output_file]
        if input_generated:
            out.append(input_file)
        for ext in [".db", ".db-shm", ".db-wal"]:
            out.append(os.path.splitext(input_file)[0] + ext)
        return out

    @pytest.fixture(scope="class")
    @classmethod
    def compare_reports(cls):
        import pandas as pd

        def _compare_reports(factual, fexpected):
            actual = pd.read_csv(factual)
            expected = pd.read_csv(fexpected)
            # rtol = 0.4 (Grain number)
            # atol = 0.002 (LAI)
            try:
                pd.testing.assert_frame_equal(
                    actual, expected, check_exact=False,
                    rtol=0.01)
            except AssertionError:
                if all(actual.columns == expected.columns):
                    print(expected.columns)
                    for x in expected.columns:
                        if pd.api.types.is_string_dtype(expected[x]):
                            continue
                        adiff = (expected[x] - actual[x]).abs()
                        rdiff = adiff / expected[x]
                        print(f"{x}: adiff = {max(adiff)}, rdiff = "
                              f"{max(rdiff.dropna())}")
                raise

        return _compare_reports

    def test_run(self, input_file, expected_file, output_file,
                 compare_reports, products):
        try:
            process = ApsimXEngine.start_direct_subprocess(
                input_file,
                csv=True,
            )
            process.wait(10)
            assert os.path.isfile(output_file)
            compare_reports(output_file, expected_file)
        finally:
            for x in products:
                if os.path.isfile(x):
                    os.remove(x)
