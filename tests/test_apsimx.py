import os
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
    ], ids=[
        "wheat", "wheat-unmanaged", "wheat-generated",
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
    def compare_reports(cls, compare_bytes):

        def _compare_reports(factual, fexpected):
            with open(factual, "r") as fd:
                actual = fd.read()
            with open(fexpected, "r") as fd:
                expected = fd.read()
            compare_bytes(actual, expected)

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
