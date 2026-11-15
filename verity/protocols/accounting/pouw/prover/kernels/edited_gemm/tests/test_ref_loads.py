"""The four references that read floored_gemm's derive.py load it from its file. Started by its path in a fresh
interpreter, from another directory, as a job script starts it, each leaves `sys.path` as it found it and binds no module
named `derive` or `ref`: the names other directories' scripts import their own modules by."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]

PROBE = """
import runpy, sys
before = list(sys.path)
names = runpy.run_path(sys.argv[1], run_name="started")
assert sys.path == before, [p for p in sys.path if p not in before]
assert not {"derive", "ref"} & set(sys.modules), sorted({"derive", "ref"} & set(sys.modules))
assert names["NCT"] == 512 and callable(names["tile_receipts"])
"""


@pytest.mark.parametrize("script", ["cgemm_ref.py", "d2sgemm_ref.py", "egemm80_ref.py", "egemm_ref.py"])
def test_a_reference_started_by_its_path_edits_no_path_and_binds_no_shared_name(script, tmp_path):
    r = subprocess.run([sys.executable, "-c", PROBE, str(HERE / script)], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
