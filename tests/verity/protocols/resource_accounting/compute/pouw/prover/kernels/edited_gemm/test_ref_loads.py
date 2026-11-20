"""The reference that reads floored_gemm's derive.py, cgemm_ref.py, loads it from its file. Started by its path in a fresh
interpreter, from another directory, as a job script starts it, it leaves `sys.path` as it found it and binds no module
named `derive` or `ref`: the names other directories' scripts import their own modules by."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = (Path(__file__).resolve().parents[9] / "verity" / "protocols" / "resource_accounting" / "compute" / "pouw" / "prover" / "kernels" / "edited_gemm")

PROBE = """
import runpy, sys
before = list(sys.path)
names = runpy.run_path(sys.argv[1], run_name="started")
assert sys.path == before, [p for p in sys.path if p not in before]
assert not {"derive", "ref"} & set(sys.modules), sorted({"derive", "ref"} & set(sys.modules))
assert names["NCT"] == 512 and callable(names["tile_receipts"])
"""


def test_the_reference_started_by_its_path_edits_no_path_and_binds_no_shared_name(tmp_path):
    r = subprocess.run([sys.executable, "-c", PROBE, str(HERE / "cgemm_ref.py")], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
