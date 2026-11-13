"""G4 (`test_g4_addressed_instances`) on workload 101's stored fifth Build, fetched from the evidence store."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from verity_vllm.check.match.per_request import addressed_instances
from verity_vllm.check.match.program_compare import load_derived

#: workload 101's fifth try: its Build (`build_request/` and `build_workload/`)
BUILD_101_TRY5 = "art:0e6911da449a5b1e47564d66e947b81fb7d81c4b10ff5b814bb72b47cc1db1d0"


@pytest.mark.slow
def test_101s_fifth_build_is_addressed_as_its_component_is_compared(tmp_path):
    """Workload 101's stored fifth Build: the address map's `n_instances` for r0 equals its request Program's compared instances plus its 32 bound
    splits constants (46,654 + 32 = 46,686), the count G4 failed on."""
    from tests.regression import store_io
    if os.environ.get("VERITY_SKIP_STORE") == "1" or not store_io.reachable():
        pytest.skip("the evidence store is not reachable here")
    root = tmp_path / "build"
    store_io.fetch(BUILD_101_TRY5, to=root, paths=("*build_request/*", "*build_workload/workload_program.json"))
    req = next(Path(p).parent for p in root.rglob("build_request/descriptor.json.gz"))
    wp = next(root.rglob("build_workload/workload_program.json"))
    declared = next(r["n_instances"] for r in json.loads(wp.read_text())["address_map"]["requests"] if r["request_id"] == "r0")
    A = load_derived(str(req), "component")
    assert (declared, len(A.inst), A.n_splits_consts_bound) == (46686, 46654, 32)
    assert addressed_instances(A) == declared
