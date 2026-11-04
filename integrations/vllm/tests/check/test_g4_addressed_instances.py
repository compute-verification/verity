"""G4 (`per_request._compare_component`): the global Program's address map declares each request's component by its root nodes
(`n_instances`), and the component the Match loads has `RootView._bind_splits` remove a single-request Program's splits constants (one
`Const32[S]` per top-p select) from its compared sequence.  So G4 compares the address map with the compared sequence plus those bound
constants (`addressed_instances`).  Workload 101's fifth Match failed G4 by exactly its 32 selects: 46,686 declared, 46,654
compared."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.pipeline.test_single_request_build_path import _compose, _derive, _write_request
from verity_vllm.check.match.per_request import addressed_instances
from verity_vllm.check.match.program_compare import load_derived
from verity_vllm.records import workload as WL

#: workload 101's fifth try: its Build (`build_request/` and `build_workload/`)
BUILD_101_TRY5 = "art:0e6911da"


@pytest.mark.slow
def test_a_single_request_components_address_map_counts_its_splits_constants(tmp_path):
    """The Build's own path on CPU: a single-request stochastic Program (a constant S, one `Const32[S]` per select) composed by GP-01."""
    V, S, LP, STEPS = 64, 4, 3, 2
    _write_request(_derive(V, S, LP, STEPS), tmp_path / "build_request", LP, STEPS, S)
    wl = tmp_path / "workload.json"
    wl.write_text(json.dumps({"schema": WL.SCHEMA, "model": "toy", "engine": {}, "variants": {"stoch-t0.8-p0.95": {"temperature": 0.8, "top_p": 0.95, "top_k": 0}},
                              "requests": [{"request_id": "r-0", "prompt_token_ids": list(range(1, LP + 1)), "max_tokens": STEPS + 1, "arrive_step": 0}],
                              "sampling": {"variant": "stoch-t0.8-p0.95", "seed": 7}, "sweep": {"execution": "bi-eager"}}))
    _compose(wl, tmp_path / "build_request", tmp_path / "build_workload", eos=0)
    declared = json.load(open(tmp_path / "build_workload" / "workload_program.json"))["address_map"]["requests"][0]["n_instances"]
    A = load_derived(str(tmp_path / "build_request"), "component")
    assert A.n_splits_consts_bound == STEPS + 1 and len(A.inst) == declared - (STEPS + 1), (len(A.inst), declared)
    assert addressed_instances(A) == declared


def test_a_component_without_splits_constants_is_counted_as_compared():
    """A batched component reads S from its `splits` Input: nothing is bound, the compared sequence is the addressed one."""
    assert addressed_instances(SimpleNamespace(inst=[{}] * 5, n_splits_consts_bound=0)) == 5
    assert addressed_instances(SimpleNamespace(inst=[{}] * 5)) == 5


@pytest.mark.slow
def test_101s_fifth_build_is_addressed_as_its_component_is_compared(tmp_path):
    """Workload 101's stored fifth Build: the address map's `n_instances` for r0 equals its request Program's compared instances plus its 32 bound
    splits constants (46,654 + 32 = 46,686), the count G4 failed on."""
    from tests.regression import store_io
    if os.environ.get("VERITY_SKIP_STORE") == "1" or not store_io.reachable():
        pytest.skip("the evidence store is not reachable here")
    root = tmp_path / "build"
    store_io._cli("fetch", BUILD_101_TRY5, "--to", str(root), "--path", "*build_request/*", "--path", "*build_workload/workload_program.json")
    req = next(Path(p).parent for p in root.rglob("build_request/descriptor.json.gz"))
    wp = next(root.rglob("build_workload/workload_program.json"))
    declared = next(r["n_instances"] for r in json.load(open(wp))["address_map"]["requests"] if r["request_id"] == "r0")
    A = load_derived(str(req), "component")
    assert (declared, len(A.inst), A.n_splits_consts_bound) == (46686, 46654, 32)
    assert addressed_instances(A) == declared
