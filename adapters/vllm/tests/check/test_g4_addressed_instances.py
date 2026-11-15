"""G4 (`per_request._compare_component`): the global Program's address map declares each request's component by its root nodes
(`n_instances`), and the component the Match loads has `RootView._bind_splits` remove a single-request Program's splits constants (one
`Const32[S]` per top-p select) from its compared sequence.  So G4 compares the address map with the compared sequence plus those bound
constants (`addressed_instances`).  Workload 101's fifth Match failed G4 by exactly its 32 selects: 46,686 declared, 46,654
compared; its stored Build is `test_g4_addressed_instances_of_record`.

The comparison itself runs on the batched toy of `test_global_match.py`, with the two claims a Build's artifact makes of a
request's component in the request's entry of the global Program (`declaration.from_gprog` puts them there): the component's
digest and the address map's count.  G4 names the request and both values where a claim is not the loaded component's."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")   # the helpers imported next derive a Program with torch

from tests.check import test_global_match as tgm
from tests.pipeline.test_single_request_build_path import (
    _compose,
    _derive,
    _write_request,
)
from verity_vllm.check.match import global_match as gm
from verity_vllm.check.match import global_match_fast as gmf
from verity_vllm.check.match.per_request import addressed_instances
from verity_vllm.check.match.program_compare import load_derived
from verity_vllm.records import workload as WL

memory = tgm.memory                          # the toy's fixture, under its own name


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
    declared = json.loads((tmp_path / "build_workload" / "workload_program.json").read_text())["address_map"]["requests"][0]["n_instances"]
    A = load_derived(str(tmp_path / "build_request"), "component")
    assert A.n_splits_consts_bound == STEPS + 1 and len(A.inst) == declared - (STEPS + 1), (len(A.inst), declared)
    assert addressed_instances(A) == declared


def test_a_component_without_splits_constants_is_counted_as_compared():
    """A batched component reads S from its `splits` Input: nothing is bound, the compared sequence is the addressed one."""
    assert addressed_instances(SimpleNamespace(inst=[{}] * 5, n_splits_consts_bound=0)) == 5
    assert addressed_instances(SimpleNamespace(inst=[{}] * 5)) == 5


@pytest.mark.parametrize("impl", [gm, gmf], ids=["record", "fast"])
def test_g4_holds_a_builds_claims_of_a_requests_component_against_the_component_the_match_loads(impl, memory, tmp_path):
    gp, oracle = tgm.assemble([tgm.R0, tgm.R1], tmp_path / "record")
    loaded: dict = {}                                                      # the fold and the components, decoded once for every run below
    unclaimed = impl.check(gp, oracle, record_alternate=False, _loaded=loaded)
    assert unclaimed["verdict"] == "PASS" and not any("global_address" in e for e in unclaimed["requests"])
    comp = {e["request_id"]: e["component"] for e in unclaimed["requests"]}
    n, digest = comp["r1"]["instances"], comp["r1"]["digest"]

    def claimed(**of_r1) -> dict:
        """The Match's document for the Program whose entries carry each component's own digest and count, but for r1's `of_r1`."""
        g = copy.deepcopy(gp)
        for q in g["requests"]:
            q["component"]["program_digest"] = comp[q["request_id"]]["digest"]
            q["global_address"] = {"root_node": None, "gates": None, "n_instances": comp[q["request_id"]]["instances"], "rule": None}
        g["requests"][1]["component"].update(of_r1.get("component", {}))
        g["requests"][1]["global_address"].update(of_r1.get("global_address", {}))
        return impl.check(g, oracle, record_alternate=False, _loaded=loaded)

    def g4(res: dict) -> list[str]:
        assert res["verdict"] == "FAIL" and res["failed_checks"] == ["G4"], (res["verdict"], res["failed_checks"])
        return tgm._check(res, "G4")["problems"]

    true = claimed()
    assert true["verdict"] == "PASS" and [e["global_address"]["n_instances"] for e in true["requests"]] == [comp["r0"]["instances"], n]
    assert g4(claimed(global_address={"n_instances": n + 32})) == [
        f"r1: address map declares {n + 32} instances for the request, the component has {n} ({n} compared + 0 bound splits constants)"]
    uncounted = claimed(global_address={"n_instances": None})              # an address map that states no count: carried, not compared
    assert uncounted["verdict"] == "PASS" and uncounted["requests"][1]["global_address"]["n_instances"] is None
    assert g4(claimed(component={"program_digest": "0" * 128})) == [
        f"r1: the global Program's component digest {'0' * 16} != the loaded component's {digest[:16]}"]
    # a component whose loader bound three splits constants out of its compared sequence is addressed by three more than it compares
    loaded["components"][comp["r1"]["dir"]].n_splits_consts_bound = 3
    assert claimed(global_address={"n_instances": n + 3})["verdict"] == "PASS"
    assert g4(claimed()) == [f"r1: address map declares {n} instances for the request, the component has {n + 3} ({n} compared + 3 bound splits constants)"]
