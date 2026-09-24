"""The replay's address rule under a v2-query manifest (P2): `ProgramIndex(rule="v2-query")` names a row's members from the v2 query's
population (`v1_bridge.Population.members_of`), the rows `request_manifest` counts.  On every row where the v1 vocabulary and the v2 query
agree the two rules address the row identically; where v2 requires more (F-r19-int-20: an interior row whose output crosses a module
boundary) the v2 rule addresses it and the v2 manifest's identity has its rows."""
from __future__ import annotations

import pytest

from verity_capture.commit import sampled_replay as SR
from verity_capture.commit.tests.test_sampled_replay import _program
from verity_vllm.query.correspondence import Correspondence
from verity_vllm.query.program_view import from_instances
from verity_vllm.query.v1_bridge import request_manifest

IMPL = ("model.layers.0.mlp.down_proj",)          # an observed implementation path: the Program's wrapper prefix is `model.`


def _v2_manifest(inst: dict) -> dict:
    P = from_instances(inst)
    res = {"program": {"digest": "p" * 64, "function": "Derived[vLLMRequest[Test]]"}, "target": {"declared": {"compute_capability": [8, 9]}}}
    return request_manifest(P, Correspondence(P, implementation_paths=IMPL), res, request_id="r0").manifest


def _address(pi: SR.ProgramIndex, r: dict) -> tuple | None:
    d = pi.describe(r)
    if d is None:
        return None
    return (d["op_path"], tuple(tuple(m) for m in d["members"]), d["row"] if d["members"] else None, d["step"])


@pytest.mark.parametrize("variant", [{}, {"bias": True}, {"rope": True}])        # (`attn`: its attention output is a leaf no Call reads)
def test_v2_rule_addresses_every_row_as_v1_where_the_populations_agree(variant):
    inst, _woff = _program(**variant)
    man = _v2_manifest(inst)
    assert man["complete"], man["unmodelled"]
    v1 = SR.ProgramIndex(inst, rule="v1")
    v2 = SR.ProgramIndex(inst, rule=SR.address_rule_of(man))
    assert v2.rule == SR.V2_QUERY and v1.prefix == v2.prefix
    diffs = [(r["i"], r["spec"], _address(v1, r), _address(v2, r)) for r in inst["rows"] if _address(v1, r) != _address(v2, r)]
    assert not diffs, diffs[:5]


def test_v2_rule_addresses_the_crossing_interior_row_v1_calls_interior():
    """F-r19-int-20 (#57's shape): the root body's `embed * normalizer` is interior by the v1 vocabulary (members_for -> []) but crosses into
    layer 0 -- the v2 query requires it as (model, out) and the v2 rule addresses exactly those rows; every other row is addressed as v1."""
    inst, _woff = _program(scale=True)
    man = _v2_manifest(inst)
    v1 = SR.ProgramIndex(inst, rule="v1")
    v2 = SR.ProgramIndex(inst, rule=SR.V2_QUERY)
    diffs = {r["i"]: (r["spec"], _address(v1, r), _address(v2, r)) for r in inst["rows"] if _address(v1, r) != _address(v2, r)}
    assert diffs and all(spec.startswith("Bf16MulScalarTensor_v1{") for spec, _a, _b in diffs.values()), diffs
    for _spec, a1, a2 in diffs.values():
        assert a1[1] == () and a2[0] == "model" and [m[0] for m in a2[1]] == ["out"]
    rows = [x for x in man["identities"] if x["op_path"] == "model" and x["output_member"] == "out"]
    assert sum(int(x["rows"]) for x in rows) == len(diffs)


@pytest.mark.parametrize("variant", [{}, {"bias": True}, {"attn": True}, {"scale": True}])
def test_v2_rule_reconciles_with_the_v2_manifest_identity_by_identity(variant):
    """Every identity of the v2 manifest with Program rows is produced by the v2-addressed rows, with the manifest's row count, and no
    addressed row is outside the manifest (`query_population`'s fail-closed checks)."""
    inst, _woff = _program(**variant)
    man = _v2_manifest(inst)
    pi = SR.ProgramIndex(inst, rule=SR.address_rule_of(man))
    addresses: dict = {}
    SR.population(pi, "r0", addresses=addresses)
    q = SR.query_population(man, addresses, set(), ranks=[0])
    checks = q["query_checks"]
    for k in ("vus_outside_query", "identities_without_rows"):
        assert checks[k]["n"] == 0, (k, checks[k])
    bad = {k: v for k, v in checks.items() if isinstance(v, dict) and v.get("n") and k not in ("vus_outside_query", "identities_without_rows")}
    assert not bad, bad


def test_address_rule_is_the_manifests_engine():
    assert SR.address_rule_of(None) == "v1"
    assert SR.address_rule_of({"query": {"engine": "v2-query"}}) == SR.V2_QUERY
    assert SR.address_rule_of({"query": {"query_id": "Q_module_v1"}}) == "v1"
    with pytest.raises(ValueError, match="unknown address rule"):
        SR.ProgramIndex({"rows": [], "params": []}, rule="v3")


@pytest.mark.parametrize("rank", [0, 1])
def test_v2_rule_addresses_a_tp_rank_program_as_v1_and_its_rank_manifest_has_v1s_identities(rank):
    """TP-04 on the v2 query: a wired rank Program's v2 manifest carries v1's identities -- the rank-local partials (`part_rank<r>` /
    `shard_rank<r>`) beside every collective output -- and the v2 rule addresses every row as the v1 rule does with the rank's TP context."""
    import copy

    from verity_capture.commit import required_manifest as RM
    from verity_capture.commit.tests.test_required_values import _tp_df
    from verity_vllm.query.compare import compare, summarize
    prog = _tp_df(rank)
    inst, wl = prog["instances"], {"sweep": {"row_id": "tp2-df", "batch": 1, "arrive_steps": [0]}}
    # the row-parallel Gemm reads its own weight shard (the fixture wires the embedding's read only): the all-reduce takes the Linear's slot "0"
    H, V, wp = 32, 16, len(inst["params"]) - 1
    for r in inst["rows"]:
        if r["spec"][:8] == "Gemm_v1{" and "mlp.down_proj/" in r["name"]:
            r["args"].append([["p", wp, V * H, V * H + H * (H // 2)]])
    P = from_instances(inst)
    man = request_manifest(P, Correspondence(P, implementation_paths=IMPL), prog["result"], request_id="r0", workload=wl, rank=rank, ranks=2).manifest
    assert man["complete"], man["unmodelled"]
    assert {x["output_member"] for x in man["identities"] if x["family"] == "tp_rank_partials"} == {f"part_rank{rank}", f"shard_rank{rank}"}
    v1m = RM.build_manifest(copy.deepcopy(prog), wl, request_id="r0", rank=rank, ranks=2)
    s = summarize(compare(v1m["identities"], man["identities"], ignore_program_digest=True))
    assert s["only_v1_n"] == s["only_v2_n"] == s["multiplicity_diff_n"] == 0, s
    tp = {"rank": rank, "world": 2}
    v1 = SR.ProgramIndex(inst, tp=tp, rule="v1", attention_paths=list(IMPL))
    v2 = SR.ProgramIndex(inst, tp=tp, rule=SR.address_rule_of(man), attention_paths=list(IMPL))
    assert v2.rule == SR.V2_QUERY
    diffs = [(r["i"], r["spec"], _address(v1, r), _address(v2, r)) for r in inst["rows"] if _address(v1, r) != _address(v2, r)]
    assert not diffs, diffs[:5]
