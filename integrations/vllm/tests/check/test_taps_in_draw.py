"""A served call boundary (binding-map entry `source: tap`) is bound, never evaluated by the Commit: `coverage.taps_in_draw` requires its
producing Calls to be VUs of the replay draw's population, else `query_checks` fails by name (`taps_outside_draw`) and the C2 verdict
names it.  On the Gemma-2-style dense Program: every norm call boundary tapped is reached; one no row produces, one of another class and
those past the executed prefix are each named."""
from __future__ import annotations

from tests.acquire.call_boundary_programs import LP, PF, build_dir, dense_program
from verity_vllm.check.c2_rules import _query_population_scope
from verity_vllm.check.replay import coverage as COV
from verity_vllm.check.replay.driver import manifest_query
from verity_vllm.check.replay.index import ProgramIndex
from verity_vllm.check.replay.population import population


def _world(tmp_path):
    b, _at = dense_program()
    doc = b.doc(LP)
    pdir, man = build_dir(tmp_path, doc, [PF[len("model."):]])
    at: dict = {}
    population(ProgramIndex(doc, program_dir=pdir, query=manifest_query(man)), "r0", addresses=at)
    return man, at


def _bmap(man, tapped):
    """A map of every call boundary of `man`, `source: tap` where `tapped(identity)`."""
    return {"entries": [{"identity": {"request_id": r["request_id"], "step": r["engine_step"], "invocation": 0, "op_path": r["op_path"],
                                      "output_member": r["output_member"]}, "source": "tap" if tapped(r) else "host"}
                        for r in man["identities"] if r.get("family") == "call_boundaries"]}


def _norm(r):
    return "layernorm/" in r["op_path"]


def _check(man, at, bmap, executed_prefix=None):
    qp = COV.query_population(man, at, set(), ranks=[0], executed_prefix=executed_prefix)
    return COV.taps_in_draw(qp, bmap, at, executed_prefix, {"r0": 0}, man)


def test_every_tapped_norm_boundary_of_the_dense_program_is_in_the_draws_population(tmp_path):
    man, at = _world(tmp_path)
    qp = _check(man, at, _bmap(man, _norm))
    got = qp["query_checks"]["taps_outside_draw"]
    n = sum(1 for r in man["identities"] if r.get("family") == "call_boundaries" and _norm(r))
    assert (got["n"], got["tapped"], got["first"]) == (0, n, []) and qp["query_checks"]["ok"] is True, qp["query_checks"]
    assert f"served call boundaries (source tap) {n}, {n} with their producing Calls in the draw's population" in qp["scope"]
    assert _query_population_scope({"population": qp})[1] is None
    plain = _check(man, at, _bmap(man, lambda r: False))
    assert plain["query_checks"]["taps_outside_draw"]["tapped"] == 0 and "served call boundaries" not in plain["scope"]


def test_a_tapped_boundary_the_draw_cannot_reach_fails_the_check_and_the_verdict_names_it(tmp_path):
    man, at = _world(tmp_path)
    bmap = _bmap(man, _norm)
    bmap["entries"].append({"identity": {"request_id": "r0", "step": 0, "invocation": 0, "op_path": "model.layers.9.input_layernorm/add_0",
                                         "output_member": "out"}, "source": "tap"})
    other = next(k for k, a in at.items() if k[1] == 1 and "/mean_" in k[2])
    at[other] = dict(at[other], **{"class": "not-evaluable", "why": "stand-in"})
    qp = _check(man, at, bmap)
    got = qp["query_checks"]["taps_outside_draw"]
    assert got["n"] == 2 and qp["query_checks"]["ok"] is False
    assert {(f["op_path"], f["why"]) for f in got["first"]} == {
        ("model.layers.9.input_layernorm/add_0", "no Program row produces it"),
        (other[2], "its producing rows are class not-evaluable, not vu (stand-in)")}
    fail = _query_population_scope({"population": qp})[1]
    assert fail is not None and "taps_outside_draw x2" in fail


def test_tapped_boundaries_past_the_executed_prefix_are_outside_the_draw(tmp_path):
    man, at = _world(tmp_path)
    ep = {"r0": {"n_exec": 2, "cap": 3, "a_r": 1, "lag": 1, "faults": []}}
    qp = _check(man, at, _bmap(man, _norm), executed_prefix=ep)
    got = qp["query_checks"]["taps_outside_draw"]
    late = sum(1 for r in man["identities"] if r.get("family") == "call_boundaries" and _norm(r) and int(r["step"]) >= 2)
    assert late and got["n"] == late and qp["query_checks"]["ok"] is False
    assert all(f["step"] >= 2 and "past the executed prefix (n_exec 2)" in f["why"] for f in got["first"])
