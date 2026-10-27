"""The Commit's B>=2 population is placed over each request's EXECUTED PREFIX OF RECORD -- cap on a
length stop, served + LAG on an EOS stop under the DECLARED lag -- never over the Program's cap shape and never by a lag read from the trace.

The record that found it: OLMoE B2 i32/o32 mixedstop: r0 served 11 (EOS 50279 at output index 10) / executed 12 under LAG 1, r1 32 / 32;
GM-01 G6 PASS with the lag exercised; the Commit required r0's steps 12..31 (bound 13808/19988, composition
'engine step 12 ... missing [r0 x1] (+19 more steps)', linkage 44/64) and failed by name.  Torch-free (the offline committer of
test_binding_negatives)."""
from __future__ import annotations

import pytest

from tests.commit.test_binding_negatives import B8_CAP, B8_LP, PD, b8_engine_rows, b8_manifest, b8_stream, commit
from verity_vllm.commit import binding as B, executed_prefix as EP
from verity_vllm.check import executed_prefix as EPL
from verity_vllm.check.replay.linkage import linkage_horizon
from verity_vllm.records.global_match import SCHEMA as GM_SCHEMA

EOS = 50279
#: the identity coverage `commit.binding.coverage_check` writes, which a binding's execution extent sits beside
COVERED = {"ok": True, "expected_n": 64, "bound_n": 64, "missing_n": 0, "ambiguous_n": 0}


def _served(lens: dict[str, int], eos_last: set[str] = frozenset()) -> dict[str, list[int]]:
    return {r: ([7] * (n - 1) + [EOS]) if r in eos_last else [7] * n for r, n in lens.items()}


def test_the_mixedstop_record_two_lengths_per_request():
    """The first real record's numbers: r0 served 11 with a declared EOS last, cap 32, LAG 1 -> executed 12 (lag row = step 11); r1 32/32."""
    caps = {"r0": 32, "r1": 32}
    e = EP.executed_prefix_of_record(["r0", "r1"], _served({"r0": 11, "r1": 32}, {"r0"}), caps=caps, eos_of_record=[EOS], lag=1)
    assert e["ok"] and e["any_eos_stop"] and e["problems"] == []
    r0, r1 = e["requests"]["r0"], e["requests"]["r1"]
    assert (r0["served_output_len"], r0["executed_prefix_len"], r0["cap_exhausted"], r0["eos_stop"]) == (11, 12, False, True)
    assert r0["eos_step"] == 10 and r0["lag_steps"] == [11] and r0["last_served_token"] == EOS
    assert (r1["served_output_len"], r1["executed_prefix_len"], r1["cap_exhausted"], r1["eos_stop"]) == (32, 32, True, False)
    assert EP.request_steps_of_record(e, {"r0": 0, "r1": 0}) == {"r0": list(range(12)), "r1": list(range(32))}
    # the same stop under a DECLARED LAG 0: executed == served (no placeholder row); the lag is the declaration, not the trace
    e0 = EP.executed_prefix_of_record(["r0", "r1"], _served({"r0": 11, "r1": 32}, {"r0"}), caps=caps, eos_of_record=[EOS], lag=0)
    assert e0["ok"] and e0["requests"]["r0"]["executed_prefix_len"] == 11 and e0["requests"]["r0"]["lag_steps"] == []
    # an EOS served exactly at the cap is a length stop for the scheduler: executed == cap, no lag row
    ec = EP.executed_prefix_of_record(["r0"], _served({"r0": 32}, {"r0"}), caps={"r0": 32}, eos_of_record=[EOS], lag=1)
    assert ec["requests"]["r0"]["executed_prefix_len"] == 32 and ec["requests"]["r0"]["cap_exhausted"] is True and not ec["any_eos_stop"]


def test_an_early_stop_the_declared_rule_cannot_place_is_refused_by_name_and_keeps_the_cap():
    caps = {"r0": 32, "r1": 32}
    served = _served({"r0": 11, "r1": 32}, {"r0"})
    # no declared LAG (undeclared Program / unattested engine): refused, never default 0 or 1; the population stays the cap's
    e = EP.executed_prefix_of_record(["r0", "r1"], served, caps=caps, eos_of_record=[EOS], lag=None)
    assert not e["ok"] and e["requests"]["r0"]["executed_prefix_len"] == 32 and "declares no LAG" in e["problems"][0] and "never default" in e["problems"][0]
    # no declared stop rule
    e = EP.executed_prefix_of_record(["r0", "r1"], served, caps=caps, eos_of_record=None, lag=1)
    assert not e["ok"] and "declares no stop rule" in e["problems"][0] and "fail closed" in e["problems"][0]
    # the last served token is not a declared EOS id (a request that stopped for no declared reason)
    e = EP.executed_prefix_of_record(["r0", "r1"], _served({"r0": 11, "r1": 32}), caps=caps, eos_of_record=[EOS], lag=1)
    assert not e["ok"] and "is not a declared EOS id" in e["problems"][0] and e["requests"]["r0"]["executed_prefix_len"] == 32
    # a LAG outside {0, 1}; served over the cap; no driver tokens for a request
    assert not EP.executed_prefix_of_record(["r0"], _served({"r0": 11}, {"r0"}), caps={"r0": 32}, eos_of_record=[EOS], lag=2)["ok"]
    assert "over the declared cap" in EP.executed_prefix_of_record(["r0"], _served({"r0": 33}), caps={"r0": 32}, eos_of_record=[EOS], lag=1)["problems"][0]
    e = EP.executed_prefix_of_record(["r0", "r1"], {"r1": [7] * 32}, caps=caps, eos_of_record=[EOS], lag=1)
    assert not e["ok"] and "no served tokens" in e["problems"][0]
    # no driver tokens at all (an older caller): every prefix is the cap, nothing refused, served_known False
    e = EP.executed_prefix_of_record(["r0", "r1"], None, caps=caps, eos_of_record=[EOS], lag=1)
    assert e["ok"] and not e["served_known"] and {k: v["executed_prefix_len"] for k, v in e["requests"].items()} == caps


def test_served_past_the_first_eos_fails_by_name_f_r17_14():
    """The Program's rule stops at the FIRST declared EOS.  Served [EOS, x, EOS] (cap 5, LAG 1) is NOT an EOS stop
    at 3 with executed 4 -- it was served past an EOS at index 0 (Continuation_v1 stops there: served [EOS], executed [1, 2]).  FAIL by name;
    the prefix stays the cap (nothing accepted).  The same for an EOS mid-list on a length stop ([x, EOS, y] at the cap) and for an
    EOS-then-more under LAG 0.  An EOS ONLY at the last served index is the EOS stop it always was."""
    e = EP.executed_prefix_of_record(["r0"], {"r0": [EOS, 3, EOS]}, caps={"r0": 5}, eos_of_record=[EOS], lag=1)
    assert not e["ok"] and not e["any_eos_stop"], e
    r0 = e["requests"]["r0"]
    assert r0["first_eos_index"] == 0 and r0["executed_prefix_len"] == 5 and r0["eos_stop"] is None and "lag_steps" not in r0
    assert e["problems"] == [f"r0: {r0['why']}"] and r0["why"].startswith("served past an EOS at index 0: served token 50279 is a declared EOS id but 2 more token(s) were served after it (served 3, cap 5)")
    assert "stops at the FIRST EOS -- served_output_len would be 1" in r0["why"]
    # an EOS in the middle of a list that reached the cap: also served past the stop (the cap does not launder it)
    ec = EP.executed_prefix_of_record(["r0"], {"r0": [7, EOS, 9]}, caps={"r0": 3}, eos_of_record=[EOS], lag=1)
    assert not ec["ok"] and "served past an EOS at index 1" in ec["problems"][0] and ec["requests"]["r0"]["cap_exhausted"] is None
    # LAG 0: the same rule (the lag is not what is being checked here)
    e0 = EP.executed_prefix_of_record(["r0"], {"r0": [EOS, 3]}, caps={"r0": 5}, eos_of_record=[EOS], lag=0)
    assert not e0["ok"] and "served past an EOS at index 0" in e0["problems"][0] and "1 more token(s)" in e0["problems"][0]
    # the positive it must not touch: EOS at the last served index only
    ok = EP.executed_prefix_of_record(["r0"], {"r0": [7, 3, EOS]}, caps={"r0": 5}, eos_of_record=[EOS], lag=1)
    assert ok["ok"] and ok["requests"]["r0"]["first_eos_index"] == 2 and ok["requests"]["r0"]["executed_prefix_len"] == 4 and ok["requests"]["r0"]["lag_steps"] == [3]
    # no declared EOS set: nothing to find (the length-stop request is fine; the early stop fails for its own reason)
    nn = EP.executed_prefix_of_record(["r0"], {"r0": [7, 3, 9]}, caps={"r0": 3}, eos_of_record=None, lag=1)
    assert nn["ok"] and nn["requests"]["r0"]["first_eos_index"] is None


def _mixed_b8(tmp_path, executed: dict[str, int], committed: dict[str, int] | None = None):
    """The B8 fixture as a mixed-stop record: the manifest declares the Program's CAP shape (every request active at [0, cap)); the engine
    committed request k over [0, committed_k) (default = executed_k) -- the rows past the stop exist only when the test plants them."""
    committed = committed or executed
    declared = {e: [(f"r{k}", B8_LP[k] if e == 0 else 1) for k in range(8) if e < B8_CAP[k]] for e in range(max(B8_CAP))}
    ran = {e: [(f"r{k}", B8_LP[k] if e == 0 else 1) for k in range(8) if e < committed[f"r{k}"]] for e in range(max(B8_CAP))}
    ran = {e: m for e, m in ran.items() if m}
    rows = b8_engine_rows(ran)
    _c, _run, bmap = commit(b8_stream(rows), request_rows=rows)
    caps = {f"r{k}": B8_CAP[k] for k in range(8)}
    arrivals = {f"r{k}": 0 for k in range(8)}
    epr = EP.executed_prefix_of_record(sorted(caps), _served(executed, {r for r, n in executed.items() if n < caps[r]}), caps=caps, eos_of_record=[EOS], lag=0)
    assert epr["ok"], epr["problems"]
    rsteps = EP.request_steps_of_record(epr, arrivals)
    man = b8_manifest(tmp_path, declared)
    return bmap, man, caps, arrivals, rsteps, epr


def test_b8_mixed_stop_population_over_the_executed_prefix_passes_and_the_cap_shape_fails_by_name(tmp_path):
    """r1 declared cap 4 stops by EOS after 2 served (LAG 0 -> executed 2), r3 cap 5 after 3; the others run to their caps."""
    executed = {f"r{k}": B8_CAP[k] for k in range(8)} | {"r1": 2, "r3": 3}
    bmap, man, caps, arrivals, rsteps, epr = _mixed_b8(tmp_path, executed)
    # the older placement (every request over its cap): missing = r1's 2 + r3's 2 unexecuted steps, composition fails at step 2
    pop_cap, hdr_cap = B.load_required_manifest(man, PD, requests=sorted(caps), steps=max(B8_CAP), arrivals=arrivals)
    cov_cap = B.coverage_check(pop_cap, bmap)
    comp_cap = B.composition_check(hdr_cap, bmap)
    assert not cov_cap["ok"] and cov_cap["missing_n"] > 0 and {m["request_id"] for m in cov_cap["missing"]} <= {"r1", "r3"}
    assert not comp_cap["ok"] and comp_cap["problems"][0]["engine_step"] == 2 and "r1 x1" in comp_cap["problems"][0]["missing"]
    # the executed-prefix placement: the requirement is each request's executed prefix; the manifest stays the Program's (cap shape)
    pop, hdr = B.load_required_manifest(man, PD, requests=sorted(caps), steps=max(B8_CAP), arrivals=arrivals, request_steps=rsteps)
    assert hdr["programmed_not_executed_n"] == len(pop_cap) - len(pop) > 0 and hdr["request_steps_of_record"]["r1"] == [0, 1]
    cov = B.coverage_check(pop, bmap)
    comp = B.composition_check(hdr, bmap, request_steps=rsteps)
    assert cov["ok"] and cov["missing_n"] == 0 and cov["extra_n"] == 0, cov["missing"][:3]
    assert comp["ok"] and comp["executed_prefix_restriction"]["members_dropped"] == 4 and comp["steps_not_executed"] == 0, comp
    assert EP.rows_past_the_stop(bmap, rsteps)["ok"]
    # every step still has its composition checked (no step was dropped: r2's cap 1 already ended at step 1 in the declaration)
    assert comp["checked_steps"] == max(B8_CAP)
    # the classification is on the record by family / request / kind, and the manifest of record is untouched (rows_n, digest)
    ne = hdr["not_executed_by_rule"]
    assert ne["n"] == hdr["programmed_not_executed_n"] and sum(ne["by_family"].values()) == ne["n"] and set(ne["by_request"]) == {"r1", "r3"}
    assert ne["by_request"]["r1"] == ne["by_request"]["r3"] and sum(ne["by_request"].values()) == ne["n"]     # 2 unexecuted steps each, every family per step
    assert set(ne["kind_by_family"].values()) <= {EP.KIND_ROW, EP.KIND_BATCH, EP.KIND_INPUT} and ne["rule"] == EP.RULE
    assert hdr["rows_n"] == hdr_cap["rows_n"] and hdr["manifest_digest"] == hdr_cap["manifest_digest"]


def _population_block(epr: dict, by_family: dict[str, int], *, applied: bool = True, n_exec: dict[str, int] | None = None) -> dict:
    """population.not_executed_by_rule as sampled_replay writes it (the keys the reconciliation reads)."""
    reqs = {rid: {"n_exec": (n_exec or {}).get(rid, r["executed_prefix_len"])} for rid, r in epr["requests"].items()}
    return {"applied": applied, "by_request": reqs, "totals": {"by_family": {f: {"N_manifest": n + 1, "N_required": 1, "N_excluded": n} for f, n in by_family.items()}}}


def test_the_commits_classification_and_the_populations_third_bucket_are_reconciled_identity_count_by_identity_count(tmp_path):
    """One mechanism on the record: n_exec per request equal, the count per ROW family equal; a batch-level
    or named-input family (outside the population split's reach) is named, never a FAIL and never read as equal."""
    executed = {f"r{k}": B8_CAP[k] for k in range(8)} | {"r1": 2, "r3": 3}
    bmap, man, caps, arrivals, rsteps, epr = _mixed_b8(tmp_path, executed)
    _pop, hdr = B.load_required_manifest(man, PD, requests=sorted(caps), steps=max(B8_CAP), arrivals=arrivals, request_steps=rsteps)
    cc = hdr["not_executed_by_rule"]
    row_fams = {f: n for f, n in cc["by_family"].items() if cc["kind_by_family"][f] == EP.KIND_ROW}
    assert row_fams, cc
    # AGREE: the population excluded exactly the Commit's row-family counts with the same n_exec
    rec = EP.reconcile_not_executed(cc, _population_block(epr, row_fams), epr)
    assert rec["ok"] and not rec["problems"] and all(v["equal"] for v in rec["n_exec"].values()) and all(v["equal"] for v in rec["by_family"].values())
    assert rec["n_exec"]["r1"] == {"commit": 2, "population": 2, "equal": True}
    # DISAGREE by count on a row family: FAIL by name (the identities would leave twice / the invariants contradict)
    fam = next(iter(row_fams))
    rec = EP.reconcile_not_executed(cc, _population_block(epr, dict(row_fams, **{fam: row_fams[fam] - 1})), epr)
    assert not rec["ok"] and fam in rec["problems"][0] and "the population excluded" in rec["problems"][0]
    # DISAGREE on n_exec (two prefixes on one record): FAIL by name
    rec = EP.reconcile_not_executed(cc, _population_block(epr, row_fams, n_exec={"r1": 3}), epr)
    assert not rec["ok"] and "r1" in rec["problems"][0] and "two prefixes" in rec["problems"][0]
    # a named-input / batch-level family the population split cannot see: recorded by name under outside_population_reach, ok unaffected
    cc2 = {**cc, "by_family": dict(cc["by_family"], input_token_ids=20, fa2_hidden_m1_stream=320),
           "kind_by_family": dict(cc["kind_by_family"], input_token_ids=EP.KIND_INPUT, fa2_hidden_m1_stream=EP.KIND_BATCH)}
    rec = EP.reconcile_not_executed(cc2, _population_block(epr, dict(row_fams, input_token_ids=0, fa2_hidden_m1_stream=0)), epr)
    assert rec["ok"] and set(rec["outside_population_reach"]) == {"input_token_ids", "fa2_hidden_m1_stream"}
    o = rec["outside_population_reach"]["input_token_ids"]
    assert (o["kind"], o["commit"], o["population_N_excluded"], o["population_reaches"]) == (EP.KIND_INPUT, 20, 0, False) and "does not reach" in o["pending"]
    assert "input_token_ids" not in rec["by_family"]
    # the population REACHES both kinds: kind_by_family names the named-input family and
    # N_excluded counts it like a row; the batch-level family states its (plane, request) cells -> both reconcile in by_family, nothing
    # outside the population's reach, and a disagreement on either FAILS by name
    pb = _population_block(epr, dict(row_fams, input_token_ids=20, fa2_hidden_m1_stream=0))
    pb["kind_by_family"] = {**{f: "row" for f in row_fams}, "input_token_ids": "named-input", "fa2_hidden_m1_stream": "batch-level"}
    pb["totals"]["by_family"]["fa2_hidden_m1_stream"]["plane_cells_not_executed"] = {"identities": 0, "cells": 320}
    pb["batch_level"] = {"fa2_hidden_m1_stream": {"identities": 512, "identities_not_executed": 0, "identities_partially_executed": 320,
                                                  "request_slots": 1024, "request_slots_not_executed": 320, "cells_not_executed": 320}}
    rec = EP.reconcile_not_executed(cc2, pb, epr)
    assert rec["ok"] and rec["total"] and rec["outside_population_reach"] == {} and rec["schema"].endswith("/v3")
    assert rec["by_family"]["input_token_ids"] == {"commit": 20, "population": 20, "equal": True, "kind": EP.KIND_INPUT, "unit": "identities"}
    assert rec["by_family"]["fa2_hidden_m1_stream"] == {"commit": 320, "population": 320, "equal": True, "kind": EP.KIND_BATCH, "unit": "cells", "population_N_excluded": 0}
    rec = EP.reconcile_not_executed(cc2, {**pb, "totals": {"by_family": dict(pb["totals"]["by_family"], input_token_ids={"N_excluded": 19})}}, epr)
    assert not rec["ok"] and "input_token_ids (named-input): the Commit classified 20 not-executed-by-rule identities, the population excluded 19" in rec["problems"][0]
    pb2 = {**pb, "totals": {"by_family": dict(pb["totals"]["by_family"], fa2_hidden_m1_stream={"N_excluded": 0, "plane_cells_not_executed": {"identities": 0, "cells": 319}})}}
    rec = EP.reconcile_not_executed(cc2, pb2, epr)
    assert not rec["ok"] and "fa2_hidden_m1_stream (batch-level): the Commit classified 320 not-executed-by-rule cells, the population excluded 319" in rec["problems"][0]
    # the cells statement may also be read from batch_level[fam].cells_not_executed alone (8eeff13eb before d13ea5153)
    pb3 = {**pb, "totals": {"by_family": dict(pb["totals"]["by_family"], fa2_hidden_m1_stream={"N_excluded": 0})}}
    rec = EP.reconcile_not_executed(cc2, pb3, epr)
    assert rec["ok"] and rec["by_family"]["fa2_hidden_m1_stream"]["population"] == 320 and rec["outside_population_reach"] == {}
    # with the manifest's engine-step-annotated identities the two families are checked ARITHMETICALLY from the
    # population's own n_exec: r1 executed 2 of cap 4 -> its named input at steps 2,3 = 2 cells; a plane row listing (r0, r1) at steps 2,3
    # = 2 (request, plane) cells of r1 while the plane identity itself stays required (r0 executes there): N_excluded 0 is right
    ids = ([{"request_id": "r1", "step": s, "engine_step": s, "family": "input_token_ids", "op_path": "prompt"} for s in range(4)]
           + [{"request_id": "*", "requests": ["r0", "r1"], "engine_step": s, "family": "fa2_hidden_m1_stream", "op_path": "h"} for s in range(4)])
    exp = EP.expected_cells_from_population(ids, _population_block(epr, row_fams))
    n_exec = {k: v["executed_prefix_len"] for k, v in epr["requests"].items()}
    plane_cells = sum(1 for s in range(4) for q in ("r0", "r1") if s >= n_exec[q])                     # r1: steps 2,3; r0: any step >= its own n_exec
    assert exp == {"input_token_ids": {"cells": 2, "rows": 4}, "fa2_hidden_m1_stream": {"cells": plane_cells, "rows": 4}} and plane_cells >= 2
    cc3 = {**cc, "by_family": dict(cc["by_family"], input_token_ids=2, fa2_hidden_m1_stream=plane_cells),
           "kind_by_family": dict(cc["kind_by_family"], input_token_ids=EP.KIND_INPUT, fa2_hidden_m1_stream=EP.KIND_BATCH)}
    rec = EP.reconcile_not_executed(cc3, _population_block(epr, dict(row_fams, input_token_ids=0, fa2_hidden_m1_stream=0)), epr, manifest_identities=ids)
    assert rec["ok"] and rec["schema"].endswith("/v3")
    o = rec["outside_population_reach"]
    assert o["input_token_ids"]["expected_from_population_facts"] == 2 and o["input_token_ids"]["equal"] and o["input_token_ids"]["population_reaches"] is False \
        and "does not reach" in o["input_token_ids"]["pending"] and "input_token_ids" not in rec["by_family"] and rec["total"] is True
    assert o["fa2_hidden_m1_stream"]["expected_from_population_facts"] == plane_cells and o["fa2_hidden_m1_stream"]["equal"] and o["fa2_hidden_m1_stream"]["population_N_excluded"] == 0
    # the Commit's cell count off by one against the population's own n_exec: FAIL by name (both kinds)
    cc4 = {**cc3, "by_family": dict(cc3["by_family"], fa2_hidden_m1_stream=plane_cells + 1)}
    rec = EP.reconcile_not_executed(cc4, _population_block(epr, dict(row_fams, input_token_ids=0, fa2_hidden_m1_stream=0)), epr, manifest_identities=ids)
    assert not rec["ok"] and "fa2_hidden_m1_stream (batch-level)" in rec["problems"][0] and f"implies {plane_cells}" in rec["problems"][0]
    cc5 = {**cc3, "by_family": dict(cc3["by_family"], input_token_ids=1)}
    rec = EP.reconcile_not_executed(cc5, _population_block(epr, dict(row_fams, input_token_ids=0, fa2_hidden_m1_stream=0)), epr, manifest_identities=ids)
    assert not rec["ok"] and "input_token_ids (named-input)" in rec["problems"][0]
    # once the population split reaches named inputs (N_excluded == the cells), the family is compared like a row family too
    rec = EP.reconcile_not_executed(cc3, _population_block(epr, dict(row_fams, input_token_ids=2, fa2_hidden_m1_stream=0)), epr, manifest_identities=ids)
    assert rec["ok"] and "input_token_ids" not in rec["outside_population_reach"] \
        and rec["by_family"]["input_token_ids"] == {"commit": 2, "population": 2, "equal": True, "kind": EP.KIND_INPUT, "unit": "identities", "expected_from_population_facts": 2}
    # a population statement of the plane cells that disagrees with its own n_exec: FAIL by name (the Commit agreeing with the statement does not save it)
    pb = _population_block(epr, dict(row_fams, input_token_ids=0, fa2_hidden_m1_stream=0))
    pb["totals"]["by_family"]["fa2_hidden_m1_stream"]["plane_cells_not_executed"] = {"identities": 0, "cells": 5}
    rec = EP.reconcile_not_executed({**cc3, "by_family": dict(cc3["by_family"], fa2_hidden_m1_stream=5)}, pb, epr, manifest_identities=ids)
    assert not rec["ok"] and f"states 5 not-executed cells, its own n_exec implies {plane_cells}" in rec["problems"][0]
    # the population's cells agreeing with its own facts and with the Commit: reconciled in by_family with the arithmetic beside it
    pb["totals"]["by_family"]["fa2_hidden_m1_stream"]["plane_cells_not_executed"] = {"identities": 0, "cells": plane_cells}
    rec = EP.reconcile_not_executed(cc3, pb, epr, manifest_identities=ids)
    assert rec["ok"] and rec["by_family"]["fa2_hidden_m1_stream"]["expected_from_population_facts"] == plane_cells and "fa2_hidden_m1_stream" not in rec["outside_population_reach"]
    # no population block at all (a tree without the population's mechanism) / not applied on an EOS-stopped record: FAIL by name
    assert not EP.reconcile_not_executed(cc, None, epr)["ok"]
    rec = EP.reconcile_not_executed(cc, _population_block(epr, row_fams, applied=False), epr)
    assert not rec["ok"] and "not applied" in rec["problems"][0]
    # the layouts-only path fills the same classification through restrict_manifest_identities(stats=)
    ids = [{"request_id": "r1", "step": 3, "family": "instance_outputs", "op_path": "m", "output_member": "out"},
           {"request_id": "*", "requests": ["r0", "r1"], "step": 3, "family": "fa2_hidden_m1_stream", "op_path": "h"},
           {"request_id": "r1", "step": 3, "family": "input_token_ids", "op_path": "prompt"},
           {"request_id": "r0", "step": 3, "family": "instance_outputs", "op_path": "m", "output_member": "out"}]
    st = EP.new_classification()
    kept, n = EP.restrict_manifest_identities(ids, {"r0": range(0, 8), "r1": range(0, 2)}, stats=st)
    assert n == 2 and len(kept) == 2 and kept[0]["requests"] == ["r0"]
    assert st["by_family"] == {"instance_outputs": 1, "fa2_hidden_m1_stream": 1, "input_token_ids": 1} and st["by_request"] == {"r1": 3}
    assert st["kind_by_family"] == {"instance_outputs": EP.KIND_ROW, "fa2_hidden_m1_stream": EP.KIND_BATCH, "input_token_ids": EP.KIND_INPUT}


def test_b8_rows_committed_past_the_stop_fail_by_name(tmp_path):
    """The engine committed r1 for 4 steps (its cap) while the driver served 2 = execution past the stop: the identity population is
    covered (extras are allowed there), so the NAMED checks are composition `unexpected` and rows_past_the_stop."""
    executed = {f"r{k}": B8_CAP[k] for k in range(8)} | {"r1": 2}
    bmap, man, caps, arrivals, rsteps, epr = _mixed_b8(tmp_path, executed, committed={f"r{k}": B8_CAP[k] for k in range(8)})
    pop, hdr = B.load_required_manifest(man, PD, requests=sorted(caps), steps=max(B8_CAP), arrivals=arrivals, request_steps=rsteps)
    assert B.coverage_check(pop, bmap)["ok"]                                       # the executed rows are all there ...
    comp = B.composition_check(hdr, bmap, request_steps=rsteps)
    assert not comp["ok"] and comp["problems"][0]["engine_step"] == 2 and comp["problems"][0]["unexpected"] == ["r1 x1"], comp["problems"][0]
    past = EP.rows_past_the_stop(bmap, rsteps)
    assert not past["ok"] and past["requests"] == {"r1": [2, 3]} and "execution past the stop" in past["error"] and "r1" in past["error"]
    # a request that committed FEWER rows than its executed prefix is still missing (unchanged rule)
    bmap2, man2, caps2, arr2, rsteps2, _ = _mixed_b8(tmp_path, {f"r{k}": B8_CAP[k] for k in range(8)} | {"r1": 3}, committed={f"r{k}": B8_CAP[k] for k in range(8)} | {"r1": 2})
    pop2, hdr2 = B.load_required_manifest(man2, PD, requests=sorted(caps2), steps=max(B8_CAP), arrivals=arr2, request_steps=rsteps2)
    cov2 = B.coverage_check(pop2, bmap2)
    assert not cov2["ok"] and {(m["request_id"], m["step"]) for m in cov2["missing"]} == {("r1", 2)}


def test_restrictions_of_the_flat_manifest_and_the_linkage_horizon():
    rsteps = {"r0": list(range(12)), "r1": list(range(32))}
    ids = ([{"family": "instance_outputs", "request_id": "r0", "step": s, "engine_step": s, "op_path": "x", "rank": 0} for s in range(32)]
           + [{"family": "instance_outputs", "request_id": "r1", "step": s, "engine_step": s, "op_path": "x", "rank": 0} for s in range(32)]
           + [{"family": "fa2_hidden_m1_stream", "request_id": "*", "requests": ["r0", "r1"], "step": "*", "engine_step": e, "op_path": "fa2.m1/L0", "rank": 0} for e in range(32)]
           + [{"family": "weights", "request_id": "*", "step": "*", "op_path": "weights", "rank": 0}])
    out, dropped = EP.restrict_manifest_identities(ids, rsteps)
    assert dropped == 20 and len(out) == len(ids) - 20
    planes = [x for x in out if x["request_id"] == "*" and x["family"] != "weights"]
    assert len(planes) == 32 and [x["requests"] for x in planes if x["engine_step"] >= 12] == [["r1"]] * 20 and planes[0]["requests"] == ["r0", "r1"]
    es = [{"engine_step": e, "members": [{"request_id": "r0", "step": e, "rows": 1}, {"request_id": "r1", "step": e, "rows": 1}], "rows": 2, "n_seq": 2} for e in range(32)]
    res, info = EP.restrict_engine_steps(es, rsteps)
    assert info == {"members_dropped": 20, "steps_dropped": 0, "steps_changed": 20} and res[12] == {"engine_step": 12, "members": [{"request_id": "r1", "step": 12, "rows": 1}], "rows": 1, "n_seq": 1}
    assert res[11] is es[11]
    # both stopped -> the tail steps have no executing request: dropped (the engine ran no such step)
    res2, info2 = EP.restrict_engine_steps(es, {"r0": list(range(12)), "r1": list(range(20))})
    assert len(res2) == 20 and info2["steps_dropped"] == 12
    # the linkage horizon at B>=2: the Program of record has the cap's steps (GP-01 composed); the executed prefix bounds it
    assert linkage_horizon(0, 32, 32, executed=12) == list(range(12)) and linkage_horizon(0, 32, 32) == list(range(32))
    assert linkage_horizon(3, 32, 12, executed=12) == list(range(3, 15)) and linkage_horizon(0, 32, 32, executed=40) == list(range(32))


def test_a_missing_match_account_leg_is_named_on_the_facts_and_faults_first(tmp_path):
    """#39 (qwen25-15b i4096, l40s4): match/capture/ was moved to /vault to free the disk while
    the Commit was queued; tokens.json went with it, a_r fell to the G6 leg and the class faulted 'a_r is not linked' / 'LAG rule not
    attested' after 47 min -- a missing INPUT read as a verdict.  The facts name the missing leg(s) (a_r_source suffix + account_missing),
    and the population's normaliser puts that fault FIRST, before (a) and (b), still fail-closed."""
    from verity_vllm.check.replay import population as POP
    e = _mixed_epr()
    # tokens.json absent, global_match.json present: exactly #39's record after the move
    acct = EPL.load_match_account(_match_account(tmp_path, write_tokens=False))
    assert acct["tokens"] is None and acct["global_match"] is not None
    link = EPL.link_to_match_account(e, acct, order=ORDER, declared_lag=1)
    binding = {"coverage": COVERED | {"match_account_link": link}}
    f = EP.executed_prefix_facts(e, binding, {"result": True}, {"r0": 0, "r1": 0}, {"eos": [EOS], "lag": 1, "declared_lag": 1})
    assert f["r0"]["account_missing"] == ["tokens.json"] and f["r1"]["account_missing"] == ["tokens.json"]
    assert "G6 lengths_by_request" in f["r0"]["a_r_source"] and f["r0"]["a_r_source"].endswith("Match account leg(s) NOT beside the Commit: tokens.json")
    assert f["r0"]["linked"] is False and f["r0"]["lag_attested"] is False
    # the consumer (the population's normaliser) accepts the fact and faults on it by name, first
    facts = EP.facts_accepted_by(POP.executed_prefix_of_record, f)
    assert "account_missing" in facts["r0"]
    rec = POP.executed_prefix_of_record("r0", **facts["r0"])
    assert rec["faults"][0].startswith("Match account leg(s) missing beside the Commit: tokens.json") and "not a verdict on the Program" in rec["faults"][0]
    assert any("a_r is not linked" in x for x in rec["faults"]) and any("LAG rule not attested" in x for x in rec["faults"])   # still fail-closed
    assert rec["account_missing"] == ["tokens.json"] and rec["mechanism"].startswith("not classified")
    # no Match record at all: both legs named; a complete record: none, and no such fault
    none = EP.executed_prefix_facts(e, {"coverage": COVERED | {"match_account_link": EPL.link_to_match_account(e, EPL.load_match_account(None), order=ORDER, declared_lag=1)}},
                                    {"result": True}, {"r0": 0}, {"eos": [EOS], "lag": 1, "declared_lag": 1})
    assert none["r0"]["account_missing"] == ["tokens.json", "global_match.json"]
    full = EPL.link_to_match_account(e, EPL.load_match_account(_match_account(tmp_path)), order=ORDER, declared_lag=1)
    ff = EP.executed_prefix_facts(e, {"coverage": COVERED | {"match_account_link": full}}, {"result": True}, {"r0": 0, "r1": 0}, {"eos": [EOS], "lag": 1, "declared_lag": 1})
    assert ff["r0"]["account_missing"] == [] and "NOT beside" not in ff["r0"]["a_r_source"]
    ok = POP.executed_prefix_of_record("r0", **EP.facts_accepted_by(POP.executed_prefix_of_record, ff)["r0"])
    assert ok["faults"] == [] and ok["account_missing"] == []


def test_the_commit_supplies_reqvals_not_executed_class_with_named_record_facts_only(tmp_path):
    """ONE mechanism for the class (sampled_replay.population.not_executed_by_rule): commit_delta is
    its CALLER -- _executed_prefix_facts derives every flag from a named record fact: a_r from the Match record's tokens.json (path + sha256
    in a_r_source), served_len_commit from this run, linked = Match-account link AND C2 boundary linkage, lag_attested from the link's legs,
    eos_declared from the stop rule, rows_past_stop from coverage.  With no Match record: a_r = the Commit's own count LABELLED so, linked False."""
    e = _mixed_epr()
    link = EPL.link_to_match_account(e, EPL.load_match_account(_match_account(tmp_path)), order=ORDER, declared_lag=1)
    assert link["ok"] and link["lag_attested"] is True and link["tokens_json"]["served_by_request"] == {"r0": 11, "r1": 32}
    binding = {"coverage": COVERED | {"match_account_link": link, "rows_past_the_stop": EP.rows_past_the_stop({"entries": []}, EP.request_steps_of_record(e, {"r0": 0, "r1": 0}))}}
    stop = {"eos": [EOS], "lag": 1, "declared_lag": 1}
    f = EP.executed_prefix_facts(e, binding, {"result": True}, {"r0": 0, "r1": 0}, stop)
    assert f["r0"] == {"a_r": 11, "a_r_source": f["r0"]["a_r_source"], "lag": 1, "cap": 32, "arrive": 0, "eos_declared": True, "served_len_commit": 11,
                       "linked": True, "lag_attested": True, "rows_past_stop": 0, "account_missing": []}
    assert f["r0"]["a_r_source"].startswith("match record ") and "tokens.json" in f["r0"]["a_r_source"] and "sha256" in f["r0"]["a_r_source"]
    assert f["r1"]["a_r"] == 32 and f["r1"]["eos_declared"] is False and f["r1"]["linked"] is True
    # the C2 boundary linkage failed -> linked False (the class faults by name in the population's normaliser)
    assert EP.executed_prefix_facts(e, binding, {"result": False}, {"r0": 0, "r1": 0}, stop)["r0"]["linked"] is False
    # committed rows of r0 past its prefix -> rows_past_stop counts them
    bm = {"entries": [{"kind": "leaf", "identity": {"request_id": "r0", "step": s}} for s in (12, 13, 14)]}
    binding2 = {"coverage": COVERED | {"match_account_link": link, "rows_past_the_stop": EP.rows_past_the_stop(bm, EP.request_steps_of_record(e, {"r0": 0, "r1": 0}))}}
    assert EP.executed_prefix_facts(e, binding2, {"result": True}, {"r0": 0, "r1": 0}, stop)["r0"]["rows_past_stop"] == 3
    # NO Match record beside the Commit: a_r is the Commit's own count, labelled, and linked is False; an undeclared LAG is -1 (out of domain)
    nolink = EPL.link_to_match_account(e, EPL.load_match_account(None), order=ORDER, declared_lag=1)
    f0 = EP.executed_prefix_facts(e, {"coverage": COVERED | {"match_account_link": nolink}}, {"result": True}, {"r0": 0}, {"eos": [EOS], "lag": None, "declared_lag": None})
    assert f0["r0"]["a_r"] == 11 and f0["r0"]["a_r_source"].startswith("the Commit's OWN served tokens") and f0["r0"]["linked"] is False
    assert f0["r0"]["lag_attested"] is False and f0["r0"]["lag"] == -1
    # with this run's served list the facts carry `served` + `eos_ids` (the recurrence's inputs); a consumer whose
    # signature predates them receives the arithmetic's facts only -- never a TypeError on an older tree, never silently dropped on a newer one
    served = _served({"r0": 11, "r1": 32}, {"r0"})
    fs = EP.executed_prefix_facts(e, binding, {"result": True}, {"r0": 0, "r1": 0}, stop, served_tokens=served)
    assert fs["r0"]["served"] == served["r0"] and fs["r0"]["eos_ids"] == [EOS] and fs["r1"]["served"] == [7] * 32
    assert {k: v for k, v in fs["r0"].items() if k not in ("served", "eos_ids")} == f["r0"]

    def older(rid, *, a_r, lag, cap, arrive=0, eos_declared=None, served_len_commit=None, linked=None, lag_attested=None, rows_past_stop=0, a_r_source=""):
        return {"rid": rid, "a_r": a_r}

    def newer(rid, *, a_r, lag, cap, arrive=0, eos_declared=None, served_len_commit=None, linked=None, lag_attested=None, rows_past_stop=0, a_r_source="",
              served=None, eos_ids=None, rule=None):
        return {"rid": rid, "served": served, "eos_ids": eos_ids}

    fo = EP.facts_accepted_by(older, fs)
    assert "served" not in fo["r0"] and "eos_ids" not in fo["r0"] and older("r0", **fo["r0"]) == {"rid": "r0", "a_r": 11}
    fn = EP.facts_accepted_by(newer, fs)
    assert newer("r0", **fn["r0"]) == {"rid": "r0", "served": served["r0"], "eos_ids": [EOS]}
    assert EP.facts_accepted_by(lambda rid, **kw: kw, fs) == fs                       # **kwargs: everything passes through


# ------------------------------------------------------------------ (a)+(b): the prefix is accepted only LINKED to the Match account
U0, U1 = "0f2b6d7e-r0", "5c1a9b3d-r1"
ORDER = {U0: "r0", U1: "r1"}


def _mixed_epr(lag: int = 1):
    return EP.executed_prefix_of_record(["r0", "r1"], _served({"r0": 11, "r1": 32}, {"r0"}), caps={"r0": 32, "r1": 32}, eos_of_record=[EOS], lag=lag)


def _match_account(tmp_path, *, served=None, tj_lag=1, gm_verdict="PASS", g6_lens=None, g6_lag=None, write_gm=True, write_tokens=True):
    """A row directory shaped like row_pod's: <row>/match/capture/tokens.json (m1_capture's fields) + <row>/global_match.json (GM-01's)."""
    import json
    served = served if served is not None else _served({"r0": 11, "r1": 32}, {"r0"})
    row = tmp_path / "row"
    (row / "match" / "capture").mkdir(parents=True, exist_ok=True)
    if write_tokens:
        tok = {"lag": {"value": tj_lag, "async_scheduling": None if tj_lag is None else bool(tj_lag)},
               "requests": [{"request_id": u, "generated_token_ids": served[r], "max_tokens": 32, "served_output_len": len(served[r]),
                             "executed_prefix_len": (len(served[r]) + tj_lag) if (len(served[r]) < 32 and tj_lag is not None) else (None if len(served[r]) < 32 else 32),
                             "cap_exhausted": len(served[r]) == 32} for u, r in ORDER.items()]}
        (row / "match" / "capture" / "tokens.json").write_text(json.dumps(tok))
    if write_gm:
        lens = g6_lens if g6_lens is not None else {"rec-r0": {"served_output_len": 11, "executed_prefix_len": 12, "cap": 32, "cap_exhausted": False},
                                                     "rec-r1": {"served_output_len": 32, "executed_prefix_len": 32, "cap": 32, "cap_exhausted": True}}
        lag = g6_lag if g6_lag is not None else {"declared": 1, "record_attests": 1, "tokens_json_lag": 1, "exercised": True, "variable_lag": False}
        gm = {"schema": GM_SCHEMA, "verdict": gm_verdict, "reasons": [], "failed_checks": [] if gm_verdict == "PASS" else ["G6"],
              "requests": [{"request_id": "rec-r0", "workload_request_id": U0, "index": 0}, {"request_id": "rec-r1", "workload_request_id": U1, "index": 1}],
              "checks": [{"id": "G6", "what": "chronology", "status": "PASS" if gm_verdict == "PASS" else "FAIL", "problems": [], "n_problems": 0,
                          "lengths_by_request": lens, "lag": lag}]}
        (row / "global_match.json").write_text(json.dumps(gm))
    return str(row / "match")


def test_a_reduced_prefix_is_accepted_only_linked_to_the_match_account_of_the_same_attempt(tmp_path):
    """The Commit's served count is linked: tokens.json served == Commit served, the same two lengths, GM-01 PASS, G6 agrees."""
    e = _mixed_epr()
    acc = EPL.load_match_account(_match_account(tmp_path))
    assert acc["tokens"] is not None and acc["global_match"] is not None and acc["notes"] == []
    L = EPL.link_to_match_account(e, acc, order=ORDER, declared_lag=1)
    assert L["ok"] and L["required"] and L["reduced_requests"] == ["r0"] and L["problems"] == [], L
    assert L["tokens_equal"] is True and L["tokens_json"]["tokens_equal"] == {"r0": True, "r1": True} and L["tokens_json"]["lag"] == 1
    # with the Commit's served lists the equality is the FULL generated_token_ids list: the same length + last token but
    # one interior id different -> not one serving transcript, first differing position named
    same_len = _served({"r0": 11, "r1": 32}, {"r0"}); same_len["r0"] = list(same_len["r0"]); same_len["r0"][4] = 99
    Lf = EPL.link_to_match_account(e, acc, order=ORDER, declared_lag=1, served_tokens=same_len)
    assert not Lf["ok"] and Lf["tokens_json"]["tokens_equal"] == {"r0": False, "r1": True} and Lf["tokens_json"]["tokens_equal_rule"] == "full generated_token_ids list"
    assert any("first differing position 4" in p and "not one serving transcript" in p for p in Lf["problems"]), Lf["problems"]
    assert EPL.link_to_match_account(e, acc, order=ORDER, declared_lag=1, served_tokens=_served({"r0": 11, "r1": 32}, {"r0"}))["ok"]
    assert L["global_match"]["verdict"] == "PASS" and L["global_match"]["requests_linked"] == ["r0", "r1"]
    assert "active-reference implementation classification only" in L["classification"] and "tau(e) = bot" in L["classification"]


def test_the_committers_own_count_is_never_sufficient_by_itself(tmp_path):
    """No Match account beside a REDUCED prefix -> problem by name; the same with no reduction is recorded, not a problem."""
    e = _mixed_epr()
    L = EPL.link_to_match_account(e, EPL.load_match_account(None), order=ORDER, declared_lag=1)
    assert not L["ok"] and L["required"]
    assert any("on the committer's own served count alone" in p and "(executed-prefix condition (a))" in p for p in L["problems"]), L["problems"]
    # tokens.json present but no global_match.json: linked to the served output only, not to the Match account -> problem
    L2 = EPL.link_to_match_account(e, EPL.load_match_account(_match_account(tmp_path, write_gm=False)), order=ORDER, declared_lag=1)
    assert not L2["ok"] and any("GM-01's account (global_match.json) is not beside" in p for p in L2["problems"]), L2["problems"]
    # every request ran its cap: nothing was placed on the committer's word -> a missing account is a note, ok stays True
    full = EP.executed_prefix_of_record(["r0", "r1"], _served({"r0": 32, "r1": 32}), caps={"r0": 32, "r1": 32}, eos_of_record=[EOS], lag=1)
    L3 = EPL.link_to_match_account(full, EPL.load_match_account(None), order=ORDER, declared_lag=1)
    assert L3["ok"] and not L3["required"] and L3["notes"] == ["no --match-dir: the Match record of this attempt is not beside the Commit"]


def test_a_transcript_that_is_not_the_matchs_or_an_unattested_lag_breaks_the_link_by_name(tmp_path):
    e = _mixed_epr()
    # (a) the Match served 12 tokens for r0 where the Commit served 11 -> not one serving transcript
    d = _match_account(tmp_path / "a", served=_served({"r0": 12, "r1": 32}, {"r0"}))
    L = EPL.link_to_match_account(e, EPL.load_match_account(d), order=ORDER, declared_lag=1)
    assert not L["ok"] and L["tokens_json"]["tokens_equal"]["r0"] is False and any("not one serving transcript" in p for p in L["problems"]), L["problems"]
    # (b) tokens.json attests LAG 0 under a Program declaring LAG 1 -> lag not attested on the record (its executed_prefix_len 11 != 12 too)
    d = _match_account(tmp_path / "b", tj_lag=0)
    L = EPL.link_to_match_account(e, EPL.load_match_account(d), order=ORDER, declared_lag=1)
    assert not L["ok"] and any("lag not attested on the record" in p and "(executed-prefix condition (b))" in p for p in L["problems"]), L["problems"]
    assert any("executed_prefix_len 11 != this rule's 12" in p for p in L["problems"]), L["problems"]
    # GM-01 FAILED -> the account this prefix must be linked to did not pass
    d = _match_account(tmp_path / "c", gm_verdict="FAIL")
    L = EPL.link_to_match_account(e, EPL.load_match_account(d), order=ORDER, declared_lag=1)
    assert not L["ok"] and any("GM-01 verdict 'FAIL'" in p for p in L["problems"]), L["problems"]
    # G6's account carries other lengths for r0 -> not the same account
    d = _match_account(tmp_path / "d", g6_lens={"rec-r0": {"served_output_len": 11, "executed_prefix_len": 11, "cap": 32, "cap_exhausted": False},
                                                 "rec-r1": {"served_output_len": 32, "executed_prefix_len": 32, "cap": 32, "cap_exhausted": True}})
    L = EPL.link_to_match_account(e, EPL.load_match_account(d), order=ORDER, declared_lag=1)
    assert not L["ok"] and any("G6 executed_prefix_len 11 != this rule's 12" in p for p in L["problems"]), L["problems"]
    # G6 reports variable lag -> no constant-lag placement is linkable
    d = _match_account(tmp_path / "e", g6_lag={"declared": 1, "record_attests": 1, "tokens_json_lag": 1, "exercised": True, "variable_lag": True})
    L = EPL.link_to_match_account(e, EPL.load_match_account(d), order=ORDER, declared_lag=1)
    assert not L["ok"] and any("VARIABLE LAG" in p for p in L["problems"]), L["problems"]


def test_a_padded_records_rows_past_the_stop_are_the_required_padding_and_padding_inside_the_prefix_is_the_prohibited_fallback():
    """Pad's padded binding map (`padded-binding-map/1`) stamps `activity` on every entry.  A not-executed row's
    entry is `bot-by-enablement` (the REQUIRED padding: tau(e) = bot by rule) and lies at a step outside the executed prefix BY CONSTRUCTION:
    it is counted, never 'execution past the stop'.  An `active` entry there (or one with no activity field: an unpadded record) IS execution
    past the stop, exactly as before.  The converse -- padding inside the executed prefix -- is the prohibited inactive fallback, named."""
    rsteps = {"r0": [0, 1, 2], "r1": [0, 1, 2, 3, 4, 5]}
    leaf = lambda rid, st, **kw: {"kind": "leaf", "identity": {"family": "instance_outputs", "request_id": rid, "step": st, "op_path": "x", "rank": 0}, **kw}  # noqa: E731
    # an unpadded record: r0 stopped at step 2 (executed 0..2) and committed rows at 3, 4 -> past the stop; nothing padded
    unpadded = {"entries": [leaf("r0", s) for s in range(5)] + [leaf("r1", s) for s in range(6)]}
    past = EP.rows_past_the_stop(unpadded, rsteps)
    assert not past["ok"] and past["requests"] == {"r0": [3, 4]} and past["entries_n"] == 2 and past["padding_entries_n"] == 0
    assert past["padded_record"] is False and "execution past the stop" in past["error"] and "inactive fallback" not in past["error"]
    # the same rows on a PADDED record: r0's steps 3..5 are bot-by-enablement (the full cap population), r1's are all active
    padded = {"schema": "padded-binding-map/1",
              "entries": [leaf("r0", s, activity="active") for s in range(3)] + [leaf("r0", s, activity=EP.ACTIVITY_PADDING) for s in range(3, 6)]
              + [leaf("r1", s, activity="active") for s in range(6)] + [{"kind": "leaf", "identity": {"family": "input_token_ids", "request_id": "r0", "step": None}, "activity": "registered-input"}]}
    ok = EP.rows_past_the_stop(padded, rsteps)
    assert ok["ok"] and ok["entries_n"] == 0 and ok["padding_entries_n"] == 3 and ok["padded_record"] is True and ok["error"] is None
    # an ACTIVE row past the stop on a padded record is still execution past the stop (the activity stamp does not launder it)
    mixed = {"entries": padded["entries"] + [leaf("r0", 6, activity="active"), leaf("r0", 7)]}
    bad = EP.rows_past_the_stop(mixed, rsteps)
    assert not bad["ok"] and bad["requests"] == {"r0": [6, 7]} and bad["padding_entries_n"] == 3 and "execution past the stop" in bad["error"]
    # padding INSIDE the executed prefix: r1 executed step 4 by the Program's rule but the record pads it -> the prohibited inactive fallback
    fallback = {"entries": [e for e in padded["entries"] if not (e["identity"]["request_id"] == "r1" and e["identity"]["step"] == 4)]
                + [leaf("r1", 4, activity=EP.ACTIVITY_PADDING)]}
    fb = EP.rows_past_the_stop(fallback, rsteps)
    assert not fb["ok"] and fb["requests"] == {} and fb["padding_inside_prefix"] == {"r1": [4]} and "inactive fallback" in fb["error"] and "r1" in fb["error"]
    assert fb["padding_entries_n"] == 3 and fb["padded_record"] is True
