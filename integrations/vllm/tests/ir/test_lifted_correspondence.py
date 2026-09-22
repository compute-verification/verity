"""LIFTING-SPEC v0.5.1 A7 (revb M-0042 F2): the identity-level correspondence table active-reference <-> lifted.

Unit tests on a hand-built required manifest (the rules: assignment by `s >= a_r + LAG`, bound/unbound by the coverage key, widths w -> w+1
from the Program's spec / the family / the dtype, the guard's gates per step, `compare` across two stopping points), then the REAL
record (sys's int-3 SmolLM2 earlystop row on the lane branch): the table's manifest digest IS the Commit's, 12,943 rows, the 20-step ⊥
suffix present and unbound, every active identity bound, every row widened.  All lifted-padded; nothing here is a panel cell."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from verity_vllm.lifting import correspondence as C
from verity_vllm.registry import lifted as L

ROW = Path("out/gen/sweep/evidence/pad/smollm2-135m__bf16__l40s__tp1__b1__i32__o32__earlystop__greedy__bi-eager/r16_0730_sys_int3c_earlystop")


def _ident(step, op, member, lo, hi, *, family="instance_outputs", dtype="bf16", spec="Gemm_v1", rid="r0", inv=0):
    d = {"family": family, "program_digest": "p", "request_id": rid, "step": step, "invocation": inv, "op_path": op, "output_member": member,
         "element_range": [lo, hi], "rank": 0, "dtype": dtype}
    if spec:
        d["spec"] = spec
    return d


def _manifest(cap, *, extra=()):
    ids = []
    for s in range(cap):
        ids.append(_ident(s, "model.layers.0.mlp", "out", 0, 8))
        ids.append(_ident(s, "runner.sampler", "sampled_token_ids", 0, 1, family="sampled_token_ids", dtype="i64", spec="TokenSelect_v1"))
        ids.append(_ident(s, "model.embed_tokens", "in", 0, 1, family="input_token_ids", dtype="i32", spec=None))
        ids.append(_ident(s, "fa2.m1/L0", "g3x3x1", 0, 12, family="fa2_hidden_m1_stream", dtype="u32", spec=None))
    ids.append(_ident("*", "model.embed_tokens.weight", "weight", 0, None, family="weights", dtype="bf16", spec=None, rid="*"))
    ids.extend(extra)
    return {"program_digest": "p", "manifest_digest": "m", "identities": ids}


def _bmap(steps, *, rid="r0", drop=()):
    ents = [{"identity": {"program_digest": "p", "request_id": "*", "step": -1, "invocation": 0, "op_path": "weights", "output_member": "registered",
                          "element_range": [0, 0], "rank": 0}, "kind": "registered-input"}]
    for s in steps:
        for op, mem, lo, hi in (("model.layers.0.mlp", "out", 0, 8), ("runner.sampler", "sampled_token_ids", 0, 1), ("model.embed_tokens", "in", 0, 1), ("fa2.m1/L0", "g3x3x1", 0, 12)):
            if (s, op) in drop:
                continue
            ents.append({"identity": {"program_digest": "p", "request_id": rid, "step": s, "invocation": 0, "op_path": op, "output_member": mem,
                                      "element_range": [lo, hi], "rank": 0}, "kind": "leaf"})
    return {"schema": "binding-map/1", "entries": ents}


WBS = {"Gemm_v1": [16], "TokenSelect_v1": [32]}


def test_a7_rows_assignment_widths_and_guard_on_a_hand_built_population():
    cap, a_r, lag = 8, 3, 1
    n_exec = L.expected_served(a_r, cap, lag)
    rows, s = C.correspondence(_manifest(cap), _bmap(range(n_exec)), commit_rid="r0", a_r=a_r, cap=cap, lag=lag, widths_by_spec=WBS)
    assert n_exec == 4 and len(rows) == cap * 4 + 1 == s["counts"]["rows"]
    assert s["counts"] == {"rows": 33, "active": 16, "bot": 16, "shared": 1, "bound_active": 16, "unbound_active": 0, "bound_bot": 0, "unbound_bot": 16,
                           "bound_shared": 1, "unbound_shared": 0, "plane_slices": 0, "plane_slices_active": 0, "plane_slices_bot": 0,
                           "planes_not_this_request": 0, "commit_key_unresolved": 0, "plane_slices_committed_present": 0, "plane_slices_committed_absent": 0, "plane_slices_committed_unplaced": 0}
    assert s["binding_rule"] == C.BINDING_RULE_LOCAL and s["commit_coverage"] is None  # a synthetic map: the local key match, NAMED as not the Commit's rule
    assert s["steps"]["active"] == [0, 1, 2, 3] and s["steps"]["bot"] == [4, 5, 6, 7] and s["steps"]["transition"] == 4 and s["steps"]["all_present"]
    assert s["checks"] == {"inactive_suffix_present": True, "no_active_identity_unbound": True, "bot_population_bound": False, "every_row_widened": True,
                           "every_row_one_identity": True}
    assert s["unique_keys"] == {"rows": 33, "unique_keys": 33, "every_row_one_identity": True, "duplicate_keys_n": 0, "duplicate_keys_first": []}
    assert all(r["invocation"] == 0 and r["rank"] == 0 for r in rows)
    # widths: the Program's leaf width (spec) for the module outputs and the sampled id, the family rule for the token / hidden-stream words
    by = s["widths"]["by_family_dtype"]
    assert (by["instance_outputs/bf16"]["w"], by["instance_outputs/bf16"]["w_lifted"], by["instance_outputs/bf16"]["w_source"]) == (16, 17, "spec")
    assert (by["sampled_token_ids/i64"]["w"], by["sampled_token_ids/i64"]["w_source"]) == (32, "spec")   # i64 storage, Value<32> leaf: the Program's width
    assert (by["input_token_ids/i32"]["w"], by["input_token_ids/i32"]["w_source"]) == (32, "family")
    assert (by["fa2_hidden_m1_stream/u32"]["w"], by["weights/bf16"]["w"], by["weights/bf16"]["w_source"]) == (32, 16, "dtype")
    assert s["widths"]["bot_words"] == [L.enc_bot(16), L.enc_bot(32)] == [65536, 2 ** 32] and s["widths"]["w_lifted_max"] == 33
    assert not s["widths"]["w_unknown"]
    # the rows themselves: a bot row holds enc(⊥) of ITS width, is unbound, and its step's guard is Active(0) / tok_in ⊥; an active row 0||v
    bot = [r for r in rows if r["assignment"] == "bot"]
    assert all(r["lifted_word"] == L.enc_bot(r["w"]) and r["active_reference"] == "unbound" and r["guard_live"] == "Active(0)" and r["tok_in"] == "bot" for r in bot)
    assert all(r["w_lifted"] == r["w"] + 1 and r["lifted_bits"] == r["n"] * (r["w"] + 1) for r in rows if r["n"] is not None)
    act = [r for r in rows if r["assignment"] == "active"]
    assert all(r["lifted_word"] == "0||v" and r["active_reference"] == "bound" for r in act)
    # the guard: steps 0..LAG unguarded (event 0 and the placeholder step are unconditionally live), then Active(1) while executed
    assert [s["per_step"][str(k)]["guard_live"] for k in range(cap)] == ["unguarded", "unguarded", "Active(1)", "Active(1)", "Active(0)", "Active(0)", "Active(0)", "Active(0)"]
    assert [s["per_step"][str(k)]["tok_in"] for k in range(cap)] == ["unguarded", "unguarded", "Active", "Active", "bot", "bot", "bot", "bot"]
    sh = [r for r in rows if r["assignment"] == "shared"]
    assert len(sh) == 1 and sh[0]["active_reference"] == "registered" and sh[0]["step"] is None and sh[0]["guard_live"] == "shared"
    assert s["label"] == "lifted-padded" and s["rules_version"] == "lifted-v1" and len(s["digest"]) == 64


def test_a7_a_coverage_hole_and_a_lifted_commit_are_told_apart():
    cap, a_r, lag = 6, 2, 0
    # a Commit missing one ACTIVE identity: a coverage hole of the executed population, named (not a lifted fact)
    _, s = C.correspondence(_manifest(cap), _bmap(range(2), drop={(1, "model.layers.0.mlp")}), commit_rid="r0", a_r=a_r, cap=cap, lag=lag, widths_by_spec=WBS)
    assert s["counts"]["unbound_active"] == 1 and s["checks"]["no_active_identity_unbound"] is False
    # a LIFTED Commit binds every bot identity too (enc(⊥) words, R6's question): bot_population_bound True
    _, s2 = C.correspondence(_manifest(cap), _bmap(range(cap)), commit_rid="r0", a_r=a_r, cap=cap, lag=lag, widths_by_spec=WBS)
    assert s2["counts"]["bound_bot"] == s2["counts"]["bot"] == 16 and s2["checks"]["bot_population_bound"] is True
    # no binding map: the active-reference column is n/a and the bound checks are None, the assignment / widths still stand
    rows3, s3 = C.correspondence(_manifest(cap), None, commit_rid="r0", a_r=a_r, cap=cap, lag=lag, widths_by_spec=WBS)
    assert all(r["active_reference"] == "n/a" for r in rows3) and s3["checks"]["no_active_identity_unbound"] is None and s3["checks"]["every_row_widened"]
    # a length stop (a_r = cap): no bot rows; the suffix check is None (nothing to be present) and padding_exercised False -- the structural
    # verdict stands, the A7 (a) evidence does not (a real ⊥ population needs a stop before the cap: OLMoE B2 r17_1653 is this case)
    _, s4 = C.correspondence(_manifest(cap), _bmap(range(cap)), commit_rid="r0", a_r=cap, cap=cap, lag=lag, widths_by_spec=WBS)
    assert s4["counts"]["bot"] == 0 and s4["checks"]["inactive_suffix_present"] is None and s4["steps"]["transition"] is None
    assert s4["padding_exercised"] is False and s2["padding_exercised"] is True and s4["checks"]["bot_population_bound"] is None
    from verity_vllm.lifting.record import _a7_ok
    assert _a7_ok(s4) is True and _a7_ok(s2) is True and _a7_ok(dict(s4, checks=dict(s4["checks"], every_row_widened=False))) is False
    # an identity with no width rule is NAMED, never guessed
    weird = _ident(0, "x.y", "out", 0, 4, family="mystery", dtype="q7", spec=None)
    _, s5 = C.correspondence(_manifest(cap, extra=[weird]), None, commit_rid="r0", a_r=a_r, cap=cap, lag=lag, widths_by_spec=WBS)
    assert s5["widths"]["w_unknown"] == {"mystery/q7": 1} and s5["checks"]["every_row_widened"] is False
    with pytest.raises(ValueError, match="NO identities"):
        C.correspondence({"identities": []}, None, commit_rid="r0", a_r=1, cap=2, lag=0)


def test_a7_compare_two_stopping_points_same_population_assignment_differs_between_the_stops(tmp_path):
    cap, lag = 8, 1
    ra, sa = C.correspondence(_manifest(cap), _bmap(range(L.expected_served(2, cap, lag))), commit_rid="r0", a_r=2, cap=cap, lag=lag, widths_by_spec=WBS)
    rb, sb = C.correspondence(_manifest(cap), _bmap(range(L.expected_served(5, cap, lag))), commit_rid="r0", a_r=5, cap=cap, lag=lag, widths_by_spec=WBS)
    pa, pb = C.write_table(ra, tmp_path / "a.jsonl.gz", sa), C.write_table(rb, tmp_path / "b.jsonl.gz", sb)
    sa2, ra2 = C.read_table(pa)
    assert sa2["digest"] == sa["digest"] and len(ra2) == len(ra)
    cmp = C.compare(ra2, C.read_table(pb)[1], sa2, C.read_table(pb)[0])
    assert cmp["population_equal"] and cmp["program_digest_equal"] and cmp["manifest_digest_equal"]
    assert cmp["n_exec"] == {"a": 3, "b": 6} and cmp["assignment_differs_on_steps"] == [3, 4, 5] == cmp["expected_differing_steps"]
    assert cmp["assignment_differs_exactly_between_the_stops"] and cmp["ok"]
    # a DIFFERENT population (one identity more) is not the same Program's: named, not ok
    rc, sc = C.correspondence(_manifest(cap, extra=[_ident(0, "model.extra", "out", 0, 2)]), None, commit_rid="r0", a_r=2, cap=cap, lag=lag, widths_by_spec=WBS)
    cmp2 = C.compare(ra, rc, sa, sc)
    assert not cmp2["population_equal"] and cmp2["only_in_b"] == [["r0", 0, "instance_outputs", "model.extra", 0, 0, "out", [0, 2]]] and not cmp2["ok"]
    # the CLI
    out = tmp_path / "cmp.json"
    assert C.main(["compare", pa, pb, "--out", str(out)]) == 0 and json.loads(out.read_text())["ok"] is True


@pytest.mark.skipif(not (ROW / "commit" / "verdict.json").exists(), reason="the R16 SmolLM2 earlystop evidence row is not in this tree")
def test_a7_on_the_real_earlystop_record_the_population_is_the_commits_and_the_bot_suffix_is_present_unbound_and_widened(tmp_path):
    """sys's int-3 SmolLM2 B1 earlystop row (a_r 11, LAG 1 attested, cap 32): the Commit's own required manifest, rebuilt here torch-free,
    has the Commit verdict's digest; the 20 ⊥ steps x 395 identities are PRESENT in the table (unbound in today's Commit), the 12 x 395
    active identities are all bound, the 303 weights registered, every row widened w -> w+1 (17 / 33), bot words 2^16 / 2^32."""
    from verity_vllm.lifting import record as R
    out = tmp_path / "rep.json"
    rc = R.main(["--build-request", str(ROW / "build_request"), "--capture", str(ROW / "match" / "capture"), "--commit", str(ROW / "commit"), "--pair", "0",
                 "--eos", "0", "--cap", "32", "--request-id", "de2d6bfc-096d-5c10-beb0-293ff914262d", "--commit-request-id", "r0", "--out", str(out),
                 "--correspondence", str(tmp_path / "corr.jsonl.gz")])
    rep = json.loads(out.read_text())
    assert rc == 0 and rep["ok"] is True and rep["checks"]["R2_activity"]["a_r"] == 11 and rep["lag"] == 1
    a7 = rep["A7_correspondence"]
    verdict = json.loads((ROW / "commit" / "verdict.json").read_text())
    assert a7["ok"] is True and a7["manifest_is_the_commits"] is True and a7["manifest_digest"] == verdict["required_manifest"]["manifest_digest"]
    assert a7["counts"] == {"rows": 12943, "active": 4740, "bot": 7900, "shared": 303, "bound_active": 4740, "unbound_active": 0, "bound_bot": 0,
                            "unbound_bot": 7900, "bound_shared": 303, "unbound_shared": 0, "plane_slices": 0, "plane_slices_active": 0, "plane_slices_bot": 0,
                            "planes_not_this_request": 0, "commit_key_unresolved": 0, "plane_slices_committed_present": 0, "plane_slices_committed_absent": 0, "plane_slices_committed_unplaced": 0}
    # F-r17b-15 (1): the binding column IS the Commit's verdict (its rule on this manifest and map): every row expanded (the 303 weights rows
    # each onto the one registered input), 5,043 bound, 7,900 missing = exactly the bot rows
    assert a7["binding_rule"] == C.BINDING_RULE_COMMIT
    cc = a7["commit_coverage"]["coverage"]
    assert cc["expected_n"] == 12943 and cc["bound_n"] == 4740 + 303 and cc["missing_n"] == 7900 and cc["ambiguous_n"] == 0 and cc["ok"] is False
    assert a7["commit_coverage"]["expansion"]["hidden_batch_rows_op_level"] in (0, None)  # B1: no batch-level plane rows (per-request hidden rows)
    assert a7["steps"]["bot"] == list(range(12, 32)) and a7["steps"]["identities_per_decode_step"] == [395] and a7["steps"]["transition"] == 12
    assert a7["checks"] == {"inactive_suffix_present": True, "no_active_identity_unbound": True, "bot_population_bound": False, "every_row_widened": True, "every_row_one_identity": True}
    assert a7["widths"]["bot_words"] == [65536, 4294967296] and a7["widths"]["w_lifted_max"] == 33 and not a7["widths"]["w_unknown"]
    assert a7["widths"]["by_family_dtype"]["instance_outputs/bf16"]["w_source"] == "spec" and a7["widths"]["by_family_dtype"]["sampled_token_ids/i64"]["w"] == 32
    # the Commit's own coverage record reconciles with the table: 7,900 missing = the bot rows, at exactly the bot steps
    cov = verdict["required_manifest"]["coverage"][0]
    assert cov["missing_n"] == a7["counts"]["unbound_bot"] and cov["steps_with_missing"] == a7["steps"]["bot"]
    summ, rows = C.read_table(tmp_path / "corr.jsonl.gz")
    assert summ["digest"] == a7["digest"] and len(rows) == 12943
    assert a7["per_step"]["12"] == {"identities": 395, "bound": 0, "unbound": 395, "elements": a7["per_step"]["12"]["elements"], "lifted_bits": a7["per_step"]["12"]["lifted_bits"],
                                    "assignment": "bot", "guard_live": "Active(0)", "tok_in": "bot", "plane_slices": 0, "plane_lifted_bits": 0,
                                    "plane_whole_launch_lifted_bits": 0, "plane_slices_committed_absent": 0}
    assert a7["per_step"]["11"]["assignment"] == "active" and a7["per_step"]["11"]["bound"] == 395 and a7["per_step"]["11"]["guard_live"] == "Active(1)"
    # `--no-correspondence` leaves the report without the table
    out2 = tmp_path / "rep2.json"
    R.main(["--build-request", str(ROW / "build_request"), "--capture", str(ROW / "match" / "capture"), "--commit", str(ROW / "commit"), "--eos", "0", "--cap", "32",
            "--request-id", "de2d6bfc-096d-5c10-beb0-293ff914262d", "--commit-request-id", "r0", "--out", str(out2), "--no-correspondence"])
    assert "A7_correspondence" not in json.loads(out2.read_text())


def _global_manifest_mixed(cap, *, requests=("r0", "r1")):
    """A hand-built GLOBAL manifest (the Commit's schema, digest over its rows) of two co-arriving requests with ONE packed per-step hidden
    plane per engine step (request `*`, `requests` = the co-scheduled set, member = the launch geometry) beside per-request rows."""
    from verity_capture.commit import required_manifest as RM
    ids = []
    es = []
    for s in range(cap):
        for rid in requests:
            ids.append(dict(_ident(s, "model.layers.0.mlp", "out", 0, 8), request_id=rid, engine_step=s))
            ids.append(dict(_ident(s, "runner.sampler", "sampled_token_ids", 0, 1, family="sampled_token_ids", dtype="i64", spec="TokenSelect_v1"), request_id=rid, engine_step=s))
        # the plane's fa2 geometry is SmolLM2-like at decode: kernel M = 3 (q-heads per kv head under GQA) with ONE row per sequence -- the
        # slot must come from the composition (rows 2 / n_seq 2), never from M (F-r17b-23 fix on the SmolLM2 B2 mixedstop record)
        ids.append(dict(_ident("*", "fa2.m1/L0", "g6x3x1", 0, 64, family="fa2_hidden_m1_stream", dtype="u32", spec=None, rid="*"), engine_step=s, requests=list(requests),
                        geometry={"HB": 6, "M": 3, "NB": 1, "BN": 128, "D": 64, "rows": len(requests), "n_seq": len(requests)}))
        es.append({"engine_step": s, "members": [{"request_id": rid, "step": s, "rows": 1} for rid in requests], "rows": len(requests), "n_seq": len(requests)})
    ids.append(_ident("*", "model.embed_tokens.weight", "weight", 0, None, family="weights", dtype="bf16", spec=None, rid="r0"))
    return {"schema": RM.SCHEMA, "program_digest": "p", "manifest_digest": RM.manifest_digest(ids), "identities": ids, "global": True, "engine_steps": es,
            "requests_declared": [{"request_id": rid, "index": k} for k, rid in enumerate(requests)],
            # the Commit's composition reference (the members its composition_check reads the RECORD's composition from)
            "composition_reference": {"input_token_ids": {"op_path": "model.embed_tokens", "output_member": "in", "per_row": 1},
                                      "sampled_token_ids": {"op_path": "runner.sampler", "output_member": "sampled_token_ids", "per_row": 1}}}


def _bmap_mixed(exec_steps: dict[str, range]):
    """The Commit's map of a mixed batch: per request, per EXECUTED step, its rows, its composition-reference entries (`model.embed_tokens/in`
    with the request's row offset in the launch and `runner.sampler/sampled_token_ids` with its sequence index -- the RECORD's composition:
    offsets by order among the requests the step ran, as the collector packs them) and the tap's per-sequence plane `s<j>/…` of the op (`j` =
    the record's sequence index) at a multi-sequence step, the packed plane `g…` at a one-sequence step, every plane bound to every request
    of the step (the op-level rule); plus the registered weights."""
    ents = [{"identity": {"program_digest": "p", "request_id": "*", "step": -1, "invocation": 0, "op_path": "weights", "output_member": "registered",
                          "element_range": [0, 0], "rank": 0}, "kind": "registered-input"}]
    all_steps = sorted({s for st in exec_steps.values() for s in st})
    for s in all_steps:
        present = [rid for rid, st in exec_steps.items() if s in st]      # the record's composition at s, packed in request order
        n_seq = len(present)
        for j, rid in enumerate(present):
            for op, mem, lo, hi, extra in (("model.layers.0.mlp", "out", 0, 8, {}),
                                           ("model.embed_tokens", "in", 0, 1, {"element_offset": j, "encoding": {"dtype": "int32", "shape": [n_seq]}}),
                                           ("runner.sampler", "sampled_token_ids", 0, 1, {"element_offset": j, "encoding": {"dtype": "int64", "shape": [n_seq, 1]}})):
                ents.append({"identity": {"program_digest": "p", "request_id": rid, "step": s, "invocation": 0, "op_path": op, "output_member": mem,
                                          "element_range": [lo, hi], "rank": 0}, "kind": "leaf", **extra})
            planes = ([(f"s{jj}/g1x1x1x4x8/S", 32) for jj in range(n_seq)] if n_seq > 1 else [("g1x1x1x4x8", 32)])
            for mem, numel in planes:                                     # every plane of the op / step bound to every request of the step
                ents.append({"identity": {"program_digest": "p", "request_id": rid, "step": s, "invocation": 0, "op_path": "fa2.m1/L0", "output_member": mem,
                                          "element_range": [0, numel], "rank": 0}, "kind": "leaf", "element_offset": 0, "encoding": {"dtype": "int32", "shape": [1, 1, 32]}})
    return {"schema": "binding-map/1", "entries": ents, "digest": "d", "entries_n": len(ents)}


def test_a3_6_packed_plane_slices_take_the_per_request_assignment_and_the_commits_own_binding_rule_decides_bound():
    """revb F-r17b-15: (1) the binding column is the Commit's verdict (binding.load_required_manifest -> coverage_check on the same
    manifest and map), (2) a packed per-step plane's (request, step) slices are rows of THAT request's table with its assignment."""
    cap, lag = 8, 1
    a = {"r0": 3, "r1": 8}                       # r0 stops (EOS at index 2 -> served 3, executed 4 under LAG 1); r1 runs to the cap
    n_exec = {r: L.expected_served(a[r], cap, lag) for r in a}
    man = _global_manifest_mixed(cap)
    bmap = _bmap_mixed({"r0": range(n_exec["r0"]), "r1": range(n_exec["r1"])})
    cov = C.commit_coverage(man, bmap, requests=["r0", "r1"], caps={"r0": cap, "r1": cap}, arrivals={"r0": 0, "r1": 0})
    # the Commit's expansion: 2 requests x 8 steps x 2 rows + 8 planes x 2 co-scheduled requests + the weights -> per-request identities
    assert cov["coverage"]["expected_n"] == 2 * cap * 2 + cap * 2 + 1 and cov["expansion"]["hidden_batch_rows_op_level"] == cap
    # what is missing under the Commit's rule is EXACTLY r0's rows and plane slices at its bot steps (4..7): 4 steps x (2 rows + 1 plane)
    assert cov["coverage"]["missing_n"] == (cap - n_exec["r0"]) * 3 and cov["coverage"]["ambiguous_n"] == 0
    assert cov["shape"].startswith("cap") and cov["of_record"] is None  # no request_steps asked: the cap-shaped (lifted) reading alone
    # the Commit's RULE OF RECORD (moe F-moe-r17-03 aad4bd3ea, R17-1) beside it: the requirement over each request's executed prefix of record
    # (one rule, two consumers: moe's executed_prefix_of_record on the served tokens = this tool's expected_served) is FULLY bound on the same
    # map, and what it does not require (programmed_not_executed_n) is EXACTLY the cap-shaped reading's missing = the bot population
    from verity_capture.commit import executed_prefix as EP
    served = {"r0": [5, 6, 0], "r1": [7] * cap}            # r0's last served token is the declared EOS 0; r1 exhausted its cap
    epr = EP.executed_prefix_of_record(["r0", "r1"], served, caps={"r0": cap, "r1": cap}, eos_of_record=[0], lag=lag)
    assert epr["ok"] and {r: epr["requests"][r]["executed_prefix_len"] for r in a} == n_exec
    steps = EP.request_steps_of_record(epr, {"r0": 0, "r1": 0})
    assert steps == {"r0": list(range(n_exec["r0"])), "r1": list(range(cap))}
    cov_r = C.commit_coverage(man, bmap, requests=["r0", "r1"], caps={"r0": cap, "r1": cap}, arrivals={"r0": 0, "r1": 0}, request_steps=steps)
    assert cov_r["coverage"] == cov["coverage"]  # the cap-shaped reading is unchanged by asking for the rule of record beside it
    ofr = cov_r["of_record"]
    assert ofr["available"] is True and ofr["coverage"]["ok"] is True and ofr["coverage"]["missing_n"] == 0 and ofr["missing_by_request"] == {}
    assert ofr["programmed_not_executed_n"] == cov["coverage"]["missing_n"] == (cap - n_exec["r0"]) * 3
    assert ofr["coverage"]["expected_n"] + ofr["programmed_not_executed_n"] == cov["coverage"]["expected_n"]
    assert ofr["request_steps_of_record"] == {"r0": [0, n_exec["r0"] - 1], "r1": [0, cap - 1]}
    # and no committed leaf of r0 lies past its stop on this map (moe's rows_past_the_stop); a row of r0 at a bot step WOULD be named
    assert EP.rows_past_the_stop(bmap, steps)["ok"] is True
    poisoned = _bmap_mixed({"r0": range(n_exec["r0"] + 1), "r1": range(cap)})
    past = EP.rows_past_the_stop(poisoned, steps)
    assert past["ok"] is False and past["requests"] == {"r0": [n_exec["r0"]]} and past["entries_n"] == 5  # mlp, embed, sampler, two per-sequence planes
    parts = C.split_by_request(man)
    tabs = {}
    for rid in ("r0", "r1"):
        part = dict(parts[rid], engine_steps=man["engine_steps"])
        rows, s = C.correspondence(part, bmap, commit_rid=rid, a_r=a[rid], cap=cap, lag=lag, widths_by_spec=WBS, commit_cov=cov)
        tabs[rid] = (rows, s)
        assert s["binding_rule"] == C.BINDING_RULE_COMMIT and s["commit_coverage"]["coverage"]["ok"] is False
        # every plane is a row of BOTH requests' tables (both co-scheduled at every engine step): one slice per request, never a 'shared' plane
        assert s["counts"]["plane_slices"] == cap and s["counts"]["planes_not_this_request"] == 0 and s["counts"]["commit_key_unresolved"] == 0
        assert s["counts"]["shared"] == 1 and s["counts"]["bound_shared"] == 1   # the weights: registered once for the workload, in every view
        pl = [r for r in rows if r.get("packed_plane")]
        assert all(r["plane_requests"] == ["r0", "r1"] and r["slice_rows"] == 1 and r["slice_step_source"] == "composition" for r in pl)
    r0_rows, r0 = tabs["r0"]
    r1_rows, r1 = tabs["r1"]
    # r0: slices at steps 0..3 active / bound, 4..7 bot / enc(bot) / unbound (the active-reference Commit binds the executed set only)
    assert {k: r0["plane_slices"][k] for k in ("n", "active", "bot", "bound_bot", "unbound_active", "steps_bot")} == \
        {"n": 8, "active": 4, "bot": 4, "bound_bot": 0, "unbound_active": 0, "steps_bot": [4, 5, 6, 7]}
    assert [r["lifted_word"] for r in r0_rows if r.get("packed_plane")] == ["0||v"] * 4 + [2 ** 32] * 4
    # F-r17b-23 (revb): a plane row's element_range is the request's SLOT of the lifted fixed plane (manifest geometry g<HB>x<M>x<NB>…: M = 1
    # row per sequence, n_seq 2 -> 2 rows of 32 elements), placed ALSO in the committed plane by the RECORD's composition (the map's
    # composition-reference entries), or named absent -- never the whole plane standing for the slice
    p0 = {r["engine_step"]: r for r in r0_rows if r.get("packed_plane")}
    p1 = {r["engine_step"]: r for r in r1_rows if r.get("packed_plane")}
    assert p0[0]["lifted_slot"] == {"seq_index": 0, "slot_rows": 1, "M_geometry": 3, "M_geometry_equals_slot_rows": False, "n_seq": 2, "rows_total": 2, "elements_per_row": 32,
                                    "row_range": [0, 1], "own_rows": [0, 1], "pad_rows": 0, "element_range": [0, 32], "elements": 32,
                                    "slot_rule": "cumulative declared member rows (sum(rows) == rows_total: no pad rows)", "source": p0[0]["lifted_slot"]["source"]}
    assert p1[0]["lifted_slot"]["row_range"] == [1, 2] and p1[0]["lifted_slot"]["element_range"] == [32, 64] and p1[0]["element_range"] == [32, 64] and p1[0]["n"] == 32
    assert p0[0]["element_range"] == [0, 32] and p0[0]["n"] == 32 and p0[0]["lifted_bits"] == 32 * 33 and p0[0]["plane_whole_launch"] == {"element_range": [0, 64], "n": 64, "lifted_bits": 64 * 33}
    # steps 0..3 (both ran): r0 = sequence 0 rows [0,1) of a 2-row launch, carried by the per-sequence plane s0/…; r1 = sequence 1 rows [1,2), s1/…
    c0, c1 = p0[0]["committed_slice"], p1[0]["committed_slice"]
    assert c0["present"] is True and (c0["seq_index"], c0["row_range"], c0["rows_total"], c0["n_seq"], c0["carrier"], c0["members"], c0["elements"], c0["elements_per_row"]) == \
        (0, [0, 1], 2, 2, "per-sequence planes", ["s0/g1x1x1x4x8/S"], 32, 32)
    assert c1["present"] is True and (c1["seq_index"], c1["row_range"], c1["members"]) == (1, [1, 2], ["s1/g1x1x1x4x8/S"])
    assert (c0["carrier_elements"], c0["carrier_pad_elements"], c0["carrier_extent_consistent"]) == (32, 0, True)
    # steps 4..7 (r1 alone): r0's slice is ABSENT from the committed plane (n_seq 1) with its bot slot's size from the LIFTED fixed shape; r1's
    # Active slice is rows [0,1) of the 1-row PACKED plane -- while in the lifted fixed plane r0 keeps slot [0,1) (enc(bot)) and r1 slot [1,2)
    for e in range(4, 8):
        a0, a1 = p0[e]["committed_slice"], p1[e]["committed_slice"]
        assert a0["present"] is False and a0["why"] == "absent (committed n_seq = 1: ['r1'])" and a0["bot_slot_elements"] == 32 and a0["bot_slot_rows"] == 1
        assert a1["present"] is True and (a1["row_range"], a1["rows_total"], a1["n_seq"], a1["carrier"], a1["members"], a1["element_range_in_carrier"]) == \
            ([0, 1], 1, 1, "packed plane", ["g1x1x1x4x8"], [0, 32])
        assert p0[e]["lifted_slot"]["row_range"] == [0, 1] and p1[e]["lifted_slot"]["row_range"] == [1, 2] and p0[e]["lifted_word"] == 2 ** 32 and p1[e]["lifted_word"] == "0||v"
    assert r0["plane_slices"]["committed_present"] == 4 and r0["plane_slices"]["committed_absent"] == 4 and r0["plane_slices"]["committed_absent_are_exactly_bot"] is True
    assert r1["plane_slices"]["committed_present"] == 8 and r1["plane_slices"]["committed_absent"] == 0 and r1["plane_slices"]["active_slices_placed_in_committed_plane"] is True
    assert r0["plane_slices"]["active_slices_placed_in_committed_plane"] is True and r0["counts"]["plane_slices_committed_unplaced"] == 0
    assert r0["per_step"]["4"]["plane_slices_committed_absent"] == 1 and r0["per_step"]["3"]["plane_slices_committed_absent"] == 0
    assert r0["checks"] == {"inactive_suffix_present": True, "no_active_identity_unbound": True, "bot_population_bound": False, "every_row_widened": True,
                            "every_row_one_identity": True}
    assert r0["counts"]["unbound_active"] == 0 and r0["counts"]["unbound_bot"] == (cap - n_exec["r0"]) * 3 and r0["padding_exercised"] is True
    # r1: the SAME planes, every slice active / 0||v / bound -- the mixed batch's object (d): at engine steps 4..7 r0's slice is bot, r1's active
    assert r1["plane_slices"]["bot"] == 0 and r1["plane_slices"]["active"] == 8 and r1["plane_slices"]["unbound_active"] == 0
    assert r1["counts"]["unbound_active"] == 0 and r1["padding_exercised"] is False and r1["checks"]["inactive_suffix_present"] is None
    for e in range(4, 8):
        a0 = next(r for r in r0_rows if r.get("packed_plane") and r["engine_step"] == e)
        a1 = next(r for r in r1_rows if r.get("packed_plane") and r["engine_step"] == e)
        assert a0["op_path"] == a1["op_path"] and a0["output_member"] == a1["output_member"]          # one plane
        assert (a0["assignment"], a0["active_reference"], a0["lifted_word"]) == ("bot", "unbound", 2 ** 32)
        assert (a1["assignment"], a1["active_reference"], a1["lifted_word"]) == ("active", "bound", "0||v")
    # the plane's bits are kept apart from the request's own elements / lifted_bits in per_step: the request's SLOT (F-r17b-23) and the whole launch
    assert r0["per_step"]["0"]["plane_slices"] == 1 and r0["per_step"]["0"]["plane_lifted_bits"] == 32 * 33 and r0["per_step"]["0"]["elements"] == 9
    assert r0["per_step"]["0"]["plane_whole_launch_lifted_bits"] == 64 * 33
    # F-r17b-18 (revb): ... and apart from the request's WIDTHS -- the planes enter neither request's by_family_dtype / lifted_bits_total
    # (summing per-request totals would count each shared plane once per extra co-scheduled request); each request reports its VIEW under
    # planes_seen, and the workload-level document counts every plane exactly once
    for s in (r0, r1):
        assert s["widths"]["excludes_planes"] is True and not any(k.startswith("fa2.m1") or "hidden" in k for k in s["widths"]["by_family_dtype"])
        assert s["widths"]["planes_seen"]["planes"] == cap and s["widths"]["planes_seen"]["lifted_bits_whole_launch"] == cap * 64 * 33
        assert s["widths"]["lifted_bits_total"] == sum(v["lifted_bits"] for v in s["widths"]["by_family_dtype"].values())
    assert r0["widths"]["planes_seen"]["by_family_dtype"]["fa2_hidden_m1_stream/u32"]["plane_slices_bot"] == 4
    assert r1["widths"]["planes_seen"]["by_family_dtype"]["fa2_hidden_m1_stream/u32"]["plane_slices_bot"] == 0
    pw = C.plane_widths(man, {"r0": WBS, "r1": WBS})
    assert pw["planes"] == cap and pw["lifted_bits_total"] == cap * 64 * 33 and pw["payload_bits_total"] == cap * 64 * 32
    assert pw["by_family_dtype"]["fa2_hidden_m1_stream/u32"]["planes"] == cap and pw["by_engine_step"]["0"] == {"planes": 1, "elements": 64, "lifted_bits": 64 * 33, "requests": 2}
    # a request's total is exactly its NON-plane rows' bits; the workload's bits are the two totals plus the planes ONCE (no plane twice)
    for rows, s in (tabs["r0"], tabs["r1"]):
        assert s["widths"]["lifted_bits_total"] == sum(r["lifted_bits"] or 0 for r in rows if not r.get("packed_plane")) > 0
        # F-r17b-23: a plane row's bits are now the request's SLOT (half of each 2-slot plane); the whole launch stays on the row beside it
        assert sum(r["lifted_bits"] or 0 for r in rows if r.get("packed_plane")) == cap * 32 * 33 == s["widths"]["planes_seen"]["slice_lifted_bits_total"]
        assert sum(r["plane_whole_launch"]["lifted_bits"] for r in rows if r.get("packed_plane")) == cap * 64 * 33
    # the two requests' slots TILE the planes: Σ slot bits == the planes once (the workload-level reconciliation)
    assert r0["widths"]["planes_seen"]["slice_lifted_bits_total"] + r1["widths"]["planes_seen"]["slice_lifted_bits_total"] == pw["lifted_bits_total"] == cap * 64 * 33
    assert r0["widths"]["lifted_bits_total"] + r1["widths"]["lifted_bits_total"] + pw["lifted_bits_total"] == \
        sum((r["lifted_bits"] or 0) for r in r0_rows if not r.get("packed_plane")) + sum((r["lifted_bits"] or 0) for r in r1_rows if not r.get("packed_plane")) + cap * 64 * 33
    # the local rule is never applied when the Commit's verdict is given: a map that binds a plane under a DIFFERENT member name still binds
    # (the Commit's op-level rule), where the local key match would say unbound
    _, s_local = C.correspondence(dict(parts["r1"], engine_steps=man["engine_steps"]), bmap, commit_rid="r1", a_r=8, cap=cap, lag=lag, widths_by_spec=WBS)
    assert s_local["binding_rule"] == C.BINDING_RULE_LOCAL and s_local["plane_slices"]["unbound_active"] == 8 and r1["plane_slices"]["unbound_active"] == 0


def test_f_r17b_23_lifted_slot_is_sized_by_the_declared_composition_never_by_the_fa2_kernel_m():
    """The slot of a request in the LIFTED fixed plane: cumulative member rows when they tile the plane (SmolLM2: 32 + 32 = 64 at prefill;
    1 + 1 = 2 at decode under GQA geometry M = 3), equal slots of rows_total / n_seq with pad rows when they do not (OLMoE step 0: 31 + 32 in a
    64-row plane -> slots of 32, r0 pads 1), and NAMED unplaceable otherwise -- the geometry's M is recorded beside, never load-bearing."""
    def man(rows_by_member, rows_total):
        return {"engine_steps": [{"engine_step": 0, "members": [{"request_id": f"r{k}", "step": 0, "rows": r} for k, r in enumerate(rows_by_member)],
                                  "rows": rows_total, "n_seq": len(rows_by_member)}]}
    def ident(member, n, rows_total, n_seq, M):
        return {"engine_step": 0, "element_range": [0, n], "requests": [f"r{k}" for k in range(n_seq)], "output_member": member, "op_path": "fa2.m1/L0",
                "geometry": {"HB": 6, "M": M, "NB": 1, "rows": rows_total, "n_seq": n_seq}}
    # SmolLM2 decode-like: M 3, one row per sequence, 3564 elements per row
    s = C.plane_slice(ident("g6x3x1", 7128, 2, 2, 3), man([1, 1], 2), "r1", rcomp=None, rplanes=None)["lifted_slot"]
    assert (s["slot_rows"], s["M_geometry"], s["M_geometry_equals_slot_rows"], s["row_range"], s["element_range"], s["elements"], s["pad_rows"]) == (1, 3, False, [1, 2], [3564, 7128], 3564, 0)
    # SmolLM2 prefill-like: M 32 == rows per sequence
    s = C.plane_slice(ident("g18x32x1", 228096, 64, 2, 32), man([32, 32], 64), "r0", rcomp=None, rplanes=None)["lifted_slot"]
    assert (s["slot_rows"], s["M_geometry_equals_slot_rows"], s["row_range"], s["element_range"], s["elements_per_row"]) == (32, True, [0, 32], [0, 114048], 3564)
    # OLMoE step 0: members 31 + 32 in a 64-row plane -> equal slots of 32; r0 owns 31 rows and pads 1
    s = C.plane_slice(ident("g16x32x1x64x128", 307200, 64, 2, 32), man([31, 32], 64), "r0", rcomp=None, rplanes=None)["lifted_slot"]
    assert (s["slot_rows"], s["row_range"], s["own_rows"], s["pad_rows"], s["element_range"], s["slot_rule"].split(" (")[0]) == (32, [0, 32], [0, 31], 1, [0, 153600], "rows_total / n_seq equal slots")
    s1 = C.plane_slice(ident("g16x32x1x64x128", 307200, 64, 2, 32), man([31, 32], 64), "r1", rcomp=None, rplanes=None)["lifted_slot"]
    assert (s1["row_range"], s1["own_rows"], s1["pad_rows"], s1["element_range"]) == ([32, 64], [32, 64], 0, [153600, 307200])
    # revb F-r17b-26: the --build-workload path on a B1 row -- n_seq 1, rows_total 1, GQA tile M 3: the slot is the whole 1-row plane (3,564), never 3 x it
    s = C.plane_slice(ident("g3x3x1", 3564, 1, 1, 3), man([1], 1), "r0", rcomp=None, rplanes=None)["lifted_slot"]
    assert (s["slot_rows"], s["M_geometry"], s["M_geometry_equals_slot_rows"], s["row_range"], s["element_range"], s["elements"], s["pad_rows"]) == (1, 3, False, [0, 1], [0, 3564], 3564, 0)
    # ... and its COMMITTED slice on a tile-rounded per-sequence carrier (3,584 = 28 x 128 for 3,564 payload elements): the slice's elements are the
    # payload's, the carrier's padded extent is named as its own field (revb F-r17b-26: "3,584 != 3,564 unexplained on the row")
    rcomp = {"steps": {0: {"n_seq": 1, "rows_total": 1, "order": ["r0"], "members": {"r0": {"seq_index": 0, "row_lo": 0, "row_hi": 1, "rows": 1}}}}}
    rplanes = {("fa2.m1/L0", 0): [{"member": "s0/g3x3x1/S", "seq": 0, "M": 3, "part": "S", "numel": 3584}]}
    c = C.plane_slice(ident("g3x3x1", 3564, 1, 1, 3), man([1], 1), "r0", rcomp=rcomp, rplanes=rplanes)["committed_slice"]
    assert (c["present"], c["elements"], c["elements_per_row"], c["element_range_in_carrier"], c["carrier_elements"], c["carrier_pad_elements"], c["carrier_extent_consistent"],
            c["M_geometry_committed"], c["M_geometry_equals_rows"]) == (True, 3564, 3564, [0, 3564], 3584, 20, True, 3, False)
    # a packed 1-sequence carrier (r1 alone at a bot step of r0): M_geometry_equals_rows is stated, not null (revb M-0394 (i))
    rplanes2 = {("fa2.m1/L0", 0): [{"member": "g1x1x1x4x8", "seq": None, "M": 1, "part": None, "numel": 3564}]}
    c = C.plane_slice(ident("g3x3x1", 3564, 1, 1, 3), man([1], 1), "r0", rcomp=rcomp, rplanes=rplanes2)["committed_slice"]
    assert (c["carrier"], c["elements"], c["carrier_elements"], c["carrier_pad_elements"], c["M_geometry_equals_rows"]) == ("packed plane", 3564, 3564, 0, True)
    # neither rule applies: named, never guessed
    s = C.plane_slice(ident("g6x3x1", 33 * 10, 33, 2, 3), man([31, 1], 33), "r1", rcomp=None, rplanes=None)["lifted_slot"]
    assert s["row_range"] == [None, None] and s["element_range"] is None and s["slot_rule"].startswith("unplaceable")



def test_f_r17b_28_a_shared_module_s_invocations_are_distinct_rows_and_compare_judges_over_the_full_key():
    """revb F-r17b-28: the shared RotaryEmbedding (canonical op_path, invocation = layer) is N identities per member per step; the table writes
    invocation + rank on every row, every row is one identity under the full key, and `compare` sees a difference confined to one invocation."""
    cap, lag, a_r = 6, 1, 3
    n_exec = L.expected_served(a_r, cap, lag)
    rot = [_ident(s, "model.layers.0.self_attn.rotary_emb", "0", 0, 4, spec="RoPE_v1", inv=k) for s in range(cap) for k in range(3)]
    man = _manifest(cap, extra=rot)
    rows, s = C.correspondence(man, _bmap(range(n_exec)), commit_rid="r0", a_r=a_r, cap=cap, lag=lag, widths_by_spec=WBS)
    assert s["counts"]["rows"] == cap * 4 + 1 + cap * 3 and s["unique_keys"]["every_row_one_identity"] and s["checks"]["every_row_one_identity"]
    rr = [r for r in rows if r["op_path"] == "model.layers.0.self_attn.rotary_emb" and r["step"] == 0]
    assert sorted(r["invocation"] for r in rr) == [0, 1, 2] and len({C._key(r) for r in rr}) == 3
    # the same population without the invocation column would collapse: 3 rows -> 1 key
    assert len({(r["request_id"], r["step"], r["family"], r["op_path"], r["output_member"], tuple(r["element_range"])) for r in rr}) == 1
    # compare: a table whose invocation-2 rows differ in assignment is NOT equal / ok -- visible only under the full key
    rows_b = [dict(r) for r in rows]
    for r in rows_b:
        if r["op_path"] == "model.layers.0.self_attn.rotary_emb" and r["invocation"] == 2 and r["step"] == 1:
            r["assignment"] = "bot"
    rep = C.compare(rows, rows_b, s, s)
    assert rep["population_equal"] is True and rep["keys"] == {"a": len(rows), "b": len(rows)} and rep["every_row_one_identity"] == {"a": True, "b": True}
    assert rep["assignment_differs_on_steps"] == [1] and rep["ok"] is False  # expected differing steps = [] (same n_exec): the difference is seen
    # a table with a duplicated row (the invocation column dropped) fails the invariant by name
    dup = [dict(r, invocation=0) for r in rows]
    u = C.unique_keys(dup)
    assert u["every_row_one_identity"] is False and u["duplicate_keys_n"] == cap and u["rows"] - u["unique_keys"] == cap * 2
    assert C.compare(dup, dup, s, s)["population_equal"] is False
