"""Live C2 oracle-compare: committed bytes vs the Match record of the same attempt.  Synthetic Match dir
(instances.jsonl + capture/snapshots.json + values + log.jsonl.gz) with two requests whose engine row order differs from the
declaration order; the collector's 'retained bytes' are a dict.  Positive + the attacks C1/C3 are blind to: relabel across
op_paths, same-name byte swap, a stale/late value, a wrong sampled id, and the partial case (a step the
Match did not snapshot -> result None, never PASS)."""
from __future__ import annotations

import gzip
import hashlib
import json
import os

import numpy as np
import pytest

from verity_vllm.check import oracle_compare as OC

U0, U1 = "aaaaaaaa-0000-0000-0000-000000000000", "bbbbbbbb-1111-1111-1111-111111111111"
H = 4                      # hidden width
ROWS = {U1: 3, U0: 5}      # engine row order at step 0: r1's 3 rows first, then r0's 5 (declaration order is r0, r1)


def _bf16(rows: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 65535, size=rows * H, dtype=np.uint16).astype("<u2").tobytes()


def make_match(tmp: str, *, snapshot_steps=(0,), sampled=None) -> tuple[str, dict]:
    """A Match dir for ONE step with ops A (model.embed_tokens: single output) and B (model.layers.0.post_attention_layernorm: (out, residual)),
    a Gemm C (model.layers.0.self_attn.qkv_proj); every op's output alloc snapshotted at its event.  Returns (dir, oracle values by (op, member))."""
    d = os.path.join(tmp, "match")
    os.makedirs(os.path.join(d, "capture", "values", "0"), exist_ok=True)
    order = [U1, U0]
    n = sum(ROWS.values())
    vals = {("model.embed_tokens", "in"): np.arange(100, 100 + n, dtype="<i4").tobytes(),
            ("model.embed_tokens", "out"): _bf16(n, 1), ("model.layers.0.post_attention_layernorm", "0"): _bf16(n, 2),
            ("model.layers.0.post_attention_layernorm", "1"): _bf16(n, 3), ("model.layers.0.self_attn.qkv_proj", "0"): _bf16(n, 4)}
    allocs = {("model.embed_tokens", "in"): 10, ("model.embed_tokens", "out"): 11, ("model.layers.0.post_attention_layernorm", "0"): 12,
              ("model.layers.0.post_attention_layernorm", "1"): 13, ("model.layers.0.self_attn.qkv_proj", "0"): 14}
    events = {"model.embed_tokens": 50, "model.layers.0.post_attention_layernorm": 60, "model.layers.0.self_attn.qkv_proj": 70}
    entries = []
    for (op, mem), data in vals.items():
        sha = hashlib.sha256(data).hexdigest()
        fn = f"values/0/{sha[:16]}.bin"
        open(os.path.join(d, "capture", fn), "wb").write(data)
        dt, isz = ("int32", 4) if mem == "in" else ("bfloat16", 2)
        entries.append({"event": events[op], "path": "args[0]" if mem == "in" else ("outs" if mem == "out" else f"outs[{mem}]"),
                        "view": {"$v": [allocs[(op, mem)], 0, [n] if mem == "in" else [n, H], [1] if mem == "in" else [H, 1], dt, "cuda:0"]},
                        "dtype": dt, "shape": [n] if mem == "in" else [n, H], "strides": [1] if mem == "in" else [H, 1], "byte_offset": 0,
                        "nbytes": len(data), "layout": "spanned-storage-bytes", "sha256": sha, "file": fn, "step": 0})
    json.dump({"schema": "verity-capture/snapshots/v1", "steps": list(snapshot_steps), "entries": entries}, open(os.path.join(d, "capture", "snapshots.json"), "w"))
    with open(os.path.join(d, "instances.jsonl"), "w") as f:
        f.write(json.dumps({"schema": "verity-gen/instances/v1", "digest": "f" * 64, "n_instances": 0,
                            "requests": [{"index": 0, "request_id": U0, "prompt_len": 5, "served_samples": 1, "max_tokens": 1},
                                         {"index": 1, "request_id": U1, "prompt_len": 3, "served_samples": 1, "max_tokens": 1}]}) + "\n")
        row = 0
        pending = []                           # the qkv_proj instances, written SHUFFLED with the fold's attention-style `row` (see below)
        for u in order:
            for pos in range(ROWS[u]):
                for op in events:
                    ov = []
                    if op == "model.embed_tokens":
                        opv = [{"alloc": 10, "offset": row, "shape": [], "strides": [], "dtype": "int32", "device": "cuda:0"}]
                        ov = [{"alloc": 11, "offset": row * H, "shape": [H], "strides": [1], "dtype": "bfloat16", "device": "cuda:0"}]
                    elif op == "model.layers.0.post_attention_layernorm":
                        opv = []
                        ov = [{"alloc": 12, "offset": row * H, "shape": [H], "strides": [1], "dtype": "bfloat16", "device": "cuda:0"},
                              {"alloc": 13, "offset": row * H, "shape": [H], "strides": [1], "dtype": "bfloat16", "device": "cuda:0"}]
                    else:
                        opv = []
                        ov = [{"alloc": 14, "offset": row * H, "shape": [H], "strides": [1], "dtype": "bfloat16", "device": "cuda:0"}]
                    rec = {"node": 0, "step": 0, "event": events[op], "pattern": "x", "spec": "X_v1{}", "row": row, "rows": n,
                           "name": f"t={pos}/{op.rsplit('.', 1)[-1]}", "hint": f"{op}/{op}.weight", "request": u, "position": pos,
                           "operand_views": opv, "output_views": ov}
                    if op == "model.layers.0.self_attn.qkv_proj":
                        # the fold's ATTENTION convention (real B2 Match record): `row` is the sequence's index inside the varlen
                        # launch and not even constant over one request's tokens ({0,1} across a 256-token prefill), `rows` = n sequences;
                        # only `position` (and the view's storage offset) orders the request's block -- sorting by `row` permutes it
                        rec["row"], rec["rows"] = (pos % 2), len(order)
                        pending.append(rec)
                    else:
                        f.write(json.dumps(rec) + "\n")
                row += 1
        for rec in sorted(pending, key=lambda r: (r["row"], -r["position"])):
            f.write(json.dumps(rec) + "\n")
    sampled = sampled if sampled is not None else [777, 555]          # per request in row order: U1 -> 777, U0 -> 555
    with gzip.open(os.path.join(d, "capture", "log.jsonl.gz"), "wt") as f:
        f.write(json.dumps({"$e": "RunHeader", "schema": "verity-capture/log/v1", "created_utc": "2026-09-19T20:00:00Z", "profile_id": "abc", "profile": {"vllm": {"version": "t"}}}) + "\n")
        f.write(json.dumps({"$e": "StepStart", "step": 0, "rows": [[u, p] for u in order for p in range(ROWS[u])]}) + "\n")
        f.write(json.dumps({"$e": "SampleEnd", "step": 0, "sampled_token_ids": sampled}) + "\n")
        f.write(json.dumps({"$e": "StepStart", "step": 1, "rows": [[u, ROWS[u]] for u in order]}) + "\n")
        f.write(json.dumps({"$e": "SampleEnd", "step": 1, "sampled_token_ids": [1, 2]}) + "\n")
    # tokens.json: the ids oracle (prompt + generated per request); the 'in' member above is arange(100..) in row order: U1 rows 0-2, U0 rows 3-7
    json.dump({"requests": [{"request_id": U1, "prompt_token_ids": [100, 101, 102], "generated_token_ids": [sampled[0]]},
                            {"request_id": U0, "prompt_token_ids": [103, 104, 105, 106, 107], "generated_token_ids": [sampled[1]]}]},
              open(os.path.join(d, "capture", "tokens.json"), "w"))
    return d, vals


def make_bmap(vals: dict, *, steps=(0,)) -> tuple[dict, dict]:
    """Binding map entries (request-local ranges, element_offset = the request's block in the engine's row order) + the committed
    members dict {(step, name): bytes} equal to the oracle."""
    names = {("model.embed_tokens", "in"): "model.embed_tokens/in", ("model.embed_tokens", "out"): "model.embed_tokens",
             ("model.layers.0.post_attention_layernorm", "0"): "model.layers.0.post_attention_layernorm/0",
             ("model.layers.0.post_attention_layernorm", "1"): "model.layers.0.post_attention_layernorm/1",
             ("model.layers.0.self_attn.qkv_proj", "0"): "model.layers.0.self_attn.qkv_proj/0"}
    entries, committed = [], {}
    for step in steps:
        off = 0
        for u, lab in ((U1, "r1"), (U0, "r0")):
            rows = ROWS[u]
            for (op, mem), data in vals.items():
                w = 1 if mem == "in" else H
                dt = "int32" if mem == "in" else "bfloat16"
                entries.append({"identity": {"program_digest": "p" * 64, "request_id": lab, "step": step, "invocation": 0, "op_path": op, "output_member": mem,
                                             "element_range": [0, rows * w], "rank": 0}, "kind": "leaf", "name": names[(op, mem)], "element_offset": off * w,
                                "encoding": {"dtype": dt, "shape": [sum(ROWS.values())] if mem == "in" else [sum(ROWS.values()), H]}})
                committed[(step, names[(op, mem)])] = data
            off += rows
        for lab, slot in (("r1", 0), ("r0", 1)):
            entries.append({"identity": {"program_digest": "p" * 64, "request_id": lab, "step": step, "invocation": 0, "op_path": "runner.sampler",
                                         "output_member": "sampled_token_ids", "element_range": [0, 1], "rank": 0}, "kind": "leaf", "name": "runner.sampler/sampled_token_ids",
                            "element_offset": slot, "encoding": {"dtype": "int64", "shape": [2]}})
        committed[(step, "runner.sampler/sampled_token_ids")] = np.array([777, 555], dtype="<i8").tobytes()
        entries.append({"identity": {"program_digest": "p" * 64, "request_id": "*", "step": step, "invocation": 0, "op_path": "fa2.m1/L0/g32x8x1", "output_member": "S",
                                     "element_range": [0, 8], "rank": 0}, "kind": "leaf", "name": "fa2.m1/L0/s0/g32x8x1/S", "element_offset": 0, "encoding": {"dtype": "int32", "shape": [8]}})
        committed[(step, "fa2.m1/L0/s0/g32x8x1/S")] = bytes(32)
    entries.append({"identity": {"program_digest": "p" * 64, "request_id": "*", "step": 0, "invocation": 0, "op_path": "weights", "output_member": "model.embed_tokens.weight",
                                 "element_range": [0, 1], "rank": 0}, "kind": "registered-input", "name": "weights", "encoding": {}})
    return {"entries": entries}, committed


def _bnd(n_req: int) -> dict:
    """A binding record that passes binding_check on its own (context ok, coverage ok, population openings skipped): B = n_req."""
    return {"context": {"ok": True}, "coverage": {"ok": True, "expected_n": 1, "bound_n": 1, "missing_n": 0}, "openings_population": {"skipped": "test", "unbound_n": 0}, "population": {"requests": n_req}}


def reader(committed: dict):
    def read(step, name, lo, hi):
        b = committed.get((step, name))
        return None if b is None else b[lo:hi]
    return read


def test_positive_two_requests_engine_order_differs(tmp_path):
    d, vals = make_match(str(tmp_path))
    bmap, committed = make_bmap(vals)
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed))
    assert res["mismatch_n"] == 0 and res["result"] is True and res["complete"], res
    assert res["compared"] == 2 * len(vals) + 2 and res["equal"] == res["compared"]
    assert res["not_compared_by_family"] == {"hidden_stream": 1, "weights": 1}
    assert res["oracle"]["requests"] == {U0: "r0", U1: "r1"} and res["oracle"]["snapshot_steps"] == [0]


def test_a_kv_layers_slots_are_no_match_value(tmp_path):
    d, vals = make_match(str(tmp_path))
    bmap, committed = make_bmap(vals)
    bmap["entries"].append({"identity": {"program_digest": "p" * 64, "request_id": "r0", "step": 0, "invocation": 0, "op_path": "model.layers.0.self_attn.qkv_proj",
                                         "output_member": "slot_mapping", "element_range": [0, ROWS[U0]], "rank": 0}, "kind": "leaf",
                            "name": "model.layers.0.self_attn.qkv_proj/slot_mapping", "element_offset": 0, "encoding": {"dtype": "int64", "shape": [ROWS[U0]]}})
    committed[(0, "model.layers.0.self_attn.qkv_proj/slot_mapping")] = np.arange(ROWS[U0], dtype="<i8").tobytes()
    assert OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed))["mismatch_n"] == 1
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed), own=frozenset({"model.layers.0.self_attn.qkv_proj/slot_mapping"}))
    assert res["mismatch_n"] == 0 and res["result"] is True and res["complete"], res
    assert res["not_compared_by_family"] == {"hidden_stream": 2, "weights": 1}


def test_relabel_across_op_paths_fails_named(tmp_path):
    """qkv_proj never acquired, post_attention_layernorm/0's bytes emitted under qkv_proj's name."""
    d, vals = make_match(str(tmp_path))
    bmap, committed = make_bmap(vals)
    committed[(0, "model.layers.0.self_attn.qkv_proj/0")] = committed[(0, "model.layers.0.post_attention_layernorm/0")]
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed))
    assert res["result"] is False and res["mismatch_n"] == 2                     # both requests' blocks of qkv_proj
    assert {m["identity"]["op_path"] for m in res["mismatch"]} == {"model.layers.0.self_attn.qkv_proj"}
    assert "qkv_proj/0" in res["why"] and "req r1" in res["why"]


def test_same_name_byte_swap_fails_named(tmp_path):
    """/0 <-> /1 bytes exchanged under identical names/shapes."""
    d, vals = make_match(str(tmp_path))
    bmap, committed = make_bmap(vals)
    a, b = committed[(0, "model.layers.0.post_attention_layernorm/0")], committed[(0, "model.layers.0.post_attention_layernorm/1")]
    committed[(0, "model.layers.0.post_attention_layernorm/0")], committed[(0, "model.layers.0.post_attention_layernorm/1")] = b, a
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed))
    assert res["result"] is False and res["mismatch_n"] == 4
    assert {m["identity"]["output_member"] for m in res["mismatch"]} == {"0", "1"}


def test_late_read_one_request_block_fails_named(tmp_path):
    """R0's rows of embed_tokens read after they were overwritten (only r0's block differs) -> r0 named, r1 equal."""
    d, vals = make_match(str(tmp_path))
    bmap, committed = make_bmap(vals)
    data = bytearray(committed[(0, "model.embed_tokens")])
    r0_lo = ROWS[U1] * H * 2                                                     # r0's block follows r1's 3 rows
    data[r0_lo:r0_lo + 2] = bytes([data[r0_lo] ^ 1, data[r0_lo + 1]])
    committed[(0, "model.embed_tokens")] = bytes(data)
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed))
    assert res["result"] is False and res["mismatch_n"] == 1
    m = res["mismatch"][0]
    assert m["identity"]["request_id"] == "r0" and m["identity"]["op_path"] == "model.embed_tokens" and "first element 0" in m["why"]


def test_wrong_sampled_id_fails_named(tmp_path):
    d, vals = make_match(str(tmp_path))
    bmap, committed = make_bmap(vals)
    committed[(0, "runner.sampler/sampled_token_ids")] = np.array([777, 556], dtype="<i8").tobytes()   # r0's id off by one
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed))
    assert res["result"] is False and res["mismatch_n"] == 1
    assert res["mismatch"][0]["identity"]["request_id"] == "r0" and res["mismatch"][0]["oracle_id"] == 555 and res["mismatch"][0]["committed_id"] == 556


def test_step_not_snapshotted_is_partial_never_pass(tmp_path):
    d, vals = make_match(str(tmp_path))
    bmap, committed = make_bmap(vals, steps=(0, 1))
    # step 1 = decode: one row per request; ids = the step-0 samples (U1 first in row order), sampled ids [1, 2] per the log
    bmap["entries"] = [e for e in bmap["entries"] if not (e["identity"]["step"] == 1 and e["identity"]["output_member"] == "in")]
    for lab in ("r1", "r0"):
        bmap["entries"].append({"identity": {"program_digest": "p" * 64, "request_id": lab, "step": 1, "invocation": 0, "op_path": "model.embed_tokens", "output_member": "in",
                                             "element_range": [0, 1], "rank": 0}, "kind": "leaf", "name": "model.embed_tokens/in", "element_offset": 0 if lab == "r1" else 1,
                                "encoding": {"dtype": "int32", "shape": [2]}})
    committed = {**committed, **{(1, k[1]): v for k, v in committed.items() if k[0] == 0}}
    committed[(1, "model.embed_tokens/in")] = np.array([777, 555], dtype="<i4").tobytes()
    committed[(1, "runner.sampler/sampled_token_ids")] = np.array([1, 2], dtype="<i8").tobytes()
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed))
    # step 1's instance outputs are not snapshotted -> not compared, partial; its ids AND sampled ids ARE compared (every step has them)
    assert res["mismatch_n"] == 0, res["mismatch"]
    assert res["steps_not_snapshotted"] == [1] and res["steps_compared"] == [0, 1]
    assert res["compared_by_family"]["input_token_ids"] == 4 and res["compared_by_family"]["sampled_token_ids"] == 4
    assert res["result"] is None and not res["complete"] and "not snapshotted" in res["why"]
    # ... but the request ATTRIBUTION is complete: every (step, request) pair's ids + sampled id compared equal (grade attribution-complete)
    assert res["grade"] == "attribution-complete" and res["attribution"]["complete"] and res["attribution"]["pairs"] == 4
    assert res["attribution"]["unverified"] == [] and res["attribution"]["ambiguous_steps"] == [] and res["attribution"]["requests"] == ["r0", "r1"]


def _two_step_case(tmp_path):
    """The two-step B2 case of test_step_not_snapshotted_is_partial_never_pass: (match dir, bmap, committed)."""
    d, vals = make_match(str(tmp_path))
    bmap, committed = make_bmap(vals, steps=(0, 1))
    bmap["entries"] = [e for e in bmap["entries"] if not (e["identity"]["step"] == 1 and e["identity"]["output_member"] == "in")]
    for lab in ("r1", "r0"):
        bmap["entries"].append({"identity": {"program_digest": "p" * 64, "request_id": lab, "step": 1, "invocation": 0, "op_path": "model.embed_tokens", "output_member": "in",
                                             "element_range": [0, 1], "rank": 0}, "kind": "leaf", "name": "model.embed_tokens/in", "element_offset": 0 if lab == "r1" else 1,
                                "encoding": {"dtype": "int32", "shape": [2]}})
    committed = {**committed, **{(1, k[1]): v for k, v in committed.items() if k[0] == 0}}
    committed[(1, "model.embed_tokens/in")] = np.array([777, 555], dtype="<i4").tobytes()
    committed[(1, "runner.sampler/sampled_token_ids")] = np.array([1, 2], dtype="<i8").tobytes()
    return d, bmap, committed


def test_b2swapeq_relabel_on_the_shared_decode_step_fails_named(tmp_path):
    """b2swapeq: r0 <-> r1 labels exchanged ONLY on the decode step where both have one row (step 1, NOT
    snapshotted by the Match): every byte and every count is unchanged, coverage/composition/openings/root-compare all pass -- the
    per-request ids oracle (tokens.json at the StepStart rows, every step) names the swap: r1's slot holds r0's token and vice versa."""
    d, bmap, committed = _two_step_case(tmp_path)
    for e in bmap["entries"]:                                                   # the collector's request->row map swapped at step 1
        i = e["identity"]
        if i["step"] == 1 and i["request_id"] in ("r0", "r1") and i["output_member"] in ("in", "sampled_token_ids"):
            e["element_offset"] = 1 - int(e["element_offset"])
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed))
    assert res["result"] is False and res["grade"] == "mismatch" and res["mismatch_n"] == 4      # ids + sampled for both requests
    assert {(m["identity"]["step"], m["identity"]["request_id"]) for m in res["mismatch"]} == {(1, "r0"), (1, "r1")}
    assert {m["family"] for m in res["mismatch"]} == {"input_token_ids", "sampled_token_ids"}
    assert res["attribution"]["complete"] is False
    from verity_vllm.check.commit_pass import commit_verdict
    commit = {"extra": {"expected_tensors": 3, "committed_tensors": 3, "errors_n": 0}, "openings": {"n": 64, "verified": 64},
              "binding": _bnd(2), "oracle_compare": res,
              "value_correspondence": {"method": "oracle-compare(partial) + root-compare-vs-independent-run", "result": False, "oracle_compare": {"why": res["why"]}}}
    ok, why = commit_verdict(commit, True, min_openings=64)
    assert not ok and "step 1" in why and "req r" in why


def test_b2_attribution_ambiguous_when_two_requests_share_ids_on_an_unsnapshotted_step(tmp_path):
    """Both requests sampled the SAME id at step 0 and so feed the same id at step 1 (not snapshotted): a swap there is undetectable by the
    ids oracle -> the step is reported ambiguous, the grade is partial (never attribution-complete), the B>=2 verdict is not PASS."""
    d, vals = make_match(str(tmp_path), sampled=[777, 777])
    bmap, committed = make_bmap(vals, steps=(0, 1))
    bmap["entries"] = [e for e in bmap["entries"] if not (e["identity"]["step"] == 1 and e["identity"]["output_member"] == "in")]
    for lab in ("r1", "r0"):
        bmap["entries"].append({"identity": {"program_digest": "p" * 64, "request_id": lab, "step": 1, "invocation": 0, "op_path": "model.embed_tokens", "output_member": "in",
                                             "element_range": [0, 1], "rank": 0}, "kind": "leaf", "name": "model.embed_tokens/in", "element_offset": 0 if lab == "r1" else 1,
                                "encoding": {"dtype": "int32", "shape": [2]}})
    committed = {**committed, **{(1, k[1]): v for k, v in committed.items() if k[0] == 0}}
    committed[(0, "runner.sampler/sampled_token_ids")] = np.array([777, 777], dtype="<i8").tobytes()
    committed[(1, "model.embed_tokens/in")] = np.array([777, 777], dtype="<i4").tobytes()
    committed[(1, "runner.sampler/sampled_token_ids")] = np.array([1, 1], dtype="<i8").tobytes()
    # the log's step-1 samples are [1, 2]: make them equal too so the step is fully ambiguous
    import gzip
    p = os.path.join(d, "capture", "log.jsonl.gz")
    lines = gzip.open(p, "rt").read().splitlines()
    lines = [json.dumps({"$e": "SampleEnd", "step": 1, "sampled_token_ids": [1, 1]}) if '"step": 1' in ln and "SampleEnd" in ln else ln for ln in lines]
    gzip.open(p, "wt").write("\n".join(lines) + "\n")
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed))
    assert res["mismatch_n"] == 0 and res["result"] is None
    assert res["attribution"]["ambiguous_steps"] == [1] and res["attribution"]["complete"] is False and res["grade"] == "partial"
    assert "ambiguous at steps [1]" in res["why"]
    from verity_vllm.check.commit_pass import commit_verdict
    commit = {"extra": {"expected_tensors": 3, "committed_tensors": 3, "errors_n": 0}, "openings": {"n": 64, "verified": 64},
              "binding": _bnd(2), "oracle_compare": res, "value_correspondence": {"method": "oracle-compare(partial)", "result": None}}
    ok, why = commit_verdict(commit, True, min_openings=64)
    assert not ok and why.startswith("C2 of record not established")          # the replay + linkage is the attribution check of record


def test_verdict_b2_requires_a_live_oracle_compare():
    """At B>=2 a Commit without the live oracle-compare (no --match-dir) is FAIL named, whatever root-compare says; at B=1 the
    root-compare with provenance remains acceptable."""
    from verity_vllm.check.commit_pass import commit_verdict
    base = {"extra": {"expected_tensors": 3, "committed_tensors": 3, "errors_n": 0}, "openings": {"n": 64, "verified": 64},
            "value_correspondence": {"method": "root-compare-vs-independent-run", "result": True}}
    # the C2 of record is the sampled replay + boundary linkage at EVERY B; a row record without it is
    # FAIL named (at B=1 too: a root-compare alone was an unsound PASS), and a complete oracle attribution no longer substitutes
    ok, why = commit_verdict({**base, "binding": _bnd(2)}, True, min_openings=64)
    assert not ok and why.startswith("C2 of record not established") and "no --program-dir" in why
    ok, why = commit_verdict({**base, "binding": _bnd(1)}, True, min_openings=64)
    assert not ok and why.startswith("C2 of record not established")
    oc = {"result": None, "compared": 0, "why": "oracle compare not run: FileNotFoundError(...)"}
    ok, why = commit_verdict({**base, "binding": _bnd(4), "oracle_compare": oc}, True, min_openings=64)
    assert not ok and why.startswith("C2 of record not established")
    oc = {"result": None, "compared": 100, "grade": "attribution-complete", "attribution": {"complete": True, "unverified_n": 0, "ambiguous_steps": []}}
    ok, why = commit_verdict({**base, "binding": _bnd(2), "oracle_compare": oc}, True, min_openings=64)
    assert not ok and why.startswith("C2 of record not established")
    # with the C2 of record present the oracle stays a diagnostic: PASS at B2 without any Match record
    c2 = {"sampled_replay": {"result": True, "grade": "complete"}, "boundary_linkage": {"result": True, "pairs": 6, "linked": 6},
          "weights_pin": {"result": True, "compared": 98}}                                   # the checkpoint pin is part of the record
    ok_, why_ = commit_verdict({**base, **c2, "binding": _bnd(2)}, True, min_openings=64)
    assert ok_ is True and why_.startswith("PASS (B=2: per-(step,request) attribution not verified on this record"), why_   # PASS, gap named


def test_verdict_gates_on_oracle_mismatch():
    from verity_vllm.check.commit_pass import commit_verdict
    commit = {"extra": {"expected_tensors": 3, "committed_tensors": 3, "errors_n": 0}, "openings": {"n": 64, "verified": 64},
              "value_correspondence": {"method": "oracle-compare + offline-suite", "result": False,
                                       "oracle_compare": {"result": False, "why": "C2 oracle mismatch: 2 identities differ from the Match oracle (first: step 0 model.layers.0.self_attn.qkv_proj/0 inv 0 req r1 range [0, 12] -- bytes differ)"}}}
    ok, why = commit_verdict(commit, True, min_openings=64)
    assert not ok and why.startswith("C2 oracle mismatch") and "qkv_proj/0" in why
    commit["value_correspondence"] = {"method": "oracle-compare(partial)", "result": None}
    assert commit_verdict(commit, True, min_openings=64) == (True, "PASS")
    commit["value_correspondence"] = {"method": "oracle-compare", "result": True}
    assert commit_verdict(commit, True, min_openings=64) == (True, "PASS")


def test_missing_oracle_files_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        OC.MatchOracle(str(tmp_path))


def _exploded_norm_match(tmp: str, *, permute: bool = False):
    """Dense's Gemma fused norm as the fold records it: ONE module hint, the forward exploded into per-op instances per row --
    Square (f32, interior), Narrow #1 (the new residual = member 1), Narrow #2 (the normed row = member 0): the two producers of the
    module's members share the family NarrowF32ToBf16_v1 and differ only by Program/event order.  Each narrowing's first operand view is
    the OUTPUT view of the instance feeding it (residual <- AddWidenedBf16_v1, normed <- MulVecF32_v1), as derive_step records.
    `permute`: the fold emitted the two independent narrowings in the other order (normed first) -- a legal dag-criterion
    fold; Program order (hence the manifest's confirmed ordinal) is unchanged."""
    d, vals = make_match(tmp)
    n = sum(ROWS.values())
    op = "model.layers.0.pre_feedforward_layernorm"
    resid, normed, square = _bf16(n, 7), _bf16(n, 8), np.arange(n * H, dtype="<f4").tobytes()
    addw, mulv = (np.arange(n * H, dtype="<f4") + 1000).tobytes(), (np.arange(n * H, dtype="<f4") + 2000).tobytes()
    ev_resid, ev_normed = (82, 81) if permute else (81, 82)
    snap = json.load(open(os.path.join(d, "capture", "snapshots.json")))
    for alloc, data, ev, dt in ((23, addw, 78, "float32"), (24, mulv, 79, "float32"), (20, square, 80, "float32"),
                                (21, resid, ev_resid, "bfloat16"), (22, normed, ev_normed, "bfloat16")):
        sha = hashlib.sha256(data).hexdigest()
        fn = f"values/0/{sha[:16]}.bin"
        open(os.path.join(d, "capture", fn), "wb").write(data)
        snap["entries"].append({"event": ev, "path": "outs", "view": {"$v": [alloc, 0, [n, H], [H, 1], dt, "cuda:0"]}, "dtype": dt, "shape": [n, H],
                                "strides": [H, 1], "byte_offset": 0, "nbytes": len(data), "layout": "spanned-storage-bytes", "sha256": sha, "file": fn, "step": 0})
    json.dump(snap, open(os.path.join(d, "capture", "snapshots.json"), "w"))
    with open(os.path.join(d, "instances.jsonl"), "a") as f:
        row = 0
        for u in [U1, U0]:
            for pos in range(ROWS[u]):
                for ev, alloc, spec, dt, src in ((78, 23, "AddWidenedBf16_v1{N=8}", "float32", None), (79, 24, "MulVecF32_v1{N=8}", "float32", None),
                                                 (80, 20, "SquareF32_v1{N=8}", "float32", None), (ev_resid, 21, "NarrowF32ToBf16_v1{N=8}", "bfloat16", 23),
                                                 (ev_normed, 22, "NarrowF32ToBf16_v1{N=8}", "bfloat16", 24)):
                    opv = [] if src is None else [{"alloc": src, "offset": row * H, "shape": [H], "strides": [1], "dtype": "float32", "device": "cuda:0"}]
                    f.write(json.dumps({"node": 0, "step": 0, "event": ev, "pattern": "gemma2.x", "spec": spec, "row": 0, "rows": 1, "name": f"t={pos}/norm",
                                        "hint": op, "request": u, "position": pos, "operand_views": opv,
                                        "output_views": [{"alloc": alloc, "offset": row * H, "shape": [H], "strides": [1], "dtype": dt, "device": "cuda:0"}]}) + "\n")
                row += 1
    entries, committed = [], {}
    off = 0
    for u, lab in ((U1, "r1"), (U0, "r0")):
        for mem in ("0", "1"):
            entries.append({"identity": {"program_digest": "p" * 64, "request_id": lab, "step": 0, "invocation": 0, "op_path": op, "output_member": mem,
                                         "element_range": [0, ROWS[u] * H], "rank": 0}, "kind": "leaf", "name": f"{op}/{mem}", "element_offset": off * H,
                            "encoding": {"dtype": "bfloat16", "shape": [n, H]}})
        off += ROWS[u]
    committed[(0, f"{op}/0")], committed[(0, f"{op}/1")] = normed, resid
    return d, {"entries": entries}, committed, op


def _biased_linear_match(tmp: str):
    """Qwen2.5's biased qkv_proj as the fold records it: per row ONE Gemm_v1 instance (the batch-invariant matmul, output =
    the PRE-bias GEMM alloc; registry Gemm_v1 has no epilogue) and ONE BiasAdd_v1 instance (aten.add(bf16, bf16), output = the tensor
    the module RETURNS), both hinted to the module.  The collector binds member "0" at the module post-hook = the post-bias bytes."""
    d, vals = make_match(tmp)
    n = sum(ROWS.values())
    op = "model.layers.0.self_attn.qkv_proj"
    pre, post = _bf16(n, 11), _bf16(n, 12)
    snap = json.load(open(os.path.join(d, "capture", "snapshots.json")))
    for alloc, data, ev in ((30, pre, 90), (31, post, 91)):
        sha = hashlib.sha256(data).hexdigest()
        fn = f"values/0/{sha[:16]}.bin"
        open(os.path.join(d, "capture", fn), "wb").write(data)
        snap["entries"].append({"event": ev, "path": "outs", "view": {"$v": [alloc, 0, [n, H], [H, 1], "bfloat16", "cuda:0"]}, "dtype": "bfloat16",
                                "shape": [n, H], "strides": [H, 1], "byte_offset": 0, "nbytes": len(data), "layout": "spanned-storage-bytes",
                                "sha256": sha, "file": fn, "step": 0})
    json.dump(snap, open(os.path.join(d, "capture", "snapshots.json"), "w"))
    with open(os.path.join(d, "instances.jsonl"), "a") as f:
        row = 0
        for u in [U1, U0]:
            for pos in range(ROWS[u]):
                for ev, alloc, spec, pat, src in ((90, 30, "Gemm_v1{K=8,N=8}", "gemm.matmul_kernel_persistent", None),
                                                  (91, 31, "BiasAdd_v1{N=8}", "aten.add", 30)):
                    opv = [] if src is None else [{"alloc": src, "offset": row * H, "shape": [H], "strides": [1], "dtype": "bfloat16", "device": "cuda:0"}]
                    f.write(json.dumps({"node": 0, "step": 0, "event": ev, "pattern": pat, "spec": spec, "row": 0, "rows": 1, "name": f"t={pos}/qkv",
                                        "hint": op, "request": u, "position": pos, "operand_views": opv,
                                        "output_views": [{"alloc": alloc, "offset": row * H, "shape": [H], "strides": [1], "dtype": "bfloat16", "device": "cuda:0"}]}) + "\n")
                row += 1
    entries, committed = [], {}
    off = 0
    for u, lab in ((U1, "r1"), (U0, "r0")):
        entries.append({"identity": {"program_digest": "p" * 64, "request_id": lab, "step": 0, "invocation": 0, "op_path": op, "output_member": "0",
                                     "element_range": [0, ROWS[u] * H], "rank": 0}, "kind": "leaf", "name": f"{op}/0", "element_offset": off * H,
                        "encoding": {"dtype": "bfloat16", "shape": [n, H]}})
        off += ROWS[u]
    committed[(0, f"{op}/0")] = post
    return d, {"entries": entries}, committed, op, pre


def test_biased_linear_member_is_read_at_the_bias_add_producer_not_the_gemm(tmp_path):
    """The manifest's producer for a biased linear's member "0" must be the BiasAdd row.  Required at the Gemm (the
    earlier rule) the oracle reads the PRE-bias GEMM output and every row of every (step, request) differs from the committed post-bias
    tensor -- exactly the Qwen2.5 finding (28 layers x 130 pairs, qkv_proj/0 only) -- a false FAIL that would be a false PASS had the
    collector bound the GEMM alloc; at the BiasAdd it is the same tensor and the compare is a check (swapped committed bytes still fail)."""
    d, bmap, committed, op, pre = _biased_linear_match(str(tmp_path))
    res_gemm = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed), producers={(op, "0"): {"spec": "Gemm_v1", "ordinal": None, "operand": None}})
    assert res_gemm["result"] is False and res_gemm["mismatch_n"] == 2, res_gemm
    assert all(m["identity"]["op_path"] == op and "bytes differ" in m["why"] for m in res_gemm["mismatch"]), res_gemm["mismatch"]
    res_bias = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed), producers={(op, "0"): {"spec": "BiasAdd_v1", "ordinal": None, "operand": None}})
    assert res_bias["compared"] == 2 and res_bias["equal"] == 2 and res_bias["mismatch_n"] == 0, res_bias
    # still a check, not a guess: a collector that committed the pre-bias GEMM alloc is named against the BiasAdd producer ...
    res_wrong = OC.oracle_compare(OC.MatchOracle(d), bmap, reader({(0, f"{op}/0"): pre}), producers={(op, "0"): {"spec": "BiasAdd_v1", "ordinal": None, "operand": None}})
    assert res_wrong["result"] is False and res_wrong["mismatch_n"] == 2, res_wrong
    # ... and without a producer the two instances per row are not silently merged into one member
    res_none = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed))
    assert res_none["result"] is not True, res_none


def test_exploded_module_is_read_at_the_manifest_producer_with_ordinal(tmp_path):
    d, bmap, committed, op = _exploded_norm_match(str(tmp_path))
    prod = {(op, "0"): {"spec": "NarrowF32ToBf16_v1", "ordinal": 1}, (op, "1"): {"spec": "NarrowF32ToBf16_v1", "ordinal": 0}}
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed), producers=prod)
    assert res["compared"] == 4 and res["equal"] == 4 and res["mismatch_n"] == 0, res
    # the same members with the residual/normed bytes exchanged under their names (on a fused norm): named, not re-guessed
    committed[(0, f"{op}/0")], committed[(0, f"{op}/1")] = committed[(0, f"{op}/1")], committed[(0, f"{op}/0")]
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed), producers=prod)
    assert res["result"] is False and res["mismatch_n"] == 4 and {m["identity"]["output_member"] for m in res["mismatch"]} == {"0", "1"}


def test_permuted_fold_order_is_read_right_by_producer_operand_and_wrong_by_ordinal_alone(tmp_path):
    """The two narrowings are independent, so a dag-criterion fold may emit them permuted while the
    Program order (the manifest's confirmed ordinal) is unchanged.  Ordinal alone reads the OTHER narrowing's bytes -> a false C2 FAIL
    (would be a false PASS had the collector made the same swap); the operand selector resolves each member by the instance that FEEDS
    it (storage identity of its first operand), so it is right in either emission order and refuses by name when it cannot decide."""
    for permute in (False, True):
        d, bmap, committed, op = _exploded_norm_match(str(tmp_path / f"p{int(permute)}"), permute=permute)
        prod_ord = {(op, "0"): {"spec": "NarrowF32ToBf16_v1", "ordinal": 1}, (op, "1"): {"spec": "NarrowF32ToBf16_v1", "ordinal": 0}}
        prod_opd = {k: dict(v, operand="MulVecF32_v1" if k[1] == "0" else "AddWidenedBf16_v1") for k, v in prod_ord.items()}
        res_ord = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed), producers=prod_ord)
        res_opd = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed), producers=prod_opd)
        assert res_opd["compared"] == 4 and res_opd["equal"] == 4 and res_opd["mismatch_n"] == 0, (permute, res_opd)
        if permute:
            assert res_ord["result"] is False and res_ord["mismatch_n"] == 4, res_ord         # the gap: event order != Program order
        else:
            assert res_ord["compared"] == 4 and res_ord["equal"] == 4, res_ord
        # the operand path stays a check, not a guess: swapped committed bytes are named in either emission order ...
        committed[(0, f"{op}/0")], committed[(0, f"{op}/1")] = committed[(0, f"{op}/1")], committed[(0, f"{op}/0")]
        res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed), producers=prod_opd)
        assert res["result"] is False and res["mismatch_n"] == 4 and {m["identity"]["output_member"] for m in res["mismatch"]} == {"0", "1"}
        # ... and an operand family no narrowing consumes (or one both consume) is refused by name, never resolved by the ordinal
        res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed), producers={k: dict(v, operand="SquareF32_v1") for k, v in prod_ord.items()})
        assert res["compared"] == 0 and res["result"] is None and any(k.startswith("producer_operand SquareF32_v1: 0 of 2") for k in res["not_compared"]), res["not_compared"]


def test_exploded_module_without_ordinal_is_ambiguous_never_guessed(tmp_path):
    d, bmap, committed, op = _exploded_norm_match(str(tmp_path))
    prod = {(op, "0"): {"spec": "NarrowF32ToBf16_v1", "ordinal": None}, (op, "1"): {"spec": "NarrowF32ToBf16_v1", "ordinal": None}}
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed), producers=prod)
    assert res["compared"] == 0 and res["mismatch_n"] == 0 and res["result"] is None
    whys = [k for k in res["not_compared"] if k.startswith("ambiguous producer")]
    assert len(whys) == 2 and all("2 instances of NarrowF32ToBf16_v1" in w and "producer_ordinal" in w for w in whys)
    assert sum(res["not_compared"][w] for w in whys) == 4                       # both members x both requests, named per member
    # a producer the fold never recorded under the module: named too
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed), producers={(op, "0"): {"spec": "RMSNorm_v1", "ordinal": None}, (op, "1"): {"spec": "RMSNorm_v1", "ordinal": None}})
    assert any(k.startswith("no fold instance of producer RMSNorm_v1") for k in res["not_compared"])
    # without a producer map the exploded module concatenates every interior op's view: a SIZE mismatch, named (never silently equal)
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed))
    assert res["result"] is False and all(m["why"].startswith("size:") for m in res["mismatch"])


def test_producers_of_reads_spec_and_ordinal_from_manifest_rows():
    man = {"identities": [{"family": "instance_outputs", "op_path": "a.b", "output_member": "0", "spec": "Gemm_v1{K=1,N=2}"},
                          {"family": "instance_outputs", "op_path": "a.n", "output_member": "1", "spec": "NarrowF32ToBf16_v1", "producer_ordinal": 0, "producer_operand": "AddWidenedBf16_v1"},
                          {"family": "instance_outputs", "op_path": "a.n", "output_member": "0", "spec": "NarrowF32ToBf16_v1", "producer_ordinal": 1},
                          {"family": "weights", "op_path": "weights", "output_member": "w", "spec": None},
                          {"family": "instance_outputs", "op_path": "a.c", "output_member": "0", "spec": "X_v1"},
                          {"family": "instance_outputs", "op_path": "a.c", "output_member": "0", "spec": "Y_v1"}]}
    p = OC.producers_of(man)
    assert p[("a.b", "0")] == {"spec": "Gemm_v1", "ordinal": None, "operand": None} and p[("a.n", "1")]["ordinal"] == 0 and p[("a.n", "0")]["ordinal"] == 1
    assert p[("a.n", "1")]["operand"] == "AddWidenedBf16_v1" and p[("a.n", "0")]["operand"] is None
    assert ("weights", "w") not in p and p[("a.c", "0")].get("conflict") is True and p[("a.c", "0")]["spec"] is None


def test_foreign_oracle_is_refused_by_name_before_any_value_compare(tmp_path):
    """A Match record of ANOTHER workload given as --match-dir: refused by name (request ids / prompt lengths / caps),
    nothing compared, result False 'foreign oracle' -- never a value mismatch that happens to point the right way."""
    d, vals = make_match(str(tmp_path))
    bmap, committed = make_bmap(vals)
    own = {"requests": [{"request_id": U0, "index": 0, "prompt_len": 5, "max_tokens": 1}, {"request_id": U1, "index": 1, "prompt_len": 3, "max_tokens": 1}]}
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed), workload=own)
    assert res["result"] is True and res["workload_check"]["ok"] is True and res["workload_check"]["compared"] == 2
    # another row's workload (one request, another uuid)
    foreign = {"requests": [{"request_id": "cccccccc-2222-2222-2222-222222222222", "index": 0, "prompt_len": 5, "max_tokens": 1}]}
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed), workload=foreign)
    assert res["result"] is False and res["grade"] == "foreign-oracle" and res["compared"] == 0 and res["mismatch_n"] == 0
    assert res["why"].startswith("foreign oracle") and "request ids differ" in res["why"]
    # same uuids, another declared shape (prompt_len) or a swapped labelling: named too
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed),
                            workload={"requests": [{"request_id": U0, "index": 0, "prompt_len": 7, "max_tokens": 1}, {"request_id": U1, "index": 1, "prompt_len": 3, "max_tokens": 1}]})
    assert res["result"] is False and "oracle prompt has 5 ids != declared prompt_len 7" in res["why"]
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed),
                            workload={"requests": [{"request_id": U1, "index": 0, "prompt_len": 3, "max_tokens": 1}, {"request_id": U0, "index": 1, "prompt_len": 5, "max_tokens": 1}]})
    assert res["result"] is False and "label r0 in the oracle != r1 declared" in res["why"]
    # no workload passed: not checked, said so, values still compared
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed))
    assert res["result"] is True and res["workload_check"]["ok"] is None
    # the verdict names it
    from verity_vllm.check.commit_pass import commit_verdict
    commit = {"extra": {"expected_tensors": 3, "committed_tensors": 3, "errors_n": 0}, "openings": {"n": 64, "verified": 64}, "binding": _bnd(2),
              "value_correspondence": {"method": "oracle-compare(partial) + offline-suite", "result": False, "oracle_compare": {"result": False, "why": "foreign oracle: x"}},
              "oracle_compare": {"result": False, "why": "foreign oracle: x", "compared": 0}}
    ok, why = commit_verdict(commit, True, min_openings=64)
    assert not ok and why.startswith("foreign oracle")


def test_verdict_names_the_failing_component_not_the_oracle_partial_note():
    """b2flip1_s5 + --expect-root: the oracle ran partial (result None, why = its coverage note) and the ROOT compare failed;
    first_fail_reason must quote the root-compare cause, never the oracle's partial note.  [R2] components carry their own result/reference."""
    from verity_vllm.check.commit_pass import commit_verdict
    partial_note = "attribution-complete: steps [2, 3, 4, 5] not snapshotted by the Match (MATCH_SNAP_STEPS); ids + sampled ids verified on every step"
    vc = {"method": "oracle-compare(partial) + offline-suite + root-compare-vs-independent-run (--expect-root)", "result": False,
          "why": "root_compare: run root 0123456789abcdef != expected root fedcba9876543210 (--expect-root)",
          "oracle_compare": {"result": None, "why": partial_note, "grade": "attribution-complete", "compared": 848, "mismatch_n": 0},
          "components": {"root_compare": {"result": False, "why": "run root 0123456789abcdef != expected root fedcba9876543210 (--expect-root)",
                                          "reference": "--expect-root (caller-pinned root; independent only if produced by another tree/pod)"},
                         "oracle_compare": {"result": None, "why": partial_note, "grade": "attribution-complete", "reference": "live Match oracle of this attempt (--match-dir)"}}}
    commit = {"extra": {"expected_tensors": 3, "committed_tensors": 3, "errors_n": 0}, "openings": {"n": 64, "verified": 64}, "binding": _bnd(1),
              "value_correspondence": vc, "oracle_compare": {"result": None, "why": partial_note, "compared": 848, "attribution": {"complete": True}}}
    ok, why = commit_verdict(commit, True, min_openings=64)
    assert not ok and why.startswith("root_compare: run root 0123456789abcdef != expected root") and "not snapshotted" not in why
    # the oracle's own failure still wins the text when IT failed
    vc["oracle_compare"] = {"result": False, "why": "C2 oracle mismatch: 1 identity differs (first: step 1 model.layers.0.mlp.down_proj/0 req r0)"}
    ok, why = commit_verdict(commit, True, min_openings=64)
    assert not ok and why.startswith("C2 oracle mismatch")
    # a replay failure recorded as a component is named as such
    vc["oracle_compare"] = {"result": None, "why": partial_note}
    vc["components"] = {"replay": {"result": False, "why": "replay root != committed root", "reference": "replay"}}
    ok, why = commit_verdict(commit, True, min_openings=64)
    assert not ok and why == "replay: replay root != committed root"


def test_f_r16_24_the_oracle_provenance_carries_the_tie_facts_of_the_match_record_it_read(tmp_path):
    """The oracle is 'this attempt's Match' by FACT, not by presence: provenance.tie carries sha256 of the Match
    record's fold_summary.json / instances.jsonl / RunHeader line so the finalizer ties it to the Match stage of record without reading the
    fold; a missing file is None, never a placeholder; another Match dir (same workload, different fold) has different digests."""
    import gzip
    import hashlib
    import os
    d, _vals = make_match(str(tmp_path))
    pv = OC.MatchOracle(d).provenance()
    tie = pv["tie"]
    assert tie["fold_summary_sha256"] is None                                      # make_match writes no fold_summary.json: named absent
    assert tie["instances_sha256"] == hashlib.sha256(open(os.path.join(d, "instances.jsonl"), "rb").read()).hexdigest()
    with gzip.open(os.path.join(d, "capture", "log.jsonl.gz"), "rb") as f:
        assert tie["run_header_sha256"] == hashlib.sha256(f.readline()).hexdigest()
    open(os.path.join(d, "fold_summary.json"), "w").write('{"digest": "abc"}')
    tie2 = OC.oracle_tie_facts(d)
    assert tie2["fold_summary_sha256"] == hashlib.sha256(b'{"digest": "abc"}').hexdigest() and tie2["instances_sha256"] == tie["instances_sha256"]
    # a second Match record of the same workload is another oracle: its tie differs (the fold digest / header line are its own)
    d2, _ = make_match(str(tmp_path / "other"))
    open(os.path.join(d2, "fold_summary.json"), "w").write('{"digest": "def"}')
    assert OC.oracle_tie_facts(d2)["fold_summary_sha256"] != tie2["fold_summary_sha256"]
    assert OC.oracle_tie_facts(str(tmp_path / "nowhere")) == {"fold_summary_sha256": None, "instances_sha256": None, "run_header_sha256": None}


def test_an_unreadable_run_header_is_named_in_the_tie_facts_not_passed_off_as_an_absent_log(tmp_path):
    """A capture/log.jsonl.gz that is there but isn't gzip has no RunHeader digest; the tie facts say it was unreadable and why, so the
    record doesn't read as a Match stage that wrote no log."""
    import os
    d, _vals = make_match(str(tmp_path))
    open(os.path.join(d, "capture", "log.jsonl.gz"), "wb").write(b'{"$e": "RunHeader"}\n')
    tie = OC.oracle_tie_facts(d)
    assert tie["run_header_sha256"] is None
    assert tie["run_header_unreadable"].startswith("unreadable: capture/log.jsonl.gz (BadGzipFile: ")
    assert "run_header_unreadable" not in OC.oracle_tie_facts(str(tmp_path / "nowhere"))


# ---- the ORACLE side: a shared module's invocation k is compared at ITS Program op -------------------------------------------------------
ROT = "model.layers.0.self_attn.rotary_emb"


def _shared_rotary_match(tmp: str, n_layers: int = 3):
    """One request (5 rows), one step; the Match's fold has the rotary q output under the PER-LAYER Program paths
    (`model.layers.k.self_attn.rotary_emb`, hint `.../rope.q`), each layer's value different.  Returns (dir, {layer: bytes})."""
    d = os.path.join(tmp, "match_rot")
    os.makedirs(os.path.join(d, "capture", "values", "0"), exist_ok=True)
    n = ROWS[U0]
    vals = {k: _bf16(n, 40 + k) for k in range(n_layers)}
    entries, lines = [], []
    for k, data in vals.items():
        sha = hashlib.sha256(data).hexdigest()
        fn = f"values/0/{sha[:16]}.bin"
        open(os.path.join(d, "capture", fn), "wb").write(data)
        entries.append({"event": 100 + k, "path": "outs[0]", "view": {"$v": [20 + k, 0, [n, H], [H, 1], "bfloat16", "cuda:0"]}, "dtype": "bfloat16",
                        "shape": [n, H], "strides": [H, 1], "byte_offset": 0, "nbytes": len(data), "layout": "spanned-storage-bytes", "sha256": sha, "file": fn, "step": 0})
        op = f"model.layers.{k}.self_attn.rotary_emb"
        for pos in range(n):
            lines.append({"node": 0, "step": 0, "event": 100 + k, "pattern": "x", "spec": "RoPE_v1{}", "row": pos, "rows": n, "name": f"t={pos}/rotary_emb",
                          "hint": f"{op}/rope.q", "request": U0, "position": pos, "operand_views": [],
                          "output_views": [{"alloc": 20 + k, "offset": pos * H, "shape": [H], "strides": [1], "dtype": "bfloat16", "device": "cuda:0"}]})
    json.dump({"schema": "verity-capture/snapshots/v1", "steps": [0], "entries": entries}, open(os.path.join(d, "capture", "snapshots.json"), "w"))
    with open(os.path.join(d, "instances.jsonl"), "w") as f:
        f.write(json.dumps({"schema": "verity-gen/instances/v1", "digest": "f" * 64, "n_instances": len(lines),
                            "requests": [{"index": 0, "request_id": U0, "prompt_len": n, "served_samples": 1, "max_tokens": 1}]}) + "\n")
        for rec in lines:
            f.write(json.dumps(rec) + "\n")
    with gzip.open(os.path.join(d, "capture", "log.jsonl.gz"), "wt") as f:
        f.write(json.dumps({"$e": "RunHeader", "schema": "verity-capture/log/v1", "created_utc": "2026-09-20T21:00:00Z", "profile_id": "abc", "profile": {}}) + "\n")
        f.write(json.dumps({"$e": "StepStart", "step": 0, "rows": [[U0, p] for p in range(n)]}) + "\n")
        f.write(json.dumps({"$e": "SampleEnd", "step": 0, "sampled_token_ids": [5]}) + "\n")
    json.dump({"requests": [{"request_id": U0, "prompt_token_ids": [1, 2, 3, 4, 5], "generated_token_ids": [5]}]}, open(os.path.join(d, "capture", "tokens.json"), "w"))
    return d, vals


def _shared_rotary_bmap(vals: dict, *, tensor_bytes: dict | None = None):
    """The collector's shape: every invocation is a meta NAMED `<ROT>/query` (ordinal 10+k), bound under the canonical path at invocation k;
    the reader is the PRODUCTION OpenedReader over those metas, committed on the per-tensor host path (ordinals 0-9: other tensors)."""
    from tests.check.opened import commit_host_steps
    n = ROWS[U0]
    entries = []
    tensors = [(f"model.other.{i}", b"\0" * 4) for i in range(10)]
    for k in vals:
        entries.append({"identity": {"program_digest": "p" * 64, "request_id": "r0", "step": 0, "invocation": k, "op_path": ROT, "output_member": "query",
                                     "element_range": [0, n * H], "rank": 0, "family": "instance_outputs"}, "kind": "leaf", "name": f"{ROT}/query", "tensor_ordinal": 10 + k,
                        "element_offset": 0, "encoding": {"dtype": "bfloat16", "shape": [n, H]}})
        tensors.append((f"{ROT}/query", (tensor_bytes or vals)[k], 10 + k))
    rd = OC.OpenedReader(*commit_host_steps({0: tensors}))
    aliases = {f"model.layers.{k}.self_attn.rotary_emb": [ROT, k] for k in vals if k > 0}
    return {"entries": entries}, rd, aliases


def test_f_moe_r17_06_shared_invocations_compare_at_their_program_op(tmp_path):
    d, vals = _shared_rotary_match(str(tmp_path))
    bmap, rd, aliases = _shared_rotary_bmap(vals)
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, rd, aliases=aliases)
    assert res["compared"] == 3 and res["equal"] == 3 and res["mismatch_n"] == 0 and res["result"] is True, res
    assert res["shared_invocations"]["compared_at_program_op"] == 2 and res["shared_invocations"]["not_compared"] == 0


def test_f_moe_r17_06_without_the_alias_table_the_exact_bytes_would_be_a_false_mismatch_so_they_are_not_compared_by_name(tmp_path):
    """The fixed reader gives invocation k its OWN bytes; read at the canonical op they would MISMATCH layer 0's oracle value for every
    k >= 1 -- a false alarm.  Without the table they are NOT COMPARED, named; invocation 0 compares at its own op."""
    d, vals = _shared_rotary_match(str(tmp_path))
    bmap, rd, aliases = _shared_rotary_bmap(vals)
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, rd)
    assert res["compared"] == 1 and res["equal"] == 1 and res["mismatch_n"] == 0 and res["complete"] is False
    assert any("no Program path in the alias table" in k and "not compared" in k for k in res["not_compared"]), res["not_compared"]


def test_f_moe_r17_06_the_collapse_is_a_named_mismatch_at_the_invocation_not_a_vacuous_equal(tmp_path):
    """Invocation 2's meta holds layer 0's bytes (what the by-name read returned for every invocation): at the Program op the compare says
    MISMATCH at invocation 2 by name -- the pre-fix path compared layer 0 with layer 0 and said equal."""
    d, vals = _shared_rotary_match(str(tmp_path))
    bad = dict(vals)
    bad[2] = vals[0]
    bmap, rd, aliases = _shared_rotary_bmap(vals, tensor_bytes=bad)
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, rd, aliases=aliases)
    assert res["compared"] == 3 and res["mismatch_n"] == 1 and res["result"] is False
    assert res["mismatch"][0]["identity"]["invocation"] == 2 and res["mismatch"][0]["identity"]["op_path"] == ROT


def test_revb_n_2_14_1_no_oracle_families_are_said_per_query_family_in_identities_not_entries(tmp_path):
    """`not_compared = {no oracle family: hidden_stream: N}` lumped two Q(P) families (moe_block_stream +
    fa2_hidden_m1_stream) under the oracle's class and counted binding ENTRIES.  The record now says it per Q(P) family and in IDENTITIES
    (the manifest's rule) with the entry count beside it; the oracle's own classes (`not_compared_by_family`) are unchanged for the verdict."""
    d, vals = make_match(str(tmp_path))
    bmap, committed = make_bmap(vals)
    # three more M1 entries for the same (layer, engine step) identity (members P / RP / O) and two moe plane entries -> 4 M1 entries, 1 identity
    for member in ("P", "RP", "O"):
        bmap["entries"].append({"identity": {"program_digest": "p" * 64, "request_id": "*", "step": 0, "invocation": 0, "op_path": "fa2.m1/L0/g32x8x1",
                                             "output_member": member, "element_range": [0, 8], "rank": 0}, "kind": "leaf",
                                "name": f"fa2.m1/L0/s0/g32x8x1/{member}", "element_offset": 0, "encoding": {"dtype": "int32", "shape": [8]}})
    for plane in ("router_w", "up"):
        bmap["entries"].append({"identity": {"program_digest": "p" * 64, "request_id": "r0", "step": 0, "invocation": 0, "op_path": "moe/L0",
                                             "output_member": plane, "element_range": [0, 8], "rank": 0}, "kind": "leaf",
                                "name": f"moe/L0/{plane}", "element_offset": 0, "encoding": {"dtype": "float32", "shape": [8]}})
    manifest = {"identities": [{"family": "fa2_hidden_m1_stream"}] + [{"family": "moe_block_stream"}] * 2 + [{"family": "weights"}] * 3
                + [{"family": "instance_outputs"}] * 5}
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed), query_families=OC.query_families_of(manifest))
    assert res["result"] is True and res["complete"]
    # the oracle's classes: unchanged (what commit_verdict reads)
    assert res["not_compared_by_family"] == {"hidden_stream": 6, "weights": 1}
    assert res["not_compared"]["no oracle family: hidden_stream"] == 6
    # the population's statement: per Q(P) family, identities from the manifest rows, entries beside them
    q = res["not_compared_by_query_family"]
    assert set(q) == {"fa2_hidden_m1_stream", "moe_block_stream", "weights"}
    assert q["fa2_hidden_m1_stream"]["identities"] == 1 and q["fa2_hidden_m1_stream"]["binding_entries"] == 4       # 4 members, ONE identity
    assert q["moe_block_stream"]["identities"] == 2 and q["moe_block_stream"]["binding_entries"] == 2
    assert q["weights"]["identities"] == 3 and q["weights"]["binding_entries"] == 1 and "registered input" in q["weights"]["why"]
    assert "kernel-interior stream" in q["fa2_hidden_m1_stream"]["why"] and res["not_compared_identities_source"] == "required manifest rows"
    # no manifest: identities are NOT stated (never entries passed off as identities), and the source says so
    res0 = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed))
    assert res0["not_compared_by_query_family"]["fa2_hidden_m1_stream"] == {"identities": None, "binding_entries": 4,
                                                                            "why": "no oracle family: kernel-interior stream (no dispatcher-level value in the Match record)"}
    assert res0["not_compared_identities_source"].startswith("no manifest supplied")
    assert OC.query_families_of(None) is None and OC.query_families_of({"identities": "x"}) is None


def test_a_padded_records_bot_by_enablement_entries_are_stated_apart_never_unverified_attribution_nor_missing_range(tmp_path):
    """The first pod run of workload 3 graded the oracle compare 'partial: request attribution unverified
    at 21 (step, request) pairs ... identity without a byte range x7665' -- every one of them a PADDING entry (r0's steps 11..31, lifted dtype,
    no execution to compare against).  A bot-by-enablement entry is counted apart per Q(P) family, its (step, request) is not an attribution
    pair, its step is not 'not snapshotted'; the verdict reads the EXECUTED entries alone -- here complete."""
    d, vals = make_match(str(tmp_path))
    bmap, committed = make_bmap(vals)
    for s in (1, 2):                                                             # r0 past its stop: padding at two later steps, dense + a shared plane
        for (op, mem), lab in ((("model.embed_tokens", "in"), "lifted33/u64"), (("model.layers.0.self_attn.qkv_proj", "0"), "lifted17/u32")):
            bmap["entries"].append({"identity": {"program_digest": "p" * 64, "request_id": "r0", "step": s, "invocation": 0, "op_path": op, "output_member": mem,
                                                 "element_range": [0, 4], "rank": 0}, "kind": "leaf", "name": f"{op}/{mem}/r0/bot", "element_offset": 0,
                                    "encoding": {"dtype": lab, "shape": [4], "committed_dtype": "i32" if mem == "in" else "bf16", "bot_word_hex": "00"},
                                    "activity": "bot-by-enablement", "block_kind": "padding", "nb": 1, "family": "input_token_ids" if mem == "in" else "instance_outputs"})
        bmap["entries"].append({"identity": {"program_digest": "p" * 64, "request_id": "r0", "step": s, "invocation": 0, "op_path": "fa2.m1/L0", "output_member": "r0/g1x8x1/bot",
                                             "element_range": [0, 8], "rank": 0}, "kind": "leaf", "name": "fa2.m1/L0/r0/g1x8x1/bot", "element_offset": 0,
                                "encoding": {"dtype": "lifted33/u64", "shape": [8], "committed_dtype": "u32", "bot_word_hex": "00"}, "activity": "bot-by-enablement",
                                "block_kind": "padding", "nb": 1, "family": "fa2_hidden_m1_stream", "shared_of": "fa2.m1/L0"})
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed))
    assert res["mismatch_n"] == 0 and res["result"] is True and res["complete"] and res["grade"] == "complete", res["why"]
    assert res["compared"] == 2 * len(vals) + 2                                    # the executed entries of step 0, as without padding
    assert res["steps_not_snapshotted"] == [] and res["steps_compared"] == [0], "padding-only steps are not 'not snapshotted'"
    assert res["attribution"]["complete"] and res["attribution"]["pairs"] == 2 and res["attribution"]["unverified"] == []
    assert "identity without a byte range (dtype/range)" not in res["not_compared"]
    assert res["padding"] == {"entries_not_compared": 6, "by_query_family": {"fa2_hidden_m1_stream": 2, "input_token_ids": 2, "instance_outputs": 2},
                              "pairs": 2, "steps": [1, 2], "pair_list": [[1, "r0"], [2, "r0"]], "rule": res["padding"]["rule"]}
    assert "bot_word" in res["padding"]["rule"]
    # with a manifest, a no-oracle family's `identities` is the FULL-cap count; on a padded record the live count is
    # stated beside it (manifest minus the padding stated apart), labelled -- leg (b): moe_block_stream 6,144 manifest = 4,224 live + 1,920 inactive
    manifest = {"identities": [{"family": "fa2_hidden_m1_stream"}] * 3 + [{"family": "instance_outputs"}] * 40}
    resq = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed), query_families=OC.query_families_of(manifest))
    q = resq["not_compared_by_query_family"]["fa2_hidden_m1_stream"]
    assert q["identities"] == 3 and q["identities_live"] == 1 and q["identities_inactive_stated_apart"] == 2 and q["identities_basis"].startswith("required manifest at full cap")
    assert "identities_live" not in res["not_compared_by_query_family"].get("fa2_hidden_m1_stream", {}) or res["not_compared_by_query_family"]["fa2_hidden_m1_stream"]["identities_live"] is None
    # the two documents agree on one number: oracle padding by family + pairs == the inactive stratum by family
    from verity_vllm.commit import padding_steps as PS
    pp = {"padding_entries": 6, "by_family": {"fa2_hidden_m1_stream": {"commit_padding_entries": 2}, "input_token_ids": {"commit_padding_entries": 2},
                                              "instance_outputs": {"commit_padding_entries": 2}}}
    c3 = PS.padding_vs_oracle(pp, res, bmap)
    assert c3["ok"] and c3["problems"] == [] and c3["entries"] == {"oracle": 6, "map": 6, "stratum": 6}, c3
    assert c3["by_family"]["oracle"] == c3["by_family"]["map"] == c3["by_family"]["stratum"] == {"fa2_hidden_m1_stream": 2, "input_token_ids": 2, "instance_outputs": 2}
    assert c3["pairs"] == {"oracle": 2, "map": 2, "pair_list_compared": True, "executed_and_padding_overlap": 0} and c3["steps"] == [1, 2]
    # a stratum that counts one entry fewer, an oracle that names a pair the map does not: both are named, not averaged
    pp_bad = {"padding_entries": 5, "by_family": {"fa2_hidden_m1_stream": {"commit_padding_entries": 1}, "input_token_ids": {"commit_padding_entries": 2},
                                                  "instance_outputs": {"commit_padding_entries": 2}}}
    bad = PS.padding_vs_oracle(pp_bad, res, bmap)
    assert not bad["ok"] and any("inactive stratum's 5" in q for q in bad["problems"]) and any("!= the inactive stratum's" in q for q in bad["problems"])
    res_bad = dict(res, padding=dict(res["padding"], pair_list=[[1, "r0"], [3, "r0"]]))
    bad2 = PS.padding_vs_oracle(pp, res_bad, bmap)
    assert not bad2["ok"] and any("oracle-only [(3, 'r0')]" in q and "map-only [(2, 'r0')]" in q for q in bad2["problems"])
    # no oracle on the record: the map/stratum half still runs, the oracle half says absent
    absent = PS.padding_vs_oracle(pp, None, bmap)
    assert absent["ok"] and absent["oracle"].startswith("absent") and absent["map_padding_entries"] == 6 and absent["map_pairs"] == 2
    # the same map WITHOUT the activity stamp (an older map) reads as before: partial, the range and the pairs named
    for e in bmap["entries"]:
        e.pop("activity", None)
    res0 = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed))
    assert res0["result"] is None and res0["grade"] == "partial" and res0["not_compared"]["identity without a byte range (dtype/range)"] == 4
    assert "padding" not in res0 and len(res0["attribution"]["unverified"]) == 2
    old = PS.padding_vs_oracle(pp, res0, bmap)                                   # a record whose oracle predates the padding block: named, not ok
    assert not old["ok"] and old["oracle"].startswith("predates the `padding` block") and old["map_padding_entries"] == 0   # (the stamps were popped above)
