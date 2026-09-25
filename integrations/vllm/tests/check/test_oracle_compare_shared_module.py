"""[moe F-moe-r17-07] The C2 ORACLE COMPARE had the alias class on ITS side too -- and it hid F-moe-r17-06 for three runs.

OLMoE B2 mixedstop, tree_int5 / int6 / int7 (the committed reader by NAME = layer 0's bytes for every rotary invocation): C2 oracle compare
'compared 8668 equal 8668' -- while the sampled replay found layers 1..15 unequal.  tree_int8 (24df89c50, the reader fixed): sampled replay
2052/2052 equal, oracle compare 1320 MISMATCHES, every one a rotary invocation 1..15 (15 layers x 44 (step, request) pairs x 2 members).
Cause: `oracle_compare` asked the Match oracle for `ident.op_path` = the CANONICAL shared path (`model.layers.0.self_attn.rotary_emb`) with
no invocation; the fold's instances are hinted per LAYER (`model.layers.L.self_attn.rotary_emb/rope.q`, 16 distinct prefixes x 126 at step 0
on the record), so `_inst[(step, canonical)]` is layer 0's instances -- the oracle answered layer 0's bytes for every invocation, and the
pre-fix committed side answered layer 0's bytes too: EQUAL on both sides' wrong bytes, a false equal 660 identities x 2 members deep.

Fix: `aliases` (manifest.op_path_aliases) inverted: (canonical, invocation k) -> the Program's per-layer op_path = the fold's hint; a shared
invocation the table does not name is NOT compared (named), never compared against another layer's oracle.

[reqval, on the train baa79a8dc] moe's four tests (d5aa44385) PORTED onto the train's interface (reqval 65588eaef + 051f4b722, the oracle side
rel picked as Shape M): the record keys are `shared_invocations.{compared_at_program_op, not_compared}` and the not-compared reason is the
F-moe-r17-06 string; the inverse table is built inside `oracle_compare` (no `alias_inverse` helper on the train) -- the same behaviour, a REAL
`MatchOracle` over a synthetic per-layer-hinted Match dir, which the train's own tests (a fake oracle) did not have."""
from __future__ import annotations

import gzip
import hashlib
import json
import os

import numpy as np

from tests.check.opened import commit_host_steps
from verity_vllm.check import oracle_compare as OC

U0 = "aaaaaaaa-0000-0000-0000-000000000000"
ROT = "model.layers.0.self_attn.rotary_emb"
H = 8                       # head width of the rotary output row
N_LAYERS = 3
POS = 2                     # r0's rows at step 0


def _rows(L: int, member: str) -> np.ndarray:
    return (np.arange(POS * H, dtype=np.uint16).reshape(POS, H) + 1000 * L + (500 if member == "1" else 0)).astype("<u2")


def make_match(tmp: str) -> str:
    """One step, one request, N_LAYERS layers; per layer the fold's two RoPE_v1 instances per row hinted `model.layers.L.self_attn.rotary_emb/rope.q|k`
    (the real record's shape: 16 distinct per-layer prefixes at step 0), each output alloc snapshotted."""
    d = os.path.join(tmp, "match")
    os.makedirs(os.path.join(d, "capture", "values", "0"), exist_ok=True)
    entries, insts = [], []
    ev = 100
    for L in range(N_LAYERS):
        for member, suf in (("0", "rope.q"), ("1", "rope.k")):
            data = _rows(L, member).tobytes()
            sha = hashlib.sha256(data).hexdigest()
            fn = f"values/0/{sha[:16]}.bin"
            open(os.path.join(d, "capture", fn), "wb").write(data)
            alloc = 10 + 2 * L + int(member)
            entries.append({"event": ev, "path": "outs", "view": {"$v": [alloc, 0, [POS, H], [H, 1], "bfloat16", "cuda:0"]}, "dtype": "bfloat16",
                            "shape": [POS, H], "strides": [H, 1], "byte_offset": 0, "nbytes": len(data), "layout": "spanned-storage-bytes",
                            "sha256": sha, "file": fn, "step": 0})
            for pos in range(POS):
                insts.append({"node": 0, "step": 0, "event": ev, "pattern": "x", "spec": "RoPE_v1{}", "row": pos, "rows": POS,
                              "name": f"t={pos}/rotary_emb", "hint": f"model.layers.{L}.self_attn.rotary_emb/{suf}", "request": U0, "position": pos,
                              "operand_views": [], "output_views": [{"alloc": alloc, "offset": pos * H, "shape": [H], "strides": [1], "dtype": "bfloat16", "device": "cuda:0"}]})
            ev += 1
    json.dump({"schema": "verity-capture/snapshots/v1", "steps": [0], "entries": entries}, open(os.path.join(d, "capture", "snapshots.json"), "w"))
    with open(os.path.join(d, "instances.jsonl"), "w") as f:
        f.write(json.dumps({"schema": "verity-gen/instances/v1", "digest": "f" * 64, "n_instances": len(insts),
                            "requests": [{"index": 0, "request_id": U0, "prompt_len": POS, "served_samples": 1, "max_tokens": 1}]}) + "\n")
        for m in insts:
            f.write(json.dumps(m) + "\n")
    with gzip.open(os.path.join(d, "capture", "log.jsonl.gz"), "wt") as f:
        f.write(json.dumps({"$e": "RunHeader", "schema": "verity-capture/log/v1", "created_utc": "2026-09-20T21:00:00Z", "profile_id": "abc", "profile": {"vllm": {"version": "t"}}}) + "\n")
        f.write(json.dumps({"$e": "StepStart", "step": 0, "rows": [[U0, p] for p in range(POS)]}) + "\n")
        f.write(json.dumps({"$e": "SampleEnd", "step": 0, "sampled_token_ids": [5]}) + "\n")
    json.dump({"requests": [{"request_id": U0, "prompt_token_ids": [1, 2], "generated_token_ids": [5]}]}, open(os.path.join(d, "capture", "tokens.json"), "w"))
    return d


def make_committed(layer_of_invocation=None):
    """The collector's shape: per invocation k one meta per member under the SAME canonical name `ROT/<member>`; binding map identities under
    the canonical op_path with invocation k.  `layer_of_invocation(k)` = whose words meta k holds (default: its own layer's -- the honest record).
    Returns (the production OpenedReader over those tensors, committed on the per-tensor host path; the binding map)."""
    who = layer_of_invocation or (lambda k: k)
    tensors, entries = [], []
    ordinal = 0
    for k in range(N_LAYERS):
        for member in ("0", "1"):
            tensors.append((f"{ROT}/{member}", _rows(who(k), member).tobytes()))
            entries.append({"identity": {"program_digest": "p" * 64, "request_id": "r0", "step": 0, "invocation": k, "op_path": ROT, "output_member": member,
                                         "element_range": [0, POS * H], "rank": 0},
                            "kind": "leaf", "name": f"{ROT}/{member}", "element_offset": 0, "tensor_ordinal": ordinal, "stream_offset": ordinal * POS * H * 2,
                            "encoding": {"dtype": "bfloat16", "shape": [POS, H]}})
            ordinal += 1
    return OC.OpenedReader(*commit_host_steps({0: tensors})), {"entries": entries}


ALIASES = {f"model.layers.{L}.self_attn.rotary_emb": [ROT, L] for L in range(1, N_LAYERS)}


def test_the_oracle_side_reads_a_shared_invocation_at_its_per_layer_path_and_the_honest_record_is_equal(tmp_path):
    d = make_match(str(tmp_path))
    rd, bmap = make_committed()
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, rd, aliases=ALIASES)
    assert res["mismatch_n"] == 0 and res["compared"] == N_LAYERS * 2 and res["equal"] == N_LAYERS * 2, res
    assert res["shared_invocations"]["compared_at_program_op"] == (N_LAYERS - 1) * 2 and res["shared_invocations"]["not_compared"] == 0 and res["not_compared"] == {}
    assert res["result"] is True and res["complete"] and res["grade"] == "complete"


def test_without_the_alias_table_an_aliased_invocation_is_not_compared_by_name_never_against_layer_zeros_oracle(tmp_path):
    """The pre-fix shape: no aliases -> the canonical path names layer 0's fold instances.  Invocation 0 compares (equal); invocations 1.. are
    NOT compared, by name -- result None (partial), never a false equal, never a false mismatch."""
    d = make_match(str(tmp_path))
    rd, bmap = make_committed()
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, rd)
    assert res["mismatch_n"] == 0 and res["compared"] == 2 and res["equal"] == 2 and res["result"] is None and res["grade"] == "partial", res
    assert list(res["not_compared"].values()) == [(N_LAYERS - 1) * 2] and "shared-module invocation >= 1 with no Program path in the alias table" in next(iter(res["not_compared"]))
    assert res["shared_invocations"] == {"compared_at_program_op": 0, "not_compared": (N_LAYERS - 1) * 2, "note": res["shared_invocations"]["note"]}


def test_the_false_equal_of_int5_int6_int7_is_gone_a_layer_zero_read_for_every_invocation_mismatches_the_per_layer_oracle(tmp_path):
    """The record that fooled the compare: committed words of invocation k = layer 0's (the by-name reader).  Against the per-layer oracle every
    invocation >= 1 MISMATCHES by name (first element 0); invocation 0 is equal.  On the real record: 1320 = 15 layers x 44 pairs x 2 members."""
    d = make_match(str(tmp_path))
    rd, bmap = make_committed(layer_of_invocation=lambda k: 0)
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, rd, aliases=ALIASES)
    assert res["result"] is False and res["mismatch_n"] == (N_LAYERS - 1) * 2 and res["equal"] == 2, res
    assert all(m["identity"]["invocation"] >= 1 and m["identity"]["op_path"] == ROT and m["why"].startswith("bytes differ (first element 0") for m in res["mismatch"])
    assert res["why"].startswith(f"C2 oracle mismatch: {(N_LAYERS - 1) * 2} identities differ from the Match oracle (first: step 0 {ROT}/0 inv 1 req r0")
    # ... and the honest record whose layer-2 meta holds layer 1's words: exactly that invocation, named
    rd2, bmap2 = make_committed(layer_of_invocation=lambda k: 1 if k == 2 else k)
    res2 = OC.oracle_compare(OC.MatchOracle(d), bmap2, rd2, aliases=ALIASES)
    assert res2["mismatch_n"] == 2 and {m["identity"]["invocation"] for m in res2["mismatch"]} == {2}


def test_a_malformed_alias_table_is_no_table_the_invocations_are_not_compared_by_name(tmp_path):
    """moe's `alias_inverse` shape test, as behaviour on the train: a table that names nothing usable == no table (fail closed, by name)."""
    d = make_match(str(tmp_path))
    rd, bmap = make_committed()
    for bad in (None, {}, {"model.layers.9.self_attn.rotary_emb": [ROT, 9]}):
        res = OC.oracle_compare(OC.MatchOracle(d), bmap, rd, aliases=bad)
        assert res["compared"] == 2 and res["result"] is None and res["shared_invocations"]["not_compared"] == (N_LAYERS - 1) * 2, (bad, res["not_compared"])
