"""ONE deliberately dumb, independent reference fixture for the alias mechanism the committed READER
(opening.CommittedStore / oracle_compare.OpenedReader, entry_read_name) and the ORACLE side (oracle_compare's inverse alias
table) SHARE.  "Reader and oracle sharing one alias mechanism prevents drift AND lets them share the same bug" -- so this file derives what
every side must say BY COUNTING (layers x members x steps x requests, invocation k = layer order, ordinal = position in acquisition order),
never by reading `op_path_aliases`, `entry_read_name`, `same_name_invocations` or the inverse table; then drives the production chain
end-to-end -- `binding.build_binding_map` (the invocation numbering), `OpenedReader` + `CommittedStore` (the reader), `oracle_compare`
over a REAL `MatchOracle` of a per-layer-hinted synthetic Match dir (the oracle) -- and checks each against the enumerator.  A bug the two
sides share fails HERE, not each other.

Rules it pins: "different logical producer/version -> never merged solely because bits/offsets coincide":
  * two invocations whose committed words are byte-identical at identical offsets stay two identities, read at two ordinals, compared
    against two oracle instances (a per-layer oracle that differs then MISMATCHES by name -- never 'equal by collapse');
  * an alias entry that points an invocation at ANOTHER module's path (a different logical producer) is never a match: the oracle has
    no rotary instance there, the invocation is NOT compared, by name;
  * a shared invocation the table does not name is NOT compared, never compared against layer 0's oracle.
Version identity (RoPE_v1 vs RoPE_v2 at one path) is the manifest's `spec` per identity (required_manifest tests), not this
fixture's: the binding map carries no spec.

It tests the COMMON MECHANISM on one synthetic record, not every production record."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import types

import numpy as np
import pytest

from tests.check.opened import commit_steps
from verity_vllm.commit import binding as B
from verity_vllm.check import oracle_compare as OC
from verity_vllm.check.replay import opening as OP

CANON = "model.layers.0.self_attn.rotary_emb"
MEMBERS = ("query", "key")                              # the shared RotaryEmbedding's two outputs (q, k)
HINT = {"query": "rope.q", "key": "rope.k"}             # the fold's per-member hint suffix (oracle_compare._ROPE_MEMBER, the record's shape)
W = 8                                                   # words per request row
U0 = "aaaaaaaa-0000-0000-0000-000000000000"


# ------------------------------------------------------------------------------------------------------------ the dumb enumerator

def per_layer_path(k: int) -> str:
    return f"model.layers.{k}.self_attn.rotary_emb"


def norm_path(k: int) -> str:
    return f"model.layers.{k}.self_attn.q_norm"


def expected_table(n_layers: int) -> dict[str, list]:
    """The alias table BY COUNTING: layer k's Program path is the canonical (layer 0) module's k-th invocation (k = layer order)."""
    return {per_layer_path(k): [CANON, k] for k in range(1, n_layers)}


def expected_identities(n_layers: int, steps: list[int], requests: list[str]) -> set[tuple]:
    """(step, request, CANONICAL op_path, member, invocation k): one identity per (step, request, layer, member) of the shared module."""
    return {(s, r, CANON, m, k) for s in steps for r in requests for k in range(n_layers) for m in MEMBERS}


def expected_ordinal(k: int, member: str) -> int:
    """Acquisition order per step: for each layer k in forward order -> q_norm/out, then rotary member 'query', then 'key'."""
    return 3 * k + 1 + MEMBERS.index(member)


def expected_words(step: int, k: int, member: str, r: int, *, coincide: bool = False) -> np.ndarray:
    """The words layer k's invocation holds for request r at `step` (distinct per (step, layer, member, request)); `coincide`: every layer
    holds LAYER 0's words (the bits/offsets-coincide negative)."""
    kk = 0 if coincide else k
    return (np.arange(W, dtype=np.uint16) + 1000 * kk + 100 * step + 10 * r + (500 if member == "key" else 0)).astype("<u2")


# ------------------------------------------------------------------------------------------------------------ the committed record

def make_record(n_layers: int, steps: list[int], n_requests: int, *, coincide: bool = False):
    """The collector's real shape: per step, metas in acquisition order; the shared module publishes one meta per INVOCATION under the SAME
    canonical name; every layer also has its own (non-shared) q_norm meta.  Rows = one per request (decode-shaped).  Committed as the pod's
    collector commits them (GPU-tree step blocks) -> (committer, run commitment)."""
    pytest.importorskip("torch")
    tensors: dict[int, list[tuple]] = {}
    for s in steps:
        ts: list[tuple] = []
        for k in range(n_layers):
            for name in (f"{norm_path(k)}/out", *(f"{CANON}/{m}" for m in MEMBERS)):
                member = name.rsplit("/", 1)[1]
                if "q_norm" in name:
                    rows = np.stack([np.arange(W, dtype=np.uint16) + 5000 + 1000 * k + 100 * s + 10 * r for r in range(n_requests)]).astype("<u2")
                else:
                    rows = np.stack([expected_words(s, k, member, r, coincide=coincide) for r in range(n_requests)])
                ts.append((name, rows.tobytes(), len(ts), "bfloat16", rows.shape))
        tensors[s] = ts
    return commit_steps(tensors)


def build_map(rec: tuple, requests: list[str]) -> dict:
    com, _run = rec
    steps_list = [types.SimpleNamespace(root=bytes([s]) * 32) for s in range(max(com._layouts) + 1)]
    request_rows = {s: [(r, i, i + 1) for i, r in enumerate(requests)] for s in com._layouts} if len(requests) > 1 else None
    return B.build_binding_map(com._layouts, steps_list, b"\x01" * 32, "p" * 64, request_rows=request_rows, request_id=requests[0])


# ------------------------------------------------------------------------------------------------------------ the synthetic Match (oracle)

def make_match(tmp: str, n_layers: int, *, words=expected_words) -> str:
    """One step, one request (r0 = U0), one row; per layer the fold's RoPE_v1 instances hinted `model.layers.k.self_attn.rotary_emb/rope.q|k`."""
    d = os.path.join(tmp, "match")
    os.makedirs(os.path.join(d, "capture", "values", "0"), exist_ok=True)
    entries, insts = [], []
    ev = 100
    for k in range(n_layers):
        for member in MEMBERS:
            data = words(0, k, member, 0).tobytes()
            sha = hashlib.sha256(data).hexdigest()
            fn = f"values/0/{sha[:16]}_{k}_{member}.bin"
            open(os.path.join(d, "capture", fn), "wb").write(data)
            alloc = 10 + 2 * k + MEMBERS.index(member)
            entries.append({"event": ev, "path": "outs", "view": {"$v": [alloc, 0, [1, W], [W, 1], "bfloat16", "cuda:0"]}, "dtype": "bfloat16",
                            "shape": [1, W], "strides": [W, 1], "byte_offset": 0, "nbytes": len(data), "layout": "spanned-storage-bytes",
                            "sha256": sha, "file": fn, "step": 0})
            insts.append({"node": 0, "step": 0, "event": ev, "pattern": "x", "spec": "RoPE_v1{}", "row": 0, "rows": 1, "name": "t=0/rotary_emb",
                          "hint": f"{per_layer_path(k)}/{HINT[member]}", "request": U0, "position": 0, "operand_views": [],
                          "output_views": [{"alloc": alloc, "offset": 0, "shape": [W], "strides": [1], "dtype": "bfloat16", "device": "cuda:0"}]})
            ev += 1
    json.dump({"schema": "verity-capture/snapshots/v1", "steps": [0], "entries": entries}, open(os.path.join(d, "capture", "snapshots.json"), "w"))
    with open(os.path.join(d, "instances.jsonl"), "w") as f:
        f.write(json.dumps({"schema": "verity-gen/instances/v1", "digest": "f" * 64, "n_instances": len(insts),
                            "requests": [{"index": 0, "request_id": U0, "prompt_len": 1, "served_samples": 1, "max_tokens": 1}]}) + "\n")
        for m in insts:
            f.write(json.dumps(m) + "\n")
    with gzip.open(os.path.join(d, "capture", "log.jsonl.gz"), "wt") as f:
        f.write(json.dumps({"$e": "RunHeader", "schema": "verity-capture/log/v1", "created_utc": "2026-09-20T23:00:00Z", "profile_id": "abc", "profile": {"vllm": {"version": "t"}}}) + "\n")
        f.write(json.dumps({"$e": "StepStart", "step": 0, "rows": [[U0, 0]]}) + "\n")
        f.write(json.dumps({"$e": "SampleEnd", "step": 0, "sampled_token_ids": [5]}) + "\n")
    json.dump({"requests": [{"request_id": U0, "prompt_token_ids": [1], "generated_token_ids": [5]}]}, open(os.path.join(d, "capture", "tokens.json"), "w"))
    return d


def _shared(bmap: dict) -> list[dict]:
    return [e for e in bmap["entries"] if e["identity"]["op_path"] == CANON]


# ============================================================================================================ the fixture's checks

N = 4
STEPS = [0, 1]
REQS = ["r0", "r1"]


def test_1_the_binding_maps_invocation_numbering_is_the_enumerators_layer_order_and_distinct_names_never_merge():
    rec = make_record(N, STEPS, len(REQS))
    bmap = build_map(rec, REQS)
    got = {(e["identity"]["step"], e["identity"]["request_id"], e["identity"]["op_path"], e["identity"]["output_member"], e["identity"]["invocation"])
           for e in _shared(bmap)}
    assert got == expected_identities(N, STEPS, REQS)                                  # one identity per (step, request, layer, member)
    for e in _shared(bmap):
        k = e["identity"]["invocation"]
        assert e["tensor_ordinal"] == expected_ordinal(k, e["identity"]["output_member"]), e     # the k-th same-name meta IS layer k's
        assert e["identity"]["element_range"] == [0, W] and e["element_offset"] == REQS.index(e["identity"]["request_id"]) * W
    # the non-shared per-layer module: N distinct names, invocation 0 each, never folded into the shared module's count
    norms = [e for e in bmap["entries"] if "q_norm" in e["identity"]["op_path"]]
    assert {e["identity"]["op_path"] for e in norms} == {norm_path(k) for k in range(N)} and {e["identity"]["invocation"] for e in norms} == {0}
    sni = OC.same_name_invocations(bmap)
    for s in STEPS:
        assert sni[(s, f"{CANON}/query")] == N and sni[(s, f"{CANON}/key")] == N and sni[(s, f"{norm_path(1)}/out")] == 1


def test_2_the_reader_returns_layer_ks_own_words_for_every_step_request_layer_member_of_the_enumeration():
    rec = make_record(N, STEPS, len(REQS))
    bmap = build_map(rec, REQS)
    table = expected_table(N)
    rd = OC.OpenedReader(*rec)
    st = OP.CommittedStore.of_record(bmap, rd, {"op_path_aliases": table})
    for s in STEPS:
        for r, rid in enumerate(REQS):
            for k in range(N):
                for m in MEMBERS:
                    w, _dt = st.words(rid, s, CANON if k == 0 else per_layer_path(k), m, 0, W)
                    assert w is not None and w.astype(np.uint16).tolist() == expected_words(s, k, m, r).tolist(), (s, rid, k, m)
            q, _ = st.words(rid, s, norm_path(2), "out", 0, W)                              # the non-shared member reads bare, its own words
            assert q.astype(np.uint16).tolist() == (np.arange(W) + 5000 + 2000 + 100 * s + 10 * r).tolist()
    # the read-name the store asks for is the enumerator's ordinal, and a bare shared name is None (never the first meta's words)
    for e in _shared(bmap):
        s, k, m = e["identity"]["step"], e["identity"]["invocation"], e["identity"]["output_member"]
        assert OC.entry_read_name(e, N) == f"{CANON}/{m}{OC.ORDINAL_SEP}{expected_ordinal(k, m)}"
        assert rd(s, f"{CANON}/{m}", 0, W * 2) is None


def test_3_the_oracle_side_reads_invocation_k_at_layer_ks_path_and_the_honest_record_is_equal_over_the_whole_enumeration(tmp_path):
    d = make_match(str(tmp_path), N)
    rec = make_record(N, [0], 1)
    bmap = build_map(rec, ["r0"])
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, OC.OpenedReader(*rec), aliases=expected_table(N))
    n_shared = len(expected_identities(N, [0], ["r0"]))
    assert res["compared"] == n_shared and res["equal"] == n_shared and res["mismatch_n"] == 0, res.get("not_compared")
    assert res["shared_invocations"]["compared_at_program_op"] == (N - 1) * len(MEMBERS) and res["shared_invocations"]["not_compared"] == 0
    # the N q_norm entries have no oracle instance in the synthetic Match: named not-compared (partial), never a mismatch, never an equal
    assert res["not_compared"] == {"no fold instance of this op at this step": N} and res["result"] is not False


def test_4_negative_bits_and_offsets_that_coincide_are_still_n_identities_and_a_differing_per_layer_oracle_mismatches_by_name(tmp_path):
    """Every layer's committed words == layer 0's (identical bits at identical request offsets).  The identities stay N x members; the reader
    reads N distinct ordinals; against the per-layer oracle layers 1.. MISMATCH exactly -- never 'equal by collapse' onto layer 0."""
    rec = make_record(N, [0], 1, coincide=True)
    bmap = build_map(rec, ["r0"])
    assert len(_shared(bmap)) == len(expected_identities(N, [0], ["r0"]))
    assert len({e["tensor_ordinal"] for e in _shared(bmap)}) == N * len(MEMBERS)          # N distinct ordinals, not one merged identity
    rd = OC.OpenedReader(*rec)
    words = [rd(0, OC.entry_read_name(e, N), 0, W * 2) for e in _shared(bmap)]
    assert len(set(words)) == len(MEMBERS)                                                 # the bits DO coincide across layers ...
    d = make_match(str(tmp_path), N)
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, rd, aliases=expected_table(N))
    assert res["result"] is False and res["mismatch_n"] == (N - 1) * len(MEMBERS) and res["equal"] == len(MEMBERS)   # ... and are not merged
    assert {m["identity"]["invocation"] for m in res["mismatch"]} == set(range(1, N))


def test_5_negative_an_alias_to_another_modules_path_is_a_different_logical_producer_never_a_match_and_an_unnamed_invocation_is_not_compared(tmp_path):
    d = make_match(str(tmp_path), N)
    rec = make_record(N, [0], 1)
    bmap = build_map(rec, ["r0"])
    rd = OC.OpenedReader(*rec)
    # (a) invocation 2 pointed at layer 2's q_norm (another module): the oracle has no rotary instance there -> NOT compared, by name; never equal
    table = expected_table(N)
    table[norm_path(2)] = [CANON, 2]
    del table[per_layer_path(2)]
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, rd, aliases=table)
    assert res["mismatch_n"] == 0 and res["equal"] == (N - 1) * len(MEMBERS) and res["compared"] == (N - 1) * len(MEMBERS), res
    # not compared by name: the 2 mis-aliased members + the N q_norm entries (the synthetic Match has no q_norm instances at all)
    assert sum(v for k, v in res["not_compared"].items() if "no fold instance" in k) == len(MEMBERS) + N, res["not_compared"]
    # (b) no table: invocation 0 compares, 1..N-1 are not compared by name, never against layer 0's oracle
    res0 = OC.oracle_compare(OC.MatchOracle(d), bmap, rd, aliases=None)
    assert res0["mismatch_n"] == 0 and res0["compared"] == len(MEMBERS) and res0["shared_invocations"]["not_compared"] == (N - 1) * len(MEMBERS)
    # (c) a table naming only SOME layers: exactly the unnamed ones are not compared
    part = {per_layer_path(1): [CANON, 1]}
    res1 = OC.oracle_compare(OC.MatchOracle(d), bmap, rd, aliases=part)
    assert res1["mismatch_n"] == 0 and res1["compared"] == 2 * len(MEMBERS) and res1["shared_invocations"]["not_compared"] == (N - 2) * len(MEMBERS)


def test_6_the_enumerator_itself_is_a_bijection_and_matches_the_records_own_table_shape():
    """The dumb table is injective both ways and equals the shape required_manifest writes ({program path: [canonical, k]}, k = layer order)."""
    t = expected_table(N)
    assert len(t) == N - 1 and len({tuple(v) for v in t.values()}) == N - 1
    assert all(v[0] == CANON and per_layer_path(v[1]) == p for p, v in t.items())
    assert len(expected_identities(N, STEPS, REQS)) == N * len(MEMBERS) * len(STEPS) * len(REQS)
