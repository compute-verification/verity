"""Form (B) qualification evidence: a COLLECTOR-SIDE perturbation with the reference observation
UNCHANGED must fail the oracle-compare.

The two identity->value paths (code provenance, rg on the release tree):
  collector   native_collect._record_step_rows (InputBatch.req_ids x scheduler_output.num_scheduled_tokens -> step_rows) -> commit_delta
              .binding_record._logical (engine id -> r<i>) -> binding.build_binding_map (request_rows split of every batched member ->
              entries with identity + element_offset + stream offsets) -> the committed bytes (oracle_compare.OpenedReader; a dict here).
  observer    vllm_adapter recorder (dispatcher observer: snapshots.json + values/<step>/*.bin, log.jsonl.gz StepStart rows / SampleEnd,
              tokens.json) + fold.py (instances.jsonl: which event/storage view is which Program instance for which request/position)
              -> oracle_compare.MatchOracle.member_bytes / ids_bytes / sampled_id.
  shared      the required manifest DICT (expected identities, member widths, producer spec/ordinal) -- the Program-derived population,
              not a mapping; and ONE engine fact read twice (the scheduler's request->row layout).  No module of one path imports the other.

Here the REAL collector-side mapping function (`binding.build_binding_map`) is driven with a perturbed `request_rows` (what
`_record_step_rows` would have recorded) or a perturbed engine-id->r<i> relabel (what `_logical` would have produced), the committed
bytes and the whole Match record untouched; `oracle_compare` must FAIL naming the affected identities.  The positive (unperturbed map
built by the same function) passes complete.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np

from verity_vllm.commit import binding as B
from verity_vllm.check import oracle_compare as OC
from tests.check.test_oracle_compare import H, ROWS, U0, U1, make_match, reader

PD = "p" * 64
# the collector's committed members of step 0 (packed over the engine's row order: U1's 3 rows first, then U0's 5), names as the hook table emits them
NAMES = {("model.embed_tokens", "in"): "model.embed_tokens/in", ("model.embed_tokens", "out"): "model.embed_tokens",
         ("model.layers.0.post_attention_layernorm", "0"): "model.layers.0.post_attention_layernorm/0",
         ("model.layers.0.post_attention_layernorm", "1"): "model.layers.0.post_attention_layernorm/1",
         ("model.layers.0.self_attn.qkv_proj", "0"): "model.layers.0.self_attn.qkv_proj/0"}


def _collector_layouts(vals: dict) -> tuple[dict[int, list], dict[tuple[int, str], bytes]]:
    """The collector's published layouts of step 0 (one meta per committed member, in stream order) + its retained bytes."""
    n = sum(ROWS.values())
    metas, committed, off = [], {}, 0
    for k, ((op, mem), data) in enumerate(vals.items()):
        shape = [n] if mem == "in" else [n, H]
        dt = "int32" if mem == "in" else "bfloat16"
        metas.append(SimpleNamespace(name=NAMES[(op, mem)], shape=shape, dtype=dt, nbytes=len(data), stream_off=off, first_leaf=off // 256, chunk=256, ordinal=k))
        committed[(0, NAMES[(op, mem)])] = data
        off += len(data)
    sampled = np.array([777, 555], dtype="<i8").tobytes()                          # per request in row order: U1 -> 777, U0 -> 555
    metas.append(SimpleNamespace(name="runner.sampler/sampled_token_ids", shape=[2], dtype="int64", nbytes=len(sampled), stream_off=off, first_leaf=off // 256, chunk=256, ordinal=len(metas)))
    committed[(0, "runner.sampler/sampled_token_ids")] = sampled
    return {0: metas}, committed


def _map(layouts, request_rows):
    steps = [SimpleNamespace(root=hashlib.sha256(b"step0").digest())]
    return B.build_binding_map(layouts, steps, hashlib.sha256(b"run").digest(), PD, request_rows=request_rows, chunk=256)


# the collector's attribution of record: engine row order U1 (r1) rows [0,3), U0 (r0) rows [3,8) -- what _record_step_rows x _logical yield
HONEST_ROWS = {0: [("r1", 0, 3), ("r0", 3, 8)]}


def test_unperturbed_collector_map_passes_complete(tmp_path):
    d, vals = make_match(str(tmp_path))
    layouts, committed = _collector_layouts(vals)
    bmap = _map(layouts, HONEST_ROWS)
    # every batched member split per request at the engine's row offsets; the sampler's ids member per request slot
    q = {e["identity"]["request_id"]: (e["identity"]["element_range"], e["element_offset"]) for e in bmap["entries"] if e["name"] == "model.layers.0.self_attn.qkv_proj/0"}
    assert q == {"r1": ([0, 3 * H], 0), "r0": ([0, 5 * H], 3 * H)}
    s = {e["identity"]["request_id"]: e["element_offset"] for e in bmap["entries"] if e["name"] == "runner.sampler/sampled_token_ids"}
    assert s == {"r1": 0, "r0": 1}
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed))
    assert res["result"] is True and res["complete"] and res["mismatch_n"] == 0, res["why"]
    assert res["compared"] == 2 * len(vals) + 2 and res["attribution"]["complete"]


def test_collector_request_attribution_swapped_observer_unchanged_fails_named(tmp_path):
    """The collector's step_rows say r0 owns the first 3 rows and r1 the last 5 (the engine's row order misread / a relabel in _logical);
    bytes, layouts, Match record all unchanged -> every per-request identity of both requests is refuted by the oracle, including the
    ids and sampled-id families that attribution rests on."""
    d, vals = make_match(str(tmp_path))
    layouts, committed = _collector_layouts(vals)
    swapped = {0: [("r0", 0, 3), ("r1", 3, 8)]}
    bmap = _map(layouts, swapped)
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed))
    assert res["result"] is False and res["grade"] == "mismatch"
    named = {(m["identity"]["request_id"], m["identity"]["op_path"], m["identity"]["output_member"]) for m in res["mismatch"]}
    assert {r for r, _o, _m in named} == {"r0", "r1"}
    assert ("r0", "model.embed_tokens", "in") in named and ("r1", "model.embed_tokens", "in") in named           # the ids oracle (tokens.json) refutes the relabel
    assert ("r0", "runner.sampler", "sampled_token_ids") in named and ("r1", "runner.sampler", "sampled_token_ids") in named
    assert res["mismatch_n"] == 2 * len(vals) + 2 and res["attribution"]["complete"] is False
    assert "C2 oracle mismatch" in res["why"]


def test_collector_row_boundary_off_by_one_observer_unchanged_fails_named(tmp_path):
    """The collector's row map puts the boundary one row early (r1 rows [0,2), r0 rows [2,8)): only the token-row members disagree at the
    boundary rows; the oracle reads each request's rows from the fold at their own positions and names both requests' blocks
    (size mismatch for r1's 2-row block vs the oracle's 3 rows; content mismatch for r0's block starting one row early)."""
    d, vals = make_match(str(tmp_path))
    layouts, committed = _collector_layouts(vals)
    bmap = _map(layouts, {0: [("r1", 0, 2), ("r0", 2, 8)]})
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed))
    assert res["result"] is False
    by_req = {}
    for m in res["mismatch"]:
        by_req.setdefault(m["identity"]["request_id"], set()).add(m["identity"]["op_path"])
    assert set(by_req) == {"r0", "r1"}
    assert "model.embed_tokens" in by_req["r1"] and "model.layers.0.self_attn.qkv_proj" in by_req["r0"]
    assert any(m["identity"]["request_id"] == "r1" and str(m.get("why", "")).startswith("size:") for m in res["mismatch"])
    # the sampler's per-request slots are untouched by a row-boundary error: the sampled ids still agree
    assert not any(m["identity"]["op_path"] == "runner.sampler" for m in res["mismatch"])


def test_collector_omits_a_request_from_the_row_map_is_a_missing_identity_not_a_pass(tmp_path):
    """The collector attributes every row to r1 (r0 dropped from step_rows): the map binds the whole packed member to r1 -> the oracle
    refutes r1's block (8 rows vs 3), and r0 has NO identity in the map at all -- C1 coverage names it (the compare cannot vouch for an
    identity the collector never bound)."""
    d, vals = make_match(str(tmp_path))
    layouts, committed = _collector_layouts(vals)
    bmap = _map(layouts, {0: [("r1", 0, 8)]})
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed))
    assert res["result"] is False and all(m["identity"]["request_id"] == "r1" for m in res["mismatch"])
    assert not any(e["identity"]["request_id"] == "r0" for e in bmap["entries"])
    pop = [B.identity(PD, "r0", 0, 0, "model.embed_tokens", "in", 0, 5), B.identity(PD, "r1", 0, 0, "model.embed_tokens", "in", 0, 3)]
    cov = B.coverage_check(pop, bmap)
    assert not cov["ok"] and cov["missing_n"] == 1 and cov["missing"][0]["request_id"] == "r0"


def test_perturbation_is_collector_side_only(tmp_path):
    """Guard for the evidence class: the perturbations above change NOTHING on the observer side -- the same MatchOracle instance serves the
    honest map (complete PASS) and the swapped map (FAIL) in one process; its provenance and value files are byte-identical before/after."""
    import os

    d, vals = make_match(str(tmp_path))
    layouts, committed = _collector_layouts(vals)
    oracle = OC.MatchOracle(d)
    files = sorted(os.path.join(dp, f) for dp, _dn, fs in os.walk(d) for f in fs)
    before = {f: hashlib.sha256(open(f, "rb").read()).hexdigest() for f in files}
    ok = OC.oracle_compare(oracle, _map(layouts, HONEST_ROWS), reader(committed))
    bad = OC.oracle_compare(oracle, _map(layouts, {0: [("r0", 0, 3), ("r1", 3, 8)]}), reader(committed))
    after = {f: hashlib.sha256(open(f, "rb").read()).hexdigest() for f in files}
    assert before == after and ok["oracle"] == bad["oracle"]
    assert ok["result"] is True and bad["result"] is False
