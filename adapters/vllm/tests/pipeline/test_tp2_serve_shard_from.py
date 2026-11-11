"""The pin's acceptance-set negative 'TP shard / reference mismatch' (`tp2_commit --serve-shard-from <rank>:<param>:<from_rank>`):
a rank serves ANOTHER rank's shard of one parameter, composed from the SAME pinned checkpoint.  Every served byte is a checkpoint byte and
the multiset of shard digests over the world is unchanged -- a whole-tensor / whole-checkpoint hash cannot see it -- but the per-rank pin
(served shard == the slice the rank Program's shard map prescribes) must FAIL naming exactly that parameter on exactly that rank.
Engine-free: the same `weights_of_record.compose` / `check` the rank worker runs, on the synthetic checkpoint of test_weights_of_record."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from verity_vllm.pipeline.tp.negatives import parse_serve_shard_from
from verity_vllm.check import weights_of_record as WR
from tests.check.test_weights_of_record import PREFIX, _program, _served, checkpoint, live_digests  # noqa: F401  (fixture)


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def test_parse_serve_shard_from_names_rank_param_and_another_rank():
    assert parse_serve_shard_from(None, 2) is None
    assert parse_serve_shard_from([], 2) is None
    one = parse_serve_shard_from(["1:model.layers.5.mlp.down_proj.weight:0"], 2)
    assert one == {"1": [{"param": "model.layers.5.mlp.down_proj.weight", "from_rank": 0}]}
    swap = parse_serve_shard_from(["0:X.weight:1", "1:X.weight:0"], 2)
    assert swap == {"0": [{"param": "X.weight", "from_rank": 1}], "1": [{"param": "X.weight", "from_rank": 0}]}
    for bad in (["1:X.weight"], ["1::0"], ["1:X.weight:1"], ["2:X.weight:0"], ["1:X.weight:5"]):
        with pytest.raises(SystemExit):
            parse_serve_shard_from(bad, 2)


@pytest.mark.parametrize("param", ["model.layers.0.mlp.down_proj.weight", "model.layers.0.mlp.gate_up_proj.weight", "model.layers.0.self_attn.qkv_proj.weight"])
def test_one_parameter_served_as_the_other_ranks_shard_fails_the_pin_by_name_and_only_there(checkpoint, param):
    d, man, t = checkpoint
    ck = WR.CheckpointIndex(str(d), checkpoints_manifest=str(man))
    recs, served = {}, {}
    for rank in (0, 1):
        prog, fields = _program(tp={"world": 2, "rank": rank})
        recs[rank] = WR.weights_of_record(prog, ck, prefix=PREFIX)
        served[rank] = _served(t, fields, tp={"world": 2, "rank": rank})
        assert WR.check(recs[rank], live_digests(served[rank]))["result"] is True          # the positive: every rank serves its own shards
    shape, bits = list(served[1][param].shape), 16
    # what the rank-1 worker does under --serve-shard-from 1:<param>:0 -- compose rank 0's slice from the pinned checkpoint and serve it
    other, note = WR.compose(param, shape, bits, ck, world=2, rank=0)
    own, _ = WR.compose(param, shape, bits, ck, world=2, rank=1)
    assert "replicated" not in str(note["shard"]) and "slice" in str(note["shard"])
    assert _sha(own) == _sha(served[1][param]) and _sha(other) == _sha(served[0][param])   # compose() is the shard map the served model follows
    assert _sha(other) != _sha(own)
    bad = dict(served[1])
    bad[param] = other
    res = WR.check(recs[1], live_digests(bad))
    assert res["result"] is False
    assert [m["served_name"] for m in res["mismatched"]] == [param], res["mismatched"]       # exactly the mis-served parameter, nothing else
    assert res["compared"] == WR.check(recs[1], live_digests(served[1]))["compared"]         # every other field still compared and equal
    assert param.split(".")[-2] in res["why"]
    # rank 0 is untouched: its pin still passes
    assert WR.check(recs[0], live_digests(served[0]))["result"] is True
    # the blindness the negative demonstrates: the multiset of served digests of <param> over the world is a permutation-invariant of the swap
    swapped = {0: dict(served[0]), 1: bad}
    swapped[0][param] = own                                                                    # 0:<param>:1 -- the two ranks swapped their shards
    before = sorted(_sha(served[r][param]) for r in (0, 1))
    after = sorted(_sha(swapped[r][param]) for r in (0, 1))
    assert before == after                                                                     # a whole-checkpoint / shard-set hash sees nothing ...
    assert WR.check(recs[0], live_digests(swapped[0]))["result"] is False                     # ... the per-rank pin fails on BOTH ranks
    assert WR.check(recs[1], live_digests(swapped[1]))["result"] is False
    assert [m["served_name"] for m in WR.check(recs[0], live_digests(swapped[0]))["mismatched"]] == [param]


def test_a_replicated_parameter_has_no_other_ranks_shard(checkpoint):
    d, man, t = checkpoint
    ck = WR.CheckpointIndex(str(d), checkpoints_manifest=str(man))
    prog, fields = _program(tp={"world": 2, "rank": 1})
    served = _served(t, fields, tp={"world": 2, "rank": 1})
    p = "model.layers.0.input_layernorm.weight"
    a, note = WR.compose(p, list(served[p].shape), 16, ck, world=2, rank=0)
    assert "replicated" in str(note["shard"]) and _sha(a) == _sha(served[p])                  # the worker refuses this spec by name (no negative possible)


def test_repo_modules_loaded_names_this_process_imports_relative_to_the_repo():
    """The import path of record is written from the RUNNING process (sys.modules), not a static closure."""
    from verity_vllm.pipeline.tp.rank_worker import repo_modules_loaded
    from verity_vllm.check import weights_of_record  # noqa: F401  -- imported here == must be listed
    r = repo_modules_loaded()
    assert r["method"].startswith("runtime:") and r["n"] == len(r["files"]) == len(set(r["files"]))
    assert "verity_vllm/pipeline/tp/rank_worker.py" in r["files"] and "verity_vllm/check/weights_of_record.py" in r["files"]
    assert all(not f.startswith("/") and not f.endswith(".pyc") for f in r["files"])
    assert r["native_collect_jit_sources"] == [] or all(f.endswith((".cpp", ".cu", ".h")) for f in r["native_collect_jit_sources"])


# ---- the decode-only (step-ranged) request-label swap negative ------------------------------------------------------------

def test_swap_spec_all_steps_and_ranged():
    from verity_vllm.pipeline.tp.rank_worker import parse_swap_spec, swap_label
    assert parse_swap_spec(None) is None
    a = parse_swap_spec(["r3", "r4"])
    assert a == {"a": "r3", "b": "r4", "lo": None, "hi": None, "mode": "all"}
    r = parse_swap_spec(["r3", "r4", 1, 37])
    assert r["mode"] == "steps" and (r["lo"], r["hi"]) == (1, 37)
    # all-steps swap: every step, and step-less consumers too
    for st in (None, 0, 5, 100):
        assert swap_label("r3", a, st) == "r4" and swap_label("r4", a, st) == "r3" and swap_label("r0", a, st) == "r0"
    # ranged swap: ONLY steps lo..hi; step 0 (the prefill of record) and step-less consumers untouched
    assert swap_label("r3", r, 0) == "r3" and swap_label("r4", r, 0) == "r4"
    assert swap_label("r3", r, 1) == "r4" and swap_label("r4", r, 37) == "r3"
    assert swap_label("r3", r, 38) == "r3"
    assert swap_label("r3", r, None) == "r3"
    assert swap_label("r0", r, 10) == "r0"
    # honest path: no rule -> identity
    assert swap_label("r3", None, 10) == "r3"


def test_ranged_swap_keeps_the_prefill_rows_and_exchanges_only_decode_labels():
    """The map the binding/attribution consumers see under the ranged swap: prefill rows (step 0) carry their true labels; on the shared
    decode steps r3 and r4 exchange labels; per-step row COUNTS per label are unchanged (each row one token) -- what coverage sees is
    identical to the honest map, which is why only the attribution can catch it."""
    from verity_vllm.pipeline.tp.rank_worker import parse_swap_spec, swap_label
    r = parse_swap_spec(["r3", "r4", 1, 37])
    step_rows = {0: [("r3", 0, 305), ("r4", 305, 313)], 1: [("r3", 0, 1), ("r4", 1, 2)], 40: [("r3", 0, 1)]}
    seen = {s: [(swap_label(l, r, s), lo, hi) for l, lo, hi in rows] for s, rows in step_rows.items()}
    assert seen[0] == step_rows[0]                                       # prefill untouched
    assert seen[1] == [("r4", 0, 1), ("r3", 1, 2)]                        # decode labels exchanged
    assert seen[40] == step_rows[40]                                      # outside the range untouched
    # per-label token counts per step: identical to the honest map (coverage-blind)
    cnt = lambda rows: sorted((l, hi - lo) for l, lo, hi in rows)
    for s in step_rows:
        assert cnt(seen[s]) == cnt(step_rows[s]) or s == 1
    assert cnt(seen[1]) == cnt(step_rows[1])


# ---- a negative that did not inject is an ERROR, never a PASS (':ulp' on a tree without parse_flip flips 0 -> PASS)

def test_negative_not_applied_is_named_not_passed():
    from verity_vllm.pipeline.tp.negatives import negatives_applied
    made = [{"rank": 0, "swap_requests_applied": None, "tp_partial_source": {"omit_site": None}},
            {"rank": 1, "swap_requests_applied": None, "tp_partial_source": {"omit_site": None}}]
    xr = [{"dumps": [{"rank": 0, "flipped_collective_inputs": 0}, {"rank": 1, "flipped_collective_inputs": 0}]}]
    a = negatives_applied("1:layers.3.mlp.down_proj:ulp", None, None, xr, made)
    assert a["flip_partial"]["applied"] is False and a["flip_partial"]["rank"] == "1"
    # the pass did not run -> unknown, still not True
    a = negatives_applied("1:layers.3.mlp.down_proj", None, None, [], made)
    assert a["flip_partial"]["applied"] is None
    # swap named requests that the workload does not have -> not applied
    a = negatives_applied(None, "1:r3:r4:1-37", None, [], made)
    assert a["swap_requests"]["applied"] is False
    # honest run: nothing requested -> nothing to check
    assert negatives_applied(None, None, None, xr, made) == {}


def test_negative_applied_is_recognised():
    from verity_vllm.pipeline.tp.negatives import negatives_applied
    made = [{"rank": 0, "swap_requests_applied": None, "tp_partial_source": {"omit_site": None}},
            {"rank": 1, "swap_requests_applied": ["r3", "r4", 1, 37], "tp_partial_source": {"omit_site": "layers.3.mlp.down_proj"}}]
    xr = [{"dumps": [{"rank": 0, "flipped_collective_inputs": 0}, {"rank": 1, "flipped_collective_inputs": 64}]}]
    a = negatives_applied("1:layers.3.mlp.down_proj:ulp", "1:r3:r4:1-37", "1:layers.3.mlp.down_proj", xr, made)
    assert all(v["applied"] is True for v in a.values()) and set(a) == {"flip_partial", "swap_requests", "omit_partial_site"}
    # the wrong rank requested (rank 0 did not flip) -> not applied
    assert negatives_applied("0:layers.3.mlp.down_proj", None, None, xr, made)["flip_partial"]["applied"] is False
