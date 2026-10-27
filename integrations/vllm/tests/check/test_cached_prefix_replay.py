"""The Commit's reading of a prefix-cache hit: a warm request's `prefix` reads resolve to the key / value rows its producers committed
(`check.replay.cached_prefix`, through the producer's own attention operands), its prompt reads to the ids its arrival step embedded
(prompt[C:]), and the boundary linkage holds that step to the declared uncached suffix."""
from __future__ import annotations

import json

import numpy as np
import pytest

from verity_vllm.check.replay import cached_prefix as CP, evaluate as EV, linkage as LK
from verity_vllm.pipeline import c2_replay as C2
from verity.ml.definitions.numerics.kernel_registry import Declined

C, RL, BS = 4, 4, 2          # cached rows, key / value leaves per row (KVH*D), block size
ROPE, QKV = "model.layers.{l}.self_attn.rotary_emb", "model.layers.{l}.self_attn.qkv_proj"
LAYOUT = {"input": 1, "C": C, "row_leaves": RL, "layers": [f"model.layers.{l}.self_attn.attn" for l in (0, 1)]}


class _Producer:
    """A cold request's Program as the resolver reads it: per layer, key rows (RoPE `key`, one member of RL leaves per history row) and
    qkv rows (`out`, the value at [2 RL, 3 RL)), and an attention per history length T whose operands read rows 0..T-1."""

    def __init__(self, n=6, own_prefix=False):
        self.rows, self.by_i, self._d = [], {}, {}
        for l in (0, 1):
            for t in range(n):
                k, v = 1000 * l + t, 1000 * l + 100 + t
                self._d[k] = {"members": [("key", RL, "bf16")], "step": 0, "row": t, "op_path": ROPE.format(l=l)}
                self._d[v] = {"members": [("out", 3 * RL, "bf16")], "step": 0, "row": t, "op_path": QKV.format(l=l)}
            for T in range(1, n + 1):
                i = 1000 * l + 500 + T
                kv = [[["p", 1, 0, C * RL]] + [["n", 1000 * l + t, 0, RL] for t in range(C, T)],
                      [["p", 1, C * RL, 2 * C * RL]] + [["n", 1000 * l + 100 + t, 2 * RL, 3 * RL] for t in range(C, T)]] if own_prefix else \
                     [[["n", 1000 * l + t, 0, RL] for t in range(T)], [["n", 1000 * l + 100 + t, 2 * RL, 3 * RL] for t in range(T)]]
                r = {"i": i, "spec": f"Attention_v4{{T={T},NH=2,KVH=1,D={RL}}}", "args": [[["n", 0, 0, 2 * RL]], *kv],
                     "name": f"model.model.layers.{l}.self_attn.attn/unified_attention_with_output[t={T - 1}]"}
                self.rows.append(r)
                self.by_i[i] = r

    def member_at(self, x, lo, hi):
        d = self._d.get(x)
        if d is None:
            return None
        name, width, dt = d["members"][0]
        return (d, name, lo, hi, dt) if 0 <= lo < hi <= width else None


def _src(producers=("r0", "r0"), arrivals=None, **kw):
    idx = {q: _Producer(**kw) for q in set(producers)}
    return CP.CachedPrefix(LAYOUT, list(producers), idx, arrivals or {q: 0 for q in producers})


def test_a_cached_key_or_value_row_is_the_row_its_producer_committed():
    src = _src()
    assert src.identities(0, C * RL) == [("r0", 0, ROPE.format(l=0), "key", j * RL, (j + 1) * RL) for j in range(C)]
    assert src.identities(C * RL, 2 * C * RL) == [("r0", 0, QKV.format(l=0), "out", 3 * RL * j + 2 * RL, 3 * RL * j + 3 * RL) for j in range(C)]
    assert src.identities(3 * C * RL + RL + 1, 3 * C * RL + 2 * RL) == [("r0", 0, QKV.format(l=1), "out", 3 * RL + 2 * RL + 1, 3 * RL + 3 * RL)]


def test_each_block_is_read_from_its_own_producer_at_its_arrival():
    src = _src(("r0", "r2"), arrivals={"r0": 0, "r2": 3})
    got = src.identities(0, C * RL)
    assert [(q, s) for q, s, *_ in got] == [("r0", 0), ("r0", 0), ("r2", 3), ("r2", 3)]


def test_a_read_across_two_planes_or_from_a_producer_that_was_itself_served_is_declined():
    with pytest.raises(Declined, match="not inside one layer"):
        _src().identities(C * RL - 1, C * RL + 1)
    with pytest.raises(Declined, match="did not compute it"):
        _src(own_prefix=True).identities(0, RL)
    with pytest.raises(Declined, match="does not hold"):
        CP.CachedPrefix(LAYOUT, ["r0", "r0"], {}, {}).identities(0, RL)


def test_the_warm_prefix_input_resolves_and_the_prompt_reads_the_uncached_suffix():
    class Store:
        def __init__(self):
            self.reads = []

        def words(self, *a):
            self.reads.append(a)
            return np.zeros(a[5] - a[4], dtype=np.uint16), None

    class Warm:
        pnames, supplied, kv_prefix, by_i = ["prompt", "prefix", "weights"], {}, dict(LAYOUT), {}
        cached_prefix = {"r1": _src()}

        def describe(self, r):
            return {"family": "Attention_v4", "statics": {}, "step": 0, "row": 0, "op_path": "x", "members": []}

        def embed_op_path(self):
            return "model.embed_tokens"

    st = Store()
    EV._evaluate_row(Warm(), st, None, {"i": 9, "spec": "X_v1", "args": [[["p", 0, C + 1, C + 2]], [["p", 1, 0, 2 * RL]]]}, "r1", 5, [],
                     depth=0, plan=True)
    assert st.reads == [("r1", 5, "model.embed_tokens", "in", 1, 2), ("r0", 0, ROPE.format(l=0), "key", 0, RL),
                        ("r0", 0, ROPE.format(l=0), "key", RL, 2 * RL)]
    with pytest.raises(Declined, match="prefix cache served"):
        EV._evaluate_row(Warm(), st, None, {"i": 9, "spec": "X_v1", "args": [[["p", 0, C - 1, C]]]}, "r1", 5, [], depth=0, plan=True)
    cold = Warm()
    cold.cached_prefix = {}
    with pytest.raises(Declined, match="no producing requests of record for r1"):
        EV._evaluate_row(cold, st, None, {"i": 9, "spec": "X_v1", "args": [[["p", 1, 0, RL]]]}, "r1", 5, [], depth=0, plan=True)


def test_the_layout_is_the_derives_prefix_file(tmp_path):
    assert CP.layout_of(str(tmp_path)) is None and CP.layout_of(None) is None
    (tmp_path / CP.PREFIX_FILE).write_text(json.dumps(dict(LAYOUT, input="1")))
    assert CP.layout_of(str(tmp_path)) == LAYOUT


def test_the_commit_hands_each_warm_request_its_producers_by_label():
    reqs = [{"request_id": "a", "cached": 0, "cached_from": []}, {"request_id": "b", "cached": 128, "cached_from": ["a"] * 8}]
    assert C2.cached_prefixes(reqs, {"a": "r0", "b": "r1"}) == {"r1": {"cached": 128, "producers": ["r0"] * 8}}


def test_the_linkage_holds_a_warm_arrival_to_the_declared_uncached_suffix():
    prompt = list(range(100, 110))

    class Store:
        def __init__(self, ids):
            self.ids = ids

        def words(self, rid, step, op, member, lo, hi):
            got = self.ids.get((rid, step))
            return (np.asarray(got[lo:hi], dtype=np.int64), "") if got is not None and hi <= len(got) else (None, f"no ids [{lo}, {hi})")

        def has(self, *a):
            return False

    reqs = [{"request_id": "r1", "arrive_step": 4, "prompt_token_ids": prompt, "steps": [4], "cached": C}]
    lk = LK.boundary_linkage({}, None, reqs, embed_op_path="model.embed_tokens", store=Store({("r1", 4): prompt[C:]}))
    assert lk["result"] is True and lk["prompt_cached"] == {"r1": C}
    bad = LK.boundary_linkage({}, None, reqs, embed_op_path="model.embed_tokens", store=Store({("r1", 4): prompt[C - 1:-1]}))
    assert bad["result"] is False and bad["unlinked"][0]["kind"] == "prompt"
    whole = LK.boundary_linkage({}, None, [dict(reqs[0], cached=0)], embed_op_path="model.embed_tokens", store=Store({("r1", 4): prompt[C:]}))
    assert whole["result"] is False
