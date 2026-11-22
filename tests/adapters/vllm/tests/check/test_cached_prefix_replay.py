"""The Commit's reading of a prefix-cache hit: a warm request's `prefix` reads resolve to the key / value rows its producers committed
(`check.replay.cached_prefix`, through the producer's own attention operands), its prompt reads to the ids its arrival step embedded
(prompt[C:]), and the boundary linkage holds that step to the declared uncached suffix."""
from __future__ import annotations

import json

import pytest

from verity_vllm.check.replay import cached_prefix as CP
from verity_vllm.definitions.numerics.kernel_registry import Declined

C, RL, BS = 4, 4, 2          # cached rows, key / value leaves per row (KVH*D), block size
ROPE, QKV = "model.layers.{l}.self_attn.rotary_emb", "model.layers.{l}.self_attn.qkv_proj"
LAYOUT = {"input": 1, "C": C, "row_leaves": RL, "layers": [f"model.layers.{layer}.self_attn.attn" for layer in (0, 1)]}


class _Producer:
    """A cold request's Program as the resolver reads it: per layer, key rows (RoPE `key`, one member of RL leaves per history row) and
    qkv rows (`out`, the value at [2 RL, 3 RL)), and an attention per history length T whose operands read rows 0..T-1."""

    def __init__(self, n=6, own_prefix=False):
        self.rows, self.by_i = [], {}
        self._d: dict[int, dict] = {}
        for layer in (0, 1):
            for t in range(n):
                k, v = 1000 * layer + t, 1000 * layer + 100 + t
                self._d[k] = {"members": [("key", RL, "bf16")], "step": 0, "row": t, "op_path": ROPE.format(l=layer)}
                self._d[v] = {"members": [("out", 3 * RL, "bf16")], "step": 0, "row": t, "op_path": QKV.format(l=layer)}
            for T in range(1, n + 1):
                i = 1000 * layer + 500 + T
                kv = [[["p", 1, 0, C * RL]] + [["n", 1000 * layer + t, 0, RL] for t in range(C, T)],
                      [["p", 1, C * RL, 2 * C * RL]] + [["n", 1000 * layer + 100 + t, 2 * RL, 3 * RL] for t in range(C, T)]] if own_prefix else \
                     [[["n", 1000 * layer + t, 0, RL] for t in range(T)], [["n", 1000 * layer + 100 + t, 2 * RL, 3 * RL] for t in range(T)]]
                r = {"i": i, "spec": f"Attention_v4{{T={T},NH=2,KVH=1,D={RL}}}", "args": [[["n", 0, 0, 2 * RL]], *kv],
                     "name": f"model.model.layers.{layer}.self_attn.attn/unified_attention_with_output[t={T - 1}]"}
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


def test_the_layout_is_the_derives_prefix_file(tmp_path):
    assert CP.layout_of(str(tmp_path)) is None and CP.layout_of(None) is None
    (tmp_path / CP.PREFIX_FILE).write_text(json.dumps(dict(LAYOUT, input="1")))
    assert CP.layout_of(str(tmp_path)) == LAYOUT


