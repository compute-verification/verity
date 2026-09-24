"""CPU tests for tp2_worker._t6_4_check (moe cf29c7e72 folded by tp; rev-cb M-0424 T6-4 / report M-0435+M-0478).

The detector compares the committed fused-MoE received all-reduce output `<L>.mlp.experts/out` with the committed next norm's `/0`
in the same step: an EQUAL pair is a late read.  tp's fold adds the not-applicable outcome for dense TP rows (no `/out` member).
"""
from __future__ import annotations

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))


def _worker():
    from verity_vllm.tp import worker as W

    cls = W.__dict__.get("TP2Worker") or next(v for v in W.__dict__.values() if isinstance(v, type) and hasattr(v, "_t6_4_check"))

    class Stub:
        _t6_4_check = cls._t6_4_check

    s = Stub()
    s._tp2_commit = {"rank": 1}
    return s


def _meta(name: str, payload: bytes):
    return types.SimpleNamespace(name=name, host=payload, stream_off=0, nbytes=len(payload))


def _com(layouts: dict):
    return types.SimpleNamespace(_layouts=layouts, _gpu_blocks=None)


def test_moe_row_distinct_bytes_is_true_and_counts_pairs():
    L = lambda i, m, b: _meta(f"model.layers.{i}.{m}", b)   # noqa: E731
    layouts = {s: [L(0, "mlp.experts/out", b"a" * 8 + bytes([s])), L(1, "input_layernorm/0", b"b" * 9), L(1, "mlp.experts/out", b"c" * 9),
                   _meta("model.norm/0", b"d" * 9)] for s in range(3)}
    r = _worker()._t6_4_check(_com(layouts))
    assert r["applicable"] is True and r["result"] is True and r["rank"] == 1
    assert (r["steps"], r["layers_compared"], r["equal_pairs"], r["missing_pairs"]) == (3, 6, 0, 0)
    assert r["pairs"][0]["next_norm"] == "model.layers.1.input_layernorm/0" and r["pairs"][1]["next_norm"] == "model.norm/0"


def test_equal_pair_is_a_late_read_and_named():
    late = b"z" * 16
    layouts = {0: [_meta("model.layers.3.mlp.experts/out", late), _meta("model.norm/0", late)]}
    r = _worker()._t6_4_check(_com(layouts))
    assert r["applicable"] is True and r["result"] is False and r["equal_pairs"] == 1
    p = r["pairs"][0]
    assert p["equal"] is True and (p["step"], p["layer"], p["next_norm"]) == (0, 3, "model.norm/0") and p["out_sha"] == p["next_norm_sha"]


def test_missing_next_norm_fails_closed():
    layouts = {0: [_meta("model.layers.0.mlp.experts/out", b"q" * 4)]}
    r = _worker()._t6_4_check(_com(layouts))
    assert r["result"] is False and r["missing_pairs"] == 1 and r["pairs"][0]["missing"] == ["next_norm"]


def test_dense_tp_row_is_not_applicable_not_false():
    layouts = {s: [_meta(f"model.layers.{i}.self_attn.o_proj/0", b"o" * 6) for i in range(4)] + [_meta("model.norm/0", b"n" * 6)] for s in range(2)}
    r = _worker()._t6_4_check(_com(layouts))
    assert r["applicable"] is False and r["result"] is None and r["layers_compared"] == 0 and r["steps"] == 0
    assert "not applicable" in r["why"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
