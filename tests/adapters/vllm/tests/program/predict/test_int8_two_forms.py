"""An int8 linear's two forms: the rows' `ScaledMmS8_v1` over the checkpoint's s8 codes and f32 scales, and the class circuit's
`ScaledMmS8_v3` slices over the same weights as the u16 words its weights registration commits (`window_weights`)."""
from __future__ import annotations

import numpy as np
import pytest
from verity.circuits import evaluation as E

from verity_vllm.definitions.registry import int8_exact as IX
from verity_vllm.pipeline import serving_rows as SRW
from verity_vllm.program.predict import window_builder as WB
from verity_vllm.program.predict import window_weights as WW

DOT = '{"fn":"BlackwellS8ImmaDot32_v1"}'


def _codes(rng, n, K, N):
    x = rng.integers(-128, 128, (n, K)).astype(np.int8)
    w = rng.integers(-128, 128, (N, K)).astype(np.int8)
    sx = (np.abs(rng.standard_normal(n)) / 127).astype(np.float32)
    sw = (np.abs(rng.standard_normal(N)) * 0.02 / 127).astype(np.float32)
    return x, sx, w, sw


def _class_words(w, sw):
    N, K = w.shape
    return WW._rows(WB.weight_param("w", K // 2), w), WW._rows(WB.weight_param("sw", 2), sw)


@pytest.mark.parametrize("K,N", [(64, 16), (128, 8)])
def test_the_class_s8_slice_on_its_registered_words_is_the_rows_s8_linear_on_the_codes(K, N):
    """`int8_exact.scaled_mm_words` (the class's slice) on the class's packing of W and its scales gives
    `int8_exact.scaled_mm` (the rows' product) bit for bit, on the reference evaluator."""
    rng = np.random.default_rng(K + N)
    n = 32
    x, sx, w, sw = _codes(rng, n, K, N)
    ww, sww = _class_words(w, sw)

    def rep(a):
        return np.broadcast_to(a, (n, *a.shape))

    want = E.evaluate_batch(IX.scaled_mm(K, N), x.view(np.uint8), sx.view(np.uint32), rep(w.view(np.uint8)),
                            rep(sw.view(np.uint32)), kernel="reference")
    got = E.evaluate_batch(IX.scaled_mm_words(K, N), x.view(np.uint8), sx.view(np.uint32), rep(ww), rep(sww),
                           kernel="reference")
    assert np.array_equal(got, want)


def test_the_rows_committed_weight_words_are_the_class_registered_words(monkeypatch):
    """Each `ScaledMmS8Coordinate_v1` unit's ``w`` row the rows commit (`s8_rows.read`) is the class's registered words of
    that weight row, then of its scale."""
    K, N = 64, 5
    rng = np.random.default_rng(3)
    x, sx, w, sw = _codes(rng, 1, K, N)
    ww, sww = _class_words(w, sw)
    spec = f"ScaledMmS8_v1{{K={K},N={N},DOT={DOT}}}"
    r = {"i": 0, "spec": spec, "args": [[["n", 0, 0, K]], [["n", 0, K, K + 1]], [["p", 0, 0, N * K]], [["p", 0, N * K, N * K + N]]]}
    d = {"statics": {"K": K, "N": N}, "op_path": "model.model.layers.0.self_attn.qkv_proj", "members": [("y", N, "bf16")],
         "row": 0, "step": 0}
    operands = [x[0].view(np.uint8), sx.view(np.uint32), w.reshape(-1).view(np.uint8), sw.view(np.uint32)]
    monkeypatch.setattr(SRW, "_operand", lambda _pi, _s, _w, _rid, _a, group: operands[r["args"].index(group)])

    class Store:
        def words(self, _rid, _step, _op, _member, lo, hi):
            return np.zeros(N, np.uint16)[lo:hi], None

    coord = f"ScaledMmS8Coordinate_v1{{K={K},DOT={DOT}}}"
    rows = [("r0", None, r, d)]
    got = SRW._read_s8(SRW.members([coord], [rows], [N], {coord: N}), rows, Store(), None, None)
    committed = np.concatenate([g[0]["rows"]["w"] for g in got]).astype("<u2")
    assert np.array_equal(committed, np.concatenate([ww, sww], axis=1))
