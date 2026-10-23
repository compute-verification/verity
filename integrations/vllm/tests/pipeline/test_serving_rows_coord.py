"""The expert-indexed pair GEMMs split into coordinates (`pipeline.serving_rows`' ``coord`` layout, ``COORDS``): a
``MoeExpertFp8Gemm(W)_v1{N}`` row is N units of its Definition at N = 1, unit n reading the row's leading operands whole and
``w[:, n, :]``, ``ws[:, n // G, :]`` of every expert, its output port the row's output word n.  The units cover the row's outputs
with nothing left over (N units under one index entry, unit n's output port word n) and compute them: the coordinate Definition
on each unit's committed rows is the row's Definition's word n."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.ops.prims import HopperE4m3MmaV2Dot32

from verity_vllm.pipeline import serving_rows as SRW
from verity_vllm.program.registry import boolean_fp8_moe as BFM
from verity_vllm.program.registry import fp8_moe_experts as X

E, K, N, G, KB = 3, 64, 40, 32, 32
NB, KG = -(-N // G), K // G
OP = "model.layers.3.mlp.experts"


def gemm(routed: bool, n: int):
    return bind(X.MoeExpertFp8GemmW if routed else X.MoeExpertFp8Gemm, E=E, K=K, N=n, G=G, KB=KB, SWAP=False, DOT=HopperE4m3MmaV2Dot32)


def _bits(a: np.ndarray) -> np.ndarray:
    return a.view(f"<u{a.dtype.itemsize}") if a.dtype.itemsize > 1 else a


def operands(routed: bool, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    def codes(n):
        c = rng.integers(0, 256, n, dtype=np.uint8)
        return np.where((c & 0x7F) == 0x7F, c ^ 1, c).astype(np.uint8)

    def scales(n):
        return ((2.0 ** rng.integers(-8, 2, n)) * rng.uniform(1, 2, n)).astype(np.float32)

    ops = {"xq": codes(K), "sx": scales(KG), "e": np.array([rng.integers(0, E)], dtype=np.int32)}
    if routed:
        ops["g"] = rng.uniform(0, 1, 1).astype(np.float32)
    return {**ops, "w": codes(E * N * K), "ws": scales(E * NB * KG)}


def evaluate(fn, leaves: list[np.ndarray]) -> np.ndarray:
    return np.asarray(evaluate_batch(fn, *leaves)).reshape(len(leaves[0]), -1).astype(np.uint64)


@pytest.fixture
def row(monkeypatch):
    def make(routed: bool, seed: int = 1, w_elems: int | None = None, width: int = N):
        ops = operands(routed, seed)
        names = list(ops)
        y = evaluate(gemm(routed, N), [_bits(ops[nm]).astype(np.uint64).reshape(1, -1) for nm in names]).reshape(-1)
        r = {"i": 7, "spec": descriptor_id(gemm(routed, N)),
             "args": [[["x", k, 0, ops[nm].size if nm != "w" or w_elems is None else w_elems]] for k, nm in enumerate(names)]}
        d = {"statics": {"E": E, "K": K, "N": N, "G": G, "KB": KB}, "op_path": OP, "members": [("down" if routed else "up", width, "bf16")],
             "row": 0, "step": 16}

        class Store:
            def words(self, rid, step, op, member, lo, hi):
                return y[lo:hi].astype(np.uint16), None

        monkeypatch.setattr(SRW, "_slices", lambda _pi, _s, _w, _rid, _a, group: [ops[names[group[0][1]]][:group[0][3]]])
        return ("r0", None, r, d), ops, y, Store()
    return make


def test_a_pair_row_holds_its_n_coordinates():
    for routed in (False, True):
        spec = descriptor_id(gemm(routed, N))
        assert SRW.unit_templates(spec) == ((spec, 1), (descriptor_id(gemm(routed, 1)), N))
        assert SRW.unit_templates(descriptor_id(gemm(routed, 1))) == ((descriptor_id(gemm(routed, 1)), 1),)
        assert SRW.template(descriptor_id(gemm(routed, 1))).layout == "coord"
    boolean = bind(BFM.MoeExpertFp8GemmWV2, E=E, K=K, N=1, G=G, KB=KB, SWAP=False, DOT=BG.HopperE4m3MmaV2Dot32)
    t = SRW.template(descriptor_id(boolean))
    assert (t.layout, t.word, t.packed) == ("coord", descriptor_id(gemm(True, 1)), (8, 32, 32, 32, 8, 32, 16))


@pytest.mark.parametrize("routed", [False, True])
def test_the_coordinates_cover_the_row_s_outputs_and_compute_them(row, routed):
    x, ops, y, store = row(routed)
    T = descriptor_id(gemm(routed, 1))
    (m,) = SRW.members([T], [[x]], [N])
    tp = SRW.template(T)
    words = [n * b // 16 for n, b in zip([ops[nm].size for nm in ops][:-2] + [E * K, E * KG], tp.packed)]
    assert (m.n, m.rows, m.outs) == (N, tuple((f"in{g}", w) for g, w in enumerate(words)), ((f"out.{'down' if routed else 'up'}", 1),))
    data, entries = SRW.read(m, [x], store, None)
    assert entries == [[0, N, "r0", 16, 3, OP, "down" if routed else "up", 0, 7]]
    assert data["outs"][m.outs[0][0]].reshape(-1).tolist() == y.tolist()
    W, WS = ops["w"].reshape(E, N, K), ops["ws"].reshape(E, NB, KG)
    iw, isw = len(m.rows) - 2, len(m.rows) - 1
    for n in range(N):
        assert data["rows"][f"in{iw}"][n].tobytes() == W[:, n, :].tobytes()
        assert data["rows"][f"in{isw}"][n].tobytes() == WS[:, n // G, :].tobytes()
    leaves = []
    for (p, _w), b in zip(m.rows, tp.packed):
        raw = np.ascontiguousarray(data["rows"][p], dtype="<u2").view(np.uint8 if b == 8 else np.dtype(f"<u{b // 8}"))
        leaves.append(raw.reshape(N, -1).astype(np.uint64))
    assert evaluate(tp.fn, leaves).reshape(-1).tolist() == y.tolist()


def test_a_row_whose_operands_are_not_the_statics_is_refused(row):
    x, _ops, _y, store = row(True, w_elems=E * N * K - 2)
    (m,) = SRW.members([descriptor_id(gemm(True, 1))], [[x]], [N])
    with pytest.raises(ValueError, match=rf"row 7 w: \['{E * N * K - 2} uint8'\] elements, not {E * N * K} of 8 bits"):
        SRW.read(m, [x], store, None)
    x, _ops, _y, store = row(True, width=N - 1)
    (m,) = SRW.members([descriptor_id(gemm(True, 1))], [[x]], [N])
    with pytest.raises(ValueError, match=rf"member down of {N - 1} elements, not its N = {N} coordinates"):
        SRW.read(m, [x], store, None)
