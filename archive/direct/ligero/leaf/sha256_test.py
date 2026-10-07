"""The SHA-256 row leaf without torch: the native fold against the core ``sha256/row/v1`` vectors, the batched padding fold, and
one compression gadget evaluated from its witness program (honest -> every constraint holds and the output is SHA-256's
compression; one flipped carry row -> a constraint fails)."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.commitments.frame_v3 import vectors as V

from ..compile import LigeroCtx, P
from ..relations import FP8_ADA_RELATION
from . import sha256 as S

L = S.Sha256Leaf()


def test_native_fold_matches_the_core_vectors() -> None:
    rows = [r for r in V.load()["row_leaves"] if r["hash"] == "sha256" and len(bytes.fromhex(r["row_bytes"])) % 64 == 0]
    assert rows
    for c in rows:
        raw, wb = bytes.fromhex(c["row_bytes"]), c["word_bits"]
        words = np.frombuffer(raw, dtype={8: "<u1", 16: "<u2", 32: "<u4"}[wb]).astype(np.int64)[None, :]
        assert L.leaf_bytes(L.native(words, word_bits=wb, role=c["role"])[0]).hex() == c["digest"]


@pytest.mark.parametrize("word_bits", (8, 16))
def test_leaf_bytes_many_is_leaf_bytes(word_bits: int) -> None:
    rows = np.random.default_rng(word_bits).integers(0, 1 << word_bits, size=(9, S.K_VU))
    for role in (1, 2):
        d = L.native(rows, word_bits=word_bits, role=role)
        assert L.leaf_bytes_many(d) == [L.leaf_bytes(r) for r in d] == L.reference_digest(rows, word_bits=word_bits, role=role)
    bad = d.copy()
    bad[3, 5] = 1 << 16
    bad[4, 0] += 1
    assert L.leaf_bytes_many(bad) == [L.leaf_bytes(r) for r in bad]
    assert L.leaf_bytes_many(np.zeros((0, S.DIGEST_ELEMS), dtype=np.int64)) == []


def _run(sys, hints: dict[str, int]) -> list[int]:
    W = [0] * sys.m

    def ev(e):
        return (e.c + sum(v * W[j] for j, v in e.lin.items())) % P

    for op in sys.program:
        k = op[0]
        if k == "hint":
            W[op[1]] = hints[op[2]] % P
        elif k == "prod":
            W[op[1]] = ev(op[2]) * ev(op[3]) % P
        elif k == "bits":
            x = ev(op[3])
            for b in range(op[2]):
                W[op[1] + b] = (x >> b) & 1
        elif k == "sels":
            x = ev(op[4])
            x = x - P if x > P // 2 else x
            for i, (a, b) in enumerate(zip(op[2], op[3])):
                W[op[1] + i] = int(a <= x <= b)
        else:
            raise ValueError(k)
    return W


def _violations(sys, W) -> int:
    def ev(e):
        return (e.c + sum(v * W[j] for j, v in e.lin.items())) % P

    return (sum(ev(c) != 0 for c in sys.linear)
            + sum(ev(q.a) * ev(q.b) % P != ev(q.c) for q in sys.quadratic))


def test_compression_gadget_computes_sha256_and_rejects_a_flipped_carry() -> None:
    ctx = LigeroCtx(FP8_ADA_RELATION.params)
    h = [S._W32([ctx.hint(f"h{w}.b{i}") for i in range(32)]) for w in range(8)]
    m = [S._W32([ctx.hint(f"m{w}.b{i}") for i in range(32)]) for w in range(16)]
    out = S.compress_gadget(ctx, h, m, "c")
    sys = ctx.sys
    assert sys.m == S.ROWS_PER_BLOCK + 8 * 32 + 16 * 32
    rng = np.random.default_rng(5)
    H = rng.integers(0, 1 << 32, size=8, dtype=np.uint64).astype(np.uint32)
    M = rng.integers(0, 1 << 32, size=16, dtype=np.uint64).astype(np.uint32)
    hints = {f"h{w}.b{i}": (int(H[w]) >> i) & 1 for w in range(8) for i in range(32)}
    hints |= {f"m{w}.b{i}": (int(M[w]) >> i) & 1 for w in range(16) for i in range(32)}
    W = _run(sys, hints)
    assert _violations(sys, W) == 0
    got = [sum(((b.c + sum(v * W[j] for j, v in b.lin.items())) % P) << i for i, b in enumerate(o.bits)) for o in out]
    assert got == [int(x) for x in S.compress_np(H[:, None], M[:, None])[:, 0]]
    sels = [op[1] for op in sys.program if op[0] == "sels"]
    W[sels[len(sels) // 2]] ^= 1
    assert _violations(sys, W) > 0


@pytest.mark.parametrize("word_bits,steps", ((8, 12), (8, 48), (16, 12), (16, 24), (16, 96)))
def test_row_sponges_are_carry_values_and_native(word_bits: int, steps: int) -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    rows = np.random.default_rng(steps).integers(0, 1 << word_bits, size=(37, S.K_VU))
    words = torch.as_tensor(rows, device="cuda").reshape(37 * steps, S.K_VU // steps)
    for role in (1, 2):
        car, dg = L.row_sponges(words, word_bits=word_bits, role=role, row_words=S.K_VU, n=37, steps=steps)
        assert np.array_equal(car.cpu().numpy(), L.carry_values(rows, word_bits=word_bits, role=role, steps=steps))
        assert np.array_equal(dg.cpu().numpy(), L.native(rows, word_bits=word_bits, role=role))
