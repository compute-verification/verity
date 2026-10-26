"""attn_ref's check data: random sequences and every word the kernels must write for them, from the Definition.

QK^T's words are `GemmCoordinate_v2{K=D, GEMM_HOPPER}` and PV's `GemmCoordinatePad_v1{T=n, K=pv_k(n), GEMM_HOPPER}`
(verity.ml.circuits.ops.gemm), pv_k(n) = max(KPV, pad16(n)) with KPV = 32, the H200 circuit's PV.  Each is evaluated by
the numpy twin of `GemmCoordinate_v2{K}` on the zero-padded operands (`evaluate_batch`, kernel "auto":
verity.ml.kernels.tc's gemm_coordinate_kernel), and the twin is held to the Definition's reference evaluator (kernel
"reference") on a sample of every sequence's words, those with a special operand first.  Any disagreement exits 1 before
anything is written.

CHECK/pool.npz, the frozen decode shape's sequences (256 query heads of head dim 64 on one KV head, 32 cached keys):
  Q [NX, 256, 64], K and V [NX, 32, 64], P [NX, 256, 32]   the inputs (bf16 words)
  S [NX, 256, 32]                                         S[i, h, j] = GemmCoordinate{K=64}(Q[i, h], K[i, j]), at any length > j
  O [NX, 32, 256, 64]                                     O[i, n-1, h, d] = GemmCoordinatePad{T=n, K=32}(P[i, h, :n],
                                                          V[i, :n, d]), the output at KV length n = 1..32
  kind [NX]                                               each sequence's generator (KINDS)
check.py builds every batch it runs from copies of these sequences, so every word of every batch has its expected word.  A
position past a length n holds the generator's words like any other (NaN, infinities and huge values in some kinds), which
the kernels must not read as values.
CHECK/odd.npz: a GQA case off the frozen shape (6 sequences, 12 query heads on 3 KV heads, head dim 128, 37 cached keys,
lengths 0, 1, 16, 17, 33 and 37): its inputs and the whole S (fill 0xFF80 past each length) and O.
CHECK/fixture.json: the seeds, kinds, word counts, the reference sample and each file's sha256.

  uv run python benchmarks/torch_gpu/attn_ref/fixture.py CHECK      (env PROCS: worker processes, default 8)
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import sys

import numpy as np

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops import gemm as G

H, D, LCAP = 256, 64, 32
KINDS = ("normal", "softmax", "wide", "unit", "tiny", "specials", "cancel_in", "cancel_across", "uniform", "subacc")
NX = 4 * len(KINDS)
SEED = 20261026
FILL = 0xFF80
KPV = 32
SPECIALS = np.array([0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x7F81, 0xFFFF, 0x0001, 0x8001, 0x007F, 0x807F,
                     0x0080, 0x7F7F, 0xFF7F, 0x3F80, 0xBF80], np.uint16)
SIGN = np.uint16(0x8000)
REF_PER_SEQ = 128
ODD = dict(B=6, H=12, Hkv=3, D=128, L=37, lens=(0, 1, 16, 17, 33, 37))


def coordinate(x: np.ndarray, w: np.ndarray, kernel: str = "auto") -> np.ndarray:
    """GemmCoordinate{K, GEMM_HOPPER} on the rows of ``x`` and ``w`` (N, K)."""
    spec = bind(G.GemmCoordinate, K=x.shape[1], **G.GEMM_HOPPER)
    return np.asarray(E.evaluate_batch(spec, np.ascontiguousarray(x), np.ascontiguousarray(w), kernel=kernel), np.uint16)


def pad16(n: int) -> int:
    return -(-n // 16) * 16


def pv_k(n: int) -> int:
    """PV's K at KV length n: attn_ref.pv's default kpv = KPV, or pad16(n) past it."""
    return max(KPV, pad16(n))


def pad_reference(x: np.ndarray, w: np.ndarray, n: int) -> np.ndarray:
    """`GemmCoordinatePad_v1{T=n, K}` by the reference evaluator on the first n words of rows ``x`` and ``w`` (N, K); at
    n = 0, which the Definition refuses, the coordinate of K words of +0."""
    if not n:
        return coordinate(x, w, kernel="reference")
    spec = bind(G.GemmCoordinatePad, T=n, K=x.shape[1], **G.GEMM_HOPPER)
    return np.asarray(E.evaluate_batch(spec, np.ascontiguousarray(x[:, :n]), np.ascontiguousarray(w[:, :n]),
                                       kernel="reference"), np.uint16)


def bf16(v: np.ndarray) -> np.ndarray:
    """Finite float32 values rounded to bf16 words (nearest, ties to even)."""
    u = np.asarray(v, np.float32).view(np.uint32).astype(np.uint64)
    return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF).astype(np.uint16)


def clustered(rng: np.random.Generator, shape: tuple, centre: np.ndarray, spread: int = 6) -> np.ndarray:
    """Finite words whose exponent fields sit within ``spread`` of a per-row ``centre`` (so products align and cancel), a
    random mantissa and sign."""
    e = np.clip(centre[..., None] + rng.integers(-spread, spread + 1, size=shape), 0, 254)
    m = rng.integers(0, 128, size=shape)
    s = rng.integers(0, 2, size=shape)
    return ((s << 15) | (e << 7) | m).astype(np.uint16)


def sprinkle(rng: np.random.Generator, a: np.ndarray, p: float) -> np.ndarray:
    hit = rng.random(a.shape) < p
    a = a.copy()
    a[hit] = rng.choice(SPECIALS, int(hit.sum()))
    return a


def sequence(i: int) -> dict:
    """Pool sequence ``i``: its kind is KINDS[i % len(KINDS)], its words drawn from its own seed."""
    kind = KINDS[i % len(KINDS)]
    rng = np.random.default_rng([SEED, i])
    normal = lambda *shape: bf16(rng.standard_normal(shape, np.float32))  # noqa: E731
    Q, K, V = normal(H, D), normal(LCAP, D), normal(LCAP, D)
    z = rng.standard_normal((H, LCAP)) * 2
    P = bf16((np.exp(z) / np.exp(z).sum(1, keepdims=True)).astype(np.float32))
    if kind == "softmax":
        q, k = rng.standard_normal((H, D)), rng.standard_normal((LCAP, D))
        Q, K = bf16(q.astype(np.float32)), bf16(k.astype(np.float32))
        sc = (q @ k.T) / 8.0
        P = bf16((np.exp(sc - sc.max(1, keepdims=True)) / np.exp(sc - sc.max(1, keepdims=True)).sum(1, keepdims=True)).astype(np.float32))
    elif kind in ("wide", "unit", "tiny"):
        def centre(n):
            if kind == "wide":
                return rng.integers(1, 255, size=n)
            if kind == "unit":
                return 127 + rng.integers(-2, 3, size=n)
            return 57 + rng.integers(-4, 5, size=n)
        Q, K, V, P = (clustered(rng, (n, c), centre(n)) for n, c in ((H, D), (LCAP, D), (LCAP, D), (H, LCAP)))
        if kind == "tiny":
            for a in (Q, K, V, P):
                sub = rng.random(a.shape) < 0.25
                a[sub] = (a[sub] & 0x807F) | rng.integers(1, 128, size=int(sub.sum())).astype(np.uint16)
    elif kind == "specials":
        Q, K, V, P = (sprinkle(rng, a, 0.05) for a in (Q, K, V, P))
    elif kind == "cancel_in":
        # every second 8 of each 16-wide k block negates the first 8: Q's heads 1 mod 4 against K's keys 1 mod 4, and P's
        # heads 1 mod 4 against every V column, cancel inside each step
        for s in range(D // 16):
            Q[1::4, 16 * s + 8:16 * s + 16] = Q[1::4, 16 * s:16 * s + 8] ^ SIGN
            K[1::4, 16 * s + 8:16 * s + 16] = K[1::4, 16 * s:16 * s + 8]
        for s in range(LCAP // 16):
            P[1::4, 16 * s + 8:16 * s + 16] = P[1::4, 16 * s:16 * s + 8] ^ SIGN
            V[16 * s + 8:16 * s + 16] = V[16 * s:16 * s + 8]
    elif kind == "cancel_across":
        # the second half of the k range negates the first: the cancellation spans steps, through the truncating adder
        Q[2::4, D // 2:] = Q[2::4, :D // 2] ^ SIGN
        K[2::4, D // 2:] = K[2::4, :D // 2]
        P[2::4, LCAP // 2:] = P[2::4, :LCAP // 2] ^ SIGN
        V[LCAP // 2:] = V[:LCAP // 2]
    elif kind == "uniform":
        Q, K, V, P = (rng.integers(0, 1 << 16, size=a.shape).astype(np.uint16) for a in (Q, K, V, P))
    elif kind == "subacc":
        # products around 2^-130, so PV's accumulator after its first step is a nonzero FP32 subnormal for most words at
        # n <= 16, where K = 32's second step is all +0
        P = clustered(rng, (H, LCAP), 55 + np.arange(H) % 10, spread=2)
        V = clustered(rng, (LCAP, D), 63 + rng.integers(-1, 2, size=LCAP), spread=2)
    return dict(Q=Q, K=K, V=V, P=P, kind=kind)


def pv_operands(P: np.ndarray, V: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows (h, d) of PV at length n: P[h, :n] ++ 0 against V[:n, d] ++ 0, each pv_k(n) wide."""
    h, d = P.shape[0], V.shape[1]
    k = pv_k(n)
    x = np.zeros((h, k), np.uint16)
    x[:, :n] = P[:, :n]
    w = np.zeros((d, k), np.uint16)
    w[:, :n] = V[:n].T
    return np.repeat(x, d, axis=0), np.tile(w, (h, 1))


def _special(a: np.ndarray) -> np.ndarray:
    """Per row: any NaN, infinity, subnormal or -0 word (a +0 is the padding)."""
    e, m = (a >> 7) & 0xFF, a & 0x7F
    return ((e == 0xFF) | ((e == 0) & (m != 0)) | (a == 0x8000)).any(axis=1)


def _sample(rng, x, w, words, n: int) -> tuple:
    sp = np.flatnonzero(_special(x) | _special(w))
    pick = np.concatenate([rng.permutation(sp)[:n // 2], rng.choice(len(words), n - min(len(sp), n // 2), replace=False)])
    return x[pick], w[pick], words[pick]


def expected(i: int) -> dict:
    """Sequence ``i``'s inputs, S and O at every length, and its reference sample's mismatches."""
    seq = sequence(i)
    Q, K, V, P = seq["Q"], seq["K"], seq["V"], seq["P"]
    S = coordinate(np.repeat(Q, LCAP, axis=0), np.tile(K, (H, 1))).reshape(H, LCAP)
    O = np.empty((LCAP, H, D), np.uint16)
    rng = np.random.default_rng([SEED, i, 1])
    ref = {"qk": [0, 0], "pv": [0, 0]}
    xs, ws, got = _sample(rng, np.repeat(Q, LCAP, axis=0), np.tile(K, (H, 1)), S.reshape(-1), REF_PER_SEQ)
    ref["qk"] = [len(got), int((coordinate(xs, ws, kernel="reference") != got).sum())]
    for n in range(1, LCAP + 1):
        x, w = pv_operands(P, V, n)
        O[n - 1] = coordinate(x, w).reshape(H, D)
        xs, ws, got = _sample(rng, x, w, O[n - 1].reshape(-1), REF_PER_SEQ // LCAP)
        ref["pv"] = [ref["pv"][0] + len(got), ref["pv"][1] + int((pad_reference(xs, ws, n) != got).sum())]
    return dict(Q=Q, K=K, V=V, P=P, S=S, O=O, kind=seq["kind"], ref=ref)


def odd() -> tuple[dict, dict]:
    """The GQA case off the frozen shape: inputs, the whole S and O, and its reference sample's mismatches."""
    B, Hq, Hkv, Dd, L, lens = ODD["B"], ODD["H"], ODD["Hkv"], ODD["D"], ODD["L"], np.array(ODD["lens"], np.int32)
    rng = np.random.default_rng([SEED, 999])
    mk = lambda *shape: sprinkle(rng, clustered(rng, shape, 127 + rng.integers(-3, 4, size=shape[:-1])), 0.03)  # noqa: E731
    Q, K, V, P = mk(B, Hq, Dd), mk(B, Hkv, L, Dd), mk(B, Hkv, L, Dd), mk(B, Hq, L)
    S = np.full((B, Hq, L), FILL, np.uint16)
    O = np.empty((B, Hq, Dd), np.uint16)
    g = Hq // Hkv
    ref = {"qk": [0, 0], "pv": [0, 0]}
    for b in range(B):
        n = int(lens[b])
        for h in range(Hq):
            kv = h // g
            if n:
                x, w = np.repeat(Q[b, h][None], n, axis=0), K[b, kv, :n]
                S[b, h, :n] = coordinate(x, w)
                r = coordinate(x, w, kernel="reference")
                ref["qk"] = [ref["qk"][0] + n, ref["qk"][1] + int((r != S[b, h, :n]).sum())]
            x, w = pv_operands(P[b, h][None], V[b, kv], n)
            O[b, h] = coordinate(x, w)
            r = pad_reference(x, w, n)
            ref["pv"] = [ref["pv"][0] + Dd, ref["pv"][1] + int((r != O[b, h]).sum())]
    return dict(Q=Q, K=K, V=V, P=P, lens=lens, S=S, O=O, H=np.int32(Hq), Hkv=np.int32(Hkv)), ref


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str]) -> int:
    out = argv[1] if len(argv) > 1 else "/tmp/attn-ref-check"
    os.makedirs(out, exist_ok=True)
    with mp.get_context("spawn").Pool(int(os.environ.get("PROCS", "8"))) as pool:
        seqs = pool.map(expected, range(NX))
    pool_doc = {k: np.stack([s[k] for s in seqs]) for k in ("Q", "K", "V", "P", "S", "O")}
    pool_doc["kind"] = np.array([s["kind"] for s in seqs])
    np.savez(os.path.join(out, "pool.npz"), **pool_doc)
    odd_doc, odd_ref = odd()
    np.savez(os.path.join(out, "odd.npz"), **odd_doc)
    ref = {k: [sum(s["ref"][k][0] for s in seqs), sum(s["ref"][k][1] for s in seqs)] for k in ("qk", "pv")}
    doc = {
        "definition": "QK^T GemmCoordinate_v2{K=D, GEMM_HOPPER}, PV GemmCoordinatePad_v1{T=n, K=max(32, pad16(n)), GEMM_HOPPER} "
                      "(verity.ml.circuits.ops.gemm), DOT HopperBF16WgmmaDot16_v1",
        "kpv": KPV,
        "twin": "verity.ml.kernels.tc.gemm_coordinate_kernel (evaluate_batch kernel auto)",
        "seed": SEED, "pool": {"sequences": NX, "kinds": list(KINDS), "H": H, "D": D, "L": LCAP,
                               "qk_words": NX * H * LCAP, "pv_words": NX * LCAP * H * D},
        "odd": {k: (list(v) if isinstance(v, tuple) else v) for k, v in ODD.items()} | {"fill": FILL},
        "reference": {"pool": {k: {"words": v[0], "mismatches": v[1]} for k, v in ref.items()},
                      "odd": {k: {"words": v[0], "mismatches": v[1]} for k, v in odd_ref.items()}},
        "files": {f: sha256(os.path.join(out, f)) for f in ("pool.npz", "odd.npz")},
    }
    with open(os.path.join(out, "fixture.json"), "w") as f:
        json.dump(doc, f, indent=1)
    print(json.dumps(doc["reference"]), file=sys.stderr)
    bad = [r["mismatches"] for part in doc["reference"].values() for r in part.values()]
    if any(bad):
        print("REFUSED: the twin and the reference evaluator disagree", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
