"""attn_bf16's check: each case's inputs and the output buffers the kernels must leave, from the Definition.

Every word the kernels write is `GemmCoordinate_v2{K, GEMM_SM120_BF16}` (verity.ml.circuits.ops.gemm), evaluated by its
numpy twin (`evaluate_batch`, kernel "auto"), and the twin is held to the Definition's reference evaluator on a sample of
each case's words (every special-valued one first); any disagreement exits 1 before anything is written:

  S[b,h,q,j] = GemmCoordinate{K=D}(Q[b,h,q,:], K[b,h,j,:])                                for q < nq[b], j < len(b,q)
  O[b,h,q,d] = GemmCoordinate{K=pad16(len)}(P[b,h,q,:len] ++ 0, V[b,h,:len,d] ++ 0)       for q < nq[b], len = len(b,q)

with len(b,q) = lens[b] - nq[b] + q + 1 (causal: a sequence's queries are its last nq positions).  Every other word of S
and O is 0xA5A5, the poison run.py fills them with: the kernels must leave it.  Inputs the kernels must not read hold
junk (NaN, infinities and huge values): the cache past lens[b], P past len(b,q), and the rows past nq[b].

CASES: the model's decode step (`decode_d64`: 1,023 sequences, 128 heads of head dim 64, every KV length 1..32), head dim
128 at every KV length 1..160, causal prefill and a prefill chunk at both head dims, specials at both (NaN, infinities,
-0, subnormals, exact cancellation, overflow, and specials in V a shorter query of the tile must not see), and PV over
4,096 keys.  Each writes CHECK/<case>.npz (Q, K, V, P, lens, nq: the inputs; S, O: the buffers after the kernels) and
CHECK/fixture.json the cases, their word counts and the reference sample's.

  uv run python benchmarks/torch_gpu/attn_bf16/fixture.py CHECK [CASE ...]
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops import gemm as G

POISON = 0xA5A5
#: bf16 words the junk past a length and the specials draw from: zeros, infinities, NaNs, subnormals, the extremes
SPECIALS = np.array([0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0x7F81, 0xFFFF, 0x0001, 0x8001, 0x007F, 0x0080, 0x7F7F,
                     0xFF7F, 0x3F80, 0xBF80], np.uint16)
JUNK = np.array([0x7FC0, 0xFFC1, 0x7F80, 0xFF80, 0x7F7F, 0xFF7F], np.uint16)
CHUNK = 1 << 20


def _decode(lens: np.ndarray) -> np.ndarray:
    return np.ones_like(lens)


def _mixed(lens: np.ndarray) -> np.ndarray:
    """Decode, a whole prompt and half a prompt's chunk, in turn."""
    i = np.arange(len(lens)) % 3
    return np.where(i == 0, 1, np.where(i == 1, lens, np.maximum(1, lens // 2)))


# name: B, H, D, Lcap, lens(B), nq(lens), the fraction of special words, sequences with specials (every n-th; 0: none)
CASES = {
    "decode_d64": (1023, 128, 64, 32, lambda B: 1 + np.arange(B) % 32, _decode, 0.02, 8),
    "decode_d128": (160, 8, 128, 160, lambda B: 1 + np.arange(B), _decode, 0.02, 8),
    "prefill_d64": (6, 4, 64, 64, lambda B: np.array([64, 33, 17, 16, 1, 48]), lambda lens: np.array([64, 33, 17, 16, 1, 20]), 0.0, 0),
    "prefill_d128": (4, 2, 128, 200, lambda B: np.array([200, 129, 31, 7]), lambda lens: np.array([200, 129, 31, 3]), 0.0, 0),
    "specials_d64": (48, 4, 64, 48, lambda B: 1 + np.arange(B), _mixed, 0.05, 1),
    "specials_d128": (24, 4, 128, 40, lambda B: 17 + np.arange(B), _mixed, 0.05, 1),
    "long_d64": (4, 2, 64, 4096, lambda B: np.array([4096, 4095, 2049, 1]), _decode, 0.0, 0),
    "long_d128": (2, 2, 128, 4100, lambda B: np.array([4100, 3000]), lambda lens: np.array([1, 40]), 0.001, 1),
}


def bf16(v: np.ndarray) -> np.ndarray:
    """``v``'s float32 words rounded to bf16 by truncation: random finite operands, any rounding serves."""
    return (np.asarray(v, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def seen(lens: np.ndarray, nq: np.ndarray, b: int, q: int) -> int:
    return int(lens[b] - nq[b] + q + 1)


def build(name: str):
    """The case's inputs: random normal Q, K and V, P uniform in [0, 1), specials in the live words of every n-th sequence
    (with exact cancellations and overflows), and junk wherever the kernels must not read."""
    B, H, D, Lcap, lens_f, nq_f, frac, every = CASES[name]
    rng = np.random.default_rng(int.from_bytes(name.encode(), "little") % 2**32)
    lens = np.asarray(lens_f(B), np.int32)
    nq = np.asarray(nq_f(lens), np.int32)
    Mq = int(nq.max())
    assert len(lens) == B and (nq >= 1).all() and (nq <= lens).all() and (lens <= Lcap).all(), name
    Q = bf16(rng.standard_normal((B, H, Mq, D), np.float32))
    K = bf16(rng.standard_normal((B, H, Lcap, D), np.float32))
    V = bf16(rng.standard_normal((B, H, Lcap, D), np.float32))
    P = bf16(rng.random((B, H, Mq, Lcap), np.float32))
    for b in range(B):
        if not (every and b % every == 0):
            continue
        L, M = int(lens[b]), int(nq[b])
        for arr, rows in ((Q, M), (K, L), (V, L), (P, M)):
            live = arr[b, :, :rows]
            hit = rng.random(live.shape) < frac
            live[hit] = rng.choice(SPECIALS, int(hit.sum()))
        h = b % H
        if L >= 2:
            # QK^T: the first query against key 0 cancels exactly in every 16-wide k block (its second 8 products negate the
            # first 8), and against key 1 every product is |q| x the largest finite, so the sum leaves FP32's range
            q, k0 = Q[b, h, 0].reshape(-1, 2, 8), K[b, h, 0].reshape(-1, 2, 8)
            q[:, 1], k0[:, 1] = q[:, 0] ^ 0x8000, k0[:, 0]
            K[b, h, 1] = 0x7F7F | (Q[b, h, 0] & 0x8000)
            # PV: the last query's first two products cancel at the largest finite in the even columns of V
            V[b, h, 0, :] = 0x7F7F
            V[b, h, 1, 0::2] = V[b, h, 0, 0::2] ^ 0x8000
            P[b, h, M - 1, :2] = 0x3F80
        # specials in V at the sequence's last key: every query but the last of a prefill must not see them
        V[b, h, L - 1, : D // 2] = rng.choice(JUNK, D // 2)
    for b in range(B):
        L, M = int(lens[b]), int(nq[b])
        K[b, :, L:] = rng.choice(JUNK, K[b, :, L:].shape)
        V[b, :, L:] = rng.choice(JUNK, V[b, :, L:].shape)
        Q[b, :, M:] = rng.choice(JUNK, Q[b, :, M:].shape)
        P[b, :, M:] = rng.choice(JUNK, P[b, :, M:].shape)
        for q in range(M):
            P[b, :, q, seen(lens, nq, b, q):] = rng.choice(JUNK, P[b, :, q, seen(lens, nq, b, q):].shape)
    return dict(Q=Q, K=K, V=V, P=P, lens=lens, nq=nq), D


def coordinate(x: np.ndarray, w: np.ndarray, kernel: str = "auto") -> np.ndarray:
    """GemmCoordinate{K, GEMM_SM120_BF16} on rows of ``x`` and ``w`` (N, K)."""
    spec = bind(G.GemmCoordinate, K=x.shape[1], **G.GEMM_SM120_BF16)
    out = np.empty(x.shape[0], np.uint16)
    for i in range(0, x.shape[0], CHUNK):
        out[i:i + CHUNK] = E.evaluate_batch(spec, x[i:i + CHUNK], w[i:i + CHUNK], kernel=kernel)
    return out


def pairs(lens: np.ndarray, nq: np.ndarray):
    """Every live (sequence, query) with its length."""
    bi, qi, li = [], [], []
    for b in range(len(lens)):
        for q in range(int(nq[b])):
            bi.append(b), qi.append(q), li.append(seen(lens, nq, b, q))
    return np.array(bi), np.array(qi), np.array(li)


def _pick(kind: str, x: np.ndarray, w: np.ndarray, words: np.ndarray, live: np.ndarray, rng: np.random.Generator, n: int = 24):
    """Up to ``n`` live rows with a special operand word and ``n`` random live ones, for `reference_sample`."""
    idx = np.flatnonzero(live)
    if not idx.size:
        return kind, x[:0], w[:0], words[:0]
    cand = rng.choice(idx, min(idx.size, 64 * n), replace=False)
    spec = cand[(np.isin(x[cand], SPECIALS[2:]) | np.isin(w[cand], SPECIALS[2:])).any(axis=1)][:n]
    p = np.concatenate([spec, rng.choice(idx, n)])
    return kind, x[p], w[p], words[p]


def expected(inp: dict, D: int, sample: list, rng: np.random.Generator):
    """S and O as the kernels must leave them, with a sample of their words' operand rows appended to ``sample``."""
    Q, K, V, P, lens, nq = (inp[k] for k in ("Q", "K", "V", "P", "lens", "nq"))
    B, H, Mq, Lcap = P.shape
    S = np.full((B, H, Mq, Lcap), POISON, np.uint16)
    O = np.full((B, H, Mq, D), POISON, np.uint16)
    bi, qi, li = pairs(lens, nq)
    hstep = max(1, CHUNK // max(1, len(bi) * Lcap))
    for h0 in range(0, H, hstep):
        hs = slice(h0, min(H, h0 + hstep))
        hh = hs.stop - h0
        x = np.broadcast_to(Q[bi, hs, qi][:, :, None, :], (len(bi), hh, Lcap, D)).reshape(-1, D)
        w = K[bi, hs].reshape(-1, D)
        words = coordinate(x, w).reshape(len(bi), hh, Lcap)
        for n in range(len(bi)):
            S[bi[n], hs, qi[n], : li[n]] = words[n, :, : li[n]]
        live = np.broadcast_to((np.arange(Lcap)[None, :] < li[:, None])[:, None, :], words.shape).reshape(-1)
        sample.append(_pick("qk", x, w, words.reshape(-1), live, rng))
    for kp in sorted({int(-(-n // 16) * 16) for n in li}):
        sel = np.flatnonzero(-(-li // 16) * 16 == kp)
        b, q, n_ = bi[sel], qi[sel], li[sel]
        keys = np.arange(kp)
        mask = keys[None, :] < n_[:, None]
        kidx = np.minimum(keys, Lcap - 1)
        hstep = max(1, CHUNK // max(1, len(sel) * D))
        for h0 in range(0, H, hstep):
            hs = slice(h0, min(H, h0 + hstep))
            hh = hs.stop - h0
            x = np.where(mask[:, None, :], P[b, hs, q][:, :, kidx], 0).astype(np.uint16)          # (n, hh, kp)
            w = np.where(mask[:, None, :, None], V[b, hs][:, :, kidx, :], 0).astype(np.uint16)    # (n, hh, kp, D)
            xr = np.broadcast_to(x[:, :, None, :], (len(sel), hh, D, kp)).reshape(-1, kp)
            wr = np.ascontiguousarray(w.transpose(0, 1, 3, 2)).reshape(-1, kp)
            words = coordinate(xr, wr).reshape(len(sel), hh, D)
            O[b, hs, q] = words
            sample.append(_pick("pv", xr, wr, words.reshape(-1), np.ones(xr.shape[0], bool), rng))
    return S, O


def reference_sample(sample: list) -> dict:
    """The twin's sampled words against the Definition's reference evaluator."""
    out = {}
    for kind in ("qk", "pv"):
        checked = mism = 0
        for _, x, w, words in (s for s in sample if s[0] == kind):
            if len(words):
                mism += int((coordinate(np.ascontiguousarray(x), np.ascontiguousarray(w), kernel="reference") != words).sum())
                checked += len(words)
        out[kind] = {"words": checked, "mismatches": mism}
    return out


def main(argv: list[str]) -> int:
    check, names = argv[1], argv[2:] or list(CASES)
    os.makedirs(check, exist_ok=True)
    summary = {"definition": bind(G.GemmCoordinate, K=64, **G.GEMM_SM120_BF16).id, "binding": "GEMM_SM120_BF16",
               "step": G.GEMM_SM120_BF16["DOT"].id, "poison": POISON, "cases": {}}
    bad = False
    for name in names:
        inp, D = build(name)
        sample: list = []
        S, O = expected(inp, D, sample, np.random.default_rng(7))
        ref = reference_sample(sample)
        bad |= any(r["mismatches"] for r in ref.values())
        np.savez(os.path.join(check, f"{name}.npz"), S=S, O=O, **inp)
        B, H, Mq, Lcap = inp["P"].shape
        _, _, li = pairs(inp["lens"], inp["nq"])
        summary["cases"][name] = {"B": B, "H": H, "D": D, "Mq": Mq, "Lcap": Lcap, "lens": [int(inp["lens"].min()), int(inp["lens"].max())],
                                  "queries": len(li), "qk_words": H * int(li.sum()), "pv_words": H * len(li) * D,
                                  "pv_k": sorted({int(-(-n // 16) * 16) for n in li}), "reference": ref}
        print(name, json.dumps(summary["cases"][name]), flush=True)
    with open(os.path.join(check, "fixture.json"), "w") as f:
        json.dump(summary, f, indent=1)
    if bad:
        print("REFUSED: the twin and the reference evaluator disagree", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
