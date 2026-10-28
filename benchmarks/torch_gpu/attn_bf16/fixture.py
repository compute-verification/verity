"""attn_bf16's check: each case's inputs and the output buffers the kernels must leave, from the Definition.

Every word the kernels write is `GemmCoordinate_v2{K, GEMM_HOPPER}` (verity.ml.circuits.ops.gemm), evaluated by its numpy
twin (`evaluate_batch`, kernel "auto"), and the twin is held to the Definition's reference evaluator on a sample of each
case's words (every special-valued one first); any disagreement exits 1 before anything is written:

  S[b,h,q,j] = GemmCoordinate{K=D}(Q[b,h,q,:], K[b,kv(h),j,:])                         for q < nq[b], j < len(b,q)
  O[b,h,q,d] = GemmCoordinate{K=kp}(P[b,h,q,:len] ++ 0, V[b,kv(h),:len,d] ++ 0)        for q < nq[b], len = len(b,q)

with len(b,q) = lens[b] - nq[b] + q + 1 (causal: a sequence's queries are its last nq positions), kv(h) = h // (H / Hkv)
and kp = max(kpv, pad16(len)), kpv the case's (KPV; the H200 model's PV pads every KV length 1..32 to K = 32).
Every other word of S and O is 0xA5A5, the poison run.py fills them with: the kernels must leave it.  Inputs the kernels must
not read hold junk (NaN, infinities and huge values): the cache past lens[b], P past len(b,q), and the rows past nq[b].

CASES, each CHECK/<case>.npz (Q, K, V, P, lens, nq: the inputs; S, O: the buffers after the kernels):
  h200_decode        the H200 model's decode step (MQA: 256 query heads of head dim 64 on one KV head, every KV length
                     1..32), 2,048 of its 20,480 sequences, with specials
  h200_ctx96         the H200 model's heads at every KV length 1..96, 512 sequences
  mha_decode_d64     MHA decode (128 heads of head dim 64, each its own KV head, every KV length 1..32)
  decode_d128        head dim 128, GQA with 2 query heads a KV head, every KV length 1..160
  prefill_d64        MQA causal prefill and a prefill chunk, head dim 64
  prefill_d128       GQA (8 query heads a KV head) causal prefill and a chunk, head dim 128
  mha_prefill_d64    MHA causal prefill and a chunk, head dim 64
  specials_d64/d128  NaN, infinities, -0, subnormals, exact cancellation, overflow, and specials in V a shorter query of
                     the tile must not see; decode, prefill and chunks in turn (MQA at 64, GQA at 128)
  long_d64/d128      PV over 4,096 keys
  s16_mqa_d64        the s16 kernels' bounds (every query sees at most 32 keys): MQA, 2 head groups, PV's K = 32, specials
                     and decode, prefill and chunks in turn
  s16_gqa_d128       the same at head dim 128, GQA with 16 query heads a KV head (4 KV heads), PV padded to 16 alone
  kinds_d64          the H200 model's decode shape over KINDS' generators, 4 sequences
                     of each kind, each at every KV length 1..32: normal, softmax rows, clustered exponents, the unit binade,
                     subnormal products, 5% specials, cancellation inside a step and across steps, uniform words, and PV
                     accumulators that are FP32 subnormals after the first step (subacc: K = 32's second step reads them)
FULL, CHECK/<case>.full.npz: the H200 model's whole decode step (20,480 sequences), its inputs made by gen.py from a seed
on both sides, so the file holds each sequence's digest of S and O (and of the inputs, by chunk) instead of the arrays.
CHECK/fixture.json lists the cases, their word counts and the reference sample's.

  uv run python benchmarks/torch_gpu/attn_bf16/fixture.py CHECK [CASE ...]   (env PROCS: FULL's worker processes)
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys

import numpy as np

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops import gemm as G

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen

POISON = 0xA5A5
SPECIALS = gen.SPECIALS
JUNK = np.array([0x7FC0, 0xFFC1, 0x7F80, 0xFF80, 0x7F7F, 0xFF7F], np.uint16)
CHUNK = 1 << 20
BINDING = G.GEMM_HOPPER
KINDS = ("normal", "softmax", "wide", "unit", "tiny", "specials", "cancel_in", "cancel_across", "uniform", "subacc")
KIND_SEED = 20261026


def _decode(lens: np.ndarray) -> np.ndarray:
    return np.ones_like(lens)


def _mixed(lens: np.ndarray) -> np.ndarray:
    """Decode, a whole prompt and half a prompt's chunk, in turn."""
    i = np.arange(len(lens)) % 3
    return np.where(i == 0, 1, np.where(i == 1, lens, np.maximum(1, lens // 2)))


# name: B, H, Hkv, D, Lcap, lens(B), nq(lens), the fraction of special words, sequences with specials (every n-th; 0: none)
CASES = {
    "h200_decode": (2048, 256, 1, 64, 32, lambda B: 1 + np.arange(B) % 32, _decode, 0.02, 8),
    "h200_ctx96": (512, 256, 1, 64, 96, lambda B: 1 + np.arange(B) % 96, _decode, 0.02, 8),
    "mha_decode_d64": (64, 128, 128, 64, 32, lambda B: 1 + np.arange(B) % 32, _decode, 0.02, 8),
    "decode_d128": (160, 8, 4, 128, 160, lambda B: 1 + np.arange(B), _decode, 0.02, 8),
    "prefill_d64": (6, 16, 1, 64, 64, lambda B: np.array([64, 33, 17, 16, 1, 48]), lambda lens: np.array([64, 33, 17, 16, 1, 20]), 0.0, 0),
    "prefill_d128": (4, 16, 2, 128, 200, lambda B: np.array([200, 129, 31, 7]), lambda lens: np.array([200, 129, 31, 3]), 0.0, 0),
    "mha_prefill_d64": (6, 4, 4, 64, 64, lambda B: np.array([64, 33, 17, 16, 1, 48]), lambda lens: np.array([64, 33, 17, 16, 1, 20]), 0.0, 0),
    "specials_d64": (48, 8, 1, 64, 48, lambda B: 1 + np.arange(B), _mixed, 0.05, 1),
    "specials_d128": (24, 8, 1, 128, 40, lambda B: 17 + np.arange(B), _mixed, 0.05, 1),
    "long_d64": (4, 8, 1, 64, 4096, lambda B: np.array([4096, 4095, 2049, 1]), _decode, 0.0, 0),
    "long_d128": (2, 2, 2, 128, 4100, lambda B: np.array([4100, 3000]), lambda lens: np.array([1, 40]), 0.001, 1),
    "s16_mqa_d64": (36, 32, 1, 64, 32, lambda B: 1 + np.arange(B) % 32, _mixed, 0.05, 2),
    "s16_gqa_d128": (24, 64, 4, 128, 32, lambda B: np.array([32, 31, 30, 17, 16, 15, 9, 8, 1, 2, 3, 24] * 2), _mixed, 0.05, 1),
    "kinds_d64": (4 * len(KINDS) * 32, 256, 1, 64, 32, lambda B: 1 + np.arange(B) % 32, _decode, 0.0, 0),
}
# PV's least K by case (default 0: pad16(len) alone)
KPV = {"h200_decode": 32, "mha_decode_d64": 32, "specials_d64": 48, "h200_decode_full": 32, "s16_mqa_d64": 32, "kinds_d64": 32}
# name: B, H, Hkv, D, Lcap, seed (gen.py's inputs: lens 1 + b mod Lcap, one query each)
FULL = {"h200_decode_full": (20480, 256, 1, 64, 32, 20261026)}
SEQ_CHUNK = 256


def pv_k(li: np.ndarray, kpv: int) -> np.ndarray:
    return np.maximum(kpv, -(-np.asarray(li) // 16) * 16)


def bf16(v: np.ndarray) -> np.ndarray:
    """``v``'s float32 words rounded to bf16 by truncation: random finite operands, any rounding serves."""
    return (np.asarray(v, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def seen(lens: np.ndarray, nq: np.ndarray, b: int, q: int) -> int:
    return int(lens[b] - nq[b] + q + 1)


def clustered(rng: np.random.Generator, shape: tuple, centre: np.ndarray, spread: int = 6) -> np.ndarray:
    """Finite words whose exponent fields sit within ``spread`` of a per-row ``centre`` (so products align and cancel), a
    random mantissa and sign."""
    e = np.clip(centre[..., None] + rng.integers(-spread, spread + 1, size=shape), 0, 254)
    m = rng.integers(0, 128, size=shape)
    s = rng.integers(0, 2, size=shape)
    return ((s << 15) | (e << 7) | m).astype(np.uint16)


def kind_sequence(i: int, H: int = 256, D: int = 64, L: int = 32) -> dict:
    """Sequence ``i`` of kinds_d64: Q (H, D), K and V (L, D) and P (H, L) of KINDS[i % len(KINDS)], from its own seed."""
    kind = KINDS[i % len(KINDS)]
    rng = np.random.default_rng([KIND_SEED, i])
    normal = lambda *shape: bf16(rng.standard_normal(shape, np.float32))  # noqa: E731
    Q, K, V = normal(H, D), normal(L, D), normal(L, D)
    z = rng.standard_normal((H, L)) * 2
    P = bf16((np.exp(z) / np.exp(z).sum(1, keepdims=True)).astype(np.float32))
    if kind == "softmax":
        q, k = rng.standard_normal((H, D)), rng.standard_normal((L, D))
        Q, K = bf16(q.astype(np.float32)), bf16(k.astype(np.float32))
        sc = (q @ k.T) / 8.0
        e = np.exp(sc - sc.max(1, keepdims=True))
        P = bf16((e / e.sum(1, keepdims=True)).astype(np.float32))
    elif kind in ("wide", "unit", "tiny"):
        def centre(n):
            if kind == "wide":
                return rng.integers(1, 255, size=n)
            if kind == "unit":
                return 127 + rng.integers(-2, 3, size=n)
            return 57 + rng.integers(-4, 5, size=n)
        Q, K, V, P = (clustered(rng, (n, c), centre(n)) for n, c in ((H, D), (L, D), (L, D), (H, L)))
        if kind == "tiny":
            for a in (Q, K, V, P):
                sub = rng.random(a.shape) < 0.25
                a[sub] = (a[sub] & 0x807F) | rng.integers(1, 128, size=int(sub.sum())).astype(np.uint16)
    elif kind == "specials":
        for a in (Q, K, V, P):
            hit = rng.random(a.shape) < 0.05
            a[hit] = rng.choice(SPECIALS, int(hit.sum()))
    elif kind == "cancel_in":
        # every second 8 of each 16-wide k segment negates the first 8: Q's heads 1 mod 4 against K's keys 1 mod 4, and P's
        # heads 1 mod 4 against every V column, cancel inside each step
        for s in range(D // 16):
            Q[1::4, 16 * s + 8:16 * s + 16] = Q[1::4, 16 * s:16 * s + 8] ^ 0x8000
            K[1::4, 16 * s + 8:16 * s + 16] = K[1::4, 16 * s:16 * s + 8]
        for s in range(L // 16):
            P[1::4, 16 * s + 8:16 * s + 16] = P[1::4, 16 * s:16 * s + 8] ^ 0x8000
            V[16 * s + 8:16 * s + 16] = V[16 * s:16 * s + 8]
    elif kind == "cancel_across":
        # the second half of the k range negates the first: the cancellation spans steps, through the truncating adder
        Q[2::4, D // 2:] = Q[2::4, :D // 2] ^ 0x8000
        K[2::4, D // 2:] = K[2::4, :D // 2]
        P[2::4, L // 2:] = P[2::4, :L // 2] ^ 0x8000
        V[L // 2:] = V[:L // 2]
    elif kind == "uniform":
        Q, K, V, P = (rng.integers(0, 1 << 16, size=a.shape).astype(np.uint16) for a in (Q, K, V, P))
    elif kind == "subacc":
        # products around 2^-130, so PV's accumulator after its first step is a nonzero FP32 subnormal for most words at
        # n <= 16, where K = 32's second step is all +0
        P = clustered(rng, (H, L), 55 + np.arange(H) % 10, spread=2)
        V = clustered(rng, (L, D), 63 + rng.integers(-1, 2, size=L), spread=2)
    return dict(Q=Q, K=K, V=V, P=P)


def build(name: str):
    """The case's inputs: random normal Q, K and V, P uniform in [0, 1), specials in the live words of every n-th sequence
    (with exact cancellations and overflows), or kinds_d64's generators; and junk wherever the kernels must not read."""
    B, H, Hkv, D, Lcap, lens_f, nq_f, frac, every = CASES[name]
    rng = np.random.default_rng(int.from_bytes(name.encode(), "little") % 2**32)
    lens = np.asarray(lens_f(B), np.int32)
    nq = np.asarray(nq_f(lens), np.int32)
    Mq = int(nq.max())
    assert len(lens) == B and (nq >= 1).all() and (nq <= lens).all() and (lens <= Lcap).all() and H % Hkv == 0, name
    if name == "kinds_d64":  # sequence b is kind sequence b // Lcap at KV length 1 + b mod Lcap
        seqs = [kind_sequence(i, H, D, Lcap) for i in range(B // Lcap)]
        Q, K, V, P = (np.stack([s[k] for s in seqs])[np.arange(B) // Lcap] for k in "QKVP")
        Q, K, V, P = Q[:, :, None], K[:, None], V[:, None], P[:, :, None]
    else:
        Q = bf16(rng.standard_normal((B, H, Mq, D), np.float32))
        K = bf16(rng.standard_normal((B, Hkv, Lcap, D), np.float32))
        V = bf16(rng.standard_normal((B, Hkv, Lcap, D), np.float32))
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
        kh = h // (H // Hkv)
        if L >= 2:
            # QK^T: the first query against key 0 cancels exactly in every 16-wide k segment (its second 8 products negate the
            # first 8), and against key 1 every product is |q| x the largest finite, so the sum leaves FP32's range
            q, k0 = Q[b, h, 0].reshape(-1, 2, 8), K[b, kh, 0].reshape(-1, 2, 8)
            q[:, 1], k0[:, 1] = q[:, 0] ^ 0x8000, k0[:, 0]
            K[b, kh, 1] = 0x7F7F | (Q[b, h, 0] & 0x8000)
            # PV: the last query's first two products cancel at the largest finite in the even columns of V
            V[b, kh, 0, :] = 0x7F7F
            V[b, kh, 1, 0::2] = V[b, kh, 0, 0::2] ^ 0x8000
            P[b, h, M - 1, :2] = 0x3F80
        # specials in V at the sequence's last key: every query but the last of a prefill must not see them
        V[b, kh, L - 1, : D // 2] = rng.choice(JUNK, D // 2)
    for b in range(B):
        L, M = int(lens[b]), int(nq[b])
        K[b, :, L:] = rng.choice(JUNK, K[b, :, L:].shape)
        V[b, :, L:] = rng.choice(JUNK, V[b, :, L:].shape)
        Q[b, :, M:] = rng.choice(JUNK, Q[b, :, M:].shape)
        P[b, :, M:] = rng.choice(JUNK, P[b, :, M:].shape)
        for q in range(M):
            P[b, :, q, seen(lens, nq, b, q):] = rng.choice(JUNK, P[b, :, q, seen(lens, nq, b, q):].shape)
    return dict(Q=Q, K=K, V=V, P=P, lens=lens, nq=nq, kpv=np.int32(KPV.get(name, 0))), D


def coordinate(x: np.ndarray, w: np.ndarray, kernel: str = "auto") -> np.ndarray:
    """GemmCoordinate{K, GEMM_HOPPER} on rows of ``x`` and ``w`` (N, K)."""
    spec = bind(G.GemmCoordinate, K=x.shape[1], **BINDING)
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
    kpv = int(inp["kpv"])
    B, H, Mq, Lcap = P.shape
    kvh = np.arange(H) // (H // K.shape[1])
    S = np.full((B, H, Mq, Lcap), POISON, np.uint16)
    O = np.full((B, H, Mq, D), POISON, np.uint16)
    bi, qi, li = pairs(lens, nq)
    hstep = max(1, CHUNK // max(1, len(bi) * Lcap))
    for h0 in range(0, H, hstep):
        hs = np.arange(h0, min(H, h0 + hstep))
        hh = len(hs)
        x = np.broadcast_to(Q[bi[:, None], hs[None, :], qi[:, None]][:, :, None, :], (len(bi), hh, Lcap, D)).reshape(-1, D)
        w = K[bi[:, None], kvh[hs][None, :]].reshape(-1, D)
        live = np.broadcast_to((np.arange(Lcap)[None, :] < li[:, None])[:, None, :], (len(bi), hh, Lcap)).reshape(-1)
        words = np.full(len(live), POISON, np.uint16)
        words[live] = coordinate(np.ascontiguousarray(x[live]), np.ascontiguousarray(w[live]))
        words = words.reshape(len(bi), hh, Lcap)
        S[bi[:, None], hs[None, :], qi[:, None]] = words
        sample.append(_pick("qk", x, w, words.reshape(-1), live, rng))
    for kp in sorted(set(pv_k(li, kpv).tolist())):
        sel = np.flatnonzero(pv_k(li, kpv) == kp)
        b, q, n_ = bi[sel], qi[sel], li[sel]
        keys = np.arange(kp)
        mask = keys[None, :] < n_[:, None]
        kidx = np.minimum(keys, Lcap - 1)
        hstep = max(1, CHUNK // max(1, len(sel) * D))
        for h0 in range(0, H, hstep):
            hs = np.arange(h0, min(H, h0 + hstep))
            hh = len(hs)
            x = np.where(mask[:, None, :], P[b[:, None], hs[None, :], q[:, None]][:, :, kidx], 0).astype(np.uint16)          # (n, hh, kp)
            w = np.where(mask[:, None, :, None], V[b[:, None], kvh[hs][None, :]][:, :, kidx, :], 0).astype(np.uint16)       # (n, hh, kp, D)
            xr = np.broadcast_to(x[:, :, None, :], (len(sel), hh, D, kp)).reshape(-1, kp)
            wr = np.ascontiguousarray(w.transpose(0, 1, 3, 2)).reshape(-1, kp)
            words = coordinate(xr, wr).reshape(len(sel), hh, D)
            O[b[:, None], hs[None, :], q[:, None]] = words
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


def full_inputs(name: str, b0: int, b1: int) -> dict:
    B, H, Hkv, D, Lcap, seed = FULL[name]
    inp = {k: gen.array(seed, k, s, b0, b1) for k, s in gen.shapes(B, H, Hkv, D, Lcap).items()}
    inp["lens"], inp["nq"] = gen.lens_nq(B, Lcap, b0, b1)
    inp["kpv"] = np.int32(KPV.get(name, 0))
    return inp


def _full_chunk(job: tuple) -> tuple:
    name, b0, b1 = job
    D = FULL[name][3]
    inp = full_inputs(name, b0, b1)
    sample: list = []
    S, O = expected(inp, D, sample, np.random.default_rng(b0))
    ref = reference_sample(sample) if b0 == 0 else None
    ins = {k: gen.digests(inp[k][None])[0] for k in ("Q", "K", "V", "P")}
    return b0, gen.digests(S), gen.digests(O), ins, ref


def full(check: str, name: str) -> dict:
    """FULL's case: each sequence's S and O digests, the inputs' digests by SEQ_CHUNK sequences, and the first chunk's
    reference sample."""
    B, H, Hkv, D, Lcap, seed = FULL[name]
    jobs = [(name, b0, min(B, b0 + SEQ_CHUNK)) for b0 in range(0, B, SEQ_CHUNK)]
    with mp.get_context("fork").Pool(int(os.environ.get("PROCS", os.cpu_count() or 1))) as pool:
        got = sorted(pool.map(_full_chunk, jobs), key=lambda r: r[0])
    Sd, Od = np.concatenate([r[1] for r in got]), np.concatenate([r[2] for r in got])
    ins = {k: np.stack([r[3][k] for r in got]) for k in ("Q", "K", "V", "P")}
    kpv = KPV.get(name, 0)
    np.savez(os.path.join(check, f"{name}.full.npz"), S=Sd, O=Od, **{f"in_{k}": v for k, v in ins.items()},
             shape=np.array([B, H, Hkv, D, Lcap, seed, SEQ_CHUNK, kpv], np.int64))
    lens, _ = gen.lens_nq(B, Lcap)
    return {"B": B, "H": H, "Hkv": Hkv, "D": D, "Mq": 1, "Lcap": Lcap, "seed": seed, "seq_chunk": SEQ_CHUNK, "kpv": kpv,
            "lens": [int(lens.min()), int(lens.max())], "queries": B, "qk_words": H * int(lens.sum()), "pv_words": H * B * D,
            "pv_k": sorted(set(pv_k(lens, kpv).tolist())), "reference": got[0][4]}


def main(argv: list[str]) -> int:
    check, names = argv[1], argv[2:] or [*CASES, *FULL]
    os.makedirs(check, exist_ok=True)
    summary = {"definition": bind(G.GemmCoordinate, K=64, **BINDING).id, "binding": "GEMM_HOPPER",
               "step": BINDING["DOT"].id, "poison": POISON, "cases": {}}
    bad = False
    for name in names:
        if name in FULL:
            row = full(check, name)
            bad |= any(r["mismatches"] for r in row["reference"].values())
            summary["cases"][name] = row
            print(name, json.dumps(row), flush=True)
            continue
        inp, D = build(name)
        sample: list = []
        S, O = expected(inp, D, sample, np.random.default_rng(7))
        ref = reference_sample(sample)
        bad |= any(r["mismatches"] for r in ref.values())
        np.savez_compressed(os.path.join(check, f"{name}.npz"), S=S, O=O, **inp)
        B, H, Mq, Lcap = inp["P"].shape
        _, _, li = pairs(inp["lens"], inp["nq"])
        summary["cases"][name] = {"B": B, "H": H, "Hkv": int(inp["K"].shape[1]), "D": D, "Mq": Mq, "Lcap": Lcap,
                                  "kpv": int(inp["kpv"]), "lens": [int(inp["lens"].min()), int(inp["lens"].max())],
                                  "queries": len(li), "qk_words": H * int(li.sum()), "pv_words": H * len(li) * D,
                                  "pv_k": sorted(set(pv_k(li, inp["kpv"]).tolist())), "reference": ref}
        print(name, json.dumps(summary["cases"][name]), flush=True)
    with open(os.path.join(check, "fixture.json"), "w") as f:
        json.dump(summary, f, indent=1)
    if bad:
        print("REFUSED: the twin and the reference evaluator disagree", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
