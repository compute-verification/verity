"""Ligerito polynomial commitment: prove / verify a multilinear evaluation f(z) = v of an N = 2^n coefficient vector
over BabyBear at z in F_{p^6}^n (paper section 6, with the tensor-claim bookkeeping of ``claims.py``).

Coins protocol (``transcript.py``; Fiat-Shamir by default; every ``challenge``/``indices`` is one verifier coin):
    header, z | root_1 | v | sumcheck round 1 (k'_1 x [g_j -> r_j]) |
    for i = 2..L:  root_i | S_{i-1} <- coin | opened rows of U_{i-1} at S_{i-1} (+ multi-path) | alpha <- coin |
                   sumcheck round i (k'_i x [g_j -> r_j]) |
    y_L (the final vector, 2^{k_L} extension elements) | S_L <- coin | opened rows of U_L (+ multi-path)

Verifier checks: every opened row against its root (Blake3 multi-path); g_j(0) + g_j(1) = running claim and
claim <- g_j(r_j); alpha-batching of the running claim with the |S| opened-row claims  < eta_s^x , y_{i-1} > =
< U_{i-1}[s, :], r-bar_{i-1} >;  finally  P_{y_L}(eta_s) = < U_L[s, :], r-bar_L >  for s in S_L and
sum_t coef_t < rowpart_t , y_L > = running claim.

Prover memory: f (int32, N x 4 B), the round-1 codeword (4 N x 4 B), the round-1 Merkle tree (32 B per codeword
row); later rounds are <= 1/64 of that.  Nothing of size N is materialised in the extension field.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ._ligero import commit_gpu as _commit_gpu
from ._ligero import merkle as ligero_merkle
from . import claims as cl
from . import device as dev
from . import ext
from . import merkle_multi
from . import stream
from .dims import Dims
from .ext import D, P
from .kernels import col_contract, row_contract
from .linalg import _powers_rows as _powers_rows_np
from .linalg import contract_columns_ext, contract_rows_ext, eq_planes, matmul_mod_p_np, powers_matrix
from .ntt_gpu import Encoder
from .transcript import Coins, FiatShamirCoins

HASH = "blake3"


# --------------------------------------------------------------------------------------------- proof container

@dataclass
class RoundOpen:
    rows: np.ndarray            # (|S|, W) int64 base words of the opened codeword rows (W = C_i or 6 C_i)
    siblings: bytes             # multi-path digests (merkle_multi.plan order)


@dataclass
class Proof:
    n: int
    kprime: list[int]
    queries: list[int]
    rate_log2: list[int]                    # per committed round
    splits: list[tuple[int, int]]           # (n1, n2) of each round's encoder (position -> eta map)
    v: np.ndarray                           # (6,); commit-first: (m, 6), one per evaluation point
    roots: list[bytes]
    sumcheck: list[list[np.ndarray]]        # per round: k'_i messages (3, 6)
    opens: list[RoundOpen]                  # per committed round
    final_y: np.ndarray                     # (2^{k_L}, 6)
    coins: int = 0
    order: str | None = None                # None = the fused order of ``prove``; "commit-first" = ``commit`` + ``prove_eval``

    @property
    def dims(self) -> Dims:
        return Dims(self.n, list(self.kprime), list(self.queries), list(self.rate_log2))

    # --- bytes ------------------------------------------------------------------------------------------
    def to_bytes(self) -> bytes:
        head = {"n": self.n, "kprime": self.kprime, "queries": self.queries, "rate_log2": self.rate_log2,
                "splits": self.splits, "coins": self.coins, "hash": HASH, "beta": ext.BETA, "ext_degree": D,
                "rows_shape": [list(o.rows.shape) for o in self.opens], "sib_len": [len(o.siblings) for o in self.opens],
                "final_len": int(self.final_y.shape[0])}
        if self.order is not None:
            head["order"] = self.order
            head["claims"] = int(np.asarray(self.v).reshape(-1, D).shape[0])
        hb = json.dumps(head, separators=(",", ":")).encode()
        parts = [len(hb).to_bytes(4, "little"), hb, _ext_bytes(self.v)]
        for i in range(len(self.kprime)):
            parts.append(self.roots[i])
            for g in self.sumcheck[i]:
                parts.append(_ext_bytes(g))
        for o in self.opens:
            parts.append(np.ascontiguousarray(o.rows.astype(np.uint32)).tobytes())
            parts.append(o.siblings)
        parts.append(_ext_bytes(self.final_y))
        return b"".join(parts)

    @staticmethod
    def from_bytes(buf: bytes) -> "Proof":
        hl = int.from_bytes(buf[:4], "little")
        head = json.loads(buf[4:4 + hl].decode())
        pos = 4 + hl
        order = head.get("order")
        v, pos = _ext_read(buf, pos, (head["claims"],) if order is not None else ())
        L = len(head["kprime"])
        roots, sumcheck = [], []
        for i in range(L):
            roots.append(buf[pos:pos + 32])
            pos += 32
            msgs = []
            for _ in range(head["kprime"][i]):
                g, pos = _ext_read(buf, pos, (3,))
                msgs.append(g)
            sumcheck.append(msgs)
        opens = []
        for shape, sl in zip(head["rows_shape"], head["sib_len"]):
            cnt = shape[0] * shape[1]
            rows = np.frombuffer(buf[pos:pos + 4 * cnt], dtype="<u4").astype(np.int64).reshape(shape)
            pos += 4 * cnt
            sib = buf[pos:pos + sl]
            pos += sl
            opens.append(RoundOpen(rows, sib))
        fy, pos = _ext_read(buf, pos, (head["final_len"],))
        assert pos == len(buf), (pos, len(buf))
        return Proof(head["n"], head["kprime"], head["queries"], head["rate_log2"], [tuple(s) for s in head["splits"]],
                     v, roots, sumcheck, opens, fy, head.get("coins", 0), order)


def _ext_bytes(a: np.ndarray) -> bytes:
    return np.ascontiguousarray(np.asarray(a, dtype=np.int64) % P).astype(np.uint32).tobytes()


def _ext_read(buf: bytes, pos: int, shape) -> tuple[np.ndarray, int]:
    cnt = D
    for s in shape:
        cnt *= s
    a = np.frombuffer(buf[pos:pos + 4 * cnt], dtype="<u4").astype(np.int64).reshape(tuple(shape) + (D,))
    return a, pos + 4 * cnt


def _z_bytes(z: np.ndarray) -> bytes:
    return _ext_bytes(z)


# --------------------------------------------------------------------------------------------- timing helper

class Timer:
    def __init__(self, device):
        self.device = device
        self.buckets: dict[str, float] = {}
        self.t0 = None
        self.cur = None

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def start(self, name: str):
        self.sync()
        self.cur, self.t0 = name, time.perf_counter()

    def stop(self):
        self.sync()
        self.buckets[self.cur] = self.buckets.get(self.cur, 0.0) + time.perf_counter() - self.t0
        self.cur = None


# --------------------------------------------------------------------------------------------- prover

def _open_rows(U: torch.Tensor, tree, S: list[int], depth: int) -> RoundOpen:
    idx = torch.as_tensor(np.asarray(S, dtype=np.int64), device=U.device)
    rows = U[:, idx].t().contiguous().to(torch.int64).cpu().numpy()        # (|S|, W)
    return RoundOpen(rows, merkle_multi.gather(tree, S, depth))


def _commit(U: torch.Tensor):
    """Blake3 Merkle over the codeword rows (= the columns of the (W, n) buffer U)."""
    if ligero_merkle._gpu_merkle is None:
        raise RuntimeError("backends/shared/hash_gpu (cupy) is required for the GPU Merkle")
    tree = ligero_merkle._gpu_merkle.commit(U, HASH)
    return ligero_merkle._gpu_merkle.digest_bytes(tree.root, HASH), tree


class _HostTree:
    """Host fallback (laptop tests, small sizes): the same tree with the blake3 package."""

    def __init__(self, U: torch.Tensor):
        words = U.t().contiguous().cpu().numpy().astype(np.uint32)          # (n, W)
        leaves = [merkle_multi.H(words[i].tobytes()) for i in range(words.shape[0])]
        levels = [leaves]
        while len(levels[-1]) > 1:
            cur = levels[-1]
            levels.append([merkle_multi.H(cur[i] + cur[i + 1]) for i in range(0, len(cur), 2)])
        self.levels_b = levels

    @property
    def root(self) -> bytes:
        return self.levels_b[-1][0]

    def gather(self, S, depth) -> bytes:
        return b"".join(self.levels_b[lv][i] for lv, i in merkle_multi.plan(S, depth))


def prove_reference(f: torch.Tensor, z: np.ndarray, dims: Dims, timer: Timer | None = None, coins: Coins | None = None) -> Proof:
    """The prototype's prover (lane ligerito-proto 7bc2fdd): host-side sumcheck tables, fp64-limb contractions.  Kept as the
    byte-identity reference for ``prove`` and as the CPU path.  f: (N,) int32 tensor (coefficients, [0, p)); z: (n, 6) numpy
    point; dims: the round structure; coins: the Coins protocol instance (default Fiat-Shamir)."""
    device = f.device
    n = dims.n
    N = 1 << n
    assert f.numel() == N and z.shape == (n, D)
    ks, kps, qs = dims.k, dims.kprime, dims.queries
    L = dims.rounds
    tm = timer or Timer(device)
    gpu = device.type == "cuda" and ligero_merkle._gpu_merkle is not None
    tr = coins if coins is not None else FiatShamirCoins()
    rates = dims.rates
    tr.absorb(b"header", json.dumps({"n": n, "kprime": kps, "queries": qs, "rate": rates}).encode())
    tr.absorb(b"z", _z_bytes(z))

    roots: list[bytes] = []
    sumcheck: list[list[np.ndarray]] = []
    opens: list[RoundOpen] = []
    splits: list[tuple[int, int]] = []
    Us: list[torch.Tensor] = []
    trees = []
    encs = []

    # ---------------- round 1: commit f
    R1, C1 = 1 << ks[0], 1 << kps[0]
    M1 = f.view(C1, R1)
    enc1 = Encoder(R1, device, rates[0])
    tm.start("commit_r1_alloc")                                           # the codeword buffer (a persistent buffer in a real prover)
    U1 = torch.empty((C1, enc1.n), dtype=torch.int32, device=device)
    tm.stop()
    tm.start("commit_r1_encode")
    enc1.encode(M1, out=U1)
    tm.stop()
    tm.start("commit_r1_merkle")
    if gpu:
        root1, tree1 = _commit(U1)
    else:
        tree1 = _HostTree(U1)
        root1 = tree1.root
    tm.stop()
    roots.append(root1)
    Us.append(U1)
    trees.append(tree1)
    encs.append(enc1)
    splits.append((enc1.n1, enc1.n2))
    tr.absorb(b"root", root1)

    # ---------------- round 1: evaluation claim and the column sumcheck (one eq term)
    tm.start("sumcheck_r1_contract")
    term0 = cl.Term(ext.one(), "eq", z=np.asarray(z, dtype=np.int64) % P, nvars=n)
    zrow = eq_planes(z[:ks[0]], device)                                   # (6, R1)
    u = col_contract(M1, zrow)                                            # (C1, 6)
    tm.stop()
    tm.start("sumcheck_host")
    sc = cl.ColumnSumcheckProver([term0], [u], ks[0], kps[0])
    v = sc.claim()
    tr.absorb(b"v", _ext_bytes(v))
    msgs = []
    for _ in range(kps[0]):
        g = sc.round_message()
        tr.absorb(b"g", _ext_bytes(g))
        r = tr.challenge(b"r", 1)[0]
        sc.fold(r)
        msgs.append(g)
    sumcheck.append(msgs)
    r1 = sc.challenges()                                                  # (k'_1, 6), r[j] <-> variable k_1 + j
    rbar = cl.rbar(r1)                                                    # (C1, 6)
    terms = [term0.restrict(ks[0], r1)]
    running = sc.claim()                                                  # = g_last(r)
    tm.stop()
    tm.start("sumcheck_r1_fold")
    y = row_contract(M1, rbar)                                            # (6, R1) int32 planes
    tm.stop()
    rbars = [rbar]

    # ---------------- rounds 2..L
    for i in range(1, L):
        R, C = 1 << ks[i], 1 << kps[i]
        Mi = y.view(D, C, R)                                              # planes
        tm.start("recursion_encode")
        enc = Encoder(R, device, rates[i])
        U = enc.encode(Mi.reshape(D * C, R))
        tm.stop()
        tm.start("recursion_merkle")
        if gpu:
            root, tree = _commit(U)
        else:
            tree = _HostTree(U)
            root = tree.root
        tm.stop()
        roots.append(root)
        Us.append(U)
        trees.append(tree)
        encs.append(enc)
        splits.append((enc.n1, enc.n2))
        tr.absorb(b"root", root)
        # open the previous round at S_{i-1}
        tm.start("serialize_openings")
        prev = i - 1
        n_prev = encs[prev].n
        S = tr.indices(b"S", n_prev, qs[prev])
        depth = n_prev.bit_length() - 1
        if gpu:
            op = _open_rows(Us[prev], trees[prev], S, depth)
        else:
            idx = torch.as_tensor(np.asarray(S, dtype=np.int64))
            op = RoundOpen(Us[prev][:, idx].t().contiguous().to(torch.int64).numpy(), trees[prev].gather(S, depth))
        opens.append(op)
        tr.absorb(b"rows", np.ascontiguousarray(op.rows.astype(np.uint32)).tobytes())
        tr.absorb(b"sib", op.siblings)
        tm.stop()
        tm.start("sumcheck_host")
        etas = encs[prev].etas(S)
        rhs = _row_claims(op.rows, rbars[prev], base=(prev == 0))        # (|S|, 6)
        new_terms = [cl.Term(ext.one(), "geom", eta=int(e), nvars=ks[prev]) for e in etas]
        alpha = tr.challenge(b"alpha", 1 + len(new_terms))                        # (1 + |S|, 6)
        for t in terms:
            t.coef = ext.mul(t.coef, alpha[0])
        for j, t in enumerate(new_terms):
            t.coef = alpha[1 + j]
        running = ext.add(ext.mul(alpha[0], running), ext.sum_ext(ext.mul(alpha[1:], rhs), axis=0))
        terms = terms + new_terms
        tm.stop()
        # contractions u_t = M~^T rowpart_t
        tm.start("recursion_contract")
        geom = [t for t in terms if t.kind == "geom"]
        eqs = [t for t in terms if t.kind == "eq"]
        assert len(eqs) == 1
        G = powers_matrix(np.array([t.eta for t in geom], dtype=np.int64), R, device) if geom else None
        zr = eq_planes(eqs[0].z[:ks[i]], device)
        u_geom, u_eq = contract_columns_ext(Mi, G, zr)
        tm.stop()
        tm.start("sumcheck_host")
        us = []
        gi = 0
        for t in terms:
            if t.kind == "geom":
                us.append(u_geom[gi])
                gi += 1
            else:
                us.append(u_eq)
        sc = cl.ColumnSumcheckProver(terms, us, ks[i], kps[i])
        assert np.array_equal(sc.claim(), running), "prover-side claim mismatch"
        msgs = []
        for _ in range(kps[i]):
            g = sc.round_message()
            tr.absorb(b"g", _ext_bytes(g))
            r = tr.challenge(b"r", 1)[0]
            sc.fold(r)
            msgs.append(g)
        sumcheck.append(msgs)
        ri = sc.challenges()
        rbar = cl.rbar(ri)
        terms = cl.restrict_terms(terms, ks[i], ri)
        running = sc.claim()
        rbars.append(rbar)
        tm.stop()
        tm.start("recursion_fold")
        y = contract_rows_ext(Mi, rbar)                                   # (6, R)
        tm.stop()

    # ---------------- final: y_L in full, open U_L
    tm.start("serialize_final")
    final_y = y.to(torch.int64).t().contiguous().cpu().numpy() % P        # (R_L, 6)
    tr.absorb(b"y", _ext_bytes(final_y))
    n_last = encs[-1].n
    S = tr.indices(b"S", n_last, qs[-1])
    depth = n_last.bit_length() - 1
    if gpu:
        op = _open_rows(Us[-1], trees[-1], S, depth)
    else:
        idx = torch.as_tensor(np.asarray(S, dtype=np.int64))
        op = RoundOpen(Us[-1][:, idx].t().contiguous().to(torch.int64).numpy(), trees[-1].gather(S, depth))
    opens.append(op)
    tm.stop()
    return Proof(n, list(kps), list(qs), list(rates), splits, v, roots, sumcheck, opens, final_y, tr.rounds)


class _NullTimer:
    def start(self, name: str):
        pass

    def stop(self):
        pass


def _commit_fast(U: torch.Tensor):
    """Blake3 Merkle over the codeword rows: the Ligero prover's fused 3-launch kernels (``commit_gpu``, same tree bytes as
    ``hash_gpu``) when the leaf fits (<= 4096 words), else ``hash_gpu`` level by level."""
    if _commit_gpu is not None and U.shape[0] <= 4096 and U.shape[1] & (U.shape[1] - 1) == 0:
        tree = _commit_gpu.commit(U)
        return ligero_merkle._gpu_merkle.digest_bytes(tree.root, HASH), tree
    return _commit(U)


def _stream_group(dims: Dims, device) -> int:
    """Round-1 streaming commit: 0 = materialise the codeword, else the column-group size.  ``LIGERITO_STREAM`` = auto (default:
    stream iff the round-1 codeword + tree + round 2 would not fit the card), 1 (always), 0 (never)."""
    mode = os.environ.get("LIGERITO_STREAM", "auto")
    C1, R1 = 1 << dims.kprime[0], 1 << dims.k[0]
    if mode == "0" or C1 % 16 or C1 > 256:
        return 0
    n1 = R1 << dims.rates[0]
    total = torch.cuda.get_device_properties(device).total_memory
    gib = 1 << 30
    msg, tree = 4 * C1 * R1, 64 * n1
    rest = 24 * R1                                                      # y_1 (6 planes)
    if dims.rounds > 1:
        R2, C2 = 1 << dims.k[1], 1 << dims.kprime[1]
        n2 = R2 << dims.rates[1]
        rest += 24 * C2 * n2 + 64 * n2
    budget = total - int(2.5 * gib)
    if mode != "1" and msg + 4 * C1 * n1 + tree + rest <= budget:
        return 0
    for g in (128, 64, 32, 16):
        if g <= C1 and C1 % g == 0 and (g < C1 or C1 == 16) and msg + 4 * g * n1 + tree <= budget:
            return g
    return 16


def _siblings(tree, S: list[int], depth: int) -> bytes:
    import cupy as cp

    pl = merkle_multi.plan_levels(S, depth)
    if pl:
        flat = cp.asarray(np.concatenate([ix for _, ix in pl]))
        parts, a = [], 0
        for lv, ix in pl:
            parts.append(tree.levels[lv][flat[a:a + len(ix)]])
            a += len(ix)
        return cp.asnumpy(cp.concatenate(parts, axis=0)).astype("<u4", copy=False).tobytes()
    return b""


def _open_fast(U: torch.Tensor, tree, S: list[int], depth: int) -> RoundOpen:
    """Opened rows + multi-path siblings with two device gathers and two copies (no per-level syncs)."""
    idx = torch.as_tensor(np.asarray(S, dtype=np.int64), device=U.device)
    rows = U.index_select(1, idx).t().contiguous().cpu().numpy().astype(np.int64)
    return RoundOpen(rows, _siblings(tree, S, depth))


def _open_recompute(M: torch.Tensor, enc: Encoder, tree, S: list[int], depth: int) -> RoundOpen:
    """Streamed round 1: the opened codeword rows U[:, s] = P_c(eta_s) recomputed from the message columns (|S| N base MACs)."""
    rows = (dev.contract_geom(M, enc.etas(S)) % P).t().contiguous().cpu().numpy()
    return RoundOpen(rows, _siblings(tree, S, depth))


COMMIT_FIRST = "commit-first"


@dataclass
class Commitment:
    """``commit``'s output: ``root`` is the commitment (absorbed into ``coins``); the rest is prover state for ``prove_eval``
    (the round-1 message view, its codeword -- None when streamed -- and Merkle tree).  Drop it after ``prove_eval``."""
    dims: Dims
    root: bytes
    coins: Coins
    M1: torch.Tensor
    enc1: Encoder
    U1: torch.Tensor | None
    tree1: object


def _header(dims: Dims, order: str | None) -> bytes:
    h = {"n": dims.n, "kprime": list(dims.kprime), "queries": list(dims.queries), "rate": list(dims.rates)}
    if order is not None:
        h["order"] = order
    return json.dumps(h).encode()


def absorb_commitment(coins: Coins, dims: Dims, root: bytes) -> None:
    """The commit-first transcript prefix (prover and verifier): header (with ``"order": "commit-first"``), root_1."""
    coins.absorb(b"header", _header(dims, COMMIT_FIRST))
    coins.absorb(b"root", root)


def _device_ok(f: torch.Tensor) -> bool:
    return f.device.type == "cuda" and dev.available(f) and ligero_merkle._gpu_merkle is not None


def _commit_r1(f: torch.Tensor, dims: Dims, tm):
    device = f.device
    assert f.numel() == 1 << dims.n and f.dtype == torch.int32
    dev.share_torch_allocator()
    R1, C1 = 1 << dims.k[0], 1 << dims.kprime[0]
    M1 = f.view(C1, R1)
    enc1 = Encoder(R1, device, dims.rates[0])
    group = _stream_group(dims, device)
    if group:
        tm.start("commit_r1_stream")
        root1, tree1 = stream.commit_streamed(M1, enc1, group, _commit_gpu)
        U1 = None
        tm.stop()
    else:
        tm.start("commit_r1_alloc")
        U1 = torch.empty((C1, enc1.n), dtype=torch.int32, device=device)
        tm.stop()
        tm.start("commit_r1_encode")
        enc1.encode(M1, out=U1)
        tm.stop()
        tm.start("commit_r1_merkle")
        root1, tree1 = _commit_fast(U1)
        tm.stop()
    return M1, enc1, U1, tree1, root1


def commit(f: torch.Tensor, dims: Dims, coins: Coins | None = None, timer: Timer | None = None) -> Commitment:
    """Commit-first API (for a relation prover whose evaluation points depend on the commitment): encode + Merkle-commit f
    ((N,) int32 CUDA tensor, [0, p)) and absorb the header + root into ``coins``.  Then run the relation's rounds on the same
    coins, then ``prove_eval``."""
    if not _device_ok(f):
        raise RuntimeError("pcs.commit needs the CUDA device path (cupy + hash_gpu)")
    tm = timer or _NullTimer()
    tr = coins if coins is not None else FiatShamirCoins()
    M1, enc1, U1, tree1, root1 = _commit_r1(f, dims, tm)
    absorb_commitment(tr, dims, root1)
    return Commitment(dims, root1, tr, M1, enc1, U1, tree1)


def prove_eval(com: Commitment, z: np.ndarray, coins: Coins | None = None, timer: Timer | None = None) -> Proof:
    """Evaluation proof for the committed f at m points: ``z`` (n, 6) or (m, n, 6).  Transcript (continuing ``commit``'s):
    absorb z (m n ext), absorb v (m ext), [m > 1: beta <- challenge(m); the round-1 claim is sum_j beta_j v_j], then the
    rounds exactly as ``prove``.  ``proof.v`` is (m, 6)."""
    zs = np.asarray(z, dtype=np.int64) % P
    if zs.ndim == 2:
        zs = zs[None]
    assert zs.shape[1:] == (com.dims.n, D)
    tr = coins if coins is not None else com.coins
    tm = timer or _NullTimer()
    tr.absorb(b"z", _z_bytes(zs))
    return _prove_rounds(com.M1, com.enc1, com.U1, com.tree1, com.root, com.dims, zs, tr, tm, COMMIT_FIRST)


def prove(f: torch.Tensor, z: np.ndarray, dims: Dims, timer: Timer | None = None, coins: Coins | None = None) -> Proof:
    """f: (N,) int32 device tensor (coefficients, [0, p)); z: (n, 6) numpy point; dims: the round structure;
    coins: the Coins protocol instance (default Fiat-Shamir).  Fused order (proto 7bc2fdd): header, z, root_1, v, ...

    Device prover (lane ligerito-pcs-fast): the sumcheck tables, every contraction and fold live on the device (integer
    kernels in ``device.py``, no fp64); the host keeps only the transcript and the O(T) batching coefficients.  Same
    transcript and proof bytes as ``prove_reference`` for the same coins.  Falls back to ``prove_reference`` off-GPU."""
    if not _device_ok(f):
        return prove_reference(f, z, dims, timer, coins)
    assert z.shape == (dims.n, D)
    tm = timer or _NullTimer()
    tr = coins if coins is not None else FiatShamirCoins()
    z = np.asarray(z, dtype=np.int64) % P
    tr.absorb(b"header", _header(dims, None))
    tr.absorb(b"z", _z_bytes(z))
    M1, enc1, U1, tree1, root1 = _commit_r1(f, dims, tm)
    tr.absorb(b"root", root1)
    return _prove_rounds(M1, enc1, U1, tree1, root1, dims, z[None], tr, tm, None)


def _prove_rounds(M1, enc1, U1, tree1, root1, dims: Dims, zs: np.ndarray, tr: Coins, tm, order: str | None) -> Proof:
    """Everything after root_1: the claims, round 1's column sumcheck, rounds 2..L, the final vector."""
    device = M1.device
    n = dims.n
    ks, kps, qs, rates = dims.k, dims.kprime, dims.queries, dims.rates
    L = dims.rounds
    m = zs.shape[0]
    roots, Us, trees, encs = [root1], [U1], [tree1], [enc1]
    splits = [(enc1.n1, enc1.n2)]
    sumcheck: list[list[np.ndarray]] = []
    opens: list[RoundOpen] = []

    # ---------------- round 1: evaluation claims and the column sumcheck (m eq terms, host tables)
    tm.start("sumcheck_r1_contract")
    us = [(dev.contract_planes(M1, dev.eq_table(zj[:ks[0]], device)) % P).cpu().numpy() for zj in zs]   # m x (C1, 6)
    tm.stop()
    tm.start("sumcheck_host")
    vs = np.stack([cl.ColumnSumcheckProver([cl.Term(ext.one(), "eq", z=zj, nvars=n)], [uj], ks[0], kps[0]).claim()
                   for zj, uj in zip(zs, us)], axis=0)                     # (m, 6)
    tr.absorb(b"v", _ext_bytes(vs))
    beta = tr.challenge(b"beta", m) if m > 1 else ext.one()[None]
    terms0 = [cl.Term(beta[j], "eq", z=zs[j], nvars=n) for j in range(m)]
    sc = cl.ColumnSumcheckProver(terms0, us, ks[0], kps[0])
    msgs = []
    for _ in range(kps[0]):
        g = sc.round_message()
        tr.absorb(b"g", _ext_bytes(g))
        r = tr.challenge(b"r", 1)[0]
        sc.fold(r)
        msgs.append(g)
    sumcheck.append(msgs)
    r1 = sc.challenges()
    rbar = cl.rbar(r1)
    running = sc.claim()
    rest = [t.restrict(ks[0], r1) for t in terms0]
    coef_eq = np.stack([t.coef for t in rest], axis=0)                   # (m, 6)
    z_eq = np.stack([t.z for t in rest], axis=0)                         # (m, k_1, 6)
    etas_g = np.zeros(0, dtype=np.int64)                                  # geom terms: eta (Tg,), coef (Tg, 6)
    coef_g = ext.zero((0,))
    tm.stop()
    tm.start("sumcheck_r1_fold")
    y = row_contract(M1, rbar)                                            # (6, R1) int32 planes
    tm.stop()
    rbars = [rbar]

    # ---------------- rounds 2..L
    for i in range(1, L):
        R, C = 1 << ks[i], 1 << kps[i]
        Mi = y.view(D, C, R)
        Mflat = Mi.reshape(D * C, R)
        tm.start("recursion_encode")
        enc = Encoder(R, device, rates[i])
        U = enc.encode(Mflat)
        tm.stop()
        tm.start("recursion_merkle")
        root, tree = _commit_fast(U)
        tm.stop()
        roots.append(root)
        Us.append(U)
        trees.append(tree)
        encs.append(enc)
        splits.append((enc.n1, enc.n2))
        tr.absorb(b"root", root)
        tm.start("serialize_openings")
        prev = i - 1
        n_prev = encs[prev].n
        S = tr.indices(b"S", n_prev, qs[prev])
        depth = n_prev.bit_length() - 1
        if Us[prev] is None:
            tm.stop()
            tm.start("commit_r1_recompute")
            op = _open_recompute(M1, encs[prev], trees[prev], S, depth)
            tm.stop()
            tm.start("serialize_openings")
        else:
            op = _open_fast(Us[prev], trees[prev], S, depth)
        opens.append(op)
        tr.absorb(b"rows", np.ascontiguousarray(op.rows.astype(np.uint32)).tobytes())
        tr.absorb(b"sib", op.siblings)
        Us[prev] = trees[prev] = None                                     # round i-1's codeword and tree are not needed any more
        tm.stop()
        tm.start("sumcheck_host")
        etas = encs[prev].etas(S)
        rhs = _row_claims(op.rows, rbars[prev], base=(prev == 0))        # (|S|, 6)
        alpha = tr.challenge(b"alpha", 1 + len(S))                        # (1 + |S|, 6)
        coef_eq = ext.mul(coef_eq, alpha[0])
        coef_g = ext.mul(coef_g, alpha[0]) if len(etas_g) else coef_g
        coef_g = np.concatenate((coef_g, alpha[1:]), axis=0)
        etas_g = np.concatenate((etas_g, np.asarray(etas, dtype=np.int64)))
        running = ext.add(ext.mul(alpha[0], running), ext.sum_ext(ext.mul(alpha[1:], rhs), axis=0))
        colg = cl.geom_colparts(etas_g, ks[i], kps[i])[:, :, 0]           # (Tg, C) base
        eqt = [cl.Term(ext.one(), "eq", z=zj, nvars=zj.shape[0]) for zj in z_eq]
        cole = np.stack([t.colpart(ks[i], kps[i]) for t in eqt], axis=0)  # (m, C, 6)
        coef = np.concatenate((coef_eq, coef_g), axis=0)                  # (m + Tg, 6): eq terms first, then geom (the proto's order)
        tm.stop()
        tm.start("recursion_contract")
        ug = dev.contract_geom(Mflat, etas_g)                            # (6C, Tg)
        ue = torch.stack([dev.contract_planes(Mflat, dev.eq_table(zj[:ks[i]], device)) for zj in z_eq], dim=0)   # (m, 6C, 6)
        dsc = dev.DeviceSumcheck(ug, ue, coef, colg, cole, C, device)
        del ug, ue
        tm.stop()
        tm.start("sumcheck_device")
        msgs, rs = [], []
        for j in range(kps[i]):
            g = dsc.message()
            if j == 0:
                assert np.array_equal(ext.add(g[0], g[1]), running), "prover-side claim mismatch"
            tr.absorb(b"g", _ext_bytes(g))
            r = tr.challenge(b"r", 1)[0]
            dsc.fold(r)
            msgs.append(g)
            rs.append(r)
        sumcheck.append(msgs)
        running = cl.interpolate_deg2(g[0], g[1], g[2], rs[-1])
        ri = np.stack(rs[::-1], axis=0)
        rbar = cl.rbar(ri)
        tm.stop()
        tm.start("sumcheck_host")
        coef_eq = np.stack([ext.mul(coef_eq[j], eqt[j].colpart_at(ks[i], ri)) for j in range(m)], axis=0)
        z_eq = z_eq[:, :ks[i]]
        coef_g = ext.mul(coef_g, cl.geom_colpart_at(etas_g, ks[i], ri))
        rbars.append(rbar)
        del dsc
        tm.stop()
        tm.start("recursion_fold")
        y = dev.fold_ext(Mi, rbar)                                        # (6, R)
        tm.stop()

    # ---------------- final: y_L in full, open U_L
    tm.start("serialize_final")
    final_y = y.to(torch.int64).t().contiguous().cpu().numpy() % P        # (R_L, 6)
    tr.absorb(b"y", _ext_bytes(final_y))
    n_last = encs[-1].n
    S = tr.indices(b"S", n_last, qs[-1])
    depth = n_last.bit_length() - 1
    op = _open_fast(Us[-1], trees[-1], S, depth) if Us[-1] is not None else _open_recompute(M1, encs[-1], trees[-1], S, depth)
    opens.append(op)
    tm.stop()
    v = vs if order == COMMIT_FIRST else vs[0]
    return Proof(n, list(kps), list(qs), list(rates), splits, v, roots, sumcheck, opens, final_y, tr.rounds, order)


def _row_claims(rows: np.ndarray, rbar: np.ndarray, base: bool) -> np.ndarray:
    """(|S|, W) opened rows -> (|S|, 6): < U[s, :], rbar >.  base: W = C base words; else W = 6 C (plane-major)."""
    if base:
        return matmul_mod_p_np(rows, rbar)                                 # (|S|, C) @ (C, 6)
    S, W = rows.shape
    C = W // D
    planes = rows.reshape(S, D, C)                                        # [s, a, c]
    out = np.zeros((S, D), dtype=np.int64)
    for a in range(D):
        part = matmul_mod_p_np(planes[:, a, :], rbar)                     # (|S|, 6): sum_c U_a[s, c] rbar[c, b]
        for b in range(D):
            k = a + b
            if k >= D:
                out[:, k - D] = (out[:, k - D] + part[:, b] * ext.BETA) % P
            else:
                out[:, k] = (out[:, k] + part[:, b]) % P
    return out


# --------------------------------------------------------------------------------------------- verifier

class Reject(Exception):
    pass


def verify(proof: Proof, z: np.ndarray, v: np.ndarray | None = None, coins: Coins | None = None) -> tuple[bool, str]:
    """Returns (accepted, reason).  ``v``: the claimed evaluation (defaults to the proof's); ``coins``: must replay the
    prover's (default Fiat-Shamir)."""
    try:
        tr = coins if coins is not None else FiatShamirCoins()
        if proof.order == COMMIT_FIRST:
            if len(proof.roots) == 0:
                raise Reject("bad proof shape")
            absorb_commitment(tr, proof.dims, proof.roots[0])
        _verify(proof, np.asarray(z, dtype=np.int64) % P, proof.v if v is None else np.asarray(v, dtype=np.int64) % P, tr)
    except Reject as e:
        return False, str(e)
    return True, "accept"


def verify_eval(proof: Proof, z: np.ndarray, v: np.ndarray | None = None, coins: Coins | None = None) -> tuple[bool, str]:
    """Commit-first proofs inside a larger transcript: ``coins`` already holds ``absorb_commitment(coins, dims, root)`` and
    whatever the relation absorbed/drew after it; ``z`` (m, n, 6) or (n, 6); ``v`` (m, 6) (default: the proof's)."""
    try:
        if proof.order != COMMIT_FIRST:
            raise Reject("not a commit-first proof")
        _verify(proof, np.asarray(z, dtype=np.int64) % P, proof.v if v is None else np.asarray(v, dtype=np.int64) % P, coins)
    except Reject as e:
        return False, str(e)
    return True, "accept"


def _verify(pf: Proof, z: np.ndarray, v: np.ndarray, tr: Coins) -> None:
    dims = pf.dims
    n, ks, kps, qs, L = dims.n, dims.k, dims.kprime, dims.queries, dims.rounds
    cf = pf.order == COMMIT_FIRST
    if pf.order not in (None, COMMIT_FIRST):
        raise Reject("unknown transcript order")
    zs = z[None] if z.ndim == 2 else z
    vs = v.reshape(-1, D) if v.ndim >= 1 else v
    m = zs.shape[0]
    if zs.shape[1:] != (n, D) or (not cf and m != 1) or vs.shape != (m, D) or m < 1:
        raise Reject("bad point shape")
    if len(pf.roots) != L or len(pf.opens) != L or len(pf.sumcheck) != L or len(pf.splits) != L or len(dims.rates) != L:
        raise Reject("bad proof shape")
    from .ntt_gpu import Encoder as _Enc

    class _Map:
        def __init__(self, R, split, rate):
            self.n = R << rate
            self.n1, self.n2 = split
            if self.n1 * self.n2 != self.n:
                raise Reject("bad encoder split")

        def etas(self, s):
            return _Enc.etas(self, s)

        def eta_index(self, s):
            return _Enc.eta_index(self, s)

    maps = [_Map(1 << ks[i], pf.splits[i], dims.rates[i]) for i in range(L)]
    if tr is None:
        tr = FiatShamirCoins()
    if cf:                                          # header + root_1 were absorbed by absorb_commitment
        tr.absorb(b"z", _z_bytes(zs))
    else:
        tr.absorb(b"header", _header(dims, None))
        tr.absorb(b"z", _z_bytes(zs[0]))
        tr.absorb(b"root", pf.roots[0])
    # round 1
    if np.asarray(pf.v).reshape(-1, D).shape != (m, D) or not np.array_equal(np.asarray(pf.v).reshape(-1, D) % P, vs):
        raise Reject("claimed evaluation differs from the proof's")
    tr.absorb(b"v", _ext_bytes(vs))
    beta = tr.challenge(b"beta", m) if m > 1 else ext.one()[None]
    terms0 = [cl.Term(beta[j], "eq", z=zs[j], nvars=n) for j in range(m)]
    claim = ext.sum_ext(ext.mul(beta, vs), axis=0)
    rs = []
    for g in pf.sumcheck[0]:
        if g.shape != (3, D):
            raise Reject("bad sumcheck message")
        tr.absorb(b"g", _ext_bytes(g))
        rs.append(tr.challenge(b"r", 1)[0])
    ok, running = cl.verify_column_rounds(claim, pf.sumcheck[0], rs)
    if not ok:
        raise Reject("sumcheck round 1 failed")
    r1 = np.stack(rs[::-1], axis=0) if rs else np.zeros((0, D), dtype=np.int64)
    rbars = [cl.rbar(r1)]
    terms = [t.restrict(ks[0], r1) for t in terms0]
    for i in range(1, L):
        tr.absorb(b"root", pf.roots[i])
        prev = i - 1
        S = tr.indices(b"S", maps[prev].n, qs[prev])
        op = pf.opens[prev]
        W = (1 << kps[prev]) * (1 if prev == 0 else D)
        if op.rows.shape != (qs[prev], W):
            raise Reject(f"opened rows of round {prev + 1} have shape {op.rows.shape}, expected {(qs[prev], W)}")
        if (op.rows < 0).any() or (op.rows >= P).any():
            raise Reject("opened row word out of range")
        tr.absorb(b"rows", np.ascontiguousarray(op.rows.astype(np.uint32)).tobytes())
        tr.absorb(b"sib", op.siblings)
        depth = maps[prev].n.bit_length() - 1
        leaves = {s: merkle_multi.leaf_digest(op.rows[j]) for j, s in enumerate(S)}
        if not merkle_multi.verify(pf.roots[prev], leaves, op.siblings, depth):
            raise Reject(f"Merkle multi-path of round {prev + 1} failed")
        etas = maps[prev].etas(S)
        rhs = _row_claims(op.rows, rbars[prev], base=(prev == 0))
        new_terms = [cl.Term(ext.one(), "geom", eta=int(e), nvars=ks[prev]) for e in etas]
        alpha = tr.challenge(b"alpha", 1 + len(new_terms))
        for t in terms:
            t.coef = ext.mul(t.coef, alpha[0])
        for j, t in enumerate(new_terms):
            t.coef = alpha[1 + j]
        running = ext.add(ext.mul(alpha[0], running), ext.sum_ext(ext.mul(alpha[1:], rhs), axis=0))
        terms = terms + new_terms
        rs = []
        for g in pf.sumcheck[i]:
            if g.shape != (3, D):
                raise Reject("bad sumcheck message")
            tr.absorb(b"g", _ext_bytes(g))
            rs.append(tr.challenge(b"r", 1)[0])
        ok, running = cl.verify_column_rounds(running, pf.sumcheck[i], rs)
        if not ok:
            raise Reject(f"sumcheck round {i + 1} failed")
        ri = np.stack(rs[::-1], axis=0) if rs else np.zeros((0, D), dtype=np.int64)
        rbars.append(cl.rbar(ri))
        terms = cl.restrict_terms(terms, ks[i], ri)
    # final vector
    y = pf.final_y % P
    RL = 1 << ks[-1]
    if y.shape != (RL, D):
        raise Reject("bad final vector shape")
    tr.absorb(b"y", _ext_bytes(y))
    S = tr.indices(b"S", maps[-1].n, qs[-1])
    op = pf.opens[-1]
    W = (1 << kps[-1]) * (1 if L == 1 else D)
    if op.rows.shape != (qs[-1], W):
        raise Reject("opened rows of the last round have a bad shape")
    if (op.rows < 0).any() or (op.rows >= P).any():
        raise Reject("opened row word out of range")
    depth = maps[-1].n.bit_length() - 1
    leaves = {s: merkle_multi.leaf_digest(op.rows[j]) for j, s in enumerate(S)}
    if not merkle_multi.verify(pf.roots[-1], leaves, op.siblings, depth):
        raise Reject("Merkle multi-path of the last round failed")
    # (a) P_y(eta_s) = < U_L[s, :], rbar_L >
    etas = maps[-1].etas(S)
    pw = _powers_rows_np(np.asarray(etas, dtype=np.int64), RL)         # (|S|, R_L)
    lhs = matmul_mod_p_np(pw, y)                                          # (|S|, 6)
    rhs = _row_claims(op.rows, rbars[-1], base=(L == 1))
    if not np.array_equal(lhs, rhs):
        raise Reject("final vector is inconsistent with the opened rows")
    # (b) sum_t coef_t < rowpart_t , y > = running
    geom = [t for t in terms if t.kind == "geom"]
    eqs = [t for t in terms if t.kind == "eq"]
    total = ext.zero()
    if geom:
        pw = _powers_rows_np(np.array([t.eta for t in geom], dtype=np.int64), RL)   # (Tg, R_L)
        ip = matmul_mod_p_np(pw, y)                                       # (Tg, 6)
        coefs = np.stack([t.coef for t in geom], axis=0)
        total = ext.add(total, ext.sum_ext(ext.mul(coefs, ip), axis=0))
    for t in eqs:
        vec = t.rowpart_eq(ks[-1])                                        # (R_L, 6)
        ip = ext.sum_ext(ext.mul(vec, y), axis=0)
        total = ext.add(total, ext.mul(t.coef, ip))
    if not np.array_equal(total, running):
        raise Reject("final sumcheck claim failed")
