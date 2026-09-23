"""The constraint layer of the Ligerito backend (lane ligerito-sumcheck): a batched degree-3 zero-check over every
constraint of a ``compile.System`` on the multilinear layout of ``layout.py``, ending in evaluation claims on the committed
polynomial.  Torch prover (reference path here; fused CUDA rounds in ``_kernels`` when cupy is present) + Python verifier.

Three sumchecks, all randomness through ``Coins.challenge(label, n)`` (``transcript.py``, lane ligerito-proto):

1. **zero-check** over ``(k, c)``, ``n_k + n_c`` variables (``k`` = constraint slot = the HIGH bits, ``c = s * l + j``):
       sum_{k,c} eq(tau, (k, c)) * (Az * Bz - Cz)(k, c) = 0,     Az = A z  etc. (tables 2^n_k x C)
   Round ``t`` binds the top remaining variable; the round polynomial is ``eq(tau_t, X) * q_t(X)`` with ``q_t`` quadratic
   (Gruen): the prover sends ``q_t(0), q_t(1), q_t(inf)`` (the X^2 coefficient), the verifier checks
   ``E_t ((1 - tau_t) q(0) + tau_t q(1)) = claim`` with ``E_t = prod_{u<t} eq(tau_u, r_u)`` and continues with
   ``E_{t+1} q(r_t)``.  Ends in the three values ``Az(r), Bz(r), Cz(r)`` with ``E * (Az Bz - Cz)(r) = claim``.
2. **rows** over ``i`` (``n_i`` variables, degree 2): ``Az(r) + g Bz(r) + g^2 Cz(r) = sum_i coef(i) z~(i, r_c)`` with
   ``coef = eq(r_k, .)^T (A + g B + g^2 C)`` (verifier-computable).  Ends in ``coef~(r_i) * z~(r_i, r_c)``, and
   ``z~(r_i, r_c) = w~(r_i, r_c) + sum_v eq(r_i, idx_v) virt_v(r_c)``: the first term is the PCS claim, the public virtual
   rows are the verifier's MLE evaluations, the shifted rows are the prover's claimed ``next_vals`` proved by
3. **shift** over ``c`` (``n_c`` variables, degree 2): ``sum_x g3^x next_x(r_c) = sum_c succ~(r_c, c) V(c)``,
   ``V(c) = sum_x g3^x W(c_x, c)``, ``succ~(r, c) = eq(r, pred(c))`` (the eq table rolled by one column, cyclic per
   sub-batch).  Ends in ``succ~(r_c, rho) * sum_x g3^x w~(bits(c_x), rho)``: ``n_links`` more PCS claims; the verifier
   evaluates ``succ~`` in closed form (``succ_mle``; the O(C) "sparse-matrix" evaluation is ``succ_mle_dense``).

Message bytes: ``(n_k + n_c + n_i + n_c) x 3 + 3 + n_links + 1 + n_links`` extension elements of 24 B.
"""
from __future__ import annotations

import hashlib
import struct
import time
from dataclasses import dataclass, field

import torch

from backends.direct.ligero.compile import P
from backends.direct.ligero.v2.fc import D, fc_inv, fc_mul, fc_one, fc_pow_seq, fc_scalar_mul
from .layout import Constraints, Layout
from .layout import SubBatch as SubBatchT

try:  # lane ligerito-proto's Coins protocol; the stub below has the same signature until it lands
    from .transcript import Coins  # type: ignore  # noqa: F401
except Exception:  # noqa: BLE001
    Coins = object  # type: ignore

__all__ = ["EvalClaim", "LocalCoins", "SumcheckProof", "eq_at", "eq_table", "prove", "succ_mle", "verify"]

MAGIC = b"LGSC0001"


# ---------------------------------------------------------------------------------------------------------------------
# Coins (stub: deterministic from a seed; the protocol is ``challenge(label, n) -> (n, 6)``)
# ---------------------------------------------------------------------------------------------------------------------

class LocalCoins:
    """Diagnostic coins: ``challenge(label, n)`` = SHAKE-256(seed || label || counter) reduced mod p per coordinate.
    Prover and verifier construct it from the same seed.  (ligerito-proto's ``transcript.LocalCoins`` replaces this.)"""

    def __init__(self, seed: bytes = b"ligerito-sumcheck", device="cpu"):
        self.seed, self.device, self.n = seed, device, 0

    def challenge(self, label: str, n: int) -> torch.Tensor:
        h = hashlib.shake_256(self.seed + b"|" + label.encode() + b"|" + struct.pack("<I", self.n))
        self.n += 1
        raw = h.digest(8 * n * D)
        vals = [int.from_bytes(raw[8 * i:8 * i + 8], "little") % P for i in range(n * D)]
        return torch.tensor(vals, dtype=torch.int64, device=self.device).reshape(n, D)


# ---------------------------------------------------------------------------------------------------------------------
# Extension-field helpers (torch int64, last axis = 6 coordinates; ``fc.py`` arithmetic)
# ---------------------------------------------------------------------------------------------------------------------

def ext_of_int(v: int, device) -> torch.Tensor:
    t = torch.zeros((D,), dtype=torch.int64, device=device)
    t[0] = v % P
    return t


def ext_of_base(a: torch.Tensor) -> torch.Tensor:
    """(...) base -> (..., 6)"""
    out = torch.zeros(a.shape + (D,), dtype=torch.int64, device=a.device)
    out[..., 0] = a.to(torch.int64) % P
    return out


def eq_table(tau: torch.Tensor) -> torch.Tensor:
    """``(2^n, 6)``: ``eq(tau, x)`` for every ``x``; variable ``t`` = bit ``t`` of ``x`` (bit ``t`` is the new HIGH bit at
    step ``t``)."""
    vec = fc_one((1,), tau.device)
    for t in range(tau.shape[0]):
        one_minus = (fc_one((), tau.device) - tau[t]) % P
        vec = torch.cat((fc_mul(vec, one_minus), fc_mul(vec, tau[t])), 0)
    return vec


def eq_at(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``eq(a, b)`` for two points ``(n, 6)`` -> ``(6,)``."""
    one = fc_one((), a.device)
    acc = one.clone()
    for t in range(a.shape[0]):
        acc = fc_mul(acc, (fc_mul(a[t], b[t]) + fc_mul((one - a[t]) % P, (one - b[t]) % P)) % P)
    return acc


def matvec_base_ext(M: torch.Tensor, e: torch.Tensor, chunk_rows: int = 256) -> torch.Tensor:
    """``(R, C) base x (C, 6) ext -> (R, 6)``, exact: fp64 GEMMs on 16-bit limbs (the coordinate sums are < 2^50 for
    ``C <= 2^18``; larger ``C`` is split)."""
    R, C = M.shape
    if C > (1 << 18):
        half = C // 2
        return (matvec_base_ext(M[:, :half], e[:half], chunk_rows) + matvec_base_ext(M[:, half:], e[half:], chunk_rows)) % P
    e = e % P
    e_lo = (e & 0xFFFF).to(torch.float64)                      # (C, 6)
    e_hi = (e >> 16).to(torch.float64)
    E = torch.cat((e_lo, e_hi), 1)                              # (C, 12)
    out = torch.empty((R, D), dtype=torch.int64, device=M.device)
    two16, two32 = pow(2, 16, P), pow(2, 32, P)
    for r0 in range(0, R, chunk_rows):
        m = M[r0:r0 + chunk_rows].to(torch.int64) % P
        m_lo = (m & 0xFFFF).to(torch.float64)
        m_hi = (m >> 16).to(torch.float64)
        pl = (m_lo @ E).to(torch.int64)                          # (r, 12): lo*lo | lo*hi
        ph = (m_hi @ E).to(torch.int64)                          # hi*lo | hi*hi
        ll, lh = pl[:, :D] % P, pl[:, D:] % P
        hl, hh = ph[:, :D] % P, ph[:, D:] % P
        out[r0:r0 + chunk_rows] = (ll + ((lh + hl) % P) * two16 + hh * two32) % P
    return out


def mle_eval_base(vec: torch.Tensor, eq_tab: torch.Tensor) -> torch.Tensor:
    """``sum_x vec[x] eq_tab[x]`` for a base vector ``(C,)`` and an eq table ``(C, 6)`` -> ``(6,)``."""
    return matvec_base_ext(vec[None, :], eq_tab)[0]


def mle_eval_ext(vec: torch.Tensor, eq_tab: torch.Tensor) -> torch.Tensor:
    """``(C, 6) x (C, 6) -> (6,)``"""
    return fc_mul(vec, eq_tab).sum(0) % P


def fold_ext(T: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Bind the top bit of axis 0 of an ext table ``(2h, ..., 6)`` to ``r``: ``T0 + r (T1 - T0)``."""
    h = T.shape[0] // 2
    T0, T1 = T[:h], T[h:]
    return (T0 + fc_mul((T1 - T0) % P, r)) % P


def succ_mle(a: torch.Tensor, b: torch.Tensor, n_j: int) -> torch.Tensor:
    """The MLE of the cyclic successor relation ``[a = b + 1 mod 2^n_j on the low n_j bits, a = b on the rest]`` at the
    points ``a, b`` ``(n, 6)``: ``sum_t (prod_{u<t} b_u (1 - a_u)) (1 - b_t) a_t prod_{u>t} eq(a_u, b_u) + prod_u b_u (1 - a_u)``
    on the ``j`` bits, times ``eq`` on the ``s`` bits.  Used as ``succ_mle(rho, r_c)`` = ``sum_c eq(rho, c) eq(r_c, pred(c))``."""
    dev = a.device
    one = fc_one((), dev)
    eqs = [(fc_mul(a[u], b[u]) + fc_mul((one - a[u]) % P, (one - b[u]) % P)) % P for u in range(n_j)]
    carry = [fc_mul(b[u], (one - a[u]) % P) for u in range(n_j)]        # b_u = 1, a_u = 0
    total = torch.zeros((D,), dtype=torch.int64, device=dev)
    pref = one.clone()
    for t in range(n_j):
        term = fc_mul(pref, fc_mul((one - b[t]) % P, a[t]))
        for u in range(t + 1, n_j):
            term = fc_mul(term, eqs[u])
        total = (total + term) % P
        pref = fc_mul(pref, carry[t])
    total = (total + pref) % P                                          # the wrap-around b = 1..1 -> a = 0..0
    for u in range(n_j, a.shape[0]):
        total = fc_mul(total, (fc_mul(a[u], b[u]) + fc_mul((one - a[u]) % P, (one - b[u]) % P)) % P)
    return total


def succ_mle_dense(a: torch.Tensor, b: torch.Tensor, n_j: int) -> torch.Tensor:
    """The same value by the O(C) "sparse-matrix" route: ``sum_c eq(a, c) eq(b, pred(c))`` with both eq tables materialised."""
    ea, eb = eq_table(a), eq_table(b)
    l = 1 << n_j
    eb_pred = torch.roll(eb.reshape(-1, l, D), 1, dims=1).reshape(-1, D)     # eb_pred[c] = eb[pred(c)]
    return mle_eval_ext(ea, eb_pred)


# ---------------------------------------------------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class EvalClaim:
    """``w~(point) = value`` on the committed polynomial; ``point[t]`` is the coordinate of variable ``t`` (bit ``t`` of the
    flat index ``x = i * C + c``): variables ``0 .. n_c - 1`` are the column ``c = s * l + j``, ``n_c .. n_vars - 1`` the row."""
    point: torch.Tensor          # (n_vars, 6)
    value: torch.Tensor          # (6,)
    name: str = ""


@dataclass
class SumcheckProof:
    rounds1: torch.Tensor        # (n_k + n_c, 3, 6): q(0), q(1), q(inf) per round of the zero-check
    abc: torch.Tensor            # (3, 6): Az(r), Bz(r), Cz(r)
    rounds2: torch.Tensor        # (n_i, 3, 6): the row sumcheck
    next_vals: torch.Tensor      # (n_links, 6): the shifted rows' values at r_c
    rounds3: torch.Tensor        # (n_c, 3, 6): the shift sumcheck
    values: torch.Tensor         # (1 + n_links, 6): the PCS claim values (w~(r_i, r_c), then w~(bits(c_x), rho))
    claims: list = field(default_factory=list)    # EvalClaim (points derived from the coins; not serialised)
    timings: dict = field(default_factory=dict)

    def n_bytes(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        def ext_bytes(t: torch.Tensor) -> bytes:
            return t.reshape(-1).to(torch.int64).cpu().numpy().astype("<u4").tobytes()
        n1, n2, n3 = self.rounds1.shape[0], self.rounds2.shape[0], self.rounds3.shape[0]
        head = MAGIC + struct.pack("<BBBBB", n1, n2, n3, self.next_vals.shape[0], self.values.shape[0])
        return head + b"".join(ext_bytes(t) for t in (self.rounds1, self.abc, self.rounds2, self.next_vals, self.rounds3, self.values))

    @classmethod
    def from_bytes(cls, b: bytes, device="cpu") -> "SumcheckProof":
        if b[:8] != MAGIC:
            raise ValueError("not a LGSC0001 sumcheck proof")
        n1, n2, n3, nl, nv = struct.unpack("<BBBBB", b[8:13])
        import numpy as np
        pos = 13

        def take(shape):
            nonlocal pos
            n = 1
            for s in shape:
                n *= s
            arr = np.frombuffer(b[pos:pos + 4 * n], dtype="<u4").astype(np.int64).reshape(shape)
            pos += 4 * n
            return torch.from_numpy(arr.copy()).to(device)
        r1 = take((n1, 3, D)); abc = take((3, D)); r2 = take((n2, 3, D)); nv_ = take((nl, D)); r3 = take((n3, 3, D)); vals = take((nv, D))
        if pos != len(b):
            raise ValueError("trailing bytes")
        return cls(r1, abc, r2, nv_, r3, vals)


# ---------------------------------------------------------------------------------------------------------------------
# Prover
# ---------------------------------------------------------------------------------------------------------------------

def build_tables(cons: Constraints, z: torch.Tensor, chunk: int = 64, dtype=torch.int64, col_chunk: int = 1 << 16) -> dict[str, torch.Tensor]:
    """``Az, Bz, Cz`` ``(K_pad, C)`` base tables from the COO matrices (gathers of rows of ``z``); ``dtype`` int32 for the
    CUDA path (accumulated in int64 per column chunk)."""
    out = {}
    C = z.shape[1]
    for side, (ks, rs, vs) in cons.coo.items():
        ks, rs, vs = (t.to(z.device) for t in (ks, rs, vs))
        T = torch.zeros((cons.K_pad, C), dtype=dtype, device=z.device)
        for c0 in range(0, C, col_chunk):
            zc = z[:, c0:c0 + col_chunk]
            acc = torch.zeros((cons.K_pad, zc.shape[1]), dtype=torch.int64, device=z.device)
            for i0 in range(0, ks.numel(), chunk):
                k, r, v = ks[i0:i0 + chunk], rs[i0:i0 + chunk], vs[i0:i0 + chunk]
                acc.index_add_(0, k, (zc.index_select(0, r).to(torch.int64) * v[:, None]) % P)
            T[:, c0:c0 + col_chunk] = (acc % P).to(dtype)
        out[side] = T
    return out


class _RefState:
    """Reference sumcheck-1 state: ext tables ``(R, Cc, 6)`` int64 (torch/fc arithmetic; any device, small sizes)."""

    def __init__(self, tabs: dict, tau_rows: torch.Tensor, tau_cols: torch.Tensor):
        self.A, self.B, self.C = (ext_of_base(tabs[s]) for s in "abc")
        self.tau_rows, self.tau_cols = tau_rows, tau_cols        # remaining taus: rows (top-down order), cols
        self.phase = "rows"

    def _weights(self):
        """(R/2, Cc, 6) eq weights over the remaining (lower) variables."""
        tr = self.tau_rows[:-1] if self.phase == "rows" else self.tau_rows
        w_rows = eq_table(tr) if self.phase == "rows" else fc_one((1,), self.A.device)
        w_cols = eq_table(self.tau_cols[:-1] if self.phase == "cols" else self.tau_cols)
        return fc_mul(w_rows[:, None], w_cols[None, :])

    def round(self) -> torch.Tensor:
        if self.phase == "rows" and self.A.shape[0] == 1:
            self.phase = "cols"
            self.A, self.B, self.C = (T.transpose(0, 1) for T in (self.A, self.B, self.C))     # (Cc, 1, 6)
        h = self.A.shape[0] // 2
        A0, A1, B0, B1, C0, C1 = self.A[:h], self.A[h:], self.B[:h], self.B[h:], self.C[:h], self.C[h:]
        w = self._weights()
        if self.phase == "cols":
            w = w.transpose(0, 1)
        q0 = fc_mul(w, (fc_mul(A0, B0) - C0) % P).reshape(-1, D).sum(0) % P
        q1 = fc_mul(w, (fc_mul(A1, B1) - C1) % P).reshape(-1, D).sum(0) % P
        qi = fc_mul(w, fc_mul((A1 - A0) % P, (B1 - B0) % P)).reshape(-1, D).sum(0) % P
        return torch.stack((q0, q1, qi))

    def fold(self, r: torch.Tensor) -> None:
        self.A, self.B, self.C = (fold_ext(T, r) for T in (self.A, self.B, self.C))
        if self.phase == "rows":
            self.tau_rows = self.tau_rows[:-1]
        else:
            self.tau_cols = self.tau_cols[:-1]

    def final(self) -> torch.Tensor:
        return torch.stack([T.reshape(D) for T in (self.A, self.B, self.C)])


def _sumcheck_deg2(U: torch.Tensor, V: torch.Tensor, coins, label: str):
    """Reference degree-2 sumcheck of ``sum_x U(x) V(x)`` over ext tables ``(2^n, 6)``, top bit first.  Returns
    ``(rounds (n, 3, 6), r (n, 6) in VARIABLE order (r[t] <-> bit t), U(r), V(r))``."""
    n = U.shape[0].bit_length() - 1
    rounds, rs = [], []
    for t in range(n):
        h = U.shape[0] // 2
        q0 = fc_mul(U[:h], V[:h]).sum(0) % P
        q1 = fc_mul(U[h:], V[h:]).sum(0) % P
        qi = fc_mul((U[h:] - U[:h]) % P, (V[h:] - V[:h]) % P).sum(0) % P
        rounds.append(torch.stack((q0, q1, qi)))
        r = coins.challenge(f"{label}/r/{t}", 1)[0].to(U.device)
        rs.append(r)
        U, V = fold_ext(U, r), fold_ext(V, r)
    r_var = torch.stack(rs[::-1]) if rs else torch.zeros((0, D), dtype=torch.int64, device=U.device)
    return torch.stack(rounds), r_var, U.reshape(D), V.reshape(D)


def prove(lay: Layout, cons: Constraints, z: torch.Tensor, coins, cheat: str | None = None, engine=None) -> SumcheckProof:
    """The three sumchecks on ``z`` (``build_z``).  ``cheat="round1"``: a dishonest prover that fixes the first round's
    ``q(1)`` so the verifier's first check passes (caught in round 1); ``cheat="all"``: every zero-check round's ``q(1)`` is
    fixed to pass, so the verifier reaches the end and the final claim ``eq(tau, r)(Az Bz - Cz)(r)`` mismatches -- negative
    tests only.
    ``engine``: ``None`` = the reference torch path (small sizes); ``"cuda"`` = the fused cupy rounds (``_kernels``)."""
    dev = z.device
    t0 = time.perf_counter()
    timings: dict = {}
    n_k, n_c, n_i = lay.n_k, lay.n_c, lay.n_i
    tau = coins.challenge("zc/tau", n_k + n_c).to(dev)                      # tau[t] <-> variable t; k bits are the top n_k
    tau_cols, tau_rows = tau[:n_c], tau[n_c:]
    tabs = build_tables(cons, z, dtype=torch.int32 if engine == "cuda" else torch.int64)
    _sync(dev)
    timings["tables"] = time.perf_counter() - t0
    # --- sumcheck 1: the zero-check --------------------------------------------------------------------------------
    t1 = time.perf_counter()
    if engine == "cuda":
        from ._kernels import CudaState
        st = CudaState(tabs, tau_rows, tau_cols)
    else:
        st = _RefState(tabs, tau_rows, tau_cols)
    del tabs
    rounds1, rs1, per_round = [], [], []
    E = fc_one((), dev)
    claim = torch.zeros((D,), dtype=torch.int64, device=dev)
    for t in range(n_k + n_c):
        tr = time.perf_counter()
        msg = st.round()
        tau_t = tau[n_k + n_c - 1 - t]
        if cheat == "all" or (cheat == "round1" and t == 0):
            # q(1) := the value making E ((1 - tau) q0 + tau q1) == claim
            one = fc_one((), dev)
            want = fc_mul(claim, fc_inv(E))
            msg[1] = fc_mul((want - fc_mul((one - tau_t) % P, msg[0])) % P, fc_inv(tau_t))
        rounds1.append(msg)
        r = coins.challenge(f"zc/r/{t}", 1)[0].to(dev)
        rs1.append(r)
        st.fold(r)
        # running claim (prover side: only for the cheat mode / self-check)
        qr = _q_at(msg, r)
        E = fc_mul(E, _eq1(tau_t, r))
        claim = fc_mul(E, qr)
        _sync(dev)
        per_round.append(time.perf_counter() - tr)
    abc = st.final()
    r1 = torch.stack(rs1[::-1])                                              # r1[t] <-> variable t of (k, c)
    r_c, r_k = r1[:n_c], r1[n_c:]
    timings["sumcheck1"] = time.perf_counter() - t1
    timings["sumcheck1_rounds"] = per_round
    # --- sumcheck 2: rows --------------------------------------------------------------------------------------------
    t2 = time.perf_counter()
    gamma = coins.challenge("rows/gamma", 1)[0].to(dev)
    coef = combined_coefficients(cons, lay, r_k, gamma)                       # (R, 6)
    eq_c = eq_table(r_c)                                                      # (C, 6)
    zc = matvec_base_ext(z, eq_c)                                             # (R, 6) = z~(., r_c)
    rounds2, r_i, coef_r, zc_r = _sumcheck_deg2(coef, zc, coins, "rows")
    # the committed part only: rows >= m are zero in the commitment
    w_only = zc.clone()
    w_only[lay.m:] = 0
    w_val = mle_eval_ext(w_only, eq_table(r_i))
    next_vals = torch.stack([zc[lay.virt[f"next:{x}"]] for x in range(lay.n_links)]) if lay.n_links else \
        torch.zeros((0, D), dtype=torch.int64, device=dev)
    timings["sumcheck2"] = time.perf_counter() - t2
    # --- sumcheck 3: the shift ---------------------------------------------------------------------------------------
    t3 = time.perf_counter()
    g3 = coins.challenge("shift/gamma", 1)[0].to(dev)
    g3p = fc_pow_seq(g3, max(lay.n_links, 1))
    V = torch.zeros((lay.C, D), dtype=torch.int64, device=dev)
    for x, cr in enumerate(lay.c_rows):
        V = (V + fc_scalar_mul(z[cr].to(torch.int64), g3p[x][None, :].expand(lay.C, D))) % P
    U = torch.roll(eq_c.reshape(lay.S, lay.l, D), 1, dims=1).reshape(lay.C, D)      # U[c] = eq_c[pred(c)]
    rounds3, rho, U_r, V_r = _sumcheck_deg2(U, V, coins, "shift")
    eq_rho = eq_table(rho)
    vals = [w_val] + [mle_eval_base(z[cr].to(torch.int64), eq_rho) for cr in lay.c_rows]
    values = torch.stack(vals)
    timings["sumcheck3"] = time.perf_counter() - t3
    timings["total"] = time.perf_counter() - t0
    claims = [EvalClaim(torch.cat((r_c, r_i)), values[0], "w(r_i, r_c)")]
    for x, cr in enumerate(lay.c_rows):
        claims.append(EvalClaim(torch.cat((rho, _bits_point(cr, n_i, dev))), values[1 + x], f"w(c_{x}, rho)"))
    return SumcheckProof(torch.stack(rounds1), abc, rounds2, next_vals, rounds3, values, claims, timings)


def _bits_point(idx: int, n: int, device) -> torch.Tensor:
    out = torch.zeros((n, D), dtype=torch.int64, device=device)
    for t in range(n):
        out[t, 0] = (idx >> t) & 1
    return out


def _sync(dev):
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize(dev)


def _eq1(tau_t: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    one = fc_one((), r.device)
    return (fc_mul(tau_t, r) + fc_mul((one - tau_t) % P, (one - r) % P)) % P


def _q_at(msg: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """``q(r)`` from ``q(0), q(1), q(inf)``: ``q0 + r (q1 - q0) + q_inf r (r - 1)``."""
    q0, q1, qi = msg[0], msg[1], msg[2]
    one = fc_one((), r.device)
    return (q0 + fc_mul(r, (q1 - q0) % P) + fc_mul(qi, fc_mul(r, (r - one) % P))) % P


def combined_coefficients(cons: Constraints, lay: Layout, r_k: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """``coef = eq(r_k, .)^T (A + gamma B + gamma^2 C)`` -> ``(R, 6)`` (the verifier's, ``nnz`` ext ops)."""
    dev = r_k.device
    eq_k = eq_table(r_k)                                                     # (K_pad, 6)
    gp = fc_pow_seq(gamma, 3)
    coef = torch.zeros((lay.R, D), dtype=torch.int64, device=dev)
    for side, g in zip("abc", gp):
        ks, rs, vs = (t.to(dev) for t in cons.coo[side])
        contrib = fc_scalar_mul(vs, fc_mul(eq_k.index_select(0, ks), g))    # (nnz, 6)
        coef.index_add_(0, rs, contrib)
        coef %= P
    return coef


# ---------------------------------------------------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------------------------------------------------

def verify(lay: Layout, cons: Constraints, pub_rows: dict[str, torch.Tensor], proof: SumcheckProof, coins,
           device="cpu") -> tuple[bool, str, list]:
    """Check the three sumchecks against the public data; returns ``(ok, reason, claims)`` -- the PCS claims the caller
    must have verified against the commitment for the whole thing to hold.  ``pub_rows``: ``layout.public_rows``."""
    dev = torch.device(device)
    one = fc_one((), dev)
    n_k, n_c, n_i = lay.n_k, lay.n_c, lay.n_i
    if proof.rounds1.shape != (n_k + n_c, 3, D) or proof.rounds2.shape != (n_i, 3, D) or proof.rounds3.shape != (n_c, 3, D) \
            or proof.next_vals.shape != (lay.n_links, D) or proof.values.shape != (1 + lay.n_links, D):
        return False, "shape mismatch", []
    for t in (proof.rounds1, proof.abc, proof.rounds2, proof.next_vals, proof.rounds3, proof.values):
        if bool((t < 0).any()) or bool((t >= P).any()):
            return False, "non-canonical field element", []
    tau = coins.challenge("zc/tau", n_k + n_c).to(dev)
    # --- sumcheck 1 --------------------------------------------------------------------------------------------------
    E = one.clone()
    claim = torch.zeros((D,), dtype=torch.int64, device=dev)
    rs = []
    for t in range(n_k + n_c):
        msg = proof.rounds1[t].to(dev)
        tau_t = tau[n_k + n_c - 1 - t]
        lhs = fc_mul(E, (fc_mul((one - tau_t) % P, msg[0]) + fc_mul(tau_t, msg[1])) % P)
        if not torch.equal(lhs, claim):
            return False, f"zero-check round {t}: (1 - tau) q(0) + tau q(1) != claim", []
        r = coins.challenge(f"zc/r/{t}", 1)[0].to(dev)
        rs.append(r)
        E = fc_mul(E, _eq1(tau_t, r))
        claim = fc_mul(E, _q_at(msg, r))
    abc = proof.abc.to(dev)
    if not torch.equal(fc_mul(E, (fc_mul(abc[0], abc[1]) - abc[2]) % P), claim):
        return False, "zero-check final: eq(tau, r) (Az Bz - Cz)(r) != claim", []
    r1 = torch.stack(rs[::-1])
    r_c, r_k = r1[:n_c], r1[n_c:]
    # --- sumcheck 2 --------------------------------------------------------------------------------------------------
    gamma = coins.challenge("rows/gamma", 1)[0].to(dev)
    gp = fc_pow_seq(gamma, 3)
    claim = (abc[0] + fc_mul(gp[1], abc[1]) + fc_mul(gp[2], abc[2])) % P
    coef = combined_coefficients(cons, lay, r_k, gamma)
    rs2 = []
    for t in range(n_i):
        msg = proof.rounds2[t].to(dev)
        if not torch.equal((msg[0] + msg[1]) % P, claim):
            return False, f"row sumcheck round {t}: q(0) + q(1) != claim", []
        r = coins.challenge(f"rows/r/{t}", 1)[0].to(dev)
        rs2.append(r)
        claim = _q_at(msg, r)
    r_i = torch.stack(rs2[::-1])
    eq_i = eq_table(r_i)
    coef_r = mle_eval_ext(coef, eq_i)
    eq_c = eq_table(r_c)
    # z~(r_i, r_c) = w~(r_i, r_c) + sum_v eq(r_i, idx_v) virt_v(r_c)   (one matvec for all public virtual rows)
    z_r = proof.values[0].to(dev).clone()
    pub_names = [n for n in lay.virt if not n.startswith("next:")]
    pub_vals = matvec_base_ext(torch.stack([pub_rows[n].to(dev) for n in pub_names]), eq_c)      # (n_pub, 6)
    for name, v in zip(pub_names, pub_vals):
        z_r = (z_r + fc_mul(eq_i[lay.virt[name]], v)) % P
    for x in range(lay.n_links):
        z_r = (z_r + fc_mul(eq_i[lay.virt[f"next:{x}"]], proof.next_vals[x].to(dev))) % P
    if not torch.equal(fc_mul(coef_r, z_r), claim):
        return False, "row sumcheck final: coef(r_i) z(r_i, r_c) != claim", []
    # --- sumcheck 3 --------------------------------------------------------------------------------------------------
    g3 = coins.challenge("shift/gamma", 1)[0].to(dev)
    g3p = fc_pow_seq(g3, max(lay.n_links, 1))
    claim = torch.zeros((D,), dtype=torch.int64, device=dev)
    for x in range(lay.n_links):
        claim = (claim + fc_mul(g3p[x], proof.next_vals[x].to(dev))) % P
    rs3 = []
    for t in range(n_c):
        msg = proof.rounds3[t].to(dev)
        if not torch.equal((msg[0] + msg[1]) % P, claim):
            return False, f"shift sumcheck round {t}: q(0) + q(1) != claim", []
        r = coins.challenge(f"shift/r/{t}", 1)[0].to(dev)
        rs3.append(r)
        claim = _q_at(msg, r)
    rho = torch.stack(rs3[::-1])
    succ = succ_mle(rho, r_c, lay.n_j)
    V_r = torch.zeros((D,), dtype=torch.int64, device=dev)
    for x in range(lay.n_links):
        V_r = (V_r + fc_mul(g3p[x], proof.values[1 + x].to(dev))) % P
    if not torch.equal(fc_mul(succ, V_r), claim):
        return False, "shift sumcheck final: succ(rho, r_c) V(rho) != claim", []
    claims = [EvalClaim(torch.cat((r_c, r_i)), proof.values[0].to(dev), "w(r_i, r_c)")]
    for x, cr in enumerate(lay.c_rows):
        claims.append(EvalClaim(torch.cat((rho, _bits_point(cr, n_i, dev))), proof.values[1 + x].to(dev), f"w(c_{x}, rho)"))
    return True, "accept (sumchecks; PCS claims pending)", claims


def ligero_subbatches(rel, runner, vus: list, l: int, S: int, device) -> tuple[list, list]:
    """The witnesses of a batch exactly as Ligero's prover materialises them: ``vus`` (the ``relchain.instances`` tuples)
    split into sub-batches of ``l`` columns (``floor(l / steps)`` VUs each), padded to ``S`` sub-batches with 0-VU
    sub-batches (all pad units).  Uses ``runner.marshal`` (public words + pub vectors), ``rel.hints`` and
    ``witness.build_witness`` -- the Ligero code PRODUCES the witness; nothing of its proof runs.  Returns
    ``(subs, n_vus)``."""
    from backends.direct.ligero.witness import build_witness
    per = l // rel.steps
    groups = [vus[i:i + per] for i in range(0, len(vus), per)]
    if len(groups) > S:
        raise ValueError(f"{len(vus)} VUs need {len(groups)} sub-batches > S = {S}")
    subs, n_vus = [], []
    for g in groups + [[]] * (S - len(groups)):
        bd = runner.marshal(list(g), min_l=l)
        assert bd["lay"].l == l, (bd["lay"].l, l)
        hints = rel.hints(rel.params, runner.sys, bd["c_d"], bd["a_d"], bd["b_d"], bd["y_d"], device)
        W = build_witness(runner.sys, bd["pub"], hints, device)
        subs.append(SubBatchT(W, bd["pub"], bd["ypub_d"], len(g)))
        n_vus.append(len(g))
    return subs, n_vus


def check_claims(lay: Layout, z: torch.Tensor, claims: list) -> bool:
    """Diagnostic: evaluate the committed polynomial (``z`` with the virtual rows zeroed) at every claim point."""
    w = z.clone()
    w[lay.m:] = 0
    for cl in claims:
        eq_c = eq_table(cl.point[:lay.n_c])
        eq_i = eq_table(cl.point[lay.n_c:])
        v = mle_eval_ext(matvec_base_ext(w, eq_c), eq_i)
        if not torch.equal(v, cl.value.to(v.device)):
            return False
    return True


# ---------------------------------------------------------------------------------------------------------------------
# Measurement harness: ``python -m backends.direct.ligerito.sumcheck --relation fp8-ada --vus 4096 --engine cuda``
# ---------------------------------------------------------------------------------------------------------------------

def bench(relation_name: str, n_vus: int, l: int, S: int, device: str, engine: str | None, negatives: bool,
          seed: int = 7) -> dict:
    """Witnesses via the Ligero code (produce only), the three sumchecks, the verifier, bytes, peak memory; optional
    negatives (violated quadratic row / chain link / lookup: the honest prover on the bad witness fails round 0, the
    cheating prover reaches the final claim and mismatches)."""
    from backends.direct.ligero.relations import relation
    from backends.direct.ligero.relchain import RelationChainRunner, instances
    from .layout import build_z, constraints, layout_for, public_rows
    dev = torch.device(device)
    out: dict = {"relation": relation_name, "vus": n_vus, "l": l, "S": S, "engine": engine or "ref"}
    t0 = time.perf_counter()
    rel = relation(relation_name)
    runner = RelationChainRunner(rel, device, -128.0)
    vus = instances(rel, n_vus, seed=seed)
    subs, nv = ligero_subbatches(rel, runner, vus, l, S, device)
    _sync(dev)
    out["t_witness"] = time.perf_counter() - t0
    lay = layout_for(runner.sys, l, nv, rel.steps)
    cons = constraints(runner.sys, lay, device)
    out.update(m=lay.m, R=lay.R, C=lay.C, K=lay.K, K_pad=cons.K_pad, n_k=lay.n_k, n_c=lay.n_c, n_i=lay.n_i,
               n_links=lay.n_links, nnz=cons.nnz(), n_sub=len([n for n in nv if n]))
    t1 = time.perf_counter()
    z = build_z(runner.sys, lay, subs, device)
    pub_subs = [s.public() for s in subs]
    del subs
    _sync(dev)
    out["t_build_z"] = time.perf_counter() - t1
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
        base_mem = torch.cuda.memory_allocated(dev)
    coins = LocalCoins(b"bench")
    proof = prove(lay, cons, z, coins, engine=engine)
    if dev.type == "cuda":
        out["peak_mem_gb"] = (torch.cuda.max_memory_allocated(dev)) / 1e9
        out["base_mem_gb"] = base_mem / 1e9
    out["timings"] = proof.timings
    out["bytes"] = proof.n_bytes()
    t2 = time.perf_counter()
    pubs = public_rows(runner.sys, lay, pub_subs, device)
    ok, why, claims = verify(lay, cons, pubs, proof, LocalCoins(b"bench"), device=device)
    _sync(dev)
    out["t_verify"] = time.perf_counter() - t2
    out["verify_ok"], out["verify_why"] = ok, why
    if negatives:
        neg = {}
        sys = runner.sys
        bit = next(r for r in sys.rows if r.kind == "bit")
        sel = [r for r in sys.rows if r.kind == "sel"]
        cases = {
            "quadratic": [(lay.row_of_sys[bit.idx], 3, 2)],
            "chain_link": [(lay.c_rows[0], 5, None), (lay.virt["next:0"], 4, "copy")],
            "lookup": [(lay.row_of_sys[sel[0].idx], 7, 1), (lay.row_of_sys[sel[1].idx], 7, 1)],
        }
        for name, edits in cases.items():
            zb = z.clone()
            for row, col, val in edits:
                if val is None:
                    zb[row, col] = (int(zb[row, col]) + 1) % P
                elif val == "copy":
                    zb[row, col] = zb[lay.c_rows[0], col + 1]
                else:
                    zb[row, col] = val
            res = {}
            for cheat in (None, "all"):
                pr = prove(lay, cons, zb, LocalCoins(b"neg"), cheat=cheat, engine=engine)
                ok_n, why_n, _ = verify(lay, cons, pubs, pr, LocalCoins(b"neg"), device=device)
                res[cheat or "honest"] = (ok_n, why_n)
            neg[name] = res
            del zb
        out["negatives"] = neg
    return out


def _main():
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--relation", default="fp8-ada")
    ap.add_argument("--vus", type=int, default=4096)
    ap.add_argument("--l", type=int, default=16384)
    ap.add_argument("--S", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--negatives", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = bench(a.relation, a.vus, a.l, a.S, a.device, a.engine, a.negatives)
    s = json.dumps(res, indent=1, default=str)
    print(s)
    if a.json:
        with open(a.json, "w") as f:
            f.write(s)


if __name__ == "__main__":
    _main()
