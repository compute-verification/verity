"""The constraint layer of the Ligerito backend (lane ligerito-sumcheck): a batched degree-3 zero-check over every
constraint of a ``compile.System`` on the multilinear layout of ``layout.py``, ending in evaluation claims on the committed
polynomial.  Torch prover (reference path here; fused cupy CUDA rounds (``CudaState``, ``engine="cuda"``)) + Python verifier.

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

Message bytes: ``9 + (n_k + n_c - 2 + n_i + n_c) x 3 + 3 + n_links + 1 + n_links`` extension elements of 24 B (the zero-check's
opening round binds the two top variables with one bivariate message of 9 elements; every later round one variable).
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

MAGIC = b"LGSC0002"


# ---------------------------------------------------------------------------------------------------------------------
# Coins (stub: deterministic from a seed; the protocol is ``challenge(label, n) -> (n, 6)``)
# ---------------------------------------------------------------------------------------------------------------------

class LocalCoins:
    """Diagnostic coins with ligerito-proto's ``transcript.py`` surface: ``challenge(label: bytes, n) -> np.ndarray (n, 6)``
    (SHAKE-256(seed || label || round) reduced mod p per coordinate), ``absorb`` ignored, ``indices``, ``rounds``.  Prover
    and verifier construct it from the same seed.  (``transcript.LocalCoins`` / ``FiatShamirCoins`` / ``LiveCoins`` replace it.)"""

    def __init__(self, seed: bytes = b"ligerito-sumcheck", device="cpu"):
        self.seed, self.device, self.rounds = seed, device, 0

    def absorb(self, label: bytes, data: bytes) -> None:
        pass

    def challenge(self, label: bytes, n: int):
        import numpy as np
        h = hashlib.shake_256(self.seed + b"|" + bytes(label) + b"|" + struct.pack("<I", self.rounds))
        self.rounds += 1
        raw = np.frombuffer(h.digest(8 * n * D), dtype="<u8").astype(object) % P
        return np.array(raw, dtype=np.int64).reshape(n, D)

    def indices(self, label: bytes, domain: int, count: int) -> list[int]:
        h = hashlib.shake_256(self.seed + b"|" + bytes(label) + b"|" + struct.pack("<I", self.rounds))
        self.rounds += 1
        out: list[int] = []
        raw = h.digest(8 * 4 * count)
        for i in range(4 * count):
            v = int.from_bytes(raw[8 * i:8 * i + 8], "little") % domain
            if v not in out:
                out.append(v)
            if len(out) == count:
                break
        return out


class _Transcript:
    """The sumcheck's view of a Coins object: torch challenges on ``device``; every prover message is absorbed under its
    label before the next coin (Fiat-Shamir needs it; LocalCoins / LiveCoins ignore it)."""

    def __init__(self, coins, device):
        self.coins, self.device = coins, torch.device(device)

    def absorb(self, label: str, t: torch.Tensor) -> None:
        absorb = getattr(self.coins, "absorb", None)
        if absorb is not None:
            absorb(label.encode(), _ext_bytes(t))

    def challenge(self, label: str, n: int) -> torch.Tensor:
        v = self.coins.challenge(label.encode(), n)
        if isinstance(v, torch.Tensor):
            return v.to(self.device, torch.int64).reshape(n, D)
        return torch.as_tensor(v, dtype=torch.int64).to(self.device).reshape(n, D)


def _ext_bytes(t: torch.Tensor) -> bytes:
    """Canonical wire form of a tensor of ext elements: coordinates as little-endian u32, row-major."""
    return (t.reshape(-1).to(torch.int64) % P).cpu().numpy().astype("<u4").tobytes()


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
    """``w~(point) = value`` on the committed polynomial; ``point[t]`` is the coordinate of variable ``t`` = bit ``t`` of the
    flat PCS index ``x = i + R * c`` (ligerito-proto's column-major layout): variables ``0 .. n_i - 1`` are the row ``i``,
    ``n_i .. n_vars - 1`` the column ``c = s * l + j``."""
    point: torch.Tensor          # (n_vars, 6)
    value: torch.Tensor          # (6,)
    name: str = ""


@dataclass
class SumcheckProof:
    round0: torch.Tensor         # (3, 3, 6): the zero-check's opening bivariate round (two top variables), M[a][b], see ``_q2_at``
    rounds1: torch.Tensor        # (n_k + n_c - 2, 3, 6): q(0), q(1), q(inf) per remaining round of the zero-check
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
        ext_bytes = _ext_bytes
        n1, n2, n3 = self.rounds1.shape[0], self.rounds2.shape[0], self.rounds3.shape[0]
        head = MAGIC + struct.pack("<BBBBB", n1, n2, n3, self.next_vals.shape[0], self.values.shape[0])
        return head + b"".join(ext_bytes(t) for t in (self.round0, self.rounds1, self.abc, self.rounds2, self.next_vals, self.rounds3, self.values))

    @classmethod
    def from_bytes(cls, b: bytes, device="cpu") -> "SumcheckProof":
        if b[:8] != MAGIC:
            raise ValueError("not a LGSC0002 sumcheck proof")
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
        r0 = take((3, 3, D)); r1 = take((n1, 3, D)); abc = take((3, D)); r2 = take((n2, 3, D)); nv_ = take((nl, D)); r3 = take((n3, 3, D)); vals = take((nv, D))
        if pos != len(b):
            raise ValueError("trailing bytes")
        return cls(r0, r1, abc, r2, nv_, r3, vals)


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

    def round2(self) -> torch.Tensor:
        """The opening bivariate round over the two top row variables: ``M (3, 3, 6)``, ``M[a][b]`` = the X1-representation
        (index ``a``: value at 0, at 1, X1^2 coefficient) of the X2-representation (index ``b``) of ``q(X1, X2)``."""
        assert self.phase == "rows" and self.A.shape[0] >= 4
        h = self.A.shape[0] // 4
        w = fc_mul(eq_table(self.tau_rows[:-2])[:, None], eq_table(self.tau_cols)[None, :])       # (h, Cc, 6)
        q = [[None] * 3 for _ in range(3)]
        A, B, C = 0, 1, 2
        corners = [(T[:h], T[h:2 * h], T[2 * h:3 * h], T[3 * h:]) for T in (self.A, self.B, self.C)]   # (00, 01, 10, 11): X1 X2

        def lin(T, b):
            """X1-linear (const, slope) of the X2-representation entry ``b`` of the bilinear T."""
            t00, t01, t10, t11 = corners[T]
            if b == 0:
                return t00, (t10 - t00) % P
            if b == 1:
                return t01, (t11 - t01) % P
            return (t01 - t00) % P, (t11 - t10 - t01 + t00) % P
        for b in range(3):
            u0, u1 = lin(A, b)
            v0, v1 = lin(B, b)
            w0, w1 = lin(C, b) if b < 2 else (torch.zeros_like(u0), torch.zeros_like(u0))
            f0 = (fc_mul(u0, v0) - w0) % P
            f1 = (fc_mul((u0 + u1) % P, (v0 + v1) % P) - (w0 + w1)) % P
            f2 = fc_mul(u1, v1)
            for a, f in enumerate((f0, f1, f2)):
                q[a][b] = fc_mul(w, f).reshape(-1, D).sum(0) % P
        return torch.stack([torch.stack(row) for row in q])

    def fold2(self, r1: torch.Tensor, r2: torch.Tensor) -> None:
        h = self.A.shape[0] // 4

        def f(T):
            t00, t01, t10, t11 = T[:h], T[h:2 * h], T[2 * h:3 * h], T[3 * h:]
            al0 = (t00 + fc_mul((t10 - t00) % P, r1)) % P
            al1 = ((t01 - t00) % P + fc_mul((t11 - t10 - t01 + t00) % P, r1)) % P
            return (al0 + fc_mul(al1, r2)) % P
        self.A, self.B, self.C = f(self.A), f(self.B), f(self.C)
        self.tau_rows = self.tau_rows[:-2]

    def final(self) -> torch.Tensor:
        return torch.stack([T.reshape(D) for T in (self.A, self.B, self.C)])


def _sumcheck_deg2(U: torch.Tensor, V: torch.Tensor, coins, label: str, engine=None):
    """Degree-2 sumcheck of ``sum_x U(x) V(x)`` over ext tables ``(2^n, 6)``, top bit first (torch reference, or the CUDA
    kernels with uniform weights).  Returns ``(rounds (n, 3, 6), r (n, 6) in VARIABLE order (r[t] <-> bit t), U(r), V(r))``."""
    n = U.shape[0].bit_length() - 1
    rounds, rs = [], []
    if engine == "cuda":
        st = CudaState.from_ext(U, V)
        for t in range(n):
            msg = st.round()
            rounds.append(msg)
            coins.absorb(f"{label}/q/{t}", msg)
            r = coins.challenge(f"{label}/r/{t}", 1)[0]
            rs.append(r)
            st.fold(r)
        fin = st.final()
        r_var = torch.stack(rs[::-1]) if rs else torch.zeros((0, D), dtype=torch.int64, device=U.device)
        return torch.stack(rounds), r_var, fin[0], fin[1]
    for t in range(n):
        h = U.shape[0] // 2
        q0 = fc_mul(U[:h], V[:h]).sum(0) % P
        q1 = fc_mul(U[h:], V[h:]).sum(0) % P
        qi = fc_mul((U[h:] - U[:h]) % P, (V[h:] - V[:h]) % P).sum(0) % P
        rounds.append(torch.stack((q0, q1, qi)))
        coins.absorb(f"{label}/q/{t}", rounds[-1])
        r = coins.challenge(f"{label}/r/{t}", 1)[0]
        rs.append(r)
        U, V = fold_ext(U, r), fold_ext(V, r)
    r_var = torch.stack(rs[::-1]) if rs else torch.zeros((0, D), dtype=torch.int64, device=U.device)
    return torch.stack(rounds), r_var, U.reshape(D), V.reshape(D)


def prove(lay: Layout, cons: Constraints, z: torch.Tensor, coins, cheat: str | None = None, engine=None) -> SumcheckProof:
    """The three sumchecks on ``z`` (``build_z``).  ``cheat="round1"``: a dishonest prover that fixes the opening round's
    ``M[1][1]`` so the verifier's first check passes (caught in the next round); ``cheat="all"``: every zero-check round's
    ``q(1)`` is fixed to pass, so the verifier reaches the end and the final claim ``eq(tau, r)(Az Bz - Cz)(r)`` mismatches --
    negative tests only.
    ``engine``: ``None`` = the reference torch path (small sizes); ``"cuda"`` = the fused cupy rounds (``CudaState``)."""
    dev = z.device
    coins = _Transcript(coins, dev)
    t0 = time.perf_counter()
    timings: dict = {}
    n_k, n_c, n_i = lay.n_k, lay.n_c, lay.n_i
    tau = coins.challenge("zc/tau", n_k + n_c)                      # tau[t] <-> variable t; k bits are the top n_k
    tau_cols, tau_rows = tau[:n_c], tau[n_c:]
    tabs = build_tables_cuda(cons, z) if engine == "cuda" else build_tables(cons, z)
    _sync(dev)
    timings["tables"] = time.perf_counter() - t0
    # --- sumcheck 1: the zero-check --------------------------------------------------------------------------------
    t1 = time.perf_counter()
    if engine == "cuda":
        st = CudaState(tabs, tau_rows, tau_cols)
    else:
        st = _RefState(tabs, tau_rows, tau_cols)
    del tabs
    rounds1, rs1, per_round = [], [], []
    E = fc_one((), dev)
    one = fc_one((), dev)
    claim = torch.zeros((D,), dtype=torch.int64, device=dev)
    # opening round: the two top variables at once (halves the extension-field volume of the remaining rounds)
    tr = time.perf_counter()
    M = st.round2()
    tau1, tau2 = tau[n_k + n_c - 1], tau[n_k + n_c - 2]
    if cheat in ("all", "round1"):
        # M[1][1] := the value making E sum_{x1,x2} eq(tau1, x1) eq(tau2, x2) M[x1][x2] == claim (= 0 here)
        other = (fc_mul(fc_mul((one - tau1) % P, (one - tau2) % P), M[0][0]) + fc_mul(fc_mul((one - tau1) % P, tau2), M[0][1])
                 + fc_mul(fc_mul(tau1, (one - tau2) % P), M[1][0])) % P
        M[1][1] = fc_mul((fc_mul(claim, fc_inv(E)) - other) % P, fc_inv(fc_mul(tau1, tau2)))
    coins.absorb("zc/q/0", M)
    r12 = coins.challenge("zc/r01", 2)
    rs1 += [r12[0], r12[1]]
    st.fold2(r12[0], r12[1])
    if cheat:
        E = fc_mul(E, fc_mul(_eq1(tau1, r12[0]), _eq1(tau2, r12[1])))
        claim = fc_mul(E, _q2_at(M, r12[0], r12[1]))
    _sync(dev)
    per_round.append(time.perf_counter() - tr)
    for t in range(2, n_k + n_c):
        tr = time.perf_counter()
        msg = st.round()
        tau_t = tau[n_k + n_c - 1 - t]
        if cheat == "all":
            # q(1) := the value making E ((1 - tau) q0 + tau q1) == claim
            want = fc_mul(claim, fc_inv(E))
            msg[1] = fc_mul((want - fc_mul((one - tau_t) % P, msg[0])) % P, fc_inv(tau_t))
        rounds1.append(msg)
        coins.absorb(f"zc/q/{t}", msg)
        r = coins.challenge(f"zc/r/{t}", 1)[0]
        rs1.append(r)
        st.fold(r)
        if cheat:
            # running claim (prover side: only the cheat mode needs it)
            qr = _q_at(msg, r)
            E = fc_mul(E, _eq1(tau_t, r))
            claim = fc_mul(E, qr)
        _sync(dev)
        per_round.append(time.perf_counter() - tr)
    abc = st.final()
    coins.absorb("zc/abc", abc)
    r1 = torch.stack(rs1[::-1])                                              # r1[t] <-> variable t of (k, c)
    r_c, r_k = r1[:n_c], r1[n_c:]
    timings["sumcheck1"] = time.perf_counter() - t1
    timings["sumcheck1_rounds"] = per_round
    # --- sumcheck 2: rows --------------------------------------------------------------------------------------------
    t2 = time.perf_counter()
    gamma = coins.challenge("rows/gamma", 1)[0]
    coef = combined_coefficients(cons, lay, r_k, gamma)                       # (R, 6)
    _sync(dev)
    timings["sumcheck2_coef"] = time.perf_counter() - t2
    eq_c = (eq_table_cuda if engine == "cuda" else eq_table)(r_c)             # (C, 6)
    _sync(dev)
    timings["sumcheck2_eq"] = time.perf_counter() - t2 - timings["sumcheck2_coef"]
    zc = (matvec_base_ext_cuda if engine == "cuda" else matvec_base_ext)(z, eq_c)      # (R, 6) = z~(., r_c)
    _sync(dev)
    timings["sumcheck2_matvec"] = time.perf_counter() - t2 - timings["sumcheck2_coef"] - timings["sumcheck2_eq"]
    rounds2, r_i, coef_r, zc_r = _sumcheck_deg2(coef, zc, coins, "rows", engine)
    # the committed part only: rows >= m are zero in the commitment
    w_only = zc.clone()
    w_only[lay.m:] = 0
    w_val = mle_eval_ext(w_only, (eq_table_cuda if engine == "cuda" else eq_table)(r_i))
    next_vals = torch.stack([zc[lay.virt[f"next:{x}"]] for x in range(lay.n_links)]) if lay.n_links else \
        torch.zeros((0, D), dtype=torch.int64, device=dev)
    coins.absorb("rows/next", next_vals)
    timings["sumcheck2"] = time.perf_counter() - t2
    # --- sumcheck 3: the shift ---------------------------------------------------------------------------------------
    t3 = time.perf_counter()
    g3 = coins.challenge("shift/gamma", 1)[0]
    g3p = fc_pow_seq(g3, max(lay.n_links, 1))
    V = torch.zeros((lay.C, D), dtype=torch.int64, device=dev)
    for x, cr in enumerate(lay.c_rows):
        V = (V + fc_scalar_mul(z[cr].to(torch.int64), g3p[x][None, :].expand(lay.C, D))) % P
    U = torch.roll(eq_c.reshape(lay.S, lay.l, D), 1, dims=1).reshape(lay.C, D)      # U[c] = eq_c[pred(c)]
    rounds3, rho, U_r, V_r = _sumcheck_deg2(U, V, coins, "shift", engine)
    eq_rho = (eq_table_cuda if engine == "cuda" else eq_table)(rho)
    if engine == "cuda":
        vals = [w_val] + list(matvec_base_ext_cuda(z.index_select(0, torch.tensor(lay.c_rows, device=dev)), eq_rho))
    else:
        vals = [w_val] + [mle_eval_base(z[cr].to(torch.int64), eq_rho) for cr in lay.c_rows]
    values = torch.stack(vals)
    coins.absorb("values", values)
    timings["sumcheck3"] = time.perf_counter() - t3
    timings["total"] = time.perf_counter() - t0
    claims = [EvalClaim(torch.cat((r_i, r_c)), values[0], "w(r_i, r_c)")]
    for x, cr in enumerate(lay.c_rows):
        claims.append(EvalClaim(torch.cat((_bits_point(cr, n_i, dev), rho)), values[1 + x], f"w(c_{x}, rho)"))
    return SumcheckProof(M, torch.stack(rounds1), abc, rounds2, next_vals, rounds3, values, claims, timings)


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


def _q2_at(M: torch.Tensor, r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    """``q(r1, r2)`` from the bivariate message ``M[a][b]``: evaluate every X2-entry's X1-representation at ``r1``, then the
    resulting X2-representation at ``r2``."""
    g = torch.stack([_q_at(M[:, b], r1) for b in range(3)])
    return _q_at(g, r2)


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
    coins = _Transcript(coins, dev)
    one = fc_one((), dev)
    n_k, n_c, n_i = lay.n_k, lay.n_c, lay.n_i
    if proof.round0.shape != (3, 3, D) or proof.rounds1.shape != (n_k + n_c - 2, 3, D) or proof.rounds2.shape != (n_i, 3, D) or proof.rounds3.shape != (n_c, 3, D) \
            or proof.next_vals.shape != (lay.n_links, D) or proof.values.shape != (1 + lay.n_links, D):
        return False, "shape mismatch", []
    for t in (proof.round0, proof.rounds1, proof.abc, proof.rounds2, proof.next_vals, proof.rounds3, proof.values):
        if bool((t < 0).any()) or bool((t >= P).any()):
            return False, "non-canonical field element", []
    tau = coins.challenge("zc/tau", n_k + n_c)
    # --- sumcheck 1 --------------------------------------------------------------------------------------------------
    E = one.clone()
    claim = torch.zeros((D,), dtype=torch.int64, device=dev)
    rs = []
    M = proof.round0.to(dev)
    tau1, tau2 = tau[n_k + n_c - 1], tau[n_k + n_c - 2]
    lhs = (fc_mul(fc_mul((one - tau1) % P, (one - tau2) % P), M[0][0]) + fc_mul(fc_mul((one - tau1) % P, tau2), M[0][1])
           + fc_mul(fc_mul(tau1, (one - tau2) % P), M[1][0]) + fc_mul(fc_mul(tau1, tau2), M[1][1])) % P
    if not torch.equal(fc_mul(E, lhs), claim):
        return False, "zero-check round 0: sum_{x1,x2} eq(tau, x) M[x1][x2] != claim", []
    coins.absorb("zc/q/0", M)
    r12 = coins.challenge("zc/r01", 2)
    rs += [r12[0], r12[1]]
    E = fc_mul(E, fc_mul(_eq1(tau1, r12[0]), _eq1(tau2, r12[1])))
    claim = fc_mul(E, _q2_at(M, r12[0], r12[1]))
    for t in range(2, n_k + n_c):
        msg = proof.rounds1[t - 2].to(dev)
        tau_t = tau[n_k + n_c - 1 - t]
        lhs = fc_mul(E, (fc_mul((one - tau_t) % P, msg[0]) + fc_mul(tau_t, msg[1])) % P)
        if not torch.equal(lhs, claim):
            return False, f"zero-check round {t}: (1 - tau) q(0) + tau q(1) != claim", []
        coins.absorb(f"zc/q/{t}", msg)
        r = coins.challenge(f"zc/r/{t}", 1)[0]
        rs.append(r)
        E = fc_mul(E, _eq1(tau_t, r))
        claim = fc_mul(E, _q_at(msg, r))
    abc = proof.abc.to(dev)
    if not torch.equal(fc_mul(E, (fc_mul(abc[0], abc[1]) - abc[2]) % P), claim):
        return False, "zero-check final: eq(tau, r) (Az Bz - Cz)(r) != claim", []
    coins.absorb("zc/abc", abc)
    r1 = torch.stack(rs[::-1])
    r_c, r_k = r1[:n_c], r1[n_c:]
    # --- sumcheck 2 --------------------------------------------------------------------------------------------------
    gamma = coins.challenge("rows/gamma", 1)[0]
    gp = fc_pow_seq(gamma, 3)
    claim = (abc[0] + fc_mul(gp[1], abc[1]) + fc_mul(gp[2], abc[2])) % P
    coef = combined_coefficients(cons, lay, r_k, gamma)
    rs2 = []
    for t in range(n_i):
        msg = proof.rounds2[t].to(dev)
        if not torch.equal((msg[0] + msg[1]) % P, claim):
            return False, f"row sumcheck round {t}: q(0) + q(1) != claim", []
        coins.absorb(f"rows/q/{t}", msg)
        r = coins.challenge(f"rows/r/{t}", 1)[0]
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
    coins.absorb("rows/next", proof.next_vals)
    # --- sumcheck 3 --------------------------------------------------------------------------------------------------
    g3 = coins.challenge("shift/gamma", 1)[0]
    g3p = fc_pow_seq(g3, max(lay.n_links, 1))
    claim = torch.zeros((D,), dtype=torch.int64, device=dev)
    for x in range(lay.n_links):
        claim = (claim + fc_mul(g3p[x], proof.next_vals[x].to(dev))) % P
    rs3 = []
    for t in range(n_c):
        msg = proof.rounds3[t].to(dev)
        if not torch.equal((msg[0] + msg[1]) % P, claim):
            return False, f"shift sumcheck round {t}: q(0) + q(1) != claim", []
        coins.absorb(f"shift/q/{t}", msg)
        r = coins.challenge(f"shift/r/{t}", 1)[0]
        rs3.append(r)
        claim = _q_at(msg, r)
    rho = torch.stack(rs3[::-1])
    succ = succ_mle(rho, r_c, lay.n_j)
    V_r = torch.zeros((D,), dtype=torch.int64, device=dev)
    for x in range(lay.n_links):
        V_r = (V_r + fc_mul(g3p[x], proof.values[1 + x].to(dev))) % P
    if not torch.equal(fc_mul(succ, V_r), claim):
        return False, "shift sumcheck final: succ(rho, r_c) V(rho) != claim", []
    coins.absorb("values", proof.values)
    claims = [EvalClaim(torch.cat((r_i, r_c)), proof.values[0].to(dev), "w(r_i, r_c)")]
    for x, cr in enumerate(lay.c_rows):
        claims.append(EvalClaim(torch.cat((_bits_point(cr, n_i, dev), rho)), proof.values[1 + x].to(dev), f"w(c_{x}, rho)"))
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
        bd = runner.marshal(list(g), min_l=l) if g else _marshal_pad(rel, runner, l, device)
        assert bd["lay"].l == l, (bd["lay"].l, l)
        hints = rel.hints(rel.params, runner.sys, bd["c_d"], bd["a_d"], bd["b_d"], bd["y_d"], device)
        W = build_witness(runner.sys, bd["pub"], hints, device)
        subs.append(SubBatchT(W, bd["pub"], bd["ypub_d"], len(g)))
        n_vus.append(len(g))
    return subs, n_vus


def _marshal_pad(rel, runner, l: int, device) -> dict:
    """A 0-VU sub-batch: ``l`` pad units (``relchain.RelationChainRunner._pad_unit``), ``ypub = 0`` -- what ``marshal``
    would build if it accepted an empty VU list."""
    import numpy as np
    from backends.direct.ligero.chain import Layout as ChainLayout
    units = [runner._pad_unit()] * l
    a = np.asarray([u[1] for u in units], dtype=np.int64)
    b = np.asarray([u[2] for u in units], dtype=np.int64)
    c_in = np.asarray([u[0] for u in units], dtype=np.int64)
    y_out = np.asarray([u[3] for u in units], dtype=np.int64)
    to_dev = lambda x: torch.as_tensor(x).to(device)  # noqa: E731
    bd = {"lay": ChainLayout(l, 0, rel.steps), "a_d": to_dev(a), "b_d": to_dev(b), "c_d": to_dev(c_in), "y_d": to_dev(y_out),
          "ypub_d": torch.zeros(l, dtype=torch.int64, device=device)}
    bd["pub"] = rel.public_vectors(runner.p, None, bd["a_d"], bd["b_d"], None, device=device)
    return bd


def check_claims(lay: Layout, z: torch.Tensor, claims: list) -> bool:
    """Diagnostic: evaluate the committed polynomial (``z`` with the virtual rows zeroed) at every claim point."""
    w = z.clone()
    w[lay.m:] = 0
    for cl in claims:
        eq_i = eq_table(cl.point[:lay.n_i])
        eq_c = eq_table(cl.point[lay.n_i:])
        v = mle_eval_ext(matvec_base_ext(w, eq_c), eq_i)
        if not torch.equal(v, cl.value.to(v.device)):
            return False
    return True


# ---------------------------------------------------------------------------------------------------------------------
# CUDA engine: fused cupy rounds for sumcheck 1
# ---------------------------------------------------------------------------------------------------------------------
# Fused CUDA rounds for the zero-check sumcheck (``sumcheck.py``): two cupy RawKernels per round.
# 
# ``sc_msg``: for every pair ``(x', x' + half)`` of the three tables, the products ``A0 B0 - C0``, ``A1 B1 - C1``,
# ``(A1 - A0)(B1 - B0)``, the eq weight ``w = w_rows[k] * w_cols[c]`` and the three weighted accumulations (block-reduced,
# one atomic per block and coordinate) -> ``q(0), q(1), q(inf)``.  ``sc_fold``: ``T0 + r (T1 - T0)`` for the three tables,
# IN PLACE (the output occupies the first half of every plane).  The message is computed before the challenge is drawn,
# as the interactive protocol requires.
# 
# Arithmetic: Montgomery form (R = 2^32) throughout the extension tables, the weights and the challenges; the base
# tables of round 1 are read in normal form and converted on the fly; the accumulators and the final values come back
# in Montgomery form and are converted by the host (``from_mont``).  Extension elements are 6 int32 planes ``(6, R, Cc)``
# with an explicit plane stride.

R_MONT = (1 << 32) % P
R2_MONT = (1 << 64) % P
R_INV = pow(1 << 32, -1, P)
NP = (-pow(P, -1, 1 << 32)) % (1 << 32)

_SRC = r"""
#define P 2013265921u
#define P64 2013265921ULL
#define NP 2013265919u
#define R2 1172168163u
#define NR 31ULL
#define R1 268435454u
typedef unsigned long long u64;
typedef unsigned int u32;

// Montgomery product: a b R^-1 mod p for a, b < p (R = 2^32)
__device__ __forceinline__ u32 mont(u32 a, u32 b) {
    u64 t = (u64)a * (u64)b;
    u32 m = (u32)t * NP;
    u32 u = (u32)((t + (u64)m * P64) >> 32);
    return u >= P ? u - P : u;
}
__device__ __forceinline__ u32 to_mont(u32 a) { return mont(a, R2); }
__device__ __forceinline__ u32 addmod(u32 a, u32 b) { u32 s = a + b; return s >= P ? s - P : s; }
__device__ __forceinline__ u32 submod(u32 a, u32 b) { return a >= b ? a - b : a + P - b; }

// ext product in Montgomery form (x^6 = 31): 36 mont + 6 u64 reductions
__device__ __forceinline__ void ext_mul(const u32* a, const u32* b, u32* out) {
    u64 t[11];
    #pragma unroll
    for (int i = 0; i < 11; i++) t[i] = 0;
    #pragma unroll
    for (int i = 0; i < 6; i++) {
        #pragma unroll
        for (int j = 0; j < 6; j++) t[i + j] += mont(a[i], b[j]);
    }
    #pragma unroll
    for (int c = 0; c < 6; c++) {
        u64 v = t[c];
        if (c + 6 < 11) v += t[c + 6] * NR;
        out[c] = (u32)(v % P64);
    }
}

__device__ __forceinline__ void load6(const int* __restrict__ T, long long ps, long long o, u32* v) {
#if BASE_IN
    v[0] = (u32)T[o];
    #pragma unroll
    for (int p = 1; p < 6; p++) v[p] = 0;
#else
    #pragma unroll
    for (int p = 0; p < 6; p++) v[p] = (u32)T[p * ps + o];
#endif
}

extern "C" __global__ void sc_msg(
    const int* __restrict__ A, const int* __restrict__ B, const int* __restrict__ C,
    const u32* __restrict__ wr, const u32* __restrict__ wc,
    u64* acc, long long hR, long long Cc, long long ps)
{
    __shared__ u64 sh[18][256];
    const long long n = hR * Cc;
    u64 s0[6], s1[6], si[6];
    #pragma unroll
    for (int p = 0; p < 6; p++) { s0[p] = 0; s1[p] = 0; si[p] = 0; }
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (long long)gridDim.x * blockDim.x) {
        const long long k = idx / Cc, c = idx - k * Cc;
        const long long o0 = k * Cc + c, o1 = (k + hR) * Cc + c;
        u32 a0[6], a1[6], b0[6], b1[6], c0[6], c1[6];
        load6(A, ps, o0, a0); load6(A, ps, o1, a1);
        load6(B, ps, o0, b0); load6(B, ps, o1, b1);
        load6(C, ps, o0, c0); load6(C, ps, o1, c1);
        u32 wrow[6], wcol[6], w[6];
        #pragma unroll
        for (int p = 0; p < 6; p++) { wrow[p] = wr[p * hR + k]; wcol[p] = wc[p * Cc + c]; }
        ext_mul(wrow, wcol, w);
        u32 t0[6], t1[6], ti[6];
#if BASE_IN
        // normal-form base products; mont(w_m, x) = (w x) in normal form, so lift x to Montgomery form first
        const u32 p0 = to_mont(submod(mont(to_mont(a0[0]), b0[0]), c0[0]));
        const u32 p1 = to_mont(submod(mont(to_mont(a1[0]), b1[0]), c1[0]));
        const u32 pi = to_mont(mont(to_mont(submod(a1[0], a0[0])), submod(b1[0], b0[0])));
        #pragma unroll
        for (int p = 0; p < 6; p++) { t0[p] = mont(w[p], p0); t1[p] = mont(w[p], p1); ti[p] = mont(w[p], pi); }
#else
        u32 p0[6], p1[6], pi[6], da[6], db[6];
        ext_mul(a0, b0, p0); ext_mul(a1, b1, p1);
        #pragma unroll
        for (int p = 0; p < 6; p++) { p0[p] = submod(p0[p], c0[p]); p1[p] = submod(p1[p], c1[p]); da[p] = submod(a1[p], a0[p]); db[p] = submod(b1[p], b0[p]); }
        ext_mul(da, db, pi);
        ext_mul(w, p0, t0); ext_mul(w, p1, t1); ext_mul(w, pi, ti);
#endif
        #pragma unroll
        for (int p = 0; p < 6; p++) { s0[p] += t0[p]; s1[p] += t1[p]; si[p] += ti[p]; }
    }
    #pragma unroll
    for (int p = 0; p < 6; p++) { sh[p][threadIdx.x] = s0[p] % P64; sh[6 + p][threadIdx.x] = s1[p] % P64; sh[12 + p][threadIdx.x] = si[p] % P64; }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            #pragma unroll
            for (int q = 0; q < 18; q++) sh[q][threadIdx.x] += sh[q][threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x < 18) atomicAdd(&acc[threadIdx.x], sh[threadIdx.x][0] % P64);
}

extern "C" __global__ void sc_fold(
    const int* __restrict__ A, const int* __restrict__ B, const int* __restrict__ C,
    int* Ao, int* Bo, int* Co, const u32* __restrict__ r6,
    long long hR, long long Cc, long long ps_in, long long ps_out)
{
    const long long n = hR * Cc;
    u32 r[6];
    #pragma unroll
    for (int p = 0; p < 6; p++) r[p] = r6[p];
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (long long)gridDim.x * blockDim.x) {
        const long long k = idx / Cc, c = idx - k * Cc;
        const long long o0 = k * Cc + c, o1 = (k + hR) * Cc + c;
        u32 a0[6], a1[6], b0[6], b1[6], c0[6], c1[6], d[6], fa[6], fb[6], fc[6];
        load6(A, ps_in, o0, a0); load6(A, ps_in, o1, a1);
        load6(B, ps_in, o0, b0); load6(B, ps_in, o1, b1);
        load6(C, ps_in, o0, c0); load6(C, ps_in, o1, c1);
#if BASE_IN
        const u32 da = to_mont(submod(a1[0], a0[0])), db = to_mont(submod(b1[0], b0[0])), dc = to_mont(submod(c1[0], c0[0]));
        a0[0] = to_mont(a0[0]); b0[0] = to_mont(b0[0]); c0[0] = to_mont(c0[0]);
        #pragma unroll
        for (int p = 0; p < 6; p++) { fa[p] = mont(r[p], da); fb[p] = mont(r[p], db); fc[p] = mont(r[p], dc); }
#else
        #pragma unroll
        for (int p = 0; p < 6; p++) d[p] = submod(a1[p], a0[p]);
        ext_mul(r, d, fa);
        #pragma unroll
        for (int p = 0; p < 6; p++) d[p] = submod(b1[p], b0[p]);
        ext_mul(r, d, fb);
        #pragma unroll
        for (int p = 0; p < 6; p++) d[p] = submod(c1[p], c0[p]);
        ext_mul(r, d, fc);
#endif
        #pragma unroll
        for (int p = 0; p < 6; p++) {
            Ao[p * ps_out + o0] = (int)addmod(fa[p], a0[p]);
            Bo[p * ps_out + o0] = (int)addmod(fb[p], b0[p]);
            Co[p * ps_out + o0] = (int)addmod(fc[p], c0[p]);
        }
    }
}



// opening round: two top row variables at once, base input.  For each quad (k', c): corners a00 a01 a10 a11 (X1 X2), the 9
// entries M[a][b] of the mixed representation of q(X1, X2) = (A B - C)(X1, X2) weighted by w = wr[k'] wc[c].
extern "C" __global__ void sc_msg2(
    const int* __restrict__ A, const int* __restrict__ B, const int* __restrict__ C,
    const u32* __restrict__ wr, const u32* __restrict__ wc,
    u64* acc, long long hR4, long long Cc)
{
    __shared__ u64 sh[54][8];
    const long long n = hR4 * Cc;
    u64 s[54];
    #pragma unroll
    for (int q = 0; q < 54; q++) s[q] = 0;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (long long)gridDim.x * blockDim.x) {
        const long long k = idx / Cc, c = idx - k * Cc;
        const long long o00 = k * Cc + c, o01 = o00 + hR4 * Cc, o10 = o01 + hR4 * Cc, o11 = o10 + hR4 * Cc;
        const u32 a00 = (u32)A[o00], a01 = (u32)A[o01], a10 = (u32)A[o10], a11 = (u32)A[o11];
        const u32 b00 = (u32)B[o00], b01 = (u32)B[o01], b10 = (u32)B[o10], b11 = (u32)B[o11];
        const u32 c00 = (u32)C[o00], c01 = (u32)C[o01], c10 = (u32)C[o10], c11 = (u32)C[o11];
        u32 wrow[6], wcol[6], w[6];
        #pragma unroll
        for (int p = 0; p < 6; p++) { wrow[p] = wr[p * hR4 + k]; wcol[p] = wc[p * Cc + c]; }
        ext_mul(wrow, wcol, w);
        #pragma unroll
        for (int b = 0; b < 3; b++) {
            u32 u0, u1, v0, v1, w0, w1;
            if (b == 0)      { u0 = a00; u1 = submod(a10, a00); v0 = b00; v1 = submod(b10, b00); w0 = c00; w1 = submod(c10, c00); }
            else if (b == 1) { u0 = a01; u1 = submod(a11, a01); v0 = b01; v1 = submod(b11, b01); w0 = c01; w1 = submod(c11, c01); }
            else             { u0 = submod(a01, a00); u1 = submod(submod(a11, a10), submod(a01, a00));
                               v0 = submod(b01, b00); v1 = submod(submod(b11, b10), submod(b01, b00)); w0 = 0; w1 = 0; }
            const u32 f0 = to_mont(submod(mont(to_mont(u0), v0), w0));
            const u32 f1 = to_mont(submod(mont(to_mont(addmod(u0, u1)), addmod(v0, v1)), addmod(w0, w1)));
            const u32 f2 = to_mont(mont(to_mont(u1), v1));
            #pragma unroll
            for (int p = 0; p < 6; p++) {
                s[(0 * 3 + b) * 6 + p] += mont(w[p], f0);
                s[(1 * 3 + b) * 6 + p] += mont(w[p], f1);
                s[(2 * 3 + b) * 6 + p] += mont(w[p], f2);
            }
        }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    #pragma unroll
    for (int q = 0; q < 54; q++) {
        u64 v = s[q] % P64;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) sh[q][warp] = v % P64;
    }
    __syncthreads();
    if (threadIdx.x < 54) {
        u64 t = 0;
        for (int wv = 0; wv < 8; wv++) t += sh[threadIdx.x][wv];
        atomicAdd(&acc[threadIdx.x], t % P64);
    }
}

// opening fold: base quads -> one ext cell: T(r1, r2) = al0(r1) + r2 al1(r1), al0 = t00 + r1 (t10 - t00), al1 = (t01 - t00) + r1 (t11 - t10 - t01 + t00)
extern "C" __global__ void sc_fold2(
    const int* __restrict__ A, const int* __restrict__ B, const int* __restrict__ C,
    int* Ao, int* Bo, int* Co, const u32* __restrict__ r1m, const u32* __restrict__ r2m,
    long long hR4, long long Cc)
{
    const long long n = hR4 * Cc;
    u32 r1[6], r2[6];
    #pragma unroll
    for (int p = 0; p < 6; p++) { r1[p] = r1m[p]; r2[p] = r2m[p]; }
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (long long)gridDim.x * blockDim.x) {
        const long long k = idx / Cc, c = idx - k * Cc;
        const long long o00 = k * Cc + c, o01 = o00 + hR4 * Cc, o10 = o01 + hR4 * Cc, o11 = o10 + hR4 * Cc;
        const int* Ts[3] = {A, B, C};
        int* To[3] = {Ao, Bo, Co};
        #pragma unroll
        for (int s = 0; s < 3; s++) {
            const u32 t00 = (u32)Ts[s][o00], t01 = (u32)Ts[s][o01], t10 = (u32)Ts[s][o10], t11 = (u32)Ts[s][o11];
            const u32 da = to_mont(submod(t10, t00)), d1 = to_mont(submod(t01, t00)), dd = to_mont(submod(submod(t11, t10), submod(t01, t00)));
            u32 al0[6], al1[6], f[6];
            #pragma unroll
            for (int p = 0; p < 6; p++) { al0[p] = mont(r1[p], da); al1[p] = mont(r1[p], dd); }
            al0[0] = addmod(al0[0], to_mont(t00));
            al1[0] = addmod(al1[0], d1);
            ext_mul(al1, r2, f);
            #pragma unroll
            for (int p = 0; p < 6; p++) To[s][p * n + o00] = (int)addmod(f[p], al0[p]);
        }
    }
}

// eq table: out[x] = prod_t (x_t ? tau[t] : 1 - tau[t]) for x < 2^n; tau planes (6, n) and (1 - tau) planes (6, n) in Montgomery form; out (x, 6) normal int64
extern "C" __global__ void sc_eq(const u32* __restrict__ tm, const u32* __restrict__ om, long long* __restrict__ out, int n)
{
    const long long N = 1LL << n;
    for (long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x; x < N; x += (long long)gridDim.x * blockDim.x) {
        u32 acc[6], f[6], tmp[6];
        acc[0] = R1; 
        #pragma unroll
        for (int p = 1; p < 6; p++) acc[p] = 0;
        for (int t = 0; t < n; t++) {
            const u32* src = ((x >> t) & 1) ? tm : om;
            #pragma unroll
            for (int p = 0; p < 6; p++) f[p] = src[p * n + t];
            ext_mul(acc, f, tmp);
            #pragma unroll
            for (int p = 0; p < 6; p++) acc[p] = tmp[p];
        }
        #pragma unroll
        for (int p = 0; p < 6; p++) out[x * 6 + p] = (long long)mont(acc[p], 1u);      // mont(a_m, 1) = a
    }
}
// base x ext matvec: out[i] = sum_c M[i, c] * e[c]   (e in Montgomery planes (6, C); out normal form, int64 (R, 6)); one block per row
extern "C" __global__ void sc_matvec(
    const int* __restrict__ M, const u32* __restrict__ e, long long* __restrict__ out, long long C)
{
    __shared__ u64 sh[6][256];
    const long long i = blockIdx.x;
    u64 acc[6];
    #pragma unroll
    for (int p = 0; p < 6; p++) acc[p] = 0;
    for (long long c = threadIdx.x; c < C; c += blockDim.x) {
        const u32 m = (u32)M[i * C + c];
        #pragma unroll
        for (int p = 0; p < 6; p++) acc[p] += mont(e[p * C + c], m);
    }
    #pragma unroll
    for (int p = 0; p < 6; p++) sh[p][threadIdx.x] = acc[p] % P64;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            #pragma unroll
            for (int p = 0; p < 6; p++) sh[p][threadIdx.x] += sh[p][threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x < 6) out[i * 6 + threadIdx.x] = (long long)(sh[threadIdx.x][0] % P64);
}
// sparse gather: T[k, c] = sum_{e in [ptr[k], ptr[k+1])} val[e] * z[row[e], c]  mod p   (CSR over the constraint slots)
extern "C" __global__ void sc_gather(
    const int* __restrict__ z, const int* __restrict__ ptr, const int* __restrict__ row, const int* __restrict__ val,
    int* __restrict__ T, long long K, long long C)
{
    const long long k = blockIdx.y;                                   // grid (ceil(C / 256), K)
    const int e0 = ptr[k], e1 = ptr[k + 1];
    for (long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x; c < C; c += (long long)gridDim.x * blockDim.x) {
        u64 acc = 0;
        for (int e = e0; e < e1; e++) acc += ((u64)(u32)val[e] * (u64)(u32)z[(long long)row[e] * C + c]) % P64;
        T[k * C + c] = (int)(acc % P64);
    }
}
"""

_KERNELS: dict = {}


def _kernel(name: str, base_in: bool):
    import cupy as cp
    key = (name, bool(base_in))
    if key not in _KERNELS:
        src = f"#define BASE_IN {1 if base_in else 0}\n" + _SRC
        _KERNELS[key] = cp.RawKernel(src, name, options=("-std=c++17",))
    return _KERNELS[key]


def _cp(t: torch.Tensor):
    import cupy as cp
    return cp.from_dlpack(t.contiguous())


def build_tables_cuda(cons: Constraints, z: torch.Tensor) -> dict[str, torch.Tensor]:
    """``build_tables`` as one fused CSR gather kernel per side (int32 output; ~12.9 GB written at 2^30)."""
    import cupy as cp
    dev = z.device
    z = z.contiguous()
    K, C = cons.K_pad, z.shape[1]
    out = {}
    kern = _kernel("sc_gather", False)
    for side, (ks, rs, vs) in cons.coo.items():
        order = torch.argsort(ks.to(dev))
        ks_s, rs_s, vs_s = ks.to(dev)[order], rs.to(dev)[order], vs.to(dev)[order]
        ptr = torch.zeros((K + 1,), dtype=torch.int64, device=dev)
        ptr[1:] = torch.cumsum(torch.bincount(ks_s, minlength=K), 0)
        T = torch.empty((K, C), dtype=torch.int32, device=dev)
        kern((int(min(1024, (C + 255) // 256)), K), (256,), (_cp(z), _cp(ptr.to(torch.int32)), _cp(rs_s.to(torch.int32)),
                                                            _cp((vs_s % P).to(torch.int32)), _cp(T), cp.int64(K), cp.int64(C)))
        out[side] = T
    return out


def eq_table_cuda(tau: torch.Tensor) -> torch.Tensor:
    """``eq_table`` as one kernel (``(2^n, 6)`` int64, normal form)."""
    import cupy as cp
    n = tau.shape[0]
    dev = tau.device
    one = fc_one((), dev)
    tm = mont_planes(tau % P)
    om = mont_planes((one - tau) % P)
    out = torch.empty((1 << n, D), dtype=torch.int64, device=dev)
    kern = _kernel("sc_eq", False)
    blocks = int(min(4096, ((1 << n) + 255) // 256))
    kern((blocks,), (256,), (_cp(tm), _cp(om), _cp(out), cp.int32(n)))
    return out


def matvec_base_ext_cuda(M: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """``matvec_base_ext`` as one kernel: ``(R, C) int32 base x (C, 6) ext -> (R, 6)``."""
    M = M.contiguous()
    if M.dtype != torch.int32:
        M = (M.to(torch.int64) % P).to(torch.int32)
    R, C = M.shape
    out = torch.empty((R, D), dtype=torch.int64, device=M.device)
    kern = _kernel("sc_matvec", False)
    import cupy as cp
    kern((R,), (256,), (_cp(M), _cp(mont_planes(e)), _cp(out), cp.int64(C)))
    return out


def to_mont(t: torch.Tensor) -> torch.Tensor:
    return (t.to(torch.int64) % P * R_MONT) % P


def from_mont(t: torch.Tensor) -> torch.Tensor:
    return (t.to(torch.int64) % P * R_INV) % P


def mont_planes(tab: torch.Tensor) -> torch.Tensor:
    """``(n, 6)`` int64 ext table -> ``(6, n)`` int32 planes in Montgomery form."""
    return to_mont(tab).t().contiguous().to(torch.int32)


def eq_prefix_planes(tau: torch.Tensor) -> list[torch.Tensor]:
    """``eq_table(tau[:t])`` for ``t = 0..n`` as Montgomery planes ``(6, 2^t)`` (built incrementally)."""
    vec = fc_one((1,), tau.device)
    out = [mont_planes(vec)]
    for t in range(tau.shape[0]):
        one_minus = (fc_one((), tau.device) - tau[t]) % P
        vec = torch.cat((fc_mul(vec, one_minus), fc_mul(vec, tau[t])), 0)
        out.append(mont_planes(vec))
    return out


class CudaState:
    """Sumcheck-1 state on the device: three tables as int32 base ``(R, Cc)`` (round 1) or ext planes; the same
    ``round()`` / ``fold()`` / ``final()`` protocol as ``sumcheck._RefState``.  Rows phase: the top row bit is folded
    while ``R > 1``; then the ``(6, 1, Cc)`` planes are read as ``(6, Cc, 1)`` and the column bits are folded the same way."""

    THREADS = 256
    MAX_BLOCKS = 4096

    def __init__(self, tabs: dict, tau_rows: torch.Tensor, tau_cols: torch.Tensor):
        dev = tabs["a"].device
        self.dev = dev
        self.R, self.Cc = tabs["a"].shape
        self.base = True
        self.T = {s: tabs[s].to(torch.int32).contiguous() for s in "abc"}      # (R, Cc) base, normal form
        self.ps = self.R * self.Cc                                              # plane stride once ext
        self.phase = "rows"
        self.wr_rows = eq_prefix_planes(tau_rows)                              # [t] = (6, 2^t)
        self.wr_cols = eq_prefix_planes(tau_cols)
        self.wc_full = self.wr_cols[-1]                                         # (6, Cc)
        self.one_col = torch.zeros((D, 1), dtype=torch.int32, device=dev)
        self.one_col[0, 0] = R_MONT

    def _weights(self, hR: int):
        if self.phase == "rows":
            return self.wr_rows[hR.bit_length() - 1], self.wc_full
        return self.wr_cols[hR.bit_length() - 1], self.one_col

    def _launch(self, n: int):
        blocks = int(min(self.MAX_BLOCKS, max(1, (n + self.THREADS - 1) // self.THREADS)))
        return (blocks,), (self.THREADS,)

    def round(self) -> torch.Tensor:
        import cupy as cp
        if self.phase == "rows" and self.R == 1:
            self.phase = "cols"
            self.R, self.Cc = self.Cc, 1
        hR = self.R // 2
        wr, wc = self._weights(hR)
        acc = torch.zeros((18,), dtype=torch.int64, device=self.dev)
        kern = _kernel("sc_msg", self.base)
        n = hR * self.Cc
        grid, block = self._launch(n)
        kern(grid, block, tuple(_cp(self.T[s]) for s in "abc") + (_cp(wr), _cp(wc), _cp(acc), cp.int64(hR), cp.int64(self.Cc), cp.int64(self.ps)))
        return from_mont(acc.reshape(3, D))

    def fold(self, r: torch.Tensor) -> None:
        import cupy as cp
        hR = self.R // 2
        r6 = to_mont(r).to(torch.int32).contiguous()
        kern = _kernel("sc_fold", self.base)
        if self.base:
            out = {s: torch.empty((D, hR, self.Cc), dtype=torch.int32, device=self.dev) for s in "abc"}
            ps_out = hR * self.Cc
            args = tuple(_cp(self.T[s]) for s in "abc") + tuple(_cp(out[s]) for s in "abc")
        else:
            out, ps_out = self.T, self.ps
            args = tuple(_cp(self.T[s]) for s in "abc") * 2
        n = hR * self.Cc
        grid, block = self._launch(n)
        kern(grid, block, args + (_cp(r6), cp.int64(hR), cp.int64(self.Cc), cp.int64(self.ps), cp.int64(ps_out)))
        if self.base:
            self.T, self.ps, self.base = out, ps_out, False
        self.R = hR

    @classmethod
    def from_ext(cls, U: torch.Tensor, V: torch.Tensor) -> "CudaState":
        """A degree-2 sumcheck ``sum_x U(x) V(x)`` over ext tables ``(2^n, 6)`` with uniform weights (``C = 0``): the same
        kernels, column phase from the start."""
        self = cls.__new__(cls)
        dev = U.device
        n = U.shape[0]
        self.dev, self.R, self.Cc, self.base, self.phase, self.ps = dev, n, 1, False, "cols", n
        self.T = {"a": mont_planes(U).reshape(D, n, 1), "b": mont_planes(V).reshape(D, n, 1),
                  "c": torch.zeros((D, n, 1), dtype=torch.int32, device=dev)}
        ones = torch.zeros((D, n), dtype=torch.int32, device=dev)
        ones[0] = R_MONT
        self.wr_cols = [ones[:, :1 << t].contiguous() for t in range(n.bit_length())]
        self.one_col = ones[:, :1].contiguous()
        return self

    def round2(self) -> torch.Tensor:
        import cupy as cp
        assert self.base and self.phase == "rows" and self.R >= 4
        hR4 = self.R // 4
        wr = self.wr_rows[hR4.bit_length() - 1]
        acc = torch.zeros((54,), dtype=torch.int64, device=self.dev)
        kern = _kernel("sc_msg2", True)
        grid, block = self._launch(hR4 * self.Cc)
        kern(grid, block, tuple(_cp(self.T[s]) for s in "abc") + (_cp(wr), _cp(self.wc_full), _cp(acc), cp.int64(hR4), cp.int64(self.Cc)))
        return from_mont(acc.reshape(3, 3, D))

    def fold2(self, r1: torch.Tensor, r2: torch.Tensor) -> None:
        import cupy as cp
        hR4 = self.R // 4
        r1m, r2m = (to_mont(r).to(torch.int32).contiguous() for r in (r1, r2))
        out = {s: torch.empty((D, hR4, self.Cc), dtype=torch.int32, device=self.dev) for s in "abc"}
        kern = _kernel("sc_fold2", True)
        grid, block = self._launch(hR4 * self.Cc)
        kern(grid, block, tuple(_cp(self.T[s]) for s in "abc") + tuple(_cp(out[s]) for s in "abc") + (_cp(r1m), _cp(r2m), cp.int64(hR4), cp.int64(self.Cc)))
        self.T, self.ps, self.base, self.R = out, hR4 * self.Cc, False, hR4

    def final(self) -> torch.Tensor:
        out = []
        for s in "abc":
            flat = self.T[s].reshape(-1)
            out.append(from_mont(torch.stack([flat[p * self.ps] for p in range(D)])))
        return torch.stack(out)


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
    proof = prove(lay, cons, z, LocalCoins(b"bench"), engine=engine)
    out["timings_cold"] = proof.timings                      # first prove in the process: includes the cupy import / kernel load
    proof = prove(lay, cons, z, LocalCoins(b"bench"), engine=engine)
    if dev.type == "cuda":
        out["peak_mem_gb"] = (torch.cuda.max_memory_allocated(dev)) / 1e9
        out["base_mem_gb"] = base_mem / 1e9
    out["timings"] = proof.timings                           # steady state
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


class RecordingCoins:
    """Wraps a coins object and records every draw ``(label, values)`` and absorb ``(label, hex)`` in order (fixtures)."""

    def __init__(self, inner):
        self.inner, self.log = inner, []

    def absorb(self, label: bytes, data: bytes) -> None:
        self.log.append({"absorb": label.decode(), "bytes": data.hex()})
        self.inner.absorb(label, data)

    def challenge(self, label: bytes, n: int):
        v = self.inner.challenge(label, n)
        self.log.append({"challenge": label.decode(), "n": n, "values": [[int(x) for x in row] for row in v]})
        return v


def fixture(relation_name: str = "fp8-ada", l: int = 64, S: int = 2, n_vus: int = 2, seed: int = 7, device: str = "cpu") -> dict:
    """A complete verifier fixture for ligerito-verify-rs: layout, constraint COO triples, public rows, proof bytes (hex), the
    claims, and the transcript (every absorb and every challenge, in order) of an honest run with ``LocalCoins(b"fixture")``."""
    from backends.direct.ligero.relations import relation
    from backends.direct.ligero.relchain import RelationChainRunner, instances
    from .layout import build_z, constraints, layout_for, public_rows
    rel = relation(relation_name)
    runner = RelationChainRunner(rel, device, -128.0)
    vus = instances(rel, n_vus, seed=seed)
    subs, nv = ligero_subbatches(rel, runner, vus, l, S, device)
    lay = layout_for(runner.sys, l, nv, rel.steps)
    cons = constraints(runner.sys, lay, device)
    z = build_z(runner.sys, lay, subs, device)
    pubs = public_rows(runner.sys, lay, [s.public() for s in subs], device)
    coins = RecordingCoins(LocalCoins(b"fixture"))
    proof = prove(lay, cons, z, coins)
    ok, why, claims = verify(lay, cons, pubs, proof, LocalCoins(b"fixture"), device=device)
    assert ok, why
    return {
        "relation": relation_name, "l": l, "S": S, "n_vus": nv, "seed": seed, "coins": "LocalCoins(b'fixture')",
        "layout": {**lay.summary(), "m": lay.m, "virt": lay.virt, "c_rows": list(lay.c_rows), "n_j": lay.n_j, "n_s": lay.n_s,
                   "row_of_sys": {int(k): int(v) for k, v in lay.row_of_sys.items()}},
        "constraints": {"K_pad": cons.K_pad, "names": cons.names,
                        **{side: [[int(a) for a in t.tolist()] for t in cons.coo[side]] for side in "abc"}},
        "public_rows": {name: [int(x) for x in v.tolist()] for name, v in pubs.items()},
        "proof_hex": proof.to_bytes().hex(),
        "claims": [{"name": c.name, "point": [[int(x) for x in row] for row in c.point.tolist()], "value": [int(x) for x in c.value.tolist()]}
                   for c in claims],
        "transcript": coins.log,
    }


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
    ap.add_argument("--fixture", default=None, help="write the toy-size verifier fixture (fp8-ada, l = 64, S = 2, CPU) to this path and exit")
    a = ap.parse_args()
    if a.fixture:
        with open(a.fixture, "w") as f:
            json.dump(fixture(), f)
        return
    res = bench(a.relation, a.vus, a.l, a.S, a.device, a.engine, a.negatives)
    s = json.dumps(res, indent=1, default=str)
    print(s)
    if a.json:
        with open(a.json, "w") as f:
            f.write(s)


if __name__ == "__main__":
    _main()
