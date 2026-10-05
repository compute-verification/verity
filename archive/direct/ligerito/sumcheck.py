"""The constraint layer of the Ligerito backend (lanes ligerito-sumcheck, ligerito-sumcheck-2): a batched degree-3 zero-check
over every constraint of a ``compile.System`` on the multilinear layout of ``layout.py``, then ONE combined degree-2 sumcheck
(rows + chain shift, batched), ending in evaluation claims on the committed polynomial.  Torch reference prover (any
device, small sizes) + fused cupy CUDA prover (``engine="cuda"``) + Python verifier.  Wire format ``LGSC0003``.

All randomness through ``Coins.challenge(label, n)`` (``transcript.py``).  A round of arity ``v`` binds the ``v`` top
remaining variables with ONE coin; its message is the round polynomial in the mixed representation ``M[a_1]..[a_v]``
(``a_j`` in ``{0, 1, inf}`` for variable ``X_j``, ``X_1`` = the top remaining variable; ``inf`` = the ``X_j^2``
coefficient), ``3^v`` extension elements; ``q(r) = contract(M, (1 - r_j, r_j, r_j (r_j - 1)))``.

1. **zero-check** over ``(k, c)`` (``n_k + n_c`` variables; ``k`` = constraint slot = the HIGH bits, ``c = s * l + j``):
       sum_{k,c} eq(tau, (k, c)) * (Az * Bz - Cz)(k, c) = 0,     Az = A z  etc.
   Gruen: round ``t`` sends ``q_t`` with ``s_t(X) = prod_j eq(tau_j, X_j) q_t(X)`` (``q_t`` degree 2 per variable); the verifier
   checks ``E * sum_{b in {0,1}^v} prod_j eq(tau_j, b_j) q_t(b) = claim`` and continues with ``E <- E prod_j eq(tau_j, r_j)``,
   ``claim <- E q_t(r)``.  The OPENING round binds the top 3 ``k`` bits straight from ``z`` (no ``Az/Bz/Cz`` base tables);
   the FINAL round sends the three tables restricted to the last ``v_f`` variables (``2^v_f`` values each) and the shifted
   rows' tables ``next_x(r_c_hi, .)`` instead of a round polynomial; its one coin gives ``r_f`` AND the batching
   coefficients ``gamma`` (A, B, C), ``g3`` (links), ``beta`` (rows vs shift).
2. **combined** over ``y`` (``n_cmb = max(n_i, n_c)`` variables, degree 2), ``h(y) = beta coef(y_lo) z~(y_lo, r_c) + succ~(r_c, y) V(y)``:
       sum_y h = beta 2^(n_cmb - n_i) (Az + g Bz + g^2 Cz)(r) + 2^(n_cmb - n_c) sum_x g3^x next_x(r_c)
   with ``coef = eq(r_k, .)^T (A + g B + g^2 C)`` (rows on the LOW ``n_i`` variables of ``y``), ``V(c) = sum_x g3^x W(c_x, c)``,
   ``succ~(r, c) = eq(r, pred(c))`` (cyclic per sub-batch).  Ends at ``rho``: ``r_i = rho[:n_i]``, ``rho_c = rho[:n_c]``;
   ``z~(r_i, r_c) = w~(r_i, r_c) + sum_v eq(r_i, idx_v) virt_v(r_c)`` (public rows + the ``next_vals``), and the
   ``n_links`` claims ``w~(bits(c_x), rho_c)``; the verifier evaluates ``succ~(rho_c, r_c)`` in closed form (``succ_mle``).

Soundness of the packing: a ``v``-variable round's difference polynomial has total degree ``<= 2v`` (``q``; zero-check) /
``2v`` (combined), so its Schwartz-Zippel error is ``2v / |F|``; the sum over rounds is ``2 n / |F|`` whatever the packing.
"""
from __future__ import annotations

import hashlib
import struct
import time
from dataclasses import dataclass, field

import torch

from archive.direct.ligero.compile import P
from archive.direct.ligero.v2.fc import D, fc_inv, fc_mul, fc_one, fc_pow_seq, fc_scalar_mul
from .layout import Constraints, Layout
from .layout import SubBatch as SubBatchT

try:  # lane ligerito-proto's Coins protocol; the stub below has the same signature
    from .transcript import Coins  # type: ignore  # noqa: F401
except Exception:  # noqa: BLE001
    Coins = object  # type: ignore

__all__ = ["EvalClaim", "LocalCoins", "Schedule", "SparseClaim", "SumcheckProof", "check_zk_claim_supports", "default_schedule",
           "eq_at", "eq_table", "prove", "succ_mle", "verify", "zk_libra_cells", "zk_mode", "zk_vf_max"]

MAGIC = b"LGSC0003"
MAGIC_ZK = b"LGSC0004"
MAX_ARITY = 4
ZK_MAX_ARITY = 6                 # LGSC0004 message rounds (3^6 = 729 ext); the engines run them on small tables only
ZK_MAX_VF = 12                   # LGSC0004 final table round (3 x 2^vf ext)
ZK_TABLE_SLACK = 4               # the prover's vf <= n_c - 4 (``zk_vf_max``); the verifier accepts any vf <= n_c


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
        self.absorb_bytes(label, _ext_bytes(t))

    def absorb_bytes(self, label: str, b: bytes) -> None:
        absorb = getattr(self.coins, "absorb", None)
        if absorb is not None:
            absorb(label.encode(), b)

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
    """``(2^n, 6)``: ``eq(tau, x)`` for every ``x``; variable ``t`` = bit ``t`` of ``x``."""
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
# Mixed representation of a round polynomial (degree <= 2 per variable)
# ---------------------------------------------------------------------------------------------------------------------

def to_corners(T: torch.Tensor, v: int) -> torch.Tensor:
    """A flat table ``(n, 6)`` -> ``(2,)*v + (n >> v, 6)``: axis ``j`` is the value of ``X_{j+1}`` (``X_1`` = the top bit)."""
    return T.reshape((2,) * v + (T.shape[0] >> v, D))


def mixed_rep(Tc: torch.Tensor, v: int) -> torch.Tensor:
    """Corners ``(2,)*v + rest`` of a multilinear table -> ``(3,)*v + rest``: per axis ``(t0, t1) -> (t0, t1, t1 - t0)``.
    The mixed representation of a product of two multilinear tables is the elementwise product of theirs."""
    for a in range(v):
        t0, t1 = Tc.select(a, 0), Tc.select(a, 1)
        Tc = torch.stack((t0, t1, (t1 - t0) % P), dim=a)
    return Tc


def noinf_mask(v: int, device) -> torch.Tensor:
    """``(3,)*v`` int64: 1 where no index is ``inf`` (the multilinear ``Cz`` has no ``X^2`` term)."""
    m = torch.ones((3,) * v, dtype=torch.int64, device=device)
    for a in range(v):
        idx = [slice(None)] * v
        idx[a] = 2
        m[tuple(idx)] = 0
    return m


def _axis_weights(r: torch.Tensor, kind: str) -> torch.Tensor:
    one = fc_one((), r.device)
    zero = torch.zeros_like(one)
    if kind == "eval":                                  # q(r) = (1 - r) q(0) + r q(1) + r (r - 1) q_inf
        return torch.stack(((one - r) % P, r % P, fc_mul(r, (r - one) % P)))
    if kind == "eq":                                    # sum_b eq(tau, b) q(b)
        return torch.stack(((one - r) % P, r % P, zero))
    return torch.stack((one, one, zero))                # sum_b q(b)


def contract(M: torch.Tensor, ws: list) -> torch.Tensor:
    """``M (3,)*v + (6,)`` contracted axis by axis with ``ws[j] (3, 6)`` -> ``(6,)``."""
    for w in ws:
        M = fc_mul(M, w.reshape((3,) + (1,) * (M.dim() - 2) + (D,))).sum(0) % P
    return M


def eval_mixed(M: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    return contract(M, [_axis_weights(r[j], "eval") for j in range(r.shape[0])])


def eq_sum_mixed(M: torch.Tensor, taus: list) -> torch.Tensor:
    return contract(M, [_axis_weights(t, "eq") for t in taus])


def bool_sum_mixed(M: torch.Tensor) -> torch.Tensor:
    v = M.dim() - 1
    one = fc_one((), M.device)
    return contract(M, [_axis_weights(one, "sum")] * v)


def fold_top(T: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Bind the top ``v = len(r)`` variables of a flat ext table ``(n, 6)``: ``r[0]`` <-> ``X_1`` (the top bit)."""
    Tc = to_corners(T, r.shape[0])
    for j in range(r.shape[0]):
        Tc = (Tc[0] + fc_mul((Tc[1] - Tc[0]) % P, r[j])) % P
    return Tc


# ---------------------------------------------------------------------------------------------------------------------
# Round schedule + messages
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Schedule:
    zc: tuple                    # arity of every zero-check message round (the first = the opening round, 3 for the CUDA engine)
    vf: int                      # variables bound by the zero-check's final (table) round
    cmb: tuple                   # arity of every combined-sumcheck round
    rb: tuple = ()               # LGSC0004 only: arity of every row-reduction round (n_i variables)

    def coins(self) -> int:
        return 1 + len(self.zc) + 1 + len(self.cmb) + len(self.rb)

    def header(self, n_links: int, n_values: int) -> bytes:
        return MAGIC + bytes([len(self.zc), *self.zc, self.vf, n_links, len(self.cmb), *self.cmb, n_values])

    def header_zk(self, n_links: int) -> bytes:
        return MAGIC_ZK + bytes([len(self.zc), *self.zc, self.vf, n_links, len(self.cmb), *self.cmb, len(self.rb), *self.rb])

    def zk_groups(self) -> tuple:
        """The Libra mask groups ``(sumcheck, arity)`` in g-block order: zero-check message rounds (the final table round has
        none), combined rounds, row-reduction rounds."""
        return tuple(("zc", v) for v in self.zc) + tuple(("cmb", v) for v in self.cmb) + tuple(("rb", v) for v in self.rb)

    def zk_coeffs(self) -> int:
        return sum(3 ** v - 1 for _, v in self.zk_groups())


def n_cmb_of(lay: Layout) -> int:
    return max(lay.n_i, lay.n_c)


def default_schedule(n_zc: int, n_cmb: int, big_log2: int = 20, big_arity: int = 2, small_arity: int = 3,
                     max_final: int = 4, opening: int = 3) -> Schedule:
    """Opening round of ``opening`` variables; ``big_arity`` per coin while the remaining table has ``> 2^big_log2`` cells
    (the O(N) passes: bivariate keeps the kernels' accumulators in registers); then ``small_arity`` per coin; the final
    table round takes the rest (``2 .. max_final`` variables).  Combined sumcheck: ``small_arity`` per coin."""
    zc = [opening]
    rem = n_zc - opening
    while rem > big_log2 and rem - big_arity > max_final:
        zc.append(big_arity)
        rem -= big_arity
    a = max(0, -(-(rem - max_final) // small_arity))
    zc += [small_arity] * a
    vf = rem - small_arity * a
    if not 1 <= vf <= max_final:
        raise ValueError(f"schedule: final arity {vf}")
    cmb = [small_arity] * (n_cmb // small_arity) + ([n_cmb % small_arity] if n_cmb % small_arity else [])
    return Schedule(tuple(zc), vf, tuple(cmb))


def balanced_arities(n: int, max_arity: int) -> tuple:
    g = -(-n // max_arity)
    return tuple(n // g + (1 if t < n % g else 0) for t in range(g))


def _tail_arities(log_table: int, n_vars: int, small_log2: int, max_arity: int = ZK_MAX_ARITY) -> list:
    """Bind ``n_vars`` variables of a table of ``2^log_table`` cells: arity 3 (``zc_msg3``) while the table has more than
    ``2^small_log2`` cells, then balanced rounds of at most ``max_arity`` (``zc_msgv`` / ``zc_red`` / ``pp_red``: 3^v entries
    per group, so only on tables that are small by then)."""
    out = []
    while n_vars > 0 and log_table > small_log2:
        a = min(3, n_vars)
        out.append(a)
        n_vars, log_table = n_vars - a, log_table - a
    return out + list(balanced_arities(n_vars, max_arity) if n_vars else ())


def lean_schedule(n_zc: int, n_cmb: int) -> Schedule:
    """LGSC0003 with fewer coins (``--schedule lean``; the default is unchanged): arity 3 (``zc_msg3``) down to 2^16 cells,
    then arity <= 4, a final table round of 4 variables; combined: arity 3 down to 2^12 cells,
    then <= 4.  fp8-ada: zc 3,3,3,3,3,4,4,3 / vf 4 / cmb 3,3,4,4,4 = 15 coins (default 18)."""
    vf = min(4, n_zc - 4)
    zc = [3] + _tail_arities(n_zc - 3, n_zc - 3 - vf, 16, MAX_ARITY)
    return Schedule(tuple(zc), vf, tuple(_tail_arities(n_cmb, n_cmb, 12, MAX_ARITY)))


def zk_default_schedule(lay: Layout, vf: int = 12) -> Schedule:
    """LGSC0004's default (coins first: one round trip ~ 0.1 s live, against ~10 ms of prover time and ~230 KB for the
    11th coin at 4096 VUs): the opening round, arity 3 (``zc_msg3``) down to 2^24 cells, then arity <= 6 (``zc_msgv``); a
    final table round of ``vf`` variables (``<= n_c - 4``: every final-table entry keeps >= 16 product-row cells); combined:
    arity 3 down to 2^18 cells, then <= 6; row reduction <= 6.  fp8-ada: zc 3,3,6,6 / vf 12 / cmb 6,6,6 / rb 6,6 = 11 coins."""
    n, n_cmb = lay.n_k + lay.n_c, n_cmb_of(lay)
    zc, rem = [3], n - 3
    vf = max(1, min(vf, zk_vf_max(lay), rem - 1, ZK_MAX_VF))
    zc += _tail_arities(rem, rem - vf, 24)
    return Schedule(tuple(zc), vf, tuple(_tail_arities(n_cmb, n_cmb, 18)), balanced_arities(lay.n_i, ZK_MAX_ARITY))


def zk_vf_max(lay: Layout) -> int:
    """The largest final-table round that keeps LGSC0004's tables blinded: entry ``b`` of ``Az, Bz, Cz`` is masked by
    functionals of ``i1`` / ``i2`` over the ``2^(n_c - vf)`` product-row cells whose low ``vf`` column bits are ``b``; ``Az`` and
    ``Cz`` need 12 independent base functionals of ``i1`` there, so ``2^(n_c - vf) >= 16``."""
    return lay.n_c - ZK_TABLE_SLACK


def zk_mode(lay: Layout, sched: Schedule) -> str:
    """``"lgsc0004"``, or ``"lgsc0004-underblinded"`` if ``vf > n_c - 4`` (ligerito-verify's ``zk_mode`` strings): a proof of
    the latter kind verifies but its final tables are not fully blinded (red-team-ligerito-3 R3-6)."""
    return "lgsc0004" if sched.vf <= zk_vf_max(lay) else "lgsc0004-underblinded"


def zk_schedule(lay: Layout, base: Schedule | None = None, rb_arity: int = ZK_MAX_ARITY) -> Schedule:
    """LGSC0004's schedule: ``zk_default_schedule`` if ``base`` is None, else ``base`` (an LGSC0003 schedule) plus the
    row-reduction rounds (``n_i`` variables, balanced, ``rb_arity`` per coin: tables of ``2^n_i`` cells)."""
    if base is None:
        return zk_default_schedule(lay)
    if base.rb:
        return base
    return Schedule(base.zc, base.vf, base.cmb, balanced_arities(lay.n_i, rb_arity))


@dataclass
class EvalClaim:
    """``w~(point) = value`` on the committed polynomial; ``point[t]`` is the coordinate of variable ``t`` = bit ``t`` of the
    flat PCS index ``x = i + R * c`` (ligerito-proto's column-major layout): variables ``0 .. n_i - 1`` are the row ``i``,
    ``n_i .. n_vars - 1`` the column ``c = s * l + j``."""
    point: torch.Tensor          # (n_vars, 6)
    value: torch.Tensor          # (6,)
    name: str = ""


@dataclass
class SparseClaim:
    """``sum_t weights[t] * w[rows[t], cols[t]] = value`` on base cells of the committed polynomial (extension weights;
    distinct cells): the Libra mask claims of LGSC0004, all on ``lay.zk.g_rows``.  The PCS index of cell ``(i, c)`` is the
    caller's (row-major ``i + R c`` or ``zk.f_index``); ligerito-proto's ``open(..., sparse=[(cells, weights, value)])``."""
    rows: torch.Tensor           # (T,) int64
    cols: torch.Tensor           # (T,) int64
    weights: torch.Tensor        # (T, 6)
    value: torch.Tensor          # (6,)
    name: str = ""


@dataclass
class SumcheckProof:
    schedule: Schedule
    zc_rounds: list              # per zero-check round: (3,)*v + (6,)
    final_abc: torch.Tensor      # (3, 2^vf, 6): Az, Bz, Cz on the last vf variables (bit t of the index <-> variable t)
    final_next: torch.Tensor     # (n_links, 2^vf, 6): next_x(r_c_hi, .) on the same variables (LGSC0004: (0, 2^vf, 6), unsent)
    cmb_rounds: list             # per combined round: (3,)*v + (6,)
    values: torch.Tensor         # LGSC0003 (1 + n_links, 6): w~(r_i, r_c), then w~(bits(c_x), rho_c)
                                 # LGSC0004 (3, 6): w~(r_i, r_c), V'(rho_c), the combined mask's claim
    claims: list = field(default_factory=list)    # EvalClaim / SparseClaim (points derived from the coins; not serialised)
    timings: dict = field(default_factory=dict)
    zk_t: torch.Tensor | None = None             # LGSC0004 (6,): the zero-check mask's claim (sent with the final tables)
    rb_rounds: list = field(default_factory=list)  # LGSC0004: per row-reduction round (3,)*v + (6,)
    values_b: torch.Tensor | None = None         # LGSC0004 (2, 6): w~(rho_i, rho_c), the row-reduction mask's claim
    zk_links: int = 0

    @property
    def zk(self) -> bool:
        return self.zk_t is not None

    @property
    def n_links(self) -> int:
        return self.zk_links if self.zk else self.final_next.shape[0]

    def header(self) -> bytes:
        if self.zk:
            return self.schedule.header_zk(self.zk_links)
        return self.schedule.header(self.n_links, self.values.shape[0])

    def n_bytes(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        if self.zk:
            parts = [self.header()] + [_ext_bytes(m) for m in self.zc_rounds] + [_ext_bytes(self.final_abc), _ext_bytes(self.zk_t)] \
                + [_ext_bytes(m) for m in self.cmb_rounds] + [_ext_bytes(self.values)] + [_ext_bytes(m) for m in self.rb_rounds] \
                + [_ext_bytes(self.values_b)]
            return b"".join(parts)
        parts = [self.header()] + [_ext_bytes(m) for m in self.zc_rounds] + [_ext_bytes(self.final_abc), _ext_bytes(self.final_next)] \
            + [_ext_bytes(m) for m in self.cmb_rounds] + [_ext_bytes(self.values)]
        return b"".join(parts)

    @classmethod
    def from_bytes(cls, b: bytes, device="cpu") -> "SumcheckProof":
        import numpy as np
        if b[:8] == MAGIC_ZK:
            return cls._from_bytes_zk(b, device)
        if b[:8] != MAGIC:
            raise ValueError("not a LGSC0003 / LGSC0004 sumcheck proof")
        pos = 8

        def u8():
            nonlocal pos
            if pos >= len(b):
                raise ValueError("truncated header")
            v = b[pos]
            pos += 1
            return v
        zc = tuple(u8() for _ in range(u8()))
        vf, nl = u8(), u8()
        cmb = tuple(u8() for _ in range(u8()))
        nv = u8()
        for a in zc + cmb + (vf,):
            if not 1 <= a <= MAX_ARITY:
                raise ValueError(f"arity {a} out of range")

        def take(shape):
            nonlocal pos
            n = 1
            for s in shape:
                n *= s
            if pos + 4 * n > len(b):
                raise ValueError("truncated")
            arr = np.frombuffer(b[pos:pos + 4 * n], dtype="<u4").astype(np.int64).reshape(shape)
            pos += 4 * n
            return torch.from_numpy(arr.copy()).to(device)
        zr = [take((3,) * v + (D,)) for v in zc]
        fabc = take((3, 1 << vf, D))
        fnext = take((nl, 1 << vf, D))
        cr = [take((3,) * v + (D,)) for v in cmb]
        vals = take((nv, D))
        if pos != len(b):
            raise ValueError("trailing bytes")
        return cls(Schedule(zc, vf, cmb), zr, fabc, fnext, cr, vals)

    @classmethod
    def _from_bytes_zk(cls, b: bytes, device="cpu") -> "SumcheckProof":
        import numpy as np
        pos = 8

        def u8():
            nonlocal pos
            if pos >= len(b):
                raise ValueError("truncated header")
            v = b[pos]
            pos += 1
            return v
        zc = tuple(u8() for _ in range(u8()))
        vf, nl = u8(), u8()
        cmb = tuple(u8() for _ in range(u8()))
        rb = tuple(u8() for _ in range(u8()))
        if any(not 1 <= a <= ZK_MAX_ARITY for a in zc + cmb + rb) or not 1 <= vf <= ZK_MAX_VF:
            raise ValueError("arity out of range")

        def take(shape):
            nonlocal pos
            n = 1
            for s in shape:
                n *= s
            if pos + 4 * n > len(b):
                raise ValueError("truncated")
            arr = np.frombuffer(b[pos:pos + 4 * n], dtype="<u4").astype(np.int64).reshape(shape)
            pos += 4 * n
            return torch.from_numpy(arr.copy()).to(device)
        zr = [take((3,) * v + (D,)) for v in zc]
        fabc = take((3, 1 << vf, D))
        zt = take((D,))
        cr = [take((3,) * v + (D,)) for v in cmb]
        vals = take((3, D))
        rr = [take((3,) * v + (D,)) for v in rb]
        vb = take((2, D))
        if pos != len(b):
            raise ValueError("trailing bytes")
        return cls(Schedule(zc, vf, cmb, rb), zr, fabc, torch.zeros((0, 1 << vf, D), dtype=torch.int64, device=device), cr, vals,
                   zk_t=zt, rb_rounds=rr, values_b=vb, zk_links=nl)


# ---------------------------------------------------------------------------------------------------------------------
# Reference (torch) sumcheck states
# ---------------------------------------------------------------------------------------------------------------------

def build_tables(cons: Constraints, z: torch.Tensor, chunk: int = 64, dtype=torch.int64, col_chunk: int = 1 << 16) -> dict[str, torch.Tensor]:
    """``Az, Bz, Cz`` ``(K_pad, C)`` base tables from the COO matrices (reference path + tests)."""
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


class _RefZC:
    """Reference zero-check state: flat ext tables ``(n, 6)`` over ``x = k * C + c`` (variable ``t`` = bit ``t``)."""

    def __init__(self, cons: Constraints, z: torch.Tensor, tau: torch.Tensor, lay: Layout):
        tabs = build_tables(cons, z)
        self.T = [ext_of_base(tabs[s]).reshape(-1, D) for s in "abc"]
        self.tau = tau
        self.nrem = tau.shape[0]

    def round(self, v: int) -> torch.Tensor:
        h = self.T[0].shape[0] >> v
        Ah, Bh, Ch = (mixed_rep(to_corners(T, v), v) for T in self.T)
        mask = noinf_mask(v, Ah.device)[(...,) + (None, None)]
        q = (fc_mul(Ah, Bh) - Ch * mask) % P
        w = eq_table(self.tau[: self.nrem - v])
        assert w.shape[0] == h
        return fc_mul(q, w).sum(-2) % P

    def fold(self, r: torch.Tensor) -> None:
        self.T = [fold_top(T, r) for T in self.T]
        self.nrem -= r.shape[0]

    def final_tables(self) -> torch.Tensor:
        return torch.stack(self.T)


class _RefPP:
    """Reference combined-sumcheck state: ``sum_y P1 Q1 + P2 Q2`` over flat ext tables."""

    def __init__(self, tabs: list):
        self.T = tabs

    def round(self, v: int) -> torch.Tensor:
        X = [mixed_rep(to_corners(T, v), v) for T in self.T]
        s = fc_mul(X[0], X[1])
        if len(X) > 2:
            s = s + fc_mul(X[2], X[3])
        return s.sum(-2) % P

    def fold(self, r: torch.Tensor) -> None:
        self.T = [fold_top(T, r) for T in self.T]


class _RowRed:
    """LGSC0004's row reduction ``sum_i s_w(i) wr(i)``: two ``(R, 6)`` tables, any arity; launch-bound, so ``_fm``."""

    def __init__(self, s_w: torch.Tensor, wr: torch.Tensor):
        self.T = [s_w, wr]

    def round(self, v: int) -> torch.Tensor:
        X = [mixed_rep(to_corners(T, v), v) for T in self.T]
        return _fm(X[0], X[1]).sum(-2) % P

    def fold(self, r: torch.Tensor) -> None:
        self.T = [_fold_fm(T, r) for T in self.T]


def _fold_fm(T: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """``fold_top`` with ``_fm``."""
    Tc = to_corners(T, r.shape[0])
    for j in range(r.shape[0]):
        Tc = (Tc[0] + _fm((Tc[1] - Tc[0]) % P, r[j])) % P
    return Tc


# ---------------------------------------------------------------------------------------------------------------------
# ZK (LGSC0004): Libra masks, one polynomial per round group
# ---------------------------------------------------------------------------------------------------------------------
# A masked sumcheck with round groups R (arity v_R) adds to its summand
#     ghat(x) = sum_R (g_R(x_R) - base_R),     g_R(X) = sum_{e in {0,1,2}^v_R, e != 0} c_{R,e} prod_j X_j^{e_j}
# with uniform extension coefficients c committed in ``lay.zk.g_rows`` (3^v_R - 1 per group: EVERY monomial of degree <= 2
# per variable except the constant, so each round's 3^v - 1 free message entries are a bijective image of fresh uniform
# coefficients -- the cross terms of a multi-variable round included), and base_R = the group's weighted cube sum
# (``eq``: sum_b eq(tau_R, b) g_R(b); ``avg``: 2^-v sum_b g_R(b)).  So sum_x w(x) ghat(x) = 0 identically: the claim is
# unchanged, no coin is spent (the coefficients are bound by the commitment before any coin), and the verifier needs one
# linear functional of the coefficients at the end, K = sum_R (g_R(r_R) - base_R): a sparse PCS claim on the g rows.

_FOLD: dict = {}


def _fm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``fc_mul`` in four torch ops, for the mask algebra's tiny (launch-bound) tensors: every product reduced, then the
    36 -> 6 fold by ``X^6 = 31`` as one weighted sum (terms < 31 p, 36 of them: < 2^42)."""
    a, b = torch.broadcast_tensors(a, b)
    if a.numel() > D << 15:
        return _fm_big(a, b)
    pr = (a[..., :, None] * b[..., None, :]) % P
    return (pr.reshape(pr.shape[:-2] + (1, D * D)) * _fold_w(a.device)).sum(-1) % P


def _fold_w(dev) -> torch.Tensor:
    """``(6, 36)``: coordinate product ``a_i b_j`` -> ext coordinate ``(i + j) % 6``, times 31 if wrapped (``X^6 = 31``)."""
    W = _FOLD.get(dev)
    if W is None:
        W = torch.zeros((D, D * D), dtype=torch.int64)
        for i in range(D):
            for j in range(D):
                W[(i + j) % D, D * i + j] = 1 if i + j < D else 31
        W = _FOLD[dev] = W.to(dev)
    return W


def _fdot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``sum_e a[e] b[e]`` over the leading axis of two ``(n, 6)`` ext tensors (= ``_fm(a, b).sum(0) % P``): the reduced
    coordinate products summed first (n < 2^32), then one fold."""
    pr = (a[:, :, None] * b[:, None, :] % P).sum(0) % P
    return (pr.reshape(1, D * D) * _fold_w(a.device)).sum(-1) % P


def _fm_big(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``_fm`` for large tensors (bandwidth-bound): six products of ``a_i`` with ``b`` rotated by ``i`` (the wrapped part
    times 31), 6 temporaries per element instead of 216."""
    b31 = (b * 31) % P
    acc = (a[..., :1] * b) % P
    for i in range(1, D):
        acc += (a[..., i:i + 1] * torch.cat((b31[..., D - i:], b[..., :D - i]), -1)) % P
    return acc % P


def _outer(ws: list) -> torch.Tensor:
    """Per-axis weights ``ws[j] (3, 6)`` -> ``(3,)*v + (6,)``: ``W[e] = prod_j ws[j][e_j]``."""
    W = ws[0]
    for w in ws[1:]:
        W = _fm(W.unsqueeze(-2), w.reshape((1,) * (W.dim() - 1) + (3, D)))
    return W


def _mono_weights(kind: str, pts: torch.Tensor | None, v: int) -> torch.Tensor:
    """Per-axis weights ``(v, 3, 6)`` of the monomials ``X^0, X^1, X^2``: ``eval`` at ``pts`` (1, r, r^2); ``eq`` sum against
    ``eq(tau, .)`` (1, tau, tau); ``avg`` (1, 1/2, 1/2).  CPU."""
    one = fc_one((v,), "cpu")
    if kind == "avg":
        h = torch.zeros((v, D), dtype=torch.int64)
        h[:, 0] = pow(2, P - 2, P)
        return torch.stack((one, h, h), 1)
    pts = pts.to("cpu") % P
    return torch.stack((one, pts, pts if kind == "eq" else _fm(pts, pts)), 1)


class ZkMask:
    """The Libra mask of one sumcheck: groups of ``arities`` (round order), coefficients ``coeffs`` ``(sum 3^v - 1, 6)`` (prover;
    None on the verifier's side), ``kind`` ``"eq"`` (zero-check: summand ``eq(tau, x) (q + ghat)``) or ``"avg"`` (plain sums),
    ``offset`` = the index of its first coefficient in the g block.  The algebra runs on the CPU (``3^v x 6`` tensors); messages
    and ``K`` are handed out on ``device``."""

    def __init__(self, arities: tuple, kind: str, offset: int, coeffs: torch.Tensor | None, device):
        self.arities, self.kind, self.offset, self.device = tuple(arities), kind, offset, device
        self.starts = [0]
        for v in self.arities:
            self.starts.append(self.starts[-1] + 3 ** v - 1)
        self.coeffs = None if coeffs is None else coeffs.to("cpu")
        self._K = torch.zeros((D,), dtype=torch.int64)
        self._base: dict = {}                                     # t -> (base_t, outer base weights (3^v, 6))
        self._msg: dict = {}                                      # t -> (g_t in the mixed representation, its ones)
        self.claim_w: list = []                                   # per bound group: (3^v - 1, 6) weights of K

    @property
    def K(self) -> torch.Tensor:
        return self._K.to(self.device)

    def _c(self, t: int) -> torch.Tensor:
        c = torch.zeros((3 ** self.arities[t], D), dtype=torch.int64)
        c[1:] = self.coeffs[self.starts[t]:self.starts[t + 1]]
        return c

    def _base_w(self, t: int, taus: list | None) -> torch.Tensor:
        if t not in self._base:
            v = self.arities[t]
            pts = torch.stack(list(taus)) if self.kind == "eq" else None
            Wb = _outer(list(_mono_weights(self.kind, pts, v))).reshape(3 ** v, D)
            base = _fdot(self._c(t), Wb) if self.coeffs is not None else None
            self._base[t] = (base, Wb)
        return self._base[t]

    def _msg_part(self, t: int, taus: list | None) -> tuple:
        if t not in self._msg:
            v = self.arities[t]
            M = self._c(t).reshape((3,) * v + (D,))
            ones = torch.ones((1,) * v, dtype=torch.int64)
            for a in range(v):              # X^0, X^1, X^2 -> (1, 1, 0), (0, 1, 0), (0, 1, 1) in (q(0), q(1), q_inf) per axis
                c0, c1, c2 = M.select(a, 0), M.select(a, 1), M.select(a, 2)
                M = torch.stack((c0, (c0 + c1 + c2) % P, c2), dim=a)
                ones = torch.cat((ones, ones, torch.zeros_like(ones)), dim=a)
            self._base_w(t, taus)
            self._msg[t] = (M, ones[..., None])
        return self._msg[t]

    def prepare(self, taus: list | None = None) -> None:
        """Every round's challenge-independent message part (``eq``: ``taus[t]`` = round ``t``'s tau entries, on the CPU so
        that nothing waits for the device): the prover runs this while the device computes the opening round."""
        for t in range(len(self.arities)):
            self._msg_part(t, None if taus is None else taus[t])

    def message(self, t: int, taus: list | None = None, n_rest: int = 0) -> torch.Tensor:
        """Round ``t``'s mask part in the mixed representation: ``K + g_t(X) - base_t`` (``eq``), times ``2^n_rest`` (``avg``)."""
        M, ones = self._msg_part(t, taus)
        base, _ = self._base_w(t, taus)
        M = (M + ones * ((self._K - base) % P)) % P
        if self.kind == "avg" and n_rest:
            M = (M * pow(2, n_rest, P)) % P
        return M.to(self.device)

    def bind(self, t: int, r: torch.Tensor, taus: list | None = None) -> None:
        """After round ``t``'s challenges ``r`` (v, 6): ``K += g_t(r) - base_t`` and record the claim weights (verifier too)."""
        v = self.arities[t]
        _, Wb = self._base_w(t, taus)
        W = (_outer(list(_mono_weights("eval", r, v))).reshape(3 ** v, D) - Wb) % P
        self.claim_w.append(W[1:])
        if self.coeffs is not None:
            self._K = (self._K + _fdot(self._c(t), W)) % P

    def claim(self, lay: Layout, value: torch.Tensor, name: str) -> SparseClaim:
        """``K = value`` as a sparse claim on the committed cells: coefficient ``u`` (global index ``offset + u``) has its
        coordinate ``d`` at g-block cell ``6 (offset + u) + d`` with weight ``W_u x^d``."""
        W = torch.cat(self.claim_w, 0)                                           # (n_coeffs, 6)
        basis = torch.eye(D, dtype=torch.int64)
        w = _fm(W[:, None, :], basis[None, :, :]).reshape(-1, D)                # (6 n_coeffs, 6)
        cells = torch.arange(D * self.offset, D * (self.offset + W.shape[0]), dtype=torch.int64)
        g_rows = torch.tensor(lay.zk.g_rows, dtype=torch.int64)
        dev = self.device
        return SparseClaim((g_rows[cells // lay.C]).to(dev), (cells % lay.C).to(dev), w.to(dev), value, name)


def zk_masks(lay: Layout, sched: Schedule, z: torch.Tensor | None, device, allow_underblinded: bool = False) -> tuple:
    """The three masks (zero-check, combined, row reduction) with their coefficients read from ``z``'s g rows (prover) or
    without (verifier, ``z = None``).  The prover's masks need ``vf <= n_c - 4`` (``zk_vf_max``) unless
    ``allow_underblinded`` (toy-size format tests: the proof verifies, its final tables are not fully blinded)."""
    n = sched.zk_coeffs()
    if D * n > min(lay.zk.g_cells, len(lay.zk.g_rows) * lay.C):
        raise ValueError(f"schedule needs {D * n} Libra cells > the layout's {lay.zk.g_cells}")
    if z is not None and not allow_underblinded and sched.vf > zk_vf_max(lay):
        raise ValueError(f"LGSC0004 prover: vf = {sched.vf} > n_c - {ZK_TABLE_SLACK} = {zk_vf_max(lay)}: the final Az/Bz/Cz "
                         f"tables would not be blinded by the product triple (red-team-ligerito-3 R3-6)")
    coeffs = None
    if z is not None:
        g = z.index_select(0, torch.tensor(lay.zk.g_rows, dtype=torch.int64, device=z.device)).reshape(-1)[:D * n]
        coeffs = (g.to(torch.int64) % P).reshape(n, D).to(device)
    out, off = [], 0
    for arities, kind in ((sched.zc, "eq"), (sched.cmb, "avg"), (sched.rb, "avg")):
        k = sum(3 ** v - 1 for v in arities)
        out.append(ZkMask(arities, kind, off, None if coeffs is None else coeffs[off:off + k], device))
        off += k
    return tuple(out)


def zk_libra_cells(lay: Layout, sched: Schedule) -> tuple[torch.Tensor, torch.Tensor]:
    """``(rows, cols)`` int64 of the ``6 * zk_coeffs`` base cells holding ``sched``'s Libra coefficients (g-block cell ``k`` at
    ``(g_rows[k // C], k % C)``; several g rows when ``C`` is small).  Nothing else may be written to or claimed on them."""
    cells = torch.arange(D * sched.zk_coeffs(), dtype=torch.int64)
    return torch.tensor(lay.zk.g_rows, dtype=torch.int64)[cells // lay.C], cells % lay.C


def check_zk_claim_supports(lay: Layout, sched: Schedule, claims: list) -> None:
    """Raise unless ``claims`` is the claim set LGSC0004's ZK argument covers (red-team-ligerito-3 R3-4): two evaluation
    claims whose row coordinates are all non-boolean (their row weights ``eq(r, .)`` reach every uniform row), then the three
    mask claims exactly on their Libra cells (zc, cmb, rb blocks in g-block order: disjoint, and inside ``g_rows``, which
    share no row with the uniform / product / mask / next rows, the virtual block or the witness rows)."""
    zr = lay.zk
    g = set(zr.g_rows)
    others = set(zr.rows()) - g | set(range(lay.m + len(lay.virt)))
    if len(g) != len(zr.g_rows) or g & others or D * sched.zk_coeffs() > min(zr.g_cells, len(zr.g_rows) * lay.C):
        raise ValueError("ZK layout: the Libra block overlaps other rows or is too small")
    if [type(c).__name__ for c in claims] != ["EvalClaim"] * 2 + ["SparseClaim"] * 3:
        raise ValueError("ZK claims: expected 2 evaluation + 3 sparse claims")
    for c in claims[:2]:
        rc = c.point[:lay.n_i].to("cpu")
        if tuple(c.point.shape) != (lay.n_vars, D) or bool(((rc[:, 1:] == 0).all(1) & (rc[:, 0] <= 1)).any()):
            raise ValueError(f"ZK claim {c.name}: a boolean row coordinate (the claim would miss uniform rows)")
    rows, cols = zk_libra_cells(lay, sched)
    off = 0
    for c, arities in zip(claims[2:], (sched.zc, sched.cmb, sched.rb)):
        k = D * sum(3 ** v - 1 for v in arities)
        if not (torch.equal(c.rows.to("cpu"), rows[off:off + k]) and torch.equal(c.cols.to("cpu"), cols[off:off + k])) \
                or tuple(c.weights.shape) != (k, D):
            raise ValueError(f"ZK claim {c.name}: not exactly its Libra cells")
        off += k


def zk_row_weights(lay: Layout, g3: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor]:
    """``(n_w, s_w)`` (R, 6): the shift's row weights -- ``n_w = sum_x g3^x e_{next_x} + g3^L e_{M_next}`` (the committed shifted
    rows, at the zero-check's column point) and ``s_w = sum_x g3^x e_{c_x} + g3^L e_M`` (the linked rows, reduced by the
    row-reduction sumcheck); ``L = n_links``: the mask link is one more link."""
    L = lay.n_links
    g3p = fc_pow_seq(g3, L + 1)
    n_w = torch.zeros((lay.R, D), dtype=torch.int64, device=device)
    s_w = torch.zeros((lay.R, D), dtype=torch.int64, device=device)
    for x in range(L):
        n_w[lay.zk.next[x]] = (n_w[lay.zk.next[x]] + g3p[x]) % P
        s_w[lay.c_rows[x]] = (s_w[lay.c_rows[x]] + g3p[x]) % P
    n_w[lay.zk.mask_next] = (n_w[lay.zk.mask_next] + g3p[L]) % P
    s_w[lay.zk.mask] = (s_w[lay.zk.mask] + g3p[L]) % P
    return n_w, s_w


# ---------------------------------------------------------------------------------------------------------------------
# Prover
# ---------------------------------------------------------------------------------------------------------------------

def _ones_index(v: int) -> tuple:
    return (1,) * v


def _fix_zc_round(M: torch.Tensor, taus: list, claim: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """Cheat: set ``M[1..1]`` so that ``E sum_b eq(tau, b) M[b] == claim``."""
    v = len(taus)
    M = M.clone()
    M[_ones_index(v)] = 0
    rest = eq_sum_mixed(M, taus)
    w1 = fc_one((), M.device)
    for t in taus:
        w1 = fc_mul(w1, t)
    M[_ones_index(v)] = fc_mul((fc_mul(claim, fc_inv(E)) - rest) % P, fc_inv(w1))
    return M


def _fix_sum_round(M: torch.Tensor, claim: torch.Tensor) -> torch.Tensor:
    """Cheat: set ``M[1..1]`` so that ``sum_b M[b] == claim``."""
    v = M.dim() - 1
    M = M.clone()
    M[_ones_index(v)] = 0
    M[_ones_index(v)] = (claim - bool_sum_mixed(M)) % P
    return M


def _next_tables(lay: Layout, z: torch.Tensor, r_hi: torch.Tensor, vf: int, engine) -> torch.Tensor:
    """``next_x(r_c_hi, b)`` for ``b in {0,1}^vf``: ``(n_links, 2^vf, 6)``; ``r_hi`` = the bound column variables ``vf .. n_c - 1``."""
    dev = z.device
    if not lay.n_links:
        return torch.zeros((0, 1 << vf, D), dtype=torch.int64, device=dev)
    B, H = 1 << vf, lay.C >> vf
    rows = torch.tensor([lay.virt[f"next:{x}"] for x in range(lay.n_links)], device=dev)
    M = z.index_select(0, rows).reshape(lay.n_links, H, B).permute(0, 2, 1).reshape(lay.n_links * B, H).contiguous()
    if engine == "cuda":
        return matvec_base_ext_cuda(M, eq_table_cuda(r_hi)).reshape(lay.n_links, B, D)
    return matvec_base_ext(M, eq_table(r_hi)).reshape(lay.n_links, B, D)


def prove(lay: Layout, cons: Constraints, z: torch.Tensor, coins, cheat: str | None = None, engine=None,
          schedule: Schedule | None = None, allow_underblinded: bool = False) -> SumcheckProof:
    """The zero-check and the combined sumcheck on ``z`` (``build_z``).  ``cheat`` (negative tests only): ``"round1"`` fixes
    the opening message so its check passes (caught in the next round); ``"zc"`` fixes every zero-check message and the
    final tables (the zero-check passes; the combined sumcheck's first round catches the false ``Az, Bz, Cz``); ``"all"``
    also fixes every combined round (the combined final check catches it).
    ``engine``: ``None`` = the torch reference (small sizes); ``"cuda"`` = the fused cupy kernels (``CudaZC`` / ``CudaPP``).
    A ZK layout (``lay.zk``; ``z`` after ``layout.fill_zk``) proves LGSC0004 (``_prove_zk``); it refuses a schedule with
    ``vf > n_c - 4`` unless ``allow_underblinded`` (``zk_masks``)."""
    if lay.zk is not None:
        return _prove_zk(lay, cons, z, coins, cheat, engine, schedule, allow_underblinded)
    dev = z.device
    n_k, n_c, n_i = lay.n_k, lay.n_c, lay.n_i
    n = n_k + n_c
    n_cmb = n_cmb_of(lay)
    sched = schedule or default_schedule(n, n_cmb)
    if sum(sched.zc) + sched.vf != n or sum(sched.cmb) != n_cmb or sched.vf > n_c:
        raise ValueError(f"schedule {sched} does not fit n = {n}, n_cmb = {n_cmb}")
    tr = _Transcript(coins, dev)
    t0 = time.perf_counter()
    timings: dict = {}
    one = fc_one((), dev)
    tr.absorb_bytes("lgsc/header", sched.header(lay.n_links, 1 + lay.n_links))
    tau = tr.challenge("zc/tau", n)                                  # tau[t] <-> variable t; k bits are the top n_k
    # --- zero-check ----------------------------------------------------------------------------------------------------
    st = CudaZC(cons, z, tau, lay) if engine == "cuda" else _RefZC(cons, z, tau, lay)
    _sync(dev)
    timings["zc_setup"] = time.perf_counter() - t0
    E, claim = one.clone(), torch.zeros((D,), dtype=torch.int64, device=dev)
    rs: list = []
    zc_msgs, per_round, msg_t = [], [], []
    nrem = n
    for t, v in enumerate(sched.zc):
        tr_ = time.perf_counter()
        M = st.round(v)
        _sync(dev)
        msg_t.append(time.perf_counter() - tr_)
        taus = [tau[nrem - 1 - j] for j in range(v)]
        if cheat in ("zc", "all") or (cheat == "round1" and t == 0):
            M = _fix_zc_round(M, taus, claim, E)
        zc_msgs.append(M)
        tr.absorb(f"zc/q/{t}", M)
        r = tr.challenge(f"zc/r/{t}", v)
        st.fold(r)
        if cheat:
            for j in range(v):
                E = fc_mul(E, _eq1(taus[j], r[j]))
            claim = fc_mul(E, eval_mixed(M, r))
        rs += list(r)
        nrem -= v
        _sync(dev)
        per_round.append(time.perf_counter() - tr_)
    tr_ = time.perf_counter()
    vf = sched.vf
    fabc = st.final_tables()                                          # (3, 2^vf, 6)
    del st
    r_hi = torch.stack(rs[::-1])[: n_c - vf]                          # variables vf .. n_c - 1
    fnext = _next_tables(lay, z, r_hi, vf, engine)
    if cheat in ("zc", "all"):
        eqt = eq_table(tau[:vf])
        prod = (fc_mul(fabc[0], fabc[1]) - fabc[2]) % P
        ones = (1 << vf) - 1
        rest = (fc_mul(prod, eqt).sum(0) - fc_mul(prod[ones], eqt[ones])) % P
        want = (fc_mul(claim, fc_inv(E)) - rest) % P
        fabc = fabc.clone()
        fabc[2, ones] = (fc_mul(fabc[0, ones], fabc[1, ones]) - fc_mul(want, fc_inv(eqt[ones]))) % P
    tr.absorb("zc/final", torch.cat((fabc, fnext), 0))
    ch = tr.challenge("zc/final", vf + 3)
    gamma, g3, beta = ch[vf], ch[vf + 1], ch[vf + 2]
    rs += list(ch[:vf])
    r_var = torch.stack(rs[::-1])                                     # r_var[t] <-> variable t of (k, c)
    r_c, r_k = r_var[:n_c], r_var[n_c:]
    _sync(dev)
    per_round.append(time.perf_counter() - tr_)
    timings["zc"] = time.perf_counter() - t0
    timings["zc_rounds"] = per_round
    timings["zc_msg"] = msg_t
    # --- combined rows + shift sumcheck --------------------------------------------------------------------------------
    t2 = time.perf_counter()
    if engine == "cuda":
        coef = combined_coefficients_cuda(cons, lay, r_k, gamma)
        eq_c = eq_table_cuda(r_c)
        zc_ = matvec_base_ext_cuda(z, eq_c)                          # (R, 6) = z~(., r_c)
    else:
        coef = combined_coefficients(cons, lay, r_k, gamma)
        eq_c = eq_table(r_c)
        zc_ = matvec_base_ext(z, eq_c)
    _sync(dev)
    timings["cmb_coef_matvec"] = time.perf_counter() - t2
    g3p = fc_pow_seq(g3, max(lay.n_links, 1))
    Vt = torch.zeros((lay.C, D), dtype=torch.int64, device=dev)
    for x, cr in enumerate(lay.c_rows):
        Vt = (Vt + fc_scalar_mul(z[cr].to(torch.int64), g3p[x][None, :].expand(lay.C, D))) % P
    Ut = torch.roll(eq_c.reshape(lay.S, lay.l, D), 1, dims=1).reshape(lay.C, D)      # U[c] = eq_c[pred(c)]
    rep_i, rep_c = 1 << (n_cmb - n_i), 1 << (n_cmb - n_c)
    tabs = [fc_mul(coef, beta).repeat(rep_i, 1), zc_.repeat(rep_i, 1), Ut.repeat(rep_c, 1), Vt.repeat(rep_c, 1)]
    pp = CudaPP(tabs) if engine == "cuda" else _RefPP(tabs)
    del tabs
    if cheat:
        eqf = eq_table(r_var[:vf])
        abc = [mle_eval_ext(fabc[s], eqf) for s in range(3)]
        claim2 = (abc[0] + fc_mul(gamma, abc[1]) + fc_mul(fc_mul(gamma, gamma), abc[2])) % P
        claim2 = fc_mul(fc_mul(beta, claim2), ext_of_int(rep_i, dev))
        nv = [mle_eval_ext(fnext[x], eqf) for x in range(lay.n_links)]
        sh = torch.zeros((D,), dtype=torch.int64, device=dev)
        for x in range(lay.n_links):
            sh = (sh + fc_mul(g3p[x], nv[x])) % P
        claim2 = (claim2 + fc_mul(sh, ext_of_int(rep_c, dev))) % P
    _sync(dev)
    timings["cmb_setup"] = time.perf_counter() - t2 - timings["cmb_coef_matvec"]
    t3 = time.perf_counter()
    cmb_msgs, rs2 = [], []
    for t, v in enumerate(sched.cmb):
        M = pp.round(v)
        if cheat == "all":
            M = _fix_sum_round(M, claim2)
        cmb_msgs.append(M)
        tr.absorb(f"cmb/q/{t}", M)
        r = tr.challenge(f"cmb/r/{t}", v)
        pp.fold(r)
        if cheat:
            claim2 = eval_mixed(M, r)
        rs2 += list(r)
    del pp
    _sync(dev)
    timings["cmb_rounds"] = time.perf_counter() - t3
    rho = torch.stack(rs2[::-1])                                      # rho[t] <-> variable t of y
    r_i, rho_c = rho[:n_i], rho[:n_c]
    w_only = zc_.clone()
    w_only[lay.m:lay.m + len(lay.virt)] = 0                           # the virtual rows are not committed (free rows above are)
    eq_i = (eq_table_cuda if engine == "cuda" else eq_table)(r_i)
    eq_rho = (eq_table_cuda if engine == "cuda" else eq_table)(rho_c)
    crow = z.index_select(0, torch.tensor(lay.c_rows, dtype=torch.int64, device=dev))
    if engine == "cuda":
        vals = [mle_eval_ext(w_only, eq_i)] + list(matvec_base_ext_cuda(crow, eq_rho))
    else:
        vals = [mle_eval_ext(w_only, eq_i)] + list(matvec_base_ext(crow, eq_rho))
    values = torch.stack(vals)
    tr.absorb("values", values)
    _sync(dev)
    timings["cmb"] = time.perf_counter() - t2
    timings["total"] = time.perf_counter() - t0
    claims = [EvalClaim(torch.cat((r_i, r_c)), values[0], "w(r_i, r_c)")]
    for x, cr in enumerate(lay.c_rows):
        claims.append(EvalClaim(torch.cat((_bits_point(cr, n_i, dev), rho_c)), values[1 + x], f"w(c_{x}, rho)"))
    return SumcheckProof(sched, zc_msgs, fabc, fnext, cmb_msgs, values, claims, timings)


def _prove_zk(lay: Layout, cons: Constraints, z: torch.Tensor, coins, cheat: str | None, engine,
              schedule: Schedule | None, allow_underblinded: bool = False) -> SumcheckProof:
    """LGSC0004: the zero-check and the combined sumcheck with Libra masks (``ZkMask``); the committed shifted rows and the mask
    link ride in the combined sumcheck's rows part (``n_w``); then the row reduction of ``V'(rho_c) = sum_i s_w(i) w~(i, rho_c)``
    to ONE claim at a random row point.  Claims: ``w~(r_i, r_c)``, ``w~(rho_i, rho_c)`` and three sparse mask claims."""
    dev = z.device
    n_k, n_c, n_i = lay.n_k, lay.n_c, lay.n_i
    n = n_k + n_c
    n_cmb = n_cmb_of(lay)
    sched = schedule if schedule is not None and schedule.rb else zk_schedule(lay, schedule)
    if sum(sched.zc) + sched.vf != n or sum(sched.cmb) != n_cmb or sum(sched.rb) != n_i or sched.vf > n_c:
        raise ValueError(f"schedule {sched} does not fit n = {n}, n_cmb = {n_cmb}, n_i = {n_i}")
    mz, mc, mr = zk_masks(lay, sched, z, dev, allow_underblinded)
    tr = _Transcript(coins, dev)
    t0 = time.perf_counter()
    timings: dict = {}
    one = fc_one((), dev)
    zero = torch.zeros((D,), dtype=torch.int64, device=dev)
    tr.absorb_bytes("lgsc/header", sched.header_zk(lay.n_links))
    tau = tr.challenge("zc/tau", n)
    # --- zero-check ----------------------------------------------------------------------------------------------------
    st = CudaZC(cons, z, tau, lay) if engine == "cuda" else _RefZC(cons, z, tau, lay)
    _sync(dev)
    timings["zc_setup"] = time.perf_counter() - t0
    E, claim = one.clone(), zero.clone()
    rs: list = []
    zc_msgs, per_round, msg_t = [], [], []
    tau_h, zc_taus, nrem = tau.cpu(), [], n
    for v in sched.zc:
        zc_taus.append([tau_h[nrem - 1 - j] for j in range(v)])
        nrem -= v
    nrem = n
    for t, v in enumerate(sched.zc):
        tr_ = time.perf_counter()
        M = st.round(v)
        if t == 0:
            for mk, tk in ((mz, zc_taus), (mc, None), (mr, None)):
                mk.prepare(tk)
        _sync(dev)
        msg_t.append(time.perf_counter() - tr_)
        taus = [tau[nrem - 1 - j] for j in range(v)]
        M = (M + mz.message(t, zc_taus[t])) % P
        if cheat in ("zc", "all") or (cheat == "round1" and t == 0):
            M = _fix_zc_round(M, taus, claim, E)
        zc_msgs.append(M)
        tr.absorb(f"zc/q/{t}", M)
        r = tr.challenge(f"zc/r/{t}", v)
        st.fold(r)
        mz.bind(t, r, taus)
        if cheat:
            for j in range(v):
                E = fc_mul(E, _eq1(taus[j], r[j]))
            claim = fc_mul(E, eval_mixed(M, r))
        rs += list(r)
        nrem -= v
        _sync(dev)
        per_round.append(time.perf_counter() - tr_)
    tr_ = time.perf_counter()
    vf = sched.vf
    fabc = st.final_tables()                                          # (3, 2^vf, 6)
    del st
    zk_t = mz.K.clone()
    if cheat in ("zc", "all"):
        eqt = eq_table(tau[:vf])
        prod = (fc_mul(fabc[0], fabc[1]) - fabc[2]) % P
        ones = (1 << vf) - 1
        rest = (fc_mul(prod, eqt).sum(0) - fc_mul(prod[ones], eqt[ones])) % P
        want = (fc_mul(claim, fc_inv(E)) - zk_t - rest) % P
        fabc = fabc.clone()
        fabc[2, ones] = (fc_mul(fabc[0, ones], fabc[1, ones]) - fc_mul(want, fc_inv(eqt[ones]))) % P
    tr.absorb("zc/final", torch.cat((fabc.reshape(-1, D), zk_t[None]), 0))
    ch = tr.challenge("zc/final", vf + 3)
    gamma, g3, beta = ch[vf], ch[vf + 1], ch[vf + 2]
    rs += list(ch[:vf])
    r_var = torch.stack(rs[::-1])
    r_c, r_k = r_var[:n_c], r_var[n_c:]
    _sync(dev)
    per_round.append(time.perf_counter() - tr_)
    timings["zc"] = time.perf_counter() - t0
    timings["zc_rounds"] = per_round
    timings["zc_msg"] = msg_t
    # --- combined: rows (+ committed shifted rows + M_next, at r_c) - shift (linked rows + M, over the columns) ------------
    t2 = time.perf_counter()
    if engine == "cuda":
        coef = combined_coefficients_cuda(cons, lay, r_k, gamma)
        eq_c = eq_table_cuda(r_c)
        zc_ = matvec_base_ext_cuda(z, eq_c)
    else:
        coef = combined_coefficients(cons, lay, r_k, gamma)
        eq_c = eq_table(r_c)
        zc_ = matvec_base_ext(z, eq_c)
    _sync(dev)
    timings["cmb_coef_matvec"] = time.perf_counter() - t2
    n_w, s_w = zk_row_weights(lay, g3, dev)
    rep_i, rep_c = 1 << (n_cmb - n_i), 1 << (n_cmb - n_c)
    P1 = ((_fm(coef, beta) + n_w) % P * rep_c) % P
    g3p = fc_pow_seq(g3, lay.n_links + 1)
    Vt = torch.zeros((lay.C, D), dtype=torch.int64, device=dev)
    for x, cr in enumerate(list(lay.c_rows) + [lay.zk.mask]):
        Vt = (Vt + fc_scalar_mul(z[cr].to(torch.int64), g3p[x][None, :].expand(lay.C, D))) % P
    Ut = (torch.roll(eq_c.reshape(lay.S, lay.l, D), 1, dims=1).reshape(lay.C, D) * (P - rep_i)) % P
    tabs = [P1.repeat(rep_i, 1), zc_.repeat(rep_i, 1), Ut.repeat(rep_c, 1), Vt.repeat(rep_c, 1)]
    pp = CudaPP(tabs) if engine == "cuda" else _RefPP(tabs)
    del tabs
    if cheat:
        eqf = eq_table(r_var[:vf])
        abc = [mle_eval_ext(fabc[s], eqf) for s in range(3)]
        claim2 = (abc[0] + fc_mul(gamma, abc[1]) + fc_mul(fc_mul(gamma, gamma), abc[2])) % P
        claim2 = fc_mul(fc_mul(beta, claim2), ext_of_int(rep_i * rep_c, dev))
    _sync(dev)
    timings["cmb_setup"] = time.perf_counter() - t2 - timings["cmb_coef_matvec"]
    t3 = time.perf_counter()
    cmb_msgs, rs2 = [], []
    nrem = n_cmb
    for t, v in enumerate(sched.cmb):
        M = pp.round(v)
        nrem -= v
        M = (M + mc.message(t, n_rest=nrem)) % P
        if cheat == "all":
            M = _fix_sum_round(M, claim2)
        cmb_msgs.append(M)
        tr.absorb(f"cmb/q/{t}", M)
        r = tr.challenge(f"cmb/r/{t}", v)
        pp.fold(r)
        mc.bind(t, r)
        if cheat:
            claim2 = eval_mixed(M, r)
        rs2 += list(r)
    del pp
    _sync(dev)
    timings["cmb_rounds"] = time.perf_counter() - t3
    rho = torch.stack(rs2[::-1])
    r_i, rho_c = rho[:n_i], rho[:n_c]
    virt = slice(lay.m, lay.m + len(lay.virt))
    w_only = zc_.clone()
    w_only[virt] = 0
    eqt_ = eq_table_cuda if engine == "cuda" else eq_table
    wr = (matvec_base_ext_cuda if engine == "cuda" else matvec_base_ext)(z, eqt_(rho_c))      # (R, 6) = z~(., rho_c)
    wr[virt] = 0
    v_link = _fm(s_w, wr).sum(0) % P
    values = torch.stack([_fm(w_only, eqt_(r_i)).sum(0) % P, v_link, mc.K])
    tr.absorb("values", values)
    _sync(dev)
    timings["cmb"] = time.perf_counter() - t2
    # --- row reduction: sum_i s_w(i) w~(i, rho_c) = V'(rho_c) ------------------------------------------------------------
    t4 = time.perf_counter()
    pr = _RowRed(s_w, wr)
    rb_msgs, rs3 = [], []
    nrem = n_i
    for t, v in enumerate(sched.rb):
        M = pr.round(v)
        nrem -= v
        M = (M + mr.message(t, n_rest=nrem)) % P
        rb_msgs.append(M)
        tr.absorb(f"rb/q/{t}", M)
        r = tr.challenge(f"rb/r/{t}", v)
        pr.fold(r)
        mr.bind(t, r)
        rs3 += list(r)
    rho_i = torch.stack(rs3[::-1])
    values_b = torch.stack([_fm(wr, eqt_(rho_i)).sum(0) % P, mr.K])
    tr.absorb("values_b", values_b)
    _sync(dev)
    timings["rb"] = time.perf_counter() - t4
    timings["total"] = time.perf_counter() - t0
    claims = [EvalClaim(torch.cat((r_i, r_c)), values[0], "w(r_i, r_c)"),
              EvalClaim(torch.cat((rho_i, rho_c)), values_b[0], "w(rho_i, rho_c)"),
              mz.claim(lay, zk_t, "mask(zc)"), mc.claim(lay, values[2], "mask(cmb)"), mr.claim(lay, values_b[1], "mask(rb)")]
    check_zk_claim_supports(lay, sched, claims)
    return SumcheckProof(sched, zc_msgs, fabc, torch.zeros((0, 1 << vf, D), dtype=torch.int64, device=dev), cmb_msgs, values,
                         claims, timings, zk_t=zk_t, rb_rounds=rb_msgs, values_b=values_b, zk_links=lay.n_links)


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
    """Check the zero-check and the combined sumcheck against the public data; returns ``(ok, reason, claims)`` -- the PCS
    claims the caller must have verified against the commitment for the whole thing to hold.  ``pub_rows``:
    ``layout.public_rows``.  The round schedule is read from the proof (any arities in ``1 .. MAX_ARITY`` that add up)."""
    if (lay.zk is not None) != proof.zk:
        return False, "ZK layout / proof format mismatch (LGSC0004 iff lay.zk)", []
    if lay.zk is not None:
        return _verify_zk(lay, cons, pub_rows, proof, coins, device)
    dev = torch.device(device)
    tr = _Transcript(coins, dev)
    one = fc_one((), dev)
    n_k, n_c, n_i = lay.n_k, lay.n_c, lay.n_i
    n = n_k + n_c
    n_cmb = n_cmb_of(lay)
    sch = proof.schedule
    vf = sch.vf
    if sum(sch.zc) + vf != n or sum(sch.cmb) != n_cmb or vf > n_c or not sch.zc \
            or any(not 1 <= a <= MAX_ARITY for a in sch.zc + sch.cmb + (vf,)):
        return False, "schedule does not fit the layout", []
    if proof.final_abc.shape != (3, 1 << vf, D) or proof.final_next.shape != (lay.n_links, 1 << vf, D) \
            or proof.values.shape != (1 + lay.n_links, D) \
            or any(m.shape != (3,) * a + (D,) for m, a in zip(proof.zc_rounds, sch.zc)) or len(proof.zc_rounds) != len(sch.zc) \
            or any(m.shape != (3,) * a + (D,) for m, a in zip(proof.cmb_rounds, sch.cmb)) or len(proof.cmb_rounds) != len(sch.cmb):
        return False, "shape mismatch", []
    for t in list(proof.zc_rounds) + list(proof.cmb_rounds) + [proof.final_abc, proof.final_next, proof.values]:
        if bool((t < 0).any()) or bool((t >= P).any()):
            return False, "non-canonical field element", []
    tr.absorb_bytes("lgsc/header", proof.header())
    tau = tr.challenge("zc/tau", n)
    # --- zero-check ----------------------------------------------------------------------------------------------------
    E = one.clone()
    claim = torch.zeros((D,), dtype=torch.int64, device=dev)
    rs: list = []
    nrem = n
    for t, v in enumerate(sch.zc):
        M = proof.zc_rounds[t].to(dev)
        taus = [tau[nrem - 1 - j] for j in range(v)]
        if not torch.equal(fc_mul(E, eq_sum_mixed(M, taus)), claim):
            return False, f"zero-check round {t}: sum_b eq(tau, b) q(b) != claim", []
        tr.absorb(f"zc/q/{t}", M)
        r = tr.challenge(f"zc/r/{t}", v)
        for j in range(v):
            E = fc_mul(E, _eq1(taus[j], r[j]))
        claim = fc_mul(E, eval_mixed(M, r))
        rs += list(r)
        nrem -= v
    fabc, fnext = proof.final_abc.to(dev), proof.final_next.to(dev)
    prod = (fc_mul(fabc[0], fabc[1]) - fabc[2]) % P
    if not torch.equal(fc_mul(E, mle_eval_ext(prod, eq_table(tau[:vf]))), claim):
        return False, "zero-check final: sum_b eq(tau, b) (Az Bz - Cz)(b) != claim", []
    tr.absorb("zc/final", torch.cat((fabc, fnext), 0))
    ch = tr.challenge("zc/final", vf + 3)
    gamma, g3, beta = ch[vf], ch[vf + 1], ch[vf + 2]
    rs += list(ch[:vf])
    r_var = torch.stack(rs[::-1])
    r_c, r_k = r_var[:n_c], r_var[n_c:]
    eqf = eq_table(r_var[:vf])
    abc = [mle_eval_ext(fabc[s], eqf) for s in range(3)]
    next_vals = [mle_eval_ext(fnext[x], eqf) for x in range(lay.n_links)]
    # --- combined ------------------------------------------------------------------------------------------------------
    gp = fc_pow_seq(gamma, 3)
    g3p = fc_pow_seq(g3, max(lay.n_links, 1))
    rep_i, rep_c = 1 << (n_cmb - n_i), 1 << (n_cmb - n_c)
    rows_claim = (abc[0] + fc_mul(gp[1], abc[1]) + fc_mul(gp[2], abc[2])) % P
    shift_claim = torch.zeros((D,), dtype=torch.int64, device=dev)
    for x in range(lay.n_links):
        shift_claim = (shift_claim + fc_mul(g3p[x], next_vals[x])) % P
    claim = (fc_mul(fc_mul(beta, rows_claim), ext_of_int(rep_i, dev)) + fc_mul(shift_claim, ext_of_int(rep_c, dev))) % P
    rs2: list = []
    for t, v in enumerate(sch.cmb):
        M = proof.cmb_rounds[t].to(dev)
        if not torch.equal(bool_sum_mixed(M), claim):
            return False, f"combined round {t}: sum_b q(b) != claim", []
        tr.absorb(f"cmb/q/{t}", M)
        r = tr.challenge(f"cmb/r/{t}", v)
        claim = eval_mixed(M, r)
        rs2 += list(r)
    rho = torch.stack(rs2[::-1])
    r_i, rho_c = rho[:n_i], rho[:n_c]
    coef = combined_coefficients(cons, lay, r_k, gamma)
    eq_i = eq_table(r_i)
    coef_r = mle_eval_ext(coef, eq_i)
    eq_c = eq_table(r_c)
    # z~(r_i, r_c) = w~(r_i, r_c) + sum_v eq(r_i, idx_v) virt_v(r_c)   (one matvec for all public virtual rows)
    values = proof.values.to(dev)
    z_r = values[0].clone()
    pub_names = [nm for nm in lay.virt if not nm.startswith("next:")]
    pub_vals = matvec_base_ext(torch.stack([pub_rows[nm].to(dev) for nm in pub_names]), eq_c)      # (n_pub, 6)
    for name, pv in zip(pub_names, pub_vals):
        z_r = (z_r + fc_mul(eq_i[lay.virt[name]], pv)) % P
    for x in range(lay.n_links):
        z_r = (z_r + fc_mul(eq_i[lay.virt[f"next:{x}"]], next_vals[x])) % P
    succ = succ_mle(rho_c, r_c, lay.n_j)
    V_r = torch.zeros((D,), dtype=torch.int64, device=dev)
    for x in range(lay.n_links):
        V_r = (V_r + fc_mul(g3p[x], values[1 + x])) % P
    if not torch.equal((fc_mul(fc_mul(beta, coef_r), z_r) + fc_mul(succ, V_r)) % P, claim):
        return False, "combined final: beta coef(r_i) z(r_i, r_c) + succ(rho, r_c) V(rho) != claim", []
    tr.absorb("values", values)
    claims = [EvalClaim(torch.cat((r_i, r_c)), values[0], "w(r_i, r_c)")]
    for x, cr in enumerate(lay.c_rows):
        claims.append(EvalClaim(torch.cat((_bits_point(cr, n_i, dev), rho_c)), values[1 + x], f"w(c_{x}, rho)"))
    return True, "accept (sumchecks; PCS claims pending)", claims


def _verify_zk(lay: Layout, cons: Constraints, pub_rows: dict[str, torch.Tensor], proof: SumcheckProof, coins,
               device="cpu") -> tuple[bool, str, list]:
    """LGSC0004's verifier (``_prove_zk``); returns the two evaluation claims and the three sparse mask claims."""
    dev = torch.device(device)
    tr = _Transcript(coins, dev)
    one = fc_one((), dev)
    n_k, n_c, n_i = lay.n_k, lay.n_c, lay.n_i
    n = n_k + n_c
    n_cmb = n_cmb_of(lay)
    sch = proof.schedule
    vf = sch.vf
    if sum(sch.zc) + vf != n or sum(sch.cmb) != n_cmb or sum(sch.rb) != n_i or vf > n_c or not sch.zc or not sch.rb \
            or any(not 1 <= a <= ZK_MAX_ARITY for a in sch.zc + sch.cmb + sch.rb) or not 1 <= vf <= ZK_MAX_VF \
            or proof.zk_links != lay.n_links:
        return False, "schedule does not fit the layout", []
    if D * sch.zk_coeffs() > min(lay.zk.g_cells, len(lay.zk.g_rows) * lay.C):
        return False, "schedule needs more Libra coefficients than the layout reserves", []
    if proof.final_abc.shape != (3, 1 << vf, D) or proof.zk_t.shape != (D,) or proof.values.shape != (3, D) \
            or proof.values_b is None or proof.values_b.shape != (2, D) \
            or len(proof.zc_rounds) != len(sch.zc) or any(m.shape != (3,) * a + (D,) for m, a in zip(proof.zc_rounds, sch.zc)) \
            or len(proof.cmb_rounds) != len(sch.cmb) or any(m.shape != (3,) * a + (D,) for m, a in zip(proof.cmb_rounds, sch.cmb)) \
            or len(proof.rb_rounds) != len(sch.rb) or any(m.shape != (3,) * a + (D,) for m, a in zip(proof.rb_rounds, sch.rb)):
        return False, "shape mismatch", []
    for t in list(proof.zc_rounds) + list(proof.cmb_rounds) + list(proof.rb_rounds) + \
            [proof.final_abc, proof.zk_t, proof.values, proof.values_b]:
        if bool((t < 0).any()) or bool((t >= P).any()):
            return False, "non-canonical field element", []
    mz, mc, mr = zk_masks(lay, sch, None, dev)
    tr.absorb_bytes("lgsc/header", proof.header())
    tau = tr.challenge("zc/tau", n)
    # --- zero-check ----------------------------------------------------------------------------------------------------
    E = one.clone()
    claim = torch.zeros((D,), dtype=torch.int64, device=dev)
    rs: list = []
    nrem = n
    for t, v in enumerate(sch.zc):
        M = proof.zc_rounds[t].to(dev)
        taus = [tau[nrem - 1 - j] for j in range(v)]
        if not torch.equal(fc_mul(E, eq_sum_mixed(M, taus)), claim):
            return False, f"zero-check round {t}: sum_b eq(tau, b) q(b) != claim", []
        tr.absorb(f"zc/q/{t}", M)
        r = tr.challenge(f"zc/r/{t}", v)
        mz.bind(t, r, taus)
        for j in range(v):
            E = fc_mul(E, _eq1(taus[j], r[j]))
        claim = fc_mul(E, eval_mixed(M, r))
        rs += list(r)
        nrem -= v
    fabc, zk_t = proof.final_abc.to(dev), proof.zk_t.to(dev)
    prod = (fc_mul(fabc[0], fabc[1]) - fabc[2]) % P
    if not torch.equal(fc_mul(E, (mle_eval_ext(prod, eq_table(tau[:vf])) + zk_t) % P), claim):
        return False, "zero-check final: sum_b eq(tau, b) (Az Bz - Cz)(b) + T != claim", []
    tr.absorb("zc/final", torch.cat((fabc.reshape(-1, D), zk_t[None]), 0))
    ch = tr.challenge("zc/final", vf + 3)
    gamma, g3, beta = ch[vf], ch[vf + 1], ch[vf + 2]
    rs += list(ch[:vf])
    r_var = torch.stack(rs[::-1])
    r_c, r_k = r_var[:n_c], r_var[n_c:]
    eqf = eq_table(r_var[:vf])
    abc = [mle_eval_ext(fabc[s], eqf) for s in range(3)]
    # --- combined ------------------------------------------------------------------------------------------------------
    gp = fc_pow_seq(gamma, 3)
    rep_i, rep_c = 1 << (n_cmb - n_i), 1 << (n_cmb - n_c)
    rows_claim = (abc[0] + fc_mul(gp[1], abc[1]) + fc_mul(gp[2], abc[2])) % P
    claim = fc_mul(fc_mul(beta, rows_claim), ext_of_int(rep_i * rep_c, dev))
    rs2: list = []
    nrem = n_cmb
    for t, v in enumerate(sch.cmb):
        M = proof.cmb_rounds[t].to(dev)
        if not torch.equal(bool_sum_mixed(M), claim):
            return False, f"combined round {t}: sum_b q(b) != claim", []
        tr.absorb(f"cmb/q/{t}", M)
        r = tr.challenge(f"cmb/r/{t}", v)
        mc.bind(t, r)
        claim = eval_mixed(M, r)
        rs2 += list(r)
    rho = torch.stack(rs2[::-1])
    r_i, rho_c = rho[:n_i], rho[:n_c]
    n_w, s_w = zk_row_weights(lay, g3, dev)
    eq_i = eq_table(r_i)
    coef_r = (fc_mul(beta, mle_eval_ext(combined_coefficients(cons, lay, r_k, gamma), eq_i)) + mle_eval_ext(n_w, eq_i)) % P
    values = proof.values.to(dev)
    z_r = values[0].clone()
    pub_names = list(lay.virt)
    pub_vals = matvec_base_ext(torch.stack([pub_rows[nm].to(dev) for nm in pub_names]), eq_table(r_c))
    for name, pv in zip(pub_names, pub_vals):
        z_r = (z_r + fc_mul(eq_i[lay.virt[name]], pv)) % P
    succ = succ_mle(rho_c, r_c, lay.n_j)
    lhs = (fc_mul(fc_mul(coef_r, z_r), ext_of_int(rep_c, dev)) - fc_mul(fc_mul(succ, values[1]), ext_of_int(rep_i, dev))
           + values[2]) % P
    if not torch.equal(lhs, claim):
        return False, "combined final: (beta coef + n_w)(r_i) z(r_i, r_c) - succ(rho, r_c) V'(rho) + mask != claim", []
    tr.absorb("values", values)
    # --- row reduction -------------------------------------------------------------------------------------------------
    claim = values[1]
    rs3: list = []
    for t, v in enumerate(sch.rb):
        M = proof.rb_rounds[t].to(dev)
        if not torch.equal(bool_sum_mixed(M), claim):
            return False, f"row-reduction round {t}: sum_b q(b) != claim", []
        tr.absorb(f"rb/q/{t}", M)
        r = tr.challenge(f"rb/r/{t}", v)
        mr.bind(t, r)
        claim = eval_mixed(M, r)
        rs3 += list(r)
    rho_i = torch.stack(rs3[::-1])
    vb = proof.values_b.to(dev)
    if not torch.equal((fc_mul(mle_eval_ext(s_w, eq_table(rho_i)), vb[0]) + vb[1]) % P, claim):
        return False, "row-reduction final: s(rho_i) w(rho_i, rho_c) + mask != claim", []
    tr.absorb("values_b", vb)
    claims = [EvalClaim(torch.cat((r_i, r_c)), values[0], "w(r_i, r_c)"),
              EvalClaim(torch.cat((rho_i, rho_c)), vb[0], "w(rho_i, rho_c)"),
              mz.claim(lay, zk_t, "mask(zc)"), mc.claim(lay, values[2], "mask(cmb)"), mr.claim(lay, vb[1], "mask(rb)")]
    return True, "accept (LGSC0004 sumchecks; PCS claims pending)", claims


# ---------------------------------------------------------------------------------------------------------------------
# Witnesses (Ligero code produces them; nothing of its proof runs)
# ---------------------------------------------------------------------------------------------------------------------

def ligero_subbatch_iter(rel, runner, vus: list, l: int, S: int, device):
    """Yield the ``S`` sub-batches of a batch one at a time, exactly as Ligero's prover materialises them: ``vus`` (the
    ``relchain.instances`` tuples) split into sub-batches of ``l`` columns (``floor(l / steps)`` VUs each), padded with
    0-VU sub-batches (all pad units).  ``runner.marshal`` (public words + pub vectors), ``rel.hints``, ``witness.build_witness``."""
    from archive.direct.ligero.witness import build_witness
    per = l // rel.steps
    groups = [vus[i:i + per] for i in range(0, len(vus), per)]
    if len(groups) > S:
        raise ValueError(f"{len(vus)} VUs need {len(groups)} sub-batches > S = {S}")
    for g in groups + [[]] * (S - len(groups)):
        bd = runner.marshal(list(g), min_l=l) if g else _marshal_pad(rel, runner, l, device)
        assert bd["lay"].l == l, (bd["lay"].l, l)
        hints = rel.hints(rel.params, runner.sys, bd["c_d"], bd["a_d"], bd["b_d"], bd["y_d"], device)
        W = build_witness(runner.sys, bd["pub"], hints, device)
        yield SubBatchT(W, bd["pub"], bd["ypub_d"], len(g))


def ligero_subbatches(rel, runner, vus: list, l: int, S: int, device) -> tuple[list, list]:
    """All ``S`` sub-batches at once (small batches / tests): ``(subs, n_vus)``."""
    subs = list(ligero_subbatch_iter(rel, runner, vus, l, S, device))
    return subs, [s.n_vus for s in subs]


def _marshal_pad(rel, runner, l: int, device) -> dict:
    """A 0-VU sub-batch: ``l`` pad units (``relchain.RelationChainRunner._pad_unit``), ``ypub = 0`` -- what ``marshal``
    would build if it accepted an empty VU list."""
    import numpy as np
    from archive.direct.ligero.chain import Layout as ChainLayout
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
    w[lay.m:lay.m + len(lay.virt)] = 0
    for cl in claims:
        if isinstance(cl, SparseClaim):
            cells = w[cl.rows.to(w.device), cl.cols.to(w.device)].to(torch.int64)
            if not torch.equal(fc_scalar_mul(cells, cl.weights.to(w.device)).sum(0) % P, cl.value.to(w.device)):
                return False
            continue
        eq_i = eq_table(cl.point[:lay.n_i])
        eq_c = eq_table(cl.point[lay.n_i:])
        v = mle_eval_ext(matvec_base_ext(w, eq_c), eq_i)
        if not torch.equal(v, cl.value.to(v.device)):
            return False
    return True


# ---------------------------------------------------------------------------------------------------------------------
# CUDA engine
# ---------------------------------------------------------------------------------------------------------------------
# Arithmetic: Montgomery form (R = 2^32) for the extension tables, weights and challenges; ``z`` and the constraint
# coefficients' products are normal form.  Extension tables are 6 int32 planes with an explicit plane stride ``ps``; a
# table of ``n`` cells occupies the first ``n`` entries of each plane (folds are in place).  Corner ``b`` of a ``v``-round
# group ``x`` sits at ``x + b * nq`` (``nq = n >> v``, ``b = sum_j X_{j+1} 2^(v-1-j)``: ``X_1`` = the top bit).
#
# zc_open3_msg / zc_open3_fold: the opening round over the top 3 ``k`` bits straight from ``z`` via the CSR constraint
#   matrices (no ``Az/Bz/Cz`` base tables: 12.9 GB and a 12.9 GB write + 2 reads saved at 2^30); the message kernel runs
#   one block per (``k'``, ``a_1`` slice), 9 of the 27 entries per thread, the ``eq`` row weight applied once per block.
# zc_msg2: a bivariate round on ext tables (the O(N) rounds), 54 accumulators per thread, block-reduced.
# zc_msg3: the trivariate analogue on big tables, X1 split over blockIdx.y.
# zc_red / pp_red: any arity <= 6, one block row per entry (arity >= 4 only on small tables).
# zc_small / pp_small: any arity <= 3 on small tables: every group writes its 3^v entries, torch sums.
# zc_fold: in-place fold of up to 4 tables by the 2^v (v <= 6) eq weights of the round's challenges.

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
#define OPEN_CPT 8
#define OPEN_T 128
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

// v < 2^39 -> v mod p: two folds by 2^32 = R1 (mod p), then at most 4p -> [0, p)
__device__ __forceinline__ u32 red39(u64 v) {
    u64 x = (v >> 32) * (u64)R1 + (v & 0xffffffffULL);
    x = (x >> 32) * (u64)R1 + (x & 0xffffffffULL);
    if (x >= 2 * P64) x -= 2 * P64;
    if (x >= 2 * P64) x -= 2 * P64;
    if (x >= P64) x -= P64;
    return (u32)x;
}

// ext product (x^6 = 31): 36 mont + 6 u64 reductions; mont(a, b_m) = a b, mont(a_m, b_m) = (a b)_m
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
        out[c] = red39(v);
    }
}

__device__ __forceinline__ u32 mix1(u32 v0, u32 v1, int a) { return a == 0 ? v0 : (a == 1 ? v1 : submod(v1, v0)); }

// block reduction of NQ u64 per thread (values < 2^32) -> atomicAdd into acc[0..NQ); blockDim = 256
template <int NQ>
__device__ __forceinline__ void block_reduce_add(const u32* s, u64* acc) {
    __shared__ u64 sh[NQ][8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    #pragma unroll
    for (int q = 0; q < NQ; q++) {
        u64 v = s[q];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) sh[q][warp] = v;
    }
    __syncthreads();
    if (threadIdx.x < NQ) {
        u64 t = 0;
        for (int wv = 0; wv < 8; wv++) t += sh[threadIdx.x][wv];
        atomicAdd(&acc[threadIdx.x], t % P64);
    }
}

// corner value of a constraint table at slot k, column c: sum_e val_m[e] z[row[e], c] (val in Montgomery form -> normal)
__device__ __forceinline__ u32 gather(const int* __restrict__ z, const int* __restrict__ ptr, const int* __restrict__ row,
                                      const u32* __restrict__ val, int k, long long C, long long c) {
    u32 acc = 0;
    const int e1 = __ldg(ptr + k + 1);
    for (int e = __ldg(ptr + k); e < e1; e++) acc = addmod(acc, mont(__ldg(val + e), (u32)__ldg(z + (long long)__ldg(row + e) * C + c)));
    return acc;
}

// ---- opening round: the top 3 k bits, straight from z.  Slot k = kk + K8 * j, j = 4 X1 + 2 X2 + X3.
// block (x: column chunk, y: kk * 3 + a1); each thread: the 9 entries M[a1][a2][a3] of its columns, weighted by wc[c]
// (normal-form sums), times wr[kk] at the end; output normal form.
extern "C" __global__ void __launch_bounds__(OPEN_T) zc_open3_msg(
    const int* __restrict__ z,
    const int* __restrict__ pa, const int* __restrict__ ra, const u32* __restrict__ va,
    const int* __restrict__ pb, const int* __restrict__ rb, const u32* __restrict__ vb,
    const int* __restrict__ pc, const int* __restrict__ rc, const u32* __restrict__ vc,
    const u32* __restrict__ wr, const u32* __restrict__ wc, u64* acc, long long K8, long long C)
{
    // blockIdx.x = kk * 3 + a1 (fast), blockIdx.y = stripe of OPEN_CPT * OPEN_T columns (consecutive blocks share it in L2);
    // the 54 per-thread accumulators live in shared memory (registers: occupancy)
    const int kk = blockIdx.x / 3, a1 = blockIdx.x - 3 * (blockIdx.x / 3);
    __shared__ u32 sacc[54][OPEN_T];
    #pragma unroll
    for (int q = 0; q < 54; q++) sacc[q][threadIdx.x] = 0;
    const long long c0 = (long long)blockIdx.y * OPEN_CPT * OPEN_T + threadIdx.x;
    for (int it = 0; it < OPEN_CPT; it++) {
        const long long c = c0 + (long long)it * OPEN_T;
        if (c >= C) break;
        u32 A4[4], B4[4], C4[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int k0 = kk + (int)K8 * j, k1 = kk + (int)K8 * (4 + j);
            if (a1 == 0) {
                A4[j] = gather(z, pa, ra, va, k0, C, c); B4[j] = gather(z, pb, rb, vb, k0, C, c); C4[j] = gather(z, pc, rc, vc, k0, C, c);
            } else if (a1 == 1) {
                A4[j] = gather(z, pa, ra, va, k1, C, c); B4[j] = gather(z, pb, rb, vb, k1, C, c); C4[j] = gather(z, pc, rc, vc, k1, C, c);
            } else {
                A4[j] = submod(gather(z, pa, ra, va, k1, C, c), gather(z, pa, ra, va, k0, C, c));
                B4[j] = submod(gather(z, pb, rb, vb, k1, C, c), gather(z, pb, rb, vb, k0, C, c));
                C4[j] = 0;
            }
        }
        u32 w[6];
        #pragma unroll
        for (int p = 0; p < 6; p++) w[p] = __ldg(wc + (long long)p * C + c);
        #pragma unroll
        for (int a2 = 0; a2 < 3; a2++) {
            #pragma unroll
            for (int a3 = 0; a3 < 3; a3++) {
                const u32 ah = mix1(mix1(A4[0], A4[1], a3), mix1(A4[2], A4[3], a3), a2);
                const u32 bh = mix1(mix1(B4[0], B4[1], a3), mix1(B4[2], B4[3], a3), a2);
                u32 f = mont(to_mont(ah), bh);
                if (a2 < 2 && a3 < 2) f = submod(f, mix1(mix1(C4[0], C4[1], a3), mix1(C4[2], C4[3], a3), a2));
                #pragma unroll
                for (int p = 0; p < 6; p++) {
                    u32& sl = sacc[(a2 * 3 + a3) * 6 + p][threadIdx.x];
                    sl = addmod(sl, mont(w[p], f));
                }
            }
        }
    }
    __syncthreads();
    // column sums of sacc (normal form) -> times wr[kk] -> atomics
    __shared__ u32 tot[54];
    if (threadIdx.x < 54) {
        u64 t = 0;
        for (int i = 0; i < OPEN_T; i++) t += sacc[threadIdx.x][i];
        tot[threadIdx.x] = (u32)(t % P64);
    }
    __syncthreads();
    if (threadIdx.x < 9) {
        u32 wrr[6], sv[6], o[6];
        #pragma unroll
        for (int p = 0; p < 6; p++) { wrr[p] = __ldg(wr + (long long)p * K8 + kk); sv[p] = tot[threadIdx.x * 6 + p]; }
        ext_mul(sv, wrr, o);
        #pragma unroll
        for (int p = 0; p < 6; p++) atomicAdd(&acc[54 * a1 + threadIdx.x * 6 + p], (u64)o[p]);
    }
}

// opening fold: T(r)[kk, c] = sum_j eq_j T[kk + K8 j, c]; eqr (8, 6) = eq(r, j) R^2 so the output is Montgomery form.
extern "C" __global__ void zc_open3_fold(
    const int* __restrict__ z,
    const int* __restrict__ pa, const int* __restrict__ ra, const u32* __restrict__ va,
    const int* __restrict__ pb, const int* __restrict__ rb, const u32* __restrict__ vb,
    const int* __restrict__ pc, const int* __restrict__ rc, const u32* __restrict__ vc,
    const u32* __restrict__ eqr, int* __restrict__ out, long long K8, long long C, long long ps)
{
    __shared__ u32 eqs[48];
    if (threadIdx.x < 48) eqs[threadIdx.x] = eqr[threadIdx.x];
    __syncthreads();
    // blockIdx.x = kk (fast), blockIdx.y = 256-column stripe: consecutive blocks reuse the stripe of z from L2
    const int kk = blockIdx.x;
    {
        const long long c = (long long)blockIdx.y * blockDim.x + threadIdx.x;
        if (c >= C) return;
        const long long idx = (long long)kk * C + c;
        #pragma unroll
        for (int s = 0; s < 3; s++) {
            const int* ptr = s == 0 ? pa : (s == 1 ? pb : pc);
            const int* row = s == 0 ? ra : (s == 1 ? rb : rc);
            const u32* val = s == 0 ? va : (s == 1 ? vb : vc);
            u32 o[6] = {0, 0, 0, 0, 0, 0};
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const u32 v = gather(z, ptr, row, val, kk + (int)K8 * j, C, c);
                #pragma unroll
                for (int p = 0; p < 6; p++) o[p] = addmod(o[p], mont(eqs[j * 6 + p], v));
            }
            #pragma unroll
            for (int p = 0; p < 6; p++) out[(long long)(s * 6 + p) * ps + idx] = (int)o[p];
        }
    }
}

// ---- bivariate round on ext tables (Montgomery).  Group x < nq: corners x + b nq, b = 2 X1 + X2.  Weight of x:
// wr[x >> cb] wc[x & (2^cb - 1)].  Output (3, 3, 6) Montgomery.
extern "C" __global__ void zc_msg2(
    const int* __restrict__ A, const int* __restrict__ B, const int* __restrict__ Cm,
    const u32* __restrict__ wr, const u32* __restrict__ wc, u64* acc,
    long long nq, int cb, long long nwr, long long nwc, long long ps)
{
    u32 s[54];
    #pragma unroll
    for (int q = 0; q < 54; q++) s[q] = 0;
    const long long cmask = (1LL << cb) - 1;
    for (long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x; x < nq; x += (long long)gridDim.x * blockDim.x) {
        const long long k2 = x >> cb, c2 = x & cmask;
        u32 wa[6], wb[6], w[6];
        #pragma unroll
        for (int p = 0; p < 6; p++) { wa[p] = __ldg(wr + p * nwr + k2); wb[p] = __ldg(wc + p * nwc + c2); }
        ext_mul(wa, wb, w);
        u32 a[4][6], b[4][6], c[4][6];
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            #pragma unroll
            for (int p = 0; p < 6; p++) {
                const long long o = p * ps + x + q * nq;
                a[q][p] = (u32)A[o]; b[q][p] = (u32)B[o]; c[q][p] = (u32)Cm[o];
            }
        }
        #pragma unroll
        for (int a1 = 0; a1 < 3; a1++) {
            #pragma unroll
            for (int a2 = 0; a2 < 3; a2++) {
                u32 ah[6], bh[6], f[6], t[6];
                #pragma unroll
                for (int p = 0; p < 6; p++) {
                    ah[p] = mix1(mix1(a[0][p], a[1][p], a2), mix1(a[2][p], a[3][p], a2), a1);
                    bh[p] = mix1(mix1(b[0][p], b[1][p], a2), mix1(b[2][p], b[3][p], a2), a1);
                }
                ext_mul(ah, bh, f);
                if (a1 < 2 && a2 < 2) {
                    #pragma unroll
                    for (int p = 0; p < 6; p++) f[p] = submod(f[p], mix1(mix1(c[0][p], c[1][p], a2), mix1(c[2][p], c[3][p], a2), a1));
                }
                ext_mul(w, f, t);
                #pragma unroll
                for (int p = 0; p < 6; p++) s[(a1 * 3 + a2) * 6 + p] = addmod(s[(a1 * 3 + a2) * 6 + p], t[p]);
            }
        }
    }
    block_reduce_add<54>(s, acc);
}

// ---- trivariate round on big ext tables: blockIdx.y = the X1 entry a1 (0, 1, inf); a thread takes the X1 slice of its
// group's 8 corners (inf: T(X1=1) - T(X1=0)) and forms the 9 (X2, X3) entries as zc_msg2 does.  Corners x + b nq,
// b = 4 X1 + 2 X2 + X3.  Output (3, 3, 3, 6) Montgomery.  (zc_red3 re-reads the corners and re-forms w for each of the 27.)
extern "C" __global__ void zc_msg3(
    const int* __restrict__ A, const int* __restrict__ B, const int* __restrict__ Cm,
    const u32* __restrict__ wr, const u32* __restrict__ wc, u64* acc,
    long long nq, int cb, long long nwr, long long nwc, long long ps)
{
    const int a1 = blockIdx.y;
    u32 s[54];
    #pragma unroll
    for (int q = 0; q < 54; q++) s[q] = 0;
    const long long cmask = (1LL << cb) - 1;
    for (long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x; x < nq; x += (long long)gridDim.x * blockDim.x) {
        const long long k2 = x >> cb, c2 = x & cmask;
        u32 wa[6], wb[6], w[6];
        #pragma unroll
        for (int p = 0; p < 6; p++) { wa[p] = __ldg(wr + p * nwr + k2); wb[p] = __ldg(wc + p * nwc + c2); }
        ext_mul(wa, wb, w);
        u32 a[4][6], b[4][6], c[4][6];
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            #pragma unroll
            for (int p = 0; p < 6; p++) {
                const long long o0 = p * ps + x + q * nq, o1 = o0 + 4 * nq;
                if (a1 == 0) {
                    a[q][p] = (u32)A[o0]; b[q][p] = (u32)B[o0]; c[q][p] = (u32)Cm[o0];
                } else if (a1 == 1) {
                    a[q][p] = (u32)A[o1]; b[q][p] = (u32)B[o1]; c[q][p] = (u32)Cm[o1];
                } else {
                    a[q][p] = submod((u32)A[o1], (u32)A[o0]); b[q][p] = submod((u32)B[o1], (u32)B[o0]); c[q][p] = 0;
                }
            }
        }
        #pragma unroll
        for (int a2 = 0; a2 < 3; a2++) {
            #pragma unroll
            for (int a3 = 0; a3 < 3; a3++) {
                u32 ah[6], bh[6], f[6], t[6];
                #pragma unroll
                for (int p = 0; p < 6; p++) {
                    ah[p] = mix1(mix1(a[0][p], a[1][p], a3), mix1(a[2][p], a[3][p], a3), a2);
                    bh[p] = mix1(mix1(b[0][p], b[1][p], a3), mix1(b[2][p], b[3][p], a3), a2);
                }
                ext_mul(ah, bh, f);
                if (a1 < 2 && a2 < 2 && a3 < 2) {
                    #pragma unroll
                    for (int p = 0; p < 6; p++) f[p] = submod(f[p], mix1(mix1(c[0][p], c[1][p], a3), mix1(c[2][p], c[3][p], a3), a2));
                }
                ext_mul(w, f, t);
                #pragma unroll
                for (int p = 0; p < 6; p++) s[(a2 * 3 + a3) * 6 + p] = addmod(s[(a2 * 3 + a3) * 6 + p], t[p]);
            }
        }
    }
    block_reduce_add<54>(s, acc + 54 * a1);
}

// ---- arity-V round (V = 4..6) on big ext tables: blockIdx.x = the entry et of the top V-2 axes (the 3^(V-2) slices of an
// x-chunk run side by side and share its corners in L2), blockIdx.y strides the groups.  A thread forms the et-slice of its
// group's 4 low (X_{V-1}, X_V) corners (an inf axis contributes T(1) - T(0)) and the 9 low entries as zc_msg2 does:
// 4^(V-1) loads per group per table (zc_red: 4^V) and one weight per slice (zc_red: per entry).  Output (3^V, 6)
// Montgomery, entry et * 9 + 3 a_{V-1} + a_V.
template <int V>
__device__ void zc_msgv_body(const int* __restrict__ A, const int* __restrict__ B, const int* __restrict__ Cm,
                             const u32* __restrict__ wr, const u32* __restrict__ wc, u64* acc,
                             long long nq, int cb, long long nwr, long long nwc, long long ps)
{
    constexpr int TA = V - 2;
    const int et = blockIdx.x;
    int base = 0, ninf = 0, infbit[TA];
    {
        int ee = et;
        #pragma unroll
        for (int j = TA - 1; j >= 0; j--) {
            const int aj = ee % 3;
            ee /= 3;
            if (aj == 2) infbit[ninf++] = 1 << (TA - 1 - j); else base |= aj << (TA - 1 - j);
        }
    }
    u32 s[54];
    #pragma unroll
    for (int q = 0; q < 54; q++) s[q] = 0;
    const long long cmask = (1LL << cb) - 1;
    for (long long x = (long long)blockIdx.y * blockDim.x + threadIdx.x; x < nq; x += (long long)gridDim.y * blockDim.x) {
        const long long k2 = x >> cb, c2 = x & cmask;
        u32 wa[6], wb[6], w[6];
        #pragma unroll
        for (int p = 0; p < 6; p++) { wa[p] = __ldg(wr + p * nwr + k2); wb[p] = __ldg(wc + p * nwc + c2); }
        ext_mul(wa, wb, w);
        u32 a[4][6], b[4][6], c[4][6];
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            #pragma unroll
            for (int p = 0; p < 6; p++) { a[q][p] = 0; b[q][p] = 0; c[q][p] = 0; }
        }
        for (int m = 0; m < (1 << ninf); m++) {
            int bt = base, neg = 0;
            for (int u = 0; u < ninf; u++) { if ((m >> u) & 1) bt |= infbit[u]; else neg ^= 1; }
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                #pragma unroll
                for (int p = 0; p < 6; p++) {
                    const long long o = p * ps + x + (long long)(bt * 4 + q) * nq;
                    const u32 va = (u32)A[o], vb = (u32)B[o];
                    a[q][p] = neg ? submod(a[q][p], va) : addmod(a[q][p], va);
                    b[q][p] = neg ? submod(b[q][p], vb) : addmod(b[q][p], vb);
                    if (ninf == 0) c[q][p] = (u32)Cm[o];
                }
            }
        }
        #pragma unroll
        for (int a1 = 0; a1 < 3; a1++) {
            #pragma unroll
            for (int a2 = 0; a2 < 3; a2++) {
                u32 ah[6], bh[6], f[6], t[6];
                #pragma unroll
                for (int p = 0; p < 6; p++) {
                    ah[p] = mix1(mix1(a[0][p], a[1][p], a2), mix1(a[2][p], a[3][p], a2), a1);
                    bh[p] = mix1(mix1(b[0][p], b[1][p], a2), mix1(b[2][p], b[3][p], a2), a1);
                }
                ext_mul(ah, bh, f);
                if (ninf == 0 && a1 < 2 && a2 < 2) {
                    #pragma unroll
                    for (int p = 0; p < 6; p++) f[p] = submod(f[p], mix1(mix1(c[0][p], c[1][p], a2), mix1(c[2][p], c[3][p], a2), a1));
                }
                ext_mul(w, f, t);
                #pragma unroll
                for (int p = 0; p < 6; p++) s[(a1 * 3 + a2) * 6 + p] = addmod(s[(a1 * 3 + a2) * 6 + p], t[p]);
            }
        }
    }
    block_reduce_add<54>(s, acc + 54 * et);
}
extern "C" __global__ void zc_msgv4(const int* A, const int* B, const int* Cm, const u32* wr, const u32* wc, u64* acc, long long nq, int cb, long long nwr, long long nwc, long long ps) { zc_msgv_body<4>(A, B, Cm, wr, wc, acc, nq, cb, nwr, nwc, ps); }
extern "C" __global__ void zc_msgv5(const int* A, const int* B, const int* Cm, const u32* wr, const u32* wc, u64* acc, long long nq, int cb, long long nwr, long long nwc, long long ps) { zc_msgv_body<5>(A, B, Cm, wr, wc, acc, nq, cb, nwr, nwc, ps); }
extern "C" __global__ void zc_msgv6(const int* A, const int* B, const int* Cm, const u32* wr, const u32* wc, u64* acc, long long nq, int cb, long long nwr, long long nwc, long long ps) { zc_msgv_body<6>(A, B, Cm, wr, wc, acc, nq, cb, nwr, nwc, ps); }

// mixed-representation entry e (sum_j a_j 3^(v-1-j)) of a table's 2^v corners at group x (ext, Montgomery)
template <int V>
__device__ __forceinline__ void mixed_at(const int* __restrict__ T, long long ps, long long x, long long nq, int e, u32* out) {
    int a[V];
    int ee = e;
    #pragma unroll
    for (int j = V - 1; j >= 0; j--) { a[j] = ee % 3; ee /= 3; }
    int base = 0, ninf = 0, infax[V];
    #pragma unroll
    for (int j = 0; j < V; j++) { if (a[j] == 2) infax[ninf++] = j; else base |= a[j] << (V - 1 - j); }
    #pragma unroll
    for (int p = 0; p < 6; p++) out[p] = 0;
    for (int m = 0; m < (1 << ninf); m++) {
        int b = base, neg = 0;
        for (int u = 0; u < ninf; u++) { if ((m >> u) & 1) b |= 1 << (V - 1 - infax[u]); else neg ^= 1; }
        #pragma unroll
        for (int p = 0; p < 6; p++) {
            const u32 v = (u32)T[p * ps + x + b * nq];
            out[p] = neg ? submod(out[p], v) : addmod(out[p], v);
        }
    }
}

template <int V>
__device__ __forceinline__ bool noinf(int e) {
    #pragma unroll
    for (int j = 0; j < V; j++) { if (e % 3 == 2) return false; e /= 3; }
    return true;
}

// small-table round of the zero-check: every group writes its 3^V weighted entries (Montgomery) to out (nq, 3^V, 6)
template <int V>
__device__ void zc_small_body(const int* __restrict__ A, const int* __restrict__ B, const int* __restrict__ Cm,
                              const u32* __restrict__ wr, const u32* __restrict__ wc, u32* __restrict__ out,
                              long long nq, int cb, long long nwr, long long nwc, long long ps)
{
    constexpr int NE = V == 1 ? 3 : (V == 2 ? 9 : 27);
    const long long cmask = (1LL << cb) - 1;
    for (long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x; x < nq; x += (long long)gridDim.x * blockDim.x) {
        u32 wa[6], wb[6], w[6];
        #pragma unroll
        for (int p = 0; p < 6; p++) { wa[p] = wr[p * nwr + (x >> cb)]; wb[p] = wc[p * nwc + (x & cmask)]; }
        ext_mul(wa, wb, w);
        for (int e = 0; e < NE; e++) {
            u32 ah[6], bh[6], ch[6], f[6], t[6];
            mixed_at<V>(A, ps, x, nq, e, ah);
            mixed_at<V>(B, ps, x, nq, e, bh);
            ext_mul(ah, bh, f);
            if (noinf<V>(e)) {
                mixed_at<V>(Cm, ps, x, nq, e, ch);
                #pragma unroll
                for (int p = 0; p < 6; p++) f[p] = submod(f[p], ch[p]);
            }
            ext_mul(w, f, t);
            #pragma unroll
            for (int p = 0; p < 6; p++) out[(x * NE + e) * 6 + p] = t[p];
        }
    }
}
// the same entries reduced in-kernel: grid (x-blocks, 3^V); acc (3^V, 6) u64, Montgomery after % p
template <int V>
__device__ void zc_red_body(const int* __restrict__ A, const int* __restrict__ B, const int* __restrict__ Cm,
                            const u32* __restrict__ wr, const u32* __restrict__ wc, u64* acc,
                            long long nq, int cb, long long nwr, long long nwc, long long ps)
{
    const int e = blockIdx.y;
    const bool ni = noinf<V>(e);
    const long long cmask = (1LL << cb) - 1;
    u32 s[6] = {0, 0, 0, 0, 0, 0};
    for (long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x; x < nq; x += (long long)gridDim.x * blockDim.x) {
        u32 wa[6], wb[6], w[6], ah[6], bh[6], ch[6], f[6], t[6];
        #pragma unroll
        for (int p = 0; p < 6; p++) { wa[p] = __ldg(wr + p * nwr + (x >> cb)); wb[p] = __ldg(wc + p * nwc + (x & cmask)); }
        ext_mul(wa, wb, w);
        mixed_at<V>(A, ps, x, nq, e, ah);
        mixed_at<V>(B, ps, x, nq, e, bh);
        ext_mul(ah, bh, f);
        if (ni) {
            mixed_at<V>(Cm, ps, x, nq, e, ch);
            #pragma unroll
            for (int p = 0; p < 6; p++) f[p] = submod(f[p], ch[p]);
        }
        ext_mul(w, f, t);
        #pragma unroll
        for (int p = 0; p < 6; p++) s[p] = addmod(s[p], t[p]);
    }
    block_reduce_add<6>(s, acc + 6 * e);
}
extern "C" __global__ void zc_red1(const int* A, const int* B, const int* Cm, const u32* wr, const u32* wc, u64* acc, long long nq, int cb, long long nwr, long long nwc, long long ps) { zc_red_body<1>(A, B, Cm, wr, wc, acc, nq, cb, nwr, nwc, ps); }
extern "C" __global__ void zc_red2(const int* A, const int* B, const int* Cm, const u32* wr, const u32* wc, u64* acc, long long nq, int cb, long long nwr, long long nwc, long long ps) { zc_red_body<2>(A, B, Cm, wr, wc, acc, nq, cb, nwr, nwc, ps); }
extern "C" __global__ void zc_red3(const int* A, const int* B, const int* Cm, const u32* wr, const u32* wc, u64* acc, long long nq, int cb, long long nwr, long long nwc, long long ps) { zc_red_body<3>(A, B, Cm, wr, wc, acc, nq, cb, nwr, nwc, ps); }
// arity 4..6 below 2^12 groups (zc_msgv above): the per-entry corner re-reads hit L2
extern "C" __global__ void zc_red4(const int* A, const int* B, const int* Cm, const u32* wr, const u32* wc, u64* acc, long long nq, int cb, long long nwr, long long nwc, long long ps) { zc_red_body<4>(A, B, Cm, wr, wc, acc, nq, cb, nwr, nwc, ps); }
extern "C" __global__ void zc_red5(const int* A, const int* B, const int* Cm, const u32* wr, const u32* wc, u64* acc, long long nq, int cb, long long nwr, long long nwc, long long ps) { zc_red_body<5>(A, B, Cm, wr, wc, acc, nq, cb, nwr, nwc, ps); }
extern "C" __global__ void zc_red6(const int* A, const int* B, const int* Cm, const u32* wr, const u32* wc, u64* acc, long long nq, int cb, long long nwr, long long nwc, long long ps) { zc_red_body<6>(A, B, Cm, wr, wc, acc, nq, cb, nwr, nwc, ps); }

template <int V>
__device__ void pp_red_body(const int* __restrict__ T, u64* acc, long long nq, long long ps)
{
    const int e = blockIdx.y;
    u32 s[6] = {0, 0, 0, 0, 0, 0};
    for (long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x; x < nq; x += (long long)gridDim.x * blockDim.x) {
        u32 u1[6], v1[6], u2[6], v2[6], f[6], g[6];
        mixed_at<V>(T, ps, x, nq, e, u1);
        mixed_at<V>(T + 6 * ps, ps, x, nq, e, v1);
        mixed_at<V>(T + 12 * ps, ps, x, nq, e, u2);
        mixed_at<V>(T + 18 * ps, ps, x, nq, e, v2);
        ext_mul(u1, v1, f);
        ext_mul(u2, v2, g);
        #pragma unroll
        for (int p = 0; p < 6; p++) s[p] = addmod(s[p], addmod(f[p], g[p]));
    }
    block_reduce_add<6>(s, acc + 6 * e);
}
extern "C" __global__ void pp_red1(const int* T, u64* acc, long long nq, long long ps) { pp_red_body<1>(T, acc, nq, ps); }
extern "C" __global__ void pp_red2(const int* T, u64* acc, long long nq, long long ps) { pp_red_body<2>(T, acc, nq, ps); }
extern "C" __global__ void pp_red3(const int* T, u64* acc, long long nq, long long ps) { pp_red_body<3>(T, acc, nq, ps); }
extern "C" __global__ void pp_red4(const int* T, u64* acc, long long nq, long long ps) { pp_red_body<4>(T, acc, nq, ps); }
extern "C" __global__ void pp_red5(const int* T, u64* acc, long long nq, long long ps) { pp_red_body<5>(T, acc, nq, ps); }
extern "C" __global__ void pp_red6(const int* T, u64* acc, long long nq, long long ps) { pp_red_body<6>(T, acc, nq, ps); }

extern "C" __global__ void zc_small1(const int* A, const int* B, const int* Cm, const u32* wr, const u32* wc, u32* out, long long nq, int cb, long long nwr, long long nwc, long long ps) { zc_small_body<1>(A, B, Cm, wr, wc, out, nq, cb, nwr, nwc, ps); }
extern "C" __global__ void zc_small2(const int* A, const int* B, const int* Cm, const u32* wr, const u32* wc, u32* out, long long nq, int cb, long long nwr, long long nwc, long long ps) { zc_small_body<2>(A, B, Cm, wr, wc, out, nq, cb, nwr, nwc, ps); }
extern "C" __global__ void zc_small3(const int* A, const int* B, const int* Cm, const u32* wr, const u32* wc, u32* out, long long nq, int cb, long long nwr, long long nwc, long long ps) { zc_small_body<3>(A, B, Cm, wr, wc, out, nq, cb, nwr, nwc, ps); }

// small-table round of the combined sumcheck: sum P1 Q1 + P2 Q2 (unweighted); tables in one (4, 6, ps) buffer
template <int V>
__device__ void pp_small_body(const int* __restrict__ T, u32* __restrict__ out, long long nq, long long ps)
{
    constexpr int NE = V == 1 ? 3 : (V == 2 ? 9 : 27);
    for (long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x; x < nq; x += (long long)gridDim.x * blockDim.x) {
        for (int e = 0; e < NE; e++) {
            u32 u1[6], v1[6], u2[6], v2[6], f[6], g[6];
            mixed_at<V>(T, ps, x, nq, e, u1);
            mixed_at<V>(T + 6 * ps, ps, x, nq, e, v1);
            mixed_at<V>(T + 12 * ps, ps, x, nq, e, u2);
            mixed_at<V>(T + 18 * ps, ps, x, nq, e, v2);
            ext_mul(u1, v1, f);
            ext_mul(u2, v2, g);
            #pragma unroll
            for (int p = 0; p < 6; p++) out[(x * NE + e) * 6 + p] = addmod(f[p], g[p]);
        }
    }
}
extern "C" __global__ void pp_small1(const int* T, u32* out, long long nq, long long ps) { pp_small_body<1>(T, out, nq, ps); }
extern "C" __global__ void pp_small2(const int* T, u32* out, long long nq, long long ps) { pp_small_body<2>(T, out, nq, ps); }
extern "C" __global__ void pp_small3(const int* T, u32* out, long long nq, long long ps) { pp_small_body<3>(T, out, nq, ps); }

// in-place fold of ntab tables (planes at T + t * tstride, plane stride ps) by the 2^v corner weights eqr (Montgomery)
extern "C" __global__ void zc_fold(int* T, int ntab, long long tstride, const u32* __restrict__ eqr, int v, long long nq, long long ps)
{
    __shared__ u32 eqs[64 * 6];
    for (int i = threadIdx.x; i < (6 << v); i += blockDim.x) eqs[i] = eqr[i];
    __syncthreads();
    for (long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x; x < nq; x += (long long)gridDim.x * blockDim.x) {
        for (int tb = 0; tb < ntab; tb++) {
            int* Tt = T + tb * tstride;
            u32 o[6] = {0, 0, 0, 0, 0, 0}, vv[6], t[6];
            for (int b = 0; b < (1 << v); b++) {
                #pragma unroll
                for (int p = 0; p < 6; p++) vv[p] = (u32)Tt[p * ps + x + b * nq];
                ext_mul(eqs + 6 * b, vv, t);
                #pragma unroll
                for (int p = 0; p < 6; p++) o[p] = addmod(o[p], t[p]);
            }
            #pragma unroll
            for (int p = 0; p < 6; p++) Tt[p * ps + x] = (int)o[p];
        }
    }
}

// coef[row] += eg[k] * val over the COO triples of all three sides (eg (3K, 6) Montgomery = eq(r_k, k) g^side; output normal)
extern "C" __global__ void sc_coef(const int* __restrict__ ks, const int* __restrict__ rs, const int* __restrict__ vs,
                                   const u32* __restrict__ eg, u64* coef, int nnz)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nnz) return;
    const int k = ks[e], r = rs[e];
    const u32 v = (u32)vs[e];
    #pragma unroll
    for (int p = 0; p < 6; p++) atomicAdd(&coef[r * 6 + p], (u64)mont(eg[k * 6 + p], v));
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
// base x ext matvec: out[i] = sum_c M[i, c] * e[c]   (e in Montgomery planes (6, C); out normal form, int64 (R, 6));
// MV_ROWS rows per block so every e load serves MV_ROWS rows
#define MV_ROWS 8
extern "C" __global__ void sc_matvec(
    const int* __restrict__ M, const u32* __restrict__ e, long long* __restrict__ out, long long R, long long C)
{
    __shared__ u64 sh[MV_ROWS * 6][8];
    const long long i0 = (long long)blockIdx.x * MV_ROWS;
    u32 acc[MV_ROWS][6];
    #pragma unroll
    for (int r = 0; r < MV_ROWS; r++) {
        #pragma unroll
        for (int p = 0; p < 6; p++) acc[r][p] = 0;
    }
    for (long long c = threadIdx.x; c < C; c += blockDim.x) {
        u32 ev[6];
        #pragma unroll
        for (int p = 0; p < 6; p++) ev[p] = __ldg(e + p * C + c);
        #pragma unroll
        for (int r = 0; r < MV_ROWS; r++) {
            if (i0 + r < R) {
                const u32 m = (u32)__ldg(M + (i0 + r) * C + c);
                #pragma unroll
                for (int p = 0; p < 6; p++) acc[r][p] = addmod(acc[r][p], mont(ev[p], m));
            }
        }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    #pragma unroll
    for (int q = 0; q < MV_ROWS * 6; q++) {
        u64 v = acc[q / 6][q % 6];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) sh[q][warp] = v;
    }
    __syncthreads();
    if (threadIdx.x < MV_ROWS * 6 && i0 + threadIdx.x / 6 < R) {
        u64 t = 0;
        for (int w = 0; w < 8; w++) t += sh[threadIdx.x][w];
        out[(i0 + threadIdx.x / 6) * 6 + threadIdx.x % 6] = (long long)(t % P64);
    }
}
"""

_MODULE: dict = {}


def _kernel(name: str):
    import cupy as cp
    if "mod" not in _MODULE:
        _MODULE["mod"] = cp.RawModule(code=_SRC, options=("-std=c++17",))
    return _MODULE["mod"].get_function(name)


def _cp(t: torch.Tensor):
    import cupy as cp
    return cp.from_dlpack(t.contiguous())


def _launch(n: int, threads: int = 256, max_blocks: int = 8192):
    return (int(min(max_blocks, max(1, (n + threads - 1) // threads))),), (threads,)


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


def _pymul(a: list, b: list) -> list:
    t = [0] * (2 * D - 1)
    for i in range(D):
        for j in range(D):
            t[i + j] += a[i] * b[j]
    return [(t[c] + 31 * t[c + D]) % P if c + D < 2 * D - 1 else t[c] % P for c in range(D)]


def _corner_eq(r: torch.Tensor) -> torch.Tensor:
    """``(2^v, 6)`` normal: ``eq(r, X(b))`` with ``b = sum_j X_{j+1} 2^(v-1-j)`` (``r[j]`` <-> ``X_{j+1}``); host arithmetic
    (the round's tiny weight table: a few hundred torch launches on the device otherwise)."""
    rs = [[int(x) % P for x in row] for row in r.tolist()]
    vec = [[1, 0, 0, 0, 0, 0]]
    for rj in reversed(rs):                     # each variable becomes the new top bit: X_1 last
        om = [(-x) % P for x in rj]
        om[0] = (om[0] + 1) % P
        vec = [_pymul(e, om) for e in vec] + [_pymul(e, rj) for e in vec]
    return torch.tensor(vec, dtype=torch.int64, device=r.device)


def _csr(cons: Constraints, dev) -> dict:
    """Per side: CSR over the constraint slots (int32 ptr (K+1), row, val in Montgomery form)."""
    cache = getattr(cons, "_csr_cache", None)
    if cache is not None and cache[0] == str(dev):
        return cache[1]
    out = {}
    K = cons.K_pad
    for side, (ks, rs, vs) in cons.coo.items():
        ks, rs, vs = ks.to(dev), rs.to(dev), vs.to(dev)
        order = torch.argsort(ks, stable=True)
        ptr = torch.zeros((K + 1,), dtype=torch.int64, device=dev)
        ptr[1:] = torch.cumsum(torch.bincount(ks, minlength=K), 0)
        out[side] = (ptr.to(torch.int32).contiguous(), rs[order].to(torch.int32).contiguous(), to_mont(vs[order]).to(torch.int32).contiguous())
    cons._csr_cache = (str(dev), out)
    return out


class CudaZC:
    """Zero-check state on the device.  Before the opening round the tables are never materialised (the opening kernels
    gather from ``z``); afterwards three ext tables in one ``(3, 6, ps)`` int32 buffer (Montgomery), folded in place.
    ``round(v)`` / ``fold(r)`` / ``final_tables()`` as ``_RefZC``."""

    THREADS = 256

    def __init__(self, cons: Constraints, z: torch.Tensor, tau: torch.Tensor, lay: Layout):
        self.dev = z.device
        self.z = z if (z.dtype == torch.int32 and z.is_contiguous()) else (z.to(torch.int64) % P).to(torch.int32).contiguous()
        self.C, self.K = z.shape[1], cons.K_pad
        self.n_c, self.n_k = lay.n_c, lay.n_k
        self.csr = _csr(cons, self.dev)
        self.tau, self.tau_r, self.tau_c = tau, tau[self.n_c:], tau[:self.n_c]
        self._planes: dict = {}
        self.T = None
        self.logn = self.n_k + self.n_c

    def _plane(self, which: str, t: int) -> torch.Tensor:
        """``eq_table(tau_part[:t])`` as Montgomery planes ``(6, 2^t)`` (built on first use)."""
        key = (which, t)
        if key not in self._planes:
            tau = self.tau_r if which == "r" else self.tau_c
            tab = eq_table_cuda(tau[:t]) if t else fc_one((1,), self.dev)
            self._planes[key] = mont_planes(tab)
        return self._planes[key]

    def _weights(self, log_after: int):
        cb = min(self.n_c, log_after)
        rb = log_after - cb
        return self._plane("r", rb), self._plane("c", cb), cb

    def _csr_args(self):
        return tuple(_cp(t) for s in "abc" for t in self.csr[s])

    def round(self, v: int) -> torch.Tensor:
        import cupy as cp
        if self.T is None:
            if v != 3:
                raise ValueError("CudaZC: the opening round binds 3 variables")
            K8 = self.K // 8
            acc = torch.zeros((54 * 3,), dtype=torch.int64, device=self.dev)
            wr, wc, _ = self._weights(self.logn - 3)
            assert wr.shape[1] == K8 and wc.shape[1] == self.C
            cols_per_block = 128 * 8                                               # OPEN_T * OPEN_CPT
            grid = (K8 * 3, -(-self.C // cols_per_block))
            _kernel("zc_open3_msg")(grid, (128,), (_cp(self.z),) + self._csr_args() + (
                _cp(wr), _cp(wc), _cp(acc), cp.int64(K8), cp.int64(self.C)))
            return (acc % P).reshape(3, 3, 3, D)
        log_after = self.logn - v
        wr, wc, cb = self._weights(log_after)
        nq = 1 << log_after
        ps = self.T.shape[2]
        A, B, Cm = self.T[0], self.T[1], self.T[2]
        if v == 2 and nq >= (1 << 12):
            acc = torch.zeros((54,), dtype=torch.int64, device=self.dev)
            grid, block = _launch(nq, self.THREADS, 4096)
            _kernel("zc_msg2")(grid, block, (_cp(A), _cp(B), _cp(Cm), _cp(wr), _cp(wc), _cp(acc), cp.int64(nq), cp.int32(cb),
                                             cp.int64(wr.shape[1]), cp.int64(wc.shape[1]), cp.int64(ps)))
            return from_mont(acc).reshape(3, 3, D)
        if v == 3 and nq >= (1 << 12):
            acc = torch.zeros((162,), dtype=torch.int64, device=self.dev)
            (gx,), block = _launch(nq, self.THREADS, 1536)
            _kernel("zc_msg3")((gx, 3), block, (_cp(A), _cp(B), _cp(Cm), _cp(wr), _cp(wc), _cp(acc), cp.int64(nq), cp.int32(cb),
                                                cp.int64(wr.shape[1]), cp.int64(wc.shape[1]), cp.int64(ps)))
            return from_mont(acc).reshape(3, 3, 3, D)
        if v > ZK_MAX_ARITY:
            raise ValueError(f"CudaZC: arity {v} > {ZK_MAX_ARITY}")
        ne = 3 ** v
        acc = torch.zeros((ne * D,), dtype=torch.int64, device=self.dev)
        if v >= 4 and nq >= (1 << 12):
            nt = 3 ** (v - 2)
            gy = int(min(-(-nq // self.THREADS), max(1, 8192 // nt)))
            _kernel(f"zc_msgv{v}")((nt, gy), (self.THREADS,), (_cp(A), _cp(B), _cp(Cm), _cp(wr), _cp(wc), _cp(acc), cp.int64(nq),
                                                                cp.int32(cb), cp.int64(wr.shape[1]), cp.int64(wc.shape[1]), cp.int64(ps)))
            return from_mont(acc % P).reshape((3,) * v + (D,))
        (gx,), block = _launch(nq, 256, max(1, 2048 // ne))
        _kernel(f"zc_red{v}")((gx, ne), block, (_cp(A), _cp(B), _cp(Cm), _cp(wr), _cp(wc), _cp(acc), cp.int64(nq), cp.int32(cb),
                                               cp.int64(wr.shape[1]), cp.int64(wc.shape[1]), cp.int64(ps)))
        return from_mont(acc % P).reshape((3,) * v + (D,))

    def fold(self, r: torch.Tensor) -> None:
        import cupy as cp
        v = r.shape[0]
        if self.T is None:
            K8 = self.K // 8
            n0 = K8 * self.C
            self.T = torch.empty((3, D, n0), dtype=torch.int32, device=self.dev)
            eqr = to_mont(to_mont(_corner_eq(r))).to(torch.int32).contiguous()          # eq * R^2: Montgomery output
            grid, block = (K8, -(-self.C // self.THREADS)), (self.THREADS,)
            _kernel("zc_open3_fold")(grid, block, (_cp(self.z),) + self._csr_args() + (
                _cp(eqr), _cp(self.T), cp.int64(K8), cp.int64(self.C), cp.int64(n0)))
            self.logn -= 3
            return
        nq = 1 << (self.logn - v)
        ps = self.T.shape[2]
        eqr = to_mont(_corner_eq(r)).to(torch.int32).contiguous()
        grid, block = _launch(nq, self.THREADS, 8192)
        _kernel("zc_fold")(grid, block, (_cp(self.T), cp.int32(3), cp.int64(D * ps), _cp(eqr), cp.int32(v), cp.int64(nq), cp.int64(ps)))
        self.logn -= v

    def final_tables(self) -> torch.Tensor:
        n = 1 << self.logn
        return from_mont(self.T[:, :, :n].permute(0, 2, 1))               # (3, n, 6)


class CudaPP:
    """Combined-sumcheck state on the device: ``P1, Q1, P2, Q2`` in one ``(4, 6, n)`` int32 buffer (Montgomery)."""

    def __init__(self, tabs: list):
        self.dev = tabs[0].device
        self.T = torch.stack([mont_planes(t) for t in tabs]).contiguous()
        self.logn = tabs[0].shape[0].bit_length() - 1

    def round(self, v: int) -> torch.Tensor:
        import cupy as cp
        nq = 1 << (self.logn - v)
        ne = 3 ** v
        acc = torch.zeros((ne * D,), dtype=torch.int64, device=self.dev)
        (gx,), block = _launch(nq, 256, max(1, 2048 // ne))
        _kernel(f"pp_red{v}")((gx, ne), block, (_cp(self.T), _cp(acc), cp.int64(nq), cp.int64(self.T.shape[2])))
        return from_mont(acc % P).reshape((3,) * v + (D,))

    def fold(self, r: torch.Tensor) -> None:
        import cupy as cp
        v = r.shape[0]
        nq = 1 << (self.logn - v)
        ps = self.T.shape[2]
        eqr = to_mont(_corner_eq(r)).to(torch.int32).contiguous()
        grid, block = _launch(nq)
        _kernel("zc_fold")(grid, block, (_cp(self.T), cp.int32(4), cp.int64(D * ps), _cp(eqr), cp.int32(v), cp.int64(nq), cp.int64(ps)))
        self.logn -= v


def combined_coefficients_cuda(cons: Constraints, lay: Layout, r_k: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """``combined_coefficients`` as one scatter kernel over the ``nnz`` COO triples (``eq(r_k)`` by ``eq_table_cuda``)."""
    import cupy as cp
    dev = r_k.device
    cache = getattr(cons, "_coo_cat", None)
    if cache is None or cache[0] != str(dev):
        K = cons.K_pad
        ks = torch.cat([cons.coo[s][0].to(dev) + i * K for i, s in enumerate("abc")]).to(torch.int32).contiguous()
        rs = torch.cat([cons.coo[s][1].to(dev) for s in "abc"]).to(torch.int32).contiguous()
        vs = torch.cat([cons.coo[s][2].to(dev) % P for s in "abc"]).to(torch.int32).contiguous()
        cons._coo_cat = cache = (str(dev), ks, rs, vs)
    _, ks, rs, vs = cache
    eq_k = eq_table_cuda(r_k)                                                  # (K, 6)
    gp = fc_pow_seq(gamma, 3)
    eg = mont_planes(torch.cat([fc_mul(eq_k, g) for g in gp])).t().contiguous()   # (3K, 6) Montgomery, row-major
    acc = torch.zeros((lay.R * D,), dtype=torch.int64, device=dev)
    nnz = ks.numel()
    _kernel("sc_coef")(((nnz + 255) // 256,), (256,), (_cp(ks), _cp(rs), _cp(vs), _cp(eg), _cp(acc), cp.int32(nnz)))
    return (acc % P).reshape(lay.R, D)


def eq_table_cuda(tau: torch.Tensor) -> torch.Tensor:
    """``eq_table`` as one kernel (``(2^n, 6)`` int64, normal form)."""
    import cupy as cp
    n = tau.shape[0]
    dev = tau.device
    one = fc_one((), dev)
    tm = mont_planes(tau % P)
    om = mont_planes((one - tau) % P)
    out = torch.empty((1 << n, D), dtype=torch.int64, device=dev)
    blocks = int(min(4096, ((1 << n) + 255) // 256))
    _kernel("sc_eq")((blocks,), (256,), (_cp(tm), _cp(om), _cp(out), cp.int32(n)))
    return out


def matvec_base_ext_cuda(M: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """``matvec_base_ext`` as one kernel: ``(R, C) int32 base x (C, 6) ext -> (R, 6)``."""
    import cupy as cp
    M = M.contiguous()
    if M.dtype != torch.int32:
        M = (M.to(torch.int64) % P).to(torch.int32)
    R, C = M.shape
    out = torch.empty((R, D), dtype=torch.int64, device=M.device)
    _kernel("sc_matvec")((-(-R // 8),), (256,), (_cp(M), _cp(mont_planes(e)), _cp(out), cp.int64(R), cp.int64(C)))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Measurement harness: ``python -m archive.direct.ligerito.sumcheck --relation fp8-ada --vus 4096 --engine cuda``
# ---------------------------------------------------------------------------------------------------------------------

def _schedule_arg(s: str | None, lay: Layout) -> Schedule | None:
    """``--schedule "3,2,2,2,2,3,3,3,3,3/4/3,3,3,3,3,3"`` (zc rounds / final / combined rounds [/ row-reduction rounds]) or a
    preset: ``default`` | ``c2`` | ``c3`` | ``lean`` (LGSC0003), ``zk`` (LGSC0004's default), ``zk-small`` (the LGSC0003
    default + rb 4,4,4: 21 coins, fewest bytes)."""
    n, n_cmb = lay.n_k + lay.n_c, n_cmb_of(lay)
    if s is None or s == "default":
        return default_schedule(n, n_cmb)
    if s == "c2":
        return default_schedule(n, n_cmb, small_arity=2, max_final=3)
    if s == "c3":
        return default_schedule(n, n_cmb, big_log2=0, big_arity=3)
    if s == "lean":
        return lean_schedule(n, n_cmb)
    if s == "zk":
        return zk_default_schedule(lay)
    if s == "zk-small":
        return zk_schedule(lay, default_schedule(n, n_cmb), rb_arity=4)
    parts = s.split("/")
    rb = tuple(int(a) for a in parts[3].split(",")) if len(parts) > 3 else ()
    return Schedule(tuple(int(a) for a in parts[0].split(",")), int(parts[1]), tuple(int(a) for a in parts[2].split(",")), rb)


def bench(relation_name: str, n_vus: int, l: int, S: int, device: str, engine: str | None, negatives: bool,
          seed: int = 7, schedule: str | None = None, reps: int = 2, zk: bool = False) -> dict:
    """Witnesses via the Ligero code (produce only; streamed into ``z`` one sub-batch at a time), the sumchecks, the
    verifier, bytes, coins, peak memory; optional negatives (violated quadratic row / chain link / lookup: the honest
    prover on the bad witness fails round 0, the cheating provers reach the combined sumcheck and mismatch)."""
    from archive.direct.ligero.relations import relation
    from archive.direct.ligero.relchain import RelationChainRunner, instances
    from .layout import constraints, fill_z, fill_zk, layout_for, public_rows
    dev = torch.device(device)
    out: dict = {"relation": relation_name, "vus": n_vus, "l": l, "S": S, "engine": engine or "ref", "zk": zk}
    if dev.type == "cuda":
        out["gpu"] = torch.cuda.get_device_name(dev)
        out["gpu_total_gb"] = torch.cuda.get_device_properties(dev).total_memory / 1e9
    t0 = time.perf_counter()
    rel = relation(relation_name)
    runner = RelationChainRunner(rel, device, -128.0)
    vus = instances(rel, n_vus, seed=seed)
    per = l // rel.steps
    nv = [len(vus[i:i + per]) for i in range(0, len(vus), per)]
    nv += [0] * (S - len(nv))
    lay = layout_for(runner.sys, l, nv, rel.steps, zk=zk)
    cons = constraints(runner.sys, lay, device)
    z = torch.zeros((lay.R, lay.C), dtype=torch.int32, device=dev)
    pub_subs = []
    for s, sb in enumerate(ligero_subbatch_iter(rel, runner, vus, l, S, device)):
        fill_z(runner.sys, lay, z, s, sb)
        pub_subs.append(SubBatchT(None, {k: v.cpu() for k, v in sb.pub.items()}, sb.ypub.cpu(), sb.n_vus))
        del sb
    _sync(dev)
    out["t_witness_and_z"] = time.perf_counter() - t0
    if zk:
        t1 = time.perf_counter()
        fill_zk(lay, z)
        _sync(dev)
        out["t_fill_zk"] = time.perf_counter() - t1
    sched = _schedule_arg(schedule, lay) if schedule is not None or not zk else None
    if zk:
        sched = zk_schedule(lay, sched)
    out.update(m=lay.m, R=lay.R, C=lay.C, K=lay.K, K_pad=cons.K_pad, n_k=lay.n_k, n_c=lay.n_c, n_i=lay.n_i,
               n_links=lay.n_links, nnz=cons.nnz(), n_sub=len([n for n in nv if n]), N_log2=lay.n_vars,
               schedule={"zc": list(sched.zc), "vf": sched.vf, "cmb": list(sched.cmb), "rb": list(sched.rb)})
    if zk:
        out["zk_rows"] = lay.summary()["zk"]
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        out["z_gb"] = z.numel() * 4 / 1e9
        out["mem_before_prove_gb"] = torch.cuda.memory_allocated(dev) / 1e9
    runs = []
    for _ in range(max(1, reps)):
        coins = LocalCoins(b"bench")
        proof = prove(lay, cons, z, coins, engine=engine, schedule=sched)
        runs.append(proof.timings)
    out["coins"] = coins.rounds
    out["timings_cold"] = runs[0]
    out["timings"] = min(runs[1:] or runs, key=lambda t: t["total"])
    out["totals"] = [t["total"] for t in runs]
    if dev.type == "cuda":
        out["peak_mem_gb"] = torch.cuda.max_memory_allocated(dev) / 1e9
        out["peak_reserved_gb"] = torch.cuda.max_memory_reserved(dev) / 1e9
    out["bytes"] = proof.n_bytes()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    t2 = time.perf_counter()
    pubs = public_rows(runner.sys, lay, pub_subs, device)
    ok, why, claims = verify(lay, cons, pubs, proof, LocalCoins(b"bench"), device=device)
    _sync(dev)
    out["t_verify"] = time.perf_counter() - t2
    out["verify_ok"], out["verify_why"] = ok, why
    out["n_claims"] = len(claims)
    if zk and ok:
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        out["claims_ok"] = check_claims(lay, z, claims)
    if negatives:
        neg = {}
        sys = runner.sys
        bit = next(r for r in sys.rows if r.kind == "bit")
        sel = [r for r in sys.rows if r.kind == "sel"]
        cases = {
            "quadratic": [(lay.row_of_sys[bit.idx], 3, 2)],
            "chain_link": [(lay.c_rows[0], 5, None), (lay.next_row(0), 4, "copy")],
            "lookup": [(lay.row_of_sys[sel[0].idx], 7, 1), (lay.row_of_sys[sel[1].idx], 7, 1)],
        }
        if zk:
            cases["zk_product"] = [(lay.zk.product[2], 9, None)]
            cases["zk_mask_link"] = [(lay.zk.mask_next, 0, None)]
        for name, edits in cases.items():
            saved = [(row, col, int(z[row, col])) for row, col, _ in edits]
            for row, col, val in edits:
                if val is None:
                    z[row, col] = (int(z[row, col]) + 1) % P
                elif val == "copy":
                    z[row, col] = z[lay.c_rows[0], col + 1]
                else:
                    z[row, col] = val
            res = {}
            for cheat in (None, "zc", "all"):
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                pr = prove(lay, cons, z, LocalCoins(b"neg"), cheat=cheat, engine=engine, schedule=sched)
                ok_n, why_n, _ = verify(lay, cons, pubs, pr, LocalCoins(b"neg"), device=device)
                res[cheat or "honest"] = (ok_n, why_n)
            for row, col, val in reversed(saved):
                z[row, col] = val
            neg[name] = res
        out["negatives"] = neg
    return out


class RecordingCoins:
    """Wraps a coins object and records every draw ``(label, values)`` and absorb ``(label, hex)`` in order (fixtures)."""

    def __init__(self, inner):
        self.inner, self.log = inner, []

    @property
    def rounds(self):
        return self.inner.rounds

    def absorb(self, label: bytes, data: bytes) -> None:
        self.log.append({"absorb": label.decode(), "bytes": data.hex()})
        self.inner.absorb(label, data)

    def challenge(self, label: bytes, n: int):
        v = self.inner.challenge(label, n)
        self.log.append({"challenge": label.decode(), "n": n, "values": [[int(x) for x in row] for row in v]})
        return v


def _claim_json(c, zk: bool) -> dict:
    if isinstance(c, SparseClaim):
        return {"name": c.name, "kind": "sparse", "rows": [int(x) for x in c.rows.tolist()], "cols": [int(x) for x in c.cols.tolist()],
                "weights": [[int(x) for x in row] for row in c.weights.tolist()], "value": [int(x) for x in c.value.tolist()]}
    return {"name": c.name, **({"kind": "eval"} if zk else {}), "point": [[int(x) for x in row] for row in c.point.tolist()],
            "value": [int(x) for x in c.value.tolist()]}


def fixture(relation_name: str = "fp8-ada", l: int = 64, S: int = 2, n_vus: int = 2, seed: int = 7, device: str = "cpu",
            schedule: Schedule | str | None = None, zk: bool = False) -> dict:
    """A complete verifier fixture for ligerito-verify-rs: layout, constraint COO triples, public rows, proof bytes (hex), the
    claims, and the transcript (every absorb and every challenge, in order) of an honest run with ``LocalCoins(b"fixture")``,
    plus negatives (proof bytes that must be rejected, with the Python verifier's reason).  ``zk``: LGSC0004 (masks from the
    fixed key ``sha256(b"fixture-masks")``; the Libra coefficient cells are included so the sparse claims can be checked).
    Toy sizes are format tests, not ZK instances: the schedule may have ``vf > n_c - 4`` (zk-small: 4 > 3 at l = 64, S = 2)."""
    from archive.direct.ligero.relations import relation
    from archive.direct.ligero.relchain import RelationChainRunner, instances
    from .layout import build_z, constraints, fill_zk, layout_for, public_rows
    rel = relation(relation_name)
    runner = RelationChainRunner(rel, device, -128.0)
    vus = instances(rel, n_vus, seed=seed)
    subs, nv = ligero_subbatches(rel, runner, vus, l, S, device)
    lay = layout_for(runner.sys, l, nv, rel.steps, zk=zk)
    cons = constraints(runner.sys, lay, device)
    z = build_z(runner.sys, lay, subs, device)
    if isinstance(schedule, str):
        schedule = _schedule_arg(schedule, lay)
    if zk:
        fill_zk(lay, z, hashlib.sha256(b"fixture-masks").digest())
        schedule = zk_schedule(lay, schedule)
    pubs = public_rows(runner.sys, lay, [s.public() for s in subs], device)
    coins = RecordingCoins(LocalCoins(b"fixture"))
    proof = prove(lay, cons, z, coins, schedule=schedule, allow_underblinded=True)
    pb = proof.to_bytes()
    ok, why, claims = verify(lay, cons, pubs, SumcheckProof.from_bytes(pb), LocalCoins(b"fixture"), device=device)
    assert ok, why
    negs = []
    for pos in sorted({len(proof.header()) + 3, len(pb) // 3, len(pb) // 2, len(pb) - 30}):
        bad = bytearray(pb)
        bad[pos] ^= 0x01
        try:
            okb, whyb, _ = verify(lay, cons, pubs, SumcheckProof.from_bytes(bytes(bad)), LocalCoins(b"fixture"), device=device)
        except ValueError as e:
            okb, whyb = False, f"parse: {e}"
        assert not okb, pos
        negs.append({"flip_byte": pos, "proof_hex": bytes(bad).hex(), "reason": whyb})
    z_bad = z.clone()
    bit = next(r for r in runner.sys.rows if r.kind == "bit")
    z_bad[lay.row_of_sys[bit.idx], 3] = 2
    for cheat in ("zc", "all"):
        pr = prove(lay, cons, z_bad, LocalCoins(b"fixture"), cheat=cheat, schedule=schedule, allow_underblinded=True)
        okb, whyb, _ = verify(lay, cons, pubs, pr, LocalCoins(b"fixture"), device=device)
        assert not okb
        negs.append({"cheat": cheat, "witness": "bit row = 2 at column 3", "proof_hex": pr.to_bytes().hex(), "reason": whyb})
    if zk:
        for what, row in (("M_next[0] += 1 (mask link not a shift)", lay.zk.mask_next), ("i3[9] += 1 (product row)", lay.zk.product[2])):
            z_bad = z.clone()
            col = 0 if row == lay.zk.mask_next else 9
            z_bad[row, col] = (int(z_bad[row, col]) + 1) % P
            pr = prove(lay, cons, z_bad, LocalCoins(b"fixture"), schedule=schedule, allow_underblinded=True)
            okb, whyb, _ = verify(lay, cons, pubs, pr, LocalCoins(b"fixture"), device=device)
            assert not okb
            negs.append({"cheat": None, "witness": what, "proof_hex": pr.to_bytes().hex(), "reason": whyb})
    out = {
        "format": "LGSC0004" if zk else "LGSC0003", "relation": relation_name, "l": l, "S": S, "n_vus": nv, "seed": seed,
        "coins": "LocalCoins(b'fixture')", "coin_count": coins.rounds,
        "schedule": {"zc": list(proof.schedule.zc), "vf": proof.schedule.vf, "cmb": list(proof.schedule.cmb),
                     **({"rb": list(proof.schedule.rb)} if zk else {})},
        "layout": {**lay.summary(), "m": lay.m, "virt": lay.virt, "c_rows": list(lay.c_rows), "n_j": lay.n_j, "n_s": lay.n_s,
                   "row_of_sys": {int(k): int(v) for k, v in lay.row_of_sys.items()}},
        "constraints": {"K_pad": cons.K_pad, "names": cons.names,
                        **{side: [[int(a) for a in t.tolist()] for t in cons.coo[side]] for side in "abc"}},
        "public_rows": {name: [int(x) for x in v.tolist()] for name, v in pubs.items()},
        "proof_hex": pb.hex(),
        "claims": [_claim_json(c, zk) for c in claims],
        "transcript": coins.log,
        "negatives": negs,
    }
    if zk:
        g = z[list(lay.zk.g_rows)].reshape(-1)[:D * proof.schedule.zk_coeffs()]
        out["zk_g_cells"] = [int(x) for x in g.tolist()]
        assert check_claims(lay, z, claims)
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
    ap.add_argument("--schedule", default=None,
                    help="default | c2 | c3 | lean | zk | zk-small | 'zc,arities/vf/cmb,arities[/rb,arities]' (LGSC0004 default: zk)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--negatives", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--fixture", default=None, help="write the toy-size verifier fixture (fp8-ada, l = 64, S = 2, CPU) to this path and exit")
    ap.add_argument("--zk", action="store_true", help="the ZK layout + LGSC0004 (bench and --fixture)")
    a = ap.parse_args()
    if a.fixture:
        with open(a.fixture, "w") as f:
            json.dump(fixture(schedule=a.schedule, zk=a.zk), f)
        return
    res = bench(a.relation, a.vus, a.l, a.S, a.device, a.engine, a.negatives, schedule=a.schedule, reps=a.reps, zk=a.zk)
    s = json.dumps(res, indent=1, default=str)
    print(s)
    if a.json:
        with open(a.json, "w") as f:
            f.write(s)


if __name__ == "__main__":
    _main()
