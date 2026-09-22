"""Class 2: the malicious-verifier simulator for the 8c protocol, written against the real verifier interface.

Lane b-zk-fix (red team F2 repaired).  The interactive derivation is now a function of the verifier's coins ALONE:
``r, rho, rc = SHAKE(H(stmt, c, r1))`` and ``cols = H(stmt, c, r2)`` -- no prover message enters either hash.  So
after the rewind the simulator knows ``(r1, r2)`` and hence ``r, rho, rc, cols`` BEFORE it builds anything, and it
prepares a consistent fake tableau without programming any hash: the simulator below runs against the real
``verify()`` under the real ``merkle.hash_bytes`` (a ``ProgrammableOracle`` with an EMPTY table is installed only to
count that zero programmed points are ever hit).  The only assumption left is the binding of the coin commitment
(a collision of Blake2b) and the hiding of the unopened Merkle leaves; the challenge hash needs no oracle.

Simulator (given V* with the ``Coins`` interface and the public statement):
  1. ``c1, c2 = V*.commitments()``.
  2. Dummy run: random ``root0``; ``V*.open1(root0)`` -> check ``c1`` (abort transcript on mismatch); uniform
     ``w0, h0, q0, v0`` of the right shapes; ``V*.open2(...)`` -> check ``c2``.  Now ``(r1, r2)`` are known.
  3. Rewind V* to after step 0.  Derive ``r, rho_lin, rho_quad, rc`` and ``cols`` from ``(stmt, c, r1)`` /
     ``(stmt, c, r2)`` exactly as the verifier does.  Sample the fake tableau: witness rows uniform polynomials of
     degree < k, ``s_prox`` uniform; ``v`` uniform on ``{v|_H = beta}``, ``h`` uniform, ``q`` uniform on
     ``{sum_H q = rhs}``; solve the mask rows ``s_lin, s_qa, u_ca`` (Lagrange at the t opened points + uniform
     slack) so every column test holds at ``cols``; ``w := r . p' + s_prox`` (holds everywhere).  Commit, replay V*:
     by binding it opens the same coins or aborts (repeat on abort, GK).
  4. Output the transcript; the real ``verify`` (real hash) must accept it.

Malicious verifiers modelled: ``AdaptiveVerifier`` (opens a slot to a value that depends on the prover's messages:
needs a collision, so the prover aborts and the simulator outputs the same abort), ``StatementAdaptiveVerifier``
(chooses ``r2`` before step 0 as a function of everything it can see -- the statement and its own commitments -- by
grinding the column set towards a target: legal, and simulated like any coins), ``AbortingVerifier`` (selective
abort on a predicate of the messages it has seen).

The indistinguishability test (``indistinguishability``) compares N real transcripts with N simulated ones
per field-element position (opened columns, w, h, v, q; two-sample chi-square on 8 value bins with Bonferroni),
then runs the two ``q`` distinguishers of ``q_leak``: with the F1 fix both must be at chance (the witness-free
consistency fraction ~ 0 on real AND simulated transcripts, 0 accumulators recovered from either).
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch

from .. import merkle, protocol
from ..chain import N_FAMILIES, chain_coefs
from ..compile import P, System
from ..field import coset_points, encode, eval_on_coset, intt, modinv, zh_on_coset
from ..protocol import (
    COIN_BYTES,
    Coins,
    Config,
    Proof,
    coin_commit,
    mask_rows,
    n_rows,
    statement_digest,
)
from ..unit import public_vectors
from ..witness import quad_p0
from . import linalg

__all__ = ["AbortTranscript", "AbortingVerifier", "AdaptiveVerifier", "ProgrammableOracle", "StatementAdaptiveVerifier",
           "indistinguishability", "simulate"]


# ---------------------------------------------------------------------------------------------------------------------
# The random oracle the challenge derivation goes through
# ---------------------------------------------------------------------------------------------------------------------

class ProgrammableOracle:
    """``merkle.hash_bytes`` (the seed hash of ``_challenge1`` / ``_challenge2``) with a programmable table.
    Everything not programmed is the real hash.  Installed for the duration of a simulation + verification."""

    def __init__(self):
        self.table: dict[bytes, bytes] = {}
        self.real = merkle.hash_bytes
        self.queries = 0
        self.programmed_hits = 0

    def __call__(self, data: bytes) -> bytes:
        self.queries += 1
        hit = self.table.get(data)
        if hit is not None:
            self.programmed_hits += 1
            return hit
        return self.real(data)

    def program(self, data: bytes, out: bytes) -> None:
        if data in self.table and self.table[data] != out:
            raise RuntimeError("oracle point already programmed differently")
        self.table[data] = out

    def install(self) -> None:
        merkle.hash_bytes = self

    def uninstall(self) -> None:
        merkle.hash_bytes = self.real


# ---------------------------------------------------------------------------------------------------------------------
# Malicious verifiers
# ---------------------------------------------------------------------------------------------------------------------

def _hb(*parts: bytes) -> bytes:
    import hashlib
    return hashlib.blake2b(b"|".join(parts), digest_size=COIN_BYTES).digest()


class AdaptiveVerifier(Coins):
    """Commits honestly, then opens slot ``slot`` to a value that depends on the prover's messages (the coin the
    commitment forbids).  It cannot produce a *valid* opening of that kind (a collision), so the prover aborts."""

    def __init__(self, slot: int):
        h = Coins.sample()
        super().__init__(h.r1, h.s1, h.r2, h.s2)
        object.__setattr__(self, "slot", slot)

    def open1(self, root):
        return (_hb(b"adaptive-r1", root), self.s1) if self.slot == 1 else (self.r1, self.s1)

    def open2(self, root, w, h, q, v):
        return (_hb(b"adaptive-r2", root, w.tobytes(), h.tobytes()), self.s2) if self.slot == 2 else (self.r2, self.s2)


class StatementAdaptiveVerifier(Coins):
    """Chooses its coins BEFORE step 0 as a function of everything it can see then -- the public statement and its
    own commitments: grinds ``r2`` over ``candidates`` values so that the column set ``cols = H(stmt, c, r2)``
    overlaps a target set as much as possible (with the b-zk-fix derivation this is computable ahead of the
    prover's first message).  Legal for an interactive verifier; the coins are then honestly committed and opened,
    and the simulator handles them like any other coins."""

    def __init__(self, cfg: Config, stmt: bytes, target_cols: list[int], candidates: int = 64):
        h = Coins.sample()
        best, best_score = h.r2, -1
        c1 = coin_commit(h.r1, h.s1)
        for i in range(candidates):
            r2 = _hb(b"stmt-adaptive", h.r2, i.to_bytes(4, "little"))
            cols = protocol._challenge2(stmt, b"", np.zeros(1, "<u4"), np.zeros(1, "<u4"), cfg.t, cfg.n, None, None,
                                        (c1, coin_commit(r2, h.s2)), r2)
            score = len(set(cols) & set(target_cols))
            if score > best_score:
                best, best_score = r2, score
        super().__init__(h.r1, h.s1, best, h.s2)
        object.__setattr__(self, "target_overlap", best_score)


class AbortingVerifier(Coins):
    """Honest coins; refuses to open slot ``slot`` (sends a malformed opening) when ``predicate(V, root, w, h, q, v)``
    holds (``V`` is the verifier itself, so the predicate may use its coins).  The selective-abort verifier of the
    Goldreich-Kahan analysis."""

    def __init__(self, predicate, slot: int = 2):
        h = Coins.sample()
        super().__init__(h.r1, h.s1, h.r2, h.s2)
        object.__setattr__(self, "predicate", predicate)
        object.__setattr__(self, "slot", slot)
        object.__setattr__(self, "aborted", [])

    def open1(self, root):
        if self.slot == 1 and self.predicate(self, root, None, None, None, None):
            self.aborted.append(1)
            return b"\0" * COIN_BYTES, b"\0" * COIN_BYTES
        return self.r1, self.s1

    def open2(self, root, w, h, q, v):
        if self.slot == 2 and self.predicate(self, root, w, h, q, v):
            self.aborted.append(2)
            return b"\0" * COIN_BYTES, b"\0" * COIN_BYTES
        return self.r2, self.s2


@dataclass
class AbortTranscript:
    slot: int
    sent: list[str]
    root: bytes


# ---------------------------------------------------------------------------------------------------------------------
# The simulator
# ---------------------------------------------------------------------------------------------------------------------

def _uniform(shape: tuple[int, ...], device) -> torch.Tensor:
    total = int(np.prod(shape))
    return protocol._uniform_symbols(total, device).reshape(shape)


def _eval_rows(coefs: torch.Tensor, n: int, chunk: int = 256) -> torch.Tensor:
    """eval_on_coset in row chunks -> int32 (frugal with device memory)."""
    out = torch.empty((coefs.shape[0], n), dtype=torch.int32, device=coefs.device)
    for lo in range(0, coefs.shape[0], chunk):
        out[lo:lo + chunk] = eval_on_coset(coefs[lo:lo + chunk], n).to(torch.int32)
    return out


def _u32(x: torch.Tensor) -> np.ndarray:
    return x.cpu().numpy().astype(np.uint32)


def simulate(V, p, sys: System, tb, cfg: Config, layout, y16, a, b, device, oracle: ProgrammableOracle | None = None,
             max_rewinds: int = 8, rng: np.random.Generator | None = None) -> Proof | AbortTranscript:
    """The simulator of the module docstring.  No hash is programmed: ``oracle`` is accepted for interface
    compatibility only and is NEVER written to (the caller may install one with an empty table to count hits)."""
    assert cfg.zk and cfg.interactive, "the simulator is for the 8b+8c protocol (interactive mode)"
    rng = rng or np.random.default_rng()
    m, l, n, D, t, k = sys.m, cfg.l, cfg.n, cfg.D, cfg.t, cfg.k
    tp = cfg.t_pad
    mr = mask_rows(cfg, m, True)
    M = n_rows(cfg, m, True)
    y16_t = torch.as_tensor(y16, dtype=torch.int64)
    stmt = statement_digest(sys, cfg, a, b, layout=layout, y16=y16_t)
    pub = public_vectors(p, None, a, b, None, device=device)
    cc = V.commitments()

    # ---- 1-2. dummy run: learn the coins ---------------------------------------------------------------------------
    root0 = os.urandom(32)
    r1, s1 = V.open1(root0)
    if len(r1) != COIN_BYTES or len(s1) != COIN_BYTES or coin_commit(r1, s1) != cc[0]:
        return AbortTranscript(1, ["coin commitments (V)", "root"], root0)
    w0 = _u32(_uniform((D, k), device))
    h0 = _u32(_uniform((D, cfg.h_len), device))
    v0 = _u32(_uniform((D, k), device))
    q0 = _u32(_uniform((D, cfg.q_len), device))
    r2, s2 = V.open2(root0, w0, h0, q0, v0)
    if len(r2) != COIN_BYTES or len(s2) != COIN_BYTES or coin_commit(r2, s2) != cc[1]:
        return AbortTranscript(2, ["coin commitments (V)", "root", "w, h, q, v"], root0)

    # ---- 3. rewind: coins known -> the challenges are known (functions of (stmt, c, r_i) alone); build a consistent
    # fake transcript for them.  Exactly the verifier's derivation, no seed of the simulator's choosing.
    r, rho_lin, rho_quad, rc = protocol._challenge1(stmt, b"", D, M, tb.L, tb.Q, device, l, cc, r1)
    cols = protocol._challenge2(stmt, b"", np.zeros(1, "<u4"), np.zeros(1, "<u4"), t, n, None, None, cc, r2)
    for attempt in range(max_rewinds):
        cols_t = torch.tensor(cols, dtype=torch.int64, device=device)
        eta = [int(x) for x in coset_points(n, cols_t.cpu()).tolist()]
        zh_c = zh_on_coset(n, l, device)[cols_t]                                    # (t,)
        zh_inv = [modinv(int(z)) for z in zh_c.tolist()]
        xk_c = protocol._xk_on_coset(n, k, device)[cols_t]

        coefs = torch.zeros((M, k), dtype=torch.int64, device=device)
        coefs[:m] = _uniform((m, k), device)                                        # fake witness rows: uniform codewords
        coefs[mr["prox"]] = _uniform((D, k), device)
        Uw = _eval_rows(coefs[:m], n)                                               # (m, n) int32
        Uo = Uw.index_select(1, cols_t).to(torch.int64)                             # (m, t)

        # linear test: v uniform on {v|_H = beta}; s_lin fits v at the opened columns
        alpha, beta = protocol._alpha_beta(tb, rho_lin, pub, m, l, device)
        v = torch.zeros((D, k), dtype=torch.int64, device=device)
        v[:, :l] = intt(beta % P)
        protocol._add_zh_times(v, _uniform((D, tp), device), l)
        v_at = eval_on_coset(v, n)[:, cols_t]                                       # (D, t)
        need = ((v_at - protocol._combine(alpha, Uo)) % P).cpu().numpy()
        for d in range(D):
            vals = [int(need[d, i]) * zh_inv[i] % P for i in range(t)]
            coefs[mr["lin"].start + d, :tp] = torch.from_numpy(linalg.interp_with_slack(eta, vals, tp, rng)).to(device)

        # quadratic test: h uniform; s_qb uniform; s_qa fits
        h = _uniform((D, cfg.h_len), device)
        coefs[mr["qb"], :tp - 1] = _uniform((D, tp - 1), device)
        P0 = quad_p0(tb, rho_quad, Uo)                                              # (D, t)
        h_at = eval_on_coset(h, n)[:, cols_t]
        qb_at = eval_on_coset(coefs[mr["qb"]], n)[:, cols_t]
        zinv_t = torch.tensor(zh_inv, dtype=torch.int64, device=device)
        need = ((h_at - (P0 * zinv_t[None, :]) % P - (xk_c[None, :] * qb_at) % P) % P).cpu().numpy()
        for d in range(D):
            coefs[mr["qa"].start + d] = torch.from_numpy(linalg.interp_with_slack(eta, [int(x) for x in need[d]], k, rng)).to(device)

        # chain test: q uniform on {sum_H q = rhs}; u_cb uniform, u_ca fits (q = Q + u_ca + x^k u_cb at the columns)
        ccf = chain_coefs(sys, layout, rc, y16_t)
        R = ccf.coef.shape[1]
        q = _uniform((D, cfg.q_len), device)
        # l (q_0 + q_l + q_2l) = rhs  ->  q_0 := rhs / l - (q_l + q_2l)
        linv = modinv(l)
        q[:, 0] = ((ccf.rhs * linv) % P - q[:, l] - q[:, 2 * l]) % P
        assert torch.equal(protocol._sum_over_H(q, l), ccf.rhs)
        Rc = encode(ccf.coef.reshape(D * R, l), n).reshape(D, R, n)[:, :, cols_t]  # (D, R, t)
        Qo = ((Rc * Uo.index_select(0, ccf.rows)[None, :, :]) % P).sum(1) % P
        q_at = eval_on_coset(q, n)[:, cols_t]
        coefs[mr["chain_b"], :l - 1] = _uniform((D, l - 1), device)
        cb_at = eval_on_coset(coefs[mr["chain_b"]], n)[:, cols_t]
        need = ((q_at - Qo - (xk_c[None, :] * cb_at) % P) % P).cpu().numpy()
        for d in range(D):
            coefs[mr["chain_a"].start + d] = torch.from_numpy(linalg.interp_with_slack(eta, [int(x) for x in need[d]], k, rng)).to(device)

        # the whole tableau, the proximity message, the commitment
        U = torch.empty((M, n), dtype=torch.int32, device=device)
        U[:m] = Uw
        del Uw
        U[m:] = _eval_rows(coefs[m:], n)
        r = r.clone()
        r[:, mr["prox"]] = 0
        w = (protocol._combine(r, coefs) + coefs[mr["prox"]]) % P
        gpu_hash = merkle.gpu_available() and str(device).startswith("cuda")
        if gpu_hash:
            root, tree = merkle.commit_device(U.contiguous())
            hash_name = merkle.GPU_HASH_NAME
            opened = _u32(U.index_select(1, cols_t))
            paths = merkle.open_device(tree, cols)
        else:
            U_cols = U.t().contiguous().cpu().numpy().astype(np.uint32)
            root, levels = merkle.commit(U_cols)
            hash_name = merkle.HASH_NAME
            opened = U_cols[cols].T.copy()
            paths = merkle.open_columns(levels, cols)
        w_np, h_np, q_np, v_np = _u32(w), _u32(h), _u32(q), _u32(v)
        del U, coefs                                             # nothing to program: the challenges never see root
        # ---- replay V* from after step 0 ---------------------------------------------------------------------------
        r1b, s1b = V.open1(root)
        if (r1b, s1b) != (r1, s1):
            if len(r1b) == COIN_BYTES and len(s1b) == COIN_BYTES and coin_commit(r1b, s1b) == cc[0]:
                raise RuntimeError("V* opened c1 to two different values: a BLAKE2b collision")
            continue                                             # aborted in the replay only: rewind again (GK)
        r2b, s2b = V.open2(root, w_np, h_np, q_np, v_np)
        if (r2b, s2b) != (r2, s2):
            if len(r2b) == COIN_BYTES and len(s2b) == COIN_BYTES and coin_commit(r2b, s2b) == cc[1]:
                raise RuntimeError("V* opened c2 to two different values: a BLAKE2b collision")
            continue
        return Proof(root, w_np, h_np, cols, opened, paths, hash_name, {"rewinds": attempt}, q_np, v_np, cc,
                     Coins(r1, s1, r2, s2))
    return AbortTranscript(0, ["rewind budget exhausted"], b"")


# ---------------------------------------------------------------------------------------------------------------------
# Indistinguishability
# ---------------------------------------------------------------------------------------------------------------------

BINS = 8


def _chi2_sf(stat: np.ndarray, dof: int) -> np.ndarray:
    """Upper tail of chi-square (Wilson-Hilferty), vectorised."""
    z = ((stat / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    return 0.5 * np.array([math.erfc(float(v) / math.sqrt(2)) for v in z.ravel()]).reshape(z.shape)


def per_position_chi2(X: np.ndarray, Y: np.ndarray) -> dict:
    """Two-sample chi-square (``BINS`` value bins) at every position of two (N, ...) sample arrays.  Returns the
    minimum p, the Bonferroni threshold, and how many positions fall below it (expected ~0.001 under the null)."""
    N = X.shape[0]
    pos = int(np.prod(X.shape[1:]))
    bx = (X.reshape(N, pos).astype(np.int64) * BINS // P).T                      # (pos, N)
    by = (Y.reshape(Y.shape[0], pos).astype(np.int64) * BINS // P).T
    cx = np.stack([(bx == i).sum(1) for i in range(BINS)], 1).astype(float)     # (pos, BINS)
    cy = np.stack([(by == i).sum(1) for i in range(BINS)], 1).astype(float)
    kx, ky = math.sqrt(cy.sum(1)[0] / cx.sum(1)[0]), math.sqrt(cx.sum(1)[0] / cy.sum(1)[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = (kx * cx - ky * cy) ** 2 / (cx + cy)
    stat = np.nansum(terms, 1)
    pv = _chi2_sf(stat, BINS - 1)
    thr = 0.001 / pos
    return {"positions": pos, "samples_per_side": [int(N), int(Y.shape[0])], "min_p": float(pv.min()),
            "bonferroni_threshold": thr, "positions_below_threshold": int((pv < thr).sum()),
            "expected_below_threshold_under_null": 0.001, "median_p": float(np.median(pv))}


def indistinguishability(real: list[Proof], sim: list[Proof], m: int) -> dict:
    """Per-position tests on every transcript component of N real vs N simulated accepting transcripts."""
    out = {}
    for name, pick in (("opened_witness_rows", lambda pr: pr.opened[:m]), ("opened_mask_rows", lambda pr: pr.opened[m:]),
                       ("w", lambda pr: pr.w), ("h", lambda pr: pr.h), ("v", lambda pr: pr.v), ("q", lambda pr: pr.q)):
        X = np.stack([pick(pr) for pr in real])
        Y = np.stack([pick(pr) for pr in sim])
        out[name] = per_position_chi2(X, Y)
    return out


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t0
