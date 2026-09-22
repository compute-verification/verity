"""The Ligero-style argument for B units of the transition unit (PROTOCOL.md): prover and verifier share nothing but
the transcript (``Proof``) and the compiled ``System`` (public: a function of ``Params`` only).

    P -> V   root                          Merkle root of the codeword columns of U = Enc(W)
    V -> P   r, rho_lin, rho_quad          D independent BabyBear combinations each (from H(root))
    P -> V   w = r W  (D x l),  h  (D x (l-1))   proximity message; quotient of the batched quadratic test by Z_H
    V -> P   cols                          t column indices (from H(root, w, h))
    P -> V   U[:, cols], Merkle paths

Verifier: (i) Enc(w)[cols] = r U[cols]; (ii) Enc(beta)[cols] = alpha U[cols] with alpha = sum rho_lin,q a_q,
beta = sum rho_lin,q (pub_q - c_q) -- the linear constraints, whose right-hand sides are public, so nothing is sent;
(iii) sum_q rho_quad,q (A_q(U) B_q(U) - C_q(U))[cols] = h(eta_c) Z_H(eta_c); (iv) the Merkle paths.
Fiat-Shamir with SHAKE-256 as the challenge expander (placeholder hash, named in the proof).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import time

import numpy as np
import torch

from verity_numerical.checker.params import Params
from verity_numerical.security import accounting as acc

from .compile import P, System
from .field import batch_modinv, coset_points, encode, intt, poly_eval, zh_on_coset, GEN, modinv
from . import merkle
from .unit import public_vectors
from .witness import Tables, apply_sparse, build_witness, tables

__all__ = ["Config", "Proof", "prove", "verify", "config_for", "soundness"]


# ---------------------------------------------------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    l: int                 # columns = units per proof (power of two)
    n: int                 # codeword length (power of two, > 2 l)
    D: int                 # independent BabyBear challenge coordinates (extension degree equivalent)
    t: int                 # opened columns
    target_log2: float

    @property
    def rate(self) -> float:
        return self.l / self.n


def config_for(l: int, target_log2: float = -128.0, rate_log2: int = 2) -> Config:
    """t from Appendix C with k = l (no ZK padding) and the extension degree from ``security.accounting``:
    BabyBear^6 for 2^-128, BabyBear^4 for 2^-80 (the sensitivity point)."""
    n = l << rate_log2
    t = acc.ligero_queries_for(target_log2, n, l, l)
    # field terms: (n + 1 + L + Q) / p^D must stay well under the target; D = 6 at -128, 4 at -80
    D = 6 if target_log2 <= -100 else 4
    return Config(l, n, D, t, target_log2)


def soundness(cfg: Config, n_linear: int, n_quadratic: int) -> dict:
    """The error terms actually achieved (Appendix C, e = (n - k) / 2, k = l, uniform combinations)."""
    F = acc.Field.prime("BabyBear", P).extension(cfg.D)
    p = acc.LigeroParams(n=cfg.n, k=cfg.l + 1, l=cfg.l, t=cfg.t, zk=False)   # k = l + 1: validate needs l < k
    terms = acc.ligero_error(p, acc.Field.prime("BabyBear", P), F, combination="uniform")
    total = sum(terms.values())
    return {"terms_log2": {k: acc.log2(v) for k, v in terms.items()}, "total_log2": acc.log2(total),
            "field": F.name, "t": cfg.t, "n": cfg.n, "l": cfg.l, "rate": cfg.rate,
            "note": "k = l + 1 stands in for the non-ZK k = l (the accounting's validate() needs l < k); "
                    "the linear test here sends nothing (the verifier encodes beta itself), so its query term is "
                    "at most the Appendix-C ((k+l)/n)^t used"}


# ---------------------------------------------------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------------------------------------------------

def _expand(seed: bytes, count: int, device) -> torch.Tensor:
    """count uniform-ish BabyBear elements from SHAKE-256(seed) (uint32 reduced mod p; bias 2^-31 relative, noted)."""
    raw = hashlib.shake_256(seed).digest(4 * count)
    arr = np.frombuffer(raw, dtype="<u4").astype(np.int64) % P
    return torch.from_numpy(arr).to(device)


def _challenge1(root: bytes, D: int, m: int, L: int, Q: int, device):
    seed = merkle.hash_bytes(b"ligero-b/c1|" + root)
    x = _expand(seed, D * (m + L + Q), device).reshape(D, m + L + Q)
    return x[:, :m], x[:, m:m + L], x[:, m + L:]


def _challenge2(root: bytes, w: np.ndarray, h: np.ndarray, t: int, n: int) -> list[int]:
    seed = merkle.hash_bytes(b"ligero-b/c2|" + root + w.tobytes() + h.tobytes())
    cols: list[int] = []
    seen = set()
    ctr = 0
    while len(cols) < t:
        raw = hashlib.shake_256(seed + ctr.to_bytes(4, "little")).digest(4 * 2 * t)
        for v in np.frombuffer(raw, dtype="<u4"):
            c = int(v) % n
            if c not in seen:
                seen.add(c)
                cols.append(c)
                if len(cols) == t:
                    break
        ctr += 1
    return cols


@dataclass
class Proof:
    root: bytes
    w: np.ndarray            # (D, l) uint32
    h: np.ndarray            # (D, l-1) uint32
    cols: list[int]
    opened: np.ndarray       # (m, t) uint32
    paths: list[list[bytes]]
    hash_name: str = merkle.HASH_NAME
    timings: dict = field(default_factory=dict)

    def n_bytes(self) -> dict:
        b = {"root": len(self.root), "w": self.w.nbytes, "h": self.h.nbytes, "cols": 4 * len(self.cols),
             "opened_columns": self.opened.nbytes, "merkle_paths": sum(32 * len(p) for p in self.paths)}
        b["total"] = sum(b.values())
        return b


# ---------------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------------

def _combine(coef: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """(D, cols) = coef (D, R) @ M (R, cols) mod p, reducing every product before the sum (R < 2^32 terms)."""
    D = coef.shape[0]
    out = torch.empty((D, M.shape[1]), dtype=torch.int64, device=M.device)
    for d in range(D):
        out[d] = ((coef[d][:, None] * M) % P).sum(0) % P
    return out


_GINV: dict = {}


def _ginv_powers(n: int, device) -> torch.Tensor:
    key = (n, str(device))
    if key not in _GINV:
        gi = modinv(GEN)
        pw = [1]
        for _ in range(n - 1):
            pw.append(pw[-1] * gi % P)
        _GINV[key] = torch.tensor(pw, dtype=torch.int64, device=device)
    return _GINV[key]


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


class _Clock:
    def __init__(self, device):
        self.device = device
        self.marks = {}
        _sync(device)
        self.t0 = time.perf_counter()
        self.last = self.t0

    def lap(self, name: str):
        _sync(self.device)
        now = time.perf_counter()
        self.marks[name] = self.marks.get(name, 0.0) + (now - self.last)
        self.last = now

    def total(self):
        _sync(self.device)
        return time.perf_counter() - self.t0


# ---------------------------------------------------------------------------------------------------------------------
# Prover
# ---------------------------------------------------------------------------------------------------------------------

def prove(p: Params, sys: System, tb: Tables, cfg: Config, pub: dict[str, torch.Tensor], hints: torch.Tensor,
          device, check: bool = False, mutate=None) -> Proof:
    """``pub``: the public vectors (from ``public_vectors``); ``hints``: (n_hints, l) from ``gen_step``.
    ``mutate(W)``: a test hook that edits the witness in place before it is committed (the mutated-witness
    negatives); an honest prover never uses it."""
    clk = _Clock(device)
    m, l, n, D = sys.m, cfg.l, cfg.n, cfg.D
    assert hints.shape[1] == l
    W = build_witness(sys, pub, hints, device)
    clk.lap("witness")
    if mutate is not None:
        mutate(W)
    if check:
        from .witness import check_constraints
        bad = check_constraints(sys, tb, W, pub)
        if bad:
            raise ValueError("witness violates constraints: " + ", ".join(bad[:5]))
        clk.lap("self_check")

    U = encode(W, n)                                              # (m, n)
    clk.lap("encode")

    U_cols = U.t().contiguous().cpu().numpy().astype(np.uint32)
    clk.lap("transfer")
    root, levels = merkle.commit(U_cols)
    clk.lap("merkle")

    r, rho_lin, rho_quad = _challenge1(root, D, m, tb.L, tb.Q, device)
    w = _combine(r, W)                                            # (D, l)
    # quadratic: batched A*B - C on the codeword domain, divided by Z_H
    A = apply_sparse(tb.quad["a"][:3], tb.quad["a"][3], tb.Q, U)
    B = apply_sparse(tb.quad["b"][:3], tb.quad["b"][3], tb.Q, U)
    T = (A * B) % P
    del A, B
    C = apply_sparse(tb.quad["c"][:3], tb.quad["c"][3], tb.Q, U)
    T = (T - C) % P
    del C
    P0 = _combine(rho_quad, T)                                    # (D, n)
    del T
    zinv = batch_modinv(zh_on_coset(n, l, device))
    hev = (P0 * zinv) % P
    hcoef = intt(hev)                                             # coefficients of h(g x): c_k g^k
    hcoef = (hcoef * _ginv_powers(n, device)) % P
    if check:
        assert (hcoef[:, l - 1:] == 0).all(), "quotient degree too high: constraints violated"
    h = hcoef[:, :l - 1]
    clk.lap("tests")

    w_np = w.cpu().numpy().astype(np.uint32)
    h_np = h.cpu().numpy().astype(np.uint32)
    cols = _challenge2(root, w_np, h_np, cfg.t, n)
    opened = U_cols[cols].T.copy()                                # (m, t)
    paths = merkle.open_columns(levels, cols)
    clk.lap("openings")
    total = clk.total()
    timings = dict(clk.marks)
    timings["total"] = total
    return Proof(root, w_np, h_np, cols, opened, paths, merkle.HASH_NAME, timings)


# ---------------------------------------------------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------------------------------------------------

def verify(p: Params, sys: System, tb: Tables, cfg: Config, c, a, b, y, proof: Proof, device) -> tuple[bool, str]:
    """Accept/reject the claim ``y[i] = tc_dot(c[i], a[i], b[i])`` for all l units.  Never touches W."""
    t0 = time.perf_counter()
    m, l, n, D, t = sys.m, cfg.l, cfg.n, cfg.D, cfg.t
    try:
        pub = public_vectors(p, c, a, b, y, device=device)
    except ValueError as e:
        return False, f"public inputs: {e}"
    if len(pub["y.s"]) != l:
        return False, "wrong number of units"
    # shapes and ranges of the transcript
    if proof.w.shape != (D, l) or proof.h.shape != (D, l - 1) or proof.opened.shape != (m, t) or len(proof.cols) != t:
        return False, "malformed proof"
    if (proof.w >= P).any() or (proof.h >= P).any() or (proof.opened >= P).any():
        return False, "transcript element out of the field"
    if len(set(proof.cols)) != t or any(not 0 <= c_ < n for c_ in proof.cols):
        return False, "bad column set"
    # challenges
    r, rho_lin, rho_quad = _challenge1(proof.root, D, m, tb.L, tb.Q, device)
    if proof.cols != _challenge2(proof.root, proof.w, proof.h, t, n):
        return False, "column challenge mismatch"
    cols = torch.tensor(proof.cols, dtype=torch.int64, device=device)
    Uo = torch.from_numpy(proof.opened.astype(np.int64)).to(device)          # (m, t)
    # (iv) Merkle
    for k, c_ in enumerate(proof.cols):
        if not merkle.verify_path(proof.root, c_, proof.opened[:, k], proof.paths[k]):
            return False, f"merkle path {k} invalid"
    # (i) proximity
    w = torch.from_numpy(proof.w.astype(np.int64)).to(device)
    if not torch.equal(encode(w, n)[:, cols], _combine(r, Uo)):
        return False, "proximity test failed"
    # (ii) linear constraints
    alpha = torch.zeros((D, m), dtype=torch.int64, device=device)
    terms = (rho_lin[:, tb.lin_q] * tb.lin_coef[None, :]) % P                 # (D, nnz)
    alpha.index_add_(1, tb.lin_j, terms)
    alpha %= P
    beta = torch.zeros((D, l), dtype=torch.int64, device=device)
    const = ((rho_lin * ((-tb.lin_c) % P)[None, :]) % P).sum(1) % P           # (D,)
    beta += const[:, None]
    pub_q = [q for q, name in enumerate(tb.lin_pub) if name is not None]
    if pub_q:
        pubmat = torch.stack([pub[tb.lin_pub[q]].to(device) % P for q in pub_q])      # (Lp, l)
        qidx = torch.tensor(pub_q, dtype=torch.int64, device=device)
        beta = (beta + _combine(rho_lin[:, qidx], pubmat)) % P
    if not torch.equal(encode(beta, n)[:, cols], _combine(alpha, Uo)):
        return False, "linear constraints failed"
    # (iii) quadratic constraints
    A = apply_sparse(tb.quad["a"][:3], tb.quad["a"][3], tb.Q, Uo)
    B = apply_sparse(tb.quad["b"][:3], tb.quad["b"][3], tb.Q, Uo)
    C = apply_sparse(tb.quad["c"][:3], tb.quad["c"][3], tb.Q, Uo)
    lhs = _combine(rho_quad, ((A * B) % P - C) % P)                           # (D, t)
    h = torch.from_numpy(proof.h.astype(np.int64)).to(device)
    eta = coset_points(n, cols)
    rhs = (poly_eval(h, eta) * zh_on_coset(n, l, device)[cols]) % P
    if not torch.equal(lhs, rhs):
        return False, "quadratic constraints failed"
    _sync(device)
    return True, f"accept ({time.perf_counter() - t0:.3f}s)"
