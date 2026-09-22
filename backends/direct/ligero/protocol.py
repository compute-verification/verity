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

import hashlib
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from verity_numerical.checker.params import Params
from verity_numerical.security import accounting as acc

from . import merkle
from .chain import N_FAMILIES, chain_coefs
from .compile import P, System
from .field import (
    GEN,
    batch_modinv,
    compiled,
    encode,
    eval_on_coset,
    intt,
    modinv,
    zh_on_coset,
)
from .unit import public_vectors
from .witness import Tables, apply_sparse, build_witness

__all__ = ["Config", "Proof", "config_for", "prove", "soundness", "verify"]


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


def config_for(l: int, target_log2: float = -128.0, rate_log2: int = 2, n_proofs: int = 1) -> Config:
    """t from Appendix C with k = l (no ZK padding) and the extension degree from ``security.accounting``:
    BabyBear^6 for 2^-128, BabyBear^4 for 2^-80 (the sensitivity point).  ``n_proofs > 1``: the batch is proved as
    that many independently committed sub-batches, so each proof gets ``accounting.per_proof_target`` (the union
    bound over the sub-batches meets ``target_log2``); ``t`` grows by ~log2(n_proofs) / 0.68 columns."""
    n = l << rate_log2
    per_proof = acc.per_proof_target(target_log2, n_proofs) if n_proofs > 1 else target_log2
    t = acc.ligero_queries_for(per_proof, n, l, l)
    # field terms: (n + 1 + L + Q) / p^D must stay well under the target; D = 6 at -128, 4 at -80
    D = 6 if target_log2 <= -100 else 4
    return Config(l, n, D, t, target_log2)


def soundness(cfg: Config, n_linear: int, n_quadratic: int, n_proofs: int = 1) -> dict:
    """The error terms actually achieved (Appendix C, e = (n - k) / 2, k = l, uniform combinations); with
    ``n_proofs`` sub-batches the reported total is ``accounting.union_over_proofs`` (never the per-proof error)."""
    F = acc.Field.prime("BabyBear", P).extension(cfg.D)
    p = acc.LigeroParams(n=cfg.n, k=cfg.l + 1, l=cfg.l, t=cfg.t, zk=False)   # k = l + 1: validate needs l < k
    terms = acc.ligero_error(p, acc.Field.prime("BabyBear", P), F, combination="uniform")
    bud = acc.Budget("B-Ligero direct (backends/direct/ligero)", {"n": cfg.n, "l": cfg.l, "t": cfg.t, "D": cfg.D})
    for k, v in terms.items():
        bud.add("encoding_opening", f"Ligero {k}", v)
    per_proof_total = bud.total()
    union = acc.union_over_proofs(bud, n_proofs) if n_proofs > 1 else bud
    return {"terms_log2": {k: acc.log2(v) for k, v in terms.items()}, "per_proof_total_log2": acc.log2(per_proof_total),
            "n_proofs": n_proofs, "total_log2": acc.log2(union.total()),
            "per_proof_target_log2": acc.per_proof_target(cfg.target_log2, n_proofs) if n_proofs > 1 else cfg.target_log2,
            "field": F.name, "t": cfg.t, "n": cfg.n, "l": cfg.l, "rate": cfg.rate,
            "note": "k = l + 1 stands in for the non-ZK k = l (the accounting's validate() needs l < k); "
                    "the linear test here sends nothing (the verifier encodes beta itself), so its query term is "
                    "at most the Appendix-C ((k+l)/n)^t used; total_log2 is the union bound over n_proofs sub-batches"}


# ---------------------------------------------------------------------------------------------------------------------
# Statement binding (red-team checklist, redteam_vu.md F1/F2 applied to this prover)
# ---------------------------------------------------------------------------------------------------------------------
#
# Every challenge is derived from H(statement, transcript so far).  The statement digest is computed by the VERIFIER
# from its own configuration and public inputs -- never read from the proof: the field, the hash, the compiled
# constraint system (a digest of its tables), the code parameters (l, n, D, t), the mode (unit / chain), the chain
# layout (units per VU, VU count), and every public word (c, a, b, y for units; a, b, y16 for chains).  Sizes come from
# the verifier's Config / System, so a proof of another l, n, m or t is malformed before any challenge is recomputed.

STATEMENT_TAG = b"ligero-b/statement|v2"
_SYS_ID: dict = {}

# Red-team hook ONLY: when False the challenges are derived from the transcript alone (the pre-fix protocol).  The
# binding fixture (``redteam_binding.py``) uses it to show the adaptive-statement forgery the fix closes.  An honest
# deployment never touches it.
_BIND_STATEMENT = True


def system_id(sys: System) -> bytes:
    """A digest of the compiled constraint system: rows (kind, name, interval), linear and quadratic tables, pins,
    hints, chain rows.  Two verifiers with the same digest check the same relation."""
    key = id(sys)
    hit = _SYS_ID.get(key)
    if hit is not None and hit[0] is sys:
        return hit[1]
    h = hashlib.blake2b(digest_size=32)
    h.update(f"m={sys.m};chain={sys.chain is not None}".encode())
    for r in sys.rows:
        h.update(f"{r.idx}:{r.kind}:{r.name}:{r.lo}:{r.hi};".encode())
    for c in sys.linear:
        h.update(f"L:{c.name}:{sorted(c.lin.items())}:{c.c}:{c.pub};".encode())
    for q in sys.quadratic:
        h.update(f"Q:{q.name}:{q.a.key()}:{q.b.key()}:{q.c.key()};".encode())
    h.update(repr(sys.pins).encode())
    h.update(repr(sys.hints).encode())
    if sys.chain is not None:
        h.update(repr(sys.chain["c"]).encode())
        h.update(repr([e.key() for e in sys.chain["y"]] + [sys.chain["y16"].key()]).encode())
    d = h.digest()
    _SYS_ID[key] = (sys, d)
    return d


def _u_bytes(x, dtype: str) -> bytes:
    arr = np.ascontiguousarray(np.asarray(x if not isinstance(x, torch.Tensor) else x.cpu().numpy(), dtype=np.int64))
    return arr.astype(dtype).tobytes()


def statement_digest(sys: System, cfg: Config, a, b, c=None, y=None, layout=None, y16=None) -> bytes:
    """The public statement as the verifier sees it (see the module comment above)."""
    h = hashlib.blake2b(digest_size=32)
    h.update(STATEMENT_TAG)
    h.update(f"|p={P}|hash=blake3|l={cfg.l}|n={cfg.n}|D={cfg.D}|t={cfg.t}|target={cfg.target_log2!r}".encode())
    h.update(b"|sys=" + system_id(sys))
    if layout is not None:
        h.update(f"|mode=chain|l={layout.l}|n_vus={layout.n_vus}|steps={layout.steps}".encode())
    else:
        h.update(b"|mode=unit")
    h.update(b"|a=" + _u_bytes(a, "<u2"))
    h.update(b"|b=" + _u_bytes(b, "<u2"))
    if layout is not None:
        h.update(b"|y16=" + _u_bytes(y16, "<u2"))
    else:
        h.update(b"|c=" + _u_bytes(c, "<u4"))
        h.update(b"|y=" + _u_bytes(y, "<u4"))
    return h.digest()


# ---------------------------------------------------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------------------------------------------------

def _expand(seed: bytes, count: int, device) -> torch.Tensor:
    """count uniform-ish BabyBear elements from SHAKE-256(seed) (uint32 reduced mod p; bias 2^-31 relative, noted)."""
    raw = hashlib.shake_256(seed).digest(4 * count)
    arr = np.frombuffer(raw, dtype="<u4").astype(np.int64) % P
    return torch.from_numpy(arr).to(device)


def _bound(stmt: bytes) -> bytes:
    return stmt if _BIND_STATEMENT else b""


def _challenge1(stmt: bytes, root: bytes, D: int, m: int, L: int, Q: int, device, l_chain: int = 0):
    seed = merkle.hash_bytes(b"ligero-b/c1|" + _bound(stmt) + root)
    extra = N_FAMILIES * l_chain
    x = _expand(seed, D * (m + L + Q + extra), device).reshape(D, m + L + Q + extra)
    rc = x[:, m + L + Q:].reshape(D, N_FAMILIES, l_chain) if l_chain else None
    return x[:, :m], x[:, m:m + L], x[:, m + L:m + L + Q], rc


def _challenge2(stmt: bytes, root: bytes, w: np.ndarray, h: np.ndarray, t: int, n: int, q: np.ndarray | None = None) -> list[int]:
    seed = merkle.hash_bytes(b"ligero-b/c2|" + _bound(stmt) + root + w.tobytes() + h.tobytes() + (q.tobytes() if q is not None else b""))
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
    q: np.ndarray | None = None   # chain mode: (D, 2l-1) uint32, the chain test's combined polynomial

    def n_bytes(self) -> dict:
        b = {"root": len(self.root), "w": self.w.nbytes, "h": self.h.nbytes, "cols": 4 * len(self.cols),
             "opened_columns": self.opened.nbytes, "merkle_paths": sum(32 * len(p) for p in self.paths),
             "q_chain": self.q.nbytes if self.q is not None else 0}
        b["total"] = sum(b.values())
        return b


# ---------------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------------

def _combine_row(c: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    return ((c[:, None] * M) % P).sum(0) % P


def _combine(coef: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """(D, cols) = coef (D, R) @ M (R, cols) mod p, reducing every product before the sum (R < 2^32 terms)."""
    D = coef.shape[0]
    out = torch.empty((D, M.shape[1]), dtype=torch.int64, device=M.device)
    row = compiled(_combine_row, M.device)
    for d in range(D):
        out[d] = row(coef[d], M)
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
          device, check: bool = False, mutate=None, chain: tuple | None = None, words: dict | None = None) -> Proof:
    """``pub``: the public vectors (from ``public_vectors``); ``hints``: (n_hints, l) from ``gen_step``.
    ``words``: the public words the statement binds -- ``{"a", "b", "c", "y"}`` (units) or ``{"a", "b"}`` (chain;
    ``y16`` comes from ``chain``) -- absorbed into every challenge (``statement_digest``).
    ``mutate(W)``: a test hook that edits the witness in place before it is committed (the mutated-witness
    negatives); an honest prover never uses it.  ``chain = (Layout, y16_pub (l,))`` runs the chain test."""
    clk = _Clock(device)
    m, l, n, D = sys.m, cfg.l, cfg.n, cfg.D
    assert hints.shape[1] == l
    assert (chain is not None) == (sys.chain is not None), "chain mode needs a chain-compiled system"
    if words is None:
        raise ValueError("prove needs the public words (statement binding)")
    if chain is not None:
        stmt = statement_digest(sys, cfg, words["a"], words["b"], layout=chain[0], y16=chain[1])
    else:
        stmt = statement_digest(sys, cfg, words["a"], words["b"], c=words["c"], y=words["y"])
    clk.lap("statement")
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

    gpu_hash = merkle.gpu_available() and str(device).startswith("cuda")
    if gpu_hash:
        U32 = U.to(torch.int32).contiguous()                       # elements < p < 2^31
        clk.lap("transfer")                                        # nothing leaves the device here
        root, tree = merkle.commit_device(U32)
        hash_name = merkle.GPU_HASH_NAME
    else:
        U_cols = U.t().contiguous().cpu().numpy().astype(np.uint32)
        clk.lap("transfer")
        root, levels = merkle.commit(U_cols)
        hash_name = merkle.HASH_NAME
    clk.lap("merkle")

    r, _rho_lin, rho_quad, rc = _challenge1(stmt, root, D, m, tb.L, tb.Q, device, l if chain else 0)
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
    q_np = None
    if chain is not None:
        layout, y16_pub = chain
        cc = chain_coefs(sys, layout, rc, y16_pub)
        R = cc.coef.shape[1]
        Renc = encode(cc.coef.reshape(D * R, l), n).reshape(D, R, n)            # R_i on the codeword domain
        Qev = ((Renc * U.index_select(0, cc.rows)[None, :, :]) % P).sum(1) % P   # (D, n)
        qcoef = (intt(Qev) * _ginv_powers(n, device)) % P
        if check:
            assert (qcoef[:, 2 * l - 1:] == 0).all(), "chain polynomial degree too high"
            assert torch.equal((l * (qcoef[:, 0] + qcoef[:, l])) % P, cc.rhs), "chain constraints violated"
        q_np = qcoef[:, :2 * l - 1].cpu().numpy().astype(np.uint32)
    clk.lap("tests")

    w_np = w.cpu().numpy().astype(np.uint32)
    h_np = h.cpu().numpy().astype(np.uint32)
    cols = _challenge2(stmt, root, w_np, h_np, cfg.t, n, q_np)
    if gpu_hash:
        cols_t = torch.tensor(cols, dtype=torch.int64, device=device)
        opened = U32.index_select(1, cols_t).cpu().numpy().astype(np.uint32)   # (m, t): the only U bytes that leave
        paths = merkle.open_device(tree, cols)
    else:
        opened = U_cols[cols].T.copy()                            # (m, t)
        paths = merkle.open_columns(levels, cols)
    clk.lap("openings")
    total = clk.total()
    timings = dict(clk.marks)
    timings["total"] = total
    return Proof(root, w_np, h_np, cols, opened, paths, hash_name, timings, q_np)


# ---------------------------------------------------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------------------------------------------------

def verify(p: Params, sys: System, tb: Tables, cfg: Config, c, a, b, y, proof: Proof, device,
           chain: tuple | None = None) -> tuple[bool, str]:
    """Accept/reject the claim ``y[i] = tc_dot(c[i], a[i], b[i])`` for all l units -- or, with
    ``chain = (Layout, y16_pub)`` (``c = y = None``), the claim that every VU of the layout casts to its
    ``y16_pub`` word from ``c_0 = 0`` over its 96 units.  Never touches W."""
    t0 = time.perf_counter()
    m, l, n, D, t = sys.m, cfg.l, cfg.n, cfg.D, cfg.t        # every size from the VERIFIER's configuration
    if (chain is not None) != (sys.chain is not None):
        return False, "chain mode mismatch"
    if not (proof.hash_name or "").startswith("blake3"):
        return False, "proof names a hash the verifier does not use"
    try:
        pub = public_vectors(p, c, a, b, y, device=device)
    except ValueError as e:
        return False, f"public inputs: {e}"
    if len(pub["a[0].s"]) != l:
        return False, "wrong number of units"
    # shapes and ranges of the transcript
    if proof.w.shape != (D, l) or proof.h.shape != (D, l - 1) or proof.opened.shape != (m, t) or len(proof.cols) != t:
        return False, "malformed proof"
    if len(proof.paths) != t or any(len(pth) != n.bit_length() - 1 for pth in proof.paths):
        return False, "malformed Merkle paths"
    if (proof.w >= P).any() or (proof.h >= P).any() or (proof.opened >= P).any():
        return False, "transcript element out of the field"
    if chain is not None:
        layout, y16_pub = chain
        y16_pub = torch.as_tensor(y16_pub, dtype=torch.int64)
        if layout.l != l or proof.q is None or proof.q.shape != (D, 2 * l - 1) or (proof.q >= P).any():
            return False, "malformed chain transcript"
        if len(y16_pub) != l or ((y16_pub < 0) | (y16_pub >= 2**p.word_bits)).any():
            return False, "y16 range"
        stmt = statement_digest(sys, cfg, a, b, layout=layout, y16=y16_pub)
    elif proof.q is not None:
        return False, "unexpected chain transcript"
    else:
        stmt = statement_digest(sys, cfg, a, b, c=c, y=y)
    if len(set(proof.cols)) != t or any(not 0 <= c_ < n for c_ in proof.cols):
        return False, "bad column set"
    # challenges: H(statement, transcript prefix) -- the statement is the verifier's, not the proof's
    r, rho_lin, rho_quad, rc = _challenge1(stmt, proof.root, D, m, tb.L, tb.Q, device, l if chain else 0)
    if proof.cols != _challenge2(stmt, proof.root, proof.w, proof.h, t, n, proof.q):
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
    rhs = (eval_on_coset(h, n)[:, cols] * zh_on_coset(n, l, device)[cols]) % P
    if not torch.equal(lhs, rhs):
        return False, "quadratic constraints failed"
    # (v) chain: the general linear test
    if chain is not None:
        cc = chain_coefs(sys, layout, rc, torch.as_tensor(y16_pub, dtype=torch.int64))
        q = torch.from_numpy(proof.q.astype(np.int64)).to(device)
        if not torch.equal((l * (q[:, 0] + q[:, l])) % P, cc.rhs):
            return False, "chain constraints failed (sum over H)"
        R = cc.coef.shape[1]
        Rc = encode(cc.coef.reshape(D * R, l), n).reshape(D, R, n)[:, :, cols]          # (D, R, t)
        expect = ((Rc * Uo.index_select(0, cc.rows)[None, :, :]) % P).sum(1) % P
        if not torch.equal(eval_on_coset(q, n)[:, cols], expect):
            return False, "chain constraints failed (opened columns)"
    _sync(device)
    return True, f"accept ({time.perf_counter() - t0:.3f}s)"
