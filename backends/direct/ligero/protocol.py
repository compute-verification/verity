"""The Ligero-style argument for B units of the transition unit (PROTOCOL.md): prover and verifier share nothing but
the transcript (``Proof``) and the compiled ``System`` (public: a function of ``Params`` only).

    V -> P   c1 = H(r1 || s1), c2 = H(r2 || s2)   step 0 (HM96 / Goldreich-Kahan): the verifier commits to the
                                                   randomness of both of its coins BEFORE the prover's first message
    P -> V   root                          Merkle root of the codeword columns of U = Enc(W)
    V -> P   (r1, s1)                      opening of coin slot 1; P checks c1 and aborts on mismatch
             r, rho_lin, rho_quad          D independent BabyBear combinations each: H(stmt, c1, c2, root, r1)
    P -> V   w = r W  (D x l),  h  (D x (l-1))   proximity message; quotient of the batched quadratic test by Z_H
    V -> P   (r2, s2)                      opening of coin slot 2; P checks c2
             cols                          t column indices: H(stmt, c1, c2, root, w, h, q, v, r2)
    P -> V   U[:, cols], Merkle paths

Three verifier messages on the critical path (c, r1, r2): ``rounds.sequential_depth = 3`` (PROTOCOL.md 8c).  In the
non-interactive (Fiat-Shamir) reading the prover plays the verifier's step 0 itself (``Coins.sample``) and the
transcript carries the commitments and the openings; the verifier recomputes ``c_i = H(r_i || s_i)``.

Verifier: (i) Enc(w)[cols] = r U[cols]; (ii) Enc(beta)[cols] = alpha U[cols] with alpha = sum rho_lin,q a_q,
beta = sum rho_lin,q (pub_q - c_q) -- the linear constraints, whose right-hand sides are public, so nothing is sent;
(iii) sum_q rho_quad,q (A_q(U) B_q(U) - C_q(U))[cols] = h(eta_c) Z_H(eta_c); (iv) the Merkle paths.
Fiat-Shamir with SHAKE-256 as the challenge expander (placeholder hash, named in the proof).
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import torch
from verity_numerical.checker.params import Params
from verity_numerical.security import accounting as acc

from . import merkle, witness
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
from .witness import Tables, build_witness, quad_p0

__all__ = ["CoinOpeningError", "Coins", "Config", "Proof", "coin_commit", "config_for", "prove", "soundness", "verify"]


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
    zk: bool = False       # HVZK masking (PROTOCOL.md section 8b as implemented): t_pad random symbols per row
    t_pad: int = 0         # k = l + t_pad; Lemma 4.15 needs t_pad > t

    @property
    def rate(self) -> float:
        return self.k / self.n

    @property
    def k(self) -> int:
        """Message length of the row code: l payload symbols (+ t_pad random ones under ZK)."""
        return self.l + self.t_pad

    @property
    def h_len(self) -> int:
        """Length of the quadratic quotient h = (P + mask) / Z_H: degree < 2k - 1 - l."""
        return 2 * self.k - 1 - self.l

    @property
    def q_len(self) -> int:
        """Length of the chain polynomial q = sum R_i p_i (+ mask): degree < l + k - 1."""
        return self.l + self.k - 1


ZK_T_PAD = 256      # random symbols per row under ZK (>= t + 1 at every configuration used; asserted in config_for)


def config_for(l: int, target_log2: float = -128.0, rate_log2: int = 2, n_proofs: int = 1, zk: bool = False,
               t_pad: int = ZK_T_PAD) -> Config:
    """t from Appendix C with k = l (no ZK padding) or k = l + t_pad (ZK: the padding is t_pad random *coefficients*
    of Z_H(x) s(x), so k need not be a power of two and n stays l << rate_log2 -- section 8b's "n = 32768" was the
    cost of rounding k to a power of two, which the coefficient form avoids) and the extension degree from
    ``security.accounting``: BabyBear^6 for 2^-128, BabyBear^4 for 2^-80 (the sensitivity point).  ``n_proofs > 1``:
    the batch is proved as that many independently committed sub-batches, so each proof gets
    ``accounting.per_proof_target`` (the union bound over the sub-batches meets ``target_log2``); ``t`` grows by
    ~log2(n_proofs) / 0.68 columns."""
    n = l << rate_log2
    per_proof = acc.per_proof_target(target_log2, n_proofs) if n_proofs > 1 else target_log2
    k = l + (t_pad if zk else 0)
    while n <= 2 * k:                     # small-l ZK configurations (l <= t_pad): the quadratic test needs n > 2k
        n *= 2
    t = acc.ligero_queries_for(per_proof, n, k, l)
    while zk and not k > l + t:           # small l: the padding lowers the rate; double n until Lemma 4.15 holds
        n *= 2
        t = acc.ligero_queries_for(per_proof, n, k, l)
    # field terms: (n + 1 + L + Q) / p^D must stay well under the target; D = 6 at -128, 4 at -80
    D = 6 if target_log2 <= -100 else 4
    return Config(l, n, D, t, target_log2, zk, t_pad if zk else 0)


def soundness(cfg: Config, n_linear: int, n_quadratic: int, n_proofs: int = 1) -> dict:
    """The error terms actually achieved (Appendix C, e = (n - k) / 2, uniform combinations); with ``n_proofs``
    sub-batches the reported total is ``accounting.union_over_proofs`` (never the per-proof error)."""
    F = acc.Field.prime("BabyBear", P).extension(cfg.D)
    k = cfg.k if cfg.zk else cfg.l + 1                                        # k = l + 1: validate needs l < k
    p = acc.LigeroParams(n=cfg.n, k=k, l=cfg.l, t=cfg.t, zk=cfg.zk)
    terms = acc.ligero_error(p, acc.Field.prime("BabyBear", P), F, combination="uniform")
    bud = acc.Budget("B-Ligero direct (backends/direct/ligero)", {"n": cfg.n, "k": k, "l": cfg.l, "t": cfg.t, "D": cfg.D})
    for k, v in terms.items():
        bud.add("encoding_opening", f"Ligero {k}", v)
    if _COMMIT_COINS:
        # step 0 (8c): the prover must not learn r_i from c_i = H(r_i || s_i).  Random-oracle hiding: q 2^-|r| with
        # q = 2^64 hash queries and |r| = 256 bits -> 2^-192 per slot (the salt is not even counted), two slots.
        # Binding (a second opening = a collision of H, 2^-128 generic) protects the PROVER's privacy, not soundness,
        # and is reported separately (privacy_terms_log2).
        hide = COIN_HIDING_TERM
        terms = {**terms, "coin_commitment_hiding": hide}
        bud.add("commitment_hash", "HM96 step-0 coin commitments: hiding vs the prover, 2 slots x q 2^-256 (q = 2^64)", hide)
    per_proof_total = bud.total()
    union = acc.union_over_proofs(bud, n_proofs) if n_proofs > 1 else bud
    return {"terms_log2": {k: acc.log2(v) for k, v in terms.items()}, "per_proof_total_log2": acc.log2(per_proof_total),
            "n_proofs": n_proofs, "total_log2": acc.log2(union.total()),
            "per_proof_target_log2": acc.per_proof_target(cfg.target_log2, n_proofs) if n_proofs > 1 else cfg.target_log2,
            "field": F.name, "t": cfg.t, "n": cfg.n, "k": cfg.k, "l": cfg.l, "rate": cfg.rate, "zk": cfg.zk, "t_pad": cfg.t_pad,
            "coin_commitment": _COMMIT_COINS,
            "privacy_terms_log2": ({"coin_commitment_binding_collision": acc.log2(acc.hash_collision_bound(256, 2**64)),
                                    "mask_sampling_bias_per_symbol": "0 (rejection sampling; the simulator's distance is "
                                                                      "ChaCha20's PRF advantage on the mask key)"} if _COMMIT_COINS else {}),
            "note": ("ZK: k = l + t_pad, every row carries t_pad > t uniform coefficients (Lemma 4.15)" if cfg.zk else
                     "k = l + 1 stands in for the non-ZK k = l (the accounting's validate() needs l < k)") + "; "
                    "the linear test here sends nothing (the verifier encodes beta itself), so its query term is "
                    "at most the Appendix-C ((k+l)/n)^t used; total_log2 is the union bound over n_proofs sub-batches"
                    + ("; + the step-0 coin-commitment hiding term (commitment_hash)" if _COMMIT_COINS else "")}


COIN_HIDING_TERM = 2 * Fraction(2**64, 2**256)      # 2 slots x q / 2^|r|, q = 2^64


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
    h.update(f"|zk={int(cfg.zk)}|k={cfg.k}".encode())
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
# Step 0: the verifier's coin commitments (HM96 / Goldreich-Kahan; PROTOCOL.md 8c)
# ---------------------------------------------------------------------------------------------------------------------
#
# The verifier's two coins (the combination challenge and the column challenge) are functions of two 32-byte random
# strings r1, r2 that the verifier commits to -- c_i = H(tag || r_i || s_i), s_i a 32-byte salt -- before the prover's
# first message, and opens one at a time, each right after the prover message it must follow.  The prover checks each
# opening against the commitment BEFORE computing anything that depends on the coin and aborts (CoinOpeningError) on a
# mismatch: the abort depends on public data only (c_i, r_i, s_i).  What a malicious verifier can no longer do: choose
# r1 after seeing root, or r2 after seeing w, h, q, v -- by binding (a second opening of c_i is a collision of H) every
# coin the prover accepts is a fixed function of the verifier's randomness sampled before root.  What the simulator
# gets: after step 0 the coins are fixed, so the simulator runs the verifier once to learn them (dummy root, random
# messages), rewinds to the state after step 0, and prepares root / messages / opened columns consistent with the now
# known coins -- the verifier is bound to open the same values or abort.  Hiding (a soundness term, the prover must
# not learn r_i from c_i): in the random-oracle model q / 2^{|r_i| + |s_i|} = q 2^-512 per slot; we book it as
# 2^-192 per slot (q = 2^64 queries, |r| = 256 bits, ignoring the salt) -- see ``soundness``.

COIN_TAG = b"ligero-b/coin|v1|"
COIN_BYTES = 32
_COMMIT_COINS = True      # red-team control ONLY (redteam_zk.py): False = the pre-8c derivation, coins = H(transcript)


class CoinOpeningError(RuntimeError):
    """The verifier opened a coin slot to a value that does not match its step-0 commitment.  ``sent`` lists the
    prover messages that had been sent when the prover aborted."""

    def __init__(self, slot: int, sent: list[str]):
        super().__init__(f"coin slot {slot}: opening does not match the commitment (verifier equivocation); "
                         f"prover aborted after sending {sent}")
        self.slot, self.sent = slot, list(sent)


def coin_commit(r: bytes, s: bytes) -> bytes:
    if len(r) != COIN_BYTES or len(s) != COIN_BYTES:
        raise ValueError("coin randomness and salt are 32 bytes each")
    return hashlib.blake2b(COIN_TAG + r + s, digest_size=32).digest()


@dataclass(frozen=True)
class Coins:
    """The verifier's step-0 randomness: two slots (r_i, s_i).  As a ``CoinSource`` it is the HONEST verifier: it
    opens exactly what it committed to, whatever the prover sent."""
    r1: bytes
    s1: bytes
    r2: bytes
    s2: bytes

    @staticmethod
    def sample() -> Coins:
        import os
        return Coins(os.urandom(COIN_BYTES), os.urandom(COIN_BYTES), os.urandom(COIN_BYTES), os.urandom(COIN_BYTES))

    def well_formed(self) -> bool:
        return all(isinstance(x, bytes) and len(x) == COIN_BYTES for x in (self.r1, self.s1, self.r2, self.s2))

    def commitments(self) -> tuple[bytes, bytes]:
        return coin_commit(self.r1, self.s1), coin_commit(self.r2, self.s2)

    # CoinSource interface (what the prover calls; a malicious verifier in redteam_zk.py overrides these)
    def open1(self, root: bytes) -> tuple[bytes, bytes]:
        return self.r1, self.s1

    def open2(self, root: bytes, w: np.ndarray, h: np.ndarray, q: np.ndarray | None, v: np.ndarray | None) -> tuple[bytes, bytes]:
        return self.r2, self.s2

    def opened(self) -> Coins:
        """The values actually opened (a malicious source may differ from its fields); honest: itself."""
        return self


def _coin_prefix(cc: tuple[bytes, bytes] | None) -> bytes:
    return b"|coins=" + cc[0] + cc[1] if (_COMMIT_COINS and cc is not None) else b""


def _coin_open(r: bytes | None) -> bytes:
    return b"|r=" + r if (_COMMIT_COINS and r is not None) else b""


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


def _challenge1(stmt: bytes, root: bytes, D: int, m: int, L: int, Q: int, device, l_chain: int = 0,
                cc: tuple[bytes, bytes] | None = None, r1: bytes | None = None):
    seed = merkle.hash_bytes(b"ligero-b/c1|" + _bound(stmt) + _coin_prefix(cc) + root + _coin_open(r1))
    extra = N_FAMILIES * l_chain
    x = _expand(seed, D * (m + L + Q + extra), device).reshape(D, m + L + Q + extra)
    rc = x[:, m + L + Q:].reshape(D, N_FAMILIES, l_chain) if l_chain else None
    return x[:, :m], x[:, m:m + L], x[:, m + L:m + L + Q], rc


def _challenge2(stmt: bytes, root: bytes, w: np.ndarray, h: np.ndarray, t: int, n: int, q: np.ndarray | None = None,
                v: np.ndarray | None = None, cc: tuple[bytes, bytes] | None = None, r2: bytes | None = None) -> list[int]:
    seed = merkle.hash_bytes(b"ligero-b/c2|" + _bound(stmt) + _coin_prefix(cc) + root + w.tobytes() + h.tobytes()
                             + (q.tobytes() if q is not None else b"") + (b"|v=" + v.tobytes() if v is not None else b"")
                             + _coin_open(r2))
    cols: list[int] = []
    seen = set()
    ctr = 0
    while len(cols) < t:
        raw = hashlib.shake_256(seed + ctr.to_bytes(4, "little")).digest(4 * 2 * t)
        for u in np.frombuffer(raw, dtype="<u4"):
            c = int(u) % n
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
    w: np.ndarray            # (D, l) uint32 evaluations on H -- ZK: (D, k) coefficients of r p + s_prox
    h: np.ndarray            # (D, 2k-1-l) uint32 (l-1 without ZK)
    cols: list[int]
    opened: np.ndarray       # (M, t) uint32, M = m witness rows + the mask rows
    paths: list[list[bytes]]
    hash_name: str = merkle.HASH_NAME
    timings: dict = field(default_factory=dict)
    q: np.ndarray | None = None   # chain mode: (D, l+k-1) uint32, the chain test's combined polynomial
    v: np.ndarray | None = None   # ZK only: (D, k) coefficients of the blinded linear-test message alpha p + Z_H s_lin
    coin_commit: tuple[bytes, bytes] | None = None   # step 0: (c1, c2)
    coins: Coins | None = None                       # the openings (r1, s1), (r2, s2) as the prover received them

    def n_bytes(self) -> dict:
        b = {"root": len(self.root), "w": self.w.nbytes, "h": self.h.nbytes, "cols": 4 * len(self.cols),
             "opened_columns": self.opened.nbytes, "merkle_paths": sum(32 * len(p) for p in self.paths),
             "q_chain": self.q.nbytes if self.q is not None else 0, "v_linear": self.v.nbytes if self.v is not None else 0,
             "coin_commitments_and_openings": 6 * COIN_BYTES if self.coin_commit is not None else 0}
        b["total"] = sum(b.values())
        return b


# ---------------------------------------------------------------------------------------------------------------------
# ZK masking (honest-verifier; PROTOCOL.md section 8b as implemented)
# ---------------------------------------------------------------------------------------------------------------------
#
# Row j of the witness becomes the polynomial  p_j(x) = interp_H(W[j])(x) + Z_H(x) s_j(x),  s_j uniform of degree
# < t_pad: it agrees with the payload on H and its values at any t <= t_pad points off H are uniform (the map
# s -> (Z_H(eta_c) s(eta_c))_c is onto: Vandermonde of t distinct points, Z_H nonzero on the coset).  The evaluation
# domain is the coset g <w_n>, disjoint from H (red-team F4: no opened column is a payload position).  Per challenge
# coordinate d, five committed mask rows (all degree < k, so the proximity test covers them like any row):
#     s_prox   uniform degree < k              proximity message  w_d = r_d . p + s_prox_d            (uniform)
#     s_lin    uniform degree < t_pad          linear-test message v_d = alpha_d . p + Z_H s_lin_d    (uniform given beta)
#     s_qa     uniform degree < k              quadratic mask  Z_H (s_qa + x^k s_qb), degree < 2k-1, vanishing on H:
#     s_qb     uniform degree < t_pad - 1      h_d = P_d / Z_H + s_qa + x^k s_qb   (uniform on all 2k-1-l coefficients)
#     s_chain  uniform degree < k - 1          chain mask Z_H s_chain (sum over H = 0):  q_d = Q_d + Z_H s_chain_d
# The verifier reconstructs every mask's contribution at the opened columns from the opened mask rows and the public
# Z_H(eta_c), eta_c^k, so a dishonest mask is just another committed row (no unchecked degree of freedom).
# Randomness (8c): a ChaCha20 counter-mode stream on the device under a fresh 32-byte key, reduced to BabyBear by
# rejection (u < 2p, then u mod p): exactly uniform given the stream (mask_sampler.py; the 8b sampler's direct
# uint32 mod p had a statistical distance of ~0.11 per symbol).

_MASK_KINDS = ("prox", "lin", "qa", "qb", "chain")


def mask_rows(cfg: Config, m: int, chain: bool) -> dict[str, slice]:
    """Row index ranges of the mask rows appended after the m witness rows (D rows per kind)."""
    if not cfg.zk:
        return {}
    out, at = {}, m
    for kind in _MASK_KINDS:
        if kind == "chain" and not chain:
            continue
        out[kind] = slice(at, at + cfg.D)
        at += cfg.D
    return out


def n_rows(cfg: Config, m: int, chain: bool) -> int:
    """Committed rows: m (+ 5 D or 4 D mask rows under ZK)."""
    mr = mask_rows(cfg, m, chain)
    return max([m] + [s.stop for s in mr.values()])


USE_DEVICE_MASKS = True     # ChaCha20 counter-mode sampler on the device (mask_sampler.py); False = host SHAKE-256


def _uniform_symbols(total: int, device) -> torch.Tensor:
    """``total`` exactly uniform BabyBear symbols (int64 on ``device``), fresh randomness.  Device: ChaCha20 stream +
    rejection (mask_sampler.py).  Host fallback: SHAKE-256 stream, the same rejection (u < 2p, then u mod p) -- the
    8b sampler's direct ``u mod p`` had a 0.11 statistical distance per symbol (mask_sampler.py docstring)."""
    import os
    if USE_DEVICE_MASKS and str(device).startswith("cuda"):
        from .mask_sampler import sampler_for
        s = sampler_for(device)
        if s is not None:
            return s.uniform(total)[0]
    key = os.urandom(32)
    acc = []
    got, ctr = 0, 0
    while got < total:
        raw = hashlib.shake_256(b"ligero-b/masks|v2|" + key + ctr.to_bytes(4, "little")).digest(4 * (total - got + (total - got) // 8 + 64))
        u = np.frombuffer(raw, dtype="<u4").astype(np.int64)
        u = u[u < 2 * P] % P
        acc.append(u)
        got += u.shape[0]
        ctr += 1
    return torch.from_numpy(np.concatenate(acc)[:total]).to(device)


def _sample_masks(cfg: Config, m: int, chain: bool, device) -> dict[str, torch.Tensor]:
    """Fresh uniform mask coefficients: ``W`` (m, t_pad) per witness row and one (D, deg) block per mask kind."""
    D, k, tp = cfg.D, cfg.k, cfg.t_pad
    shapes = {"W": (m, tp), "prox": (D, k), "lin": (D, tp), "qa": (D, k), "qb": (D, tp - 1)}
    if chain:
        shapes["chain"] = (D, k - 1)
    total = sum(a * b for a, b in shapes.values())
    flat = _uniform_symbols(total, device)
    out, at = {}, 0
    for name, (a, b) in shapes.items():
        out[name] = flat[at:at + a * b].reshape(a, b)
        at += a * b
    return out


def _add_zh_times(coefs: torch.Tensor, s: torch.Tensor, l: int) -> torch.Tensor:
    """coefs (R, >= l + deg s) += Z_H(x) s(x) = x^l s(x) - s(x), in place (mod p)."""
    d = s.shape[1]
    coefs[:, :d] = (coefs[:, :d] - s) % P
    coefs[:, l:l + d] = (coefs[:, l:l + d] + s) % P
    return coefs


def _xk_on_coset(n: int, k: int, device) -> torch.Tensor:
    """eta_c^k for every column c (cached)."""
    from .field import Domain, root_of_unity
    key = ("xk", n, k, str(device))
    hit = Domain._cache.get(key)
    if hit is None:
        w = root_of_unity(n)
        gk, wk = pow(GEN, k, P), pow(w, k, P)
        vals, cur = [], gk
        for _ in range(n):
            vals.append(cur)
            cur = cur * wk % P
        hit = torch.tensor(vals, dtype=torch.int64, device=device)
        Domain._cache[key] = hit
    return hit


def _fold_to_H(coefs: torch.Tensor, l: int) -> torch.Tensor:
    """Evaluations on H of polynomials given by (R, K >= l) coefficient rows: reduce mod x^l - 1, then NTT_l."""
    from .field import ntt
    K = coefs.shape[1]
    acc_ = coefs[:, :l].clone()
    for lo in range(l, K, l):
        seg = coefs[:, lo:min(lo + l, K)]
        acc_[:, :seg.shape[1]] = (acc_[:, :seg.shape[1]] + seg) % P
    return ntt(acc_ % P)


def _sum_over_H(coefs: torch.Tensor, l: int) -> torch.Tensor:
    """sum_{h in H} q(h) = l * sum_{l | j} q_j for (R, K) coefficient rows."""
    return (l * coefs[:, ::l].sum(1)) % P


def _alpha_beta(tb: Tables, rho_lin: torch.Tensor, pub: dict, m: int, l: int, device):
    """The batched linear test's combination alpha (D, m) over the rows and its public right-hand side beta (D, l)
    (evaluations on H): sum_j alpha_j W[j] = beta for an honest witness."""
    D = rho_lin.shape[0]
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
    return alpha, beta


# ---------------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------------

USE_SIMT_ENCODER = True     # the int32 Montgomery radix-4 kernel (encode_simt.py) on CUDA; False = the torch int64 NTT


def _simt_encoder(cfg: Config, device, plain: bool = False):
    """The SIMT encoder for this configuration (``plain``: no mask fold-in, for the chain's R_i rows), or None when
    it does not apply (CPU, no cupy, or n != 4 l -- the small-l ZK configurations double n)."""
    if not USE_SIMT_ENCODER or not str(device).startswith("cuda") or cfg.n != 4 * cfg.l or cfg.l < 256:
        return None
    try:
        from . import encode_simt
        if not encode_simt.available():
            return None
        return encode_simt.encoder_for(cfg.l, cfg.n, 0 if plain or not cfg.zk else cfg.t_pad, device)
    except Exception:  # noqa: BLE001 -- no NVRTC / cupy: the torch path
        return None


def _combine_row(c: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    return ((c[:, None] * M) % P).sum(0) % P


def _combine(coef: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """(D, cols) = coef (D, R) @ M (R, cols) mod p, reducing every product before the sum (R < 2^32 terms)."""
    D = coef.shape[0]
    if witness.USE_FUSED:
        from . import tests_fused
        f = tests_fused.fused_for(D, M.device)
        if f is not None:
            return f.lincomb(coef, M)
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
          device, check: bool = False, mutate=None, chain: tuple | None = None, words: dict | None = None,
          coins: Coins | None = None) -> Proof:
    """``pub``: the public vectors (from ``public_vectors``); ``hints``: (n_hints, l) from ``gen_step``.
    ``words``: the public words the statement binds -- ``{"a", "b", "c", "y"}`` (units) or ``{"a", "b"}`` (chain;
    ``y16`` comes from ``chain``) -- absorbed into every challenge (``statement_digest``).
    ``mutate(W)``: a test hook that edits the witness in place before it is committed (the mutated-witness
    negatives); an honest prover never uses it.  ``chain = (Layout, y16_pub (l,))`` runs the chain test.
    ``coins``: the verifier's step-0 coin source (interactive: the verifier's own ``Coins`` or a malicious source
    with the same interface); ``None`` = non-interactive, the prover samples them (Fiat-Shamir reading)."""
    clk = _Clock(device)
    m, l, n, D = sys.m, cfg.l, cfg.n, cfg.D
    assert hints.shape[1] == l
    assert (chain is not None) == (sys.chain is not None), "chain mode needs a chain-compiled system"
    if words is None:
        raise ValueError("prove needs the public words (statement binding)")
    src = coins if coins is not None else Coins.sample()
    ccom = src.commitments()                                      # step 0: received before anything is computed
    sent = ["coin commitments (V)"]
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

    k = cfg.k
    mr = mask_rows(cfg, m, chain is not None)
    Mrows = n_rows(cfg, m, chain is not None)
    enc = _simt_encoder(cfg, device)                              # None: the torch int64 NTT (CPU, or no cupy)
    if cfg.zk:
        S = _sample_masks(cfg, m, chain is not None, device)
        clk.lap("zk_masks")
        # coefficient rows of every committed polynomial: witness rows interp_H(W) + Z_H s_j, then the mask rows
        coefs = torch.zeros((Mrows, k), dtype=torch.int64, device=device)
        if enc is not None:
            # SIMT kernel: codeword of interp_H(W_j) + Z_H s_j on the coset (the mask folded in per coset, 8c) and the
            # natural-order coefficients of interp_H(W_j); the 4D / 5D mask rows (degree < k) through the torch NTT
            U = torch.empty((Mrows, n), dtype=torch.int32, device=device)
            _, C = enc.encode(W, S["W"], want_coefs=True, out=U[:m])
            coefs[:m, :l] = C
        else:
            coefs[:m, :l] = intt(W % P)
        _add_zh_times(coefs[:m], S["W"], l)
        for kind, sl in mr.items():
            blk = S[kind]
            coefs[sl, :blk.shape[1]] = blk
        del S
        if enc is not None:
            U[m:] = eval_on_coset(coefs[m:], n).to(torch.int32)
        else:
            U = eval_on_coset(coefs, n)                           # (M, n), the coset g <w_n> (disjoint from H)
    else:
        coefs = None
        U = enc.encode(W)[0] if enc is not None else encode(W, n)   # (m, n); int32 from the kernel
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

    M = U.shape[0]
    sent.append("root")
    r1, s1 = src.open1(root)                                      # V opens coin slot 1 (after root)
    if len(r1) != COIN_BYTES or len(s1) != COIN_BYTES or coin_commit(r1, s1) != ccom[0]:
        raise CoinOpeningError(1, sent)                           # public-data abort, before any coin-dependent message
    r, rho_lin, rho_quad, rc = _challenge1(stmt, root, D, M, tb.L, tb.Q, device, l if chain else 0, ccom, r1)
    v_np = None
    if cfg.zk:
        r[:, mr["prox"]] = 0                                      # the proximity mask enters with coefficient 1
        w = (_combine(r, coefs) + coefs[mr["prox"]]) % P          # (D, k) coefficients, uniform
        alpha, _beta = _alpha_beta(tb, rho_lin, pub, m, l, device)
        v = _combine(alpha, coefs[:m])                            # (D, k): alpha . p, = beta on H
        _add_zh_times(v, coefs[mr["lin"]][:, :cfg.t_pad], l)     # + Z_H s_lin (the committed row is s_lin)
        v_np = v.cpu().numpy().astype(np.uint32)
    else:
        w = _combine(r, W)                                        # (D, l) evaluations on H
    clk.lap("tests_w")
    # quadratic: batched A*B - C on the codeword domain (+ the mask), divided by Z_H
    P0 = quad_p0(tb, rho_quad, U[:m])                             # (D, n): split form (witness.quad_p0)
    zinv = batch_modinv(zh_on_coset(n, l, device))
    hev = (P0 * zinv) % P
    if cfg.zk:                                                    # + s_qa + x^k s_qb  (the mask / Z_H)
        hev = (hev + U[mr["qa"]] + (_xk_on_coset(n, k, device)[None, :] * U[mr["qb"]]) % P) % P
    hcoef = intt(hev)                                             # coefficients of h(g x): c_k g^k
    hcoef = (hcoef * _ginv_powers(n, device)) % P
    if check:
        assert (hcoef[:, cfg.h_len:] == 0).all(), "quotient degree too high: constraints violated"
    h = hcoef[:, :cfg.h_len]
    clk.lap("tests_quad")
    q_np = None
    if chain is not None:
        layout, y16_pub = chain
        cc = chain_coefs(sys, layout, rc, y16_pub)
        R = cc.coef.shape[1]
        penc = _simt_encoder(cfg, device, plain=True)
        Renc = (penc.encode(cc.coef.reshape(D * R, l))[0].to(torch.int64) if penc is not None
                else encode(cc.coef.reshape(D * R, l), n)).reshape(D, R, n)     # R_i on the codeword domain
        Qev = ((Renc * U.index_select(0, cc.rows)[None, :, :]) % P).sum(1) % P   # (D, n)
        if cfg.zk:
            Qev = (Qev + (zh_on_coset(n, l, device)[None, :] * U[mr["chain"]]) % P) % P
        qcoef = (intt(Qev) * _ginv_powers(n, device)) % P
        if check:
            assert (qcoef[:, cfg.q_len:] == 0).all(), "chain polynomial degree too high"
            assert torch.equal(_sum_over_H(qcoef[:, :cfg.q_len], l), cc.rhs), "chain constraints violated"
        q_np = qcoef[:, :cfg.q_len].cpu().numpy().astype(np.uint32)
    clk.lap("tests_chain")
    clk.marks["tests"] = clk.marks["tests_w"] + clk.marks["tests_quad"] + clk.marks["tests_chain"]   # the bucket

    w_np = w.cpu().numpy().astype(np.uint32)
    h_np = h.cpu().numpy().astype(np.uint32)
    sent.append("w, h" + (", q" if q_np is not None else "") + (", v" if v_np is not None else ""))
    r2, s2 = src.open2(root, w_np, h_np, q_np, v_np)              # V opens coin slot 2 (after the test messages)
    if len(r2) != COIN_BYTES or len(s2) != COIN_BYTES or coin_commit(r2, s2) != ccom[1]:
        raise CoinOpeningError(2, sent)
    cols = _challenge2(stmt, root, w_np, h_np, cfg.t, n, q_np, v_np, ccom, r2)
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
    return Proof(root, w_np, h_np, cols, opened, paths, hash_name, timings, q_np, v_np, ccom, Coins(r1, s1, r2, s2))


# ---------------------------------------------------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------------------------------------------------

def verify(p: Params, sys: System, tb: Tables, cfg: Config, c, a, b, y, proof: Proof, device,
           chain: tuple | None = None, coins: Coins | None = None) -> tuple[bool, str]:
    """Accept/reject the claim ``y[i] = tc_dot(c[i], a[i], b[i])`` for all l units -- or, with
    ``chain = (Layout, y16_pub)`` (``c = y = None``), the claim that every VU of the layout casts to its
    ``y16_pub`` word from ``c_0 = 0`` over its 96 units.  Never touches W.
    ``coins``: interactive -- the verifier's OWN step-0 randomness; the transcript must carry exactly these openings
    (a proof made in another verifier's session is rejected).  ``None``: non-interactive -- the openings in the
    transcript are checked against the commitments in the transcript."""
    t0 = time.perf_counter()
    m, l, n, D, t, k = sys.m, cfg.l, cfg.n, cfg.D, cfg.t, cfg.k     # every size from the VERIFIER's configuration
    if (chain is not None) != (sys.chain is not None):
        return False, "chain mode mismatch"
    # step 0 / openings: c_i = H(r_i || s_i) for both slots; interactive: they are this verifier's own coins
    if _COMMIT_COINS:
        pc = proof.coins
        if proof.coin_commit is None or pc is None or not pc.well_formed() or len(proof.coin_commit) != 2:
            return False, "malformed coin commitments / openings"
        if coins is not None and (pc != coins or proof.coin_commit != coins.commitments()):
            return False, "coins are not this verifier's step-0 coins"
        if tuple(proof.coin_commit) != pc.commitments():
            return False, "coin opening does not match its commitment"
        ccom = tuple(proof.coin_commit)
    else:
        ccom, pc = None, Coins(b"\0" * 32, b"\0" * 32, b"\0" * 32, b"\0" * 32)
    mr = mask_rows(cfg, m, chain is not None)
    M = n_rows(cfg, m, chain is not None)
    if not (proof.hash_name or "").startswith("blake3"):
        return False, "proof names a hash the verifier does not use"
    try:
        pub = public_vectors(p, c, a, b, y, device=device)
    except ValueError as e:
        return False, f"public inputs: {e}"
    if len(pub["a[0].s"]) != l:
        return False, "wrong number of units"
    # shapes and ranges of the transcript
    w_shape = (D, k) if cfg.zk else (D, l)
    if proof.w.shape != w_shape or proof.h.shape != (D, cfg.h_len) or proof.opened.shape != (M, t) or len(proof.cols) != t:
        return False, "malformed proof"
    if cfg.zk != (proof.v is not None) or (cfg.zk and (proof.v.shape != (D, k) or (proof.v >= P).any())):
        return False, "malformed linear-test message"
    if len(proof.paths) != t or any(len(pth) != n.bit_length() - 1 for pth in proof.paths):
        return False, "malformed Merkle paths"
    if (proof.w >= P).any() or (proof.h >= P).any() or (proof.opened >= P).any():
        return False, "transcript element out of the field"
    if chain is not None:
        layout, y16_pub = chain
        y16_pub = torch.as_tensor(y16_pub, dtype=torch.int64)
        if layout.l != l or proof.q is None or proof.q.shape != (D, cfg.q_len) or (proof.q >= P).any():
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
    r, rho_lin, rho_quad, rc = _challenge1(stmt, proof.root, D, M, tb.L, tb.Q, device, l if chain else 0, ccom, pc.r1)
    if proof.cols != _challenge2(stmt, proof.root, proof.w, proof.h, t, n, proof.q, proof.v, ccom, pc.r2):
        return False, "column challenge mismatch"
    cols = torch.tensor(proof.cols, dtype=torch.int64, device=device)
    Uo = torch.from_numpy(proof.opened.astype(np.int64)).to(device)          # (M, t)
    zh = zh_on_coset(n, l, device)[cols]                                      # (t,)
    # (iv) Merkle
    for j, c_ in enumerate(proof.cols):
        if not merkle.verify_path(proof.root, c_, proof.opened[:, j], proof.paths[j]):
            return False, f"merkle path {j} invalid"
    # (i) proximity: w is a degree < k polynomial; it must agree with r . U (+ the prox mask row) at the columns
    w = torch.from_numpy(proof.w.astype(np.int64)).to(device)
    if cfg.zk:
        r[:, mr["prox"]] = 0
        if not torch.equal(eval_on_coset(w, n)[:, cols], (_combine(r, Uo) + Uo[mr["prox"]]) % P):
            return False, "proximity test failed"
    elif not torch.equal(encode(w, n)[:, cols], _combine(r, Uo)):
        return False, "proximity test failed"
    # (ii) linear constraints: alpha . p = beta on H; ZK: the prover sends v = alpha . p + Z_H s_lin and the
    # verifier checks v|_H = beta (public) and v at the columns against the opened rows
    alpha, beta = _alpha_beta(tb, rho_lin, pub, m, l, device)
    if cfg.zk:
        v = torch.from_numpy(proof.v.astype(np.int64)).to(device)
        if not torch.equal(_fold_to_H(v, l), beta):
            return False, "linear constraints failed (public values)"
        expect = (_combine(alpha, Uo[:m]) + (zh[None, :] * Uo[mr["lin"]]) % P) % P
        if not torch.equal(eval_on_coset(v, n)[:, cols], expect):
            return False, "linear constraints failed"
    elif not torch.equal(encode(beta, n)[:, cols], _combine(alpha, Uo)):
        return False, "linear constraints failed"
    # (iii) quadratic constraints
    lhs = quad_p0(tb, rho_quad, Uo[:m])                                       # (D, t)
    if cfg.zk:                                                                # + Z_H (s_qa + x^k s_qb)
        xk = _xk_on_coset(n, k, device)[cols]
        lhs = (lhs + (zh[None, :] * ((Uo[mr["qa"]] + (xk[None, :] * Uo[mr["qb"]]) % P) % P)) % P) % P
    h = torch.from_numpy(proof.h.astype(np.int64)).to(device)
    rhs = (eval_on_coset(h, n)[:, cols] * zh[None, :]) % P
    if not torch.equal(lhs, rhs):
        return False, "quadratic constraints failed"
    # (v) chain: the general linear test
    if chain is not None:
        cc = chain_coefs(sys, layout, rc, torch.as_tensor(y16_pub, dtype=torch.int64))
        q = torch.from_numpy(proof.q.astype(np.int64)).to(device)
        if not torch.equal(_sum_over_H(q, l), cc.rhs):
            return False, "chain constraints failed (sum over H)"
        R = cc.coef.shape[1]
        Rc = encode(cc.coef.reshape(D * R, l), n).reshape(D, R, n)[:, :, cols]          # (D, R, t)
        expect = ((Rc * Uo.index_select(0, cc.rows)[None, :, :]) % P).sum(1) % P
        if cfg.zk:
            expect = (expect + (zh[None, :] * Uo[mr["chain"]]) % P) % P
        if not torch.equal(eval_on_coset(q, n)[:, cols], expect):
            return False, "chain constraints failed (opened columns)"
    _sync(device)
    return True, f"accept ({time.perf_counter() - t0:.3f}s)"
