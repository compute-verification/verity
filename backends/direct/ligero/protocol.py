"""The Ligero-style argument for B units of the transition unit (PROTOCOL.md): prover and verifier share nothing but
the transcript (``Proof``) and the compiled ``System`` (public: a function of ``Params`` only).

    V -> P   c1 = H(r1 || s1), c2 = H(r2 || s2)   step 0 (HM96 / Goldreich-Kahan): the verifier commits to the
                                                   randomness of both of its coins BEFORE the prover's first message
    P -> V   root                          Merkle root of the codeword columns of U = Enc(W)
    V -> P   (r1, s1)                      opening of coin slot 1; P checks c1 and aborts on mismatch
             r, rho_lin, rho_quad          D independent BabyBear combinations each: SHAKE(H(stmt, c1, c2, r1))
    P -> V   w = r W  (D x l),  h  (D x (l-1))   proximity message; quotient of the batched quadratic test by Z_H
    V -> P   (r2, s2)                      opening of coin slot 2; P checks c2
             cols                          t column indices: H(stmt, c1, c2, r2)
    P -> V   U[:, cols], Merkle paths

Three verifier messages on the critical path (c, r1, r2): ``rounds.sequential_depth = 3`` (PROTOCOL.md 8c).  The
challenges are functions of the verifier's coins alone (no prover message enters them: lane b-zk-fix, red team F2).
``Config.mode = "fiat-shamir"`` is the OTHER protocol: no step 0, challenges ``H(stmt, root)`` and
``H(stmt, root, w, h, q, v)``, transcript without coins, soundness under the ``q x eps`` rule (``soundness``); the
two modes are mutually exclusive (``prove`` / ``verify`` reject the other mode's coin usage; the statement digest
binds the mode).

Verifier: (i) Enc(w)[cols] = r U[cols]; (ii) Enc(beta)[cols] = alpha U[cols] with alpha = sum rho_lin,q a_q,
beta = sum rho_lin,q (pub_q - c_q) -- the linear constraints, whose right-hand sides are public, so nothing is sent;
(iii) sum_q rho_quad,q (A_q(U) B_q(U) - C_q(U))[cols] = h(eta_c) Z_H(eta_c); (iv) the Merkle paths.
Fiat-Shamir with SHAKE-256 as the challenge expander (placeholder hash, named in the proof).
"""
from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import torch
from verity_numerical.checker.params import Params
from verity_numerical.security import accounting as acc

from . import auth as auth_mod
from . import chain_term, merkle, witness
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

__all__ = ["BF16_HOOKS", "MODES", "MODE_FIAT_SHAMIR", "MODE_INTERACTIVE", "CoinOpeningError", "Coins", "Config", "Proof",
           "RelationHooks", "batch_bound", "coin_commit", "config_for", "log2_sum", "prove", "soundness", "verify"]


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
    mode: str = "interactive"   # "interactive" (8c: verifier coins, challenges from the coins alone) | "fiat-shamir"

    @property
    def interactive(self) -> bool:
        return self.mode == MODE_INTERACTIVE

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
FS_ZK_T_PAD = 512   # Fiat-Shamir mode: t ~ 288 at 2^-128 no longer fits 256 (red team F3 / ZK_CONSTRUCTION.md 3.3)

MODE_INTERACTIVE = "interactive"
MODE_FIAT_SHAMIR = "fiat-shamir"
MODES = (MODE_INTERACTIVE, MODE_FIAT_SHAMIR)

# Fiat-Shamir soundness rule of the campaign (ZK_CONSTRUCTION.md 3.3, zk_cost.fiat_shamir_reparameterisation): the
# generic state-restoration bound eps_FS <= q x eps_round with q = 2^60 hash queries.  Every statistical term of the
# interactive protocol is multiplied by 2^60 in ``soundness``; ``config_for`` picks t (and the extension degree D)
# so that the union over the sub-batches still meets the target.
FS_QUERIES_LOG2 = 60


def _field_terms_ok(D: int, n: int, k: int, l: int, t: int, per_proof: float, zk: bool, shift_log2: int,
                    chain_extras: int = 0) -> bool:
    """Appendix-C query terms (with the k that sized t) + field terms (n + 3) / p^D + the chain test's field term,
    times 2^shift, vs the target."""
    e = (n - k) // 2
    query = Fraction(n - e, n) ** t + Fraction(k + l, n) ** t + Fraction(2 * k, n) ** t
    field = Fraction(n + 3, P ** D) + chain_term.field_term(chain_extras, D)
    return acc.log2((query + field) * Fraction(2) ** shift_log2) <= per_proof


def config_for(l: int, target_log2: float = -128.0, rate_log2: int = 2, n_proofs: int = 1, zk: bool = False,
               t_pad: int | None = None, mode: str = MODE_INTERACTIVE, chain_extras: int = 0) -> Config:
    """t from Appendix C with k = l + 1 (no ZK padding: the conservative stand-in for k = l that ``soundness`` reports
    and ligero-verify gates on) or k = l + t_pad (ZK: the padding is t_pad random *coefficients*
    of Z_H(x) s(x), so k need not be a power of two and n stays l << rate_log2 -- section 8b's "n = 32768" was the
    cost of rounding k to a power of two, which the coefficient form avoids) and the extension degree from
    ``security.accounting``: BabyBear^6 for 2^-128, BabyBear^4 for 2^-80 (the sensitivity point).  ``n_proofs > 1``:
    the batch is proved as that many independently committed sub-batches, so each proof gets
    ``accounting.per_proof_target`` (the union bound over the sub-batches meets ``target_log2``); ``t`` grows by
    ~log2(n_proofs) / 0.68 columns.

    ``mode``: ``interactive`` (statistical bound, live verifier coins) or ``fiat-shamir`` (the per-proof target is
    shifted by ``FS_QUERIES_LOG2``: more opened columns, ``t_pad = 512`` under ZK so Lemma 4.15 still holds, and the
    smallest extension degree whose field terms survive the 2^60 factor -- D = 7 at 2^-128).

    ``chain_extras``: the system's chain constraints with monomial coefficients (``chain.extra_constraints``); their
    field term (``chain_term.field_term``) comes out of the query budget, so ``t`` is sized with it."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    fs = mode == MODE_FIAT_SHAMIR
    if t_pad is None:
        t_pad = FS_ZK_T_PAD if fs else ZK_T_PAD
    n = l << rate_log2
    per_proof = acc.per_proof_target(target_log2, n_proofs) if n_proofs > 1 else target_log2
    shift = FS_QUERIES_LOG2 if fs else 0
    query_target = per_proof - shift
    D0 = 6 if target_log2 <= -100 else 4
    if chain_extras:
        # the queries keep at least half the budget: a larger D rather than many more columns (Fiat-Shamir: D = 7)
        while float(chain_term.field_term(chain_extras, D0)) * 2 > 2.0 ** query_target:
            D0 += 1
            if D0 > 12:
                raise acc.AccountingError("no extension degree <= 12 leaves the chain test's field term under the target")
        query_target = math.log2(2.0 ** query_target - float(chain_term.field_term(chain_extras, D0)))
    k = l + (t_pad if zk else 0)
    k_acc = k if zk else l + 1            # the k ``soundness`` (and ligero-verify) evaluates the bound at (BV-D1)
    while n <= 2 * k:                     # small-l ZK configurations (l <= t_pad): the quadratic test needs n > 2k
        n *= 2
    t = acc.ligero_queries_for(query_target, n, k_acc, l)
    while zk and not k > l + t:           # small l: the padding lowers the rate; double n until Lemma 4.15 holds
        n *= 2
        t = acc.ligero_queries_for(query_target, n, k_acc, l)
    # field terms: (n + 1 + 2) / p^D must stay under the (shifted) target; D = 6 at -128 interactive, 7 Fiat-Shamir
    D = D0
    while not _field_terms_ok(D, n, k_acc, l, t, per_proof, zk, shift, chain_extras):
        D += 1
        if D > 12:
            raise acc.AccountingError("no extension degree <= 12 meets the target")
    return Config(l, n, D, t, target_log2, zk, t_pad if zk else 0, mode)


FP_COIN_MAX_PROB = chain_term.COIN_MAX_PROB   # a coin is a uint32 reduced mod p (_expand): no residue has > 3 preimages


def fingerprint_collision(coords: int, steps: int, l: int, n_op: int) -> Fraction:
    """The fingerprint constraints' collision term (chain.py, "Fingerprint constraints"; PROTOCOL.md "Row sharing"): one
    of the ``n_op x (l // steps)`` fingerprinted operand slots of a proof holds a row of ``steps`` columns x 16 lanes
    that differs from the shared row it is matched to, yet agrees with it on all ``coords`` coordinates.  Per slot and
    coordinate the difference of the two fingerprints is a nonzero polynomial of degree < 16 steps in rho_x
    (Schwartz-Zippel), so the event has probability <= (16 steps - 1) x max_r Pr[rho_x = r]; the coins are uint32 mod p
    (``_expand``, no rejection), so that maximum is 3 / 2^32 rather than 1 / p (+0.49 bits per coordinate); the
    coordinates are independent; the union is over every slot (a VU's row and its W column are two events)."""
    return n_op * (l // steps) * ((16 * steps - 1) * FP_COIN_MAX_PROB) ** coords


def soundness(cfg: Config, n_linear: int, n_quadratic: int, n_proofs: int = 1, fp: tuple[int, int, int] | None = None,
              chain_extras: int = 0) -> dict:
    """The error terms actually achieved (Appendix C, e = (n - k) / 2, uniform combinations); with ``n_proofs``
    sub-batches the reported total is ``accounting.union_over_proofs`` (never the per-proof error).  Fiat-Shamir
    mode: every statistical term is multiplied by 2^FS_QUERIES_LOG2 (the campaign's ``q x eps`` rule) and no coin
    term is booked (there is no step 0).  ``fp = (coords, steps, n_op)``: the proof carries fingerprint constraints in the
    coin mode (the GEMM side of row sharing): the ``fingerprint_collision`` term is booked per proof over its
    ``n_op x (l // steps)`` fingerprinted slots -- the event that a VU's operand row differs from the shared row it is
    matched to yet has the same fingerprints (a statistical term: rho is drawn after both rows are committed, so it
    takes the Fiat-Shamir factor as well).  ``chain_extras``: the system's chain constraints with monomial coefficients
    (``chain.extra_constraints``, more than three linked rows); their field term ``chain_field``
    (``chain_term.field_term``, red-team-bligero-real-k A1) is booked per proof, with the Fiat-Shamir factor."""
    F = acc.Field.prime("BabyBear", P).extension(cfg.D)
    k = cfg.k if cfg.zk else cfg.l + 1                                        # k = l + 1: validate needs l < k
    p = acc.LigeroParams(n=cfg.n, k=k, l=cfg.l, t=cfg.t, zk=cfg.zk)
    terms = acc.ligero_error(p, acc.Field.prime("BabyBear", P), F, combination="uniform")
    if chain_extras:
        terms = {**terms, "chain_field": chain_term.field_term(chain_extras, cfg.D)}
    if fp is not None:
        coords, steps, n_op = fp
        terms = {**terms, "fingerprint_collision": fingerprint_collision(coords, steps, cfg.l, n_op)}
    fs_factor = Fraction(2) ** FS_QUERIES_LOG2 if not cfg.interactive else Fraction(1)
    terms = {name: v * fs_factor for name, v in terms.items()}
    bud = acc.Budget("B-Ligero direct (backends/direct/ligero)", {"n": cfg.n, "k": k, "l": cfg.l, "t": cfg.t, "D": cfg.D, "mode": cfg.mode})
    for name, v in terms.items():
        bud.add("encoding_opening", f"Ligero {name}" + (f" x 2^{FS_QUERIES_LOG2} (Fiat-Shamir)" if not cfg.interactive else ""), v)
    if cfg.interactive:
        # step 0 (8c): the prover must not learn r_i from c_i = H(r_i || s_i).  Random-oracle hiding: q 2^-|r| with
        # q = 2^64 hash queries and |r| = 256 bits -> 2^-192 per slot (the salt is not even counted), two slots.
        # Binding (a second opening = a collision of H, 2^-128 generic) protects the PROVER's privacy, not soundness,
        # and is reported separately (privacy_terms_log2).
        hide = COIN_HIDING_TERM
        terms = {**terms, "coin_commitment_hiding": hide}
        bud.add("commitment_hash", "HM96 step-0 coin commitments: hiding vs the prover, 2 slots x q 2^-256 (q = 2^64)", hide)
    per_proof_total = bud.total()
    union = acc.union_over_proofs(bud, n_proofs) if n_proofs > 1 else bud
    return {"terms_log2": {name: acc.log2(v) for name, v in terms.items()}, "per_proof_total_log2": acc.log2(per_proof_total),
            **({"chain_extras": chain_extras, "chain_degree": chain_term.degree(chain_extras)} if chain_extras else {}),
            "n_proofs": n_proofs, "total_log2": acc.log2(union.total()),
            "per_proof_target_log2": acc.per_proof_target(cfg.target_log2, n_proofs) if n_proofs > 1 else cfg.target_log2,
            "field": F.name, "t": cfg.t, "n": cfg.n, "k": cfg.k, "l": cfg.l, "rate": cfg.rate, "zk": cfg.zk, "t_pad": cfg.t_pad,
            "mode": cfg.mode, "coin_commitment": cfg.interactive,
            "fiat_shamir_queries_log2": None if cfg.interactive else FS_QUERIES_LOG2,
            "bound_model": ("interactive: statistical, live verifier coins committed at step 0 (HM96); no Fiat-Shamir loss"
                            if cfg.interactive else
                            f"non-interactive: eps_FS <= 2^{FS_QUERIES_LOG2} x eps_round (BCS16 state restoration; "
                            "ZK_CONSTRUCTION.md 3.3); challenges = H(statement, transcript prefix)"),
            "hash_assumption": ("collision resistance of Blake2b (coin commitments) + Blake3 (Merkle); the challenge "
                                "expander is a PRG of the verifier's coins" if cfg.interactive else
                                "random oracle for the challenge hash (soundness AND zero knowledge)"),
            "zk_statement": ((("malicious-verifier ZK (simulator by rewinding V* after step 0, CRH only); "
                               "operands a, b and words y16 are PUBLIC in this statement, so privacy is vacuous today")
                              if cfg.interactive else
                              "HVZK; NIZK only in the programmable random-oracle model; operands a, b and words y16 PUBLIC "
                              "in this statement, so privacy is vacuous today") if cfg.zk else "none (no masking)"),
            "privacy_terms_log2": ({"coin_commitment_binding_collision": acc.log2(acc.hash_collision_bound(256, 2**64)),
                                    "mask_sampling_bias_per_symbol": "0 (rejection sampling; the simulator's distance is "
                                                                      "ChaCha20's PRF advantage on the mask key)"} if cfg.interactive else {}),
            "note": ("ZK: k = l + t_pad, every row carries t_pad > t uniform coefficients (Lemma 4.15)" if cfg.zk else
                     "k = l + 1 stands in for the non-ZK k = l (the accounting's validate() needs l < k)") + "; "
                    "the linear test here sends nothing (the verifier encodes beta itself), so its query term is "
                    "at most the Appendix-C ((k+l)/n)^t used; total_log2 is the union bound over n_proofs sub-batches"
                    + ("; + the step-0 coin-commitment hiding term (commitment_hash)" if cfg.interactive else
                       f"; every statistical term x 2^{FS_QUERIES_LOG2} (Fiat-Shamir, q x eps)")}


COIN_HIDING_TERM = 2 * Fraction(2**64, 2**256)      # 2 slots x q / 2^|r|, q = 2^64


# ---------------------------------------------------------------------------------------------------------------------
# Batch verification: the union bound over the sub-batches presented (PROTOCOL.md 8c, "Batch verification")
# ---------------------------------------------------------------------------------------------------------------------
#
# A batch is N independently committed sub-batch proofs.  Each proof's bound 2^b_i (b_i = per_proof_total_log2 of
# ``soundness()``, recomputed by the verifier from THAT proof's parameters) is a statement about one commitment; the
# batch claim "sound to 2^-target" is true only through the union bound sum_i 2^b_i <= 2^-target, and a proof's
# n_proofs is a commitment to its batch only if it equals the number of proofs actually presented.  Neither was a
# verifier check before lane b-batch-bound: N proofs each claiming n_proofs = 1 passed on their own 2^-132.9 while the
# batch was 2^-128.3.  The same rule, the same reason strings, in backends/ligero-verify (batch_bound in verify.rs).

def _bits_str(bits: float) -> str:
    """The target in a reason string: ``128`` for an integer, else two decimals (identical in the Rust verifier)."""
    return f"{int(bits)}" if float(bits).is_integer() else f"{bits:.2f}"


def log2_sum(xs: list[float]) -> float:
    """log2(sum_i 2^x_i) without leaving the log domain (the x_i are ~ -130)."""
    if not xs:
        return float("-inf")
    m = max(xs)
    if m == float("-inf"):
        return m
    return m + math.log2(sum(2.0 ** (x - m) for x in xs))


def batch_bound(per_proof_log2: list[float], n_proofs_claimed: list[int], target_bits: float = 128.0,
                names: list[str] | None = None) -> dict:
    """The batch verifier's accounting gate on N = len(per_proof_log2) sub-batch proofs that each verified on their own.
    ``per_proof_log2[i]`` is the verifier's recomputation of proof i's bound (``soundness(cfg_i, ...)["per_proof_total_log2"]``,
    never a prover-supplied number); ``n_proofs_claimed[i]`` is the count proof i's statement was sized for.  Accepts iff
    every claimed count equals N and ``sum_i 2^b_i <= 2^-target_bits``.  ``batch_log2`` = log2 of the sum (the batch's
    bits are its negative); reason strings are shared with the Rust verifier."""
    N = len(per_proof_log2)
    if N == 0 or len(n_proofs_claimed) != N:
        raise ValueError("batch_bound needs one bound and one claimed n_proofs per sub-batch")
    names = names or [f"sub_{i:02d}" for i in range(N)]
    total = log2_sum([float(b) for b in per_proof_log2])
    out = {"n_subbatches": N, "target_bits": float(target_bits), "batch_log2": total,
           "per_proof_log2": [float(b) for b in per_proof_log2], "n_proofs_claimed": [int(k) for k in n_proofs_claimed]}
    for name, k in zip(names, n_proofs_claimed):
        if int(k) != N:
            return {**out, "accepted": False, "reason": f"batch: n_proofs mismatch ({name}: n_proofs = {int(k)}, {N} sub-batches presented)"}
    if total > -float(target_bits):
        return {**out, "accepted": False,
                "reason": f"batch: union bound 2^{total:.2f} over {N} sub-batches exceeds the target 2^-{_bits_str(target_bits)}"}
    return {**out, "accepted": True,
            "reason": f"accept: all {N} sub-batches accepted; union bound 2^{total:.2f} <= 2^-{_bits_str(target_bits)}"}


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


@dataclass(frozen=True)
class RelationHooks:
    """What ``prove`` / ``verify`` need from a relation beyond its compiled ``System``: the verifier's public decode and
    the statement's word widths.  The default (``BF16_HOOKS``) is the BF16 unit of ``unit.py`` and reproduces the
    pre-hook transcript byte for byte; ``fp8/`` supplies its own (``--relation fp8-ada``)."""
    public_vectors: object
    word_dtype: str = "<u2"        # operand words a, b in the statement digest
    y16_dtype: str = "<u2"         # the chain-end public words
    y16_bits: int | None = None    # range of the chain-end words (None: 2^p.word_bits)
    tag: bytes = b""               # domain separation of a non-default relation (b"" for BF16)
    # The VU shape (red-team-leaf-3 H1 / H2; verifier-only, never absorbed): the compiled system is column-uniform, so it
    # does not fix how many columns a VU spans.  ``steps``: the relation's columns per VU (``Relation.steps``); a chain
    # statement with another layout is refused.  ``max_steps``: the leaf's structural bound (``AjtaiLeaf.max_steps`` = n:
    # past it the negacyclic digest is not injective).  None: unchecked (the unit-mode / test systems).
    steps: int | None = None
    max_steps: int | None = None


BF16_HOOKS = RelationHooks(public_vectors)


def layout_error(layout, rel: RelationHooks) -> str | None:
    """The chain statement's layout against the relation's VU shape (the Rust verifier's ``check_vu_shape``)."""
    if rel.max_steps is not None and layout.steps > rel.max_steps:
        return (f"statement: steps = {layout.steps} columns per VU exceeds the leaf's bound {rel.max_steps} "
                "(ring degree: the digest is not binding beyond it)")
    if rel.steps is not None and layout.steps != rel.steps:
        return f"statement: steps = {layout.steps} columns per VU, the relation's VU is {rel.steps} columns"
    return None

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
        if sys.chain.get("y16") is not None:
            h.update(repr([e.key() for e in sys.chain["y"]] + [sys.chain["y16"].key()]).encode())
        else:   # component ends (chain.py): the expression AND the bit field it is compared against
            h.update(repr([e.key() for e in sys.chain["y"]]).encode())
            h.update(repr([("end", Y.key(), int(lo), int(bits)) for Y, lo, bits in sys.chain["y_end"]]).encode())
        if sys.chain.get("fp") is not None:     # the fingerprint constraints (chain.py): lane rows, coordinates, mode
            fp = sys.chain["fp"]
            h.update(repr(("fp", [list(map(int, op)) for op in fp["lanes"]], int(fp["coords"]), fp["mode"])).encode())
    d = h.digest()
    _SYS_ID[key] = (sys, d)
    return d


def _u_bytes(x, dtype: str) -> bytes:
    arr = np.ascontiguousarray(np.asarray(x if not isinstance(x, torch.Tensor) else x.cpu().numpy(), dtype=np.int64))
    return arr.astype(dtype).tobytes()


def statement_digest(sys: System, cfg: Config, a, b, c=None, y=None, layout=None, y16=None,
                     rel: RelationHooks = BF16_HOOKS, auth: auth_mod.AuthStatement | None = None, fp=None) -> bytes:
    """The public statement as the verifier sees it (see the module comment above).  ``auth`` (chain mode,
    ``authentication = included``): the committed-object block of ``auth.py`` -- the three tree references (binding,
    owner, count, root) and the claimed VU indices -- so the challenges are bound to "the words at these leaf indices
    under these roots", not just to the words.  ``fp`` (chain mode, a fingerprinted chain in the "public" mode of
    chain.py -- the hash side of row sharing): the ``FpArgs`` (rho, F) the fingerprint constraints take as statement
    values."""
    h = hashlib.blake2b(digest_size=32)
    h.update(STATEMENT_TAG)
    h.update(rel.tag)
    h.update(f"|p={P}|hash=blake3|l={cfg.l}|n={cfg.n}|D={cfg.D}|t={cfg.t}|target={cfg.target_log2!r}".encode())
    h.update(f"|zk={int(cfg.zk)}|k={cfg.k}|mode={cfg.mode}".encode())
    h.update(b"|sys=" + system_id(sys))
    if layout is not None:
        h.update(f"|mode=chain|l={layout.l}|n_vus={layout.n_vus}|steps={layout.steps}".encode())
    else:
        h.update(b"|mode=unit")
    h.update(b"|a=" + _u_bytes(a, rel.word_dtype))
    h.update(b"|b=" + _u_bytes(b, rel.word_dtype))
    if layout is not None:
        h.update(b"|y16=" + _u_bytes(y16, rel.y16_dtype))
    else:
        h.update(b"|c=" + _u_bytes(c, "<u4"))
        h.update(b"|y=" + _u_bytes(y, "<u4"))
    if auth is not None:
        if layout is None:
            raise ValueError("authentication is defined for the chain statement only")
        h.update(auth_mod.digest_bytes(auth))
    if fp is not None:
        if layout is None:
            raise ValueError("public fingerprints are defined for the chain statement only")
        h.update(b"|fp_rho=" + _u_bytes(fp.rho, "<u4"))
        h.update(b"|fp_F=" + _u_bytes(fp.F, "<u4"))
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
#
# Challenge derivation (lane b-zk-fix, red team F2).  In the INTERACTIVE mode the challenges are functions of the
# verifier's coins ALONE:  r, rho, rc = SHAKE(H("c1" || stmt || c1 c2 || r1)),  cols = H("c2" || stmt || c1 c2 || r2)
# -- neither root nor w, h, q, v enters.  This is what lets the simulator work with a collision-resistant hash only:
# after the rewind it knows (r1, r2), hence r, rho, rc and cols, BEFORE it builds root and the messages, so it can
# prepare a tableau consistent with them without programming any hash.  Soundness is unaffected: the coins are
# hidden from the prover by c_i until it has sent the message the coin must follow (hiding term), and the prover
# cannot bias r1 / r2 (they are the verifier's), so r and cols are uniform and independent of root / w / h / q / v
# exactly as in the public-coin protocol -- the statement and the commitments in the seed are domain separation,
# not entropy.  In the FIAT-SHAMIR mode there is no step 0 and the challenges are H(stmt, transcript prefix) as in
# the pre-8c protocol; the random-oracle model is then the assumption for soundness AND for zero knowledge.

COIN_TAG = b"ligero-b/coin|v1|"
COIN_BYTES = 32


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
    return b"|coins=" + cc[0] + cc[1] if cc is not None else b""


def _coin_open(r: bytes | None) -> bytes:
    return b"|r=" + r if r is not None else b""


# ---------------------------------------------------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------------------------------------------------

def _expand(seed: bytes, count: int, device) -> torch.Tensor:
    """count uniform-ish BabyBear elements from SHAKE-256(seed) (uint32 reduced mod p; bias 2^-31 relative, noted)."""
    raw = hashlib.shake_256(seed).digest(4 * count)
    if str(device).startswith("cuda"):
        # the same uint32 -> int64 -> mod p reduction, on the device: one 4-byte-per-element copy instead of the host
        # astype + mod + 8-byte copy (lane hostphase; identical values)
        u = torch.from_numpy(np.frombuffer(raw, dtype="<u4").view(np.int32).copy()).to(device)
        return (u.to(torch.int64) & 0xFFFFFFFF) % P
    arr = np.frombuffer(raw, dtype="<u4").astype(np.int64) % P
    return torch.from_numpy(arr).to(device)


def _bound(stmt: bytes) -> bytes:
    return stmt if _BIND_STATEMENT else b""


def seed1_preimage(stmt: bytes, root: bytes, cc: tuple[bytes, bytes] | None, r1: bytes | None) -> bytes:
    """Interactive (``cc, r1`` given): the verifier's coin alone -- ``H(tag || stmt || c1 c2 || r1)``, no root.
    Fiat-Shamir (``cc is None``): ``H(tag || stmt || root)``."""
    if cc is not None:
        if r1 is None:
            raise ValueError("interactive challenge 1 needs the opened coin r1")
        return b"ligero-b/c1|v2|" + _bound(stmt) + _coin_prefix(cc) + _coin_open(r1)
    return b"ligero-b/c1|" + _bound(stmt) + root


def seed2_preimage(stmt: bytes, root: bytes, w: np.ndarray, h: np.ndarray, q: np.ndarray | None, v: np.ndarray | None,
                   cc: tuple[bytes, bytes] | None, r2: bytes | None, fp: np.ndarray | None = None) -> bytes:
    """Interactive: ``H(tag || stmt || c1 c2 || r2)`` -- none of root, w, h, q, v.  Fiat-Shamir: the transcript, the
    fingerprint message ``fp`` (chain.py "coin" mode) included after v."""
    if cc is not None:
        if r2 is None:
            raise ValueError("interactive challenge 2 needs the opened coin r2")
        return b"ligero-b/c2|v2|" + _bound(stmt) + _coin_prefix(cc) + _coin_open(r2)
    return (b"ligero-b/c2|" + _bound(stmt) + root + w.tobytes() + h.tobytes()
            + (q.tobytes() if q is not None else b"") + (b"|v=" + v.tobytes() if v is not None else b"")
            + (b"|fp=" + fp.tobytes() if fp is not None else b""))


def fp_coin_coords(sys: System) -> int:
    """The fingerprint coins challenge 1 expands for ``sys``: ``coords`` in the chain's "coin" mode (chain.py), else 0."""
    fp = sys.chain.get("fp") if sys.chain is not None else None
    return int(fp["coords"]) if fp is not None and fp["mode"] == "coin" else 0


def _challenge1(stmt: bytes, root: bytes, D: int, m: int, L: int, Q: int, device, l_chain: int = 0,
                cc: tuple[bytes, bytes] | None = None, r1: bytes | None = None, n_fp: int = 0):
    """The challenge block: per coordinate ``r | rho_lin | rho_quad | rc (7 l)`` and, for a fingerprinted chain in the
    coin mode, ``n_fp`` more elements of which coordinate 0's are the fingerprint coins rho (the others are unused)."""
    seed = merkle.hash_bytes(seed1_preimage(stmt, root, cc, r1))
    extra = N_FAMILIES * l_chain
    width = m + L + Q + extra + n_fp
    x = _expand(seed, D * width, device).reshape(D, width)
    rc = x[:, m + L + Q:m + L + Q + extra].reshape(D, N_FAMILIES, l_chain) if l_chain else None
    rho = x[0, width - n_fp:] if n_fp else None
    return x[:, :m], x[:, m:m + L], x[:, m + L:m + L + Q], rc, rho


def _challenge2(stmt: bytes, root: bytes, w: np.ndarray, h: np.ndarray, t: int, n: int, q: np.ndarray | None = None,
                v: np.ndarray | None = None, cc: tuple[bytes, bytes] | None = None, r2: bytes | None = None,
                fp: np.ndarray | None = None) -> list[int]:
    seed = merkle.hash_bytes(seed2_preimage(stmt, root, w, h, q, v, cc, r2, fp))
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
    auth: list[list[bytes]] | None = None            # authentication = included: one Merkle multiproof per tree (a, b, y)
    fp: np.ndarray | None = None                     # fingerprint message (chain.py coin mode): (n_vus, n_op, C) uint32 after coin 1
    # NOT part of the proof (never serialized, never read by a verifier): the coins rho the prover's challenge 1 expanded,
    # kept for the prover of the paired hash-side proof whose statement takes them (relchain.SharedHashedRunner)
    rho: np.ndarray | None = None

    def n_bytes(self) -> dict:
        b = {"root": len(self.root), "w": self.w.nbytes, "h": self.h.nbytes, "cols": 4 * len(self.cols),
             "opened_columns": self.opened.nbytes, "merkle_paths": sum(32 * len(p) for p in self.paths),
             "q_chain": self.q.nbytes if self.q is not None else 0, "v_linear": self.v.nbytes if self.v is not None else 0,
             "coin_commitments_and_openings": 6 * COIN_BYTES if self.coin_commit is not None else 0,
             "auth_multiproofs": sum(32 * len(s) for s in self.auth) if self.auth is not None else 0,
             "fingerprints": self.fp.nbytes if self.fp is not None else 0}
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
#     u_ca     degree < k                      chain mask u_d = u_ca + x^k u_cb: degree < l + k - 1 = q_len, uniform
#     u_cb     uniform degree < l - 1          subject to sum_H u = 0 (u_ca[0] adjusted):  q_d = Q_d + u_d
#                                              (AHIV22 u_add; red team F1: the old Z_H s_chain vanished ON H, so q|_H
#                                              was Q|_H in the clear -- every accumulator recoverable)
# The verifier reconstructs every mask's contribution at the opened columns from the opened mask rows and the public
# Z_H(eta_c), eta_c^k, so a dishonest mask is just another committed row (no unchecked degree of freedom).
# Randomness (8c): a ChaCha20 counter-mode stream on the device under a fresh 32-byte key, reduced to BabyBear by
# rejection (u < 2p, then u mod p): exactly uniform given the stream (mask_sampler.py; the 8b sampler's direct
# uint32 mod p had a statistical distance of ~0.11 per symbol).

_MASK_KINDS = ("prox", "lin", "qa", "qb", "chain_a", "chain_b")
_CHAIN_KINDS = ("chain_a", "chain_b")


def mask_rows(cfg: Config, m: int, chain: bool) -> dict[str, slice]:
    """Row index ranges of the mask rows appended after the m witness rows (D rows per kind)."""
    if not cfg.zk:
        return {}
    out, at = {}, m
    for kind in _MASK_KINDS:
        if kind in _CHAIN_KINDS and not chain:
            continue
        out[kind] = slice(at, at + cfg.D)
        at += cfg.D
    return out


def n_rows(cfg: Config, m: int, chain: bool) -> int:
    """Committed rows: m (+ 6 D or 4 D mask rows under ZK)."""
    mr = mask_rows(cfg, m, chain)
    return max([m] + [s.stop for s in mr.values()])


USE_DEVICE_MASKS = True     # ChaCha20 counter-mode sampler on the device (mask_sampler.py); False = host SHAKE-256


def _uniform_symbols(total: int, device, key_dev: torch.Tensor | None = None) -> torch.Tensor:
    """``total`` exactly uniform BabyBear symbols (int64 on ``device``), fresh randomness.  Device: ChaCha20 stream +
    rejection (mask_sampler.py).  Host fallback: SHAKE-256 stream, the same rejection (u < 2p, then u mod p) -- the
    8b sampler's direct ``u mod p`` had a 0.11 statistical distance per symbol (mask_sampler.py docstring).
    ``key_dev``: the ChaCha key already on the device (drawn by the caller; the commit graph's static input)."""
    import os
    if USE_DEVICE_MASKS and str(device).startswith("cuda"):
        from .mask_sampler import sampler_for
        s = sampler_for(device)
        if s is not None:
            return s.uniform(total, key_dev=key_dev)[0]
    if key_dev is not None:
        raise RuntimeError("device mask key without the device sampler")
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


def _sample_masks(cfg: Config, m: int, chain: bool, device, key_dev: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Fresh uniform mask coefficients: ``W`` (m, t_pad) per witness row and one (D, deg) block per mask kind.
    ``key_dev``: the sampler key on the device (the commit graph draws it eagerly and derives inside the graph)."""
    D, k, tp, l = cfg.D, cfg.k, cfg.t_pad, cfg.l
    shapes = {"W": (m, tp), "prox": (D, k), "lin": (D, tp), "qa": (D, k), "qb": (D, tp - 1)}
    if chain:
        shapes["chain_a"] = (D, k)
        shapes["chain_b"] = (D, l - 1)
    total = sum(a * b for a, b in shapes.values())
    flat = _uniform_symbols(total, device, key_dev)
    out, at = {}, 0
    for name, (a, b) in shapes.items():
        out[name] = flat[at:at + a * b].reshape(a, b).clone()
        at += a * b
    if chain:
        # u = u_ca + x^k u_cb (degree < l + k - 1) must sum to zero over H: sum_H u = l * sum_{l | j} u_j.  Fix the
        # constant coefficient (index 0, itself one of the l | j terms) so that the sum vanishes; the other
        # l + k - 2 coefficients stay uniform -- u is uniform on the zero-sum subspace (AHIV22 u_add).
        s = _sum_over_H(_chain_mask_coefs(cfg, out["chain_a"], out["chain_b"]), l)          # (D,)
        out["chain_a"][:, 0] = (out["chain_a"][:, 0] - s * modinv(l)) % P
    return out


def _chain_mask_coefs(cfg: Config, ca: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
    """(D, l + k - 1) coefficients of u = u_ca + x^k u_cb."""
    u = torch.zeros((ca.shape[0], cfg.q_len), dtype=torch.int64, device=ca.device)
    u[:, :cfg.k] = ca
    u[:, cfg.k:cfg.k + cb.shape[1]] = (u[:, cfg.k:cfg.k + cb.shape[1]] + cb) % P
    return u


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


def _zh_inv_on_coset(n: int, l: int, device) -> torch.Tensor:
    """1 / Z_H(eta_c) for every column c: ``batch_modinv(zh_on_coset(...))``, cached per (n, l, device) -- the Fermat
    ladder is ~60 kernels per proof otherwise (lane hostphase)."""
    from .field import Domain
    key = ("zh_inv", n, l, str(device))
    hit = Domain._cache.get(key)
    if hit is None:
        hit = batch_modinv(zh_on_coset(n, l, device))
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


def _pubmat(tb: Tables, pub: dict, device) -> torch.Tensor | None:
    """(Lp, l) the public columns the linear constraints read, reduced mod p (None without public columns) -- the
    per-sub-batch input of ``_alpha_beta`` (a static buffer of the tests graph receives it)."""
    pub_q, _ = _lin_pub_index(tb, device)
    if not pub_q:
        return None
    return torch.stack([pub[tb.lin_pub[q]].to(device) for q in pub_q]) % P



def _capture_failed(e: BaseException) -> None:
    """A CUDA-graph capture that failed falls back to eager -- but SILENTLY it hides real breaks (lane merge-val: a device
    sync inside the captured body left torch's generator in capture state and the *other* in-flight sub-batch's replay
    raised 'Offset increment outside graph capture').  Warn once per exception text; LIGERO_GRAPH_STRICT=1 re-raises."""
    import os, warnings
    if os.environ.get("LIGERO_GRAPH_STRICT"):
        raise e
    msg = f"CUDA graph capture failed, running eagerly: {type(e).__name__}: {str(e)[:200]}"
    if msg not in _CAPTURE_WARNED:
        _CAPTURE_WARNED.add(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)


_CAPTURE_WARNED: set = set()

def _alpha_beta(tb: Tables, rho_lin: torch.Tensor, pub: dict | None, m: int, l: int, device,
                pubmat: torch.Tensor | None = None, with_beta: bool = True):
    """The batched linear test's combination alpha (D, m) over the rows and its public right-hand side beta (D, l)
    (evaluations on H): sum_j alpha_j W[j] = beta for an honest witness.  ``pubmat``: ``_pubmat(tb, pub, device)``
    precomputed (the tests graph), else it is built from ``pub`` here.  ``with_beta=False``: (alpha, None) (the ZK
    prover sends v = alpha . p, never beta)."""
    D = rho_lin.shape[0]
    alpha = torch.zeros((D, m), dtype=torch.int64, device=device)
    terms = (rho_lin[:, tb.lin_q] * tb.lin_coef[None, :]) % P                 # (D, nnz)
    alpha.index_add_(1, tb.lin_j, terms)
    alpha %= P
    if not with_beta:
        return alpha, None
    beta = torch.zeros((D, l), dtype=torch.int64, device=device)
    const = ((rho_lin * ((-tb.lin_c) % P)[None, :]) % P).sum(1) % P           # (D,)
    beta += const[:, None]
    pub_q, qidx = _lin_pub_index(tb, device)
    if pub_q:
        # one stack + one reduction (was one ``% P`` kernel per public column; the index tensor is cached) -- hostphase
        if pubmat is None:
            pubmat = _pubmat(tb, pub, device)                                         # (Lp, l)
        beta = (beta + _combine(rho_lin.index_select(1, qidx), pubmat)) % P
    return alpha, beta


_LIN_PUB: dict = {}


def _lin_pub_index(tb: Tables, device):
    key = (id(tb), str(device))
    hit = _LIN_PUB.get(key)
    if hit is None or hit[0] is not tb:
        pub_q = [q for q, name in enumerate(tb.lin_pub) if name is not None]
        hit = (tb, pub_q, torch.tensor(pub_q, dtype=torch.int64, device=device))
        _LIN_PUB[key] = hit
    return hit[1], hit[2]


# ---------------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------------

USE_SIMT_ENCODER = True     # the int32 Montgomery radix-4 kernel (encode_simt.py) on CUDA; False = the torch int64 NTT


def _simt_encoder(cfg: Config, device, plain: bool = False):
    """The SIMT encoder for this configuration (``plain``: no mask fold-in, for the chain's R_i rows), or None when
    it does not apply (CPU, no cupy, or n/l not in {2, 4, 8}, l outside [256, 16384])."""
    if (not USE_SIMT_ENCODER or not str(device).startswith("cuda") or cfg.n not in (2 * cfg.l, 4 * cfg.l, 8 * cfg.l)
            or cfg.l < 256 or cfg.l > 16384):
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


def _intt_ginv(g, x: torch.Tensor, n: int, device) -> torch.Tensor:
    """``intt(x) * g^-i mod p`` (x canonical (D, n) int64): one fused kernel when available, else ``g("intt", ...)``."""
    gi = _ginv_powers(n, device)
    if witness.USE_FUSED:
        from . import tests_fused
        f = tests_fused.fused_for(x.shape[0], x.device)
        if f is not None and f.intt_ok(x):
            return f.intt_scaled(x, gi)
    return (g("intt", intt, x) * gi) % P


def _fused_lincomb2(C1: torch.Tensor, C2: torch.Tensor, M: torch.Tensor):
    """The fused kernels when they take (C1 @ M, C2 @ M[:R2]) in one pass here, else None."""
    if not witness.USE_FUSED:
        return None
    from . import tests_fused
    f = tests_fused.fused_for(C1.shape[0], M.device)
    return f if f is not None and f.lincomb2_ok(C1, C2, M) else None


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


NTT_GRAPHS = True      # the prover's three torch NTT chains (h, q quotients; mask-row coset evaluation) as CUDA graphs (lane hp2-host)
_TGRAPHS: dict = {}


class _TensorGraph:
    """CUDA graph of ``fn(x)`` for one fixed input shape / dtype: captured once (two eager warm-up runs on a side stream,
    then the capture), replayed as ONE launch -- the torch NTT is 16 compiled butterfly stages, each ~0.1 ms of host
    dispatch (dynamo guards, slices, cat), i.e. 5-7 ms of main-thread time per NTT for < 1 ms of GPU work; the prover
    runs three per sub-batch.  Keyed per stream (static buffers must not be shared by in-flight sub-batches)."""

    def __init__(self, fn, x: torch.Tensor):
        self.inp = torch.zeros_like(x)
        s = torch.cuda.Stream(device=x.device)
        s.wait_stream(torch.cuda.current_stream(x.device))
        with torch.cuda.stream(s):
            for _ in range(2):
                fn(self.inp)
        torch.cuda.current_stream(x.device).wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = fn(self.inp)

    def run(self, x: torch.Tensor) -> torch.Tensor:
        self.inp.copy_(x)
        self.graph.replay()
        return self.out.clone()


def _graphed(name: str, fn, x: torch.Tensor) -> torch.Tensor:
    """``fn(x)`` through a per-(name, shape, dtype, device, stream) CUDA graph on CUDA (``NTT_GRAPHS``), eagerly otherwise."""
    if not NTT_GRAPHS or x.device.type != "cuda":
        return fn(x)
    key = (name, tuple(x.shape), x.dtype, str(x.device), torch.cuda.current_stream(x.device).cuda_stream)
    g = _TGRAPHS.get(key)
    if g is None:
        try:
            g = _TensorGraph(fn, x.contiguous())
        except Exception as e:  # noqa: BLE001 -- capture unsupported here: eager
            g = False
            _capture_failed(e)
        _TGRAPHS[key] = g
    return g.run(x) if g else fn(x)


HOST_TRACE = False      # ``_Clock.host`` marks: wall time between the prover's host-side points (diagnostics, timings["host_*"])
EVENT_CLOCK = True      # phase laps as CUDA events (read once at the end) instead of a device sync per lap (lane hp2-host)


class _Clock:
    """Phase timer.  CUDA with ``EVENT_CLOCK``: every lap records an event on the current stream and the phase times are
    read once at ``total()`` (event-to-event = the stream's timeline between the marks, host gaps included; ``total`` is
    wall-clock after one stream sync) -- no host sync per phase, so the phases of a sub-batch can overlap.  Otherwise
    (CPU, or ``EVENT_CLOCK = False``): the legacy sync-per-lap timer."""

    def __init__(self, device):
        self.device = device
        self.marks: dict[str, float] = {}
        self.events = str(device).startswith("cuda") and EVENT_CLOCK
        self._laps: list = []
        if self.events:
            self._last_ev = torch.cuda.Event(enable_timing=True)
            self._last_ev.record()
        else:
            _sync(device)
        self.t0 = time.perf_counter()
        self.last = self.t0

    def host(self, name: str):
        """Host-side (wall) time since the previous host mark, as ``host_<name>`` -- only with ``HOST_TRACE`` (diagnostics)."""
        if HOST_TRACE:
            now = time.perf_counter()
            self.marks["host_" + name] = self.marks.get("host_" + name, 0.0) + (now - self.last)
            self.last = now

    def lap(self, name: str):
        if self.events:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._laps.append((name, self._last_ev, ev))
            self._last_ev = ev
            return
        _sync(self.device)
        now = time.perf_counter()
        self.marks[name] = self.marks.get(name, 0.0) + (now - self.last)
        self.last = now

    def total(self):
        if self.events:
            torch.cuda.current_stream(self.device).synchronize()
            for name, a, b in self._laps:
                self.marks[name] = self.marks.get(name, 0.0) + a.elapsed_time(b) / 1e3
            self._laps = []
        else:
            _sync(self.device)
        return time.perf_counter() - self.t0


# ---------------------------------------------------------------------------------------------------------------------
# Prover
# ---------------------------------------------------------------------------------------------------------------------
#
# ``prove`` runs ``_prove_stages`` -- the prover as a generator that yields at the points where the host must wait for
# something (pipeline.py): the root's D2H copy; the SHAKE-256 challenge expansion (squeezed on a worker thread into
# pinned memory); the test messages' D2H copy; the openings' D2H copy.  Sequentially (``prove``) the generator is driven
# to completion on the caller's stream: the same transcript as before, with the per-object ``.cpu()`` syncs replaced by
# one pinned non-blocking copy per message and the phase clock replaced by events.  ``pipeline.prove_many`` interleaves
# several sub-batches' generators on their own streams (lane hp2-host).  Bit-exactness: every value that reaches the
# transcript is the same tensor as before, copied through pinned memory (int32 <-> uint32 bit patterns of residues < p).

def _pin(slot, device, name: str, nbytes: int) -> torch.Tensor:
    """A pinned uint8 buffer of >= ``nbytes`` for ``name`` in this slot (the caller's slot, or the device's default slot)."""
    from . import pipeline
    if slot is None:
        slot = pipeline.slot_for(device, 0)
    return slot.pinned(name, nbytes)


def _event(device, blocking: bool = False):
    """An event on the current stream.  ``blocking`` (pipelined prover): the host waits without spinning a core -- the
    squeeze worker and the other slots' launches need the CPU; sequential prover: the default spin-wait keeps the main
    thread's core hot for the inline squeeze that follows the root (measured 7.4 ms hot vs ~10 ms after a sleep)."""
    if str(device).startswith("cuda"):
        ev = torch.cuda.Event(blocking=blocking)
        ev.record()
        return ev
    return None


def _cupy_ctx(device):
    from . import pipeline
    if str(device).startswith("cuda"):
        try:
            return pipeline.cupy_stream_context()
        except Exception:  # noqa: BLE001 -- no cupy: nothing to redirect
            pass
    return pipeline._NullContext()


def _expand_pinned(seed: bytes, count: int, device, slot, name: str):
    """(future, finish): ``future`` completes when SHAKE-256(seed) has been squeezed into pinned memory (worker thread,
    GIL released -- pipeline.shake_into); ``finish()`` then uploads it (non-blocking) and reduces mod p on the device:
    the same ``uint32 -> int64 & 0xFFFFFFFF % p`` map as ``_expand``."""
    from . import pipeline
    nbytes = 4 * count
    buf = _pin(slot, device, name, nbytes)[:nbytes]
    fut = pipeline.squeeze(seed, buf.numpy(), inline=slot is None)

    def finish(raw_out: torch.Tensor | None = None):
        if raw_out is not None:                                    # the tests graph's static input: reduced in the graph
            raw_out.copy_(buf.view(torch.int32), non_blocking=True)
            return None
        u = buf.view(torch.int32).to(device, non_blocking=True)
        return (u.to(torch.int64) & 0xFFFFFFFF) % P

    return fut, finish


def _fp_setup(sys: System, x: torch.Tensor, M: int, tb: Tables, l_chain: int, lanes, fp_pub, layout):
    """The fingerprint data of the chain test (chain.py): coin mode -- rho from the challenge block's tail and F the
    honest fingerprints of the lane rows (the prover's message); public mode -- ``fp_pub`` as given.  ``(FpArgs, F)``
    with F None in the public mode; ``(None, None)`` for a chain without fingerprint constraints."""
    from .chain import FpArgs, fp_values
    spec = sys.chain.get("fp") if sys.chain is not None else None
    if spec is None:
        return None, None
    if spec["mode"] == "coin":
        C = int(spec["coords"])
        base = M + tb.L + tb.Q + N_FAMILIES * l_chain
        rho = x[0, base:base + C]
        if lanes is None:
            raise ValueError("fingerprint coin mode: the prover needs the lane rows")
        F = fp_values(lanes, rho, len(spec["lanes"]), layout.steps, l_chain)
        return FpArgs(rho, F), F
    if fp_pub is None:
        raise ValueError("fingerprint public mode: the chain tuple must carry (Layout, y16, FpArgs)")
    return fp_pub, None


FUSED_CHAINQ = True     # the chain quotient's codeword-domain sum through tests_fused.chain_quotient (bit-exact)


def _fused_chainq(D: int, U: torch.Tensor):
    if not (FUSED_CHAINQ and witness.USE_FUSED and U.is_cuda and U.dtype == torch.int32):
        return None
    from . import tests_fused
    return tests_fused.fused_for(D, U.device)


def _tests_compute(sys: System, tb: Tables, cfg: Config, x: torch.Tensor, coefs, U, W, pub, pubmat, y16_pub, layout,
                   mr: dict, device, check: bool = False, nested_graphs: bool = True, lanes=None, fp_pub=None):
    """The prover's tests from the challenge block ``x`` (D, width): (w, h, q, v, F) on the device -- the batched linear
    test (``w``; ``v`` = alpha . p in the ZK mode), the quadratic quotient ``h``, the chain quotient ``q`` (chain
    mode) and the fingerprint message ``F`` (a fingerprinted chain in the coin mode, from the lane rows ``lanes``;
    else None).  Shared by the eager prover and the captured tests graph (``nested_graphs=False`` while capturing: the
    NTT chains run eagerly inside the capture)."""
    m, l, n, k, D = sys.m, cfg.l, cfg.n, cfg.k, cfg.D
    M = U.shape[0]
    l_chain = l if layout is not None else 0
    r, rho_lin, rho_quad = x[:, :M], x[:, M:M + tb.L], x[:, M + tb.L:M + tb.L + tb.Q]
    rc_hi = M + tb.L + tb.Q + N_FAMILIES * l_chain
    rc = x[:, M + tb.L + tb.Q:rc_hi].reshape(D, N_FAMILIES, l_chain) if l_chain else None
    fp, F = _fp_setup(sys, x, M, tb, l_chain, lanes, fp_pub, layout) if l_chain else (None, None)
    g = (lambda name, fn, t: _graphed(name, fn, t)) if nested_graphs else (lambda name, fn, t: fn(t))
    v = None
    if cfg.zk:
        r[:, mr["prox"]] = 0                                      # the proximity mask enters with coefficient 1
        alpha, _ = _alpha_beta(tb, rho_lin, pub, m, l, device, pubmat=pubmat, with_beta=False)
        ft = _fused_lincomb2(r, alpha, coefs)
        if ft is not None:                                        # both combinations in one pass over coefs
            rw, v = ft.lincomb2(r, alpha, coefs)
            w = (rw + coefs[mr["prox"]]) % P
        else:
            w = (_combine(r, coefs) + coefs[mr["prox"]]) % P      # (D, k) coefficients, uniform
            v = _combine(alpha, coefs[:m])                        # (D, k): alpha . p, = beta on H
        _add_zh_times(v, coefs[mr["lin"]][:, :cfg.t_pad], l)     # + Z_H s_lin (the committed row is s_lin)
    else:
        w = _combine(r, W)                                        # (D, l) evaluations on H
    # quadratic: batched A*B - C on the codeword domain (+ the mask), divided by Z_H
    P0 = quad_p0(tb, rho_quad, U[:m])                             # (D, n): split form (witness.quad_p0)
    zinv = _zh_inv_on_coset(n, l, device)
    hev = (P0 * zinv) % P
    if cfg.zk:                                                    # + s_qa + x^k s_qb  (the mask / Z_H)
        hev = (hev + U[mr["qa"]] + (_xk_on_coset(n, k, device)[None, :] * U[mr["qb"]]) % P) % P
    hcoef = _intt_ginv(g, hev, n, device)                         # coefficients of h(g x): c_k g^k, times g^-k
    if check:
        assert (hcoef[:, cfg.h_len:] == 0).all(), "quotient degree too high: constraints violated"
    h = hcoef[:, :cfg.h_len]
    q = None
    if layout is not None:
        cc = chain_coefs(sys, layout, rc, y16_pub, fp)
        R = cc.coef.shape[1]
        penc = _simt_encoder(cfg, device, plain=True)
        # R_i on the codeword domain, in chunks of CHAIN_ENC_CHUNK coefficient rows: a leaf/ composition links 2n + 3
        # rows (131 / 259 for ajtai-n64 / n128) and one (D, R, n) int64 tensor plus its product would be 1.6 GB each at
        # l = 16384 (the n128 bench OOMed at 24 GB); the partial sums are reduced mod P, so the result is bit-identical.
        # Each chunk takes the fused int32 pass when available (one chunk, R <= CHAIN_ENC_CHUNK, is the unchunked path).
        ft = _fused_chainq(D, U)
        Qev = None
        for r0 in range(0, R, CHAIN_ENC_CHUNK):
            r1 = min(R, r0 + CHAIN_ENC_CHUNK)
            cf, rows = cc.coef[:, r0:r1].reshape(D * (r1 - r0), l), cc.rows[r0:r1]
            if ft is not None and penc is not None:
                part = ft.chain_quotient(penc.encode(cf)[0].reshape(D, r1 - r0, n), U, rows)
            else:
                Renc = (penc.encode(cf)[0].to(torch.int64) if penc is not None else encode(cf, n)).reshape(D, r1 - r0, n)
                part = ((Renc * U.index_select(0, rows)[None, :, :]) % P).sum(1) % P   # (D, n)
            Qev = part if Qev is None else (Qev + part) % P
        if cfg.zk:                                                # + u_ca + x^k u_cb (uniform, zero sum over H)
            Qev = (Qev + U[mr["chain_a"]] + (_xk_on_coset(n, k, device)[None, :] * U[mr["chain_b"]]) % P) % P
        qcoef = _intt_ginv(g, Qev, n, device)
        if check:
            assert (qcoef[:, cfg.q_len:] == 0).all(), "chain polynomial degree too high"
            assert torch.equal(_sum_over_H(qcoef[:, :cfg.q_len], l), cc.rhs), "chain constraints violated"
        q = qcoef[:, :cfg.q_len]
    return w, h, q, v, F


CHAIN_ENC_CHUNK = int(os.environ.get("LIGERO_CHAIN_ENC_CHUNK", "64"))   # chain-test coefficient rows encoded per chunk (bounded memory for wide chains)
TESTS_GRAPH = True     # the whole tests chain (challenge block -> w, h, q, v in pinned memory) as ONE CUDA graph (lane hp2-host)
TESTS_STATIC_KEEP = 2  # static sets kept per (stream, system), most recent first: a run whose last sub-batch is ragged
#                        (relchain's hash side at l_H = 8192 then 4096) alternates two configurations on one stream
_TESTS_STATIC: dict = {}


class _TestsStatic:
    """The static buffers of the tests graph for one (stream, configuration): the committed codeword ``U`` and the
    coefficient rows ``coefs`` (the encoder writes them in place, so the graph reads them at fixed addresses), the
    witness ``W`` (non-ZK: a copy), the raw challenge block ``xraw`` (the squeeze, uploaded from pinned memory), the
    public columns ``pubmat`` and the chain ends ``y16``; the graph writes w, h, q, v into the slot's pinned ``msgs``
    buffer (D2H memcpy nodes).  ~45 launches and ~5 ms of main-thread dispatch per sub-batch become one replay."""

    def __init__(self, key, cfg: Config, sys: System, tb: Tables, m: int, Mrows: int, width: int, chain: bool, device,
                 steps: int = 0):
        self.key, self.sys, self.tb = key, sys, tb
        n, k, l, D = cfg.n, cfg.k, cfg.l, cfg.D
        self.U = torch.empty((Mrows, n), dtype=torch.int32, device=device)
        # int32 residues (< p < 2^31): the encoder returns int32 coefficients, the fused tests read u32 -- half the
        # traffic of int64 for the 431 MB coefficient matrix (the copy into it and the three combinations over it)
        self.coefs = torch.empty((Mrows, k), dtype=torch.int32, device=device) if cfg.zk else None
        self.W = None if cfg.zk else torch.empty((m, l), dtype=torch.int32, device=device)
        self.xraw = torch.empty((D * width,), dtype=torch.int32, device=device)
        pub_q, _ = _lin_pub_index(tb, device)
        self.pubmat = torch.empty((len(pub_q), l), dtype=torch.int64, device=device) if pub_q else None
        self.y16 = torch.empty((l,), dtype=torch.int64, device=device) if chain else None
        # a fingerprinted chain (chain.py): the lane rows copied out of the witness at commit (coin mode: F is computed from
        # them inside the tests graph) or the statement's rho / F (public mode), static inputs of the graph
        self.lanes = self.fp_rho = self.fp_F = None
        spec = sys.chain.get("fp") if (chain and sys.chain is not None) else None
        fp_shape = None
        if spec is not None:
            from .chain import fp_lane_rows
            self.lane_rows = fp_lane_rows(sys.chain).to(device)
            C, n_op = int(spec["coords"]), len(spec["lanes"])
            fp_shape = (l // steps, n_op, C)
            if spec["mode"] == "coin":
                self.lanes = torch.empty((len(self.lane_rows), l), dtype=torch.int64, device=device)
            else:
                self.fp_rho = torch.empty((C,), dtype=torch.int64, device=device)
                self.fp_F = torch.empty(fp_shape, dtype=torch.int64, device=device)
        # the message layout (int32 residues, ``msgs`` pinned buffer): w | h | q | v | fp
        shapes = [("w", (D, k if cfg.zk else l)), ("h", (D, cfg.h_len))]
        if chain:
            shapes.append(("q", (D, cfg.q_len)))
        if cfg.zk:
            shapes.append(("v", (D, k)))
        if spec is not None and spec["mode"] == "coin":
            shapes.append(("fp", fp_shape))
        self.offs, at = {}, 0
        for name, shape in shapes:
            self.offs[name] = (at, shape)
            at += 4 * math.prod(shape)
        self.nbytes = at
        # the tests graph per chain layout ((l, n_vus, steps) or None): the chain test's coefficient masks are a
        # function of the layout (a ragged last sub-batch has its own), captured per layout on the shared buffers
        self.graphs: dict = {}     # layout key -> (CUDAGraph | False, pinned msgs data_ptr)
        # the commit graph's static inputs (witness kernel + masks + encoder -> U, coefs): public rows, hints, mask key
        self.pub_rows = torch.empty((len(sys.pins), l), dtype=torch.int64, device=device)
        self.hints = torch.empty((len(sys.hints), l), dtype=torch.int64, device=device)
        self.mkey = torch.empty((8,), dtype=torch.int32, device=device)
        self.commit_graph = None   # None: not captured yet; False: eager
        # the Merkle commit inside the commit graph (MERKLE_GRAPH): static tree buffers + the pinned root it lands in
        self.merkle = self.root_pin = None
        if MERKLE_GRAPH and merkle.gpu_available():
            from . import openings_device
            self.merkle = openings_device.StaticMerkle(Mrows, n, device)
            self.root_pin = torch.empty((32,), dtype=torch.uint8, pin_memory=True)


def _graph_run(holder, attr: str, body, device) -> None:
    """Run ``body()`` through the CUDA graph stored at ``holder.<attr>``: captured at the first call (two eager
    warm-ups on a side stream, then the capture), replayed afterwards; ``False`` (capture failed here) = eager."""
    g = getattr(holder, attr)
    if g is None:
        try:
            cur = torch.cuda.current_stream(device)
            s = torch.cuda.Stream(device=device)
            s.wait_stream(cur)
            with torch.cuda.stream(s):
                for _ in range(2):
                    body()
            cur.wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                body()
        except Exception as e:  # noqa: BLE001 -- capture unsupported here: eager
            g = False
            _capture_failed(e)
        setattr(holder, attr, g)
    if g:
        g.replay()
    else:
        body()


COMMIT_GRAPH = True    # witness kernel + mask sampler + encoder + mask-row coset NTT as ONE CUDA graph per stream (lane hp2-host)
MERKLE_GRAPH = True    # ... + the Blake3 Merkle commit and the root's D2H copy in the same graph (21 launches + 1 memcpy node)


def _fused_witness(sys: System, device):
    """The fused witness kernel for ``sys`` when the device witness path applies (else None: no commit graph)."""
    from . import witness as wmod
    if wmod.IMPL != "device" or not sys.pins or not str(device).startswith("cuda"):
        return None
    try:
        from .witness_device import fused_for
        return fused_for(sys, device)
    except Exception:  # noqa: BLE001
        return None


def _tests_static(cfg: Config, sys: System, tb: Tables, m: int, Mrows: int, width: int, chain: bool, device, steps: int = 0):
    """The static set of the current stream, system and configuration (up to ``TESTS_STATIC_KEEP`` per (stream,
    system): the paired prover of relchain.SharedHashedRunner alternates two systems on one stream, and each keeps its
    buffers and captured graphs; the least recently used configuration is dropped beyond that)."""
    skey = (str(device), torch.cuda.current_stream(device).cuda_stream, id(sys))
    key = (cfg.l, cfg.n, cfg.k, cfg.D, cfg.zk, cfg.t_pad, cfg.h_len, cfg.q_len, m, Mrows, width, chain, steps)
    sets = _TESTS_STATIC.setdefault(skey, [])
    st = next((s for s in sets if s.key == key and s.sys is sys and s.tb is tb), None)
    if st is not None:
        sets.remove(st)
    else:
        st = _TestsStatic(key, cfg, sys, tb, m, Mrows, width, chain, device, steps)
        while len(sets) >= TESTS_STATIC_KEEP:
            old = sets.pop()
            for g in [g_ for g_, _ in old.graphs.values()] + [old.commit_graph]:
                if g:
                    g.reset()
    sets.insert(0, st)
    return st


def _tests_graph_run(st: _TestsStatic, sys: System, tb: Tables, cfg: Config, mr: dict, layout, mbuf: torch.Tensor,
                     device) -> None:
    """Run the tests chain on the static set: replay the graph (captured at the first call for this stream and
    configuration: two eager warm-ups on a side stream, then the capture), or eagerly if the capture failed."""
    D = cfg.D

    def body():
        from .chain import FpArgs
        x = ((st.xraw.to(torch.int64) & 0xFFFFFFFF) % P).reshape(D, -1)
        fp_pub = FpArgs(st.fp_rho, st.fp_F) if st.fp_rho is not None else None
        w, h, q, v, F = _tests_compute(sys, tb, cfg, x, st.coefs, st.U, st.W, None, st.pubmat, st.y16, layout, mr,
                                       device, check=False, nested_graphs=False, lanes=st.lanes, fp_pub=fp_pub)
        for name, tsr in (("w", w), ("h", h), ("q", q), ("v", v), ("fp", F)):
            if tsr is None:
                continue
            o, shape = st.offs[name]
            mbuf[o:o + 4 * tsr.numel()].view(torch.int32).view(shape).copy_(tsr.to(torch.int32).contiguous(),
                                                                             non_blocking=True)

    lkey = (layout.l, layout.n_vus, layout.steps) if layout is not None else None
    hit = st.graphs.get(lkey)
    if hit is not None and hit[1] != mbuf.data_ptr():              # the pinned buffer moved (grown): re-capture
        if hit[0]:
            hit[0].reset()
        hit = None
    if hit is None:
        try:
            cur = torch.cuda.current_stream(device)
            s = torch.cuda.Stream(device=device)
            s.wait_stream(cur)
            with torch.cuda.stream(s):
                for _ in range(2):
                    body()
            cur.wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                body()
        except Exception as e:  # noqa: BLE001 -- capture unsupported here: eager on the static set
            g = False
            _capture_failed(e)
        hit = (g, mbuf.data_ptr())
        st.graphs[lkey] = hit
    if hit[0]:
        hit[0].replay()
    else:
        body()


def prove(p: Params, sys: System, tb: Tables, cfg: Config, pub: dict[str, torch.Tensor], hints: torch.Tensor,
          device, check: bool = False, mutate=None, chain: tuple | None = None, words: dict | None = None,
          coins: Coins | None = None, rel: RelationHooks = BF16_HOOKS, auth: auth_mod.AuthStatement | None = None,
          auth_open=None, hints_fn=None) -> Proof:
    """``pub``: the public vectors (from ``public_vectors``); ``hints``: (n_hints, l) from ``gen_step``.
    ``words``: the public words the statement binds -- ``{"a", "b", "c", "y"}`` (units) or ``{"a", "b"}`` (chain;
    ``y16`` comes from ``chain``) -- absorbed into every challenge (``statement_digest``).
    ``mutate(W)``: a test hook that edits the witness in place before it is committed (the mutated-witness
    negatives); an honest prover never uses it.  ``chain = (Layout, y16_pub (l,))`` runs the chain test.
    ``coins``: the verifier's step-0 coin source -- REQUIRED in the interactive mode (the verifier's own ``Coins`` or
    a malicious source with the same interface) and FORBIDDEN in the Fiat-Shamir mode (there is no step 0; the
    challenges are hashes of the transcript).  The two modes are mutually exclusive (red team F3).
    ``auth`` (chain mode): the committed-object block absorbed into the statement (``auth.py``); ``auth_open()``
    returns the three Merkle multiproofs the transcript then carries (prover work, timed as ``auth_openings``).
    ``hints_fn``: with ``hints=None``, a callable producing the hints inside the prover (timed ``hints_host``, excluded
    from ``timings["total"]`` as the runners' separately measured hints were)."""
    from . import pipeline
    gen = _prove_stages(p, sys, tb, cfg, pub, hints, device, check, mutate, chain, words, coins, rel, auth, auth_open,
                        hints_fn, None)
    return pipeline.run_to_completion(gen)


def prove_stages(*args, slot=None, fp_sink=None, **kw):
    """The prover as a stage generator for ``pipeline.prove_many`` (same arguments as ``prove`` plus ``slot``, ``fp_sink``)."""
    return _prove_stages(*args, **kw, slot=slot, fp_sink=fp_sink)


def _coin_future(src, name: str, *args):
    """A coin source's ``name`` (``commitments`` / ``open1`` / ``open2``) as a waitable, when it offers one: a source
    that talks to a live verifier (``live.LiveCoins``) has ``<name>_future(*args)`` returning a ``Future`` the stage
    YIELDS instead of blocking on the socket -- the pipeline driver runs the other sub-batches meanwhile and resumes this
    one when the reply is in.  ``None`` for a local ``Coins`` (the blocking call is used)."""
    fn = getattr(src, name + "_future", None)
    return fn(*args) if fn is not None else None


def _prove_stages(p: Params, sys: System, tb: Tables, cfg: Config, pub: dict[str, torch.Tensor], hints, device,
                  check=False, mutate=None, chain=None, words=None, coins=None, rel: RelationHooks = BF16_HOOKS,
                  auth=None, auth_open=None, hints_fn=None, slot=None, fp_sink=None):
    """``chain[2]`` (a fingerprinted chain's public ``FpArgs``) may be a ``Future``: it is awaited after the root (the
    commitment does not depend on it; the statement digest and the tests do).  ``fp_sink``: a ``Future`` set to
    ``(rho, F)`` as soon as a coin-mode fingerprint message is on the host (before the openings)."""
    from concurrent.futures import Future
    from . import pipeline
    _t_enter = time.perf_counter()
    clk = _Clock(device)
    if HOST_TRACE:
        clk.marks["host_clock_init"] = time.perf_counter() - _t_enter
    m, l, n, D = sys.m, cfg.l, cfg.n, cfg.D
    assert (chain is not None) == (sys.chain is not None), "chain mode needs a chain-compiled system"
    if words is None:
        raise ValueError("prove needs the public words (statement binding)")
    if (auth is not None) != (auth_open is not None) or (auth is not None and chain is None):
        raise ValueError("authentication needs the chain mode, an AuthStatement and an opening function")
    if cfg.interactive:
        if coins is None:
            raise ValueError("interactive mode: prove needs the verifier's step-0 coin source (coins=...)")
        src = coins
        # step 0: V's coin commitments, received before any prover MESSAGE.  A non-blocking source hands them out as a
        # Future (its ROOT is held back until they are in: ``live.LiveCoins``); resolved right after ``open1`` below.
        ccom_fut = _coin_future(src, "commitments")
        ccom = src.commitments() if ccom_fut is None else None
        sent = ["coin commitments (V)"]
    else:
        if coins is not None:
            raise ValueError("fiat-shamir mode has no step 0: coins must be None")
        src, ccom, sent = None, None, []
    cuda = str(device).startswith("cuda")
    # the statement digest (blake2b of the public words, ~1.6 ms of host hashing) on a worker thread; needed at
    # challenge 1 only.  ``words["y16"]`` (a host copy of the chain's y16) keeps the worker off the device.
    fp_later = chain is not None and len(chain) > 2 and isinstance(chain[2], Future)
    if chain is not None:
        y16_for_digest = words.get("y16", chain[1])
        fp_public = len(chain) > 2 and (sys.chain.get("fp") or {}).get("mode") == "public"

        def submit_stmt(fp_args):
            return pipeline.POOL.submit(statement_digest, sys, cfg, words["a"], words["b"], layout=chain[0],
                                        y16=y16_for_digest, rel=rel, auth=auth, fp=fp_args if fp_public else None)
        stmt_fut = None if fp_later else submit_stmt(chain[2] if len(chain) > 2 else None)
    else:
        stmt_fut = pipeline.POOL.submit(statement_digest, sys, cfg, words["a"], words["b"], c=words["c"], y=words["y"], rel=rel)
    clk.lap("statement")
    clk.host("stmt_submit")
    if hints is None:
        hints = hints_fn()
        clk.lap("hints_host")
    clk.host("hints")
    assert hints.shape[1] == l
    k = cfg.k
    mr = mask_rows(cfg, m, chain is not None)
    Mrows = n_rows(cfg, m, chain is not None)
    enc = _simt_encoder(cfg, device)                              # None: the torch int64 NTT (CPU, or no cupy)
    l_chain = l if chain else 0
    n_fp = fp_coin_coords(sys) if chain else 0
    width = Mrows + tb.L + tb.Q + N_FAMILIES * l_chain + n_fp     # the challenge block: r | rho_lin | rho_quad | rc | rho_fp
    fp_spec = sys.chain.get("fp") if chain else None
    fp_pub = chain[2] if (chain is not None and len(chain) > 2) else None
    if fp_spec is not None and fp_spec["mode"] == "public" and fp_pub is None:
        raise ValueError("fingerprint public mode: chain = (Layout, y16, FpArgs)")
    # the graphs' static set (CUDA + SIMT encoder, no self-check asserts): U and coefs live in it
    st = (_tests_static(cfg, sys, tb, m, Mrows, width, chain is not None, device, chain[0].steps if chain else 0)
          if TESTS_GRAPH and cuda and enc is not None and not check else None)
    lane_rows = None
    if fp_spec is not None:                                       # the static set holds its own (a pageable upload syncs)
        from .chain import fp_lane_rows
        lane_rows = st.lane_rows if st is not None else fp_lane_rows(sys.chain).to(device)
    fw = _fused_witness(sys, device) if (st is not None and COMMIT_GRAPH and mutate is None) else None
    graph_merkle = False
    if fw is not None:
        # the commit graph: witness kernel -> (masks ->) encoder (-> mask-row coset NTT), U / coefs in the static
        # set.  Eager part: the public rows and the hints into the static inputs, the mask key drawn here (same
        # os.urandom order as the eager prover) and uploaded from pinned memory.
        st.pub_rows.copy_(torch.stack([pub[name].to(device) for _, name in sys.pins]))
        st.pub_rows %= P
        st.hints.copy_(hints)
        clk.lap("witness")
        if cfg.zk:
            import os
            mkey = os.urandom(32)
            kbuf = _pin(slot, device, "mkey", 32)[:32]
            kbuf.numpy()[:] = np.frombuffer(mkey, dtype=np.uint8)
            st.mkey.copy_(kbuf.view(torch.int32), non_blocking=True)
        clk.lap("zk_masks")

        def commit_body():
            Wg = fw.run(st.pub_rows, st.hints)                       # (m, l) int32
            if st.lanes is not None:                                  # the fingerprint's lane rows (coin mode)
                st.lanes.copy_(Wg.index_select(0, st.lane_rows))
            if cfg.zk:
                with _cupy_ctx(device):
                    Sg = _sample_masks(cfg, m, chain is not None, device, key_dev=st.mkey)
                cg = st.coefs
                # the witness rows are written in full below (k = l + t_pad: interp_H(W) on [0, l), then Z_H s = x^l s - s
                # with s on [l, k) and -s folded into [0, t_pad)); only the mask rows' tails need zeros
                cg[m:].zero_()
                _, Cg = enc.encode(Wg, Sg["W"], want_coefs=True, out=st.U[:m])
                cg[:m, :l] = Cg
                sW = Sg["W"]
                tp_ = sW.shape[1]
                cg[:m, :tp_] = (cg[:m, :tp_] - sW) % P                # _add_zh_times, first half
                cg[:m, l:l + tp_] = sW                                # second half: (0 + s) % p == s (s canonical)
                if l + tp_ < cg.shape[1]:
                    cg[:m, l + tp_:].zero_()
                for kind_, sl_ in mr.items():
                    blk_ = Sg[kind_]
                    cg[sl_, :blk_.shape[1]] = blk_
                st.U[m:] = eval_on_coset(cg[m:].to(torch.int64), n).to(torch.int32)
            else:
                enc.encode(Wg, out=st.U)
                st.W.copy_(Wg)
            if graph_merkle:
                with _cupy_ctx(device):
                    st.merkle.run(st.U)                               # the tree into the static levels
                st.root_pin.view(torch.int32).copy_(st.merkle.root, non_blocking=True)   # D2H memcpy node

        graph_merkle = st.merkle is not None and merkle.gpu_available()
        clk.host("commit_prep")
        _graph_run(st, "commit_graph", commit_body, device)
        clk.host("commit_replay")
        W, coefs, U = st.W, st.coefs, st.U
    else:
        W = build_witness(sys, pub, hints, device)
        clk.lap("witness")
        if mutate is not None:
            mutate(W)
        if st is not None and st.lanes is not None:
            st.lanes.copy_(W.index_select(0, st.lane_rows))
        if check:
            from .witness import check_constraints
            bad = check_constraints(sys, tb, W, pub)
            if bad:
                raise ValueError("witness violates constraints: " + ", ".join(bad[:5]))
            clk.lap("self_check")
    if fw is not None:
        pass
    elif cfg.zk:
        with _cupy_ctx(device):
            S = _sample_masks(cfg, m, chain is not None, device)
        clk.lap("zk_masks")
        # coefficient rows of every committed polynomial: witness rows interp_H(W) + Z_H s_j, then the mask rows
        coefs = st.coefs.zero_() if st is not None else torch.zeros((Mrows, k), dtype=torch.int64, device=device)
        if enc is not None:
            # SIMT kernel: codeword of interp_H(W_j) + Z_H s_j on the coset (the mask folded in per coset, 8c) and the
            # natural-order coefficients of interp_H(W_j) written as int64 straight into coefs[:m, :l] (slot buffers reused)
            U = st.U if st is not None else torch.empty((Mrows, n), dtype=torch.int32, device=device)
            enc.encode(W, S["W"], out=U[:m], coefs_out=coefs[:m, :l])
        else:
            coefs = torch.zeros((Mrows, k), dtype=torch.int64, device=device)
            coefs[:m, :l] = intt(W % P)
        _add_zh_times(coefs[:m], S["W"], l)
        for kind, sl in mr.items():
            blk = S[kind]
            coefs[sl, :blk.shape[1]] = blk
        del S
        if enc is not None:
            if Mrows > m:
                enc.encode_coefs(coefs[m:], out=U[m:])            # the 4D / 5D mask rows (degree < k), same kernel
        else:
            U = eval_on_coset(coefs, n)                           # (M, n), the coset g <w_n> (disjoint from H)
    else:
        coefs = None
        if st is not None:
            U = enc.encode(W, out=st.U)[0]
            st.W.copy_(W)                                         # the linear test reads W (static for the graph)
        else:
            U = enc.encode(W)[0] if enc is not None else encode(W, n)   # (m, n); int32 from the kernel
    clk.lap("encode")

    gpu_hash = merkle.gpu_available() and cuda
    if gpu_hash:
        from . import openings_device
        if graph_merkle:                                           # committed inside the commit graph (MERKLE_GRAPH)
            U32, tree, root_buf = st.U, st.merkle, st.root_pin
            clk.lap("transfer")
        else:
            U32 = U.to(torch.int32).contiguous()                   # elements < p < 2^31
            clk.lap("transfer")                                    # nothing leaves the device here
            with _cupy_ctx(device):
                root_dev, tree = openings_device.commit_device_async(U32)
            root_buf = _pin(slot, device, "root", 32)[:32]
            root_buf.view(torch.int32).copy_(root_dev, non_blocking=True)
        hash_name = merkle.GPU_HASH_NAME
        clk.lap("merkle")
        clk.host("launch_to_merkle")
        yield _event(device, slot is not None)                                       # -- wait: the root reaches the host
        clk.host("wait_root")
        root = root_buf.numpy().tobytes()
    else:
        U_cols = U.t().contiguous().cpu().numpy().astype(np.uint32)
        clk.lap("transfer")
        root, levels = merkle.commit(U_cols)
        hash_name = merkle.HASH_NAME
        clk.lap("merkle")
        yield None

    M = U.shape[0]
    sent.append("root")
    if fp_later:
        if not fp_pub.done():
            yield fp_pub                                           # -- wait: the fingerprint statement (the other proof)
        fp_pub = fp_pub.result()
        stmt_fut = submit_stmt(fp_pub)
        clk.host("wait_fp")
    if not stmt_fut.done():
        yield stmt_fut
    stmt = stmt_fut.result()
    clk.host("wait_stmt")
    if cfg.interactive:
        fut = _coin_future(src, "open1", root)                   # V opens coin slot 1 (after root)
        if fut is not None:
            yield fut                                              # -- wait: the verifier's OPEN1 (network, off the driver)
            r1, s1 = fut.result()
            ccom = ccom_fut.result()                               # in before ROOT went out (the source gates it)
        else:
            r1, s1 = src.open1(root)
        if len(r1) != COIN_BYTES or len(s1) != COIN_BYTES or coin_commit(r1, s1) != ccom[0]:
            raise CoinOpeningError(1, sent)                       # public-data abort, before any coin-dependent message
    else:
        r1 = s1 = None
    # challenge 1: SHAKE-256 of the seed, squeezed on a worker thread into pinned memory, reduced on the device
    seed1 = merkle.hash_bytes(seed1_preimage(stmt, root, ccom, r1))
    layout = chain[0] if chain is not None else None
    if st is not None:
        # the tests graph: the squeeze lands in the static input; ONE replay computes w, h, q, v and copies them into
        # the pinned message buffer
        fut, finish = _expand_pinned(seed1, D * width, device, slot, "c1")
        clk.host("seed1")
        yield fut                                                  # -- wait: the squeeze (host, off-thread)
        clk.host("wait_shake")
        finish(raw_out=st.xraw)
        if st.pubmat is not None:
            st.pubmat.copy_(_pubmat(tb, pub, device))
        if chain is not None:
            st.y16.copy_(chain[1].to(device))
        if st.fp_rho is not None:                                  # public fingerprints (the hash side): static inputs
            # host tensors go up from the slot's pinned memory, non-blocking: a pageable copy synchronises the stream
            for nm, dst, src_t in (("fp_rho", st.fp_rho, fp_pub.rho), ("fp_F", st.fp_F, fp_pub.F)):
                if src_t.device.type == "cpu":
                    nbytes = dst.numel() * dst.element_size()
                    buf = _pin(slot, device, nm, nbytes)[:nbytes].view(dst.dtype).view(dst.shape)
                    buf.copy_(src_t.reshape(dst.shape))
                    src_t = buf
                dst.copy_(src_t, non_blocking=True)
        mbuf = _pin(slot, device, "msgs", st.nbytes)[:st.nbytes]
        _tests_graph_run(st, sys, tb, cfg, mr, layout, mbuf, device)
        offs = st.offs
        clk.lap("tests_w")                                         # the graph's laps: tests_w carries the replay
        clk.lap("tests_quad")
        clk.lap("tests_chain")
        clk.host("launch_tests_qc")
        yield _event(device, slot is not None)                     # -- wait: w, h, q, v reach the host
        clk.host("wait_msgs")
        got = {}
        for name, (o, shape) in offs.items():
            got[name] = mbuf[o:o + 4 * math.prod(shape)].numpy().view(np.uint32).reshape(shape).copy()
        w_np, h_np, q_np, v_np, fp_np = got["w"], got["h"], got.get("q"), got.get("v"), got.get("fp")
    else:
        if cuda:
            fut, finish = _expand_pinned(seed1, D * width, device, slot, "c1")
            clk.host("seed1")
            yield fut                                              # -- wait: the squeeze (host, off-thread)
            clk.host("wait_shake")
            x = finish().reshape(D, width)
        else:
            x = _expand(seed1, D * width, device).reshape(D, width)
        y16_pub = chain[1] if chain is not None else None
        lanes = W.index_select(0, lane_rows) if (lane_rows is not None and W is not None) else None
        w, h, q, v, F = _tests_compute(sys, tb, cfg, x, coefs, U, W, pub, None, y16_pub, layout, mr, device, check=check,
                                       lanes=lanes, fp_pub=fp_pub)
        clk.lap("tests_w")
        clk.lap("tests_quad")
        clk.lap("tests_chain")
        clk.host("launch_tests_qc")
    # the test messages leave the device as ONE pinned non-blocking copy (int32: residues < p); host copies below
    if st is None and cuda:
        msgs = ([("w", w), ("h", h)] + ([("q", q)] if q is not None else []) + ([("v", v)] if v is not None else [])
                + ([("fp", F)] if F is not None else []))
        offs, at = {}, 0
        for name, tsr in msgs:
            offs[name] = (at, tuple(tsr.shape))
            at += 4 * tsr.numel()
        mbuf = _pin(slot, device, "msgs", at)[:at]
        for name, tsr in msgs:
            o, shape = offs[name]
            mbuf[o:o + 4 * tsr.numel()].view(torch.int32).view(shape).copy_(tsr.to(torch.int32).contiguous(), non_blocking=True)
        yield _event(device, slot is not None)                                       # -- wait: w, h, q, v reach the host
        clk.host("wait_msgs")
        got = {}
        for name, (o, shape) in offs.items():
            got[name] = mbuf[o:o + 4 * math.prod(shape)].numpy().view(np.uint32).reshape(shape).copy()
        w_np, h_np, q_np, v_np, fp_np = got["w"], got["h"], got.get("q"), got.get("v"), got.get("fp")
    elif st is None:
        w_np = w.cpu().numpy().astype(np.uint32)
        h_np = h.cpu().numpy().astype(np.uint32)
        q_np = q.cpu().numpy().astype(np.uint32) if q is not None else None
        v_np = v.cpu().numpy().astype(np.uint32) if v is not None else None
        fp_np = F.cpu().numpy().astype(np.uint32) if F is not None else None
    rho_np = None
    if fp_np is not None:
        fp_np = np.ascontiguousarray(fp_np[:layout.n_vus])        # the real VUs' fingerprints (pad slots are zero)
        if st is not None:                                         # coordinate 0's tail of the challenge block
            rho_np = ((st.xraw.reshape(-1)[width - n_fp:width].to(torch.int64) & 0xFFFFFFFF) % P).cpu().numpy().astype(np.uint32)
        else:
            rho_np = x[0, width - n_fp:].cpu().numpy().astype(np.uint32)
        if fp_sink is not None:
            fp_sink.set_result((rho_np, fp_np))

    sent.append("w, h" + (", q" if q_np is not None else "") + (", v" if v_np is not None else "") + (", fp" if fp_np is not None else ""))
    if cfg.interactive:
        if fp_np is not None and hasattr(src, "provide_fp"):
            # a live verifier takes F with the test message: fixed before slot 2 here and before the paired side's slot 1
            src.provide_fp(fp_np)
        fut = _coin_future(src, "open2", root, w_np, h_np, q_np, v_np)   # V opens coin slot 2 (after the test messages)
        if fut is not None:
            yield fut                                              # -- wait: the verifier's OPEN2 (network, off the driver)
            r2, s2 = fut.result()
        else:
            r2, s2 = src.open2(root, w_np, h_np, q_np, v_np)
        if len(r2) != COIN_BYTES or len(s2) != COIN_BYTES or coin_commit(r2, s2) != ccom[1]:
            raise CoinOpeningError(2, sent)
    else:
        r2 = s2 = None
    cols = _challenge2(stmt, root, w_np, h_np, cfg.t, n, q_np, v_np, ccom, r2, fp_np)
    clk.host("challenge2")
    t = len(cols)
    if gpu_hash:
        cbuf = _pin(slot, device, "cols", 8 * t)[:8 * t].view(torch.int64)
        cbuf.numpy()[:] = cols
        cols_t = cbuf.to(device, non_blocking=True)
        opened_dev = U32.index_select(1, cols_t)                  # (M, t) int32: the only U bytes that leave
        sib = openings_device.gather_paths_device(tree, cols_t)   # (t, depth, 8) int32
        depth = sib.shape[1]
        nb1, nb2 = 4 * M * t, 4 * sib.numel()
        obuf = _pin(slot, device, "open", nb1 + nb2)[:nb1 + nb2]
        obuf[:nb1].view(torch.int32).view(M, t).copy_(opened_dev.contiguous(), non_blocking=True)
        if nb2:
            obuf[nb1:nb1 + nb2].view(torch.int32).view(sib.shape).copy_(sib.contiguous(), non_blocking=True)
        clk.lap("openings")
        clk.host("launch_openings")
        yield _event(device, slot is not None)                                       # -- wait: the openings reach the host
        clk.host("wait_openings")
        opened = openings_device.opened_from_pinned(obuf[:nb1].view(torch.int32), M, t)
        paths = openings_device.paths_from_raw(obuf[nb1:nb1 + nb2].numpy().tobytes(), t, depth)
        clk.host("unpack_openings")
    else:
        opened = U_cols[cols].T.copy()                            # (m, t)
        paths = merkle.open_columns(levels, cols)
        clk.lap("openings")
    auth_sibs = None
    if auth is not None:                                          # the Merkle multiproofs of the boundary words
        auth_sibs = [list(s) for s in auth_open()]
        clk.lap("auth_openings")
    total = clk.total()
    timings = dict(clk.marks)
    timings["tests"] = timings.get("tests_w", 0.0) + timings.get("tests_quad", 0.0) + timings.get("tests_chain", 0.0)
    timings["total"] = total - timings.get("hints_host", 0.0)
    return Proof(root, w_np, h_np, cols, opened, paths, hash_name, timings, q_np, v_np, ccom,
                 Coins(r1, s1, r2, s2) if cfg.interactive else None, auth_sibs, fp_np, rho_np)


# ---------------------------------------------------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------------------------------------------------

def verify(p: Params, sys: System, tb: Tables, cfg: Config, c, a, b, y, proof: Proof, device,
           chain: tuple | None = None, coins: Coins | None = None, rel: RelationHooks = BF16_HOOKS,
           auth: auth_mod.AuthStatement | None = None, auth_procs: int | None = None,
           auth_timing: dict | None = None, out: dict | None = None) -> tuple[bool, str]:
    """Accept/reject the claim ``y[i] = tc_dot(c[i], a[i], b[i])`` for all l units -- or, with
    ``chain = (Layout, y16_pub)`` (``c = y = None``), the claim that every VU of the layout casts to its
    ``y16_pub`` word from ``c_0 = 0`` over its 96 units.  Never touches W.  A fingerprinted chain (chain.py):
    ``chain = (Layout, y16_pub, FpArgs)`` in the public mode (the hash side of row sharing: rho and F are the
    verifier's statement values); in the coin mode the message ``proof.fp`` is checked against the chain and ``out``
    (if given) receives ``out["rho"]`` -- the coins THIS verifier derived, for the paired hash-side statement.
    ``coins``: interactive mode -- REQUIRED: the verifier's OWN step-0 randomness; the transcript must carry exactly
    these openings (a proof made in another verifier's session, or with prover-chosen coins, is rejected).
    Fiat-Shamir mode -- must be ``None``: there is no step 0 and the transcript carries no coins.
    ``auth`` (chain mode, ``authentication = included``): the verifier's OWN committed-object block (roots, VU
    indices); the statement digest absorbs it and, after the Ligero tests, the transcript's three multiproofs must
    open exactly the published words at the claimed leaf indices under the roots (``auth.verify_auth``)."""
    t0 = time.perf_counter()
    m, l, n, D, t, k = sys.m, cfg.l, cfg.n, cfg.D, cfg.t, cfg.k     # every size from the VERIFIER's configuration
    if (chain is not None) != (sys.chain is not None):
        return False, "chain mode mismatch"
    if chain is not None:
        why = layout_error(chain[0], rel)
        if why is not None:
            return False, why
    if auth is not None and chain is None:
        return False, "authentication is defined for the chain statement only"
    if (auth is not None) != (proof.auth is not None):
        return False, "authentication block / multiproofs mismatch"
    # step 0 / openings: c_i = H(r_i || s_i) for both slots; interactive: they are this verifier's own coins
    if cfg.interactive:
        if coins is None:
            return False, "interactive mode: the verifier must supply its own step-0 coins"
        pc = proof.coins
        if proof.coin_commit is None or pc is None or not pc.well_formed() or len(proof.coin_commit) != 2:
            return False, "malformed coin commitments / openings"
        own = (coins.r1, coins.s1, coins.r2, coins.s2)            # field-wise: ``coins`` may be a Coins subclass (V*)
        if (pc.r1, pc.s1, pc.r2, pc.s2) != own or tuple(proof.coin_commit) != coins.commitments():
            return False, "coins are not this verifier's step-0 coins"
        ccom = tuple(proof.coin_commit)
    else:
        if coins is not None:
            return False, "fiat-shamir mode has no step 0 (coins must be None)"
        if proof.coin_commit is not None or proof.coins is not None:
            return False, "fiat-shamir transcript must not carry coins"
        ccom, pc = None, Coins(None, None, None, None)
    mr = mask_rows(cfg, m, chain is not None)
    M = n_rows(cfg, m, chain is not None)
    if not (proof.hash_name or "").startswith("blake3"):
        return False, "proof names a hash the verifier does not use"
    try:
        pub = rel.public_vectors(p, c, a, b, y, device=device)
    except ValueError as e:
        return False, f"public inputs: {e}"
    if sys.pins and len(pub[sys.pins[0][1]]) != l:          # the first pinned vector (a[0].s for BF16/FP8, a[0].n for FP4)
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
    fp_pub = fp_args = None
    n_fp = 0
    if chain is not None:
        layout, y16_pub = chain[0], chain[1]
        y16_pub = torch.as_tensor(y16_pub, dtype=torch.int64)
        if layout.l != l or proof.q is None or proof.q.shape != (D, cfg.q_len) or (proof.q >= P).any():
            return False, "malformed chain transcript"
        if len(y16_pub) != l or ((y16_pub < 0) | (y16_pub >= 2 ** (rel.y16_bits or p.word_bits))).any():
            return False, "y16 range"
        if auth is not None and len(auth.vu_index) != layout.n_vus:
            return False, "authentication block does not match the layout"
        # the fingerprint constraints (chain.py): coin mode -- the message F (n_vus, n_op, C) in the transcript, rho from
        # challenge 1; public mode -- (rho, F) from the verifier's statement, no message
        spec = sys.chain.get("fp")
        n_slots = l // layout.steps
        if spec is not None:
            from .chain import FpArgs
            n_op, C = len(spec["lanes"]), int(spec["coords"])
            if spec["mode"] == "coin":
                n_fp = C
                if proof.fp is None or proof.fp.shape != (layout.n_vus, n_op, C) or (proof.fp >= P).any():
                    return False, "malformed fingerprint message"
                if len(chain) > 2 and chain[2] is not None:
                    return False, "fingerprint coin mode takes no public fingerprints"
            else:
                if proof.fp is not None:
                    return False, "unexpected fingerprint message"
                if len(chain) < 3 or chain[2] is None:
                    return False, "fingerprint public mode: the statement must give rho and F"
                fp_pub = chain[2]
                if tuple(fp_pub.F.shape) != (n_slots, n_op, C) or tuple(fp_pub.rho.shape) != (C,):
                    return False, "malformed public fingerprints"
                if (fp_pub.F >= P).any() or (fp_pub.F < 0).any() or (fp_pub.rho >= P).any() or (fp_pub.rho < 0).any():
                    return False, "public fingerprints out of the field"
                if (fp_pub.F[layout.n_vus:] != 0).any():
                    return False, "public fingerprints of pad slots must be zero"
        elif proof.fp is not None or (len(chain) > 2 and chain[2] is not None):
            return False, "unexpected fingerprint data"
        stmt = statement_digest(sys, cfg, a, b, layout=layout, y16=y16_pub, rel=rel, auth=auth, fp=fp_pub)
    elif proof.q is not None or proof.fp is not None:
        return False, "unexpected chain transcript"
    else:
        stmt = statement_digest(sys, cfg, a, b, c=c, y=y, rel=rel)
    if len(set(proof.cols)) != t or any(not 0 <= c_ < n for c_ in proof.cols):
        return False, "bad column set"
    # challenges: H(statement, transcript prefix) -- the statement is the verifier's, not the proof's
    r, rho_lin, rho_quad, rc, rho_fp = _challenge1(stmt, proof.root, D, M, tb.L, tb.Q, device, l if chain else 0, ccom, pc.r1, n_fp)
    if proof.cols != _challenge2(stmt, proof.root, proof.w, proof.h, t, n, proof.q, proof.v, ccom, pc.r2, proof.fp):
        return False, "column challenge mismatch"
    if n_fp:
        from .chain import FpArgs
        F = torch.zeros((n_slots, proof.fp.shape[1], n_fp), dtype=torch.int64, device=device)
        F[:layout.n_vus] = torch.from_numpy(proof.fp.astype(np.int64)).to(device)
        fp_args = FpArgs(rho_fp, F)
        if out is not None:
            out["rho"] = rho_fp.cpu().numpy().astype(np.uint32)
    elif fp_pub is not None:
        from .chain import FpArgs
        fp_args = FpArgs(torch.as_tensor(fp_pub.rho, dtype=torch.int64).to(device), torch.as_tensor(fp_pub.F, dtype=torch.int64).to(device))
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
        cc = chain_coefs(sys, layout, rc, torch.as_tensor(y16_pub, dtype=torch.int64), fp_args)
        q = torch.from_numpy(proof.q.astype(np.int64)).to(device)
        if not torch.equal(_sum_over_H(q, l), cc.rhs):
            return False, "chain constraints failed (sum over H)"
        R = cc.coef.shape[1]
        Rc = torch.cat([encode(cc.coef[:, r0:r0 + CHAIN_ENC_CHUNK].reshape(-1, l), n).reshape(D, -1, n)[:, :, cols]
                        for r0 in range(0, R, CHAIN_ENC_CHUNK)], 1)                     # (D, R, t), bounded memory
        expect = ((Rc * Uo.index_select(0, cc.rows)[None, :, :]) % P).sum(1) % P
        if cfg.zk:                                                            # + u_ca(eta_c) + eta_c^k u_cb(eta_c)
            xk = _xk_on_coset(n, k, device)[cols]
            expect = (expect + Uo[mr["chain_a"]] + (xk[None, :] * Uo[mr["chain_b"]]) % P) % P
        if not torch.equal(eval_on_coset(q, n)[:, cols], expect):
            return False, "chain constraints failed (opened columns)"
    _sync(device)
    # (vi) authentication: the published words are the committed words (verifier-side openings; auth.py)
    if auth is not None:
        t_auth = time.perf_counter()
        real = layout.n_vus * layout.steps
        a_np = np.asarray(a.cpu().numpy() if isinstance(a, torch.Tensor) else a, dtype=np.int64)
        b_np = np.asarray(b.cpu().numpy() if isinstance(b, torch.Tensor) else b, dtype=np.int64)
        y_np = y16_pub.cpu().numpy()[layout.steps - 1:real:layout.steps]
        ok, why = auth_mod.verify_auth(auth, proof.auth, a_words=a_np[:real].reshape(-1), b_words=b_np[:real].reshape(-1),
                                       y_words=y_np, K=layout.steps * a_np.shape[1], steps=layout.steps, procs=auth_procs)
        if auth_timing is not None:
            auth_timing["seconds"] = auth_timing.get("seconds", 0.0) + (time.perf_counter() - t_auth)
        if not ok:
            return False, why
    return True, f"accept ({time.perf_counter() - t0:.3f}s)"
