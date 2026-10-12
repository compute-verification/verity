"""The tables of the approved spec (``kb/TABLES.md``, approved 2026-09-25): a labelled PREVIEW beside the frozen canonical tables.

    python -m verity_numerical.bench.views --root <store root> [--format md|json|entities] [--rules spec|unfiltered|legacy]
                                           [--security-bits N] [--include-archived]
    python -m verity_numerical.bench.views --root <store root> --parity [--snapshot NAME | --take-snapshot] [--record]

Until the switch (a labelled 6 PM PT digest) the published tables stay :mod:`verity_numerical.bench.tables` and
:mod:`~verity_numerical.bench.drilldown`; these views are previews and are never shown as the published numbers.

* **Table 1** -- backends and statements with their security, one entry per backend family followed by one row per backend
  and statement (:data:`CONFIGURATIONS`): the statement (the full committed relation or a weaker one) and what the verifier sees of x, W and
  y; the commitment scheme (:data:`SCHEMES`); every hash of the soundness argument with its role, property and class; ZK class
  and verifier model; soundness with the whole-proof bound at the batching of each cell it holds and the dominant term; the
  proximity bounds (regime, proven, machine-checked or conjectured); transferability; assumptions; the pin; ``archived``; its cells (computed).
* **Table 2** -- end-to-end overhead ``N / P`` per subcircuit, hardware class and admissible commitment scheme (one line per
  ``verity.ml.circuits.targets.target.TARGETS`` entry and scheme some row's statement declares), a native column, one column per
  backend family (:data:`FAMILIES`).  ``N`` is the
  compute-bound datasheet estimate (:func:`native`); ``P`` is inputs per second end to end, committing the batch's values and
  then proving (:func:`throughput`).  A cell is the family's best admissible backend and statement (highest ``P``, ties to the lower
  artifact id) that proves the full relation over a core-defined scheme and passes the view's filters (:class:`Rules`: by
  default a whole-proof bound of 2^-128), marked ``(prov.)`` while its statement has no red-team clearance on
  that subcircuit and ``(alg.)`` when its soundness argument uses an algebraic hash (flagged in Table 1).
* **Table 3** -- where the time goes: seconds per input of commitment and of each proving bucket, with shares of end to end,
  and the cell split into serving overhead (commitment x N) and proving overhead (proving x N).

Every rejected result is listed with its first reason (:data:`REASON_LETTERS`, the spec's order); ``--format json`` has them all.

The v1 adapter: ``bench-result/v1`` results carry no ``configuration`` id (a backend and statement), so theirs is the ``drilldown.VARIANTS`` entry
that recognises them (:func:`configuration_of`) and their family the ``tables.CANDIDATES`` entry; name recognisers live nowhere
else in these views.  A result names its commitment time as the measurement :data:`COMMIT` and its
protocol as the ``protocol`` / ``sweep`` blocks (``benchmark-standard`` §6.1, ``bench-result/v2``); no v1 result records the
protocol, so no v1 result meets it (reason M).

``--rules legacy`` renders with today's parameters (2^-128 on target and achieved, B = 4096, K = 1536, the frozen input range,
the bare and ``+ in-proof hash`` columns, proving time only): the switch's parity gate, which ``--parity`` checks against a
fresh :func:`tables.build` of the same store (and, with ``--snapshot NAME``, against that ``tables --snapshot``'s members).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import json
import math
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from verity.core import claims as claims_registry
from verity.ml.circuits.targets.target import BF16_HOPPER, FIRST, FP8_ADA_MMA, FP8_HOPPER_WGMMA, NVFP4_SM120, TARGETS, Target
from verity_numerical.bench import census, contract
from verity_numerical.bench import drilldown as D
from verity_numerical.bench import tables as T

__all__ = ["COMMIT", "CONFIGURATIONS", "FAMILIES", "LEGACY", "REASON_LETTERS", "SPEC", "UNFILTERED", "Configuration", "Family",
           "Rules", "Views", "build", "configuration_of", "main", "native", "parity", "render_entities", "render_json", "render_markdown", "throughput"]

PREVIEW = ("PREVIEW of the approved tables spec (kb/TABLES.md): not the published tables. The published numbers switch once, in a "
           "6 PM PT digest labelled as a change of method; until then the canonical tables are `verity_numerical.bench.tables`.")
#: The banner of the published render (``--published``, the steward's ``published = "views"``): the switch has happened.
PUBLISHED = ("The published tables, under the approved spec (kb/TABLES.md), since the switch digest labelled as a change of method. "
             "The frozen 2026-09-22 tables (`verity_numerical.bench.tables`) remain as a drill-down beside this render (`<stamp>-tables-frozen.md`).")

#: Commitment seconds per batch: the warm median of committing the batch's values afresh, as lane hash-commit writes it (with
#: ``commit.cold_seconds``, ``commit.reps`` and ``e2e.seconds`` = this + ``t.total``).
COMMIT = "commit.seconds"
#: v1 in-proof-hash results record their commitment once, before the timed runs (possibly a tree cache load): shown, never counted.
COMMIT_V1 = "relation.hash.commit_seconds"
#: One-time setup of a run (keys, circuit setup), outside ``t.total``: Table 3 shows it beside the cell, never in it.
SETUP = "fixed.setup_seconds"
MIN_RUNS = 5
#: How Table 1 flags a configuration whose soundness argument uses an algebraic hash; its Table 2 cells carry ``(alg.)``.
ALGEBRAIC_FLAG = "algebraic hash, not for highest-stakes use"

#: The spec's reject reasons, in the order they are checked; ``B`` exists only under the legacy rules (today's batch and K).
REASON_LETTERS: dict[str, str] = {
    "R": "weaker statement than the full relation", "Z": "not the family's declared class",
    "S": "commitments on a scheme verity.ml.commitments does not define", "L": "fails the view's security filter",
    "U": "not independently verified", "F": "proof files not preserved", "I": "another input set", "K": "prover not on the line's GPU",
    "P": "phase-sum", "M": "protocol not met", "D": "capped by a red-team verdict", "X": "other", "B": "legacy: batch != 4096 or K != 1536"}
_ORDER = {k: i for i, k in enumerate(REASON_LETTERS)}


# --- commitment schemes -----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Scheme:
    """A commitment scheme with the hash of its leaves (one Table 2 line per subcircuit, hardware class and admissible scheme).
    ``core`` is where ``verity.ml.commitments`` defines it (``None``: not defined there, so inadmissible); ``standard``: built only on
    standard hashes (SHA-256, BLAKE3) -- an algebraic scheme is admissible and flagged (:data:`ALGEBRAIC_FLAG`)."""

    id: str
    label: str
    core: str | None
    standard: bool

    @property
    def admissible(self) -> bool:
        return self.core is not None


SCHEMES: tuple[Scheme, ...] = (
    Scheme("frame-v3/sha256", "frame-v3 · SHA-256", "verity.ml.commitments.merkle (frame-v3 word leaves; the SHA-256 row leaf is core-schemes')", True),
    Scheme("frame-v3/blake3-row", "frame-v3 · keyed-BLAKE3 rows",
           "verity.ml.commitments.merkle trees; the keyed-BLAKE3 row-leaf schema is promoted from archive/direct/ligero/leaf/blake3.py (core-schemes)", True),
    Scheme("vllm-v1", "vllm-v1 · SHA-256", "verity.ml.commitments.vllm_v1 (position / fa2h / id leaves, step and stream root bindings)", True),
    Scheme("frame-v3-sha512/hm96-sha512", "frame-v3 · SHA-512, hm96-sha512 leaves",
           "verity.ml.commitments.frame_v3 (the SHA-512 variant) over verity.ml.commitments.hm96 hm96-sha512/v1 leaves", True),
    Scheme("frame-v3/poseidon2-row", "frame-v3 · Poseidon2 rows", "verity.ml.commitments.rowleaf (leaf/v2h)", False),
    Scheme("frame-v3/poseidon2-tile", "frame-v3 · Poseidon2 64x64 tiles", None, False),
    Scheme("frame-v3/ajtai-row", "frame-v3 · Ajtai rows", None, False),
)
_SCHEME = {s.id: s for s in SCHEMES}


# --- Table 1: families, backends and statements ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Family:
    """One Table 2 column: the backend family, its declared class (the ``tables.CANDIDATES`` entry's) and what it is."""

    id: str
    declared_classes: tuple[str, ...]
    description: str
    setup: str
    interaction: str
    note: str = ""
    development: str = "active"


#: Backend families kept in the repo and the tables but frozen (Daniel, 2026-09-26): tests keep passing and cells stay; no new
#: features, no cells for new subcircuits, re-run only when a shared change would break their tests.  B-Ligero stays the ZK comparison.
FROZEN = ("A-GKR", "B-Ligero", "SP1")
NOT_ACTIVE = "not actively developed"


#: Which composed backends a family column holds (Table 1's family row): route (a) stays A-GKR's (brief tables-switch).
FAMILY_NOTES = {
    "A-GKR": "route (a), the A-GKR prime side with Flock's keyed-BLAKE3 row hashes and the sigma link in one live session, is an "
             "A-GKR configuration and fills this column",
    "Flock": "pure Flock only: the relation and its row hashes in one binary-field circuit; route (a) is in the A-GKR column",
}

#: Flock is its own backend family (Daniel, 2026-09-25); the frozen tables have no Flock column, so its family is declared here.  The
#: v1 adapter recognises it by ``software.backend.name`` starting ``flock``.  Declared class NON_ZK_PROOF: Flock is not zero-knowledge
#: yet (the Flock lane may change it).
FLOCK = T.Candidate(
    id="Flock", description="Flock: Ligerito-family binary-field prover (GF(2^128)) over the union circuit of the unit and its keyed-BLAKE3 row leaves",
    zk="as configured", verifier_model="as configured", soundness="as configured", setup="none (no trusted setup)",
    interaction="live verifier coins (flock-128-r2); Fiat–Shamir variant below 2^-128", assumptions="as configured",
    proof_classes=("NON_ZK_PROOF",), drilldown_classes=(),
    backend_names=("flock",))
CANDIDATES = (*T.CANDIDATES, FLOCK)
FAMILIES: tuple[Family, ...] = tuple(Family(c.id, c.proof_classes, c.description, c.setup, c.interaction, FAMILY_NOTES.get(c.id, ""),
                                            NOT_ACTIVE if c.id in FROZEN else "active") for c in CANDIDATES)
_FAMILY = {f.id: f for f in FAMILIES}
_CANDIDATE = {c.id: c for c in CANDIDATES}


def candidate_for(backend: Any) -> T.Candidate | None:
    """The backend family of a result: ``tables.candidate_for``, else Flock."""
    return T.candidate_for(backend) or (FLOCK if FLOCK.recognises(backend) else None)


@dataclass(frozen=True)
class Hash:
    """One hash the soundness argument relies on: its role (statement commitment, proof-internal commitment, Fiat–Shamir and
    coins), the property assumed (collision resistance, random oracle) and whether it is algebraic."""

    name: str
    role: str
    prop: str
    algebraic: bool = False

    def text(self) -> str:
        return f"{self.name} ({self.role}; {self.prop}; {'algebraic' if self.algebraic else 'standard'})"


@dataclass(frozen=True)
class Proximity:
    """One code-based component's proximity bound: the decoding regime its terms use and whether the bound is proven (with its
    theorem), machine-checked in the configuration's own soundness proof (``MACHINE_CHECKED``), or conjectured."""

    component: str
    regime: str
    basis: str

    def text(self) -> str:
        return f"{self.component}: {self.regime}, {self.basis}"


FULL = "full relation"
#: Route (a): A-GKR's prime proof with Flock's keyed-BLAKE3 rows and the sigma link (drilldown's ``VARIANTS`` entry of the same id).
ROUTE_A = "A-GKR route (a): prime + Flock rows + sigma link"
_WEAK_PUBLIC = "weaker: operands public, no commitment (the verifier could recompute y)"
_WEAK_PRIVATE = "weaker: operands private and bound to nothing"
_WEAK_REHASH = "weaker: operands published for the verifier to re-hash (it could recompute y)"
_STMT, _INTERNAL, _COINS = "statement commitment", "proof-internal commitment", "Fiat–Shamir / coins"
_CR, _RO = "collision resistance", "random oracle"
_B_HASHES = (Hash("BLAKE3", _INTERNAL, _CR), Hash("BLAKE2b", _COINS, _CR + " (coin commitments)"), Hash("SHAKE-256", _COINS, "PRG of the verifier's coins"))
_B_FS_HASHES = (Hash("BLAKE3", _INTERNAL, _CR), Hash("BLAKE3 / BLAKE2b", _COINS, _RO + " (Fiat–Shamir)"), Hash("SHAKE-256", _COINS, _RO))
_SHA256_STMT = Hash("SHA-256", _STMT, _CR + " (frame-v3 trees)")
_LIGERO_UDR = (Proximity("Ligero column tests (interleaved RS, rate 1/4)", "unique decoding", "proven (AHIV22 App. C; BCIKS20 Thm 1.2)"),)
_B_DOMINANT = ("proximity-gap term n/|F| per sub-batch proof (n = code length); the whole-proof bound adds log2(proofs), so it grows "
               "with the batch through the number of proofs")
_B_ASSUME = "BabyBear with its extension for challenges; interactive: statistical given the committed coins (HM96)"
_SP1_HASHES = (Hash("Poseidon2-KoalaBear", _INTERNAL, _CR + " (Merkle trees)", True), Hash("Poseidon2-KoalaBear", _COINS, _RO + " (challenger)", True))
_SP1_PROX = (Proximity("FRI", "unique decoding", "proven (soundcalc UDR terms; BCIKS20)"),)
_SP1_DOMINANT = ("field-bound terms over KoalaBear^4 (|F| = 2^123.95), LogUp-GKR 2^-100.3 per shard; the whole-proof bound adds "
                 "log2(shards), and shards grow with the batch")
_SP1_ASSUME = "KoalaBear^4 challenges; zerocheck and jagged-reduction lemmas as SP1 states them"
_AGKR_PROX = (Proximity("Ligero IRS polynomial commitment", "unique decoding", "proven (BCIKS20 Thm 1.2, n/|F|^sigma)"),)
#: A proximity bound proved inside the configuration's own Lean soundness theorem, so Table 1 cites no assumption for it.
MACHINE_CHECKED = "machine-checked"
_FLOCK_MCA = (f"{MACHINE_CHECKED} (Lean table_sound: Haböck ePrint 2025/2110, with BCHKS25's MCA bound proved from DKT26, "
              "ArkLib #907)")
_AGKR_DOMINANT = "IRS query term (2^-130.2 at t = 192) and the LogUp identity term; the (n+1)/|F|^sigma term grows with the committed length n"


@dataclass(frozen=True)
class Configuration:
    """One Table 1 row: a backend and a statement.  ``id`` is the ``drilldown.VARIANTS`` entry it adapts (v1 results carry
    no ``configuration`` id); the ZK and per-proof soundness words are that entry's.  ``scheme`` is the :data:`SCHEMES` id its statements bind (``None``: the statement
    commits nothing); ``hashes`` is every hash the soundness argument relies on.  ``unsupported`` maps a target name to the
    backend's declared reason (``n/s``)."""

    id: str
    family: str | None
    statement: str
    ports: str
    scheme: str | None
    commitment: str
    hashes: tuple[Hash, ...]
    zk: str
    soundness: str
    dominant_term: str
    proximity: tuple[Proximity, ...]
    transferability: str
    assumptions: str
    pin: str
    interaction: str = ""
    archived: bool = False
    unsupported: Mapping[str, str] = field(default_factory=dict)

    @property
    def full_relation(self) -> bool:
        return self.statement == FULL

    def algebraic(self) -> tuple[Hash, ...]:
        return tuple(h for h in self.hashes if h.algebraic)

    def scheme_problem(self) -> str | None:
        """Why the statement's commitments are on an inadmissible scheme (reason S), or ``None``."""
        s = _SCHEME.get(self.scheme) if self.scheme else None
        if s is None:
            return f"{self.id} binds no scheme of verity.ml.commitments"
        if s.core is None:
            return f"{self.id} commits with {s.label}, which verity.ml.commitments does not define yet"
        return None


def _cfg(vid: str, statement: str, ports: str, scheme: str | None, commitment: str, hashes: tuple[Hash, ...], dominant: str,
         proximity: tuple[Proximity, ...], transferability: str, assumptions: str, pin: str, **kw: Any) -> Configuration:
    v = D._BY_ID[vid]
    return Configuration(vid, v.candidate, statement, ports, scheme, commitment, hashes, v.zk, v.soundness, dominant, proximity,
                         transferability, assumptions, pin, **kw)


_B_PIN = "ligero-verify's per-relation system pin (src/relation.rs) and statement digest"
_B_LIVE = "interactive: live-verifier-only (a file replay of the runner's coins is not transferable, D7)"
_B_BARE = ("x, W, y public", None, "none: the statement publishes the operands; the witness is under BLAKE3 Merkle trees")
_H_PORTS = "x, W committed (private); y committed"
_IN_PROOF = "; the proof computes the row digests and publishes them, the verifier checks them against the trees natively"
_B_KW = dict(dominant=_B_DOMINANT, proximity=_LIGERO_UDR, transferability=_B_LIVE)
CONFIGURATIONS: tuple[Configuration, ...] = (
    *(_cfg(vid, _WEAK_PUBLIC, *_B_BARE, _B_HASHES, assumptions=_B_ASSUME, pin=_B_PIN, **_B_KW) for vid in (
        "B-Ligero v1", "B-Ligero v3 (private selection)", "B-Ligero v3 fused hints", "B-Ligero v3x4 fused (l=4096)",
        "B-Ligero v1x4 (fold)", "B-Ligero v3x4 (fold)", "B-Ligero v2 (public selection)", "B-Ligero v2x4 (public selection)")),
    _cfg("B-Ligero Fiat–Shamir (HVZK)", _WEAK_PUBLIC, *_B_BARE, _B_FS_HASHES, _B_DOMINANT + "; plus the Fiat–Shamir query term",
         _LIGERO_UDR, "transferable (Fiat–Shamir)", "BabyBear with its extension for challenges", _B_PIN),
    _cfg("B-Ligero + Poseidon2 per row", FULL, _H_PORTS, "frame-v3/poseidon2-row",
         "frame-v3 SHA-256 trees over Poseidon2-BabyBear (w24) row digests (leaf/v2h), one per operand row" + _IN_PROOF,
         (_SHA256_STMT, Hash("Poseidon2-BabyBear", _STMT, _CR + " (row digests)", True), *_B_HASHES), assumptions=_B_ASSUME,
         pin=_B_PIN + "; leaf schema poseidon2-babybear-w24/row/v2h", **_B_KW),
    _cfg("B-Ligero + Poseidon2 shared 64x64 tile", FULL, _H_PORTS + " (a fingerprint discloses linear information, F8)",
         "frame-v3/poseidon2-tile", "SHA-256 trees over Poseidon2-BabyBear digests of 64x64 shared tiles" + _IN_PROOF,
         (_SHA256_STMT, Hash("Poseidon2-BabyBear", _STMT, _CR + " (tile digests)", True), *_B_HASHES),
         assumptions=_B_ASSUME + "; the fingerprint-collision bound (F7)", pin=_B_PIN, **_B_KW),
    _cfg("B-Ligero + BLAKE3 in circuit", FULL, _H_PORTS, "frame-v3/blake3-row",
         "frame-v3 SHA-256 trees over role-keyed, n_chunks-framed keyed-BLAKE3 row digests" + _IN_PROOF,
         (_SHA256_STMT, Hash("BLAKE3 (keyed)", _STMT, _CR + " (row digests)"), *_B_HASHES), assumptions=_B_ASSUME,
         pin=_B_PIN + "; leaf schema blake3-keyed/row/*", **_B_KW),
    _cfg("B-Ligero + SHA-256 in circuit", FULL, _H_PORTS, "frame-v3/sha256",
         "frame-v3 SHA-256 trees over sha256/row/v1 row digests (prefix block and padding block by the verifier)" + _IN_PROOF,
         (_SHA256_STMT, Hash("SHA-256", _STMT, _CR + " (row digests)"), *_B_HASHES), assumptions=_B_ASSUME,
         pin=_B_PIN + "; leaf schema sha256/row/v1", **_B_KW),
    _cfg("B-Ligero + vllm-v1 SHA-256 in circuit", FULL, _H_PORTS, "vllm-v1",
         "vllm-v1 SHA-256 trees over position leaves of every operand row (all compressions in-circuit), bound by step roots "
         "over verifier-derived domains whose digests the statement absorbs" + _IN_PROOF,
         (_SHA256_STMT, Hash("SHA-256", _STMT, _CR + " (position leaves)"), *_B_HASHES), assumptions=_B_ASSUME,
         pin=_B_PIN + "; leaf schema vllm-v1/pos-leaf/v0, step-root domains", **_B_KW),
    _cfg("B-Ligero + Ajtai n=64", FULL, _H_PORTS + " (binding only, not hiding)", "frame-v3/ajtai-row",
         "SHA-256 trees over Ajtai (Ring-SIS, n = 64) row digests" + _IN_PROOF,
         (_SHA256_STMT, Hash("Ajtai n=64 (linear digest)", _STMT, "Ring-SIS binding", True), *_B_HASHES),
         assumptions=_B_ASSUME + "; Ring-SIS over F_p[X]/(X^64 + 1); rows of <= 64 columns (steps-pin)", pin=_B_PIN + "; steps <= n", **_B_KW),
    _cfg("B-Ligero + Ajtai n=128", FULL, _H_PORTS + " (binding only, not hiding)", "frame-v3/ajtai-row",
         "SHA-256 trees over Ajtai (Ring-SIS, n = 128) row digests" + _IN_PROOF,
         (_SHA256_STMT, Hash("Ajtai n=128 (linear digest)", _STMT, "Ring-SIS binding", True), *_B_HASHES),
         assumptions=_B_ASSUME + "; Ring-SIS over F_p[X]/(X^128 + 1); rows of <= 128 columns (steps-pin)", pin=_B_PIN + "; steps <= n", **_B_KW),
    _cfg("B-Ligero native external authentication", _WEAK_REHASH, "x, W, y public", "frame-v3/sha256",
         "frame-v3 SHA-256 trees over the words of x, W, y, recomputed by the verifier from the published words",
         (_SHA256_STMT, *_B_HASHES), assumptions=_B_ASSUME, pin=_B_PIN, **_B_KW),
    _cfg("Ligerito", "weaker: the a/b words are statement rows (public)", "x, W public", None, "none (BLAKE3 Merkle over the witness)",
         _B_HASHES, "PCS term per level (Ligerito (17)) plus field terms; sumcheck 184/|F|",
         (Proximity("Ligerito PCS", "unique decoding", "proven (Ligerito paper eq. (17), sampling base ((1+rho)/2)^|S|)"),),
         "interactive (live verify session)", "binary extension field; the PCS ZK gap is open (red-team-ligerito-3)", "unknown"),
    _cfg("A-GKR CPU (Rust)", _WEAK_PRIVATE, "x, W private (unbound); y public endpoints", None,
         "none for the operands (SHA-256 Merkle over the RS columns is proof-internal)",
         (Hash("SHA-256", _INTERNAL, _CR), Hash("SHA-256", _COINS, _RO + " (Fiat–Shamir)")), _AGKR_DOMINANT, _AGKR_PROX,
         "transferable (SHA-256 Fiat–Shamir)", "Goldilocks^3 (or BabyBear^6); LogUp fractional-sum soundness", "unknown"),
    _cfg("A-GKR GPU (torch/Triton)", _WEAK_PRIVATE, "x, W private (unbound); y public endpoints", None,
         "none for the operands (SHA-512 Merkle over the RS columns is proof-internal)",
         (Hash("SHA-512", _INTERNAL, _CR), Hash("SHA-256", _COINS, _RO + " (Fiat–Shamir coins)")), _AGKR_DOMINANT, _AGKR_PROX,
         "transferable (SHA-256 Fiat–Shamir coins)", "BabyBear^6; LogUp fractional-sum soundness",
         "verity-gkr-verify refuses a circuit set its pins.txt does not list (PR #13)"),
    _cfg(ROUTE_A, FULL, "x, W committed (private) as keyed-BLAKE3 rows proved in Flock; y public", "frame-v3/blake3-row",
         "A-GKR prime proof of the relation, with the operands committed as keyed-BLAKE3 rows (blake3-keyed/row/v2) proved in Flock "
         "(flock-128-r2, live coins) and joined to the prime side by the sigma link (two GF(2^128) points); frame-v3 SHA-256 trees",
         (_SHA256_STMT, Hash("BLAKE3 (keyed)", _STMT, _CR + " (row leaves, proved in Flock)"), Hash("SHA-256", _COINS, "live transcript and root_F binding"),
          Hash("SHA-512", _INTERNAL, _CR + " (Merkle on the A-GKR Ligero side)")),
         "A-GKR's interactive terms (2^-130.19 whole proof); Flock flock-128-r2 2^-205 and the sigma link 2^-243.9 / 2^-246.4 do not bind",
         (*_AGKR_PROX, Proximity("Ligerito PCS in Flock (binary field)", "Johnson regime (list decoding)", _FLOCK_MCA)),
         "interactive: live-verifier-only (live prime and Flock coins in one session)",
         "collision resistance of SHA-256, keyed BLAKE3 and SHA-512 is a Table 1 assumption (decision 57); BabyBear^6 and GF(2^128) fields",
         "verity-gkr-verify pins (PR #13) plus the Flock profile params and the link version (session-2xgf2_128/v2, verity/link-L/v2)"),
    _cfg("SP1 stock (relation-only / bare guest)", _WEAK_PRIVATE, "x, W guest input (unbound)", None, "none for the operands",
         _SP1_HASHES, _SP1_DOMINANT, _SP1_PROX, "transferable (STARK, Fiat–Shamir)", _SP1_ASSUME,
         "verifying key (guest ELF digest), sp1-sdk =6.4.0"),
    _cfg("SP1 stock, FRI query count raised (otherwise unmodified)", _WEAK_PRIVATE, "x, W guest input (unbound)", None,
         "none for the operands", _SP1_HASHES, _SP1_DOMINANT + "; the FRI query term is 2^-134.7", _SP1_PROX,
         "transferable (STARK, Fiat–Shamir)", _SP1_ASSUME, "verifying key (guest ELF digest), sp1 6.6.0 with core FRI queries 175"),
    _cfg("SP1 stock, committed (frame-v3 SHA-256 rows)", FULL, _H_PORTS, "frame-v3/sha256",
         "frame-v3 SHA-256 trees over sha256/row/v1 row digests (SP1's SHA-256 precompile), one per operand row" + _IN_PROOF,
         (_SHA256_STMT, Hash("SHA-256", _STMT, _CR + " (row digests)"), *_SP1_HASHES), _SP1_DOMINANT, _SP1_PROX,
         "transferable (STARK, Fiat–Shamir)", _SP1_ASSUME,
         "verifying key of the relation-committed guest ELF, sp1-sdk =6.4.0; row schema sha256/row/v1"),
    _cfg("SP1 stock, committed (vllm-v1 SHA-256 position leaves)", FULL, _H_PORTS, "vllm-v1",
         "vllm-v1 SHA-256 trees over position leaves (SP1's SHA-256 precompile), one per operand row, bound by step roots over "
         "verifier-derived domains" + _IN_PROOF,
         (_SHA256_STMT, Hash("SHA-256", _STMT, _CR + " (position leaves)"), *_SP1_HASHES), _SP1_DOMINANT, _SP1_PROX,
         "transferable (STARK, Fiat–Shamir)", _SP1_ASSUME,
         "verifying key of the relation-committed-vllm guest ELF, sp1-sdk =6.4.0; vllm-v1 position leaves, step-root domains"),
    _cfg("SP1 precompile (TC_DOT chip)", _WEAK_PRIVATE, "x, W guest input (unbound)", None, "none for the operands", _SP1_HASHES,
         _SP1_DOMINANT, _SP1_PROX, "transferable (STARK, Fiat–Shamir)", _SP1_ASSUME + "; the TC_DOT chip's constraints",
         "verifying key plus the TC_DOT fork commit"),
)
_B_LIVE_MODEL = ("interactive protocol: 3 round trips with a live verifier (the step-0 coin commitment, then the column and opening "
                 "challenges), the verifier online for the whole proof; results verified from files (the runner's coins replayed) are "
                 "charged the protocol's sequential depth (3 per proof x the sub-batch proofs run one after another; derived: file-verified, no "
                 "measured RTT), network time = (proof + statement bytes) ÷ bandwidth")
_INTERACTION = {
    **{c.id: _B_LIVE_MODEL for c in CONFIGURATIONS if c.family == "B-Ligero"},
    "B-Ligero Fiat–Shamir (HVZK)": "non-interactive (Fiat–Shamir): rounds = 0, no one online",
    "Ligerito": "live verify session: the verifier is online for the session's rounds",
    "A-GKR CPU (Rust)": "non-interactive (SHA-256 Fiat–Shamir): rounds = 0; the interactive object is not implemented",
    "A-GKR GPU (torch/Triton)": "non-interactive (SHA-256 Fiat–Shamir coins): rounds = 0; the interactive object is not implemented",
    ROUTE_A: "live verifier: the prime side's coins and Flock's coins in one session (PROTOCOL 17.6); the verifier must be online for "
             "the whole proof",
    **{c.id: "non-interactive (Fiat–Shamir STARK): rounds = 0, no one online" for c in CONFIGURATIONS if c.family == "SP1"},
}
FLOCK_CONFIG = Configuration(
    "Flock (flock-128-r2, live coins)", "Flock", FULL, _H_PORTS, "frame-v3/blake3-row",
    "frame-v3 SHA-256 trees over keyed-BLAKE3 row digests; the union circuit proves the digests (chunk CVs) and y are public "
    "(lane flock-backend's design, unreviewed)",
    (_SHA256_STMT, Hash("BLAKE3 (keyed)", _STMT, _CR + " (row digests)"),
     Hash("BLAKE3 / SHA-256", _INTERNAL, _CR + " (Ligerito Merkle trees: BLAKE3 in the CPU prover, SHA-256 in Flock-CUDA)"),
     Hash("SHA-256", _COINS, "commit-before-coin binding of the live transcript")),
    "No: not zero-knowledge yet (declared class NON_ZK_PROOF; the Flock lane may change it)",
    "2^-205 for flock-128-r2 (two Fast100 runs with live verifier coins, no grinding credit; the table theorems at η = 1/200, "
    "soundness ASSUMPTIONS.md; the cells record the earlier, still valid 2^-195.5); red-team-flock: NOT GRANTED until "
    "R1–R8 (reps bound to one commitment, a live challenger); under Fiat–Shamir at most 2^-75.6",
    "the Ligerito query term per run (2^-97.8), squared by the two live-coin reps; it does not grow with the batch",
    (Proximity("Ligerito PCS (binary field)", "Johnson regime (list decoding)", _FLOCK_MCA),),
    "interactive: live-verifier-only", "GF(2^128) PIOP sites; the aarch64 AG r1 path gives about 14 bits of nonce choice (red-team-flock GAP)",
    "Flock b684b12 profile params pinned (commitment.params == expected)",
    interaction="live verifier: coins for every squeeze of both reps (Fiat–Shamir cannot reach 2^-128 for this profile); the verifier "
                "must be online for the whole proof")
#: Pure Flock over vllm-v1 commitments (``software.backend.name`` ``flock-vllm-v1``, statement ``verity/flock-vllm-block/v1``).
FLOCK_VLLM_CONFIG = dataclasses.replace(
    FLOCK_CONFIG, id="Flock over vllm-v1 (flock-128-r2, live coins)", scheme="vllm-v1",
    commitment="vllm-v1 SHA-256 trees over position leaves of every operand row, bound by step roots over the verifier's port "
               "domains; the verifier recomputes every bound root from its own instance file (VL1) and the circuit proves the fold "
               "(statement verity/flock-vllm-block/v1)",
    hashes=(_SHA256_STMT, Hash("SHA-256", _STMT, _CR + " (position leaves)"), *FLOCK_CONFIG.hashes[2:]),
    soundness="2^-205 per proof for flock-128-r2 (live verifier coins, no grinding credit), union 2^-203 over 4 proofs (the cells "
              "record the earlier, still valid 2^-195.44); "
              "red-team-flock: GRANTED WITH CONDITIONS at ff1c1e3f (VL1 closed at af6aae19)")
#: M0's generic circuit (statement ``verity/flock-circuit``): one pinned GF(2) circuit per template, the hm96-sha512 leaves
#: proved inside it and the frame-v3-sha512 trees over them recomputed by the verifier (the Lean verifier's PROTOCOL.md
#: §16.10, ``HmRow.checkPublic``).  The unit wording, the absent privacy claim and the time and AND qualifiers follow
#: red-team-flock-3's statement review (a plain row would call the attention unit a head, or read the hm96 leaves as hiding).
FLOCK_CIRCUIT_CONFIG = dataclasses.replace(
    FLOCK_CONFIG, id="C-Flock, generic circuit, SHA-512 in-circuit commitments", scheme="frame-v3-sha512/hm96-sha512",
    ports="per instance: one query row (attention: one query over T keys) or one output coordinate (GEMM); x, W committed; "
          "y public, equal to the IR reference on every proved instance",
    commitment="hm96-sha512 leaves proved in the circuit (the benchmark's salts are public); the frame-v3-sha512 trees over "
               "them recomputed by the verifier from the public file and held to its header's roots; commitment timed "
               "separately as serving's hm96-sha512 leaf and tree commitment of the committed values",
    hashes=(Hash("SHA-512", _STMT, _CR + " (frame-v3-sha512 trees, recomputed by the verifier)"),
            Hash("SHA-512", _STMT, _CR + " (hm96-sha512 leaves, in circuit)"),
            *FLOCK_CONFIG.hashes[2:3],
            Hash("SHA-512", _COINS, "commit-before-coin binding of the live transcript")),
    zk="No: NON_ZK_PROOF; no privacy claim (M0's mask slots are zero and the benchmark's hm96 salts are public)",
    soundness="2^-205 per table for flock-128-r2 (fast100 x2, live verifier coins); knowledge soundness under SHA-512 CR and "
              "Ligerito's named axioms",
    assumptions=FLOCK_CONFIG.assumptions + "; template instances as stated (M0's provisional unit cover), all proved, no unit draw",
    pin="verity/flock-circuit pinned circuit file (PR #83); the Lean verifier of record for this format (PR #142)",
    interaction="live verifier: coins for every squeeze of both reps; prover end-to-end time (witness, proof, live rounds, upload), "
                "verification separate")
CIRCUIT_FLAT, CIRCUIT_TYPED = "verity/flock-circuit", "verity/flock-circuit/types"
#: red-team-flock-3's C2 for the typed statement (Q4): #277 and #268 on main, and the typed path calling #268's checks.  Until it
#: holds, no typed cell or row says the circuit computes its type.
TYPED_C2 = False
#: Why a typed cell is not a Table 2 cell (nor in the workload headline): until C2 holds and Table 2 gives the typed statement
#: lines of its own, a typed cell would compete with a flat one for the same line.
TYPED_NOT_IN_TABLE2 = (f"{CIRCUIT_TYPED}: not a Table 2 cell until C2 holds (#277 and #268 on main, the typed path calling #268's "
                       "checks) and Table 2 shows typed and flat statements on separate lines")
#: M0's typed statement (``verity/flock-circuit/types``, PR #281): the same circuit route with typed ports.  Its own row and
#: backend, so a typed cell is never merged with a flat one (red-team-flock-3's Q4).
FLOCK_CIRCUIT_TYPED_CONFIG = dataclasses.replace(
    FLOCK_CIRCUIT_CONFIG, id="C-Flock, typed circuit (verity/flock-circuit/types), SHA-512 in-circuit commitments",
    zk=FLOCK_CIRCUIT_CONFIG.zk + "; the sessions accepted by the Rust live verifier (flock-circuit serve), no Lean replay"
       + ("; the circuit computes its type (#268's checks)" if TYPED_C2 else ""),
    pin=f"{CIRCUIT_TYPED} pinned circuit file (circuit_format flock-circuit/types, PR #281); compared with the flat statement per "
        "instance only")
#: M0's hidden-output statements (the Lean verifier's PROTOCOL.md §16.13, PR #757): the same names, every output port an hm96 row
#: hashed in the circuit, the identity's ``outputs`` saying so.  Before #757 the names meant the public-output forms (``@e51e2b86``,
#: ``types@210d32e1``), so a result is read by its identity, never by its name, and hidden and public results never share a cell.
_HIDDEN_PORTS = ("per instance: one query row (attention: one query over T keys) or one output coordinate (GEMM); x, W and y "
                 "committed, y as hm96-sha512 rows hashed in the circuit (the public file holds no output word); y equal to the IR "
                 "reference on every proved instance")
_HIDDEN_ZK = ("No: NON_ZK_PROOF; no privacy claim (M0's transcript is a function of the witness, output rows included: its mask "
              "slots are zero; and the benchmark's hm96 salts are public)")
FLOCK_CIRCUIT_HIDDEN_CONFIG = dataclasses.replace(
    FLOCK_CIRCUIT_CONFIG, id="C-Flock, generic circuit, hidden outputs, SHA-512 in-circuit commitments", ports=_HIDDEN_PORTS,
    zk=_HIDDEN_ZK, pin="verity/flock-circuit pinned circuit file with hidden outputs (PR #757); the Lean verifier of record "
                       "(Stmt.setupHidden: FailClosed.flock_headline_exec_hidden, in the forms the proof covers)")
FLOCK_CIRCUIT_TYPED_HIDDEN_CONFIG = dataclasses.replace(
    FLOCK_CIRCUIT_TYPED_CONFIG, id="C-Flock, typed circuit (verity/flock-circuit/types), hidden outputs, SHA-512 in-circuit commitments",
    ports=_HIDDEN_PORTS, zk=_HIDDEN_ZK + FLOCK_CIRCUIT_TYPED_CONFIG.zk[len(FLOCK_CIRCUIT_CONFIG.zk):],
    pin=f"{CIRCUIT_TYPED} pinned circuit file with hidden outputs (PR #757); compared with the flat statement per instance only")
HIDDEN_NOT_IN_TABLE2 = ("hidden outputs (PR #757): not a Table 2 cell until Table 2 shows hidden-output and public-output statements on "
                        "separate lines")
CONFIGURATIONS = (*(dataclasses.replace(c, interaction=_INTERACTION[c.id]) for c in CONFIGURATIONS), FLOCK_CONFIG, FLOCK_VLLM_CONFIG,
                  FLOCK_CIRCUIT_CONFIG, FLOCK_CIRCUIT_TYPED_CONFIG, FLOCK_CIRCUIT_HIDDEN_CONFIG, FLOCK_CIRCUIT_TYPED_HIDDEN_CONFIG)
_CONFIG = {c.id: c for c in CONFIGURATIONS}
assert set(_CONFIG) == set(D._BY_ID) | {FLOCK_CONFIG.id, FLOCK_VLLM_CONFIG.id, FLOCK_CIRCUIT_CONFIG.id, FLOCK_CIRCUIT_TYPED_CONFIG.id,
                                        FLOCK_CIRCUIT_HIDDEN_CONFIG.id, FLOCK_CIRCUIT_TYPED_HIDDEN_CONFIG.id}, \
    "every drilldown.VARIANTS entry is a declared configuration"
TYPED_CONFIGS = frozenset({FLOCK_CIRCUIT_TYPED_CONFIG.id, FLOCK_CIRCUIT_TYPED_HIDDEN_CONFIG.id})


def outputs_hidden(meta: Mapping[str, Any]) -> bool:
    """True iff the result's backend identity says its outputs are hidden rows (``outputs: "hidden: ..."``)."""
    fp = meta.get("workload_fingerprint") if isinstance(meta.get("workload_fingerprint"), Mapping) else {}
    identity = _get(fp, "software", "backend", "identity")
    return isinstance(identity, Mapping) and str(identity.get("outputs", "")).startswith("hidden")

TABLE1_COLUMNS = ("Configuration", "Flag", "Statement", "x, W, y", "Commitment scheme", "Hashes (role; property; class)", "ZK and verifier model",
                  "Soundness (whole-proof bound at its cells; dominant term)", "Proximity bounds", "Interaction model", "Transferability",
                  "Assumptions", "Pin", "Archived", "Cells")


def configuration_of(meta: Mapping[str, Any], configurations: Mapping[str, Configuration] | None = None) -> Configuration | None:
    """The declared configuration of a ``bench-result/v1`` result (the v1 adapter: the ``drilldown.VARIANTS`` entry that recognises
    it), or ``None``."""

    cfgs = configurations or _CONFIG
    v = D.classify(D.features(meta))
    if v is not None:
        return cfgs.get(v.id)
    fp = meta.get("workload_fingerprint") if isinstance(meta.get("workload_fingerprint"), Mapping) else {}
    name = _get(fp, "software", "backend", "name")
    if not FLOCK.recognises(name):
        return None
    if fp.get("statement") == CIRCUIT_FLAT:
        return cfgs.get((FLOCK_CIRCUIT_HIDDEN_CONFIG if outputs_hidden(meta) else FLOCK_CIRCUIT_CONFIG).id)
    if fp.get("statement") == CIRCUIT_TYPED:
        return cfgs.get((FLOCK_CIRCUIT_TYPED_HIDDEN_CONFIG if outputs_hidden(meta) else FLOCK_CIRCUIT_TYPED_CONFIG).id)
    return cfgs.get(FLOCK_VLLM_CONFIG.id if str(name).lower().startswith("flock-vllm-v1") else FLOCK_CONFIG.id)


def circuit_facts(meta: Mapping[str, Any]) -> dict[str, Any] | None:
    """A circuit-statement result's AND count, its convention and the verifier that accepted it (red-team-flock-3's Q4); ``None``
    for any other result.  A flat result recorded before #281 has no ``ands_counted``: its count is the tensor-core units' only."""
    fp = meta.get("workload_fingerprint") if isinstance(meta.get("workload_fingerprint"), Mapping) else {}
    if fp.get("statement") not in (CIRCUIT_FLAT, CIRCUIT_TYPED):
        return None
    counted = fp.get("ands_counted")
    verifier = fp.get("verifier") if isinstance(fp.get("verifier"), Mapping) else {}
    draw = fp.get("unit_draw")
    return {"statement": fp["statement"], "ands_per_instance": fp.get("ands_per_instance"), "rows_per_instance": fp.get("rows_per_instance"),
            "ands_label": "ANDs per instance: tensor-core units + tail" if counted else "ANDs per instance: units only (recorded before #281)",
            "not_counted": "in-circuit SHA-512 and hm96: proved but not counted",
            "accepted_by": verifier.get("accepted_by") or "the Rust live verifier (flock-circuit serve)",
            "computes_its_type": TYPED_C2 if fp["statement"] == CIRCUIT_TYPED else None,
            **({"unit_draw": f"{draw} (units_per_vu 1: a unit draw counts instances)"} if draw else {})}


# --- native throughput N ----------------------------------------------------------------------------------------------------------

#: The work model until ``verity.silicon`` declares per-primitive costs: a GemmCoordinate<K> is K/16 tensor-core steps of 16
#: multiply-adds (32 FLOP each) plus one FP32 -> operand-dtype conversion, which never binds; the peaks are ``contract.native_peak_of``.
WORK_MODEL = "work-model/v0: K/16 tensor-core steps x 32 FLOP (+1 conversion, not binding); peaks Target.native_peak"


def native(target: Target, K: int = T.K) -> dict[str, Any]:
    """``N`` for the target's subcircuit on its hardware class: inputs per second = 1 / max over units of ops / peak, the
    tensor cores being the busiest unit of a GEMM coordinate; with the busiest unit, the peak entry and the work model."""

    steps, flop = K // 16, (K // 16) * 32
    peak = contract.native_peak_of(target)
    return {"instances_per_second": peak.flop_per_second / flop, "unit": "tensor cores", "ops_per_instance": flop, "steps": steps,
            "dtype": (T.dtype_of(target) or "?").upper(), "peak_flop_per_second": peak.flop_per_second, "device": peak.device,
            "source": census.source(peak), "work_model": WORK_MODEL}


# --- the view's rules -------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Rules:
    """A view's parameters.  ``security_log2``: the whole-proof bound filter (``None`` = unfiltered); ``standard_hashes``: an
    optional filter (off in the published view) dropping configurations with an algebraic hash anywhere in the soundness
    argument.  ``interaction_validated``: an interactive result must carry the interaction record and be within tolerance of the
    reference profile (M otherwise); off only for a preview.  ``legacy``: today's
    parameters (the parity render): B = 4096 and K = 1536, the frozen range, security.target also <= the filter, the bare and
    ``+ in-proof hash`` columns instead of the full relation, proving time only, no custody or protocol check."""

    name: str
    security_log2: float | None = -128
    standard_hashes: bool = False
    legacy: bool = False
    include_archived: bool = False
    interaction_validated: bool = True
    #: previews of two pending acceptance decisions (Daniel), never published: the interaction check one-sided (pass when the
    #: measured wall is at most 1.1 x the model), and non-GEMM N on activations only (per-row weights and tables cached)
    interaction_one_sided: bool = False
    native_cached: bool = False
    #: a third preview: the workload headline counts a cleared cell on any GPU (N still the served hardware line's)
    headline_any_hardware: bool = False
    #: a fourth preview: the headline counts a cell whose ``omitted`` label names verifier-evaluated parts (top-p sampling)
    headline_count_omitted: bool = False
    #: Daniel's standing rule (2026-09-26), published: a cell whose verifier evaluates any part (attention's softmax, the RMSNorms'
    #: row tail, sampling's keep word and noise: the census's ``verifier_evaluated``) does not cover its subcircuit; overrides
    #: ``headline_count_omitted``
    headline_strict_verifier: bool = False


SPEC = Rules("spec", headline_strict_verifier=True)
UNFILTERED = Rules("unfiltered", security_log2=None)
LEGACY = Rules("legacy", legacy=True)
RULES = {r.name: r for r in (SPEC, UNFILTERED, LEGACY)}


def _num(v: Any) -> bool:
    return T._num(v)


def _get(d: Any, *keys: str) -> Any:
    return D._get(d, *keys)


def whole_proof_log2(fp: Mapping[str, Any]) -> float | None:
    """The whole-proof bound the accountant recorded at the result's batching (``security.achieved_log2``), else its target."""
    sec = fp.get("security") if isinstance(fp.get("security"), Mapping) else {}
    for k in ("achieved_log2", "target"):
        if _num(sec.get(k)):
            return float(sec[k])
    return None


# --- interaction: t = compute + rounds x RTT + bytes / bandwidth (reporting standard, Daniel 2026-09-25) --------------------------

#: The reference network the Table 2 cell is evaluated at (``census.REFERENCE_NETWORK``, ``benchmarks/census/networks.json``: same datacenter); the
#: interaction drill-down adds ``census.SENSITIVITY_NETWORKS``.
REF_RTT_S, REF_BANDWIDTH_BPS = census.REFERENCE_NETWORK.rtt_s, census.REFERENCE_NETWORK.bandwidth_bps
#: A measured interactive run must be within this fraction of the formula at the reference profile.
INTERACTION_TOLERANCE = 0.10


#: A result's statement bytes (sent beside the proof): the relation's statements over the run's sub-batch proofs.
STATEMENT_BYTES = ("relation.statement_bytes_total", "relation.hash.statement_bytes_total")


def interaction(meta: Mapping[str, Any], interactive: bool = False) -> dict[str, Any]:
    """The interaction record of a result.  ``live``: it ran against a live verifier (live-verifier evidence, or a measured RTT,
    network wait or ``t.total_live``).

    A **live** result's record is measured: ``rounds`` (round trips on the critical path: ``rounds.sequential_depth``, else
    ``rounds.protocol``), ``bytes_up`` (``net.bytes_out``, else the proof bytes), ``bytes_down`` (``net.bytes_in``, the coins),
    ``rtt_ms`` (``net.rtt_ms``) with ``rtt_method`` (the ``interaction`` block's, or the live-verifier evidence's), and the wall
    split: ``prover_s`` (``t.total``), ``verifier_s`` (verifier compute the prover waited on: ``verify.critical_s``, else 0 -- the
    measured wait includes it) and ``wait_s`` (``net.wait_seconds``, else ``t.total_live - t.total``).  A record whose ``interaction``
    block says ``t_total_includes_wait`` (Flock's live coins) has its prover compute in ``live.prover_compute_seconds`` (else
    ``t.total`` minus the wait).  The prover's own per-round live-coin handling, measured on loopback
    (``live.loopback_round_seconds``), is prover compute: rounds x that (at most the measured wait) moves from the wait into
    ``prover_s``, and only the excess over loopback counts as network wait (``loopback_s`` records the amount moved).

    Any other result is a proof verified from its files and its record is derived (``derived``): bytes up = the proof bytes
    (``proof_bytes``, else ``transcript.bytes``) plus the statement bytes (:data:`STATEMENT_BYTES`), no measured RTT.  Its
    rounds (``rounds_source``): a Fiat–Shamir proof has none (0); an ``interactive`` protocol's file-replayed transcript (every
    B-interactive cell today) pays the protocol's sequential depth, ``"derived: file-verified"``: the per-proof depth
    (``rounds.sequential_depth``, else ``rounds.protocol``) x the proofs that run one after another -- ``N_subbatches``
    sub-batch proofs, ``software.backend.pipeline`` of them in flight at once, so ceil(N_subbatches / pipeline) waves (root's
    ruling 2026-09-26)."""

    m = T._measurements(meta)
    fp = meta.get("workload_fingerprint") if isinstance(meta.get("workload_fingerprint"), Mapping) else {}
    block = meta.get("interaction") if isinstance(meta.get("interaction"), Mapping) else fp.get("interaction")
    block = block if isinstance(block, Mapping) else {}
    lv = _get(meta, "validation", "evidence", "live_verifier")
    lv = lv if isinstance(lv, Mapping) else {}
    num = lambda k: float(m[k]) if _num(m.get(k)) else None  # noqa: E731
    proof = next((v for v in (num("proof_bytes"), num("transcript.bytes")) if v is not None), None)
    live = bool(lv) or any(num(k) is not None for k in ("net.rtt_ms", "net.wait_seconds", contract.T_TOTAL_LIVE))
    if not live:
        stmt = next((v for v in (num(k) for k in STATEMENT_BYTES) if v is not None), 0.0)
        rounds, source = 0.0, "fiat-shamir"
        if interactive:
            depth = next((v for v in (num("rounds.sequential_depth"), num("rounds.protocol")) if v is not None), 0.0)
            proofs, flight = fp.get("N_subbatches"), _get(fp, "software", "backend", "pipeline")
            proofs = proofs if _num(proofs) and proofs >= 1 else 1
            waves = math.ceil(proofs / flight) if _num(flight) and flight >= 1 else proofs
            rounds, source = depth * waves, "derived: file-verified"
        return {"live": False, "derived": True, "rounds": rounds, "rounds_source": source, "bytes_up": (proof or 0.0) + stmt, "bytes_down": 0.0,
                "proof_bytes": proof, "statement_bytes": stmt, "rtt_ms": None, "rtt_method": None,
                "prover_s": num(contract.T_TOTAL), "verifier_s": 0.0, "wait_s": None, "loopback_s": 0.0,
                "loopback_round_s": None, "loopback_method": None}
    rounds = num("rounds.sequential_depth")
    rounds = num("rounds.protocol") if rounds is None else rounds
    up = next((v for v in (num("net.bytes_out"), proof) if v is not None), None)
    wait = num("net.wait_seconds")
    if wait is None and num(contract.T_TOTAL_LIVE) is not None and num(contract.T_TOTAL) is not None:
        wait = max(0.0, num(contract.T_TOTAL_LIVE) - num(contract.T_TOTAL))
    compute = num(contract.T_TOTAL)
    if block.get("t_total_includes_wait") is True and compute is not None and wait is not None:
        compute = next((v for v in (num("live.prover_compute_seconds"),) if v is not None), compute - wait)
    loop, moved = num("live.loopback_round_seconds"), 0.0
    if loop is not None and rounds and wait is not None and compute is not None:
        moved = min(wait, rounds * loop)
        compute, wait = compute + moved, wait - moved
    return {"live": True, "derived": False, "rounds": rounds, "rounds_source": "measured", "bytes_up": up or 0.0, "bytes_down": num("net.bytes_in") or 0.0,
            "proof_bytes": proof, "statement_bytes": None, "rtt_ms": num("net.rtt_ms"),
            "rtt_method": block.get("rtt_method") or lv.get("rtt_method"), "bandwidth_bps": num("net.bandwidth_bps"), "prover_s": compute,
            "verifier_s": num("verify.critical_s") or 0.0, "wait_s": wait, "loopback_s": moved,
            "loopback_round_s": loop if moved else None, "loopback_method": block.get("loopback_method") if moved else None}


def interaction_time(ix: Mapping[str, Any], rtt_s: float = REF_RTT_S, bandwidth_bps: float = REF_BANDWIDTH_BPS) -> float:
    """t = compute + rounds x RTT + bytes / bandwidth, compute = prover compute + the verifier compute the prover waits on."""
    return ((ix["prover_s"] or 0.0) + ix["verifier_s"] + (ix["rounds"] or 0.0) * rtt_s
            + (ix["bytes_up"] + ix["bytes_down"]) * 8 / bandwidth_bps)


def interaction_deviation(ix: Mapping[str, Any]) -> float | None:
    """(measured - model) / model at the run's own RTT and bandwidth (the reference's when none was measured), for a live result
    with its rounds, RTT and wait; else ``None``."""
    if not ix["live"] or ix["rounds"] is None or ix["rtt_ms"] is None or ix["wait_s"] is None:
        return None
    model = interaction_time(ix, ix["rtt_ms"] / 1e3, ix.get("bandwidth_bps") or REF_BANDWIDTH_BPS)
    return ((ix["prover_s"] or 0.0) + ix["wait_s"] - model) / model if model > 0 else None


def interaction_problem(ix: Mapping[str, Any], one_sided: bool = False) -> str | None:
    """Why a result does not meet the interaction standard, or ``None``.  A non-interactive (file-verified) result's record is derived
    and always meets it.  A live-verifier result records its rounds, its measured RTT with how it was measured and its network wait,
    and its measured wall time (prover compute + network wait) is within :data:`INTERACTION_TOLERANCE` of the formula evaluated at
    the run's own measured RTT (``net.rtt_ms``) and bandwidth (``net.bandwidth_bps``, else the reference's): the check validates
    the model against the run itself; the published figure is the formula at the reference profile (root's ruling 2026-09-25)."""
    if not ix["live"]:
        return None
    if ix["rounds"] is None:
        return "interaction: a live-verifier result without its rounds (rounds.sequential_depth)"
    missing = [k for k, ok in (("measured RTT (net.rtt_ms)", ix["rtt_ms"] is not None), ("how the RTT was measured", bool(ix["rtt_method"])),
                               ("network wait (net.wait_seconds or t.total_live)", ix["wait_s"] is not None)) if not ok]
    if missing:
        return f"interaction: {ix['rounds']:.0f} rounds but no {', no '.join(missing)}"
    bw = ix.get("bandwidth_bps") or REF_BANDWIDTH_BPS
    model = interaction_time(ix, ix["rtt_ms"] / 1e3, bw)
    measured = (ix["prover_s"] or 0.0) + ix["wait_s"]
    dev = interaction_deviation(ix)
    if dev is not None and (dev if one_sided else abs(dev)) > INTERACTION_TOLERANCE:
        return (f"interaction: measured {measured:.3g} s is {100 * dev:+.0f}% from the model's {model:.3g} s at the run's "
                f"own network ({ix['rtt_ms']:.3g} ms RTT, {bw / 1e9:.3g} Gb/s{'' if ix.get('bandwidth_bps') else ' (reference; not measured)'}; "
                f"tolerance ±{100 * INTERACTION_TOLERANCE:.0f}%)")
    return None


def throughput(meta: Mapping[str, Any], rules: Rules, interactive: bool = False) -> dict[str, Any] | None:
    """``P``: instances per timed run over its end-to-end seconds (committing the batch, :data:`COMMIT`, then the interaction
    formula at the reference profile: prover compute + rounds x 1 ms + bytes / 100 Gb/s; ``interactive``: the configuration's
    protocol is interactive, :func:`interaction`); under the legacy rules proving only.  ``None`` without a positive ``t.total`` and batch."""

    fp = meta.get("workload_fingerprint") if isinstance(meta.get("workload_fingerprint"), Mapping) else {}
    m = T._measurements(meta)
    t, n = m.get(contract.T_TOTAL), fp.get("B")
    if not (_num(t) and t > 0 and _num(n) and n > 0):
        return None
    commit = 0.0 if rules.legacy else (float(m[COMMIT]) if _num(m.get(COMMIT)) else 0.0)
    ix = interaction(meta, interactive)
    net = 0.0 if rules.legacy else interaction_time(ix) - float(t)
    e2e = float(t) + net + commit
    return {"instances": n, "proving_s": float(t), "commit_s": commit, "interaction_s": net, "end_to_end_s": e2e,
            "instances_per_second": n / e2e, "commit_v1_s": m.get(COMMIT_V1) if _num(m.get(COMMIT_V1)) else None, "interaction": ix}


# --- admissibility ----------------------------------------------------------------------------------------------------------------


@dataclass
class Ctx:
    """What the predicate reads about one result."""

    art: str
    meta: Mapping[str, Any]
    labels: list[Any]
    attempt: Mapping[str, Any] | None
    target: Target
    family: Family
    config: Configuration | None
    statement_labels: list[list[Any]]      #: each result's labels, for every result of this configuration, target and subcircuit
    subcircuit: str | None = None                         #: a statement other than the target's own (:func:`result_subcircuit`)
    input_sets: Mapping[str, tuple[str, Mapping[str, Any]]] = field(default_factory=dict)  #: registered input sets (:func:`input_set_index`)
    variables: dict[str, int] = field(default_factory=dict)  #: its per-input work-model variables (attention's T: :func:`input_variables`)
    key_class: dict[str, Any] | None = None               #: a class cell's key counts and per-T timings (:func:`key_class_of`)


def _column(ctx: Ctx, rules: Rules) -> str:
    """The Table 2 column a result competes in: its family; under the legacy rules family + authentication column."""
    if not rules.legacy:
        return ctx.family.id
    auth = T.column_authentication(ctx.meta["workload_fingerprint"])
    return ctx.family.id + T.AUTH_COLUMN_SUFFIX[auth]


def _custody(store: Any, ctx: Ctx) -> str | None:
    """Why the result's proof files are not shown preserved in the catalog, or ``None``: a proof or run-files artifact referenced
    by the result or its producing attempt, each on the remote as the catalog records it.  (``research data preserved`` is the full check.)"""
    try:
        refs = store.get_manifest(ctx.art).to_doc().get("refs") or {}
    except Exception:  # noqa: BLE001
        refs = {}
    outs = (ctx.attempt or {}).get("outputs") if isinstance((ctx.attempt or {}).get("outputs"), Mapping) else {}
    files = [v for v in (refs.get("proof"), refs.get("run_files"), outs.get("run_files"), outs.get("run_record")) if isinstance(v, str)]
    if not files:
        return "no proof files: neither the result (refs.proof / refs.run_files) nor its attempt names a dump"
    def on_remote(a: str) -> bool:
        row = store.index.artifact(a)
        return bool(row and row.get("remote")) or "remote" in store.index.replicas(a)

    try:
        missing = [a for a in files if not on_remote(a)]
    except Exception:  # noqa: BLE001  (a store without a catalog)
        return "proof files unknown to the catalog"
    return f"proof files {', '.join(missing)} have no remote replica recorded (`research data preserved`)" if missing else None


def _protocol(meta: Mapping[str, Any], config: Configuration | None) -> str | None:
    """Why the result was not measured under the protocol, or ``None``: warm, >= 5 timed runs, uncontended, a plateau point, and
    (for a full-relation statement) its commitment timed (:data:`COMMIT`)."""
    fp = meta.get("workload_fingerprint") if isinstance(meta.get("workload_fingerprint"), Mapping) else {}
    proto = meta.get("protocol") if isinstance(meta.get("protocol"), Mapping) else fp.get("protocol")
    sweep = meta.get("sweep") if isinstance(meta.get("sweep"), Mapping) else fp.get("sweep")
    if not isinstance(proto, Mapping):
        return "protocol not recorded (warm, timed runs, contention verdict): a bench-result/v1 result"
    runs = proto.get("runs")
    if proto.get("warm") is not True or not (_num(runs) and runs >= MIN_RUNS) or proto.get("contended") is not False:
        return f"protocol: warm {proto.get('warm')!r}, runs {runs!r}, contended {proto.get('contended')!r} (need warm, >= {MIN_RUNS}, uncontended)"
    if not isinstance(sweep, Mapping) or sweep.get("plateau") is not True:
        return "not a plateau point of a recorded sweep"
    if config is not None and config.full_relation and not _num(T._measurements(meta).get(COMMIT)):
        return f"commitment not timed ({COMMIT} absent)"
    return None


def _security(fp: Mapping[str, Any], rules: Rules, config: Configuration | None) -> list[tuple[str, str]]:
    out = []
    if rules.standard_hashes and config is not None and config.algebraic():
        out.append(("L", "an algebraic hash in the soundness argument: " + "; ".join(h.text() for h in config.algebraic())))
    if rules.security_log2 is None:
        return out
    if rules.legacy:
        out = []
        tgt, ach = _get(fp, "security", "target"), _get(fp, "security", "achieved_log2")
        if not _num(tgt) or tgt > rules.security_log2:
            out.append(("L", f"security.target is {tgt!r}, need <= {rules.security_log2:g}"))
        if ach is not None and (not _num(ach) or ach > rules.security_log2):
            out.append(("L", f"security.achieved_log2 is {ach!r}, need <= {rules.security_log2:g}"))
        return out
    bound = whole_proof_log2(fp)
    if bound is None or bound > rules.security_log2:
        out.append(("L", f"whole-proof bound {_fmt_bound(bound)} is below the filter 2^{rules.security_log2:g}"))
    return out


def _recipe(text: str) -> Callable[[int], str]:
    return lambda n: hashlib.sha256(text.format(n=n).encode()).hexdigest()


#: The synthetic sets as streams: instance i is a function of (generator, seed, i) alone (``relchain._instance``,
#: ``fp4.chain.instances_fp4``), so the set over [0, n) is a prefix-consistent extension of the frozen [0, 4096).  Their
#: ``manifest_sha256`` hashes the generator, seed and n (``relchain.instances_digest``, ``fp4.chain.instances_digest``); each
#: recipe here reproduces the digest ``contract`` pins at n = 4096 (tested).  The A100 set is a fixture of 4,096 inputs with
#: no stream beyond them: a larger batch repeats instances, which can inflate throughput, so it is rejected (I).
STREAMS: dict[str, Callable[[int], str]] = {
    FP8_ADA_MMA.name: _recipe(f"fp8-ada synthetic|seed={contract.FP8_ADA_INSTANCES_SEED}|n={{n}}|K={T.K}"),
    BF16_HOPPER.name: _recipe(f"bf16-hopper synthetic|seed={contract.BF16_HOPPER_INSTANCES_SEED}|n={{n}}|K={T.K}"),
    FP8_HOPPER_WGMMA.name: _recipe(f"fp8-hopper synthetic|seed={contract.FP8_HOPPER_INSTANCES_SEED}|n={{n}}|K={T.K}"),
    NVFP4_SM120.name: _recipe(f"fp4-nvf4 synthetic|seed={contract.NVFP4_INSTANCES_SEED}|n={{n}}|K={T.K}|families="
                              + ",".join(contract.NVFP4_INSTANCES_FAMILIES)),
}
_STREAM_SEEDS = {FP8_ADA_MMA.name: contract.FP8_ADA_INSTANCES_SEED, BF16_HOPPER.name: contract.BF16_HOPPER_INSTANCES_SEED,
                 FP8_HOPPER_WGMMA.name: contract.FP8_HOPPER_INSTANCES_SEED, NVFP4_SM120.name: contract.NVFP4_INSTANCES_SEED}


def instance_range(target: Target, inst: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """``(how, None)`` when ``inst`` is the target's declared input set over [0, n) -- ``how`` is ``"frozen"`` (exactly the
    frozen ref), ``"prefix"`` (a prefix of it) or ``"stream"`` (a prefix-consistent extension of a synthetic set, identified by
    generator, seed and n) -- else ``(None, why)``, ``why`` ``None`` when it simply names another set."""

    want = T.frozen_instances(target)
    if want is None or any(inst.get(k) != want[k] for k in ("dataset", "tier")):
        return None, None
    rng = inst.get("range")
    if not (isinstance(rng, (list, tuple)) and len(rng) == 2 and rng[0] == 0 and isinstance(rng[1], int) and rng[1] > 0):
        return None, None
    n, size = rng[1], want["range"][1]
    if n == size:
        return ("frozen", None) if inst.get("manifest_sha256") == want["manifest_sha256"] else (None, None)
    recipe = STREAMS.get(target.name)
    if recipe is not None:
        seed = inst.get("seed", _STREAM_SEEDS[target.name])
        if seed == _STREAM_SEEDS[target.name] and inst.get("manifest_sha256") == recipe(n):
            return ("prefix" if n < size else "stream"), None
        return None, None
    if inst.get("manifest_sha256") != want["manifest_sha256"]:
        return None, None
    if n < size:
        return "prefix", None
    return None, (f"instances [0, {n}) of {want['tier']}: the frozen set holds {size} instances and has no stream beyond them, "
                  f"so this batch repeats instances (repeats can inflate throughput)")


def _declared_side(target: Target, frozen: Any, inst: Mapping[str, Any]) -> str | None:
    """Why an ``instance-equiv/v1`` document's ``frozen`` side is not the target's declared set over the candidate's range, or
    ``None``.  The side is an instance ref as the runner writes it (dataset, tier, range, manifest_sha256, and for a synthetic
    set seed and recipe): the frozen ref itself, a prefix of it, or a synthetic stream over [0, n) by generator, seed and n
    (:func:`instance_range`).  A fixture's repeats (A100 past 4,096) never are."""

    if not isinstance(frozen, Mapping):
        return "frozen is not an instance ref"
    if T._ref_key(frozen) != T._ref_key(T.frozen_instances(target)):
        how, why = instance_range(target, frozen)
        if how is None:
            return why or "frozen is neither FROZEN_INSTANCES[target] nor the declared set's stream over its range (generator, seed and n)"
    fr, cr = frozen.get("range"), inst.get("range")
    if not (isinstance(fr, (list, tuple)) and isinstance(cr, (list, tuple)) and len(fr) == len(cr) == 2 and fr[1] - fr[0] == cr[1] - cr[0]):
        return f"frozen range {fr!r} and candidate range {cr!r} do not hold the same number of instances"
    return None


def stream_equivalence(inst: Mapping[str, Any], target: Target, equivs: Sequence[T.InstanceEquiv] | None,
                       result_producers: set[str]) -> tuple[str | None, str | None]:
    """:func:`tables.instance_equivalence` with the frozen side generalised by :func:`_declared_side`: ``(art id, None)`` of an
    ``instance-equiv/v1`` document that shows the result's instances equal to the declared set over the same range and is
    ``verified=accepted`` by a non-producer, else ``(None, why)``.  One document shape serves 4,096 and any stream n."""

    pool = [e for e in equivs or () if e.doc.get("target") == target.name and T._ref_key(e.doc.get("candidate")) == T._ref_key(inst)]
    if not pool:
        return None, T.unreadable_equivs(equivs, target)
    judged = []
    for e in pool:
        problems = [x for x in T._equiv_content(e.doc, target, inst, e.unreadable) if not x.startswith("frozen is not FROZEN_INSTANCES")]
        side = _declared_side(target, e.doc.get("frozen"), inst)
        judged.append((e, problems + ([side] if side else []), T._equiv_verdict(e, result_producers)))
    contra = [e for e, _, v in judged if v is None and e.doc.get("equal") is not True]
    if contra:
        return None, f"{contra[0].artifact}, verified, says equal={contra[0].doc.get('equal')!r}"
    ok = [e for e, c, v in judged if not c and v is None]
    if ok:
        return ok[0].artifact, None
    return None, "; ".join(f"{e.artifact}: {c[0]}" if c else str(v) for e, c, v in judged)


def _equiv_of(ctx: Ctx, rules: Rules, equivs: Sequence[T.InstanceEquiv]) -> str | None:
    """The equivalence artifact a re-packed result counts through (its footnote), or ``None``."""
    if rules.legacy:
        return T.instances_equiv_of(ctx.meta, ctx.labels, ctx.attempt, ctx.target, equivs)
    inst, want = ctx.meta["workload_fingerprint"].get("instances"), T.frozen_instances(ctx.target)
    if not isinstance(inst, Mapping) or want is None or not T.instances_diff(inst, want) or instance_range(ctx.target, inst)[0] is not None:
        return None
    return stream_equivalence(inst, ctx.target, equivs, T.producers(ctx.attempt, ctx.labels, ctx.meta))[0]


#: A result's ``relation`` (before any ``-x4-k<K>`` / ``+<leaf>`` suffix) naming its GEMM coordinate's semantics (the backends'
#: relation names: ``verity_flock.bench.PROFILES``, ``ligero.templates.gemm_coordinate.RELATIONS``).
RELATION_SEMANTICS = {"bf16-ampere": "sm80.mma.m16n8k16.bf16", "bf16-hopper": "sm90.mma.m16n8k16.bf16",
                      "bf16-hopper-wgmma": "sm90.wgmma.m64n8k16.bf16", "fp8-ada": "sm89.mma.m16n8k32.e4m3",
                      "fp8-hopper": "sm90.wgmma.m64n8k32.e4m3", "fp4-nvf4": "sm120.mma.m16n8k64.e2m1.nvf4"}
#: The kinds of a registered input set a real-K result's inputs name.
INPUT_SET_KINDS = ("input-set/v1", "vllm-vu-set/v1")


#: Hardware lines no target anchors: a target's instruction on another device (``README.md``'s Glossary, "Cell": prover
#: hardware is its own axis).  #101 was served on an L40S, whose BF16 GEMM is the Ampere-class mma the first target states.  Every result on such
#: a line is a statement of its own (:func:`result_subcircuit`), so the line has rows only where it has results.
L40S_BF16 = dataclasses.replace(FIRST, name="l40s-48gb/bf16/sm80.mma.m16n8k16.bf16", anchor_device="NVIDIA L40S 48GB", status="draft", dossier=None)
#: Every Table 2 line by name: the targets, then the hardware-only lines.
LINES: dict[str, Target] = {**TARGETS, L40S_BF16.name: L40S_BF16}


def rehome(meta: Mapping[str, Any], target: Target) -> Target:
    """The line a result belongs on when its prover ran on another device than ``target``'s: for a GEMM statement the line of
    the same instruction on the prover's device, for another template the line :func:`hardware_target` picks; ``target``
    itself when the prover is on its device or no line fits (then the result is rejected K as before)."""
    gpu = _get(meta.get("workload_fingerprint") or {}, "hardware", "gpu", "name")
    if T.same_sku(gpu, contract.native_peak_of(target).device):
        return target
    sub = stamped_subcircuit(meta)
    if not is_gemm(sub):
        return hardware_target(meta, sub) or target
    return next((t for t in LINES.values() if t.instruction == target.instruction and T.same_sku(gpu, contract.native_peak_of(t).device)), target)


def _sem_slug(semantics: str) -> str:
    return "-".join(x for x in semantics.split(".") if not re.fullmatch(r"m\d+n\d+k\d+", x))


def result_subcircuit(meta: Mapping[str, Any], target: Target) -> str | None:
    """The census subcircuit of a result's statement when it is not its target's own -- a statement of its own, so a Table 2
    row of its own (``README.md``'s Glossary, "Subcircuit": K and the semantics are part of it) -- else ``None``.

    K is the fingerprint's.  The semantics are the ``bench.cell`` stamp's subcircuit, else the result's ``relation``
    (:data:`RELATION_SEMANTICS`), else the target's instruction.  Semantics the census records as one (``stood_for_by``) are
    one statement: at the target's K the target's own subcircuit, at another K the one that is stood for (root's ruling
    2026-09-26 02:44Z: real-K H100 BF16 cells use the wgmma subcircuit's id)."""
    fp = meta.get("workload_fingerprint") if isinstance(meta.get("workload_fingerprint"), Mapping) else {}
    extra = target.name not in TARGETS
    home = census.subcircuit_with("gemm-coordinate", K=T.K, semantics=target.instruction) if extra else census.subcircuit_of(target.name)
    named = stamped_subcircuit(meta)
    if named and named.split("/")[0] != home.template:
        return named                                         # another template (a non-GEMM subcircuit): its own statement
    K = fp.get("K") if isinstance(fp.get("K"), int) and not isinstance(fp.get("K"), bool) else home.parameters.K
    sem = None
    stamped = (meta.get("cell") or {}).get("subcircuit") if isinstance(meta.get("cell"), Mapping) else None
    if isinstance(stamped, str):
        known = census.SUBCIRCUITS.get(stamped)
        slug = stamped.rsplit("/", 1)[-1]
        sem = known.parameters.semantics if known else next(
            (x.parameters.semantics for x in census.SUBCIRCUITS.values() if _sem_slug(x.parameters.semantics) == slug), None)
    if sem is None:
        rel = fp.get("relation") if isinstance(fp.get("relation"), str) else meta.get("relation")
        sem = RELATION_SEMANTICS.get(re.split(r"-x\d+|-k\d+|\+", rel)[0]) if isinstance(rel, str) else None
    sem = sem or home.parameters.semantics
    stands = {census.SUBCIRCUITS[x].parameters.semantics: census.SUBCIRCUITS[census.stood_for_by(x)].parameters.semantics
              for x in census.SUBCIRCUITS if census.stood_for_by(x)}
    if K == home.parameters.K and not extra:
        sem = stands.get(sem, sem)
        if sem == home.parameters.semantics:
            return None
    else:
        sem = {v: k for k, v in stands.items()}.get(sem, sem)
    found = census.subcircuit_with(home.template, K=K, semantics=sem)
    return found.id if found else f"{home.template}/k{K}/{_sem_slug(sem)}"


def stamped_subcircuit(meta: Mapping[str, Any]) -> str | None:
    """The subcircuit a result names itself: the ``bench.cell`` stamp's, else the fingerprint's or the meta's ``subcircuit``."""
    fp = meta.get("workload_fingerprint") if isinstance(meta.get("workload_fingerprint"), Mapping) else {}
    cell = meta.get("cell") if isinstance(meta.get("cell"), Mapping) else {}
    return next((v for v in (cell.get("subcircuit"), fp.get("subcircuit"), meta.get("subcircuit")) if isinstance(v, str)), None)


#: GPU model tokens a prover's ``hardware.gpu.name`` carries, as the census hardware lines' devices spell them.
_GPU_TOKENS = ("A100", "H100", "L40S", "4090", "5090")
_DTYPE_TOKENS = (("bf16", "bf16"), ("e4m3", "e4m3"), ("fp8", "e4m3"), ("e2m1", "e2m1"), ("nvf4", "e2m1"))


def hardware_target(meta: Mapping[str, Any], sub: str | None) -> Target | None:
    """The target whose census hardware line a non-GEMM result ran on, for a result with no ``profile`` (a ``bench.cell`` cell
    on a subcircuit of another template): the prover GPU's model and the subcircuit's datatype pick the line; ``None`` when none
    matches (then the result is rejected X as before)."""
    gpu = str(_get(meta.get("workload_fingerprint") or {}, "hardware", "gpu", "name") or "")
    token = next((t for t in _GPU_TOKENS if t in gpu), None)
    tail = (sub or "").rsplit("/", 1)[-1].lower()
    dtype = next((d for key, d in _DTYPE_TOKENS if key in tail), None)
    if token is None:
        return None
    on = [t for t in LINES.values() if token in contract.native_peak_of(t).device]
    if dtype is None:                      # a subcircuit without a tensor datatype (fp32 sampling): its N is memory-bound, any line of the device
        return next((t for t in on if contract.native_peak_of(t).dtype == "bf16"), on[0] if on else None)
    return next((t for t in on if contract.native_peak_of(t).dtype == dtype), None)


def is_gemm(sub: str | None) -> bool:
    return sub is None or sub.split("/")[0] == "gemm-coordinate"


def native_of(target: Target, sub: str | None, cached: bool = False, variables: Mapping[str, int] | None = None) -> dict[str, Any]:
    """``N`` of a row: the GEMM work model (:func:`native`) at the subcircuit's K, or for another template its census
    memory-bandwidth work model (:func:`memory_native`) at the row's per-input ``variables`` (attention's T)."""
    return native(target, subcircuit_k(sub) if sub else T.K) if is_gemm(sub) else memory_native(target, sub, cached, variables)


def input_variables(meta: Mapping[str, Any], sub: str | None, input_sets: Mapping[str, tuple[str, Mapping[str, Any]]]) -> tuple[dict[str, int], str | None]:
    """The per-input variables a result's statement fixes (its work model's ``variables`` its expressions use: attention's T) and
    ``None``; ``({}, why)`` when one is missing or recorded differently in two places; ``({}, None)`` for a template without
    them.  A value is read wherever a result records it: the fingerprint's or the cell stamp's ``statics`` / ``variables``, the
    instances ref, the lowered unit's name (flock-ir-lowering's ``attention-head/fa2-bf16/tc-step/d64-bn128/t4``), the input set's
    name, and the registered set's parameters, relation statics or name."""
    known = census.SUBCIRCUITS.get(sub or "")
    wm = census.SUBCIRCUIT_TEMPLATES[known.template].work_model if known else None
    names = [v for v in vars(getattr(wm, "variables", None) or SimpleNamespace()) if v in str(wm.ops_per_input) + str(getattr(wm, "bytes_per_input", ""))]
    if not names:
        return {}, None
    fp = meta.get("workload_fingerprint") if isinstance(meta.get("workload_fingerprint"), Mapping) else {}
    cell = meta.get("cell") if isinstance(meta.get("cell"), Mapping) else {}
    inst = fp.get("instances") if isinstance(fp.get("instances"), Mapping) else {}
    keys = [inst.get(k) for k in ("art", "set_art", "content_digest", "manifest_sha256")] + [(cell.get("input_set") or {}).get("art")
                                                                                              if isinstance(cell.get("input_set"), Mapping) else None]
    reg = next((input_sets[k][1] for k in keys if isinstance(k, str) and k in input_sets), {})
    out: dict[str, int] = {}
    for name in names:
        seen: dict[int, list[str]] = {}

        def put(v: Any, where: str) -> None:
            if isinstance(v, bool):
                return
            if isinstance(v, int) or (isinstance(v, str) and v.isdigit()):
                seen.setdefault(int(v), []).append(where)

        for where, d in (("statics", fp.get("statics")), ("variables", fp.get("variables")), ("instance_statics", fp.get("instance_statics")),
                         ("cell.statics", cell.get("statics")), ("cell.variables", cell.get("variables")), ("instances", inst),
                         ("input set parameters", reg.get("parameters")),
                         ("input set relation statics", (reg.get("relation") or {}).get("statics") if isinstance(reg.get("relation"), Mapping) else None)):
            if isinstance(d, Mapping):
                put(d.get(name), where)
        lname = name.lower()
        for where, text in (("unit", _get(fp, "software", "backend", "unit")), ("input set", inst.get("tier")), ("input set", inst.get("set")),
                            ("registered set", reg.get("set"))):
            m = re.search(rf"(?:^|[-_/.:]){lname}(\d+)$", text) if isinstance(text, str) else None
            if m:
                put(m.group(1), where)
        if not seen:
            return {}, f"its work model needs {name} (a per-input variable) and the result records none"
        if len(seen) > 1:
            return {}, f"{name} is recorded differently: " + "; ".join(f"{v} ({', '.join(w)})" for v, w in sorted(seen.items()))
        out[name] = next(iter(seen))
    return out, None


def key_class_of(meta: Mapping[str, Any]) -> dict[str, Any] | str | None:
    """A class cell (flock-ir-lowering's class-pin attention statement: one pin per key-count class, the batch spans several T):
    the class range (``key_class.T``), its key counts (``key_counts`` {T: heads}) and per-T timings (``per_key_count`` {T: heads,
    e2e_s, ...}), with each T that is creditable (listed in ``key_counts``, inside the class range, with a positive heads count
    and ``e2e_s`` in ``per_key_count``) and ``P`` at it (heads / e2e_s); a reason string when the shape is broken; ``None`` for a
    result that is not a class cell."""
    fp = meta.get("workload_fingerprint") if isinstance(meta.get("workload_fingerprint"), Mapping) else {}
    counts = fp.get("key_counts")
    if not isinstance(counts, Mapping) or not counts:
        return None
    klass = fp.get("key_class") if isinstance(fp.get("key_class"), Mapping) else {}
    rng = klass.get("T")
    if not (isinstance(rng, (list, tuple)) and len(rng) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in rng) and rng[0] <= rng[1]):
        return f"a class cell (key_counts) without a key_class.T range, got {rng!r}"
    per = meta.get("per_key_count") if isinstance(meta.get("per_key_count"), Mapping) else fp.get("per_key_count")
    per = per if isinstance(per, Mapping) else {}
    credit, skipped = {}, {}
    for k, heads in counts.items():
        t = int(k) if isinstance(k, (int, str)) and str(k).isdigit() else None
        e = per.get(str(k)) if isinstance(per.get(str(k)), Mapping) else per.get(t) if isinstance(per.get(t), Mapping) else None
        if t is None:
            skipped[str(k)] = "not a key count"
        elif not rng[0] <= t <= rng[1]:
            skipped[str(t)] = f"outside the class {rng[0]}–{rng[1]}"
        elif e is None or not _num(e.get("e2e_s")) or e["e2e_s"] <= 0 or not _num(e.get("heads")) or e["heads"] <= 0:
            skipped[str(t)] = "no per_key_count timing"
        else:
            credit[t] = {"heads": int(e["heads"]), "e2e_s": float(e["e2e_s"]), "P": e["heads"] / e["e2e_s"], "listed_heads": heads}
    if not credit:
        return f"a class cell with no creditable key count ({'; '.join(f'T {k}: {v}' for k, v in skipped.items()) or 'none listed'})"
    return {"range": [int(rng[0]), int(rng[1])], "pin": klass.get("pin"), "credit": dict(sorted(credit.items())), "skipped": skipped}


def class_native(target: Target, sub: str, cached: bool, klass: Mapping[str, Any]) -> dict[str, Any]:
    """``N`` of a class cell: per T its native N (:func:`memory_native` at that T) and, over the cell's own inputs, the
    work-weighted N = heads / sum(heads_T / N_T), so the cell's N / P is its work-weighted overhead."""
    per = {t: native_of(target, sub, cached, {"T": t}) for t in klass["credit"]}
    heads = sum(x["heads"] for x in klass["credit"].values())
    n_eff = heads / sum(x["heads"] / per[t]["instances_per_second"] for t, x in klass["credit"].items())
    first = per[min(per)]
    lo, hi = klass["range"]
    return {**first, "instances_per_second": n_eff, "per_key_count": {t: n["instances_per_second"] for t, n in per.items()},
            "basis": (f"work-weighted over the class cell's key counts T = {', '.join(map(str, klass['credit']))} (class {lo}–{hi}): "
                      f"heads / sum(heads_T / N_T), N_T by {first['basis']}")}


def memory_native(target: Target, sub: str, cached: bool = False, variables: Mapping[str, int] | None = None) -> dict[str, Any]:
    """``N`` for a memory-bound subcircuit: instances per second = the hardware line's memory bandwidth / the template's bytes per
    input (``benchmarks/census/subcircuits.json`` ``work_model.bytes_per_input``), with the basis for the footnote.  Operations are not
    binding (``ops_per_instance`` is the template's op count, for reference)."""
    known = census.SUBCIRCUITS[sub]
    wm = census.SUBCIRCUIT_TEMPLATES[known.template].work_model
    env = {**{k: v for k, v in vars(known.parameters).items() if isinstance(v, int)}, **(variables or {})}
    expr = getattr(wm, "bytes_per_input_cached", None) if cached else None
    nbytes, ops = census.evaluate(expr or wm.bytes_per_input, env), census.evaluate(wm.ops_per_input, env)
    hw = contract.native_peak_of(target)
    bw = getattr(hw, "memory_bytes_per_second", None)
    if not _num(bw):
        raise ValueError(f"{hw.id}: no memory bandwidth in the census for {sub}'s memory-bandwidth work model")
    version = WORK_MODEL.split(":")[0]
    return {"instances_per_second": bw / nbytes, "unit": "memory", "ops_per_instance": ops, "bytes_per_instance": nbytes, "steps": None,
            "dtype": (T.dtype_of(target) or "?").upper(), "peak_flop_per_second": hw.flop_per_second, "memory_bytes_per_second": bw,
            "device": hw.device, "source": "; ".join(x.label for x in getattr(hw, "memory_sources", ())) or census.source(hw),
            "work_model": f"{version}: memory-bandwidth bound, {expr or wm.bytes_per_input} bytes per input over the line's memory bandwidth",
            "basis": (f"{version} memory-bandwidth work model: {nbytes} bytes per input ({expr or wm.bytes_per_input}: "
                      f"{wm.bytes_cached_basis if expr else wm.bytes_basis}; "
                      f"{wm.bytes_source}) over {bw / 1e12:.3g} TB/s ({hw.id}); ops {ops} per input ({wm.ops_per_input}) are not binding")}


def subcircuit_k(sub: str) -> int:
    known = census.SUBCIRCUITS.get(sub)
    return int(known.parameters.K) if known else int(re.search(r"/k(\d+)/", sub).group(1))


def input_set_index(store: Any) -> dict[str, tuple[str, Mapping[str, Any]]]:
    """Every registered input set (:data:`INPUT_SET_KINDS`) by its artifact id and by its content digest: ``(art, meta)``."""
    out: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for kind in INPUT_SET_KINDS:
        try:
            rows = store.index.select_artifacts(kind=kind)
        except Exception:  # noqa: BLE001  (a store without a catalog)
            return out
        for row in rows:
            meta = D._doc(store, row).get("meta") or {}
            out[row["id"]] = (row["id"], meta)
            if isinstance(meta.get("content_digest"), str):
                out[meta["content_digest"]] = (row["id"], meta)
    return out


def _set_k(meta: Mapping[str, Any]) -> int | None:
    for v in ((meta.get("parameters") or {}).get("K") if isinstance(meta.get("parameters"), Mapping) else None,
              ((meta.get("relation") or {}).get("statics") or {}).get("K") if isinstance(meta.get("relation"), Mapping) else None):
        if isinstance(v, int):
            return v
    m = re.search(r"/k(\d+)/", str(meta.get("subcircuit") or ""))
    return int(m.group(1)) if m else None


def _captured_set_is(meta: Mapping[str, Any], sub: str | None) -> bool:
    """Whether a captured ``vllm-vu-set/v1`` set (which records its IR ``relation``, not a census id) is of ``sub``: its
    ``relation.definition`` is one of the template's IR Definitions in the census, and its integer statics equal the
    subcircuit's integer parameters.  numpy-free (the steward renders without numpy; ``bench.templates.identify`` needs it)."""
    known = census.SUBCIRCUITS.get(sub or "")
    rel = meta.get("relation") if isinstance(meta.get("relation"), Mapping) else {}
    if known is None or not isinstance(rel.get("definition"), str):
        return False
    names = {n for d in getattr(census.SUBCIRCUIT_TEMPLATES[known.template], "ir_definitions", ()) for n in re.findall(r"[A-Za-z][A-Za-z0-9_]*", d)}
    if rel["definition"] not in names:
        return False
    statics = rel.get("statics") if isinstance(rel.get("statics"), Mapping) else {}
    ints = {k: v for k, v in vars(known.parameters).items() if isinstance(v, int) and not isinstance(v, bool)}
    return all(statics.get(k) == v for k, v in ints.items())


def _input_set_codes(ctx: Ctx) -> list[tuple[str, str]]:
    """Why a result on a statement of its own (another K or semantics) does not run on a registered input set of that K, or
    ``[]``: its inputs name an ``input-set/v1`` / ``vllm-vu-set/v1`` artifact (the ``bench.cell`` stamp's art, else the set's
    content digest), of the statement's K, over a range inside the set."""
    fp = ctx.meta["workload_fingerprint"]
    inst = fp.get("instances")
    if not isinstance(inst, Mapping):
        return [("I", "instances is null: no input set named")]
    stamp = ((ctx.meta.get("cell") or {}).get("input_set") or {}) if isinstance(ctx.meta.get("cell"), Mapping) else {}
    keys = [stamp.get("art"), inst.get("set_art"), inst.get("content_digest"), stamp.get("content_digest"), inst.get("manifest_sha256")]
    hit = next((ctx.input_sets[k] for k in keys if isinstance(k, str) and k in ctx.input_sets), None)
    if hit is None:
        return [("I", f"inputs name no registered input set ({' | '.join(INPUT_SET_KINDS)}; "
                      f"digest {inst.get('content_digest') or inst.get('manifest_sha256')!r})")]
    art, meta = hit
    if not is_gemm(ctx.subcircuit):
        if meta.get("subcircuit") != ctx.subcircuit and not _captured_set_is(meta, ctx.subcircuit):
            return [("I", f"input set {art} is of {meta.get('subcircuit')!r}, the statement's subcircuit is {ctx.subcircuit}")]
    else:
        K, sk = subcircuit_k(ctx.subcircuit), _set_k(meta)
        if sk != K:
            return [("I", f"input set {art} is of K = {sk}, the statement {ctx.subcircuit} is of K = {K}")]
    rng, n = inst.get("range"), meta.get("n")
    if isinstance(rng, (list, tuple)) and len(rng) == 2 and isinstance(n, int) and not (0 <= rng[0] < rng[1] <= n):
        return [("I", f"inputs {list(rng)} leave input set {art} (n = {n})")]
    return []


def _instances(ctx: Ctx, rules: Rules, equivs: Sequence[T.InstanceEquiv], self_: set[str]) -> list[tuple[str, str]]:
    fp = ctx.meta["workload_fingerprint"]
    want, inst = T.frozen_instances(ctx.target), fp.get("instances")
    if want is None:
        return [("I", f"target {ctx.target.name!r} has no frozen input set yet")]
    if not isinstance(inst, Mapping):
        return [("I", "instances is null: not measured on the frozen input set")]
    diff = T.instances_diff(inst, want)
    if not diff:
        return []
    if not rules.legacy:
        how, repeats = instance_range(ctx.target, inst)
        if how is not None:
            return []
        if repeats:
            return [("I", repeats)]
    eq, why = (T.instance_equivalence if rules.legacy else stream_equivalence)(inst, ctx.target, equivs, self_)
    if eq is not None:
        return []
    return [("I", "instances differ from the frozen set on " + ", ".join(f"{k} ({inst.get(k)!r} vs {want[k]!r})" for k in diff)
             + (f" (instance-equiv: {why})" if why else ""))]


def reject_codes(store: Any, ctx: Ctx, rules: Rules, equivs: Sequence[T.InstanceEquiv]) -> list[tuple[str, str]]:
    """Every reason the result is not an admissible Table 2 cell under ``rules``, as ``(letter, reason)`` sorted in the spec's
    order (:data:`REASON_LETTERS`); ``[]`` means admissible."""

    p: list[tuple[str, str]] = []
    meta, fp = ctx.meta, ctx.meta["workload_fingerprint"]
    labs = T._labels(ctx.labels)
    self_ = T.producers(ctx.attempt, labs, meta)
    pc = fp.get("proof_class")
    # R / S: the statement (legacy: today's authentication columns)
    if rules.legacy:
        auth = T.column_authentication(fp)
        if fp.get("authentication") != auth:
            p.append(("S" if fp.get("authentication") == T.SHARED_AUTHENTICATION else "X",
                      f"authentication {fp.get('authentication')!r}: legacy columns are {' | '.join(T.AUTH_MODES)}"))
        elif auth == "included-hash":
            p += T.in_proof_hash_codes(fp)
    elif ctx.config is None:
        p.append(("X", "no declared configuration recognises this result (drilldown.VARIANTS)"))
    else:
        if not ctx.config.full_relation:
            p.append(("R", f"{ctx.config.id}: {ctx.config.statement}"))
        elif ctx.config.scheme_problem():
            p.append(("S", ctx.config.scheme_problem()))
        if ctx.config.archived and not rules.include_archived:
            p.append(("X", f"{ctx.config.id} is archived (hidden by default)"))
    # Z: the family's declared class (diagnostic modes never enter)
    if pc not in ctx.family.declared_classes:
        p.append(("Z", f"proof_class {pc!r} is not {ctx.family.id}'s declared class {' | '.join(ctx.family.declared_classes)}"))
    p += _security(fp, rules, ctx.config)
    if ctx.attempt is None and not self_:
        p.append(("U", "not independently verified: no attempt and no named producer"))
    elif T._verified_by(labs, self_) is None:
        p.append(("U", f"not independently verified: no accepting verification label by a non-producer (producers: {', '.join(sorted(self_)) or 'unknown'})"))
    if not rules.legacy:
        why = _custody(store, ctx)
        if why:
            p.append(("F", why))
    p += _input_set_codes(ctx) if ctx.subcircuit and not rules.legacy else _instances(ctx, rules, equivs, self_)
    gpu = _get(fp, "hardware", "gpu") if isinstance(_get(fp, "hardware", "gpu"), Mapping) else None
    if not T.same_sku(gpu.get("name") if gpu else None, contract.native_peak_of(ctx.target).device):
        p.append(("K", f"same-SKU: prover ran on {(gpu or {}).get('name')!r}, the line's device is {contract.native_peak_of(ctx.target).device!r}"))
    variant = T.device_variant(gpu, ctx.target)
    if variant:
        p.append(("K", variant))
    problems = contract.validate(meta)
    if problems:
        p.append(("P" if any(T.PHASE_SUM_PROBLEM.match(x) for x in problems) else "X", T._contract_reason(problems)))
    if rules.legacy:
        if fp.get("K") != T.K:
            p.append(("B", f"K is {fp.get('K')!r}, legacy K = {T.K}"))
        if fp.get("B") != T.B:
            p.append(("B", f"B is {fp.get('B')!r}, legacy B = {T.B}"))
        if "B_real" in fp and fp.get("B_real") != fp.get("B"):
            p.append(("B", f"B_real {fp.get('B_real')!r} != B {fp.get('B')!r}"))
    else:
        why = _protocol(meta, ctx.config)
        if why:
            p.append(("M", why))
        why = interaction_problem(interaction(meta), rules.interaction_one_sided) if rules.interaction_validated else None
        if why:
            p.append(("M", why))
    ceiling, verdict = T._redteam_ceiling(T._labels(ctx.labels), self_)        # a verdict caps only the result it names
    if not rules.legacy and not is_gemm(ctx.subcircuit) and verdict is None and statement_grant(ctx, self_, pc) is None:
        p.append(("U", f"not red-team cleared: a statement on a new template ({ctx.subcircuit}) needs a red-team proof_class verdict "
                       "on its results before it is a cell"))
    if verdict is not None and isinstance(pc, str) and T.RANK.get(ceiling or "", -1) < T.RANK.get(pc, -1):
        p.append(("D", f"capped: {verdict.by} ({verdict.ref or 'no ref'}, {verdict.ts}) asserts proof_class={ceiling}, below {pc}"))
    state, flab = T.cell_finding(ctx.labels)
    if state == "pulled":
        p.append(("D", f"pulled: {T.finding_note(flab)}"))
    return sorted(p, key=lambda x: _ORDER[x[0]])


def statement_grant(ctx: Ctx, self_: set[str], pc: Any) -> T._Label | None:
    """The red-team verdict that clears the result's statement at its class ``pc``: the latest verdict on some result of the same
    configuration, target and subcircuit that grants at least ``pc``, or ``None``.  Each result's own latest verdict counts,
    so a peer's downgrade (a placement finding on that one run) neither clears nor caps the others."""
    need = T.RANK.get(pc, -1) if isinstance(pc, str) else -1
    grants = [v for v in (T._redteam_ceiling(T._labels(labels), self_)[1] for labels in ctx.statement_labels)
              if v is not None and T.RANK.get(str(v.value), -1) >= need]
    return max(grants, key=lambda v: v.ts) if grants else None


def provisional(ctx: Ctx) -> str | None:
    """Why the cell is provisional, or ``None``: neither the result's own red-team ``proof_class`` verdict nor one on another
    result of the configuration's statement for this subcircuit (same configuration and target) grants the result's class."""
    self_ = T.producers(ctx.attempt, T._labels(ctx.labels), ctx.meta)
    pc = (ctx.meta.get("workload_fingerprint") or {}).get("proof_class")
    ceiling, verdict = T._redteam_ceiling(T._labels(ctx.labels), self_)
    if verdict is None and statement_grant(ctx, self_, pc) is None:
        return f"no red-team clearance of {ctx.config.id if ctx.config else ctx.family.id}'s statement on {ctx.subcircuit or ctx.target.name}"
    return None


# --- build ------------------------------------------------------------------------------------------------------------------------


@dataclass
class ViewCell:
    line: str
    column: str
    artifact: str
    configuration: str | None
    native: float
    throughput: dict[str, Any]
    overhead: float
    bound_log2: float | None
    provisional: str | None
    legacy: T.Cell
    batching: dict[str, Any]
    scheme: str | None = None
    unfiltered: dict[str, Any] | None = None
    algebraic: list[str] = field(default_factory=list)   #: the algebraic hashes of its soundness argument (Table 1's flag)
    instances: dict[str, Any] = field(default_factory=dict)  #: the result's instance ref
    setup_s: float | None = None                          #: one-time setup (``fixed.setup_seconds``), outside the cell
    subcircuit: str | None = None                         #: its statement's subcircuit when not its target's own (a real-K row)
    variables: dict[str, int] = field(default_factory=dict)  #: the per-input variables its statement fixes (attention's T)

    def to_json(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["legacy"] = self.legacy.to_json()
        if d["subcircuit"] is None:
            del d["subcircuit"]
        if not d["variables"]:
            del d["variables"]
        return d


@dataclass
class Views:
    rules: Rules
    lines: list[dict[str, Any]] = field(default_factory=list)
    cells: list[ViewCell] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    below_bar: list[dict[str, Any]] = field(default_factory=list)  #: verified results labelled below_bar=true (:func:`below_bar_entry`)
    not_families: dict[str, list[str]] = field(default_factory=dict)
    configurations: tuple[Configuration, ...] = CONFIGURATIONS
    unreadable: list[str] = field(default_factory=list)   #: artifacts the views could not read here (payload not local, ...)
    admissible: list[ViewCell] = field(default_factory=list)  #: every admissible result as a cell (``cells`` holds each family's best)
    security_of: dict[str, dict[str, Any]] = field(default_factory=dict)  #: each admissible result's ``security`` block, by artifact
    key_class_of: dict[str, dict[str, Any]] = field(default_factory=dict)  #: each admissible class cell's key counts (:func:`key_class_of`)
    omitted_of: dict[str, Any] = field(default_factory=dict)  #: each admissible result's latest ``omitted`` label (what it leaves out)
    domain_of: dict[str, str] = field(default_factory=dict)  #: each admissible result's operand domain (``domain`` label, else its fingerprint)
    circuit_of: dict[str, dict[str, Any]] = field(default_factory=dict)  #: each admissible circuit-statement result's :func:`circuit_facts`
    prover_gpu: dict[str, str | None] = field(default_factory=dict)  #: each admissible result's ``hardware.gpu.name``, by artifact
    interaction_raw: dict[str, dict[str, Any]] = field(default_factory=dict)  #: each admissible result's raw round and loopback records
    #: the attempts labelled ``interaction_attempt_of`` a cell, by that cell's artifact (IX1): ``{artifact, attempt, deviation}``
    interaction_attempts: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    input_sets: dict[str, tuple[str, Mapping[str, Any]]] = field(default_factory=dict)  #: registered input sets (:func:`input_set_index`)

    def columns(self) -> list[str]:
        if not self.rules.legacy:
            return [f.id for f in FAMILIES]
        hashed = [f.id + T.AUTH_COLUMN_SUFFIX["included-hash"] for f in FAMILIES
                  if any(ln["cells"].get(f.id + T.AUTH_COLUMN_SUFFIX["included-hash"]) for ln in self.lines)]
        return [f.id for f in FAMILIES] + hashed


def _batching(fp: Mapping[str, Any]) -> dict[str, Any]:
    return {"total": fp.get("B"), "per_proof": fp.get("B_proof"), "proofs": fp.get("N_subbatches"),
            "in_flight": _get(fp, "software", "backend", "pipeline"), "columns_per_proof": _get(fp, "security", "rs_l")}


def _results(store: Any, rules: Rules) -> list[tuple[str, Mapping[str, Any]]]:
    """Every ``bench-result/v1`` as ``(art, envelope)``.  The spec rules also read a result registered with a lane summary as its
    meta and the result envelope as its JSON payload (the drill-down's ``_envelope``; route (a)'s cell art:4b52879f is one); the
    legacy rules read the meta only, as the frozen tables do."""
    out = []
    for row in store.index.select_artifacts(kind=T.RESULT_KIND):
        doc = D._doc(store, row)
        meta = doc.get("meta") or {}
        if not rules.legacy and not isinstance(meta.get("workload_fingerprint"), Mapping):
            env, _, payload = D._envelope(store, row["id"], doc)
            if env is not None and payload:
                meta = env
        out.append((row["id"], meta))
    return out


#: The label that sends a verified result below the tables' 2^-128 bar to its own drill-down (never a Table 2 cell, never dropped).
BELOW_BAR = "below_bar"
#: The label on an interaction attempt that is not its cell's figure: the cell's art id; the cell lists it with its deviation (IX1).
INTERACTION_ATTEMPT_OF = "interaction_attempt_of"


def _latest(labels: Sequence[Any], key: str) -> Any:
    found = [lab for lab in T._labels(labels) if lab.key == key]
    return found[-1].value if found else None


def below_bar_entry(art: str, meta: Mapping[str, Any], labels: Sequence[Any], cand: Any) -> dict[str, Any]:
    """A below-bar result as the drill-down shows it: what it proves (its IR Definition and gate family), on which input set and
    GPU, its own soundness per proof mode (core, compressed), its proving seconds per instance, proof bytes, and who verified it."""
    fp = meta.get("workload_fingerprint") if isinstance(meta.get("workload_fingerprint"), Mapping) else {}
    m = T._measurements(meta)
    num = lambda k: float(m[k]) if _num(m.get(k)) else None  # noqa: E731
    verified = [lab for lab in T._labels(labels) if lab.key == "verified"]
    inst = fp.get("instances") if isinstance(fp.get("instances"), Mapping) else {}
    return {"artifact": art, "backend_family": _fam(cand.id), "definition": _get(fp, "definition", "id"), "gate_family": fp.get("family"),
            "B": fp.get("B"), "input_set": inst.get("set"), "input_set_art": inst.get("set_art"), "served_row": inst.get("row"), "prover_gpu": _get(fp, "hardware", "gpu", "name"),
            "proof_class": fp.get("proof_class"),
            "modes": {mode: {"soundness_bits": num(f"{mode}.soundness_bits"), "prove_seconds": num(f"{mode}.prove_seconds"),
                             "seconds_per_instance": num(f"{mode}.seconds_per_instance"), "proof_bytes": num(f"{mode}.proof_bytes"),
                             "verify_seconds": num(f"{mode}.host_verify_seconds")}
                      for mode in (fp.get("modes") or ["core"])},
            "bar_bits": 128, "verified": verified[-1].value if verified else None, "verified_by": verified[-1].by if verified else None}


def build(store: Any, rules: Rules = SPEC, configurations: Sequence[Configuration] = CONFIGURATIONS) -> Views:
    """Read the store, apply :func:`reject_codes` under ``rules`` and assemble Tables 1-3 (rendering is separate)."""

    V = Views(rules=rules, configurations=tuple(configurations))
    cfgs = {c.id: c for c in configurations}
    equivs = T.instance_equivs(store)
    V.unreadable = [f"{e.artifact}: {e.unreadable}" for e in equivs if e.unreadable]
    entries: list[Ctx] = []
    for art, meta in _results(store, rules):
        fp = meta.get("workload_fingerprint") if isinstance(meta.get("workload_fingerprint"), Mapping) else {}
        backend = _get(fp, "software", "backend", "name")
        cand = candidate_for(backend)
        if cand is None:
            V.not_families.setdefault(backend if isinstance(backend, str) else "(no software.backend.name)", []).append(art)
            continue
        rid, attempt = T._attempt_of(store, art)
        labels = list(store.labels(art))
        of = _latest(labels, INTERACTION_ATTEMPT_OF)
        if not rules.legacy and of:
            cell = str(of) if str(of).startswith("art:") else f"art:{of}"
            V.interaction_attempts.setdefault(cell, []).append(
                {"artifact": art, "attempt": rid, "deviation": interaction_deviation(interaction(meta))})
            V.rejected.append({"artifact": art, "family": cand.id, "configuration": None, "line": None, "attempt": rid,
                               "codes": [("D", f"an interaction attempt of {cell}, listed with that cell (IX1)")]})
            continue
        if not rules.legacy and str(_latest(labels, BELOW_BAR)).lower() == "true":     # written as a bool or, off-vocabulary, "true"
            V.below_bar.append(below_bar_entry(art, meta, labels, cand))
            continue
        named = stamped_subcircuit(meta)
        if not rules.legacy and named and not is_gemm(named) and named not in census.SUBCIRCUITS:
            V.rejected.append({"artifact": art, "family": cand.id, "configuration": None, "line": None, "attempt": rid,
                               "codes": [("X", f"subcircuit {named!r} is not a census subcircuit (a bound template id such as "
                                               f"rope-head/d64/neox-bf16)")]})
            continue
        target = TARGETS.get(fp.get("profile")) if isinstance(fp.get("profile"), str) else None
        if target is None and not rules.legacy and not is_gemm(stamped_subcircuit(meta)):
            target = hardware_target(meta, stamped_subcircuit(meta))
        if target is not None and not rules.legacy:
            target = rehome(meta, target)
        config = configuration_of(meta, cfgs)
        if target is None:
            V.rejected.append({"artifact": art, "family": cand.id, "configuration": config.id if config else None, "line": None,
                               "attempt": rid, "codes": [("X", f"profile {fp.get('profile')!r} is not a target of verity.ml.circuits.targets.target")]})
            continue
        entries.append(Ctx(art, meta, labels, attempt, target, _FAMILY[cand.id], config, []))
    if not rules.legacy:
        sets = V.input_sets = input_set_index(store)
        for c in entries:
            c.subcircuit, c.input_sets = result_subcircuit(c.meta, c.target), sets
        unmodelled = []
        for c in entries:
            klass = key_class_of(c.meta) if not is_gemm(c.subcircuit) else None
            if isinstance(klass, dict):
                c.key_class, c.variables, why = klass, {}, None
            elif isinstance(klass, str):
                why = klass
            else:
                c.variables, why = input_variables(c.meta, c.subcircuit, sets)
            try:
                if why is None and c.key_class:
                    class_native(c.target, c.subcircuit, rules.native_cached, c.key_class)
                elif why is None:
                    native_of(c.target, c.subcircuit, rules.native_cached, c.variables)
            except KeyError as e:
                why = f"its work model needs {e.args[0]}, which varies per input, and the result records none"
            if why is not None:          # a per-input work-model variable (attention's key count T) the result does not fix
                rid, _ = T._attempt_of(store, c.art)
                V.rejected.append({"artifact": c.art, "family": c.family.id, "configuration": c.config.id if c.config else None,
                                   "line": c.target.name, "subcircuit": c.subcircuit, "attempt": rid,
                                   "codes": [("X", f"native N is not modelled for {c.subcircuit}: {why}")]})
                unmodelled.append(c)
        entries = [c for c in entries if c not in unmodelled]
    by_statement: dict[tuple[str | None, str, str | None], list[list[Any]]] = {}
    for c in entries:
        by_statement.setdefault((c.config.id if c.config else None, c.target.name, c.subcircuit), []).append(list(c.labels))
    for c in entries:
        c.statement_labels = by_statement[(c.config.id if c.config else None, c.target.name, c.subcircuit)] if c.config else [list(c.labels)]

    line_schemes: list[str | None] = [None] if rules.legacy else [
        s.id for s in SCHEMES if s.admissible and any(c.scheme == s.id for c in configurations)]

    def scheme_of(c: Ctx) -> str | None:
        return c.config.scheme if not rules.legacy and c.config is not None and c.config.scheme in line_schemes else None

    Key = tuple[Any, ...]

    def vkey(c: Ctx) -> tuple[tuple[str, Any], ...]:
        if c.key_class:                 # a class cell: a row per key-count class
            return (("T", f"{c.key_class['range'][0]}–{c.key_class['range'][1]}"),)
        return tuple(sorted(c.variables.items()))

    def n_of(c: Ctx) -> dict[str, Any]:
        if c.key_class:
            return class_native(c.target, c.subcircuit, rules.native_cached, c.key_class)
        return native_of(c.target, c.subcircuit, rules.native_cached, c.variables)

    pools: dict[Key, list[tuple[Ctx, dict[str, Any]]]] = {}
    unfiltered: dict[Key, list[tuple[Ctx, dict[str, Any]]]] = {}
    for c in entries:
        if c.config is not None and c.config.id in TYPED_CONFIGS | {FLOCK_CIRCUIT_HIDDEN_CONFIG.id}:
            # never the best of a typed and a flat cell, or of a hidden-output and a public-output one (different statements are
            # never merged): out of Table 2 until Table 2 gives them lines of their own (a typed one also until C2 holds, whatever
            # TYPED_C2 says)
            rid, _ = T._attempt_of(store, c.art)
            why = TYPED_NOT_IN_TABLE2 if c.config.id in TYPED_CONFIGS else HIDDEN_NOT_IN_TABLE2
            V.rejected.append({"artifact": c.art, "family": c.family.id, "configuration": c.config.id, "line": c.target.name,
                               "subcircuit": c.subcircuit, "attempt": rid, "codes": [("X", why)]})
            continue
        codes = reject_codes(store, c, rules, equivs)
        P = throughput(c.meta, rules, c.config is not None and not c.config.transferability.startswith("transferable"))
        key = (c.target.name, c.subcircuit, scheme_of(c), _column(c, rules), vkey(c))
        if not codes and P is not None:
            pools.setdefault(key, []).append((c, P))
            continue
        if (rules.security_log2 is not None or rules.standard_hashes) and all(k == "L" for k, _ in codes) and P is not None:
            unfiltered.setdefault(key, []).append((c, P))
        rid, _ = T._attempt_of(store, c.art)
        entry = {"artifact": c.art, "family": c.family.id, "configuration": c.config.id if c.config else None, "line": c.target.name,
                 "scheme": key[2], "column": key[3], "attempt": rid, "codes": codes or [("X", "no positive t.total and batch")]}
        if c.subcircuit:
            entry["subcircuit"] = c.subcircuit
        if c.variables:
            entry["variables"] = dict(c.variables)
        if P is not None and not rules.legacy and c.config is not None and c.config.full_relation:
            commit = P["commit_s"] or P["commit_v1_s"] or 0.0
            n = n_of(c)["instances_per_second"]
            entry["end_to_end_preview"] = {"commit_s": commit, "commit_warm": bool(P["commit_s"]),
                                           "overhead": n * (P["proving_s"] + commit) / P["instances"]}
        V.rejected.append(entry)

    def best(pool: list[tuple[Ctx, dict[str, Any]]]) -> tuple[Ctx, dict[str, Any]] | None:
        return min(pool, key=lambda cp: (-cp[1]["instances_per_second"], cp[0].art)) if pool else None

    def pick(c: Ctx, P: Mapping[str, Any], N: float) -> dict[str, Any]:
        return {"artifact": c.art, "configuration": c.config.id if c.config else None, "bound_log2": whole_proof_log2(c.meta["workload_fingerprint"]),
                "hashes": [h.text() for h in (c.config.hashes if c.config else ())], "overhead": N / P["instances_per_second"]}

    def make_cell(c: Ctx, P: dict[str, Any], target: Target, scheme: str | None, col: str, N: Mapping[str, Any],
                  sub: str | None = None, variables: tuple[tuple[str, int], ...] = ()) -> ViewCell:
        vb = T.independent_verification(c.meta, c.labels, c.attempt)
        V.security_of[c.art] = dict(c.meta["workload_fingerprint"].get("security") or {})
        if _latest(c.labels, "omitted"):
            V.omitted_of[c.art] = _latest(c.labels, "omitted")
        fp_dom = contract.identity(c.meta["workload_fingerprint"])["domain"]
        dom = _latest(c.labels, "domain") or (fp_dom if fp_dom in contract.DOMAINS else None)
        if dom:
            V.domain_of[c.art] = str(dom)
        if (facts := circuit_facts(c.meta)) is not None:
            V.circuit_of[c.art] = facts
        if c.key_class:
            V.key_class_of[c.art] = c.key_class
            N = n_of(c)                 # its own key counts' work-weighted N (the row's is its first cell's)
        m = T._measurements(c.meta)
        block = c.meta.get("interaction") if isinstance(c.meta.get("interaction"), Mapping) else c.meta["workload_fingerprint"].get("interaction")
        V.interaction_raw[c.art] = {"rounds_protocol": m.get("rounds.protocol"), "sequential_depth": m.get("rounds.sequential_depth"),
                                    "loopback_round_s": m.get("live.loopback_round_seconds"),
                                    "loopback_method": block.get("loopback_method") if isinstance(block, Mapping) else None}
        gpu = _get(c.meta["workload_fingerprint"], "hardware", "gpu")
        V.prover_gpu[c.art] = gpu.get("name") if isinstance(gpu, Mapping) else None
        return ViewCell(line=target.name, column=col, artifact=c.art, configuration=c.config.id if c.config else None,
                        native=N["instances_per_second"], throughput=P, overhead=N["instances_per_second"] / P["instances_per_second"],
                        bound_log2=whole_proof_log2(c.meta["workload_fingerprint"]),
                        provisional=None if rules.legacy else provisional(c),
                        legacy=T._cell(c.meta, c.art, T._attempt_of(store, c.art)[0], target, _CANDIDATE[c.family.id],
                                       vb.by if vb else None, _equiv_of(c, rules, equivs),
                                       _open_finding(c.labels)),
                        batching=_batching(c.meta["workload_fingerprint"]), scheme=scheme,
                        algebraic=[h.text() for h in (c.config.algebraic() if c.config else ())],
                        instances=dict(c.meta["workload_fingerprint"].get("instances") or {}),
                        setup_s=T._measurements(c.meta).get(SETUP) if _num(T._measurements(c.meta).get(SETUP)) else None,
                        subcircuit=sub, variables=dict(variables))

    columns = [f.id for f in FAMILIES] + ([f.id + T.AUTH_COLUMN_SUFFIX["included-hash"] for f in FAMILIES] if rules.legacy else [])
    def add_line(target: Target, sub: str | None, scheme: str | None, N: Mapping[str, Any], variables: tuple[tuple[str, int], ...] = ()) -> None:
        line: dict[str, Any] = {"line": target.name, "scheme": scheme, "scheme_label": _SCHEME[scheme].label if scheme else None,
                                "status": target.status, "device": contract.native_peak_of(target).device, "native": N,
                                "instruction": target.instruction, "cells": {}, "empty": {}}
        if sub:
            known = census.SUBCIRCUITS.get(sub)
            line.update(subcircuit=sub, subcircuit_display=(known.display if known else sub) + "".join(f" · {k} = {v}" for k, v in variables))
        if variables:
            line["variables"] = dict(variables)
        for col in columns:
            key = (target.name, sub, scheme, col, variables)
            hit = best(pools.get(key, []))
            if hit is None:
                line["cells"][col] = None
                line["empty"][col] = _empty_reason(V.rejected, target, scheme, line_schemes, col, configurations, sub)
                if unfiltered.get(key):
                    u = pick(*best(unfiltered[key]), N["instances_per_second"])
                    line["empty"][col] = {"state": "below-filter", "unfiltered": u, "text": (
                        f"only below the filter: unfiltered pick {u['configuration']} {u['artifact']}, bound {_fmt_bound(u['bound_log2'])}, "
                        f"hashes {'; '.join(u['hashes'])}, {T.fmt_overhead(u['overhead'])}")}
                continue
            c, P = hit
            cell = make_cell(c, P, target, scheme, col, N, sub, variables)
            V.admissible += [cell if o.art == c.art else make_cell(o, oP, target, scheme, col, N, sub, variables) for o, oP in pools[key]]
            alt = best(pools.get(key, []) + unfiltered.get(key, []))
            if alt is not None and alt[0].art != c.art:
                cell.unfiltered = pick(*alt, N["instances_per_second"])
            line["cells"][col] = cell
            V.cells.append(cell)
        V.lines.append(line)

    for target in TARGETS.values():
        N = native(target)
        for scheme in line_schemes:
            add_line(target, None, scheme, N)
    # statements of their own (another K, or semantics the target's subcircuit does not stand for): a row per scheme with a result
    own: dict[tuple[str, str, tuple[tuple[str, Any], ...]], set[str | None]] = {}
    first_class: dict[tuple[str, str, tuple[tuple[str, Any], ...]], Ctx] = {}
    for c in entries:
        if c.subcircuit:
            own.setdefault((c.target.name, c.subcircuit, vkey(c)), set()).add(scheme_of(c))
            if c.key_class:
                first_class.setdefault((c.target.name, c.subcircuit, vkey(c)), c)
    for target in LINES.values():
        for sub, vs in sorted(((x, v) for t, x, v in own if t == target.name),
                              key=lambda xv: (not is_gemm(xv[0]), subcircuit_k(xv[0]) if is_gemm(xv[0]) else 0, xv[0],
                                              tuple((k, (1, int(v.split("–")[0])) if isinstance(v, str) else (0, v)) for k, v in xv[1]))):
            fc = first_class.get((target.name, sub, vs))
            N = class_native(target, sub, rules.native_cached, fc.key_class) if fc else native_of(target, sub, rules.native_cached, dict(vs))
            for scheme in line_schemes:
                if scheme in own[(target.name, sub, vs)]:
                    add_line(target, sub, scheme, N, vs)
    return V


def _open_finding(labels: Sequence[Any]) -> str | None:
    state, lab = T.cell_finding(labels)
    return T.finding_note(lab) if state == "open" else None


def _empty_reason(rejected: Sequence[Mapping[str, Any]], target: Target, scheme: str | None, line_schemes: Sequence[str | None],
                  col: str, configurations: Sequence[Configuration], sub: str | None = None) -> dict[str, Any]:
    """Why a cell is empty: ``n/s`` when every configuration of the family declares the subcircuit unsupported, else what its
    results lack (weaker statements or undefined schemes only, or the first reasons that apply).  A result on no line's
    scheme counts on every scheme line of its subcircuit and hardware class."""
    fam = [c for c in configurations if c.family == col]
    if fam and all(target.name in c.unsupported for c in fam):
        return {"state": "n/s", "text": "; ".join(sorted({c.unsupported[target.name] for c in fam}))}
    mine = [r for r in rejected if r.get("line") == target.name and r.get("column") == col and r.get("subcircuit") == sub
            and (r.get("scheme") == scheme or r.get("scheme") not in line_schemes)]
    if not mine:
        return {"state": "none", "text": "no result"}
    if all(r["codes"][0][0] in ("R", "S") for r in mine):
        weak = sorted({str(r["configuration"]) for r in mine if r["codes"][0][0] == "R"})
        bad = sorted({str(r["configuration"]) for r in mine if r["codes"][0][0] == "S"})
        parts = ([f"a weaker statement ({', '.join(weak)})"] if weak else []) + ([f"a scheme verity.ml.commitments does not define ({', '.join(bad)})"] if bad else [])
        return {"state": "weaker", "text": "results only for " + " or ".join(parts)}
    firsts: dict[str, int] = {}
    for r in mine:
        firsts[r["codes"][0][0]] = firsts.get(r["codes"][0][0], 0) + 1
    return {"state": "rejected", "text": "no admissible result; first reasons " + ", ".join(f"{k}×{n}" for k, n in sorted(firsts.items(), key=lambda kv: _ORDER[kv[0]]))}


# --- parity (the switch's gate 2) -------------------------------------------------------------------------------------------------


def parity(store: Any, frozen: T.Tables | None = None, members: Sequence[str] | None = None) -> list[str]:
    """Differences between the legacy-rules views and today's canonical tables over the same store: every cell's artifact and
    ratio (and with ``members``, a ``tables --snapshot``'s candidate-cell members).  ``[]`` = parity."""

    frozen = frozen or T.build(store)
    V = build(store, LEGACY)
    want = {(c.target, c.candidate + T.AUTH_COLUMN_SUFFIX.get(c.authentication, "")): c for c in frozen.cells}
    got = {(c.line, c.column): c for c in V.cells}
    out = []
    for key in sorted(set(want) | set(got)):
        w, g = want.get(key), got.get(key)
        if w is None or g is None:
            out.append(f"{key[0]} · {key[1]}: tables {w.artifact if w else '—'}, views {g.artifact if g else '—'}")
        elif w.artifact != g.artifact:
            out.append(f"{key[0]} · {key[1]}: tables {w.artifact}, views {g.artifact}")
        elif not math.isclose(w.overhead, g.overhead, rel_tol=1e-9) or T.fmt_overhead(w.overhead) != T.fmt_overhead(g.overhead):
            out.append(f"{key[0]} · {key[1]}: {w.artifact} ratio tables {w.overhead!r}, views {g.overhead!r}")
    if members is not None:
        mine = {c.artifact for c in V.cells}
        snap = set(members) - {r["native_measured"]["artifact"] for r in frozen.rows if r.get("native_measured")}
        for a in sorted(snap - mine):
            out.append(f"snapshot member {a} is no legacy-rules cell")
        for a in sorted(mine - snap):
            out.append(f"legacy-rules cell {a} is not in the snapshot")
    return out


PARITY_KIND = "tables-parity/v1"


def parity_record(store: Any, snapshot: tuple[str, str, Sequence[str]] | None, frozen: T.Tables | None = None) -> dict[str, Any]:
    """The recorded parity render (TABLES.md "The switch", condition 2): :func:`parity` against ``snapshot`` = (name, art id,
    members), with every legacy-rules cell's artifact and ratio beside the frozen one."""
    frozen = frozen or T.build(store)
    diffs = parity(store, frozen, members=snapshot[2] if snapshot else None)
    old = {(c.target, c.candidate + T.AUTH_COLUMN_SUFFIX.get(c.authentication, "")): c for c in frozen.cells}
    cells = [{"line": c.line, "column": c.column, "artifact": c.artifact, "views_overhead": c.overhead,
              "tables_overhead": old[(c.line, c.column)].overhead if (c.line, c.column) in old else None, "printed": T.fmt_overhead(c.overhead)}
             for c in build(store, LEGACY).cells]
    return {"schema": "verity/tables-parity/v1", "rules": LEGACY.name, "ok": not diffs, "differences": diffs,
            "snapshot": snapshot[0] if snapshot else None, "snapshot_artifact": snapshot[1] if snapshot else None,
            "tables_cells": len(frozen.cells), "cells": cells}


# --- rendering --------------------------------------------------------------------------------------------------------------------


def fmt_rate(x: float) -> str:
    m, e = f"{x:.1e}".split("e")
    return f"{m}e{int(e)}/s"


def fmt_per_instance(s: float) -> str:
    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("µs", 1e-6), ("ns", 1e-9)):
        if abs(s) >= scale:
            return f"{s / scale:.3g} {unit}"
    return f"{s / 1e-12:.3g} ps"


def _fmt_bound(b: float | None) -> str:
    return f"2^{b:.1f}" if b is not None else "unknown"


def _instances_text(target_name: str, inst: Mapping[str, Any], sub: str | None = None) -> str:
    rng = inst.get("range") or ["?", "?"]
    if sub:
        return f"inputs [{rng[0]}, {rng[1]}) of input set {inst.get('dataset') or inst.get('set') or inst.get('tier', '?')} ({inst.get('source') or 'source not recorded'})"
    how, _ = instance_range(LINES[target_name], inst) if target_name in LINES and inst else (None, None)
    kind = {"stream": "a prefix-consistent extension of the frozen set", "prefix": "a prefix of the frozen set",
            "frozen": "the frozen set"}.get(how or "", "re-packed")
    return f"instances [{rng[0]}, {rng[1]}) of {inst.get('tier', '?')} ({kind})"


def _native_text(N: Mapping[str, Any]) -> str:
    if N.get("memory_bytes_per_second") and N["unit"] == "memory":
        return f"{fmt_rate(N['instances_per_second'])} · memory, {N['memory_bytes_per_second'] / 1e12:.3g} TB/s"
    return f"{fmt_rate(N['instances_per_second'])} · {N['unit']}, {N['dtype']} {T._fmt_tflops(N['peak_flop_per_second'])}"


def table1_rows(V: Views) -> list[tuple[str, list[str]]]:
    held: dict[str, list[tuple[str, float | None]]] = {}
    for c in V.cells:
        if c.configuration:
            held.setdefault(c.configuration, []).append((_cell_name(c), c.bound_log2))
    out: list[tuple[str, list[str]]] = []
    for fam in [*FAMILIES, None]:
        fid = fam.id if fam else None
        for cfg in (c for c in V.configurations if c.family == fid or (fam is None and c.family not in _FAMILY)):
            cells = held.get(cfg.id, [])
            bounds = "; ".join(f"{_fmt_bound(b)} at {where}" for where, b in cells)
            sch = _SCHEME.get(cfg.scheme) if cfg.scheme else None
            scheme = (f"{sch.label} ({sch.id}; {'admissible' if sch.admissible else 'inadmissible'}): " if sch else "none: ") + cfg.commitment
            soundness = (bounds + " — " if bounds else "") + cfg.soundness + "; dominant term: " + cfg.dominant_term
            flag = f"**{ALGEBRAIC_FLAG}** ({', '.join(sorted({h.name for h in cfg.algebraic()}))})" if cfg.algebraic() else ""
            out.append((fid or "(no Table 2 family)", [cfg.id, flag, cfg.statement, cfg.ports, scheme, "; ".join(h.text() for h in cfg.hashes),
                                                        cfg.zk, soundness, "; ".join(x.text() for x in cfg.proximity), cfg.interaction or T.EMPTY,
                                                        cfg.transferability,
                                                        cfg.assumptions, cfg.pin, "yes" if cfg.archived else "no",
                                                        ", ".join(w for w, _ in cells) or T.EMPTY]))
    return out


def _cell_name(c: ViewCell) -> str:
    return f"{c.subcircuit or c.line}{' · ' + _SCHEME[c.scheme].label if c.scheme else ''} · {column_header(c.column)}"


def overheads(c: ViewCell) -> tuple[float, float]:
    """``(serving overhead, proving overhead)`` of a cell: commitment and proving seconds per input, each times N; they sum
    to the cell's ``N / P``."""
    P = c.throughput
    return c.native * P["commit_s"] / P["instances"], c.native * P["proving_s"] / P["instances"]


def _fmt_x(v: float) -> str:
    return "0×" if v == 0 else T.fmt_overhead(v)


def stages(c: ViewCell) -> dict[str, Any]:
    """Table 3 in overhead units: each stage's seconds per input × N (commitment, interaction -- network wait at the reference
    profile --, the six buckets, joints, Other), which sum to
    the cell's ``N / P``; and the one-time setup (``fixed.setup_seconds``) as seconds and amortized ×native over the measured
    batch (setup × N ÷ instances), outside that sum."""
    P, ph = c.throughput, c.legacy.phases
    k = c.native / P["instances"]
    setup = c.setup_s
    return {"commitment": P["commit_s"] * k, "interaction": P.get("interaction_s", 0.0) * k,
            "buckets": {b: (v * k if v is not None else None) for b, v in ph["buckets"].items()},
            "joints": {name: {"covers": j["covers"], "overhead": j["seconds"] * k} for name, j in ph["joints"].items()},
            "other": ph["other"] * k if _num(ph["other"]) else None,
            "setup": {"seconds": float(setup), "amortized": float(setup) * k, "instances": P["instances"]} if _num(setup) and setup > 0 else None}


def _family_text(fam: Family) -> str:
    return (f"{fam.description}; declared class {' | '.join(fam.declared_classes)}; setup {fam.setup}; {fam.interaction}"
            + (f"; {fam.note}" if fam.note else ""))


def _mark_names(c: ViewCell) -> list[str]:
    """The Table 2 marks of a cell, in print order: ``provisional``, ``algebraic``, ``under-reverification``."""
    return [m for m, on in (("provisional", c.provisional), ("algebraic", c.algebraic), ("under-reverification", c.legacy.finding)) if on]


_MARK_TEXT = {"provisional": "(prov.)", "algebraic": "(alg.)", "under-reverification": f"({T.UNDER_REVERIFICATION})"}


#: A live run whose own round trips (rounds x its measured RTT) exceed this share of its measured wall is footnoted: its wall is
#: the network's, not the prover's (red-team-flock 2026-09-26: the RTX 5090 cells' 3.2 ms separate-verifier RTT).
RTT_DOMINATES = 0.5


def rtt_share(ix: Mapping[str, Any]) -> float | None:
    """A live result's measured round-trip time (rounds x its own RTT) as a share of its measured wall (prover compute + network
    wait); ``None`` when not live or not measured."""
    if not ix.get("live") or not _num(ix.get("rtt_ms")) or not _num(ix.get("rounds")):
        return None
    wall = (ix.get("prover_s") or 0.0) + (ix.get("wait_s") or 0.0)
    return (ix["rounds"] * ix["rtt_ms"] / 1e3) / wall if wall > 0 else None


def _cell_note(c: ViewCell, col: str) -> str:
    P, b = c.throughput, c.batching
    text = (f"{c.artifact} — {c.configuration or col}{', scheme ' + c.scheme if c.scheme else ''}; P {fmt_rate(P['instances_per_second'])} ({P['instances']} instances in "
            f"{fmt_per_instance(P['end_to_end_s'])}: commitment {fmt_per_instance(P['commit_s'])}, proving {fmt_per_instance(P['proving_s'])}); "
            f"batching total {b['total']}, per proof {b['per_proof'] or '?'}, proofs {b['proofs'] or '?'}, in flight {b['in_flight'] or '?'}; "
            f"{_instances_text(c.legacy.target, c.instances, c.subcircuit)}; whole-proof bound {_fmt_bound(c.bound_log2)}; host {c.legacy.host or '?'}; {c.legacy.verification_note()}")
    if c.legacy.instances_equiv:
        text += f"; {T.INSTANCE_EQUIV_NOTE} ({c.legacy.instances_equiv})"
    ixl = P.get("interaction") or {}
    if ixl.get("loopback_s"):
        text += (f"; live-coin handling {1e3 * ixl['loopback_round_s']:.3g} ms per round, measured on loopback, counted as prover compute "
                 f"({ixl['loopback_method'] or 'method not recorded'})")
    share = rtt_share(ixl)
    if share is not None and share > RTT_DOMINATES:
        text += (f"; separate verifier at {ixl['rtt_ms']:.3g} ms measured RTT: its {ixl['rounds']:.0f} round trips are {100 * min(share, 1):.0f}% of "
                 f"the measured wall, so that wall is the network's; P is the formula at the 1 ms reference and cells are ranked on it, "
                 f"never on measured total time")
    if c.provisional:
        text += f"; provisional: {c.provisional}"
    if c.legacy.finding:
        text += f"; {c.legacy.finding}"
    if c.algebraic:
        text += f"; {ALGEBRAIC_FLAG} (Table 1, {c.configuration}: {'; '.join(c.algebraic)})"
    if c.unfiltered:
        u = c.unfiltered
        text += (f"; unfiltered pick: {u['configuration']} {u['artifact']}, bound {_fmt_bound(u['bound_log2'])}, hashes "
                 f"{'; '.join(u['hashes'])}, {T.fmt_overhead(u['overhead'])}")
    return text


def table2_grid(V: Views) -> list[dict[str, Any]]:
    """Table 2 as printed, one entry per line: its row ``label``, the native column and each family column as ``display`` (the
    printed text without its footnote number) and ``note`` (the footnote text), with ``state`` (``cell``, or the empty
    cell's: ``n/s``, ``none``, ``weaker``, ``rejected``, ``below-filter``), ``marks`` and the ``cell`` itself.  The markdown
    numbers the footnotes in this order: each line's native column, then its family columns."""
    out = []
    for ln in V.lines:
        N = ln["native"]
        g: dict[str, Any] = {
            "line": ln, "label": f"{N['device']} · {N['dtype']} · {ln.get('subcircuit_display') or ln['line']}" + (f" · {ln['scheme_label']}" if ln["scheme"] else ""),
            "native": {"display": _native_text(N), "note": f"{ln.get('subcircuit') or ln['line']}: {N['basis']}; {N['source']}" if N.get("basis") else
                       f"{ln.get('subcircuit') or ln['line']}: {N['work_model']}; {N['ops_per_instance']} tensor-core FLOP per instance "
                                                           f"({N['steps']} steps); peak {N['peak_flop_per_second']:.4g} FLOP/s, {N['source']}"},
            "cells": {}}
        for col in V.columns():
            c = ln["cells"].get(col)
            if c is None:
                e = ln["empty"].get(col) or {"state": "none", "text": "no result"}
                where = f"{ln.get('subcircuit') or ln['line']}{' · ' + ln['scheme_label'] if ln['scheme'] else ''} · {col}"
                g["cells"][col] = {"display": "n/s" if e["state"] == "n/s" else T.EMPTY, "note": f"{where}: {e['text']}", "state": e["state"],
                                   "marks": [], "cell": None}
                continue
            marks = _mark_names(c)
            g["cells"][col] = {"display": " ".join([T.fmt_overhead(c.overhead), *(_MARK_TEXT[m] for m in marks)]), "note": _cell_note(c, col),
                               "state": "cell", "marks": marks, "cell": c}
        out.append(g)
    return out


def table3_row(c: ViewCell) -> dict[str, str]:
    """Table 3's printed row of a cell: ``commitment``, ``interaction``, each of :data:`contract.BUCKETS`, ``other``, ``total`` and
    ``setup`` (a fused stage prints once in its first bucket, the continuation mark in the others)."""
    st = stages(c)
    share = lambda v: f"{_fmt_x(v)} ({100 * v / c.overhead:.0f}%)"  # noqa: E731
    cells = {b: share(v) for b, v in st["buckets"].items() if v is not None}
    for j in st["joints"].values():
        first, *rest = j["covers"]
        cells[first] = f"{'+'.join(j['covers'])}: {share(j['overhead'])}"
        cells.update({b: T.JOINT_CONTINUATION for b in rest})
    su = st["setup"]
    return {"commitment": share(st["commitment"]), "interaction": share(st["interaction"]), **{b: cells.get(b, T.EMPTY) for b in contract.BUCKETS},
            "other": share(st["other"]) if st["other"] is not None else T.EMPTY, "total": f"{T.fmt_overhead(c.overhead)} (100%)",
            "setup": f"{T.fmt_seconds(su['seconds'])} · {_fmt_x(su['amortized'])} over {su['instances']}" if su else T.EMPTY}


def render_markdown(V: Views, published: bool = False) -> str:
    rules = V.rules
    filt = ", ".join(([f"whole-proof bound 2^{rules.security_log2:g} or better"] if rules.security_log2 is not None else [])
                     + (["no algebraic hash in the soundness argument"] if rules.standard_hashes else [])) or "unfiltered"
    out = [f"> {PUBLISHED if published else PREVIEW}", "", f"Rules: `{rules.name}` ({filt}" + ("; legacy parameters: B = 4096, K = 1536, bare and + in-proof hash "
                                                                 "columns, proving time only" if rules.legacy else "") + ").", "",
           "## Table 1 — Backend configurations and their security", ""]
    rows1, fam_seen = [], None
    for fid, row in table1_rows(V):
        if fid != fam_seen:
            fam = _FAMILY.get(fid)
            desc = _family_text(fam) if fam else "not a Table 2 column"
            rows1.append([f"**{column_header(fid) if fid else fid}**", "", desc, *([""] * (len(TABLE1_COLUMNS) - 3))])
            fam_seen = fid
        rows1.append(row)
    out += [T._md_table(TABLE1_COLUMNS, rows1), "",
            "Statement: the full relation commits x, W and y with the configuration's scheme and proves y = S(x, W) over the committed "
            "values; its scheme must be defined in verity.ml.commitments (reason S otherwise). A configuration whose soundness argument "
            f"uses an algebraic hash (Poseidon2 and the like) is admissible and flagged: {ALGEBRAIC_FLAG}. Bounds "
            "are the accountant's whole-proof figures at each cell's batching (`security.achieved_log2`), from proven proximity bounds only.", "",
            "## Table 2 — End-to-end overhead per subcircuit, hardware class and commitment scheme", "",
            "N = compute-bound datasheet estimate, instances/s (busiest unit, peak entry); P = instances/s end to end at the reference "
            "network (same datacenter, 1 ms RTT, 100 Gb/s: t = compute + rounds × RTT + bytes ÷ bandwidth) "
            f"(committing the batch, `{COMMIT}`, then proving){'; legacy: proving only' if rules.legacy else ''}; cell = N / P of the "
            "family's best admissible configuration (highest P). `(prov.)` = no red-team clearance of the statement yet; "
            f"`({T.UNDER_REVERIFICATION})` = the latest `finding` label on the result is open (a `PULLED` one rejects it, D); "
            f"`(alg.)` = its soundness argument uses an algebraic hash: {ALGEBRAIC_FLAG} (Table 1's flag); "
            f"`{T.EMPTY}` = no admissible result; `n/s` = declared unsupported.", ""]
    notes: list[str] = []

    def note(text: str) -> str:
        notes.append(text)
        return f"[{len(notes)}]"

    cols = V.columns()
    rows2 = [[g["label"], g["native"]["display"] + " " + note(g["native"]["note"]),
              *[g["cells"][col]["display"] + " " + note(g["cells"][col]["note"]) for col in cols]] for g in table2_grid(V)]
    out += [T._md_table(("Subcircuit · hardware · scheme", "Native N", *map(column_header, cols)), rows2), ""]

    out += ["## Table 3 — Where the overhead comes from (each stage × N, share of the cell)", "",
            "Each stage is its contribution to the Table 2 overhead: its time per instance × N, so commitment + interaction + the six "
            "buckets + Other add up to the cell (Total). Commitment is the serving overhead; interaction is the network wait at the "
            "reference profile (rounds × 1 ms + bytes ÷ 100 Gb/s, plus verifier compute the prover waits on); the buckets and Other are "
            "the proving overhead. A fused stage "
            f"prints once in its first column, `{T.JOINT_CONTINUATION}` in the others. Setup, one-time (keys, circuit setup: "
            "`fixed.setup_seconds`) is outside the cell: its seconds, and ×native amortized over the measured batch (setup × N ÷ the "
            "batch's instances). Seconds per instance are in the drill-down below.", ""]
    rows3, rows3s = [], []
    for c in V.cells:
        r3 = table3_row(c)
        su = stages(c)["setup"]
        rows3.append([_cell_name(c), r3["commitment"], r3["interaction"], *[r3[b] for b in contract.BUCKETS], r3["other"], r3["total"], r3["setup"]])
        P, ph = c.throughput, c.legacy.phases
        n, total = P["instances"], P["end_to_end_s"]
        pct = lambda v: f"{fmt_per_instance(v / n)} ({100 * v / total:.0f}%)"  # noqa: E731
        secs = {b: pct(v) for b, v in ph["buckets"].items() if v is not None}
        for j in ph["joints"].values():
            first, *rest = j["covers"]
            secs[first] = f"{'+'.join(j['covers'])}: {pct(j['seconds'])}"
            secs.update({b: T.JOINT_CONTINUATION for b in rest})
        rows3s.append([_cell_name(c), pct(P["commit_s"]), pct(P.get("interaction_s", 0.0)), *[secs.get(b, T.EMPTY) for b in contract.BUCKETS],
                       pct(ph["other"]) if _num(ph["other"]) else T.EMPTY, f"{fmt_per_instance(total / n)} (100%)",
                       T.fmt_seconds(su["seconds"]) if su else T.EMPTY])
    head = ("Cell", "Commitment", "Interaction", "Witness", "Encoding + witness commitment", "Arithmetic", "Lookup", "ZK additional", "Serialization", "Other")
    out += [T._md_table((*head, "Total (the cell)", "Setup, one-time (not in the total)"), rows3)
            if rows3 else f"(no Table 2 cell {T.EMPTY} nothing to decompose)", ""]
    derived = [c for c in V.cells if c.throughput["interaction"]["derived"]]
    if derived:
        shares = [100 * stages(c)["interaction"] / c.overhead for c in derived]
        out += [f"Interaction note: {len(derived)} of {len(V.cells)} cells are verified from files, so their interaction is derived "
                f"(no measured RTT; rounds = 0 under Fiat–Shamir, else the protocol's sequential depth; network time = rounds × 1 ms + "
                f"(proof + statement bytes) ÷ 100 Gb/s). That interaction is {min(shares):.0f}–{max(shares):.0f}% of those cells at the "
                f"reference network: proof size is a real cost.", ""]
    if rows3s:
        out += ["### Table 3 drill-down — seconds per instance (share of end to end)", "",
                T._md_table((*head, "End to end", "Setup, one-time (s)"), rows3s), ""]
        rowsi = []
        for c in V.cells:
            ix, P = c.throughput["interaction"], c.throughput
            at = lambda rtt, bw=REF_BANDWIDTH_BPS: c.native * (P["commit_s"] + interaction_time(ix, rtt, bw)) / P["instances"]  # noqa: E731
            rtt = (f"{ix['rtt_ms']:.3g} ms ({ix['rtt_method'] or 'method not recorded'})" if ix["rtt_ms"] is not None
                   else "— (file-verified)" if ix["derived"] else T.EMPTY)
            rounds = f"{ix['rounds']:.0f} (derived)" if ix["derived"] else f"{ix['rounds']:.0f}" if ix["rounds"] is not None else T.UNMEASURED
            rowsi.append([_cell_name(c), rounds, D._bytes(ix["bytes_up"]),
                          D._bytes(ix["bytes_down"]), rtt, T.fmt_seconds(ix["prover_s"] or 0.0), T.fmt_seconds(ix["verifier_s"]),
                          T.fmt_seconds(ix["wait_s"]) if ix["wait_s"] is not None else T.UNMEASURED,
                          *[T.fmt_overhead(at(n.rtt_s, n.bandwidth_bps)) for n in (census.REFERENCE_NETWORK, *census.SENSITIVITY_NETWORKS)]])
        out += ["### Interaction drill-down — the formula t = compute + rounds × RTT + bytes ÷ bandwidth", "",
                T._md_table(("Cell", "Rounds", "Bytes up", "Bytes down", "Measured RTT (method)", "Prover compute", "Verifier compute (waited on)",
                             "Measured network wait", "Cell at 1 ms, 100 Gb/s (reference)", "At 20 ms, 100 Gb/s", "At 100 ms, 100 Gb/s",
                             "At 1 ms, 10 Gb/s"), rowsi), ""]
    if not rules.legacy:
        from verity_numerical.bench import headline as H  # noqa: PLC0415  (headline reads these views)

        out += H.render_markdown(H.headline(V))
    if notes:
        out += ["Footnotes:", *[f"[{i}] {n}" for i, n in enumerate(notes, 1)], ""]
    out.append(f"Rejected ({len(V.rejected)} result{'s' if len(V.rejected) != 1 else ''}; artifact → first reason; letters: "
               + ", ".join(f"{k} {v}" for k, v in REASON_LETTERS.items() if rules.legacy or k != "B") + "):")
    for r in V.rejected:
        k, why = r["codes"][0]
        pre = r.get("end_to_end_preview")
        how = f"{COMMIT} {pre['commit_s']:.3g} s" if pre and pre["commit_warm"] else f"{COMMIT_V1} {pre['commit_s']:.3g} s, once before the timed runs" if pre else ""
        extra = f" [end to end with its commitment ({how}): {T.fmt_overhead(pre['overhead'])}]" if pre else ""
        out.append(f"- {r['artifact']} ({r['configuration'] or r['family']} / {r['line'] or '?'}) → {k}: {why}{extra}")
    if V.unreadable:
        out += ["", f"Unreadable here ({len(V.unreadable)}; the render cannot use them until their payloads are fetched):",
                *[f"- {u}" for u in V.unreadable]]
    if V.not_families:
        n = sum(len(v) for v in V.not_families.values())
        out += ["", f"Not a backend family's result ({n}; by software.backend.name): "
                + ", ".join(f"{name} ×{len(a)}" for name, a in sorted(V.not_families.items(), key=lambda kv: (-len(kv[1]), kv[0])))]
    return "\n".join(out) + "\n"


def render_json(V: Views, published: bool = False) -> dict[str, Any]:
    return {
        "schema": "verity/tables-views/v1", **({"published": PUBLISHED} if published else {"preview": PREVIEW}), "rules": dataclasses.asdict(V.rules),
        "reason_letters": REASON_LETTERS, "schemes": [{**dataclasses.asdict(x), "admissible": x.admissible} for x in SCHEMES],
        "work_model": WORK_MODEL,
        "table1": {"columns": list(TABLE1_COLUMNS), "families": [dataclasses.asdict(f) for f in FAMILIES],
                   "configurations": [{**dataclasses.asdict(c), "unsupported": dict(c.unsupported), "full_relation": c.full_relation}
                                      for c in V.configurations],
                   "rows": [{"family": f, "row": r} for f, r in table1_rows(V)]},
        "table2": {"columns": V.columns(), "lines": [{**{k: v for k, v in ln.items() if k != "cells"},
                                                     "cells": {k: (c.to_json() if c else None) for k, c in ln["cells"].items()}}
                                                    for ln in V.lines]},
        "table3": [{"cell": _cell_name(c), "artifact": c.artifact, "scheme": c.scheme, "instances": c.throughput["instances"],
                    "commit_s": c.throughput["commit_s"], "end_to_end_s": c.throughput["end_to_end_s"],
                    "serving_overhead": overheads(c)[0], "proving_overhead": overheads(c)[1], "stages_overhead": stages(c),
                    **c.legacy.phases} for c in V.cells],
        "rejected": [{**r, "codes": [list(x) for x in r["codes"]]} for r in V.rejected],
        "not_families": V.not_families,
        "unreadable": V.unreadable,
    }


# --- entities: the render as charted by the docs site -----------------------------------------------------------------------------

ENTITIES_SCHEMA = "verity/tables-entities/v1"
#: The ZK class of a configuration, from the leading words of its declared ZK text (the text rides alongside).  A B-Ligero
#: configuration that runs "as the B-Ligero mode it runs in" is one of its live interactive modes: malicious-verifier ZK.
ZK_CLASSES = (("malicious-verifier zk", "malicious-verifier-zk"), ("as the b-ligero mode it runs in", "malicious-verifier-zk"),
              ("hvzk", "hvzk"), ("no", "none"), ("unknown", "unknown"))
#: Table 3's stages in print order, with the column titles: the phases a result's overhead splits into.
#: An "Other" share in [-OTHER_CLAMP, 0) is shown as 0 (``clamped``, the raw value kept); anything more negative is ``flagged``.
OTHER_CLAMP = 0.005
PHASES = (("commitment", "Commitment"), ("interaction", "Interaction"), ("witness", "Witness"),
          ("encoding_commitment", "Encoding + witness commitment"), ("arithmetic", "Arithmetic"), ("lookup", "Lookup"),
          ("zk_additional", "ZK additional"), ("serialization", "Serialization"), ("other", "Other"))


def zk_class(cfg: Configuration) -> str:
    """The short ZK class (:data:`ZK_CLASSES`) of a configuration; ``ValueError`` when its text matches none."""
    t = cfg.zk.lower()
    for prefix, cls in ZK_CLASSES:
        if t.startswith(prefix):
            return cls
    raise ValueError(f"{cfg.id}: no ZK class for {cfg.zk!r}")


def split_list(text: str) -> list[str]:
    """``text``'s ``"; "``-separated items outside parentheses (``"; ".join`` of the result is ``text``)."""
    out, depth, cur = [], 0, ""
    i = 0
    while i < len(text):
        ch = text[i]
        depth += (ch == "(") - (ch == ")")
        if depth == 0 and text.startswith("; ", i):
            out.append(cur)
            cur, i = "", i + 2
            continue
        cur += ch
        i += 1
    return [*out, cur] if cur or out else []


def slug(text: str) -> str:
    return "-".join("".join(ch if ch.isalnum() else " " for ch in text.lower()).split())


def subcircuit_id(target_name: str | None) -> str | None:
    """The census subcircuit (a fully bound template) a target name denotes (its ``legacy_target``)."""
    return census.subcircuit_of(target_name).id if target_name else None


def verity_commit() -> str | None:
    """The commit of the source tree this renderer runs from (``None`` outside a git checkout)."""
    import subprocess  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


#: ``verity.core.claims`` instances of the hash names Table 1 uses (a name joining two hashes names both).
HASH_INSTANCES = {"SHA-256": ("sha-256",), "SHA-512": ("sha-512",), "BLAKE3": ("blake3",), "BLAKE3 (keyed)": ("blake3",),
                  "BLAKE2b": ("blake2b",), "SHAKE-256": ("shake-256",), "BLAKE3 / BLAKE2b": ("blake3", "blake2b"),
                  "Poseidon2-BabyBear": ("poseidon2-babybear",), "Poseidon2-KoalaBear": ("poseidon2-koalabear",),
                  "Ajtai n=64 (linear digest)": ("ajtai-n64",), "BLAKE3 / SHA-256": ("blake3", "sha-256"), "Ajtai n=128 (linear digest)": ("ajtai-n128",)}
#: ``verity.core.claims`` properties by the leading words of a hash's assumed property; a hash assumed to be a random oracle
#: is cited once, as the ``random-oracle`` model.
HASH_PROPERTIES = (("collision resistance", "cr"), ("expected-time collision resistance", "ecr"), ("commit-before-coin binding", "cr"), ("live transcript and root_f binding", "cr"),
                   ("prg", "xof"), ("ring-sis", "sis"))
PROXIMITY_REGIMES = {"unique decoding": "rs-proximity-udr", "johnson regime (list decoding)": "rs-proximity-johnson"}
#: ``verity.core.claims`` ids of a Table 1 assumption, by a phrase it contains.
ASSUMPTION_PHRASES = (("logup fractional-sum soundness", "logup"),)
#: An ``unknown`` ZK text claims nothing, so it maps to ``no-zk`` (Verity claims no zero knowledge for it; the text says why).
_ZK_GUARANTEE = {"malicious-verifier-zk": "malicious-verifier-zk", "hvzk": "hvzk", "none": "no-zk", "unknown": "no-zk"}
#: B-Ligero's malicious-verifier ZK also rests on the random-oracle model for its unsalted Merkle leaves
#: (archive/direct/ligero/PROTOCOL.md, red-team-zk-2 G3): the unopened columns are hidden only by Blake3(column).
_LIGERO_MERKLE_HIDING = ("proof-internal commitment (Merkle hiding: unsalted leaves)",
                         "BLAKE3 (proof-internal commitment; random oracle (hiding of the unopened, unsalted Merkle leaves Blake3(column)); "
                         "red-team-zk-2 G3)")


def _role(h: Hash) -> str:
    """A hash's role as the configuration declares it: its role, with the detail its assumed property gives in parentheses
    (``statement commitment (row digests)``), or the property itself when it is not a bare property name
    (``PRG of the verifier's coins``)."""
    prop = h.prop
    if "(" in prop and prop.endswith(")"):
        detail = prop[prop.index("(") + 1:-1]
    else:
        detail = "" if prop.lower() in (_CR, _RO) else prop
    return f"{h.role} ({detail})" if detail else h.role


def claims_of(cfg: Configuration, bits: float | None = None) -> list[dict[str, Any]]:
    """A configuration's security claims: each a ``property`` (a ``verity.core.claims`` id with its Table 1 text, and for soundness
    ``param``, the weakest whole-proof bound at its cells as TeX) with the ``assumptions`` it rests on, each an id (``None``
    and ``unmapped`` when the renderer has none yet), its ``role`` as the configuration declares it (a hash's role and
    detail, a proximity bound's component; empty where nothing is declared) and its Table 1 text.

    * soundness rests on every hash of the soundness argument but those assumed to be random oracles, the proximity bounds
      but those machine-checked in its own proof, and the configuration's other assumptions;
    * ZK: malicious-verifier ZK whose text names the coin commitment rests on the hashes that commit the coins; HVZK on an
      honest verifier;
    * non-interactive (transferable) rests on the random oracle, once per hash modelled as one; an interactive
      configuration is designated-verifier and rests on nothing further;
    * setup: transparent, resting on nothing."""

    def prop(id_: str | None, text: str, param: str | None = None) -> dict[str, Any]:
        return {"id": id_, "param": param, "text": text, "unmapped": id_ is None}

    def rests(id_: str | None, role: str, text: str) -> dict[str, Any]:
        return {"id": id_, "role": role, "text": text, "unmapped": id_ is None}

    def hash_ids(h: Hash) -> list[str | None]:
        p = next((pid for words, pid in HASH_PROPERTIES if h.prop.lower().startswith(words)), None)
        return [f"{p}/{i}" for i in HASH_INSTANCES.get(h.name, ())] if p else []

    transferable = cfg.transferability.startswith("transferable")
    oracles = [h for h in cfg.hashes if h.prop.lower().startswith("random oracle")]
    sound = [rests(i, _role(h), h.text()) for h in cfg.hashes if h not in oracles for i in (hash_ids(h) or [None])]
    sound += [rests(PROXIMITY_REGIMES.get(x.regime.lower()) if x.basis.startswith("proven") else None, x.component, x.text())
              for x in cfg.proximity if not x.basis.startswith(MACHINE_CHECKED)]
    sound += [rests(next((i for words, i in ASSUMPTION_PHRASES if words in a.lower()), None), "", a) for a in split_list(cfg.assumptions)]
    if transferable:
        sound += [rests("random-oracle", _role(h), h.text()) for h in oracles]
    cls = zk_class(cfg)
    zk_rests: list[dict[str, Any]] = []
    if cls == "malicious-verifier-zk" and "coin commitment" in cfg.zk.lower():
        zk_rests = [rests(i, _role(h), h.text()) for h in cfg.hashes if "coin commitment" in h.prop.lower() for i in (hash_ids(h) or [None])]
    if cls == "malicious-verifier-zk" and cfg.family == "B-Ligero":
        zk_rests.append(rests("random-oracle", *_LIGERO_MERKLE_HIDING))
    elif cls == "hvzk":
        zk_rests = [rests("honest-verifier", "zero knowledge only (soundness does not rest on it)", cfg.zk)]
    claims = [{"property": prop("computational-soundness" if transferable else "statistical-soundness", cfg.soundness,
                                f"2^{{-{bits:.1f}}}" if bits is not None else None), "assumptions": sound},
              {"property": prop(_ZK_GUARANTEE.get(cls), cfg.zk), "assumptions": zk_rests}]
    if transferable:
        claims.append({"property": prop("non-interactive", cfg.transferability),
                       "assumptions": [rests("random-oracle", _role(h), h.text()) for h in oracles]})
    elif cfg.transferability.startswith("interactive"):
        claims.append({"property": prop("designated-verifier", cfg.transferability), "assumptions": []})
    else:
        claims.append({"property": prop(None, cfg.transferability), "assumptions": []})
    fam = _FAMILY.get(cfg.family or "")
    if fam:
        claims.append({"property": prop("transparent-setup" if fam.setup.startswith("none") else None, fam.setup), "assumptions": []})
    return claims


#: The families' Glossary names (Daniel, 2026-09-25): a letter, then the family.  Column ids (``FAMILIES``) stay the cell keys
#: the parity gate compares with the frozen tables; the markdown prints the names (:func:`column_header`).
FAMILY_NAMES = {"A-GKR": "A-GKR", "B-Ligero": "B-Ligero", "Flock": "C-Flock", "SP1": "D-SP1"}


def _fam(fid: str | None) -> str | None:
    return FAMILY_NAMES.get(fid, fid) if fid else fid


def column_header(col: str) -> str:
    """A Table 2 column's printed header: its family's name (``SP1`` -> ``D-SP1``), then any authentication suffix."""
    fid = next((f for f in FAMILY_NAMES if col == f or col.startswith(f + " ")), None)
    return FAMILY_NAMES[fid] + col[len(fid):] if fid else col


#: Composed backends with their own id (``README.md``'s Glossary, "Backend"): route (a) renders in the A-GKR column, and its
#: profile adds C-Flock's assumptions and the sigma link's.
COMPOSED = {ROUTE_A: "A-route-a"}


def backend_of(cfg: Configuration) -> str | None:
    """The backend (an implementation of the Backend interface) a configuration belongs to: ``<letter>-interactive`` or
    ``<letter>-fs``, or a composed backend's own id (:data:`COMPOSED`); ``None`` outside the backend families."""
    if cfg.id in COMPOSED:
        return COMPOSED[cfg.id]
    return f"{_fam(cfg.family)[0]}-{'fs' if cfg.transferability.startswith('transferable') else 'interactive'}" if cfg.family else None


#: Scheme ids for the site, naming the framing and the leaf hash (the tree hash is the framing's: frame-v3 and vllm-v1 trees
#: are SHA-256), by the renderer's scheme id (``legacy_id``; the markdown prints those).  A new id only where the committed
#: bytes differ.
SCHEME_IDS = {"frame-v3/sha256": "frame-v3/sha-256", "frame-v3/blake3-row": "frame-v3/blake3-keyed", "vllm-v1": "vllm-v1/sha-256",
              "frame-v3/poseidon2-row": "frame-v3/poseidon2-babybear", "frame-v3/poseidon2-tile": "frame-v3/poseidon2-babybear-tile64",
              "frame-v3/ajtai-row": "frame-v3/ajtai"}
#: The ``ligero-verify`` leaf schemes (``src/leaf.rs``) that commit each scheme's leaves: a circuit, not a scheme.  ``blake3``
#: and ``blake3-xob`` share the schema ``blake3-keyed/row/v2``, its params digest and leaf bytes, so they commit the same bytes
#: (``leaf/conformance_test.py`` asserts it): one scheme, two circuits.
LEAF_CIRCUITS = {"frame-v3/sha-256": ("sha256",), "frame-v3/blake3-keyed": ("blake3", "blake3-xob"), "vllm-v1/sha-256": ("vllm-v1",),
                 "frame-v3/poseidon2-babybear": ("poseidon2",), "frame-v3/poseidon2-babybear-tile64": (), "frame-v3/ajtai": ("ajtai-n64", "ajtai-n128")}


def scheme_id(legacy: str | None) -> str | None:
    return SCHEME_IDS.get(legacy, legacy) if legacy else legacy


def mode_of(cfg: Mapping[str, Any]) -> str:
    """``fs`` (Fiat–Shamir: non-interactive, transferable) or ``interactive``."""
    return "fs" if cfg["transferability"].startswith("transferable") else "interactive"


def backends(configurations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Table 1 as backends (each implements the Backend interface): per backend family and interaction mode,
    ``<letter>-interactive`` or ``<letter>-fs`` (``A-fs``: A-GKR under Fiat–Shamir), and composed backends by their own id
    (``A-route-a``), each with its member configurations and its claims written out in full.  Each claim slot (soundness,
    ZK, interaction, setup) holds the property its members share (``id`` ``None`` and ``unmapped`` when they differ or have
    none; ``texts`` are theirs), resting on every assumption any member rests on for it, with the roles the code declares and
    the members that cite it; the soundness ``param`` is the weakest of the members'.  A composed backend's soundness also
    rests on its components' assumptions (``from``: the component).  Statement-commitment hashes are left out: they are the
    statement's (its scheme's), and a result's full assumption set is its backend's plus its statement's.  ``implementations``
    names the renderer's configurations a backend covers (an implementation detail): those with a result in this render when any
    has one (the profile is what the published results rest on); the others are ``declared_without_results``."""
    order = {"interactive": 0, "route": 1, "fs": 2}
    ids = sorted({c["backend"] for c in configurations if c["backend"]},
                 key=lambda b: (b[0], order["route"] if b not in (f"{b[0]}-interactive", f"{b[0]}-fs") else order[b.split("-", 1)[1]], b))
    out = []
    for bid in ids:
        declared = [c for c in configurations if c["backend"] == bid]
        members = [c for c in declared if c.get("cells")] or declared
        claims = []
        for slot in range(max(len(c["claims"]) for c in members)):
            props = [c["claims"][slot]["property"] for c in members if slot < len(c["claims"])]
            pids, texts = {p["id"] for p in props}, list(dict.fromkeys(p["text"] for p in props))
            pid = next(iter(pids)) if len(pids) == 1 and len(props) == len(members) else None
            params = [p["param"] for p in props if p.get("param")]
            rests: dict[str, dict[str, Any]] = {}
            for c in members:
                for a in (c["claims"][slot]["assumptions"] if slot < len(c["claims"]) else ()):
                    if a["role"].startswith(_STMT):
                        continue                      # the statement's (its scheme's hashes), not the backend's
                    e = rests.setdefault(a["id"] or a["text"], {"id": a["id"], "roles": [], "texts": [], "unmapped": a["unmapped"],
                                                                "implementations": [], "from": None})
                    for key, v in (("roles", a["role"]), ("texts", a["text"]), ("implementations", c["variant"])):
                        if v and v not in e[key]:
                            e[key].append(v)
            claims.append({"property": {"id": pid, "param": max(params, key=lambda t: float(t[2:].strip("{}"))) if params else None,
                                        "texts": texts, "unmapped": pid is None},
                           "assumptions": list(rests.values())})
        fam = members[0]["backend_family"]
        entry = {"id": bid, "display": bid, "backend_family": fam, "column": fam,
                 "mode": "fs" if bid.endswith("-fs") else "interactive", "composed": bid in COMPOSED.values(), "components": [],
                 "implementations": [c["variant"] for c in members],
                 "declared_without_results": [c["variant"] for c in declared if c not in members], "claims": claims}
        if entry["composed"]:
            _compose(entry, members, configurations)
        entry["properties"] = [c["property"]["id"] for c in claims if c["property"]["id"]]
        entry["assumptions"] = list(dict.fromkeys(a["id"] for c in claims for a in c["assumptions"] if a["id"]))
        out.append(entry)
    return out


def _compose(entry: dict[str, Any], members: Sequence[Mapping[str, Any]], configurations: Sequence[Mapping[str, Any]]) -> None:
    """Route (a)'s profile: its soundness also rests on C-Flock's soundness assumptions and on the sigma link -- the link's
    bound as its configuration states it, and the hash its code declares binding the live transcript and ``root_F``."""
    sound = entry["claims"][0]["assumptions"]
    for a in sound:
        if any("root_f" in t.lower() for t in a["texts"]):
            a["from"] = "link"
    have = {a["id"] or a["texts"][0] for a in sound}
    for c in configurations:
        if c["backend_family"] == "C-Flock":
            for a in c["claims"][0]["assumptions"]:
                if (a["id"] or a["text"]) not in have and not a["role"].startswith(_STMT):
                    sound.append({"id": a["id"], "roles": [a["role"]] if a["role"] else [], "texts": [a["text"]], "unmapped": a["unmapped"],
                                  "implementations": [c["variant"]], "from": "C-Flock"})
                    have.add(a["id"] or a["text"])
    links = [m for c in members for m in re.findall(r"sigma link [^;,)]*", c["soundness"]["text"] + "; " + c["soundness"]["dominant_term"])]
    for t in dict.fromkeys(x.strip() for x in links):
        sound.append({"id": None, "roles": ["sigma link"], "texts": [t], "unmapped": True, "implementations": [c["variant"] for c in members], "from": "link"})
    entry["components"] = ["C-Flock", "link"]


def scheme_entities(configurations: Sequence[Configuration]) -> list[dict[str, Any]]:
    """Each commitment scheme under its site id (:data:`SCHEME_IDS`; ``legacy_id`` the renderer's): ``name``, a one-line
    ``description`` (the commitment text of the first full-relation configuration that binds it, up to its first ``;``), its
    ``tree_hash`` and ``leaf_hashes`` (``verity.core.claims`` instances of the statement-commitment hashes its configurations declare:
    the tree's role names the trees), ``algebraic`` when any of them is an algebraic instance, the ``assumptions`` those hashes
    are (with the roles and configurations that declare them), and the ``leaf_circuits`` that commit its leaves."""
    from verity.core.claims import INSTANCES  # noqa: PLC0415

    out = []
    for sch in SCHEMES:
        binders = [c for c in configurations if c.scheme == sch.id]
        rests: dict[str, dict[str, Any]] = {}
        tree: list[str] = []
        leaves: list[str] = []
        for c in binders:
            for h in c.hashes:
                if h.role != _STMT:
                    continue
                prop = next((pid for words, pid in HASH_PROPERTIES if h.prop.lower().startswith(words)), None)
                insts = HASH_INSTANCES.get(h.name, ())
                (tree if "trees" in _role(h) else leaves).extend(i for i in insts if i not in (tree if "trees" in _role(h) else leaves))
                for i in ([f"{prop}/{x}" for x in insts] if prop else []) or [None]:
                    e = rests.setdefault(i or h.text(), {"id": i, "unmapped": i is None, "roles": [], "implementations": []})
                    for key, v in (("roles", _role(h)), ("implementations", c.id)):
                        if v not in e[key]:
                            e[key].append(v)
        full = next((c for c in binders if c.full_relation), binders[0] if binders else None)
        sid = scheme_id(sch.id)
        out.append({"id": sid, "legacy_id": sch.id, "name": sch.label, "description": full.commitment.split(";")[0] if full else None,
                    "tree_hash": tree[0] if tree else None, "leaf_hashes": leaves,
                    "algebraic": any(INSTANCES[i].algebraic for i in (*tree, *leaves) if i in INSTANCES),
                    "leaf_circuits": list(LEAF_CIRCUITS.get(sid, ())),
                    "assumptions": [e["id"] for e in rests.values() if e["id"]], "assumption_details": list(rests.values()),
                    "admissible": sch.admissible, "standard": sch.standard, "core": sch.core, "implementations": [c.id for c in binders]})
    return out


def best_standard_hash(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per backend family and subcircuit, the lowest-overhead cell whose commitment scheme is built on standard hashes only and whose
    soundness argument uses no algebraic hash (ties to the lower artifact id)."""
    best: dict[tuple[str, str], Mapping[str, Any]] = {}
    for r in results:
        sch = _SCHEME.get(r["legacy_scheme"] or "")
        if sch is None or not sch.standard or r["state"]["algebraic"]:
            continue
        k = (r["backend_family"], r["subcircuit"])
        if k not in best or (r["overhead"]["value"], r["artifact"]) < (best[k]["overhead"]["value"], best[k]["artifact"]):
            best[k] = r
    return [{"backend_family": r["backend_family"], "subcircuit": r["subcircuit"], "result": r["id"], "backend": r["backend"],
             "statement": r["statement"], "scheme": r["scheme"], "overhead": r["overhead"]}
            for r in sorted(best.values(), key=lambda r: (r["backend_family"], r["subcircuit"]))]


#: How a live run measured its RTT, as published: the recorded ``rtt_method`` text names hosts and ports, so it stays in the
#: evidence store (the result's own record) and the entities render carries only one of these.
RTT_METHODS = ("tcp-connect", "open-connection-ping", "unclassified")


def rtt_method_enum(text: str | None) -> str | None:
    """The published RTT method of a recorded ``rtt_method`` text (:data:`RTT_METHODS`), ``None`` when none is recorded."""
    if not text:
        return None
    t = text.lower()
    if "ping" in t and "connection" in t:
        return "open-connection-ping"
    if "tcp connect" in t:
        return "tcp-connect"
    return "unclassified"


def interaction_attempts(V: Views, c: ViewCell, ix: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every interaction attempt of a live cell (IX1): the cell's own (``counted``) first, then each result labelled
    ``interaction_attempt_of`` it (a label may name the cell by an art-id prefix).  Empty for a file-verified result."""
    if not ix["live"]:
        return []
    linked = [a for key, attempts in V.interaction_attempts.items() if c.artifact.startswith(key) for a in attempts]
    return [{"artifact": c.artifact, "deviation": interaction_deviation(ix), "counted": True},
            *({**a, "counted": False} for a in sorted(linked, key=lambda a: a["artifact"]))]


def interaction_fields(ix: Mapping[str, Any], P: Mapping[str, Any], raw: Mapping[str, Any]) -> dict[str, Any]:
    """A result's interaction record, enough to recompute its overhead at any network:
    ``overhead = N x (commit_s + prover_compute_s + verifier_compute_s + rounds x RTT + (bytes_up + bytes_down) x 8 / bandwidth) / inputs``
    (``inputs``: the inputs per timed run).
    ``rounds`` is what the formula uses (``rounds.sequential_depth``, else ``rounds.protocol``; for a file-verified result 0 under
    Fiat–Shamir, else the protocol's sequential depth over its sequential sub-batch proofs), ``rounds_source`` says which;
    ``prover_compute_s`` already counts the rounds x loopback moved out of the measured wait (``loopback_moved_s``);
    ``measured_rtt.method`` is an :data:`RTT_METHODS` value, never the recorded text."""
    num = lambda v: float(v) if _num(v) else None  # noqa: E731
    return {
        "live": ix["live"], "derived": ix["derived"], "rounds": ix["rounds"], "rounds_source": ix["rounds_source"],
        "rounds_protocol": num(raw.get("rounds_protocol")), "sequential_depth": num(raw.get("sequential_depth")),
        "bytes_up": ix["bytes_up"], "bytes_down": ix["bytes_down"], "proof_bytes": ix["proof_bytes"], "statement_bytes": ix["statement_bytes"],
        "commit_s": P["commit_s"], "prover_compute_s": ix["prover_s"], "verifier_compute_s": ix["verifier_s"], "network_wait_s": ix["wait_s"],
        "measured_rtt": {"ms": ix["rtt_ms"], "method": rtt_method_enum(ix["rtt_method"])} if ix["rtt_ms"] is not None else None,
        "measured_bandwidth_bps": ix.get("bandwidth_bps"),
        "loopback_per_round": ({"s": num(raw.get("loopback_round_s")), "method": raw.get("loopback_method")}
                               if num(raw.get("loopback_round_s")) is not None else None),
        "loopback_moved_s": ix["loopback_s"], "inputs": P["instances"],
        "rtt_share": rtt_share(ix), "rtt_dominated": bool((rtt_share(ix) or 0) > RTT_DOMINATES),
        "formula": "overhead = N * (commit_s + prover_compute_s + verifier_compute_s + rounds * rtt_s + (bytes_up + bytes_down) * 8 / bandwidth_bps) / inputs",
    }


def best_per_key(cells: Sequence[ViewCell], key: Callable[[ViewCell], str]) -> list[ViewCell]:
    """Every admissible result, one per key (backend x statement x input set x prover hardware): the highest ``P``, ties to
    the lower artifact id.  Unlike Table 2's cell (each family's best), every backend's best is kept."""
    best: dict[str, ViewCell] = {}
    for c in cells:
        k = key(c)
        if k not in best or (-c.throughput["instances_per_second"], c.artifact) < (-best[k].throughput["instances_per_second"], best[k].artifact):
            best[k] = c
    return list(best.values())


def configuration_records(V: Views) -> list[dict[str, Any]]:
    """The renderer's configurations as records, the input of :func:`backends` (an implementation detail: the entities render
    emits backends and statements, not configurations).  Each has its backend, variant, scheme, ZK class, soundness bits (the
    weakest whole-cell bound at its cells), hashes, texts and ``claims`` (:func:`claims_of`)."""
    cells_of: dict[str, list[ViewCell]] = {}
    for c in V.cells:
        cells_of.setdefault(c.configuration or "", []).append(c)
    rows1 = table1_rows(V)
    configurations = []
    for (fid, row), cfg in zip(rows1, [c for fid in [*(f.id for f in FAMILIES), None]
                                       for c in V.configurations if c.family == fid or (fid is None and c.family not in _FAMILY)]):
        held = cells_of.get(cfg.id, [])
        bits = [-c.bound_log2 for c in held if c.bound_log2 is not None]
        fam = _FAMILY.get(cfg.family or "")
        claims = claims_of(cfg, min(bits) if bits else None)
        configurations.append({
            "id": slug(cfg.id), "variant": cfg.id, "cells": len(held), "backend_family": _fam(cfg.family), "scheme": cfg.scheme,
            "backend": backend_of(cfg),
            "statement": {"full_relation": cfg.full_relation, "text": cfg.statement}, "ports": cfg.ports, "commitment": cfg.commitment,
            "zk": {"class": zk_class(cfg), "text": cfg.zk},
            "soundness": {"bits": min(bits) if bits else None, "text": cfg.soundness, "dominant_term": cfg.dominant_term},
            "hashes": [{"name": h.name, "role": h.role, "property": h.prop, "class": "algebraic" if h.algebraic else "standard"} for h in cfg.hashes],
            "hash_class": "algebraic" if cfg.algebraic() else "standard", "algebraic_flag": ALGEBRAIC_FLAG if cfg.algebraic() else None,
            "proximity": [dataclasses.asdict(x) for x in cfg.proximity],
            "interaction": split_list(cfg.interaction), "transferability": cfg.transferability, "assumptions_text": split_list(cfg.assumptions),
            "claims": claims, "properties": [c["property"]["id"] for c in claims if c["property"]["id"]],
            "assumptions": list(dict.fromkeys(a["id"] for c in claims for a in c["assumptions"] if a["id"])),
            "models": ["fiat-shamir" if cfg.transferability.startswith("transferable") else "live-verifier"],
            "params": {c["property"]["id"]: c["property"]["param"] for c in claims if c["property"]["id"] and c["property"]["param"]},
            "unmapped": [x["text"] for c in claims for x in (c["property"], *c["assumptions"]) if x["unmapped"]],
            "setup_text": fam.setup if fam else None, "pin": cfg.pin, "archived": cfg.archived, "unsupported": dict(cfg.unsupported),
            "display": dict(zip(TABLE1_COLUMNS, row))})
        assert row[0] == cfg.id
    return configurations


def render_entities(V: Views, published: bool = False, rendered_at: str | None = None, commit: str | None = None) -> dict[str, Any]:
    """The render as entities a site can chart and follow across renders by id (schema :data:`ENTITIES_SCHEMA`):

    * ``subcircuits`` -- one per subcircuit, a fully bound subcircuit template: ``subcircuit_template`` (``gemm-coordinate``)
      with its bound ``parameters`` (``K``; ``semantics``, the exact instruction), ``legacy_target`` (the target name), the
      census hardware line (id, short name, full device, peak), datatype (id and display), tensor-core step, frozen instance
      set and native N;
    * ``backend_families`` -- A-GKR, B-Ligero, C-Flock, D-SP1, each with its markdown column header as ``column``;
    * ``backends`` -- the primary shape of Table 1's claims (:func:`backends`): one per backend family and interaction mode
      (``A-interactive``, ``A-fs``, ...), its claims written out in full;
    * ``statements`` -- one per subcircuit and commitment scheme (``<subcircuit>+<scheme>``): the full committed relation a
      backend lowers and proves, with its scheme's assumptions; a result's full assumption set is its backend's plus these;
    * ``schemes`` -- each commitment scheme's id, name, one-line description and the hash assumptions it rests on, from the
      configurations that bind it (:func:`scheme_entities`); a result's ``scheme`` is one of these ids;
    * ``results`` -- every admissible backend's best result per subcircuit and scheme: its backend and backend family, its
      row ``key`` (backend, statement, input set, prover hardware), ``hardware`` (the semantics chip, which
      supplies N) and ``prover_hardware`` (where the prover ran, which supplies P), overhead, the phases
      that sum to it, one-time setup apart, state (provisional, algebraic, verified, under re-verification), the artifact
      and the render time;
    * ``best_standard_hash`` -- per backend family and subcircuit, the lowest-overhead cell on standard hashes only;
    * ``tables`` -- the printed layout: every Table 1 / 2 / 3 cell's exact display string, numeric value, marks and footnote;
    * ``workload_headline`` -- per census workload and backend, the share of the served run's native work its cleared cells
      cover and the work-weighted overhead over that share, in both bases, uncovered subcircuits listed with why
      (:mod:`verity_numerical.bench.headline`); ``None`` under the legacy rules;
    * ``claims`` -- the ``verity.core.claims`` entries cited (the site imports these); ``census`` -- the ``benchmarks/census/`` entries as read
      at render time; ``provenance`` -- render time, verity commit, rules, banner.

    Ids: subcircuits by their census id, backends ``<letter>-<mode>`` (or a composed backend's own id), statements
    ``<subcircuit>+<scheme>``, results ``<backend>@<statement>@<input set>@<prover hardware>``, census entries by their
    census id."""

    rendered_at = rendered_at or _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    grid = table2_grid(V)
    cfg_of = {c.id: c for c in V.configurations}
    sets = {s.target: s.id for s in census.INPUT_SETS.values()}
    circuits = []
    for target in TARGETS.values():
        N = native(target)
        peak = contract.native_peak_of(target)
        dt = census.DATATYPES[peak.dtype]
        sub = census.subcircuit_of(target.name)
        circuits.append({"id": sub.id, "display": sub.display, "subcircuit_template": sub.template, "parameters": vars(sub.parameters), "legacy_target": target.name,
                         "hardware": peak.id, "hardware_name": peak.name, "device": peak.device,
                         "peak": {"value": peak.flop_per_second, "unit": "FLOP/s", "display": T._fmt_tflops(peak.flop_per_second)},
                         "dtype": peak.dtype, "dtype_display": dt.display, "hardware_display": f"{peak.name} · {dt.display}",
                         "accumulate": census.SUBCIRCUIT_TEMPLATES[sub.template].accumulate, "step": target.transition_unit,
                         "status": target.status, "input_set": sets.get(target.name),
                         "input_set_provenance": census.INPUT_SETS[sets[target.name]].provenance if target.name in sets else None,
                         "fidelity_notes": [{"note": n.note, "sources": [vars(x) for x in n.sources]} for n in sub.fidelity_notes],
                         "native": {"value": N["instances_per_second"], "unit": "instances/s", "display": _native_text(N),
                                    "peak_flop_per_second": N["peak_flop_per_second"], "ops_per_instance": N["ops_per_instance"],
                                    "work_model": N["work_model"]}})

    for ln in V.lines:
        sub_id = ln.get("subcircuit")
        if not sub_id or any(c["id"] == sub_id for c in circuits):
            continue
        target, known = LINES[ln["line"]], census.SUBCIRCUITS.get(sub_id)
        peak = contract.native_peak_of(target)
        dt = census.DATATYPES[peak.dtype]
        N = ln["native"]
        circuits.append({"id": sub_id, "display": known.display if known else sub_id, "subcircuit_template": known.template if known else sub_id.split("/")[0],
                         "parameters": vars(known.parameters) if known else {"K": subcircuit_k(sub_id)}, "legacy_target": None,
                         "hardware": peak.id, "hardware_name": peak.name, "device": peak.device,
                         "peak": {"value": peak.flop_per_second, "unit": "FLOP/s", "display": T._fmt_tflops(peak.flop_per_second)},
                         "dtype": peak.dtype, "dtype_display": dt.display, "hardware_display": f"{peak.name} · {dt.display}",
                         "accumulate": census.SUBCIRCUIT_TEMPLATES[known.template].accumulate if known else "fp32",
                         "step": target.transition_unit if is_gemm(sub_id) else None, "status": target.status, "input_set": None,
                         "input_set_provenance": None,
                         "fidelity_notes": [{"note": n.note, "sources": [vars(x) for x in n.sources]} for n in (known.fidelity_notes if known else [])],
                         "native": {"value": N["instances_per_second"], "unit": "instances/s", "display": _native_text(N),
                                    "peak_flop_per_second": N["peak_flop_per_second"], "ops_per_instance": N["ops_per_instance"],
                                    "work_model": N["work_model"]}})

    configurations = configuration_records(V)

    backend_by_cfg = {c.id: backend_of(c) for c in V.configurations}
    sets_by_ref = {(x.dataset, x.tier): x for x in census.INPUT_SETS.values()}

    backend_list = backends(configurations)
    backend_assumptions = {b["id"]: b["assumptions"] for b in backend_list}
    scheme_list = scheme_entities(V.configurations)
    scheme_by_id = {x["id"]: x for x in scheme_list}

    def sub_of(target_name: str | None, sub: str | None) -> str | None:
        return sub or subcircuit_id(target_name)

    def statement_id(target_name: str, legacy_scheme: str | None, sub: str | None = None) -> str:
        return f"{sub_of(target_name, sub)}+{scheme_id(legacy_scheme)}"

    def prover_of(c: ViewCell) -> str | None:
        dtype = contract.native_peak_of(LINES[c.line]).dtype
        return next((h.id for h in census.HARDWARE.values() if h.dtype == dtype and T.same_sku(V.prover_gpu.get(c.artifact), h.device)), None)

    def key_of(target_name: str, legacy_scheme: str | None, prover: str | None, own: str | None = None) -> dict[str, Any]:
        """A row key: the statement (subcircuit and scheme, with the subcircuit's template and bound parameters) and the
        prover hardware."""
        sub = census.SUBCIRCUITS.get(own) if own else census.subcircuit_of(target_name)
        return {"statement": statement_id(target_name, legacy_scheme, own), "subcircuit": own if sub is None else sub.id,
                "subcircuit_template": sub.template if sub else own.split("/")[0],
                "parameters": vars(sub.parameters) if sub else {"K": subcircuit_k(own)}, "scheme": scheme_id(legacy_scheme), "prover_hardware": prover}

    def registered_set(c: ViewCell) -> tuple[str, Mapping[str, Any]] | None:
        keys = (c.instances.get("set_art"), c.instances.get("content_digest"), c.instances.get("manifest_sha256"))
        return next((V.input_sets[k] for k in keys if isinstance(k, str) and k in V.input_sets), None)

    def input_set_id(c: ViewCell) -> str | None:
        if c.subcircuit:
            hit = registered_set(c)
            return hit[0] if hit else None
        return (sets_by_ref.get((c.instances.get("dataset"), c.instances.get("tier"))) or SimpleNamespace(id=None)).id

    def result_key(c: ViewCell) -> str:
        """backend x statement x input set x prover hardware; the typed circuit statement is its own statement, so a typed
        result never stands in for a flat one (red-team-flock-3's Q4)."""
        typed = "+types" if c.configuration in TYPED_CONFIGS else ""
        return (f"{backend_by_cfg.get(c.configuration or '') or slug(c.column)}@{statement_id(c.line, c.scheme, c.subcircuit)}{typed}"
                f"@{input_set_id(c)}@{prover_of(c)}")

    def input_set_of(c: ViewCell) -> dict[str, Any] | None:
        if c.subcircuit:
            hit = registered_set(c)
            return {"id": hit[0], "set": hit[1].get("set"), "provenance": hit[1].get("source"), "range": c.instances.get("range"),
                    "how": "registered"} if hit else None
        x = sets_by_ref.get((c.instances.get("dataset"), c.instances.get("tier")))
        if x is None:
            return None
        how, _ = instance_range(LINES[c.line], c.instances) if c.instances else (None, None)
        return {"id": x.id, "provenance": x.provenance, "range": c.instances.get("range"), "how": how or "equivalent"}

    def phases_of(c: ViewCell) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        st, r3 = stages(c), table3_row(c)
        share = (lambda v: v / c.overhead) if c.overhead else (lambda v: None)  # noqa: E731
        out = [{"id": "commitment", "kind": "stage", "value": st["commitment"], "share": share(st["commitment"]), "display": r3["commitment"]},
               {"id": "interaction", "kind": "stage", "value": st["interaction"], "share": share(st["interaction"]), "display": r3["interaction"]}]
        out += [{"id": b, "kind": "stage", "value": v, "share": share(v), "display": r3[b]} for b, v in st["buckets"].items() if v is not None]
        out += [{"id": name, "kind": "span", "covers": list(j["covers"]), "from": j["covers"][0], "to": j["covers"][-1],
                 "value": j["overhead"], "share": share(j["overhead"]), "display": r3[j["covers"][0]]} for name, j in st["joints"].items()]
        other = {"value": st["other"], "share": share(st["other"]) if st["other"] is not None else None}
        if st["other"] is not None:
            sh = other["share"] or 0.0
            clamped = -OTHER_CLAMP <= sh < 0
            out.append({"id": "other", "kind": "stage", "value": 0.0 if clamped else st["other"], "share": 0.0 if clamped else sh,
                        "display": r3["other"], "raw": st["other"], "clamped": clamped, "flagged": sh < -OTHER_CLAMP})
        return out, other

    def result_of(c: ViewCell) -> dict[str, Any]:
        P, ix = c.throughput, c.throughput["interaction"]
        su = stages(c)["setup"]
        phases, other = phases_of(c)
        cfg = cfg_of.get(c.configuration or "")
        per_proof = V.security_of.get(c.artifact, {}).get("per_proof_total_log2")
        peak = contract.native_peak_of(LINES[c.line])
        prover, backend, stmt = prover_of(c), backend_by_cfg.get(c.configuration or ""), statement_id(c.line, c.scheme, c.subcircuit)
        stmt_assumptions = scheme_by_id[scheme_id(c.scheme)]["assumptions"] if scheme_id(c.scheme) in scheme_by_id else []
        return {
            "id": result_key(c), "key": {"backend": backend, "statement": stmt, "input_set": input_set_id(c), "prover_hardware": prover},
            "backend": backend, "backend_family": _fam(c.column), "statement": stmt, "implementation": c.configuration,
            "subcircuit": sub_of(c.line, c.subcircuit), **({"variables": dict(c.variables)} if c.variables else {}),
            **({"key_class": {"T": V.key_class_of[c.artifact]["range"], "pin": V.key_class_of[c.artifact]["pin"],
                              "key_counts": {str(t): x["heads"] for t, x in V.key_class_of[c.artifact]["credit"].items()},
                              "not_credited": V.key_class_of[c.artifact]["skipped"]}} if c.artifact in V.key_class_of else {}),
            "scheme": scheme_id(c.scheme), "legacy_scheme": c.scheme,
            "hardware": peak.id, "prover_hardware": prover, "input_set": input_set_of(c),
            "assumptions": {"backend": backend_assumptions.get(backend or "", []), "statement": stmt_assumptions,
                            "all": list(dict.fromkeys([*backend_assumptions.get(backend or "", []), *stmt_assumptions]))},
            "best": any(b.artifact == c.artifact and b.line == c.line and b.scheme == c.scheme and b.subcircuit == c.subcircuit for b in V.cells),
            "artifact": c.artifact, "rendered_at": rendered_at, "network": census.REFERENCE_NETWORK.id,
            "overhead": {"value": c.overhead, "display": T.fmt_overhead(c.overhead)},
            "N": {"value": c.native, "unit": "instances/s", "work_model": WORK_MODEL.split(":")[0], "peak": peak.id},
            "P": {"value": P["instances_per_second"], "unit": "instances/s", "network": census.REFERENCE_NETWORK.id},
            "phases": phases, "other_raw": other["value"], "serving_overhead": overheads(c)[0], "proving_overhead": overheads(c)[1],
            "setup": {"seconds": su["seconds"], "amortized_overhead": su["amortized"], "instances": su["instances"],
                      "display": table3_row(c)["setup"]} if su else None,
            "soundness": {"cell_log2": c.bound_log2, "cell_bits": -c.bound_log2 if c.bound_log2 is not None else None,
                          "per_proof_log2": float(per_proof) if _num(per_proof) else None, "proofs": c.batching.get("proofs")},
            "state": {"provisional": bool(c.provisional), "provisional_reason": c.provisional, "algebraic": bool(c.algebraic),
                      "algebraic_hashes": list(c.algebraic), "verified": bool(c.legacy.verified_by), "verified_by": c.legacy.verified_by,
                      "under_reverification": c.legacy.finding},
            "marks": _mark_names(c), "zk_class": zk_class(cfg) if cfg else None, "domain": V.domain_of.get(c.artifact),
            **({"circuit": V.circuit_of[c.artifact]} if c.artifact in V.circuit_of else {}),
            "throughput": {"inputs": P["instances"], **{k: P[k] for k in ("instances_per_second", "end_to_end_s", "commit_s", "proving_s", "interaction_s")}},
            "batching": dict(c.batching), "instances": dict(c.instances),
            "interaction": interaction_fields(ix, P, V.interaction_raw.get(c.artifact, {})),
            "interaction_attempts": interaction_attempts(V, c, ix),
            "overhead_at": {n.id: c.native * (P["commit_s"] + interaction_time(ix, n.rtt_s, n.bandwidth_bps)) / P["instances"]
                            for n in (census.REFERENCE_NETWORK, *census.SENSITIVITY_NETWORKS)} if not V.rules.legacy else None}

    results = [result_of(c) for c in best_per_key(V.admissible, result_key)]

    table2_lines, footnotes = [], []

    def footnote(text: str) -> int:
        footnotes.append({"number": len(footnotes) + 1, "text": text})
        return len(footnotes)

    for g in grid:
        ln = g["line"]
        line_prover = contract.native_peak_of(LINES[ln["line"]]).id
        own = ln.get("subcircuit")
        line = {"statement": statement_id(ln["line"], ln["scheme"], own), "subcircuit": sub_of(ln["line"], own), "scheme": scheme_id(ln["scheme"]),
                "legacy_scheme": ln["scheme"], "key": key_of(ln["line"], ln["scheme"], line_prover, own), "label": g["label"],
                "native": {"display": g["native"]["display"], "value": ln["native"]["instances_per_second"], "footnote": footnote(g["native"]["note"]),
                           "work_model": WORK_MODEL.split(":")[0], "peak": contract.native_peak_of(LINES[ln["line"]]).id},
                "cells": {}}
        for col, gc in g["cells"].items():
            c = gc["cell"]
            line["cells"][_fam(col)] = {"display": gc["display"], "value": c.overhead if c else None, "marks": gc["marks"], "state": gc["state"],
                                        "footnote": footnote(gc["note"]), "result": result_key(c) if c else None,
                                        "backend": backend_by_cfg.get(c.configuration or "") if c else None}
        table2_lines.append(line)

    families = sorted(({**dataclasses.asdict(f), "id": _fam(f.id), "letter": _fam(f.id)[0], "column": column_header(f.id), "declared_classes": list(f.declared_classes),
                        "display": _family_text(f)} for f in FAMILIES), key=lambda f: f["id"])
    columns = sorted((_fam(c) for c in V.columns()), key=lambda c: (c.split(" ")[0], c))
    def labels(ids: Sequence[str]) -> str:
        return ", ".join(claims_registry.get(i).label for i in ids)

    development = {_fam(f.id): f.development for f in FAMILIES}
    for b in backend_list:
        b["development"] = development.get(b["backend_family"], "active")
    table1 = [{"backend": b["id"], "backend_family": b["backend_family"], "mode": b["mode"], "development": b["development"],
               "cells": {"Backend": b["display"], "Backend family": b["backend_family"], "Mode": b["mode"], "Development": b["development"],
                         "Guarantees": labels(b["properties"]),
                         "Assumptions": labels(b["assumptions"]),
                         "Unmapped": "; ".join(t for c in b["claims"] for x in (c["property"], *c["assumptions"]) if x["unmapped"] for t in x["texts"])}}
              for b in backend_list]
    statements = {}
    for ln in V.lines:
        own = ln.get("subcircuit")
        sid = statement_id(ln["line"], ln["scheme"], own)
        sch = scheme_by_id.get(scheme_id(ln["scheme"]) or "")
        disp = ln.get("subcircuit_display") if own else census.subcircuit_of(ln["line"]).display
        statements[sid] = {"id": sid, "display": f"{disp} · {sch['name'] if sch else 'no scheme'}",
                           "subcircuit": sub_of(ln["line"], own), "scheme": scheme_id(ln["scheme"]),
                           "assumptions": sch["assumptions"] if sch else [], "assumption_details": sch["assumption_details"] if sch else []}
    phase_cols = [pid for pid, _ in PHASES]
    from verity_numerical.bench import headline as H  # noqa: PLC0415  (headline reads these views)

    workload_headline = None if V.rules.legacy else H.to_json(H.headline(V), {r["artifact"]: r["id"] for r in results})
    for b in backend_list:                    # Table 1: the operand domains its results cover (Daniel 2026-09-26: total by default)
        b["domains"] = sorted({r["domain"] for r in results if r.get("backend") == b["id"] and r.get("domain")})
    return {
        "schema": ENTITIES_SCHEMA,
        "provenance": {"rendered_at": rendered_at, "verity_commit": commit if commit is not None else verity_commit(), "rules": dataclasses.asdict(V.rules),
                       "published": published, "banner": PUBLISHED if published else PREVIEW},
        "axes": {"subcircuits": [c["id"] for c in circuits], "schemes": [scheme_id(s.id) for s in SCHEMES if s.admissible],
                 "statements": list(statements), "backends": [b["id"] for b in backend_list], "backend_families": columns,
                 "phases": phase_cols, "networks": [n.id for n in (census.REFERENCE_NETWORK, *census.SENSITIVITY_NETWORKS)]},
        "census": {name: {e["id"]: e for e in census.raw(name)} for name in ("hardware", "datatypes", "networks", "subcircuits", "input_sets", "workloads")},
        "claims": {i: dataclasses.asdict(claims_registry.get(i)) for i in sorted({*(i for b in backend_list for k in ("properties", "assumptions")
                                                                                     for i in b[k]),
                                                                                   *(i for st in statements.values() for i in st["assumptions"])})},
        "subcircuits": circuits,
        "schemes": scheme_list,
        "statements": list(statements.values()),
        "backend_families": families,
        "backends": backend_list,
        "results": results,
        "best_standard_hash": best_standard_hash(results),
        "tables": {
            "table1": {"columns": ["Backend", "Backend family", "Mode", "Guarantees", "Assumptions", "Unmapped"], "rows": table1},
            "table2": {"columns": ["native", *columns], "column_labels": {_fam(c): column_header(c) for c in V.columns()}, "lines": table2_lines,
                       "footnotes": footnotes},
            "table3": {"columns": [*phase_cols, "total", "setup"], "labels": dict(PHASES),
                       "rows": [{"result": result_key(c), "key": key_of(c.line, c.scheme, prover_of(c), c.subcircuit), "backend_family": _fam(c.column),
                                 "backend": backend_by_cfg.get(c.configuration or ""), "label": _cell_name(c), "cells": table3_row(c)}
                                for c in V.cells]},
        },
        "workload_headline": workload_headline,
        "below_bar": sorted(V.below_bar, key=lambda e: (e["backend_family"] or "", str(e["definition"]), e["B"] or 0, e["artifact"])),
        "rejected": [{"artifact": r["artifact"], "implementation": r["configuration"], "backend_family": _fam(r["family"]),
                      "subcircuit": r.get("subcircuit") or subcircuit_id(r["line"]), "scheme": scheme_id(r.get("scheme")), "reason": r["codes"][0][0], "text": r["codes"][0][1]} for r in V.rejected],
    }


# --- CLI --------------------------------------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, metavar="ROOT", help="store root (default $RESEARCH_STORE or ~/.research/store)")
    ap.add_argument("--format", choices=("md", "json", "entities"), default="md",
                    help="md; json (the raw views); entities (the docs site's render: subcircuits, statements, backends, results, the printed tables)")
    ap.add_argument("--rules", choices=tuple(RULES), default=SPEC.name)
    ap.add_argument("--security-bits", type=float, default=None, metavar="N", help="filter to a whole-proof bound of 2^-N or better")
    ap.add_argument("--include-archived", action="store_true")
    ap.add_argument("--unvalidated-interaction", action="store_true",
                    help="preview: do not require interactive results to carry the interaction record and meet the tolerance")
    ap.add_argument("--standard-hashes-only", action="store_true", help="drop backends and statements with an algebraic hash in the soundness argument")
    ap.add_argument("--preview-one-sided-interaction", action="store_true",
                    help="PREVIEW (pending decision, never published): the interaction check passes when measured <= 1.1 x the model")
    ap.add_argument("--preview-any-hardware-headline", action="store_true",
                    help="PREVIEW (pending decision, never published): the workload headline counts cleared cells on any GPU")
    ap.add_argument("--preview-strict-verifier", action="store_true",
                    help="PREVIEW (Daniel's standing rule, not yet published): a cell whose verifier evaluates any part covers nothing")
    ap.add_argument("--preview-count-omitted", action="store_true",
                    help="PREVIEW (pending decision, never published): the headline counts cells whose omitted label names verifier-evaluated parts")
    ap.add_argument("--preview-cached-weights", action="store_true",
                    help="PREVIEW (pending decision, never published): non-GEMM native N on activations only (weights/tables cached)")
    ap.add_argument("--parity", action="store_true", help="compare the legacy-rules views with a fresh `tables` build; exit 1 on a difference")
    ap.add_argument("--snapshot", default=None, metavar="NAME", help="with --parity: also compare with this `tables --snapshot`'s members")
    ap.add_argument("--take-snapshot", action="store_true",
                    help="with --parity: first take the `tables --snapshot` (tables-<UTC date>) to compare with")
    ap.add_argument("--record", action="store_true", help=f"with --parity: store the parity render as a {PARITY_KIND} artifact (its id on stderr)")
    ap.add_argument("--published", action="store_true", help="the published render (after the switch): the published banner, not the preview's")
    ns = ap.parse_args(argv)
    from research.store.local import LocalStore  # noqa: PLC0415  (only the CLI needs the store package)

    st = LocalStore(ns.root)
    if ns.parity:
        frozen, snap = T.build(st), None
        if ns.take_snapshot:
            if not frozen.cells:
                print("views: no populated tables cell, no snapshot to take", file=sys.stderr)
                return 2
            art = T.take_snapshot(st, frozen)
            snap = (T.snapshot_name(), art, T.cell_artifacts(frozen))
        elif ns.snapshot:
            rows = [r for r in st.index.select_artifacts(kind="dataset-snapshot/v1") if (D._doc(st, r).get("meta") or {}).get("name") == ns.snapshot]
            if not rows:
                print(f"views: no dataset-snapshot/v1 named {ns.snapshot!r}", file=sys.stderr)
                return 2
            snap = (ns.snapshot, rows[-1]["id"], list((D._doc(st, rows[-1]).get("refs") or {}).get("members") or []))
        rec = parity_record(st, snap, frozen)
        if ns.record:
            refs = {k: v for k, v in (("snapshot", rec["snapshot_artifact"]), ("cells", sorted({c["artifact"] for c in rec["cells"]}))) if v}
            rec["artifact"] = st.put_artifact(PARITY_KIND, {k: v for k, v in rec.items() if k != "snapshot_artifact"}, refs)
            print(f"parity recorded as {rec['artifact']}" + (f" (snapshot {rec['snapshot']} {rec['snapshot_artifact']})" if snap else ""), file=sys.stderr)
        if ns.format == "json":
            print(json.dumps(rec, indent=2))
        else:
            print("parity: ok (the legacy-rules views select exactly today's cells)" if rec["ok"] else "\n".join(["parity: DIFFERENT", *rec["differences"]]))
            for c in rec["cells"]:
                print(f"  {c['line']} · {c['column']}: {c['artifact']} {c['printed']}")
        return 0 if rec["ok"] else 1
    rules = RULES[ns.rules]
    if ns.security_bits is not None:
        rules = dataclasses.replace(rules, name=f"{rules.name}@{ns.security_bits:g}", security_log2=-ns.security_bits)
    if ns.include_archived:
        rules = dataclasses.replace(rules, include_archived=True)
    if ns.unvalidated_interaction:
        rules = dataclasses.replace(rules, name=f"{rules.name}+unvalidated-interaction", interaction_validated=False)
    if ns.preview_one_sided_interaction:
        rules = dataclasses.replace(rules, name=f"{rules.name}+preview-one-sided-interaction", interaction_one_sided=True)
    if ns.preview_any_hardware_headline:
        rules = dataclasses.replace(rules, name=f"{rules.name}+preview-any-hardware-headline", headline_any_hardware=True)
    if ns.preview_strict_verifier:
        rules = dataclasses.replace(rules, name=f"{rules.name}+preview-strict-verifier", headline_strict_verifier=True)
    if ns.preview_count_omitted:
        rules = dataclasses.replace(rules, name=f"{rules.name}+preview-count-omitted", headline_count_omitted=True)
    if ns.preview_cached_weights:
        rules = dataclasses.replace(rules, name=f"{rules.name}+preview-cached-weights", native_cached=True)
    if ns.standard_hashes_only:
        rules = dataclasses.replace(rules, name=f"{rules.name}+standard-hashes", standard_hashes=True)
    V = build(st, rules)
    if ns.format == "json":
        print(json.dumps(render_json(V, ns.published), indent=2, sort_keys=True, default=str))
    elif ns.format == "entities":
        print(json.dumps(render_entities(V, ns.published), indent=2, default=str))
    else:
        print(render_markdown(V, ns.published), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
