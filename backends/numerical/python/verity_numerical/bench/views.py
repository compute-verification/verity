"""The tables of the approved spec (``kb/TABLES.md``, approved 2026-09-25): a labelled PREVIEW beside the frozen canonical tables.

    python -m verity_numerical.bench.views --root <store root> [--format md|json] [--rules spec|unfiltered|legacy]
                                           [--security-bits N] [--include-archived]
    python -m verity_numerical.bench.views --root <store root> --parity [--snapshot NAME]

Until the switch (a labelled 6 PM PT digest) the published tables stay :mod:`verity_numerical.bench.tables` and
:mod:`~verity_numerical.bench.drilldown`; these views are previews and are never shown as the published numbers.

* **Table 1** -- backend configurations and their security, one entry per backend family followed by its configurations
  (:data:`CONFIGURATIONS`): the statement (the full committed relation or a weaker one) and what the verifier sees of x, W and
  y; the commitment scheme (:data:`SCHEMES`); every hash of the soundness argument with its role, property and class; ZK class
  and verifier model; soundness with the whole-proof bound at the batching of each cell it holds and the dominant term; the
  proximity bounds (regime, proven or conjectured); transferability; assumptions; the pin; ``archived``; its cells (computed).
* **Table 2** -- end-to-end overhead ``N / P`` per subcircuit, hardware class and admissible commitment scheme (one line per
  ``verity.verification.target.TARGETS`` entry and scheme some configuration declares), a native column, one column per
  backend family (:data:`FAMILIES`).  ``N`` is the
  compute-bound datasheet estimate (:func:`native`); ``P`` is instances per second end to end, committing the batch's values and
  then proving (:func:`throughput`).  A cell is the family's best admissible configuration (highest ``P``, ties to the lower
  artifact id) that proves the full relation over a core-defined scheme and passes the view's filters (:class:`Rules`: by
  default a whole-proof bound of 2^-128), marked ``(prov.)`` while its configuration's statement has no red-team clearance on
  that subcircuit and ``(alg.)`` when its soundness argument uses an algebraic hash (flagged in Table 1).
* **Table 3** -- where the time goes: seconds per instance of commitment and of each proving bucket, with shares of end to end,
  and the cell split into serving overhead (commitment x N) and proving overhead (proving x N).

Every rejected result is listed with its first reason (:data:`REASON_LETTERS`, the spec's order); ``--format json`` has them all.

The v1 adapter: ``bench-result/v1`` results carry no configuration id, so their configuration is the ``drilldown.VARIANTS`` entry
that recognises them (:func:`configuration_of`) and their family the ``tables.CANDIDATES`` entry; name recognisers live nowhere
else in these views.  A result names its commitment time as the measurement :data:`COMMIT` and its
protocol as the ``protocol`` / ``sweep`` blocks (``benchmark-standard`` §6.1, ``bench-result/v2``); no v1 result records the
protocol, so no v1 result meets it (reason M).

``--rules legacy`` renders with today's parameters (2^-128 on target and achieved, B = 4096, K = 1536, the frozen instance range,
the bare and ``+ in-proof hash`` columns, proving time only): the switch's parity gate, which ``--parity`` checks against a
fresh :func:`tables.build` of the same store (and, with ``--snapshot NAME``, against that ``tables --snapshot``'s members).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from verity.verification.target import BF16_HOPPER, FP8_ADA_MMA, FP8_HOPPER_WGMMA, NVFP4_SM120, TARGETS, Target
from verity_numerical.bench import contract
from verity_numerical.bench import drilldown as D
from verity_numerical.bench import tables as T

__all__ = ["COMMIT", "CONFIGURATIONS", "FAMILIES", "LEGACY", "REASON_LETTERS", "SPEC", "UNFILTERED", "Configuration", "Family",
           "Rules", "Views", "build", "configuration_of", "main", "native", "parity", "render_json", "render_markdown", "throughput"]

PREVIEW = ("PREVIEW of the approved tables spec (kb/TABLES.md): not the published tables. The published numbers switch once, in a "
           "6 PM PT digest labelled as a change of method; until then the canonical tables are `verity_numerical.bench.tables`.")

#: Commitment seconds per batch: the warm median of committing the batch's values afresh, as lane hash-commit writes it (with
#: ``commit.cold_seconds``, ``commit.reps`` and ``e2e.seconds`` = this + ``t.total``).
COMMIT = "commit.seconds"
#: v1 in-proof-hash results record their commitment once, before the timed runs (possibly a tree cache load): shown, never counted.
COMMIT_V1 = "relation.hash.commit_seconds"
MIN_RUNS = 5
#: How Table 1 flags a configuration whose soundness argument uses an algebraic hash; its Table 2 cells carry ``(alg.)``.
ALGEBRAIC_FLAG = "algebraic hash, not for highest-stakes use"

#: The spec's reject reasons, in the order they are checked; ``B`` exists only under the legacy rules (today's batch and K).
REASON_LETTERS: dict[str, str] = {
    "R": "weaker statement than the full relation", "Z": "not the family's declared class",
    "S": "commitments on a scheme verity.commitments does not define", "L": "fails the view's security filter",
    "U": "not independently verified", "F": "proof files not preserved", "I": "another instance set", "K": "prover not on the line's GPU",
    "P": "phase-sum", "M": "protocol not met", "D": "capped by a red-team verdict", "X": "other", "B": "legacy: batch != 4096 or K != 1536"}
_ORDER = {k: i for i, k in enumerate(REASON_LETTERS)}


# --- commitment schemes -----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Scheme:
    """A commitment scheme with the hash of its leaves (one Table 2 line per subcircuit, hardware class and admissible scheme).
    ``core`` is where ``verity.commitments`` defines it (``None``: not defined there, so inadmissible); ``standard``: built only on
    standard hashes (SHA-256, BLAKE3) -- an algebraic scheme is admissible and flagged (:data:`ALGEBRAIC_FLAG`)."""

    id: str
    label: str
    core: str | None
    standard: bool

    @property
    def admissible(self) -> bool:
        return self.core is not None


SCHEMES: tuple[Scheme, ...] = (
    Scheme("frame-v3/sha256", "frame-v3 · SHA-256", "verity.commitments.merkle (frame-v3 word leaves; the SHA-256 row leaf is core-schemes')", True),
    Scheme("frame-v3/blake3-row", "frame-v3 · keyed-BLAKE3 rows",
           "verity.commitments.merkle trees; the keyed-BLAKE3 row-leaf schema is promoted from backends/direct/ligero/leaf/blake3.py (core-schemes)", True),
    Scheme("vllm-v1", "vllm-v1 · SHA-256", None, True),
    Scheme("frame-v3/poseidon2-row", "frame-v3 · Poseidon2 rows", "verity.commitments.rowleaf (leaf/v2h)", False),
    Scheme("frame-v3/poseidon2-tile", "frame-v3 · Poseidon2 64x64 tiles", None, False),
    Scheme("frame-v3/ajtai-row", "frame-v3 · Ajtai rows", None, False),
)
_SCHEME = {s.id: s for s in SCHEMES}


# --- Table 1: families and configurations ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Family:
    """One Table 2 column: the backend family, its declared class (the ``tables.CANDIDATES`` entry's) and what it is."""

    id: str
    declared_classes: tuple[str, ...]
    description: str
    setup: str
    interaction: str


FAMILIES: tuple[Family, ...] = tuple(Family(c.id, c.proof_classes, c.description, c.setup, c.interaction) for c in T.CANDIDATES)
_FAMILY = {f.id: f for f in FAMILIES}
_CANDIDATE = {c.id: c for c in T.CANDIDATES}


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
    theorem) or conjectured."""

    component: str
    regime: str
    basis: str

    def text(self) -> str:
        return f"{self.component}: {self.regime}, {self.basis}"


FULL = "full relation"
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
_AGKR_DOMINANT = "IRS query term (2^-130.2 at t = 192) and the LogUp identity term; the (n+1)/|F|^sigma term grows with the committed length n"


@dataclass(frozen=True)
class Configuration:
    """One Table 1 row.  ``id`` is the ``drilldown.VARIANTS`` entry it adapts (v1 results carry no configuration id); the ZK and
    per-proof soundness words are that entry's.  ``scheme`` is the :data:`SCHEMES` id its statements bind (``None``: the statement
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
            return f"{self.id} binds no scheme of verity.commitments"
        if s.core is None:
            return f"{self.id} commits with {s.label}, which verity.commitments does not define yet"
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
    _cfg("SP1 stock (relation-only / bare guest)", _WEAK_PRIVATE, "x, W guest input (unbound)", None, "none for the operands",
         _SP1_HASHES, _SP1_DOMINANT, _SP1_PROX, "transferable (STARK, Fiat–Shamir)", _SP1_ASSUME,
         "verifying key (guest ELF digest), sp1-sdk =6.4.0"),
    _cfg("SP1 stock, FRI query count raised (otherwise unmodified)", _WEAK_PRIVATE, "x, W guest input (unbound)", None,
         "none for the operands", _SP1_HASHES, _SP1_DOMINANT + "; the FRI query term is 2^-134.7", _SP1_PROX,
         "transferable (STARK, Fiat–Shamir)", _SP1_ASSUME, "verifying key (guest ELF digest), sp1 6.6.0 with core FRI queries 175"),
    _cfg("SP1 precompile (TC_DOT chip)", _WEAK_PRIVATE, "x, W guest input (unbound)", None, "none for the operands", _SP1_HASHES,
         _SP1_DOMINANT, _SP1_PROX, "transferable (STARK, Fiat–Shamir)", _SP1_ASSUME + "; the TC_DOT chip's constraints",
         "verifying key plus the TC_DOT fork commit"),
)
_CONFIG = {c.id: c for c in CONFIGURATIONS}
assert set(_CONFIG) == set(D._BY_ID), "every drilldown.VARIANTS entry is a declared configuration"

TABLE1_COLUMNS = ("Configuration", "Flag", "Statement", "x, W, y", "Commitment scheme", "Hashes (role; property; class)", "ZK and verifier model",
                  "Soundness (whole-proof bound at its cells; dominant term)", "Proximity bounds", "Transferability", "Assumptions",
                  "Pin", "Archived", "Cells")


def configuration_of(meta: Mapping[str, Any], configurations: Mapping[str, Configuration] | None = None) -> Configuration | None:
    """The declared configuration of a ``bench-result/v1`` result (the v1 adapter: the ``drilldown.VARIANTS`` entry that recognises
    it), or ``None``."""

    v = D.classify(D.features(meta))
    return (configurations or _CONFIG).get(v.id) if v else None


# --- native throughput N ----------------------------------------------------------------------------------------------------------

#: The work model until ``verity.silicon`` declares per-primitive costs: a GemmCoordinate<K> is K/16 tensor-core steps of 16
#: multiply-adds (32 FLOP each) plus one FP32 -> operand-dtype conversion, which never binds; the peaks are ``Target.native_peak``.
WORK_MODEL = "work-model/v0: K/16 tensor-core steps x 32 FLOP (+1 conversion, not binding); peaks Target.native_peak"


def native(target: Target, K: int = T.K) -> dict[str, Any]:
    """``N`` for the target's subcircuit on its hardware class: instances per second = 1 / max over units of ops / peak, the
    tensor cores being the busiest unit of a GEMM coordinate; with the busiest unit, the peak entry and the work model."""

    steps, flop = K // 16, (K // 16) * 32
    peak = target.native_peak
    return {"instances_per_second": peak.flop_per_second / flop, "unit": "tensor cores", "ops_per_instance": flop, "steps": steps,
            "dtype": (T.dtype_of(target) or "?").upper(), "peak_flop_per_second": peak.flop_per_second, "device": peak.device,
            "source": peak.source, "work_model": WORK_MODEL}


# --- the view's rules -------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Rules:
    """A view's parameters.  ``security_log2``: the whole-proof bound filter (``None`` = unfiltered); ``standard_hashes``: an
    optional filter (off in the published view) dropping configurations with an algebraic hash anywhere in the soundness
    argument.  ``legacy``: today's
    parameters (the parity render): B = 4096 and K = 1536, the frozen range, security.target also <= the filter, the bare and
    ``+ in-proof hash`` columns instead of the full relation, proving time only, no custody or protocol check."""

    name: str
    security_log2: float | None = -128
    standard_hashes: bool = False
    legacy: bool = False
    include_archived: bool = False


SPEC = Rules("spec")
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


def throughput(meta: Mapping[str, Any], rules: Rules) -> dict[str, Any] | None:
    """``P``: instances per timed run over its end-to-end seconds (committing the batch, :data:`COMMIT`, then proving); under the
    legacy rules proving only.  ``None`` without a positive ``t.total`` and batch."""

    fp = meta.get("workload_fingerprint") if isinstance(meta.get("workload_fingerprint"), Mapping) else {}
    m = T._measurements(meta)
    t, n = m.get(contract.T_TOTAL), fp.get("B")
    if not (_num(t) and t > 0 and _num(n) and n > 0):
        return None
    commit = 0.0 if rules.legacy else (float(m[COMMIT]) if _num(m.get(COMMIT)) else 0.0)
    return {"instances": n, "proving_s": float(t), "commit_s": commit, "end_to_end_s": float(t) + commit,
            "instances_per_second": n / (float(t) + commit), "commit_v1_s": m.get(COMMIT_V1) if _num(m.get(COMMIT_V1)) else None}


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
    statement_labels: list[Any]


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
#: recipe here reproduces the digest ``contract`` pins at n = 4096 (tested).  The A100 set is a fixture of 4,096 instances with
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
    """``(how, None)`` when ``inst`` is the target's declared instance set over [0, n) -- ``how`` is ``"frozen"`` (exactly the
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


def _instances(ctx: Ctx, rules: Rules, equivs: Sequence[T.InstanceEquiv], self_: set[str]) -> list[tuple[str, str]]:
    fp = ctx.meta["workload_fingerprint"]
    want, inst = T.frozen_instances(ctx.target), fp.get("instances")
    if want is None:
        return [("I", f"target {ctx.target.name!r} has no frozen instance set yet")]
    if not isinstance(inst, Mapping):
        return [("I", "instances is null: not measured on the frozen instance set")]
    diff = T.instances_diff(inst, want)
    if not diff:
        return []
    if not rules.legacy:
        how, repeats = instance_range(ctx.target, inst)
        if how is not None:
            return []
        if repeats:
            return [("I", repeats)]
    eq, why = T.instance_equivalence(inst, ctx.target, equivs, self_)
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
    p += _instances(ctx, rules, equivs, self_)
    gpu = _get(fp, "hardware", "gpu") if isinstance(_get(fp, "hardware", "gpu"), Mapping) else None
    if not T.same_sku(gpu.get("name") if gpu else None, ctx.target.native_peak.device):
        p.append(("K", f"same-SKU: prover ran on {(gpu or {}).get('name')!r}, the line's device is {ctx.target.native_peak.device!r}"))
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
    verdicts = T._labels(ctx.labels if rules.legacy else ctx.statement_labels)
    ceiling, verdict = T._redteam_ceiling(verdicts, self_)
    if verdict is not None and isinstance(pc, str) and T.RANK.get(ceiling or "", -1) < T.RANK.get(pc, -1):
        p.append(("D", f"capped: {verdict.by} ({verdict.ref or 'no ref'}, {verdict.ts}) asserts proof_class={ceiling}, below {pc}"))
    state, flab = T.cell_finding(ctx.labels)
    if state == "pulled":
        p.append(("D", f"pulled: {T.finding_note(flab)}"))
    return sorted(p, key=lambda x: _ORDER[x[0]])


def provisional(ctx: Ctx) -> str | None:
    """Why the cell is provisional, or ``None``: no red-team ``proof_class`` verdict on the configuration's statement for this
    subcircuit (any result of the same configuration and target) grants the result's class."""
    self_ = T.producers(ctx.attempt, T._labels(ctx.labels), ctx.meta)
    ceiling, verdict = T._redteam_ceiling(T._labels(ctx.statement_labels), self_)
    if verdict is None:
        return f"no red-team clearance of {ctx.config.id if ctx.config else ctx.family.id}'s statement on {ctx.target.name}"
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

    def to_json(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["legacy"] = self.legacy.to_json()
        return d


@dataclass
class Views:
    rules: Rules
    lines: list[dict[str, Any]] = field(default_factory=list)
    cells: list[ViewCell] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    not_families: dict[str, list[str]] = field(default_factory=dict)
    configurations: tuple[Configuration, ...] = CONFIGURATIONS

    def columns(self) -> list[str]:
        if not self.rules.legacy:
            return [f.id for f in FAMILIES]
        hashed = [f.id + T.AUTH_COLUMN_SUFFIX["included-hash"] for f in FAMILIES
                  if any(ln["cells"].get(f.id + T.AUTH_COLUMN_SUFFIX["included-hash"]) for ln in self.lines)]
        return [f.id for f in FAMILIES] + hashed


def _batching(fp: Mapping[str, Any]) -> dict[str, Any]:
    return {"total": fp.get("B"), "per_proof": fp.get("B_proof"), "proofs": fp.get("N_subbatches"),
            "in_flight": _get(fp, "software", "backend", "pipeline"), "columns_per_proof": _get(fp, "security", "rs_l")}


def build(store: Any, rules: Rules = SPEC, configurations: Sequence[Configuration] = CONFIGURATIONS) -> Views:
    """Read the store, apply :func:`reject_codes` under ``rules`` and assemble Tables 1-3 (rendering is separate)."""

    V = Views(rules=rules, configurations=tuple(configurations))
    cfgs = {c.id: c for c in configurations}
    equivs = T.instance_equivs(store)
    entries: list[Ctx] = []
    for art, meta in T._artifacts(store, T.RESULT_KIND):
        fp = meta.get("workload_fingerprint") if isinstance(meta.get("workload_fingerprint"), Mapping) else {}
        backend = _get(fp, "software", "backend", "name")
        cand = T.candidate_for(backend)
        if cand is None:
            V.not_families.setdefault(backend if isinstance(backend, str) else "(no software.backend.name)", []).append(art)
            continue
        rid, attempt = T._attempt_of(store, art)
        target = TARGETS.get(fp.get("profile")) if isinstance(fp.get("profile"), str) else None
        config = configuration_of(meta, cfgs)
        if target is None:
            V.rejected.append({"artifact": art, "family": cand.id, "configuration": config.id if config else None, "line": None,
                               "attempt": rid, "codes": [("X", f"profile {fp.get('profile')!r} is not a target of verity.verification.target")]})
            continue
        entries.append(Ctx(art, meta, list(store.labels(art)), attempt, target, _FAMILY[cand.id], config, []))
    by_statement: dict[tuple[str | None, str], list[Any]] = {}
    for c in entries:
        by_statement.setdefault((c.config.id if c.config else None, c.target.name), []).extend(c.labels)
    for c in entries:
        c.statement_labels = by_statement[(c.config.id if c.config else None, c.target.name)] if c.config else c.labels

    line_schemes: list[str | None] = [None] if rules.legacy else [
        s.id for s in SCHEMES if s.admissible and any(c.scheme == s.id for c in configurations)]

    def scheme_of(c: Ctx) -> str | None:
        return c.config.scheme if not rules.legacy and c.config is not None and c.config.scheme in line_schemes else None

    Key = tuple[str, str | None, str]
    pools: dict[Key, list[tuple[Ctx, dict[str, Any]]]] = {}
    unfiltered: dict[Key, list[tuple[Ctx, dict[str, Any]]]] = {}
    for c in entries:
        codes = reject_codes(store, c, rules, equivs)
        P = throughput(c.meta, rules)
        key = (c.target.name, scheme_of(c), _column(c, rules))
        if not codes and P is not None:
            pools.setdefault(key, []).append((c, P))
            continue
        if (rules.security_log2 is not None or rules.standard_hashes) and all(k == "L" for k, _ in codes) and P is not None:
            unfiltered.setdefault(key, []).append((c, P))
        rid, _ = T._attempt_of(store, c.art)
        entry = {"artifact": c.art, "family": c.family.id, "configuration": c.config.id if c.config else None, "line": c.target.name,
                 "scheme": key[1], "column": key[2], "attempt": rid, "codes": codes or [("X", "no positive t.total and batch")]}
        if P is not None and not rules.legacy and c.config is not None and c.config.full_relation:
            commit = P["commit_s"] or P["commit_v1_s"] or 0.0
            n = native(c.target)["instances_per_second"]
            entry["end_to_end_preview"] = {"commit_s": commit, "commit_warm": bool(P["commit_s"]),
                                           "overhead": n * (P["proving_s"] + commit) / P["instances"]}
        V.rejected.append(entry)

    def best(pool: list[tuple[Ctx, dict[str, Any]]]) -> tuple[Ctx, dict[str, Any]] | None:
        return min(pool, key=lambda cp: (-cp[1]["instances_per_second"], cp[0].art)) if pool else None

    def pick(c: Ctx, P: Mapping[str, Any], N: float) -> dict[str, Any]:
        return {"artifact": c.art, "configuration": c.config.id if c.config else None, "bound_log2": whole_proof_log2(c.meta["workload_fingerprint"]),
                "hashes": [h.text() for h in (c.config.hashes if c.config else ())], "overhead": N / P["instances_per_second"]}

    columns = [f.id for f in FAMILIES] + ([f.id + T.AUTH_COLUMN_SUFFIX["included-hash"] for f in FAMILIES] if rules.legacy else [])
    for target in TARGETS.values():
        N = native(target)
        for scheme in line_schemes:
            line: dict[str, Any] = {"line": target.name, "scheme": scheme, "scheme_label": _SCHEME[scheme].label if scheme else None,
                                    "status": target.status, "device": target.native_peak.device, "native": N,
                                    "instruction": target.instruction, "cells": {}, "empty": {}}
            for col in columns:
                key = (target.name, scheme, col)
                hit = best(pools.get(key, []))
                if hit is None:
                    line["cells"][col] = None
                    line["empty"][col] = _empty_reason(V.rejected, target, scheme, line_schemes, col, configurations)
                    if unfiltered.get(key):
                        u = pick(*best(unfiltered[key]), N["instances_per_second"])
                        line["empty"][col] = {"state": "below-filter", "unfiltered": u, "text": (
                            f"only below the filter: unfiltered pick {u['configuration']} {u['artifact']}, bound {_fmt_bound(u['bound_log2'])}, "
                            f"hashes {'; '.join(u['hashes'])}, {T.fmt_overhead(u['overhead'])}")}
                    continue
                c, P = hit
                vb = T.independent_verification(c.meta, c.labels, c.attempt)
                cell = ViewCell(line=target.name, column=col, artifact=c.art, configuration=c.config.id if c.config else None,
                                native=N["instances_per_second"], throughput=P, overhead=N["instances_per_second"] / P["instances_per_second"],
                                bound_log2=whole_proof_log2(c.meta["workload_fingerprint"]),
                                provisional=None if rules.legacy else provisional(c),
                                legacy=T._cell(c.meta, c.art, T._attempt_of(store, c.art)[0], target, _CANDIDATE[c.family.id],
                                               vb.by if vb else None, T.instances_equiv_of(c.meta, c.labels, c.attempt, target, equivs),
                                               _open_finding(c.labels)),
                                batching=_batching(c.meta["workload_fingerprint"]), scheme=scheme,
                                algebraic=[h.text() for h in (c.config.algebraic() if c.config else ())],
                                instances=dict(c.meta["workload_fingerprint"].get("instances") or {}))
                alt = best(pools.get(key, []) + unfiltered.get(key, []))
                if alt is not None and alt[0].art != c.art:
                    cell.unfiltered = pick(*alt, N["instances_per_second"])
                line["cells"][col] = cell
                V.cells.append(cell)
            V.lines.append(line)
    return V


def _open_finding(labels: Sequence[Any]) -> str | None:
    state, lab = T.cell_finding(labels)
    return T.finding_note(lab) if state == "open" else None


def _empty_reason(rejected: Sequence[Mapping[str, Any]], target: Target, scheme: str | None, line_schemes: Sequence[str | None],
                  col: str, configurations: Sequence[Configuration]) -> dict[str, Any]:
    """Why a cell is empty: ``n/s`` when every configuration of the family declares the subcircuit unsupported, else what its
    results lack (weaker statements or undefined schemes only, or the first reasons that apply).  A result on no line's
    scheme counts on every scheme line of its subcircuit and hardware class."""
    fam = [c for c in configurations if c.family == col]
    if fam and all(target.name in c.unsupported for c in fam):
        return {"state": "n/s", "text": "; ".join(sorted({c.unsupported[target.name] for c in fam}))}
    mine = [r for r in rejected if r.get("line") == target.name and r.get("column") == col
            and (r.get("scheme") == scheme or r.get("scheme") not in line_schemes)]
    if not mine:
        return {"state": "none", "text": "no result"}
    if all(r["codes"][0][0] in ("R", "S") for r in mine):
        weak = sorted({str(r["configuration"]) for r in mine if r["codes"][0][0] == "R"})
        bad = sorted({str(r["configuration"]) for r in mine if r["codes"][0][0] == "S"})
        parts = ([f"a weaker statement ({', '.join(weak)})"] if weak else []) + ([f"a scheme verity.commitments does not define ({', '.join(bad)})"] if bad else [])
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


def _instances_text(target_name: str, inst: Mapping[str, Any]) -> str:
    rng = inst.get("range") or ["?", "?"]
    how, _ = instance_range(TARGETS[target_name], inst) if target_name in TARGETS and inst else (None, None)
    kind = {"stream": "a prefix-consistent extension of the frozen set", "prefix": "a prefix of the frozen set",
            "frozen": "the frozen set"}.get(how or "", "re-packed")
    return f"instances [{rng[0]}, {rng[1]}) of {inst.get('tier', '?')} ({kind})"


def _native_text(N: Mapping[str, Any]) -> str:
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
                                                        cfg.zk, soundness, "; ".join(x.text() for x in cfg.proximity), cfg.transferability,
                                                        cfg.assumptions, cfg.pin, "yes" if cfg.archived else "no",
                                                        ", ".join(w for w, _ in cells) or T.EMPTY]))
    return out


def _cell_name(c: ViewCell) -> str:
    return f"{c.line}{' · ' + _SCHEME[c.scheme].label if c.scheme else ''} · {c.column}"


def overheads(c: ViewCell) -> tuple[float, float]:
    """``(serving overhead, proving overhead)`` of a cell: commitment and proving seconds per instance, each times N; they sum
    to the cell's ``N / P``."""
    P = c.throughput
    return c.native * P["commit_s"] / P["instances"], c.native * P["proving_s"] / P["instances"]


def render_markdown(V: Views) -> str:
    rules = V.rules
    filt = ", ".join(([f"whole-proof bound 2^{rules.security_log2:g} or better"] if rules.security_log2 is not None else [])
                     + (["no algebraic hash in the soundness argument"] if rules.standard_hashes else [])) or "unfiltered"
    out = [f"> {PREVIEW}", "", f"Rules: `{rules.name}` ({filt}" + ("; legacy parameters: B = 4096, K = 1536, bare and + in-proof hash "
                                                                 "columns, proving time only" if rules.legacy else "") + ").", "",
           "## Table 1 — Backend configurations and their security", ""]
    rows1, fam_seen = [], None
    for fid, row in table1_rows(V):
        if fid != fam_seen:
            fam = _FAMILY.get(fid)
            desc = f"{fam.description}; declared class {' | '.join(fam.declared_classes)}; setup {fam.setup}; {fam.interaction}" if fam else "not a Table 2 column"
            rows1.append([f"**{fid}**", "", desc, *([""] * (len(TABLE1_COLUMNS) - 3))])
            fam_seen = fid
        rows1.append(row)
    out += [T._md_table(TABLE1_COLUMNS, rows1), "",
            "Statement: the full relation commits x, W and y with the configuration's scheme and proves y = S(x, W) over the committed "
            "values; its scheme must be defined in verity.commitments (reason S otherwise). A configuration whose soundness argument "
            f"uses an algebraic hash (Poseidon2 and the like) is admissible and flagged: {ALGEBRAIC_FLAG}. Bounds "
            "are the accountant's whole-proof figures at each cell's batching (`security.achieved_log2`), from proven proximity bounds only.", "",
            "## Table 2 — End-to-end overhead per subcircuit, hardware class and commitment scheme", "",
            "N = compute-bound datasheet estimate, instances/s (busiest unit, peak entry); P = instances/s end to end "
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
    rows2 = []
    for ln in V.lines:
        N = ln["native"]
        row = [f"{N['device']} · {N['dtype']} · {ln['line']}" + (f" · {ln['scheme_label']}" if ln["scheme"] else ""),
               _native_text(N) + " " + note(f"{ln['line']}: {N['work_model']}; {N['ops_per_instance']} tensor-core FLOP per instance "
                                            f"({N['steps']} steps); peak {N['peak_flop_per_second']:.4g} FLOP/s, {N['source']}")]
        for col in cols:
            c = ln["cells"].get(col)
            if c is None:
                e = ln["empty"].get(col) or {"state": "none", "text": "no result"}
                where = f"{ln['line']}{' · ' + ln['scheme_label'] if ln['scheme'] else ''} · {col}"
                row.append(("n/s " if e["state"] == "n/s" else f"{T.EMPTY} ") + note(f"{where}: {e['text']}"))
                continue
            P, b = c.throughput, c.batching
            text = (f"{c.artifact} — {c.configuration or col}{', scheme ' + c.scheme if c.scheme else ''}; P {fmt_rate(P['instances_per_second'])} ({P['instances']} instances in "
                    f"{fmt_per_instance(P['end_to_end_s'])}: commitment {fmt_per_instance(P['commit_s'])}, proving {fmt_per_instance(P['proving_s'])}); "
                    f"batching total {b['total']}, per proof {b['per_proof'] or '?'}, proofs {b['proofs'] or '?'}, in flight {b['in_flight'] or '?'}; "
                    f"{_instances_text(c.legacy.target, c.instances)}; whole-proof bound {_fmt_bound(c.bound_log2)}; host {c.legacy.host or '?'}; {c.legacy.verification_note()}")
            if c.legacy.instances_equiv:
                text += f"; {T.INSTANCE_EQUIV_NOTE} ({c.legacy.instances_equiv})"
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
            row.append(T.fmt_overhead(c.overhead) + (" (prov.)" if c.provisional else "") + (" (alg.)" if c.algebraic else "")
                       + (f" ({T.UNDER_REVERIFICATION})" if c.legacy.finding else "") + " " + note(text))
        rows2.append(row)
    out += [T._md_table(("Subcircuit · hardware · scheme", "Native N", *cols), rows2), ""]

    out += ["## Table 3 — Where the time goes (per instance, share of end to end)", "",
            "Commitment = computing the commitments the statement names; proving = the six buckets and Other (proving total − Σ buckets − "
            f"Σ joints); end to end = commitment + proving. A fused stage prints once in its first column, `{T.JOINT_CONTINUATION}` in the others. "
            "Serving overhead = commitment time × N, proving overhead = proving time × N; they add up to the Table 2 cell.", ""]
    rows3 = []
    for c in V.cells:
        P, ph = c.throughput, c.legacy.phases
        n, total = P["instances"], P["end_to_end_s"]
        pct = lambda v: f"{fmt_per_instance(v / n)} ({100 * v / total:.0f}%)"  # noqa: E731
        cells: dict[str, str] = {b: pct(v) for b, v in ph["buckets"].items() if v is not None}
        for j in ph["joints"].values():
            first, *rest = j["covers"]
            cells[first] = f"{'+'.join(j['covers'])}: {pct(j['seconds'])}"
            cells.update({b: T.JOINT_CONTINUATION for b in rest})
        serving, proving = overheads(c)
        rows3.append([_cell_name(c), pct(P["commit_s"]), *[cells.get(b, T.EMPTY) for b in contract.BUCKETS],
                      pct(ph["other"]) if _num(ph["other"]) else T.EMPTY, f"{fmt_per_instance(total / n)} (100%)",
                      T.fmt_overhead(serving) if serving > 0 else "0×", T.fmt_overhead(proving)])
    out += [T._md_table(("Cell", "Commitment", "Witness", "Encoding + witness commitment", "Arithmetic", "Lookup", "ZK additional",
                         "Serialization", "Other", "End to end", "Serving overhead", "Proving overhead"), rows3)
            if rows3 else f"(no Table 2 cell {T.EMPTY} nothing to decompose)", ""]
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
    if V.not_families:
        n = sum(len(v) for v in V.not_families.values())
        out += ["", f"Not a backend family's result ({n}; by software.backend.name): "
                + ", ".join(f"{name} ×{len(a)}" for name, a in sorted(V.not_families.items(), key=lambda kv: (-len(kv[1]), kv[0])))]
    return "\n".join(out) + "\n"


def render_json(V: Views) -> dict[str, Any]:
    return {
        "schema": "verity/tables-views/v1", "preview": PREVIEW, "rules": dataclasses.asdict(V.rules),
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
                    "serving_overhead": overheads(c)[0], "proving_overhead": overheads(c)[1], **c.legacy.phases} for c in V.cells],
        "rejected": [{**r, "codes": [list(x) for x in r["codes"]]} for r in V.rejected],
        "not_families": V.not_families,
    }


# --- CLI --------------------------------------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, metavar="ROOT", help="store root (default $RESEARCH_STORE or ~/.research/store)")
    ap.add_argument("--format", choices=("md", "json"), default="md")
    ap.add_argument("--rules", choices=tuple(RULES), default=SPEC.name)
    ap.add_argument("--security-bits", type=float, default=None, metavar="N", help="filter to a whole-proof bound of 2^-N or better")
    ap.add_argument("--include-archived", action="store_true")
    ap.add_argument("--standard-hashes-only", action="store_true", help="drop configurations with an algebraic hash in the soundness argument")
    ap.add_argument("--parity", action="store_true", help="compare the legacy-rules views with a fresh `tables` build; exit 1 on a difference")
    ap.add_argument("--snapshot", default=None, metavar="NAME", help="with --parity: also compare with this `tables --snapshot`'s members")
    ns = ap.parse_args(argv)
    from research.store.local import LocalStore  # noqa: PLC0415  (only the CLI needs the store package)

    st = LocalStore(ns.root)
    if ns.parity:
        members = None
        if ns.snapshot:
            rows = [r for r in st.index.select_artifacts(kind="dataset-snapshot/v1") if (D._doc(st, r).get("meta") or {}).get("name") == ns.snapshot]
            if not rows:
                print(f"views: no dataset-snapshot/v1 named {ns.snapshot!r}", file=sys.stderr)
                return 2
            members = list((D._doc(st, rows[-1]).get("refs") or {}).get("members") or [])
        diffs = parity(st, members=members)
        if ns.format == "json":
            print(json.dumps({"schema": "verity/tables-parity/v1", "snapshot": ns.snapshot, "ok": not diffs, "differences": diffs}, indent=2))
        else:
            print("parity: ok (the legacy-rules views select exactly today's cells)" if not diffs else "\n".join(["parity: DIFFERENT", *diffs]))
        return 0 if not diffs else 1
    rules = RULES[ns.rules]
    if ns.security_bits is not None:
        rules = dataclasses.replace(rules, name=f"{rules.name}@{ns.security_bits:g}", security_log2=-ns.security_bits)
    if ns.include_archived:
        rules = dataclasses.replace(rules, include_archived=True)
    if ns.standard_hashes_only:
        rules = dataclasses.replace(rules, name=f"{rules.name}+standard-hashes", standard_hashes=True)
    V = build(st, rules)
    if ns.format == "json":
        print(json.dumps(render_json(V), indent=2, sort_keys=True, default=str))
    else:
        print(render_markdown(V), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
