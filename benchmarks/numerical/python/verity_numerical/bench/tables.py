"""The proof campaign's three canonical result tables, frozen 2026-09-22 (``archive/AGENTS.md``, "Canonical tables").

    python -m verity_numerical.bench.tables --root <store root> [--format md|json] [--snapshot]

Reads ``bench-result/v1``, ``native-peak/v1`` and ``instance-equiv/v1`` artifacts and their labels from the local evidence
store (``research.store.LocalStore``), applies the frozen validity predicate (:func:`reject_reasons`) and renders:

* **Table 1** -- the candidate proof systems (:data:`CANDIDATES`): definitional, no performance, no status;
* **Table 2** -- headline proving overhead by target: one row per ``verity.ml.circuits.targets.target.TARGETS`` entry, one
  column per candidate; a cell is the *best valid* implementation of that candidate (min ``t.total``) run on the row's
  native device SKU, printed as ``R_native_spec / R_proved`` with ``R_proved = 2KB / t.total``;
* **Table 3** -- the prover phase decomposition of every populated Table 2 cell (the six buckets, joints, ``Other``).

These are the only canonical tables; everything else (memory, committed elements, hash bytes, verifier seconds,
transcript bytes, the RTT sweep, coupling diagnostics) is drill-down behind a Table 2 or Table 3 cell.  ``--format json``
gives the machine-readable object (per-cell artifact ids, ``R_proved``, and the reasons every rejected result was
rejected); ``--snapshot`` freezes the cell artifacts into a ``dataset-snapshot/v1`` named ``tables-<UTC date>``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import json
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from verity.ml.circuits.silicon.instructions import INSTRUCTIONS
from verity.ml.circuits.targets.target import BF16_HOPPER, FIRST, FP8_ADA_MMA, FP8_HOPPER_WGMMA, NVFP4_SM120, TARGETS, Target
from verity_numerical.bench import census, contract

try:  # the label keys are pinned with the store (`research.store.vocab`, tools/research); verity-numerical does not depend on that package
    from research.store import vocab as _vocab

    _CANDIDATE, _LABEL, _PROOF_CLASS, _LANE = _vocab.CANDIDATE, _vocab.LABEL, _vocab.PROOF_CLASS, _vocab.LANE
    _FINDING = _vocab.FINDING
    _INDEPENDENTLY_VERIFIED, _VERIFIED, _VERIFIED_ACCEPTED = _vocab.INDEPENDENTLY_VERIFIED, _vocab.VERIFIED, _vocab.VERIFIED_ACCEPTED
    _VERIFIED_REJECTED = _vocab.VERIFIED_REJECTED
    _KEY_ALIASES, _VALUE_ALIASES = dict(_vocab.KEY_ALIASES), dict(_vocab.VALUE_ALIASES)
except ImportError:  # pragma: no cover  -- the same spellings, so the predicate still imports where the store package is absent (tested equal)
    _CANDIDATE, _LABEL, _PROOF_CLASS, _LANE = "candidate", "label", "proof_class", "lane"
    _FINDING = "finding"
    _INDEPENDENTLY_VERIFIED, _VERIFIED, _VERIFIED_ACCEPTED = "independently_verified", "verified", "accepted"
    _VERIFIED_REJECTED = "rejected"
    _KEY_ALIASES, _VALUE_ALIASES = {"notes": "note", "verify_note": "note", "batch": "B", "checker": "verifier"}, {("candidate", "A"): "A-GKR"}


def canonical_label(key: Any, value: Any) -> tuple[str, Any]:
    """An assertion as read: old spellings of a key (``notes`` -> ``note``, ``batch`` -> ``B``, ``checker`` -> ``verifier``) and of a
    value (``candidate=A`` -> ``A-GKR``) mean the canonical one; the words ``true`` / ``false`` under a bool key mean the bool.
    Readers accept every spelling ever written; writers (``ledger_to_store``, ``research data label``) store only the canonical."""
    k = _KEY_ALIASES.get(str(key), str(key))
    if isinstance(value, str):
        value = _VALUE_ALIASES.get((k, value), value)
        if k in _BOOL_KEYS and value in ("true", "false"):
            value = value == "true"
    return k, value


_BOOL_KEYS = frozenset({_INDEPENDENTLY_VERIFIED, "zk", "model_agrees", "breakthrough"})

__all__ = [
    "AUTHENTICATION",
    "AUTH_MODES",
    "B",
    "BF16_INSTANCES_MANIFEST_SHA256",
    "CANDIDATES",
    "CAPTION",
    "Candidate",
    "Cell",
    "FROZEN_INSTANCES",
    "FROZEN_ON",
    "JOINT_CONTINUATION",
    "K",
    "NATIVE_PEAK_KIND",
    "NATIVE_PEAK_METHOD",
    "PRODUCER_LABEL_KEYS",
    "PRODUCER_META_KEYS",
    "PRODUCER_NAME_LABEL_KEYS",
    "IN_PROOF_HASH",
    "INSTANCE_EQUIV_SCHEMA",
    "InstanceEquiv",
    "PHASE_SUM_PROBLEM",
    "RANK",
    "REASON_LETTERS",
    "REDTEAM_GLOB",
    "RESULT_KIND",
    "SOUNDNESS_LOG2",
    "TABLE1_COLUMNS",
    "TABLE2_COLUMNS",
    "TABLE3_COLUMNS",
    "Tables",
    "VERIFIED_LABELS",
    "ZK_VALUES",
    "build",
    "candidate_for",
    "canonical_label",
    "cell_finding",
    "cell_artifacts",
    "column_authentication",
    "dtype_of",
    "fmt_overhead",
    "fmt_seconds",
    "frozen_instances",
    "independent_verification",
    "instance_equivalence",
    "instance_equivs",
    "instances_diff",
    "instances_equiv_of",
    "main",
    "normalise_sku",
    "phases",
    "in_proof_hash_codes",
    "producers",
    "reject_codes",
    "reject_reasons",
    "render_json",
    "render_markdown",
    "same_sku",
    "device_variant",
    "snapshot_name",
    "take_snapshot",
]

# --- caption constants (frozen) -----------------------------------------------------------------------------------------

FROZEN_ON = "2026-09-22"
K = census.subcircuit_of(FIRST.name).parameters.K
B = census.GEMM_COORDINATE.reference_batch
#: log2 of the soundness error: the target AND the achieved (union-bounded over sub-batches) must both be <= this.
SOUNDNESS_LOG2 = -128
AUTHENTICATION = "excluded"
#: Table 2 prints one column per candidate per authentication mode that has a valid cell: ``excluded`` (the bare relation) and
#: ``included-hash`` (the operand rows hashed INSIDE the proof with Poseidon2-BabyBear, the digests bound to the Merkle roots;
#: lane hash-relation, user decision 2026-09-23 05:39Z: "ZK-friendly hashes as part of the proof").
AUTH_MODES: tuple[str, ...] = (AUTHENTICATION, "included-hash")
AUTH_COLUMN_SUFFIX = {AUTHENTICATION: "", "included-hash": " + in-proof hash"}
CAPTION: dict[str, Any] = {
    "K": K,
    "B": B,
    "soundness": f"2^{SOUNDNESS_LOG2} (target and achieved, union-bounded over sub-batches)",
    "execution": "dense",
    "accumulation": "fp32",
    "flop_per_fma": 2,
    "instances": "the frozen input set of the row's target (FROZEN_INSTANCES)",
    "native_denominator": "datasheet dense tensor-core peak of the exact target device (Target.native_peak)",
    "authentication": "excluded (bare columns) | included-hash (columns ' + in-proof hash': Poseidon2-BabyBear row hashing inside the proof, "
                      "digests checked against the operand Merkle roots; x, W private)",
    "cell": "best valid implementation of the candidate in its declared proof class: min t.total; printed as R_native_spec / R_proved, R_proved = 2KB / t.total",
    "same_sku": "the prover ran on the same physical device SKU as the row's native device (hardware.gpu.name ~ native_peak.device)",
    "device_variant": "the board is the reference part: hardware.gpu.memory_total_bytes within 12 % of native_peak.memory_bytes (a same-named re-memoried board is not)",
}

RESULT_KIND = "bench-result/v1"
NATIVE_PEAK_KIND = "native-peak/v1"
#: ``benchmarks/native_peak`` 's canonical GEMM method string (the shape every GEMM measurement's ``method`` starts with).
NATIVE_PEAK_METHOD_CUBLAS = "cublas dense gemm 8192^3, median of 5"
#: What the measured column is, for the caption (selection rule in :func:`_native_measured`).
NATIVE_PEAK_METHOD = ("the best dense GEMM through the vendor library on the row's device x dtype, median of 5, the call and size in the "
                      "footnote; an instruction-level microbenchmark only when no GEMM measurement exists, and then said so")
#: The BF16 frozen set: sha256 of ``fixtures/bench-instances/v1/manifest.json`` (``verity_numerical.bench.instances.manifest_sha256``).
BF16_INSTANCES_MANIFEST_SHA256 = census.INPUT_SETS["bench-instances-v1"].manifest_sha256
#: The frozen instance ref a valid result of each target must carry, ``instances`` field for field.  A target absent here
#: has no frozen set yet: every result of it is rejected with that reason.  The synthetic sets (FP8 Ada, the two Hopper
#: relations) are pinned by their recipe digest, the BF16 Ampere set by its fixture manifest (``benchmarks/census/input_sets.json``).
FROZEN_INSTANCES: dict[str, dict[str, Any]] = {
    s.target: contract.instances_ref(s.tier, 0, s.size, s.manifest_sha256, dataset=s.dataset) for s in census.INPUT_SETS.values()}

#: Label keys whose asserter is the *producer* of a result (the lane's own row labels, ``ledger_to_store``); a verification
#: label by any of them, or by the producing attempt's ``execution.campaign``, is a self-assertion and does not count.
PRODUCER_LABEL_KEYS = (_CANDIDATE, _LABEL)
#: Label keys whose *value* names the producing lane (``lane=<lane>``, for a result whose meta does not carry it); the asserter
#: of such a label is not thereby a producer (the coordinator back-fills attribution).
PRODUCER_NAME_LABEL_KEYS = (_LANE,)
#: Meta keys of a result that name who produced or registered it.  ``research data put`` records no registrar (the manifest is
#: kind, meta, refs, payload; the catalog has no such column), so a hand-registered result -- no attempt -- is attributed only by
#: its own meta: ``lane`` (the producing lane), ``by`` / ``registered_by`` (the lane that ran ``put``), ``constructed_by`` (prose
#: that starts with the lane's name).  Each value's leading lane-name token is a producer.
PRODUCER_META_KEYS = ("lane", "by", "registered_by", "constructed_by")
#: The labels that mean "an independent verifier accepted this result": ``(key, accepted values)``.  The store's back-fill of
#: Rust-verifier acceptance used ``independently_verified=true`` (by ``h100-bestvsbest``, ``coordinator``); ``verified=accepted``
#: is the name the freeze spec gives it.  Either counts when the asserter is not a producer.
VERIFIED_LABELS: tuple[tuple[str, tuple[Any, ...]], ...] = ((_INDEPENDENTLY_VERIFIED, (True, "true")), (_VERIFIED, (_VERIFIED_ACCEPTED,)))
REDTEAM_GLOB = "red-team-*"
#: Proof classes in order of strength; a red-team ``proof_class`` verdict below the fingerprint's class is a downgrade.  ``COMPONENT``
#: is the ledger's class for a component benchmark (``bench.ledger.PROOF_CLASSES``), below every ``contract.PROOF_CLASSES`` entry; an
#: unknown class ranks below all of them (``RANK.get(c, -1)``).  The one ordering for every table (``store_tables`` reads this one).
RANK: dict[str, int] = {c: i for i, c in enumerate(("COMPONENT", "NO_PROOF", "ARITHMETIC_DIAGNOSTIC", "NON_ZK_PROOF_DIAGNOSTIC", "NON_ZK_PROOF",
                                                    "COMPLETE_HVZK_BACKEND", "COMPLETE_ZK_BACKEND"))}
SOUND_CLASSES = ("NON_ZK_PROOF", "COMPLETE_HVZK_BACKEND", "COMPLETE_ZK_BACKEND")

# --- Table 1: the candidates -----------------------------------------------------------------------------------------------

TABLE1_COLUMNS = ("Candidate", "Description", "ZK", "Verifier model", "Soundness", "Setup", "Interaction", "Security assumptions")
#: The only words a ``ZK`` cell may start a variant with (variants are ``;``-separated, a parenthesis may follow).
ZK_VALUES = ("No", "HVZK", "Malicious-verifier ZK")
JOINT_CONTINUATION = "↞"


@dataclass(frozen=True)
class Candidate:
    """One row of Table 1 plus what Table 2 needs: the declared proof class(es) a cell may hold, the drill-down classes,
    and the recogniser over ``software.backend.name`` (a name matches when it equals a pattern or starts with it followed
    by a non-alphanumeric character: ``"b-ligero chain (archive/direct/ligero)"`` and ``"b-ligero chain, fp8-ada relation
    (...)"`` both match ``"b-ligero chain"``)."""

    id: str
    description: str
    zk: str
    verifier_model: str
    soundness: str
    setup: str
    interaction: str
    assumptions: str
    proof_classes: tuple[str, ...]
    drilldown_classes: tuple[str, ...]
    backend_names: tuple[str, ...]

    def recognises(self, backend_name: Any) -> bool:
        if not isinstance(backend_name, str):
            return False
        name = backend_name.strip().lower()
        return any(re.match(rf"^{re.escape(p.lower())}(?![a-z0-9])", name) for p in self.backend_names)

    def table1_row(self) -> list[str]:
        return [self.id, self.description, self.zk, self.verifier_model, self.soundness, self.setup, self.interaction, self.assumptions]


CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        id="A-GKR",
        description="specialized data-parallel GKR/sumcheck prover (checker v2 + LogUp lookups) for the exact numerical relation",
        zk="No",
        verifier_model="honest (non-ZK, soundness only)",
        soundness="2^-128",
        setup="none (no trusted setup)",
        interaction="interactive; Fiat–Shamir variant non-interactive",
        assumptions="collision-resistant hash (SHA-512 Merkle); Fiat–Shamir in the random-oracle model for the non-interactive variant; "
                    "LogUp fractional-sum soundness over BabyBear^6",
        # The user's label for a complete, non-ZK proof is NON_ZK_PROOF_DIAGNOSTIC ("diagnostic" relative to the ZK requirement, not
        # an incomplete proof): every A-GKR emitter writes it.  NON_ZK_PROOF is the ledger's spelling (SP1 sound core).  Both are A's class;
        # ARITHMETIC_DIAGNOSTIC (relation only, no proof) is not.
        proof_classes=("NON_ZK_PROOF_DIAGNOSTIC", "NON_ZK_PROOF"),
        drilldown_classes=(),
        backend_names=("verity-gkr",),
    ),
    Candidate(
        id="B-Ligero",
        description="specialized Ligero-style interleaved-RS proof over the exact numerical checker",
        zk="Malicious-verifier ZK (interactive, coin-commitment; transcript is live-verifier-only per D7); HVZK (Fiat–Shamir variant, transferable)",
        verifier_model="malicious (interactive) / honest (Fiat–Shamir)",
        soundness="2^-128",
        setup="none (no trusted setup)",
        interaction="interactive; Fiat–Shamir variant non-interactive",
        assumptions="collision-resistant hash (BLAKE3/BLAKE2b); SHAKE-256 mask sampling; random oracle only for the Fiat–Shamir variant; "
                    "proximity-gap bound model named in the fingerprint (security.bound_model)",
        proof_classes=("COMPLETE_ZK_BACKEND",),
        drilldown_classes=("COMPLETE_HVZK_BACKEND",),
        backend_names=("b-ligero chain",),
    ),
    Candidate(
        id="SP1",
        description="general-purpose zkVM baseline executing the exact relation",
        zk="as configured",
        verifier_model="as configured",
        soundness="as configured",
        setup="none (no trusted setup)",
        interaction="non-interactive (Fiat–Shamir)",
        assumptions="SP1 / STARK hash assumptions",
        proof_classes=SOUND_CLASSES,
        drilldown_classes=(),
        backend_names=("sp1",),
    ),
)


def candidate_for(backend_name: Any) -> Candidate | None:
    """The Table 1 candidate whose recogniser matches ``software.backend.name``; ``None`` for a component benchmark or an
    unknown backend."""

    hits = [c for c in CANDIDATES if c.recognises(backend_name)]
    return hits[0] if hits else None


# --- the same-SKU invariant --------------------------------------------------------------------------------------------------

_SINGLE_SKU_FAMILIES = {"4090": "rtx-4090", "5090": "rtx-5090", "l40s": "l40s", "l40": "l40", "t4": "t4", "a10": "a10", "l4": "l4"}
_MULTI_SKU_FAMILIES = ("a100", "h100", "h200", "b200")
_FORMS = ("sxm4", "sxm5", "sxm", "pcie", "nvl")
_NOISE = {"nvidia", "geforce", "tesla", "graphics", "device", "gpu", "hbm2e", "hbm2", "hbm3", "hbm3e"}


def normalise_sku(name: Any) -> str | None:
    """A canonical SKU token for a device name as ``nvidia-smi`` or a datasheet writes it, so the same physical part compares
    equal: ``"NVIDIA A100-SXM4-80GB"`` == ``"NVIDIA A100 SXM4 80GB"`` -> ``a100-sxm4-80gb``; ``"NVIDIA H100 80GB HBM3"`` ==
    ``"NVIDIA H100 SXM5 80GB"`` -> ``h100-sxm5-80gb`` (the 80 GB HBM3 part *is* the SXM5 board; the PCIe part says PCIe);
    ``"NVIDIA GeForce RTX 4090"`` -> ``rtx-4090``.  A family with several SKUs whose form factor or memory is not stated
    keeps an ``unknown`` component, so it never equals a fully named SKU.  Unknown families give the cleaned name.  ``None``
    for a non-string / empty name."""

    if not isinstance(name, str) or not name.strip():
        return None
    s = re.sub(r"[-_/,()]+", " ", name.lower())
    s = re.sub(r"(\d+)\s*gb\b", r"\1gb", s)
    toks = [t for t in s.split() if t not in _NOISE]
    hbm3 = "hbm3" in name.lower().replace("-", "").replace(" ", "")
    for t in toks:
        if t in _SINGLE_SKU_FAMILIES:
            return _SINGLE_SKU_FAMILIES[t]
    family = next((t for t in toks if t in _MULTI_SKU_FAMILIES), None)
    if family is None:
        return "-".join(toks) or None
    mem = next((t for t in toks if re.fullmatch(r"\d+gb", t)), None)
    form = next((t for t in toks if t in _FORMS), None)
    if form == "sxm":
        form = {"a100": "sxm4", "h100": "sxm5", "h200": "sxm5"}.get(family, "sxm")
    if family == "h100" and form is None and hbm3:
        form = "sxm5"
    if family == "h100" and form in ("sxm5", "pcie") and mem is None:
        mem = "80gb"
    return f"{family}-{form or 'unknown'}-{mem or 'unknown'}"


#: Fractional deviation of the reported device memory from the datasheet's that still counts as the reference part
#: (``torch.cuda`` reports ~0.5 % less than the nominal total; a re-memoried board reports ~2x).
MEMORY_TOLERANCE = 0.12


def device_variant(gpu: Mapping[str, Any] | None, target: Target) -> str | None:
    """A reject reason when the prover's board is a *variant* of the row's SKU -- same name, very different memory -- or
    ``None``.  The name check (:func:`same_sku`) is by ``nvidia-smi`` string, which a re-memoried board keeps; the
    datasheet peak the row divides by is the reference part's, so such a board is not a valid cell.  Unchecked when the
    target states no ``memory_bytes`` or the fingerprint reports none."""

    ref = contract.native_peak_of(target).memory_bytes
    mem = (gpu or {}).get("memory_total_bytes")
    if not ref or not _num(mem) or mem <= 0:
        return None
    if abs(mem / ref - 1) > MEMORY_TOLERANCE:
        return (f"device variant: {mem / 2**30:.1f} GiB of device memory on {(gpu or {}).get('name')!r}, the reference "
                f"{contract.native_peak_of(target).device!r} has {ref / 2**30:.0f} GiB -- not the datasheet part the row divides by")
    return None


def same_sku(a: Any, b: Any) -> bool:
    """Whether two device names normalise to the same SKU (``False`` when either cannot be normalised)."""

    na, nb = normalise_sku(a), normalise_sku(b)
    return na is not None and na == nb


def dtype_of(target: Target) -> str | None:
    """The target's operand dtype (``bf16``, ``e4m3``, ``e2m1``): the instruction registry's ``a_dtype``."""

    ins = INSTRUCTIONS.get(target.instruction) if target.instruction else None
    return ins.a_dtype if ins else None


#: ``native-peak/v1`` measurement dtypes that name a block-scaled *format* over an operand dtype: ``benchmarks/native_peak
#: --dtype nvfp4`` times the sm_120 ``kind::mxf4nvf4`` instruction on E2M1 operands (UE4M3 block scales), so it is the
#: measured peak of an ``e2m1`` target.
NATIVE_DTYPE_ALIASES = {"nvfp4": "e2m1", "mxfp4": "e2m1"}


def native_dtype(meta: dict[str, Any]) -> str | None:
    """The operand dtype a ``native-peak/v1`` measurement is of (its ``dtype`` through ``NATIVE_DTYPE_ALIASES``)."""

    d = meta.get("dtype")
    return NATIVE_DTYPE_ALIASES.get(d, d)


def frozen_instances(target: Target) -> dict[str, Any] | None:
    return FROZEN_INSTANCES.get(target.name)


# --- labels --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Label:
    """An assertion as the predicates read it (key and value canonical, see :func:`canonical_label`).  ``path`` is the assertion
    file when the label came from the catalog (``store_tables``), None when it came as a ``Label`` object."""

    key: str
    value: Any
    by: str
    ts: str
    ref: str | None
    path: str | None = None


def _labels(labels: Iterable[Any]) -> list[_Label]:
    """``research.store.model.Label`` objects or plain dicts -> the fields the predicate reads, in ts order."""

    out = []
    for lab in labels:
        g = (lambda k: lab.get(k)) if isinstance(lab, Mapping) else (lambda k: getattr(lab, k, None))
        key, value = canonical_label(g("key"), g("value"))
        out.append(_Label(key, value, str(g("by")), str(g("ts") or ""), g("ref")))
    return sorted(out, key=lambda l: l.ts)


_LANE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _lane_name(v: Any) -> str | None:
    m = _LANE_TOKEN.match(v.strip()) if isinstance(v, str) else None
    return m.group(0) if m else None


def producers(attempt: Mapping[str, Any] | None, labels: Iterable[Any], meta: Mapping[str, Any] | None = None) -> set[str]:
    """The asserters that are the result's own producer: the producing attempt's ``execution.campaign``, whoever asserted a
    :data:`PRODUCER_LABEL_KEYS` label on it (the lane's row labels), the lane a :data:`PRODUCER_NAME_LABEL_KEYS` label names, and
    the lanes its ``meta`` names under :data:`PRODUCER_META_KEYS` (the registrar of a hand-registered result).  A verification by
    one of these is a self-assertion."""

    out: set[str] = set()
    if isinstance(attempt, Mapping):
        camp = (attempt.get("execution") or {}).get("campaign") if isinstance(attempt.get("execution"), Mapping) else None
        if isinstance(camp, str) and camp:
            out.add(camp)
    for lab in _labels(labels):
        if lab.key in PRODUCER_LABEL_KEYS:
            out.add(lab.by)
        if lab.key in PRODUCER_NAME_LABEL_KEYS and _lane_name(lab.value):
            out.add(_lane_name(lab.value))
    if isinstance(meta, Mapping):
        out |= {n for n in (_lane_name(meta.get(k)) for k in PRODUCER_META_KEYS) if n}
    return out


def _verified_by(labels: Sequence[_Label], self_: set[str]) -> _Label | None:
    for lab in labels:
        if lab.by in self_:
            continue
        for key, accepted in VERIFIED_LABELS:
            if lab.key == key and lab.value in accepted:
                return lab
    return None


def independent_verification(result: Mapping[str, Any], labels: Iterable[Any], attempt: Mapping[str, Any] | None) -> _Label | None:
    """The accepting :data:`VERIFIED_LABELS` label by a non-producer (:func:`producers`), or ``None``.  A result with no attempt
    and no named producer has none: nobody can be shown independent of an unknown producer."""

    labs = _labels(labels)
    self_ = producers(attempt, labs, result)
    if attempt is None and not self_:
        return None
    return _verified_by(labs, self_)


def _redteam_ceiling(labels: Sequence[_Label], self_: set[str], by_glob: str = REDTEAM_GLOB) -> tuple[str | None, _Label | None]:
    """The class the latest ``proof_class`` event by an asserter matching ``by_glob`` (the red team by default) granted -- one event
    = one by + ref + ts; several assertions in it grant the highest -- or ``(None, None)`` when no such asserter spoke.  Downgrades
    stand, nothing is upgraded, a later re-audit supersedes an earlier verdict.  ``labels`` are in ts order; asserters in ``self_``
    (the producers) do not count."""

    verdicts = [l for l in labels if l.key == _PROOF_CLASS and l.by not in self_ and fnmatch.fnmatchcase(l.by, by_glob)]
    if not verdicts:
        return None, None
    latest = verdicts[-1]
    event = [l for l in verdicts if (l.by, l.ref, l.ts) == (latest.by, latest.ref, latest.ts)]
    top = max(event, key=lambda l: RANK.get(str(l.value), -1))
    return str(top.value), top


#: The research coordinator's cell marks (``lanes/coordinator/evidence/marked-cells.py``): the LATEST ``finding`` label on a
#: result, by ts, decides.  ``UNDER RE-VERIFICATION ...`` marks its cell with a footnote while it stands; a later ``CLEARED ...``
#: drops the mark; ``PULLED ...`` rejects the result like a red-team downgrade, and so does ``SUPERSEDED ...`` (its producer re-ran
#: the cell, e.g. on the spine's input sets), and so does any ``superseded_by`` label (the vocabulary's key, naming the
#: replacement).  Any other latest finding changes nothing.
FINDING_OPEN, FINDING_CLEARED, FINDING_PULLED, FINDING_SUPERSEDED = "UNDER RE-VERIFICATION", "CLEARED", "PULLED", "SUPERSEDED"
UNDER_REVERIFICATION = "under re-verification"


def cell_finding(labels: Iterable[Any]) -> tuple[str | None, _Label | None]:
    """``("open" | "cleared" | "pulled" | None, the latest finding label)`` of a result (``(None, None)`` without one)."""

    labs = _labels(labels)
    superseded = [lab for lab in labs if lab.key == "superseded_by" and str(lab.value).strip()]
    if superseded:                                          # the vocabulary's key: a replacement exists, whatever the findings say
        return "pulled", superseded[-1]
    found = [lab for lab in labs if lab.key == _FINDING]
    if not found:
        return None, None
    latest = max(found, key=lambda lab: (lab.ts, str(lab.value)))
    value = str(latest.value)
    for state, prefix in (("open", FINDING_OPEN), ("cleared", FINDING_CLEARED), ("pulled", FINDING_PULLED), ("pulled", FINDING_SUPERSEDED)):
        if value.startswith(prefix):
            return state, latest
    return None, latest


def finding_note(lab: _Label) -> str:
    return f"{lab.value} ({lab.by}, {lab.ts})"


# --- the validity predicate -------------------------------------------------------------------------------------------------


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _measurements(result: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for m in result.get("measurements") or []:
        if isinstance(m, Mapping) and isinstance(m.get("name"), str):
            out[m["name"]] = m.get("value")
    return out


#: One letter per reject reason, assigned where :func:`reject_codes` writes the reason (drill-down D2, campaign morning-tables
#: BRIEF §1).  A contract reason is ``P`` when any of its problems is phase accounting (:data:`PHASE_SUM_PROBLEM`).
REASON_LETTERS: dict[str, str] = {"U": "not independently verified", "I": "different input set", "P": "phase-sum",
                                  "S": "shared hashing", "Z": "class/ZK not the column's", "K": "prover not on the row's SKU",
                                  "B": "batch != 4096", "X": "other"}
#: ``contract.validate_measurements`` 's wording of a phase-accounting failure: the buckets and joints exceed ``t.total`` beyond
#: ``contract.TOLERANCE``, or one bucket is counted in two places (user decision 2026-09-24: such a result is out of Tables 2/3).
PHASE_SUM_PROBLEM = re.compile(r"^(buckets and joints sum to .* more than t\.total|bucket '[a-z_]+' is counted twice)")
#: Column ``<candidate> + in-proof hash`` (``included-hash``) is Poseidon2-BabyBear per row, no sharing (user decision keep_frozen,
#: 2026-09-24): the fingerprint's ``hash`` names it and, where the emitter records the leaf interface (lane leaf-iface D0), so do
#: ``leaf.scheme`` / ``leaf.schema``; ``sharing`` is ``none``.  Another leaf (Ajtai, BLAKE3, SHA-256) or a shared tile is a drill-down.
IN_PROOF_HASH, IN_PROOF_LEAF, IN_PROOF_SHARING = "poseidon2-babybear-w24", "poseidon2", "none"
SHARED_AUTHENTICATION = "included-hash-shared"
_LEAF_FAMILIES = (("ajtai", "an Ajtai (Ring-SIS) leaf"), ("blake3", "a BLAKE3 in-circuit leaf"), ("sha256", "a SHA-256 in-circuit leaf"),
                  ("poseidon2", "a Poseidon2 leaf"))


def in_proof_hash_codes(fp: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Why an ``included-hash`` fingerprint is not column 2's hashing (Poseidon2-BabyBear per row, no sharing), as ``(letter, reason)``."""

    out: list[tuple[str, str]] = []
    h = fp.get("hash")
    leaf = fp.get("leaf") if isinstance(fp.get("leaf"), Mapping) else {}
    scheme, schema = leaf.get("scheme"), leaf.get("schema")
    if not (h == IN_PROOF_HASH and scheme in (None, IN_PROOF_LEAF) and (schema is None or str(schema).startswith(IN_PROOF_HASH + "/"))):
        named = " ".join(str(x).lower() for x in (h, scheme, schema) if x is not None)
        family = next((d for key, d in _LEAF_FAMILIES if key in named), "an unrecognised leaf" if named else "no named leaf")
        out.append(("X", f"leaf hash {h!r} (leaf.scheme {scheme!r}, schema {schema!r}) is {family}: the ' + in-proof hash' column is "
                         f"{IN_PROOF_HASH} per row, no sharing"))
    sharing, rows = fp.get("sharing"), fp.get("row_sharing")
    if sharing != IN_PROOF_SHARING or rows is not None:
        shared = rows is not None or sharing is not None
        out.append(("S" if shared else "X", f"sharing {sharing!r}{' with row_sharing' if rows is not None else ''}: the ' + in-proof hash' "
                                            f"column hashes every operand row per VU, sharing {IN_PROOF_SHARING!r} (a shared tile is a drill-down)"))
    return out


def reject_codes(result: Mapping[str, Any], labels: Iterable[Any], attempt: Mapping[str, Any] | None, target: Target,
                 candidate: Candidate, proof_classes: Sequence[str] | None = None, authentication: str = AUTHENTICATION,
                 equivs: Sequence[InstanceEquiv] | None = None) -> list[tuple[str, str]]:
    """Every reason ``result`` (a ``bench-result/v1`` meta: the result.json) is NOT a valid Table 2 cell for ``target`` x
    ``candidate``, as ``(letter, reason)`` (:data:`REASON_LETTERS`); ``[]`` means valid.  ``proof_classes`` overrides the candidate's
    declared classes (drill-down); ``equivs`` are the store's ``instance-equiv/v1`` artifacts (:func:`instance_equivs`).  Frozen:

    contract ``validate()`` empty; ``profile`` == the target; ``K == 1536``, ``B == 4096`` (and ``B_real == B`` when present);
    ``security.target <= -128`` and ``security.achieved_log2 <= -128`` when present; ``instances`` equal to the target's frozen
    ref, or re-packed with an accepted equivalence (:func:`instance_equivalence`); ``proof_class`` in the candidate's declared
    class(es); ``authentication`` the column's (``excluded``, or ``included-hash`` with Poseidon2-BabyBear per row and no sharing,
    :func:`in_proof_hash_codes`); the same-SKU invariant; independently verified (a :data:`VERIFIED_LABELS` label by a non-producer,
    :func:`producers`); not downgraded by a ``red-team-*`` verdict; not pulled by a ``finding`` label (:func:`cell_finding`)."""

    p: list[tuple[str, str]] = []
    labs = _labels(labels)
    self_ = producers(attempt, labs, result)
    problems = contract.validate(result)
    if problems:
        p.append(("P" if any(PHASE_SUM_PROBLEM.match(x) for x in problems) else "X", _contract_reason(problems)))
    fp = result.get("workload_fingerprint") if isinstance(result, Mapping) else None
    if not isinstance(fp, Mapping):
        return p + [("X", "workload_fingerprint missing")]
    if fp.get("profile") != target.name:
        p.append(("X", f"profile {fp.get('profile')!r} is not the row's target {target.name!r}"))
    if fp.get("K") != K:
        p.append(("X", f"K is {fp.get('K')!r}, the tables are at K = {K}"))
    if fp.get("B") != B:
        p.append(("B", f"B is {fp.get('B')!r}, the tables are at B = {B}"))
    if "B_real" in fp and fp.get("B_real") != fp.get("B"):
        p.append(("B", f"B_real {fp.get('B_real')!r} != B {fp.get('B')!r}: a padded batch is not the frozen workload"))
    sec = fp.get("security") if isinstance(fp.get("security"), Mapping) else {}
    tgt = sec.get("target")
    if not _num(tgt) or tgt > SOUNDNESS_LOG2:
        p.append(("X", f"security.target is {tgt!r}, need <= {SOUNDNESS_LOG2}"))
    ach = sec.get("achieved_log2")
    if ach is not None and (not _num(ach) or ach > SOUNDNESS_LOG2):
        p.append(("X", f"security.achieved_log2 is {ach!r}, need <= {SOUNDNESS_LOG2} (union bound over sub-batches)"))
    want = frozen_instances(target)
    inst = fp.get("instances")
    if want is None:
        p.append(("I", f"target {target.name!r} has no frozen input set yet"))
    elif not isinstance(inst, Mapping):
        p.append(("I", "instances is null: not measured on the frozen input set"))
    else:
        diff = instances_diff(inst, want)
        if diff:
            eq, why = instance_equivalence(inst, target, equivs, self_)
            if eq is None:
                p.append(("I", "instances differ from the frozen set on " + ", ".join(f"{k} ({inst.get(k)!r} vs {want[k]!r})" for k in diff)
                          + (f" (instance-equiv: {why})" if why else "")))
    classes = tuple(proof_classes) if proof_classes else candidate.proof_classes
    pc = fp.get("proof_class")
    if pc not in classes:
        p.append(("Z", f"proof_class {pc!r} is not {candidate.id}'s declared class {' | '.join(classes)}"))
    if fp.get("authentication") != authentication:
        p.append(("S" if fp.get("authentication") == SHARED_AUTHENTICATION else "X",
                  f"authentication {fp.get('authentication')!r}: this column is {authentication!r} (Table 2 columns: {' | '.join(AUTH_MODES)})"))
    elif authentication == "included-hash":
        p += in_proof_hash_codes(fp)
    hw = fp.get("hardware") if isinstance(fp.get("hardware"), Mapping) else {}
    gpu = hw.get("gpu") if isinstance(hw.get("gpu"), Mapping) else None
    gpu_name = gpu.get("name") if gpu else None
    if not same_sku(gpu_name, contract.native_peak_of(target).device):
        p.append(("K", f"same-SKU: prover ran on {gpu_name!r}, the row's native device is {contract.native_peak_of(target).device!r}"))
    variant = device_variant(gpu, target)
    if variant:
        p.append(("K", variant))
    if attempt is None and not self_:
        p.append(("U", "not independently verified: no attempt and no named producer (meta lane / by, a lane label, a candidate / label "
                       "asserter), so no verification can be shown independent"))
    elif _verified_by(labs, self_) is None:
        keys = " or ".join(f"{k}={'|'.join(map(str, v))}" for k, v in VERIFIED_LABELS)
        p.append(("U", f"not independently verified: no label {keys} by a non-producer (producers: {', '.join(sorted(self_)) or 'unknown'})"))
    ceiling, verdict = _redteam_ceiling(labs, self_)
    if verdict is not None and isinstance(pc, str) and RANK.get(ceiling or "", -1) < RANK.get(pc, -1):
        p.append(("Z", f"downgraded: {verdict.by} ({verdict.ref or 'no ref'}, {verdict.ts}) asserts proof_class={ceiling}, below {pc}"))
    state, lab = cell_finding(labs)
    if state == "pulled":
        p.append(("Z", f"pulled: {finding_note(lab)}"))
    return p


def reject_reasons(result: Mapping[str, Any], labels: Iterable[Any], attempt: Mapping[str, Any] | None, target: Target,
                   candidate: Candidate, proof_classes: Sequence[str] | None = None, authentication: str = AUTHENTICATION,
                   equivs: Sequence[InstanceEquiv] | None = None) -> list[str]:
    """The reasons of :func:`reject_codes` without their letters; ``[]`` means ``result`` is a valid Table 2 cell."""

    return [r for _, r in reject_codes(result, labels, attempt, target, candidate, proof_classes, authentication, equivs)]


def instances_equiv_of(result: Mapping[str, Any], labels: Iterable[Any], attempt: Mapping[str, Any] | None, target: Target,
                       equivs: Sequence[InstanceEquiv] | None) -> str | None:
    """The ``instance-equiv/v1`` artifact through which a re-packed ``result`` counts as the frozen set (for its footnote), or ``None``."""

    fp = result.get("workload_fingerprint") if isinstance(result.get("workload_fingerprint"), Mapping) else {}
    inst, want = fp.get("instances"), frozen_instances(target)
    if not isinstance(inst, Mapping) or want is None or not instances_diff(inst, want):
        return None
    return instance_equivalence(inst, target, equivs, producers(attempt, labels, result))[0]


def _canon(v: Any) -> Any:
    return list(v) if isinstance(v, (list, tuple)) else v


# --- re-packed instances: instance-equiv/v1 (user decision 2026-09-24) -------------------------------------------------------------

#: A result whose ``instances`` differ from the frozen ref counts as the frozen set only through an artifact of this schema
#: (campaign morning-tables BRIEF §5, fixed; lane fused-phases' ``bench.instance_equiv`` writes it) that shows the decoded x, W, y
#: of both sides byte-identical and that a non-producer labelled ``verified=accepted``.  The document is the artifact's meta
#: (``research data put --kind instance-equiv/v1 --meta @doc.json``) or, failing that, its JSON payload file (local blob only).
INSTANCE_EQUIV_SCHEMA = "instance-equiv/v1"
INSTANCE_EQUIV_ARRAYS = ("x", "W", "y")
INSTANCE_EQUIV_NOTE = "instances re-packed; decoded values byte-identical to the frozen set"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_PAYLOAD_JSON_MAX = 1 << 20


@dataclass(frozen=True)
class InstanceEquiv:
    """One ``instance-equiv/v1`` artifact as the predicate reads it: the document, the artifact's own meta (its attribution:
    ``lane`` / ``by``), its labels and producing attempt."""

    artifact: str
    doc: Mapping[str, Any]
    meta: Mapping[str, Any] = field(default_factory=dict)
    labels: tuple[Any, ...] = ()
    attempt: Mapping[str, Any] | None = None
    #: why the document could not be read (its meta is not the document and its JSON payload is not usable here), else ``None``
    unreadable: str | None = None


def _ref_key(ref: Any) -> str:
    return json.dumps(ref, sort_keys=True, default=str)


def instances_diff(inst: Mapping[str, Any], want: Mapping[str, Any]) -> list[str]:
    """The fields of the frozen ref ``want`` that the result's ``inst`` does not carry equal."""

    return [k for k in ("dataset", "tier", "range", "manifest_sha256") if _canon(inst.get(k)) != _canon(want[k])]


def _equiv_content(doc: Mapping[str, Any], target: Target, inst: Mapping[str, Any], unreadable: str | None = None) -> list[str]:
    if unreadable:
        return [unreadable]
    p = []
    if doc.get("schema") != INSTANCE_EQUIV_SCHEMA:
        p.append(f"schema is {doc.get('schema')!r}")
    if doc.get("target") != target.name:
        p.append(f"target is {doc.get('target')!r}")
    frozen = doc.get("frozen")
    if target.name not in FROZEN_INSTANCES:
        return p + ["the target has no frozen set"]
    if not isinstance(frozen, Mapping) or instances_diff(frozen, FROZEN_INSTANCES[target.name]):
        p.append("frozen is not FROZEN_INSTANCES[target] (dataset, tier, range, manifest_sha256)")
    if _ref_key(doc.get("candidate")) != _ref_key(inst):
        p.append("candidate is not the result's instances ref field for field")
    if doc.get("equal") is not True:
        p.append(f"equal is {doc.get('equal')!r}")
    arrays = doc.get("arrays") if isinstance(doc.get("arrays"), Mapping) else {}
    missing = [a for a in INSTANCE_EQUIV_ARRAYS if a not in arrays]
    if missing:
        p.append("arrays lack " + ", ".join(missing))
    for name, a in arrays.items():
        f, c = (a.get("frozen_sha256"), a.get("candidate_sha256")) if isinstance(a, Mapping) else (None, None)
        if not (isinstance(f, str) and _HEX64.match(f) and f == c):
            p.append(f"arrays.{name}: frozen_sha256 {f!r} != candidate_sha256 {c!r} (or not a sha256)")
    return p


def _equiv_verdict(eq: InstanceEquiv, result_producers: set[str]) -> str | None:
    """Why ``eq`` is not ``verified=accepted`` by a non-producer (neither its own producers nor the result's), or ``None``: the
    latest ``verified`` verdict by a non-producer must be ``accepted``."""

    labs = _labels(eq.labels)
    own = producers(eq.attempt, labs, eq.meta)
    if eq.attempt is None and not own:
        return (f"{eq.artifact} names no producer (no attempt, meta lane / by, lane label): `research data label {eq.artifact} "
                f"lane=<producer>`, then a non-producer's verified=accepted")
    excluded = own | result_producers
    verdicts = [lab for lab in labs if lab.key == _VERIFIED and lab.by not in excluded]
    if not verdicts:
        return f"{eq.artifact} has no verified={_VERIFIED_ACCEPTED} by a non-producer (producers: {', '.join(sorted(excluded))})"
    if verdicts[-1].value != _VERIFIED_ACCEPTED:
        return f"{eq.artifact}: the latest verdict by a non-producer is verified={verdicts[-1].value} ({verdicts[-1].by})"
    return None


def instance_equivalence(inst: Mapping[str, Any], target: Target, equivs: Sequence[InstanceEquiv] | None,
                         result_producers: set[str]) -> tuple[str | None, str | None]:
    """``(art id, None)`` of an ``instance-equiv/v1`` artifact that shows the result's ``inst`` equal to the frozen set of ``target``
    and is ``verified=accepted`` by a non-producer; else ``(None, why)`` -- ``why`` names what the artifacts for this pair (same
    target and candidate ref) lack, ``None`` when there is none.  A verified document of the same pair that says ``equal`` is not
    true contradicts any acceptance."""

    pool = [e for e in equivs or () if e.doc.get("target") == target.name and _ref_key(e.doc.get("candidate")) == _ref_key(inst)]
    if not pool:
        return None, unreadable_equivs(equivs, target)
    judged = [(e, _equiv_content(e.doc, target, inst, e.unreadable), _equiv_verdict(e, result_producers)) for e in pool]
    contra = [e for e, _, v in judged if v is None and e.doc.get("equal") is not True]
    if contra:
        return None, f"{contra[0].artifact}, verified, says equal={contra[0].doc.get('equal')!r}"
    ok = [e for e, c, v in judged if not c and v is None]
    if ok:
        return ok[0].artifact, None
    return None, "; ".join(f"{e.artifact}: {c[0]}" if c else str(v) for e, c, v in judged)


def payload_json(store: Any, doc: Mapping[str, Any]) -> tuple[Any, str | None]:
    """``(the artifact's JSON payload, None)``, or ``(None, why)``: no file payload, ``payload not local`` (the manifest is in the
    catalog but its blob is not in this store: `research data fetch <art>`, or a render refresh with payloads), too large, or not
    JSON."""
    pay = doc.get("payload") if isinstance(doc.get("payload"), Mapping) else None
    if not pay or pay.get("type") != "file" or not isinstance(pay.get("sha256"), str):
        return None, "no file payload"
    try:
        path = store.object_path(pay["sha256"])
        if not path.is_file():
            return None, f"payload not local (blob {pay['sha256'][:12]}… is not in this store: `research data fetch`)"
        if path.stat().st_size > _PAYLOAD_JSON_MAX:
            return None, f"payload of {path.stat().st_size} bytes is over the {_PAYLOAD_JSON_MAX}-byte JSON limit"
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, ValueError, AttributeError) as e:
        return None, f"payload unreadable ({type(e).__name__})"


def unreadable_equivs(equivs: Sequence[InstanceEquiv] | None, target: Target) -> str | None:
    """The unreadable ``instance-equiv/v1`` artifacts that may be the target's (their meta names it, or names no target), as
    one reason -- a document that could not be read never matches a result, so say so instead of staying silent."""
    bad = [f"{e.artifact}: {e.unreadable}" for e in equivs or () if e.unreadable and e.meta.get("target") in (None, target.name)]
    return "; ".join(bad) if bad else None


def _payload_json(store: Any, doc: Mapping[str, Any]) -> Any:
    return payload_json(store, doc)[0]


def instance_equivs(store: Any) -> list[InstanceEquiv]:
    """Every ``instance-equiv/v1`` artifact in the store's catalog (by kind, or by ``meta.schema``), with its labels and attempt."""

    rows: dict[str, Mapping[str, Any]] = {}
    for sel in ({"kind": INSTANCE_EQUIV_SCHEMA}, {"where": {"schema": INSTANCE_EQUIV_SCHEMA}}):
        try:
            rows.update({r["id"]: r for r in store.index.select_artifacts(**sel)})
        except Exception:  # noqa: BLE001  (a store without a catalog, or without this facet)
            continue
    out = []
    for art_id, row in sorted(rows.items()):
        doc = row.get("manifest")
        if not isinstance(doc, Mapping):
            doc = store.get_manifest(art_id).to_doc()
        meta = doc.get("meta") or {}
        body, why = (meta, None) if meta.get("schema") == INSTANCE_EQUIV_SCHEMA else payload_json(store, doc)
        if body is not None and not isinstance(body, Mapping):
            body, why = None, "payload is not a JSON object"
        if body is None and why == "no file payload":
            body, why = meta, None                  # the meta is all there is: judge it as the document (its own schema says what is wrong)
        _, att = _attempt_of(store, art_id)
        out.append(InstanceEquiv(art_id, body if isinstance(body, Mapping) else meta, meta, tuple(store.labels(art_id)), att, unreadable=why))
    return out


def _contract_reason(problems: Sequence[str]) -> str:
    return f"contract: {problems[0]}" + (f" (+{len(problems) - 1} more)" if len(problems) > 1 else "")


# --- phases (Table 3) -------------------------------------------------------------------------------------------------------------


def phases(result: Mapping[str, Any]) -> dict[str, Any]:
    """``{"total", "buckets": {bucket: seconds | None}, "joints": {name: {"covers": [...], "seconds"}}, "other"}`` of a result:
    ``Other = t.total - sum(buckets) - sum(joints)`` (the frozen identity ``T_total = witness + encoding_commitment + arithmetic
    + lookup + zk_additional + serialization + other``).  A bucket covered by a joint is ``None``."""

    m = _measurements(result)
    total = m.get(contract.T_TOTAL)
    buckets: dict[str, float | None] = {b: (m.get(f"t.{b}") if _num(m.get(f"t.{b}")) else None) for b in contract.BUCKETS}
    joints: dict[str, dict[str, Any]] = {}
    for name, v in m.items():
        parts = contract.parse_joint(name) if name.startswith("t.joint.") else None
        if parts and _num(v):
            joints[name] = {"covers": list(parts), "seconds": v}
            for b in parts:
                buckets[b] = None
    summed = sum(v for v in buckets.values() if v is not None) + sum(j["seconds"] for j in joints.values())
    other = (total - summed) if _num(total) else None
    return {"total": total, "buckets": buckets, "joints": joints, "other": other}


# --- build -------------------------------------------------------------------------------------------------------------------------


@dataclass
class Cell:
    target: str
    candidate: str
    artifact: str
    attempt: str | None
    proof_class: str
    t_total: float
    r_proved: float
    overhead: float
    hardware: str | None
    phases: dict[str, Any]
    mode: str | None = None            #: the fingerprint's ``mode`` (``interactive`` | ``fiat-shamir``), when stated
    verified_by: str | None = None     #: who asserted the accepting verification label (a non-producer)
    host: str | None = None            #: the pod's host CPU (``hardware.cpu``): host-side prover phases are paid there
    authentication: str = AUTHENTICATION  #: the fingerprint's ``authentication`` (which Table 2 column the cell fills)
    hash_name: str | None = None       #: ``included-hash``: the fingerprint's ``hash`` (e.g. poseidon2-babybear-w24)
    sharing: str | None = None         #: ``included-hash``: the fingerprint's ``sharing`` (none | tile64x64)
    instances_equiv: str | None = None  #: re-packed instances: the accepted ``instance-equiv/v1`` artifact that makes them the frozen set
    finding: str | None = None         #: an open ``UNDER RE-VERIFICATION`` finding (:func:`cell_finding`): the cell carries a mark

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def verification_note(self) -> str:
        """The footnote's account of the verification: who accepted, and -- for an interactive transcript -- that the file
        verifier could only replay the runner's step-0 coins (D7: live-verifier-only evidence; the Fiat-Shamir row of the
        same run set is the transferable one)."""
        who = f"verified by {self.verified_by}" if self.verified_by else "verified"
        if self.mode == "interactive" and self.verified_by and self.verified_by.startswith("live-verifier"):
            return (f"accepted by the LIVE verifier {self.verified_by} (its own step-0 coins, independent of the prover; session "
                    f"recorded); interactive transcript, so the dump is evidence only together with that session (D7)")
        if self.mode == "interactive":
            return (f"{who} against the runner's step-0 coins (interactive transcript: not transferable evidence per D7; "
                    f"the Fiat–Shamir variant is)")
        if self.mode == "fiat-shamir":
            return f"{who} (Fiat–Shamir transcript: transferable)"
        return who


@dataclass
class Tables:
    rows: list[dict[str, Any]] = field(default_factory=list)          # Table 2 rows (one per target)
    cells: list[Cell] = field(default_factory=list)                    # populated Table 2 cells, row then candidate order
    drilldown: list[dict[str, Any]] = field(default_factory=list)      # best valid result in a candidate's drill-down class
    rejected: list[dict[str, Any]] = field(default_factory=list)       # every rejected candidate result with its reasons
    not_candidates: dict[str, list[str]] = field(default_factory=dict)  # backend name -> results that are no Table 1 candidate's
    also_valid: list[dict[str, Any]] = field(default_factory=list)     # valid results that were not the best
    snapshot: str | None = None


def _rate(t_total: float) -> float:
    return contract.proved_flop_per_second(t_total, B, K)


def _attempt_of(store: Any, art_id: str) -> tuple[str | None, Mapping[str, Any] | None]:
    """The producing attempt (from the catalog only: never a remote round-trip)."""

    try:
        rids = store.index.attempts_by_output(art_id)
    except Exception:  # noqa: BLE001  (a store without a catalog)
        return None, None
    for rid in rids:
        doc = store.index.attempt(rid)
        if isinstance(doc, Mapping):
            return rid, doc
    return (rids[0], None) if rids else (None, None)


def _artifacts(store: Any, kind: str) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for row in store.index.select_artifacts(kind=kind):
        doc = row.get("manifest")
        if not isinstance(doc, Mapping):
            doc = store.get_manifest(row["id"]).to_doc()
        out.append((row["id"], doc.get("meta") or {}))
    return out


def gemm_method(meta: Mapping[str, Any]) -> bool:
    """Whether a measurement's *primary* method (the clause before the first ``:`` / ``;`` of ``method``) is a dense GEMM through
    the vendor library (``cublas`` / ``cuBLASLt`` / ``gemm``), as opposed to an instruction-level microbenchmark (register-
    resident MMAs issued back to back, which can exceed the datasheet at sustained clocks above the rated boost -- lane
    fp4-proof's 1880 T on the RTX 5090 against 1676 T; its method text mentions GEMMs only to say it bounds them).  Both are
    kept in the store; the column prefers the GEMM so the rows share one methodology."""
    m = str(meta.get("method") or "").lower()
    head = m.replace(";", ":").split(":", 1)[0]
    return "gemm" in head or "cublas" in head


def _native_candidates(art_id: str, meta: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The measurements one ``native-peak/v1`` artifact contributes: its primary ``flop_per_second`` / ``method``, plus its
    ``gemm_attempt`` when the microbenchmark tool also ran the library GEMM beside the instruction (``supported`` with a
    positive ``flop_per_second``) -- that GEMM is the column's methodology and is cited under the same artifact id."""
    out = []
    flop = meta.get("flop_per_second")
    if _num(flop) and flop > 0:
        out.append({"artifact": art_id, "flop_per_second": flop, "method": meta.get("method"), "gemm": gemm_method(meta)})
    g = meta.get("gemm_attempt")
    if isinstance(g, Mapping) and g.get("supported") and _num(g.get("flop_per_second")) and g["flop_per_second"] > 0:
        how = ", ".join(str(x) for x in (g.get("variant"), f"n={g['n']}" if g.get("n") else None, g.get("note")) if x)
        out.append({"artifact": art_id, "flop_per_second": g["flop_per_second"], "gemm": True,
                    "method": f"library dense gemm recorded beside the instruction microbenchmark ({how}), median of {len(g.get('seconds_all') or []) or '?'}"})
    return out


def _native_measured(store: Any, target: Target) -> dict[str, Any] | None:
    """The best ``native-peak/v1`` measurement of the target's device and dtype, or ``None``: the max ``flop_per_second`` among
    the dense-GEMM measurements (:func:`gemm_method`, including a microbenchmark artifact's ``gemm_attempt``), falling back
    to the max of any method only when no GEMM was recorded (the footnote prints the method either way)."""

    best = None
    for art_id, meta in _artifacts(store, NATIVE_PEAK_KIND):
        if not same_sku(meta.get("device"), contract.native_peak_of(target).device) or native_dtype(meta) != dtype_of(target):
            continue
        acc = meta.get("accumulate")
        if acc is not None and acc != contract.native_peak_of(target).accumulate:
            continue
        for c in _native_candidates(art_id, meta):
            if best is None or (c["gemm"], c["flop_per_second"]) > (best["gemm"], best["flop_per_second"]):
                best = {**c, "fraction_of_spec": c["flop_per_second"] / contract.native_peak_of(target).flop_per_second, "device": meta.get("device"),
                        "dtype": meta.get("dtype"), "driver": meta.get("driver"), "cuda": meta.get("cuda"), "sass": meta.get("sass")}
    return best


def build(store: Any) -> Tables:
    """Read the store, apply the predicate, and assemble the three tables' data (rendering is separate)."""

    T = Tables()
    results = _artifacts(store, RESULT_KIND)
    equivs = instance_equivs(store)
    valid: dict[tuple[str, str, str], list[Cell]] = {}     # (target, candidate, authentication)
    drill: dict[tuple[str, str, str], list[Cell]] = {}
    for art_id, meta in results:
        fp = meta.get("workload_fingerprint") if isinstance(meta.get("workload_fingerprint"), Mapping) else {}
        backend = ((fp.get("software") or {}).get("backend") or {}).get("name") if isinstance(fp.get("software"), Mapping) else None
        cand = candidate_for(backend)
        target = TARGETS.get(fp.get("profile")) if isinstance(fp.get("profile"), str) else None
        if cand is None:
            T.not_candidates.setdefault(backend if isinstance(backend, str) else "(no software.backend.name)", []).append(art_id)
            continue
        rid, attempt = _attempt_of(store, art_id)
        entry = {"artifact": art_id, "candidate": cand.id, "target": target.name if target else None, "attempt": rid,
                 "backend": backend, "profile": fp.get("profile"), "proof_class": fp.get("proof_class")}
        if target is None:
            problems = contract.validate(meta)
            T.rejected.append({**entry, "reasons": [f"profile {fp.get('profile')!r} is not a target of verity.ml.circuits.targets.target"]
                               + ([_contract_reason(problems)] if problems else [])})
            continue
        labels = store.labels(art_id)
        auth = column_authentication(fp)
        reasons = reject_reasons(meta, labels, attempt, target, cand, authentication=auth, equivs=equivs)
        vb = independent_verification(meta, labels, attempt)
        verified_by = vb.by if vb is not None else None
        eq = instances_equiv_of(meta, labels, attempt, target, equivs)
        state, flab = cell_finding(labels)
        open_finding = finding_note(flab) if state == "open" else None
        if reasons:
            T.rejected.append({**entry, "reasons": reasons})
            for dc in cand.drilldown_classes:
                if not reject_reasons(meta, labels, attempt, target, cand, proof_classes=(dc,), authentication=auth, equivs=equivs):
                    drill.setdefault((target.name, cand.id, dc), []).append(_cell(meta, art_id, rid, target, cand, verified_by, eq, open_finding))
            continue
        valid.setdefault((target.name, cand.id, auth), []).append(_cell(meta, art_id, rid, target, cand, verified_by, eq, open_finding))

    for target in TARGETS.values():
        row: dict[str, Any] = {"target": target.name, "status": target.status, "device": contract.native_peak_of(target).device, "dtype": dtype_of(target),
                               "instruction": target.instruction, "native_spec_flop_per_second": contract.native_peak_of(target).flop_per_second,
                               "native_measured": _native_measured(store, target), "cells": {}, "committed": {}}
        for cand in CANDIDATES:
            for auth in AUTH_MODES:
                slot = row["cells"] if auth == AUTHENTICATION else row["committed"]
                pool = sorted(valid.get((target.name, cand.id, auth), []), key=lambda c: (c.t_total, c.artifact))
                if pool:
                    slot[cand.id] = pool[0]
                    T.cells.append(pool[0])
                    for c in pool[1:]:
                        T.also_valid.append({"artifact": c.artifact, "candidate": cand.id, "target": target.name, "authentication": auth,
                                             "t_total": c.t_total, "best": pool[0].artifact})
                else:
                    slot[cand.id] = None
            for dc in cand.drilldown_classes:
                dpool = sorted(drill.get((target.name, cand.id, dc), []), key=lambda c: (c.t_total, c.artifact))
                if dpool:
                    T.drilldown.append({"target": target.name, "candidate": cand.id, "proof_class": dc, **dpool[0].to_json()})
        T.rows.append(row)
    return T


def column_authentication(fp: Mapping[str, Any]) -> str:
    """The authentication mode of the Table 2 column that judges a result: its fingerprint's when that names a column
    (:data:`AUTH_MODES`), else the bare column's (where ``included`` / ``included-hash-shared`` are then rejected)."""
    a = fp.get("authentication")
    return a if a in AUTH_MODES else AUTHENTICATION


def _mode_of(fp: Mapping[str, Any]) -> str | None:
    """``interactive`` | ``fiat-shamir`` from the fingerprint: an explicit ``mode`` when the emitter states one, else the
    ``security.bound_model`` prefix B-Ligero's emitters write (``interactive: ...`` / ``non-interactive: ...``)."""
    m = fp.get("mode")
    if isinstance(m, str) and m in ("interactive", "fiat-shamir"):
        return m
    sec = fp.get("security") if isinstance(fp.get("security"), Mapping) else {}
    bm = sec.get("bound_model")
    if isinstance(bm, str):
        if bm.startswith("interactive"):
            return "interactive"
        if bm.startswith("non-interactive"):
            return "fiat-shamir"
    return None


def _cell(meta: Mapping[str, Any], art_id: str, rid: str | None, target: Target, cand: Candidate, verified_by: str | None = None,
          instances_equiv: str | None = None, finding: str | None = None) -> Cell:
    fp = meta["workload_fingerprint"]
    t_total = float(_measurements(meta)[contract.T_TOTAL])
    r = _rate(t_total)
    gpu = (fp.get("hardware") or {}).get("gpu") or {}
    mode = _mode_of(fp)
    cpu = (fp.get("hardware") or {}).get("cpu") or {}
    host = None
    if isinstance(cpu, Mapping) and isinstance(cpu.get("model"), str):
        host = cpu["model"] + (f" x{cpu['count']}" if _num(cpu.get("count")) else "")
    auth = column_authentication(fp)
    return Cell(target=target.name, candidate=cand.id, artifact=art_id, attempt=rid, proof_class=fp["proof_class"], t_total=t_total, r_proved=r,
                overhead=contract.native_peak_of(target).flop_per_second / r, hardware=gpu.get("name"), phases=phases(meta), mode=mode, verified_by=verified_by,
                host=host, authentication=auth, hash_name=fp.get("hash") if isinstance(fp.get("hash"), str) else None,
                sharing=fp.get("sharing") if isinstance(fp.get("sharing"), str) else None, instances_equiv=instances_equiv, finding=finding)


def cell_artifacts(T: Tables) -> list[str]:
    """The artifacts a snapshot freezes: every populated Table 2 candidate cell and every native-measured cell."""

    ids = [c.artifact for c in T.cells] + [r["native_measured"]["artifact"] for r in T.rows if r.get("native_measured")]
    return sorted(set(ids))


def take_snapshot(store: Any, T: Tables) -> str:
    """``--snapshot``: freeze :func:`cell_artifacts` as the ``dataset-snapshot/v1`` named :func:`snapshot_name`; -> its art id."""
    return store.snapshot(snapshot_name(), cell_artifacts(T), note=f"canonical tables (frozen {FROZEN_ON}): the Table 2 cell artifacts")


def snapshot_name(now: _dt.datetime | None = None) -> str:
    now = now or _dt.datetime.now(_dt.UTC)
    return f"tables-{now.astimezone(_dt.UTC).strftime('%Y-%m-%d')}"


# --- rendering ------------------------------------------------------------------------------------------------------------------------

TABLE2_COLUMNS = ("Device", "Datatype / target", "Native dense peak, spec", "Native dense peak, measured", *(c.id for c in CANDIDATES))


def committed_candidates(T: "Tables") -> list["Candidate"]:
    """The candidates with at least one populated ``included-hash`` cell: each gets a ``<id> + in-proof hash`` column after the
    bare columns (a candidate without committed runs does not get an empty column)."""
    return [c for c in CANDIDATES if any(r["committed"].get(c.id) for r in T.rows)]


def table2_columns(T: "Tables") -> tuple[str, ...]:
    return (*TABLE2_COLUMNS, *(c.id + AUTH_COLUMN_SUFFIX["included-hash"] for c in committed_candidates(T)))
TABLE3_COLUMNS = ("Device / target", "Candidate", "Total prover time", "Witness", "Encoding + commitment", "Arithmetic", "Lookup",
                  "ZK additional", "Serialization", "Other")
_BUCKET_COLUMN = dict(zip(contract.BUCKETS, TABLE3_COLUMNS[3:9]))
EMPTY = "—"
UNMEASURED = "…"


def fmt_overhead(x: float) -> str:
    """``2.6e7×``: one decimal of mantissa, a bare exponent."""

    m, e = f"{x:.1e}".split("e")
    return f"{m}e{int(e)}×"


def fmt_seconds(x: float) -> str:
    if abs(x) >= 1000:
        return f"{x:.0f} s"
    return f"{x:.3g} s"


def _fmt_tflops(flop: float) -> str:
    t = flop / 1e12
    return f"{t:.0f} T" if t >= 10 else f"{t:.1f} T"


def _md_escape(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def _md_table(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "| " + " | ".join(_md_escape(c) for c in columns) + " |"
    sep = "|" + "|".join(" --- " for _ in columns) + "|"
    body = ["| " + " | ".join(_md_escape(str(c)) for c in r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def _phase_cells(ph: Mapping[str, Any]) -> list[str]:
    """The six bucket columns + Other of a Table 3 row.  A joint prints once, ``a+b: 0.41 s (37%)``, in the first column it
    covers; the other covered columns hold :data:`JOINT_CONTINUATION`."""

    total = ph["total"]
    pct = lambda v: f"{100 * v / total:.0f}%" if _num(total) and total > 0 else "?%"  # noqa: E731
    cells: dict[str, str] = {}
    for b, v in ph["buckets"].items():
        if v is not None:
            cells[b] = f"{fmt_seconds(v)} ({pct(v)})"
    for j in ph["joints"].values():
        first, *rest = j["covers"]
        cells[first] = f"{'+'.join(j['covers'])}: {fmt_seconds(j['seconds'])} ({pct(j['seconds'])})"
        for b in rest:
            cells[b] = JOINT_CONTINUATION
    out = [cells.get(b, EMPTY) for b in contract.BUCKETS]
    other = ph["other"]
    out.append(f"{fmt_seconds(other)} ({pct(other)})" if _num(other) else EMPTY)
    return out


def caption_text() -> str:
    return (f"K = {K}; B = {B}; soundness 2^{SOUNDNESS_LOG2} (target and achieved, union-bounded over sub-batches); dense execution; "
            f"FP32 accumulation; FMA = 2 FLOP; the frozen input set of the row's target; native denominator = datasheet dense peak of the "
            f"exact target device (`Target.native_peak`); authentication = {AUTHENTICATION} in the candidate columns, included-hash in the "
            f"' + in-proof hash' columns (Poseidon2-BabyBear hashing of the operand rows inside the proof, digests bound to the Merkle roots; "
            f"x, W private). Cell = best valid implementation of the "
            f"candidate in its declared proof class (min `t.total`), run on the row's device SKU, printed as R_native_spec / R_proved with "
            f"R_proved = 2KB / t.total. Native measured = `{NATIVE_PEAK_KIND}` ({NATIVE_PEAK_METHOD}), `{UNMEASURED}` when absent; "
            f"`{EMPTY}` = no valid result; `({UNDER_REVERIFICATION})` = the latest `finding` label on the cell's artifact is open "
            f"(footnote), and a `PULLED` finding rejects the result.")


def render_markdown(T: Tables) -> str:
    out: list[str] = [f"## Table 1 — Candidate proof systems (frozen {FROZEN_ON})", "",
                      _md_table(TABLE1_COLUMNS, [c.table1_row() for c in CANDIDATES]), "",
                      "## Table 2 — Headline proving overhead by target", "", caption_text(), ""]
    notes: list[str] = []

    def note(text: str) -> str:
        notes.append(text)
        return f"[{len(notes)}]"

    rows2: list[list[str]] = []
    for r in T.rows:
        nm = r["native_measured"]
        measured = UNMEASURED
        if nm:
            how = nm.get("method") or "method not recorded"
            if not nm.get("gemm"):
                how = f"INSTRUCTION-LEVEL MICROBENCHMARK, not a GEMM (no GEMM measurement of this device x dtype in the store): {how}"
            measured = f"{_fmt_tflops(nm['flop_per_second'])} ({nm['fraction_of_spec']:.2f}) " + note(
                f"{nm['artifact']} — native peak of {nm['device']} {nm['dtype']}: {nm['flop_per_second']:.4g} FLOP/s, {how}")
        line = [r["device"], f"{(r['dtype'] or '?').upper()} {r['instruction'] or ''} · {r['target']}".strip(),
                _fmt_tflops(r["native_spec_flop_per_second"]), measured]
        def cell_text(c: Cell | None, cand: Candidate) -> str:
            if c is None:
                return EMPTY
            extra = ""
            if c.authentication != AUTHENTICATION:
                extra = f"; authentication {c.authentication} (hash {c.hash_name or '?'}, sharing {c.sharing or '?'})"
            if c.instances_equiv:
                extra += f"; {INSTANCE_EQUIV_NOTE} ({c.instances_equiv})"
            if c.finding:
                extra += f"; {c.finding}"
            return fmt_overhead(c.overhead) + (f" ({UNDER_REVERIFICATION})" if c.finding else "") + " " + note(
                f"{c.artifact} — {cand.id} {c.proof_class} on {c.hardware}" + (f" (host {c.host})" if c.host else "") +
                f"; attempt {c.attempt or 'unknown'}; t.total {c.t_total:.4g} s; R_proved {c.r_proved:.4g} FLOP/s{extra}; {c.verification_note()}")

        for cand in CANDIDATES:
            line.append(cell_text(r["cells"].get(cand.id), cand))
        for cand in committed_candidates(T):
            line.append(cell_text(r["committed"].get(cand.id), cand))
        rows2.append(line)
    out += [_md_table(table2_columns(T), rows2), ""]
    for d in T.drilldown:
        out.append(f"- drill-down {d['candidate']} {d['proof_class']} on {d['target']}: {fmt_overhead(d['overhead'])} "
                   f"({d['artifact']}, t.total {d['t_total']:.4g} s)")
    if T.drilldown:
        out.append("")
    out += ["## Table 3 — Prover phase decomposition", "",
            f"T_total = witness + encoding_commitment + arithmetic + lookup + zk_additional + serialization + other; Other = t.total − Σ buckets − Σ joints "
            f"(≥ −1 %·t.total). A fused stage `t.joint.a+b` prints once as `a+b: 0.41 s (37%)` in the first bucket column it covers; "
            f"`{JOINT_CONTINUATION}` marks the other covered columns.", ""]
    rows3 = []
    by_target = {r["target"]: r for r in T.rows}
    for c in T.cells:
        rows3.append([f"{by_target[c.target]['device']} / {c.target}", c.candidate + AUTH_COLUMN_SUFFIX.get(c.authentication, ""),
                      fmt_seconds(c.t_total), *_phase_cells(c.phases)])
    out += [_md_table(TABLE3_COLUMNS, rows3) if rows3 else f"(no populated Table 2 cell {EMPTY} nothing to decompose)", ""]
    if notes:
        out += ["Footnotes:", *[f"[{i}] {n}" for i, n in enumerate(notes, 1)], ""]
    out.append(f"Rejected ({len(T.rejected)} candidate result{'s' if len(T.rejected) != 1 else ''}; artifact → first reason):")
    for r in T.rejected:
        who = f"{r['candidate']} / {r['target'] or r['profile'] or '?'}"
        out.append(f"- {r['artifact']} ({who}) → {r['reasons'][0]}")
    if T.not_candidates:
        n = sum(len(v) for v in T.not_candidates.values())
        out += ["", f"Not a Table 1 candidate ({n} result{'s' if n != 1 else ''}: component benchmarks and other backends, by software.backend.name):"]
        for name, arts in sorted(T.not_candidates.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            out.append(f"- {name} ×{len(arts)}")
    if T.also_valid:
        out.append("")
        out.append("Valid but not the best (superseded by the cell's artifact):")
        for r in T.also_valid:
            out.append(f"- {r['artifact']} ({r['candidate']} / {r['target']}) t.total {r['t_total']:.4g} s → {r['best']}")
    if T.snapshot:
        out += ["", f"Snapshot: {T.snapshot}"]
    return "\n".join(out) + "\n"


def render_json(T: Tables) -> dict[str, Any]:
    rows = []
    for r in T.rows:
        rows.append({**{k: v for k, v in r.items() if k not in ("cells", "committed")},
                     "cells": {k: (c.to_json() if c else None) for k, c in r["cells"].items()},
                     "committed": {k: (c.to_json() if c else None) for k, c in r["committed"].items()}})
    return {
        "schema": "verity/canonical-tables/v1",
        "frozen": FROZEN_ON,
        "caption": CAPTION,
        "frozen_instances": FROZEN_INSTANCES,
        "table1": {"columns": list(TABLE1_COLUMNS), "rows": [c.table1_row() for c in CANDIDATES],
                   "candidates": [{"id": c.id, "proof_classes": list(c.proof_classes), "drilldown_classes": list(c.drilldown_classes),
                                   "backend_names": list(c.backend_names)} for c in CANDIDATES]},
        "table2": {"columns": list(table2_columns(T)), "rows": rows, "authentication_modes": list(AUTH_MODES)},
        "table3": {"columns": list(TABLE3_COLUMNS), "identity": "T_total = witness + encoding_commitment + arithmetic + lookup + zk_additional + serialization + other",
                   "rows": [{"target": c.target, "candidate": c.candidate, "artifact": c.artifact, **c.phases} for c in T.cells]},
        "drilldown": T.drilldown,
        "rejected": T.rejected,
        "not_candidates": T.not_candidates,
        "also_valid": T.also_valid,
        "snapshot": T.snapshot,
    }


# --- CLI ---------------------------------------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, metavar="ROOT", help="store root (default $RESEARCH_STORE or ~/.research/store)")
    ap.add_argument("--format", choices=("md", "json"), default="md")
    ap.add_argument("--snapshot", action="store_true", help="freeze the cell artifacts into a dataset-snapshot/v1 named tables-<UTC date>")
    ns = ap.parse_args(argv)
    from research.store.local import LocalStore  # noqa: PLC0415  (only the CLI needs the store package)

    st = LocalStore(ns.root)
    T = build(st)
    if ns.snapshot:
        members = cell_artifacts(T)
        if not members:
            print("tables: no populated cell, no snapshot written", file=sys.stderr)
            return 1
        T.snapshot = take_snapshot(st, T)
    if ns.format == "json":
        print(json.dumps(render_json(T), indent=2, sort_keys=True, default=str))
    else:
        print(render_markdown(T), end="")
    if T.snapshot:
        print(f"snapshot {T.snapshot} ({len(cell_artifacts(T))} member(s))", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
