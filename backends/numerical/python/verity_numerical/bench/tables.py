"""The proof campaign's three canonical result tables, frozen 2026-09-22 (``backends/AGENTS.md``, "Canonical tables").

    python -m verity_numerical.bench.tables --root <store root> [--format md|json] [--snapshot]

Reads ``bench-result/v1`` and ``native-peak/v1`` artifacts and their labels from the local evidence store
(``research.store.LocalStore``), applies the frozen validity predicate (:func:`reject_reasons`) and renders:

* **Table 1** -- the candidate proof systems (:data:`CANDIDATES`): definitional, no performance, no status;
* **Table 2** -- headline proving overhead by target: one row per ``verity.verification.target.TARGETS`` entry, one
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

from verity.ml.tc.instructions import INSTRUCTIONS
from verity.verification.target import BF16_HOPPER, FIRST, FP8_ADA_MMA, FP8_HOPPER_WGMMA, NVFP4_SM120, TARGETS, Target
from verity_numerical.bench import contract

try:  # the label keys are pinned with the store (`research.store.vocab`, tools/research); verity-numerical does not depend on that package
    from research.store import vocab as _vocab

    _CANDIDATE, _LABEL, _PROOF_CLASS = _vocab.CANDIDATE, _vocab.LABEL, _vocab.PROOF_CLASS
    _INDEPENDENTLY_VERIFIED, _VERIFIED, _VERIFIED_ACCEPTED = _vocab.INDEPENDENTLY_VERIFIED, _vocab.VERIFIED, _vocab.VERIFIED_ACCEPTED
    _KEY_ALIASES, _VALUE_ALIASES = dict(_vocab.KEY_ALIASES), dict(_vocab.VALUE_ALIASES)
except ImportError:  # pragma: no cover  -- the same spellings, so the predicate still imports where the store package is absent (tested equal)
    _CANDIDATE, _LABEL, _PROOF_CLASS = "candidate", "label", "proof_class"
    _INDEPENDENTLY_VERIFIED, _VERIFIED, _VERIFIED_ACCEPTED = "independently_verified", "verified", "accepted"
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
    "RANK",
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
    "cell_artifacts",
    "dtype_of",
    "fmt_overhead",
    "fmt_seconds",
    "frozen_instances",
    "main",
    "normalise_sku",
    "phases",
    "producers",
    "reject_reasons",
    "render_json",
    "render_markdown",
    "same_sku",
    "device_variant",
    "snapshot_name",
]

# --- caption constants (frozen) -----------------------------------------------------------------------------------------

FROZEN_ON = "2026-09-22"
K = 1536
B = 4096
#: log2 of the soundness error: the target AND the achieved (union-bounded over sub-batches) must both be <= this.
SOUNDNESS_LOG2 = -128
AUTHENTICATION = "excluded"
CAPTION: dict[str, Any] = {
    "K": K,
    "B": B,
    "soundness": f"2^{SOUNDNESS_LOG2} (target and achieved, union-bounded over sub-batches)",
    "execution": "dense",
    "accumulation": "fp32",
    "flop_per_fma": 2,
    "instances": "the frozen instance set of the row's target (FROZEN_INSTANCES)",
    "native_denominator": "datasheet dense tensor-core peak of the exact target device (Target.native_peak)",
    "authentication": AUTHENTICATION + " (stated here until `included` runs exist)",
    "cell": "best valid implementation of the candidate in its declared proof class: min t.total; printed as R_native_spec / R_proved, R_proved = 2KB / t.total",
    "same_sku": "the prover ran on the same physical device SKU as the row's native device (hardware.gpu.name ~ native_peak.device)",
    "device_variant": "the board is the reference part: hardware.gpu.memory_total_bytes within 12 % of native_peak.memory_bytes (a same-named re-memoried board is not)",
}

RESULT_KIND = "bench-result/v1"
NATIVE_PEAK_KIND = "native-peak/v1"
#: ``tools/native_peak`` 's canonical GEMM method string (the shape every GEMM measurement's ``method`` starts with).
NATIVE_PEAK_METHOD_CUBLAS = "cublas dense gemm 8192^3, median of 5"
#: What the measured column is, for the caption (selection rule in :func:`_native_measured`).
NATIVE_PEAK_METHOD = ("the best dense GEMM through the vendor library on the row's device x dtype, median of 5, the call and size in the "
                      "footnote; an instruction-level microbenchmark only when no GEMM measurement exists, and then said so")
#: The BF16 frozen set: sha256 of ``fixtures/bench-instances/v1/manifest.json`` (``verity_numerical.bench.instances.manifest_sha256``).
BF16_INSTANCES_MANIFEST_SHA256 = "059103cf9bd55ee83cbd4bb14ae6db2f60db2cb4ddf85cdc22b1cecee6e4eeea"
#: The frozen instance ref a valid result of each target must carry, ``instances`` field for field.  A target absent here
#: has no frozen set yet: every result of it is rejected with that reason.  The synthetic sets (FP8 Ada, the two Hopper
#: relations) are pinned by their recipe digest (``contract.*_INSTANCES_DIGEST``), the BF16 Ampere set by its fixture manifest.
FROZEN_INSTANCES: dict[str, dict[str, Any]] = {
    FIRST.name: contract.instances_ref("vu-k1536", 0, B, BF16_INSTANCES_MANIFEST_SHA256),
    FP8_ADA_MMA.name: contract.instances_ref(contract.FP8_ADA_INSTANCES_TIER, 0, B, contract.FP8_ADA_INSTANCES_DIGEST,
                                             dataset=contract.FP8_ADA_INSTANCES_DATASET),
    BF16_HOPPER.name: contract.instances_ref(contract.BF16_HOPPER_INSTANCES_TIER, 0, B, contract.BF16_HOPPER_INSTANCES_DIGEST,
                                             dataset=contract.BF16_HOPPER_INSTANCES_DATASET),
    FP8_HOPPER_WGMMA.name: contract.instances_ref(contract.FP8_HOPPER_INSTANCES_TIER, 0, B, contract.FP8_HOPPER_INSTANCES_DIGEST,
                                                  dataset=contract.FP8_HOPPER_INSTANCES_DATASET),
    NVFP4_SM120.name: contract.instances_ref(contract.NVFP4_INSTANCES_TIER, 0, B, contract.NVFP4_INSTANCES_DIGEST,
                                             dataset=contract.NVFP4_INSTANCES_DATASET),
}

#: Label keys whose asserter is the *producer* of a result (the lane's own row labels, ``ledger_to_store``); a verification
#: label by any of them, or by the producing attempt's ``execution.campaign``, is a self-assertion and does not count.
PRODUCER_LABEL_KEYS = (_CANDIDATE, _LABEL)
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
    by a non-alphanumeric character: ``"b-ligero chain (backends/direct/ligero)"`` and ``"b-ligero chain, fp8-ada relation
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
        assumptions="collision-resistant hash (BLAKE3 Merkle); Fiat–Shamir in the random-oracle model for the non-interactive variant; "
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

    ref = target.native_peak.memory_bytes
    mem = (gpu or {}).get("memory_total_bytes")
    if not ref or not _num(mem) or mem <= 0:
        return None
    if abs(mem / ref - 1) > MEMORY_TOLERANCE:
        return (f"device variant: {mem / 2**30:.1f} GiB of device memory on {(gpu or {}).get('name')!r}, the reference "
                f"{target.native_peak.device!r} has {ref / 2**30:.0f} GiB -- not the datasheet part the row divides by")
    return None


def same_sku(a: Any, b: Any) -> bool:
    """Whether two device names normalise to the same SKU (``False`` when either cannot be normalised)."""

    na, nb = normalise_sku(a), normalise_sku(b)
    return na is not None and na == nb


def dtype_of(target: Target) -> str | None:
    """The target's operand dtype (``bf16``, ``e4m3``, ``e2m1``): the instruction registry's ``a_dtype``."""

    ins = INSTRUCTIONS.get(target.instruction) if target.instruction else None
    return ins.a_dtype if ins else None


#: ``native-peak/v1`` measurement dtypes that name a block-scaled *format* over an operand dtype: ``tools/native_peak
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


def producers(attempt: Mapping[str, Any] | None, labels: Iterable[Any]) -> set[str]:
    """The asserters that are the result's own producer: the producing attempt's ``execution.campaign`` and whoever asserted
    a :data:`PRODUCER_LABEL_KEYS` label on it (the lane's row labels).  A verification by one of these is a self-assertion."""

    out: set[str] = set()
    if isinstance(attempt, Mapping):
        camp = (attempt.get("execution") or {}).get("campaign") if isinstance(attempt.get("execution"), Mapping) else None
        if isinstance(camp, str) and camp:
            out.add(camp)
    for lab in _labels(labels):
        if lab.key in PRODUCER_LABEL_KEYS:
            out.add(lab.by)
    return out


def _verified_by(labels: Sequence[_Label], self_: set[str]) -> _Label | None:
    for lab in labels:
        if lab.by in self_:
            continue
        for key, accepted in VERIFIED_LABELS:
            if lab.key == key and lab.value in accepted:
                return lab
    return None


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


# --- the validity predicate -------------------------------------------------------------------------------------------------


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _measurements(result: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for m in result.get("measurements") or []:
        if isinstance(m, Mapping) and isinstance(m.get("name"), str):
            out[m["name"]] = m.get("value")
    return out


def reject_reasons(result: Mapping[str, Any], labels: Iterable[Any], attempt: Mapping[str, Any] | None, target: Target,
                   candidate: Candidate, proof_classes: Sequence[str] | None = None) -> list[str]:
    """Every reason ``result`` (a ``bench-result/v1`` meta: the result.json) is NOT a valid Table 2 cell for ``target`` x
    ``candidate``; ``[]`` means valid.  ``proof_classes`` overrides the candidate's declared classes (drill-down).  Frozen:

    contract ``validate()`` empty; ``profile`` == the target; ``K == 1536``, ``B == 4096`` (and ``B_real == B`` when present);
    ``security.target <= -128`` and ``security.achieved_log2 <= -128`` when present; ``instances`` equal to the target's frozen
    ref; ``proof_class`` in the candidate's declared class(es); ``authentication == "excluded"``; the same-SKU invariant;
    independently verified (a :data:`VERIFIED_LABELS` label by a non-producer); not downgraded by a ``red-team-*`` verdict."""

    p: list[str] = []
    problems = contract.validate(result)
    if problems:
        p.append(_contract_reason(problems))
    fp = result.get("workload_fingerprint") if isinstance(result, Mapping) else None
    if not isinstance(fp, Mapping):
        return p + ["workload_fingerprint missing"]
    if fp.get("profile") != target.name:
        p.append(f"profile {fp.get('profile')!r} is not the row's target {target.name!r}")
    if fp.get("K") != K:
        p.append(f"K is {fp.get('K')!r}, the tables are at K = {K}")
    if fp.get("B") != B:
        p.append(f"B is {fp.get('B')!r}, the tables are at B = {B}")
    if "B_real" in fp and fp.get("B_real") != fp.get("B"):
        p.append(f"B_real {fp.get('B_real')!r} != B {fp.get('B')!r}: a padded batch is not the frozen workload")
    sec = fp.get("security") if isinstance(fp.get("security"), Mapping) else {}
    tgt = sec.get("target")
    if not _num(tgt) or tgt > SOUNDNESS_LOG2:
        p.append(f"security.target is {tgt!r}, need <= {SOUNDNESS_LOG2}")
    ach = sec.get("achieved_log2")
    if ach is not None and (not _num(ach) or ach > SOUNDNESS_LOG2):
        p.append(f"security.achieved_log2 is {ach!r}, need <= {SOUNDNESS_LOG2} (union bound over sub-batches)")
    want = frozen_instances(target)
    inst = fp.get("instances")
    if want is None:
        p.append(f"target {target.name!r} has no frozen instance set yet")
    elif not isinstance(inst, Mapping):
        p.append("instances is null: not measured on the frozen instance set")
    else:
        diff = [k for k in ("dataset", "tier", "range", "manifest_sha256") if _canon(inst.get(k)) != _canon(want[k])]
        if diff:
            p.append("instances differ from the frozen set on " + ", ".join(f"{k} ({inst.get(k)!r} vs {want[k]!r})" for k in diff))
    classes = tuple(proof_classes) if proof_classes else candidate.proof_classes
    pc = fp.get("proof_class")
    if pc not in classes:
        p.append(f"proof_class {pc!r} is not {candidate.id}'s declared class {' | '.join(classes)}")
    if fp.get("authentication") != AUTHENTICATION:
        p.append(f"authentication {fp.get('authentication')!r}: the caption states {AUTHENTICATION!r}")
    hw = fp.get("hardware") if isinstance(fp.get("hardware"), Mapping) else {}
    gpu = hw.get("gpu") if isinstance(hw.get("gpu"), Mapping) else None
    gpu_name = gpu.get("name") if gpu else None
    if not same_sku(gpu_name, target.native_peak.device):
        p.append(f"same-SKU: prover ran on {gpu_name!r}, the row's native device is {target.native_peak.device!r}")
    variant = device_variant(gpu, target)
    if variant:
        p.append(variant)
    labs = _labels(labels)
    self_ = producers(attempt, labs)
    if _verified_by(labs, self_) is None:
        keys = " or ".join(f"{k}={'|'.join(map(str, v))}" for k, v in VERIFIED_LABELS)
        p.append(f"not independently verified: no label {keys} by a non-producer (producers: {', '.join(sorted(self_)) or 'unknown'})")
    ceiling, verdict = _redteam_ceiling(labs, self_)
    if verdict is not None and isinstance(pc, str) and RANK.get(ceiling or "", -1) < RANK.get(pc, -1):
        p.append(f"downgraded: {verdict.by} ({verdict.ref or 'no ref'}, {verdict.ts}) asserts proof_class={ceiling}, below {pc}")
    return p


def _canon(v: Any) -> Any:
    return list(v) if isinstance(v, (list, tuple)) else v


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

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def verification_note(self) -> str:
        """The footnote's account of the verification: who accepted, and -- for an interactive transcript -- that the file
        verifier could only replay the runner's step-0 coins (D7: live-verifier-only evidence; the Fiat-Shamir row of the
        same run set is the transferable one)."""
        who = f"verified by {self.verified_by}" if self.verified_by else "verified"
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
        if not same_sku(meta.get("device"), target.native_peak.device) or native_dtype(meta) != dtype_of(target):
            continue
        acc = meta.get("accumulate")
        if acc is not None and acc != target.native_peak.accumulate:
            continue
        for c in _native_candidates(art_id, meta):
            if best is None or (c["gemm"], c["flop_per_second"]) > (best["gemm"], best["flop_per_second"]):
                best = {**c, "fraction_of_spec": c["flop_per_second"] / target.native_peak.flop_per_second, "device": meta.get("device"),
                        "dtype": meta.get("dtype"), "driver": meta.get("driver"), "cuda": meta.get("cuda"), "sass": meta.get("sass")}
    return best


def build(store: Any) -> Tables:
    """Read the store, apply the predicate, and assemble the three tables' data (rendering is separate)."""

    T = Tables()
    results = _artifacts(store, RESULT_KIND)
    valid: dict[tuple[str, str], list[Cell]] = {}
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
            T.rejected.append({**entry, "reasons": [f"profile {fp.get('profile')!r} is not a target of verity.verification.target"]
                               + ([_contract_reason(problems)] if problems else [])})
            continue
        labels = store.labels(art_id)
        reasons = reject_reasons(meta, labels, attempt, target, cand)
        labs = _labels(labels)
        vb = _verified_by(labs, producers(attempt, labs))
        verified_by = vb.by if vb is not None else None
        if reasons:
            T.rejected.append({**entry, "reasons": reasons})
            for dc in cand.drilldown_classes:
                if not reject_reasons(meta, labels, attempt, target, cand, proof_classes=(dc,)):
                    drill.setdefault((target.name, cand.id, dc), []).append(_cell(meta, art_id, rid, target, cand, verified_by))
            continue
        valid.setdefault((target.name, cand.id), []).append(_cell(meta, art_id, rid, target, cand, verified_by))

    for target in TARGETS.values():
        row: dict[str, Any] = {"target": target.name, "status": target.status, "device": target.native_peak.device, "dtype": dtype_of(target),
                               "instruction": target.instruction, "native_spec_flop_per_second": target.native_peak.flop_per_second,
                               "native_measured": _native_measured(store, target), "cells": {}}
        for cand in CANDIDATES:
            pool = sorted(valid.get((target.name, cand.id), []), key=lambda c: (c.t_total, c.artifact))
            if pool:
                row["cells"][cand.id] = pool[0]
                T.cells.append(pool[0])
                for c in pool[1:]:
                    T.also_valid.append({"artifact": c.artifact, "candidate": cand.id, "target": target.name, "t_total": c.t_total,
                                         "best": pool[0].artifact})
            else:
                row["cells"][cand.id] = None
            for dc in cand.drilldown_classes:
                dpool = sorted(drill.get((target.name, cand.id, dc), []), key=lambda c: (c.t_total, c.artifact))
                if dpool:
                    T.drilldown.append({"target": target.name, "candidate": cand.id, "proof_class": dc, **dpool[0].to_json()})
        T.rows.append(row)
    return T


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


def _cell(meta: Mapping[str, Any], art_id: str, rid: str | None, target: Target, cand: Candidate, verified_by: str | None = None) -> Cell:
    fp = meta["workload_fingerprint"]
    t_total = float(_measurements(meta)[contract.T_TOTAL])
    r = _rate(t_total)
    gpu = (fp.get("hardware") or {}).get("gpu") or {}
    mode = _mode_of(fp)
    cpu = (fp.get("hardware") or {}).get("cpu") or {}
    host = None
    if isinstance(cpu, Mapping) and isinstance(cpu.get("model"), str):
        host = cpu["model"] + (f" x{cpu['count']}" if _num(cpu.get("count")) else "")
    return Cell(target=target.name, candidate=cand.id, artifact=art_id, attempt=rid, proof_class=fp["proof_class"], t_total=t_total, r_proved=r,
                overhead=target.native_peak.flop_per_second / r, hardware=gpu.get("name"), phases=phases(meta), mode=mode, verified_by=verified_by,
                host=host)


def cell_artifacts(T: Tables) -> list[str]:
    """The artifacts a snapshot freezes: every populated Table 2 candidate cell and every native-measured cell."""

    ids = [c.artifact for c in T.cells] + [r["native_measured"]["artifact"] for r in T.rows if r.get("native_measured")]
    return sorted(set(ids))


def snapshot_name(now: _dt.datetime | None = None) -> str:
    now = now or _dt.datetime.now(_dt.UTC)
    return f"tables-{now.astimezone(_dt.UTC).strftime('%Y-%m-%d')}"


# --- rendering ------------------------------------------------------------------------------------------------------------------------

TABLE2_COLUMNS = ("Device", "Datatype / target", "Native dense peak, spec", "Native dense peak, measured", *(c.id for c in CANDIDATES))
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
            f"FP32 accumulation; FMA = 2 FLOP; the frozen instance set of the row's target; native denominator = datasheet dense peak of the "
            f"exact target device (`Target.native_peak`); authentication = {AUTHENTICATION}. Cell = best valid implementation of the "
            f"candidate in its declared proof class (min `t.total`), run on the row's device SKU, printed as R_native_spec / R_proved with "
            f"R_proved = 2KB / t.total. Native measured = `{NATIVE_PEAK_KIND}` ({NATIVE_PEAK_METHOD}), `{UNMEASURED}` when absent; "
            f"`{EMPTY}` = no valid result.")


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
        for cand in CANDIDATES:
            c: Cell | None = r["cells"].get(cand.id)
            if c is None:
                line.append(EMPTY)
            else:
                line.append(fmt_overhead(c.overhead) + " " + note(
                    f"{c.artifact} — {cand.id} {c.proof_class} on {c.hardware}" + (f" (host {c.host})" if c.host else "") +
                    f"; attempt {c.attempt or 'unknown'}; t.total {c.t_total:.4g} s; R_proved {c.r_proved:.4g} FLOP/s; {c.verification_note()}"))
        rows2.append(line)
    out += [_md_table(TABLE2_COLUMNS, rows2), ""]
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
        rows3.append([f"{by_target[c.target]['device']} / {c.target}", c.candidate, fmt_seconds(c.t_total), *_phase_cells(c.phases)])
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
        rows.append({**{k: v for k, v in r.items() if k != "cells"}, "cells": {k: (c.to_json() if c else None) for k, c in r["cells"].items()}})
    return {
        "schema": "verity/canonical-tables/v1",
        "frozen": FROZEN_ON,
        "caption": CAPTION,
        "frozen_instances": FROZEN_INSTANCES,
        "table1": {"columns": list(TABLE1_COLUMNS), "rows": [c.table1_row() for c in CANDIDATES],
                   "candidates": [{"id": c.id, "proof_classes": list(c.proof_classes), "drilldown_classes": list(c.drilldown_classes),
                                   "backend_names": list(c.backend_names)} for c in CANDIDATES]},
        "table2": {"columns": list(TABLE2_COLUMNS), "rows": rows},
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
        T.snapshot = st.snapshot(snapshot_name(), members, note=f"canonical tables (frozen {FROZEN_ON}): the Table 2 cell artifacts")
    if ns.format == "json":
        print(json.dumps(render_json(T), indent=2, sort_keys=True, default=str))
    else:
        print(render_markdown(T), end="")
    if T.snapshot:
        print(f"snapshot {T.snapshot} ({len(cell_artifacts(T))} member(s))", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
