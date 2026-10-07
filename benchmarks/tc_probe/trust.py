"""Trust: when we trust a silicon circuit.  Computed from evidence, never asserted.

A *subject* is one exact tensor-core relation on one architecture: a :class:`~verity.ml.circuits.targets.target.Target` (its
``instruction`` and ``pipeline`` model) or, for instructions no target wraps yet, a registry
:class:`~verity.ml.circuits.silicon.instructions.Instruction` with its model.  Its trust **level** is the highest of

    UNPROBED        no hardware evidence for the instruction on the anchor SKU and nothing to inherit
    HYPOTHESIS      a model exists and agrees somewhere (any SKU), or is inherited from a sibling instruction
    PINNED          the model matches the silicon on the anchor SKU (see PINNED criteria)
    TRUSTED         PINNED and the relation, the references and the discrepancy log all close (TRUSTED criteria)
    TRUSTED_STABLE  TRUSTED on >= 2 physical devices or >= 2 driver/ptxas versions with identical words

whose criteria *all* pass, where every criterion is a predicate over store evidence (artifact ids and labels, attempt
conditions) and repository fixtures (``verity/ml/circuits/silicon/discrepancy_log.json``, dataset manifests, reference replicas).  Nothing
here reads a hand-set ``status`` field: ``Instruction.status`` is reported next to the computed level so that the two
can disagree visibly.

The store is any object with the :class:`research.store.local.LocalStore` surface (``index.select_artifacts``,
``get_manifest``, ``labels``, ``index.attempts_by_output``, ``get_attempt``, ``has_local``, ``fetch``); the CLI resolves
``--store-class`` by name at run time.

Label contract (the exact keys this module reads; producers write them, nobody normalises later):

    tc-evidence/v1     instruction=<registry id>  hardware=<sku family: a100|h100|rtx4090|rtx5090|...>  model=<model name>
                       model_agrees=true|false  evidence=<sweep|golden_replay|...>  model_status=hypothesis
                       family=frozen-instances  instances_dataset=<dataset>  instances_tier=<tier>  mismatches=<n>   (hardware-in-the-loop)
                       family=canary                                                                                (benchmarks/tc_probe --canary)
                       device=<GPU UUID>      (optional; else attempt hw.gpu_uuid / payload gpu_uuid; TRUSTED_STABLE counts these, not hostnames)
                       instances-hw payload:  tc_evidence.{instances_dataset, instances_tier, mismatches, model, model_agrees, negative,
                                              manifest_sha256, hostname} is read when the labels are absent (benchmarks/tc_probe/instances_hw.py)
    bench-result/v1    relation=<RELATION_LABELS value> | target=<target name>  independently_verified=true|false
                       proof_class=<COMPLETE_ZK_BACKEND|COMPLETE_HVZK_BACKEND|NON_ZK_PROOF_DIAGNOSTIC|...>  hardware=<family>
                       honest=<n>  negatives=<n>                                     (when validation.detail is not the checker.sweep form)
    any                trust_downgrade=<trigger id>  /  trust_downgrade=resolved:<D-id>                              (invalidation)
    trust-dossier/v1   target=<subject>  instruction=<id>  level=<LEVEL>  anchor=<family>                          (written by --publish)

Dossier: ``python benchmarks/tc_probe/trust.py <target|instruction|--all> [--root STORE] [--format md|json] [--publish]``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Iterable

from verity.ml.circuits.silicon import models as _models
from verity.ml.circuits.silicon.instructions import INSTRUCTIONS, Instruction
from verity.ml.circuits.silicon.models import Model
from verity.ml.circuits.targets import target as _target
from verity.ml.circuits.targets.target import TARGETS, Target

__all__ = [
    "ANCHOR_SKU", "CANONICAL_FAMILIES", "Criterion", "DOSSIER_KIND", "EvidenceRef", "HONEST_MIN", "INDEPENDENT_REFERENCES",
    "INVALIDATION_TRIGGERS", "Level", "N_MIN", "NEGATIVES_MIN", "REGRESSION_TILES", "RELATION_LABELS", "RELITIGATION_PROCEDURE",
    "TrustReport", "assess", "d_class", "main", "publish", "registry_level", "render_markdown", "sku_family", "subjects",
]


# --- levels and thresholds ---------------------------------------------------------------------------------------------------


class Level(IntEnum):
    UNPROBED = 0
    HYPOTHESIS = 1
    PINNED = 2
    TRUSTED = 3
    TRUSTED_STABLE = 4


def registry_level(status: str | None) -> Level:
    """The ladder level a registry ``Instruction.status`` asserts (``pinned`` -> PINNED, ``hypothesis`` -> HYPOTHESIS, ``unprobed`` ->
    UNPROBED).  The registry may never sit above the computed level (TRUSTED and above are computed only): ``python
    benchmarks/tc_probe/trust.py --all --publish`` reports every instruction whose status exceeds its level, and the registry is
    downgraded citing the dossier."""
    if status is None:
        return Level.UNPROBED
    try:
        return Level[str(status).upper()]
    except KeyError:
        raise ValueError(f"registry status {status!r} is not a trust-ladder level ({', '.join(l.name.lower() for l in Level)})") from None


#: Elements with 0 mismatches on the anchor SKU for PINNED.  One default ``tc_probe --sweep`` is ~2.5e7 elements, so 1e7 demands a
#: full sweep (the 1.4e5-element FP8-family run alone cannot pin) while one run suffices; at 1e7 independent draws a per-element
#: defect of rate 5e-7 escapes with probability e^-5 < 1%.
N_MIN = 10_000_000
#: Honest verification units the prover's relation must accept (bit-exact against ``Model.step``) and mutations it must reject.
HONEST_MIN = 2_000
NEGATIVES_MIN = 100
#: Words an agreeing anchor run must hold to count as a second device / environment for TRUSTED_STABLE (a canary is ~5-30k).
STABLE_MIN_ELEMENTS = 1_000

DOSSIER_KIND = "trust-dossier/v1"
DOSSIER_SCHEMA = "verity/trust-dossier/v1"

#: SKU string -> family (the same table as ``research.env.SKU_FAMILY``; the anchor is a *family*: any A100 is the A100 anchor).
SKU_FAMILY: tuple[tuple[str, str], ...] = (
    (r"\bH100\b", "h100"), (r"\bB200\b", "b200"), (r"\bA100\b", "a100"), (r"\bL40S\b", "l40s"), (r"\bL40\b", "l40"),
    (r"\bRTX\s*4090\b", "rtx4090"), (r"\bRTX\s*5090\b", "rtx5090"), (r"\bRTX\s*PRO\s*6000\b", "rtxpro6000"),
)

#: The anchor SKU of an architecture when neither a Target nor the instruction names one (a Target's anchor is its
#: ``anchor_device``, then the registry instruction's ``anchor_device``).
ANCHOR_SKU: dict[str, str] = {
    "sm_80": "NVIDIA A100 SXM4 80GB", "sm_89": "NVIDIA GeForce RTX 4090", "sm_90": "NVIDIA H100 SXM5 80GB",
    "sm_100": "NVIDIA B200", "sm_120": "NVIDIA GeForce RTX 5090",
}

#: The canonical family set for PINNED: role -> tc_probe family names that cover it (``replay:<fragment>`` = a replay file).
#: Every role must be covered on the anchor with 0 mismatches for the subject's model (specials are recorded, not asserted:
#: presence suffices).
#: Each role lists tc_probe's family names first, then tc_probe_fp4's (block-scaled FP4 on sm_120) equivalents.
CANONICAL_FAMILIES: dict[str, tuple[str, ...]] = {
    "random-sweep": ("randn_zero_acc", "randn_random_acc", "uniform_bits", "random_zero_acc", "random_random_scales", "uniform_encodings"),
    "adder-width": ("adder_width_cancellation", "large_cancellation", "acc_window"),
    "zero-products": ("tiny_acc_zero_products", "zero_products_nonzero_acc", "tiny_acc", "signed_zero_matrix"),
    "accumulator-lsb-walk": ("acc_only_lsb_walk", "acc_offset_walk"),
    "specials": ("specials", "replay:mma_special"),
}
UNASSERTED_ROLES = {"specials"}

#: Regression tiles per discrepancy-log entry: the tc_probe family (``family:<name>``) or replay file (``replay:<fragment>``) that
#: reproduces the finding.  A *resolved* entry must have its tiles replayed on the anchor with 0 mismatches (PINNED); an entry
#: with an empty tuple has no hardware tile (proof-level, excluded-domain or kernel-level) and passes vacuously; a resolved entry
#: missing from this table fails until someone names its tile.  The canary replays every tile listed here for the subject.
REGRESSION_TILES: dict[str, tuple[str, ...]] = {
    "D1": (),                                                        # excluded: non-finite domain (recorded by the specials replay)
    "D2": ("family:randn_random_acc", "replay:mma_special"),         # sm_80 finite-domain parameters, arbitrary accumulators
    "D3": (),                                                        # closed-form normalisation: a proof, no distinguishing tile
    "D4": ("family:signed_zero", "family:cancellation"),             # zero results and their sign
    "D5": (),                                                        # excluded: epilogue NaN word, outside the instruction
    "D6": (),                                                        # kernel-level accumulation order: not an instruction tile
    "D7": ("family:randn_random_acc",),                              # arbitrary incoming accumulator
    "D8": ("family:near_overflow",),                                 # saturation and hold
    "D9": ("family:zero_acc_tiny_cancellation",),                    # zero accumulator's exponent
    "D10": ("replay:golden/ada_e4m3_m16n8k32.json", "family:adder_width_cancellation", "family:zero_products_nonzero_acc"),
    "D11": ("replay:probe3",),                                       # autoproof nvfp4 vectors (model5 vs the refuted v0)
    "D12": ("replay:probe3",),                                       # what autoproof's pin covers
    # D13-D20 tiles are tc_probe_fp4 families (lane fp4-re, RTX 5090, model BlockScaledAlignAdd: 0 mismatches / 15.7M words).
    "D13": ("family:ue8m0_underflow",),                              # resolved: mxf4 ue8m0 underflow sign is +0
    "D14": ("family:randn_zero_acc", "family:zero_products_nonzero_acc"),  # sm_90 FP8 mma.sync lowering
    "D15": ("family:fp8_floor_probe", "family:zero_products_nonzero_acc"),  # resolved: the FP8 adder floor is unobservable in [-152, -126] (pin-bf16)
    "D16": ("family:signed_zero", "family:signed_zero_matrix"),      # resolved: every zero result is +0 (acc sign dropped)
    "D17": ("family:hot_a_only", "family:hot_clip_regime", "family:anchor_participation"),  # resolved: grid anchored on leading scale exponent - 27, acc window 36
    "D18": ("family:random_zero_acc", "family:ue8m0_scale_extremes"),  # resolved: mxf4 = same adder with ue8m0 as mantissa 8 @ 2^(code-130)
    "D19": ("family:acc_subnormal",),                                # resolved: subnormal-result regime anchors on the exponent field, floor 2^-174
    "D20": (),                                                       # excluded: NaN/inf accumulators and overflow (outside the domain; recorded by specials)
}

#: bench-result ``relation`` label values per target (the coordinator's convention today); ``target=<name>`` is the go-forward key.
RELATION_LABELS: dict[str, tuple[str, ...]] = {
    "first-campaign-target/2026-09-21": ("bf16-ampere-k1536",),
    "fp8-ada-mma-draft/2026-09-22": ("fp8-ada-k1536",),
    "fp8-hopper-wgmma-draft/2026-09-22": ("fp8-hopper-k1536",),
}

#: Proof classes that prove the whole relation contract (ZK or not); RELATION-ONLY diagnostics are excluded by the checker text.
FULL_CONTRACT_PROOF_CLASSES = ("COMPLETE_ZK_BACKEND", "COMPLETE_HVZK_BACKEND", "NON_ZK_PROOF_DIAGNOSTIC")

#: Model-name aliases used by producers that predate the registry names (fp4-5090's ``model5``).
MODEL_ALIASES: dict[str, tuple[str, ...]] = {
    # model5 was the registry model for sm_120 FP4 until lane fp4-re refuted it (D16/D17: 29,118 mismatches on the RTX 5090) and
    # replaced it with BlockScaledAlignAdd; fp4-5090's model5-only sweeps are evidence about model5, not about the current model.
    # tc_probe names the E5M2 hypotheses as the E4M3 pipeline with E5M2 products; the registry's E5M2 models are those pipelines
    "ada_e5m2_m16n8k32": ("ada_e4m3_m16n8k32+e5m2_product",),
    "hopper_e5m2_wgmma_k32": ("hopper_e4m3_wgmma_k32+e5m2_product",),
    # lane tcgen05 tested the sm_120 models as the hypothesis on the B200 and labelled its captures with the sm_120 names; the
    # sm_100 registry models are those parameters under an sm_100 name (verity.ml.circuits.silicon.models).
    "blackwell_sm100_tcgen05_nvf4": ("blackwell_sm120_nvf4_m16n8k64", "blackwell_sm120_nvf4"),
    "blackwell_sm100_tcgen05_mxf4": ("blackwell_sm120_mxf4_m16n8k64", "blackwell_sm120_mxf4"),
}


@dataclass(frozen=True)
class IndependentReference:
    """Something other than our probe that computed the same words: a source-close simulator, a second compiled instruction path or
    a second harness's capture replicated into the repo.  ``paths`` must exist in the repo; agreement is *computed*: a replay of
    ``replay`` on the anchor with 0 mismatches, a replica whose structured conformance says 0 mismatches for this instruction and
    model, or -- for a reference that disagrees -- every D-entry citing its paths resolved with hardware as arbiter."""

    name: str
    kind: str
    paths: tuple[str, ...]
    replay: str | None = None
    replica: str | None = None
    note: str = ""


_HAWKEYE = ("verity/ml/circuits/silicon/hawkeye/PIN.json", "benchmarks/numerical/python/verity_numerical/reference/hawkeye_close.py",
            "benchmarks/numerical/tests/reference/test_hawkeye_close.py")
INDEPENDENT_REFERENCES: dict[str, tuple[IndependentReference, ...]] = {
    "sm80.mma.m16n8k16.bf16": (
        IndependentReference("hawkeye-source-close", "simulator", _HAWKEYE, note="Hawkeye's pinned A100 simulator, ported source-close"),
        IndependentReference("veritor-tc-total-a100-specials", "second-harness",
                             ("verity/ml/circuits/ops/tests/fixtures/tc-total-2026-09-07/a100/mma_special.reduced.json.gz",),
                             replay="a100/mma_special", note="veritor's tc-total A100 captures (31,290 structured cases) replayed by tc_probe"),
    ),
    "sm89.mma.m16n8k16.bf16": (
        IndependentReference("veritor-golden-ada-bf16", "second-harness", ("verity/ml/circuits/ops/tests/fixtures/golden/ada_bf16_m16n8k16.json",),
                             replay="golden/ada_bf16_m16n8k16.json", note="veritor gpu/tensor-core-semantics RTX 4090 golden records"),
        IndependentReference("hawkeye-source-close", "simulator", _HAWKEYE),
    ),
    "sm90.mma.m16n8k16.bf16": (
        IndependentReference("hawkeye-source-close", "simulator", _HAWKEYE, note="Hawkeye (H100) recovered groups (16,), width 26, floor -133"),
    ),
    "sm90.wgmma.m64n8k16.bf16": (
        IndependentReference("veritor-h100-wgmma-bf16", "second-compiled-path",
                             ("verity/ml/circuits/ops/tests/fixtures/tc-hopper-2026-09-07/special.reduced.json.gz",
                              "verity/ml/circuits/ops/tests/fixtures/tc-hopper-2026-09-07/summary.json"),
                             replay="tc-hopper-2026-09-07/special.reduced",
                             note="veritor lane HP: Triton tl.dot -> wgmma m64n128k16 (HGMMA.64x128x16), 307,281,920 words; its 44,046 records "
                                  "(14 families, non-finite included) replayed by tc_probe --replay-capture"),
    ),
    "sm89.mma.m16n8k32.e4m3": (
        IndependentReference("veritor-golden-ada-e4m3", "second-harness", ("verity/ml/circuits/ops/tests/fixtures/golden/ada_e4m3_m16n8k32.json",),
                             replay="golden/ada_e4m3_m16n8k32.json", note="veritor gpu/tensor-core-semantics RTX 4090 golden records (24.9M-element campaign)"),
        IndependentReference("hawkeye-hopper-fp8", "simulator", _HAWKEYE, note="Hawkeye's Hopper FP8 simulator does not describe Ada (D10)"),
    ),
    "sm90.wgmma.m64n8k32.e4m3": (
        IndependentReference("veritor-h100-wgmma-fp8", "second-compiled-path", ("verity/ml/circuits/silicon/captures/veritor-tc-fp8-2026-09-08.json",),
                             replica="verity/ml/circuits/silicon/captures/veritor-tc-fp8-2026-09-08.json",
                             note="veritor lane QZ: Triton tl.dot -> wgmma m64n64k32, 144,543,744 words"),
        IndependentReference("hawkeye-hopper-fp8", "simulator", _HAWKEYE, note="Hawkeye (H100 FP8, zero accumulator only): groups (32,), width 14, floor -139"),
    ),
    "sm120.mma.m16n8k64.e2m1.nvf4": (
        IndependentReference("autoproof-model5", "second-harness",
                             ("verity/ml/circuits/silicon/captures/autoproof-SEMANTICS.lock.v1.json", "verity/ml/circuits/silicon/captures/autoproof-probe-vectors.json"),
                             replay="probe3", note="autoproof's RTX 5090 probe vectors (1,601 of 1,574,998 dots) replayed by tc_probe_fp4"),
    ),
}

#: Events that re-open trust.  ``label`` is how the event is recorded in the store (on an artifact attributed to the subject).
INVALIDATION_TRIGGERS: tuple[dict[str, str], ...] = (
    {"id": "canary-mismatch", "event": "a canary or probe run on the anchor SKU records a mismatch for the pinned model",
     "label": "trust_downgrade=canary-mismatch", "detects": "a model error, a silicon revision or an environment fault (F1: the Triton-cache incident)"},
    {"id": "toolchain-change", "event": "the anchor's driver, ptxas/nvcc or CUDA toolkit changes",
     "label": "re-run `tc_probe --canary` on the anchor; a mismatch is canary-mismatch", "detects": "a lowering change (SASS count, a new MMA path)"},
    {"id": "verifier-rejected-honest", "event": "an independent verifier rejects an honest run of the relation",
     "label": "trust_downgrade=verifier-rejected-honest", "detects": "relation != model, or a prover/verifier defect"},
    {"id": "new-d-entry", "event": "a new discrepancy-log entry names the target or its instruction",
     "label": "verity/ml/circuits/silicon/discrepancy_log.json entry with status open (read directly)", "detects": "any disagreement between paper, simulator, reference and hardware"},
    {"id": "redteam-downgrade", "event": "a red team records a finding against the relation or its checker",
     "label": "trust_downgrade=redteam-downgrade", "detects": "a forgery, an unsound contract or a checker gap"},
)

#: What happens when a trigger fires; frozen with the criteria so the procedure cannot drift from them.
RELITIGATION_PROCEDURE: tuple[str, ...] = (
    "1. Reproduce on the anchor SKU: re-run the failing tiles (canary tiles_*.npz / the rejected instances) with `tc_probe` on the "
    "anchor family; record the run as tc-evidence/v1.  A non-reproducing failure is an environment fault: record it, restore the "
    "environment, re-run the canary; trust is unchanged only if the canary passes.",
    "2. Bisect model vs relation vs prover with the stored tiles: (a) hardware word vs Model.step -> model or silicon; (b) Model.step "
    "vs the prover's checker on the same words -> relation; (c) checker accepts but the verifier rejects -> prover/verifier.",
    "3. Open a D-entry in verity/ml/circuits/silicon/discrepancy_log.json naming the target (profile), the hardware evidence (run ids, artifact ids) "
    "and the owner; status open.",
    "4. The level downgrades automatically: an open D-entry blocks TRUSTED, a mismatch on the anchor blocks PINNED, a "
    "trust_downgrade label blocks TRUSTED; nobody edits a level.  Re-publish the dossier so the downgrade is frozen with its evidence.",
    "5. Resolve with hardware as arbiter: the model, relation or prover changes; the fix is replayed on the anchor (tiles + canary), "
    "the D-entry becomes resolved with evidence_refs, its tile joins REGRESSION_TILES, the trust_downgrade label is closed with "
    "trust_downgrade=resolved:<D-id>; `assess` recomputes the level.",
)


# --- small helpers -------------------------------------------------------------------------------------------------------------


def sku_family(sku: str | None) -> str | None:
    """SKU string -> family name (``NVIDIA A100 80GB PCIe`` -> ``a100``); None when no row matches."""
    if not sku:
        return None
    for pat, fam in SKU_FAMILY:
        if re.search(pat, sku):
            return fam
    return None


def _sm(arch: str | None) -> str | None:
    """``sm_90a`` -> ``sm_90``."""
    if not arch:
        return None
    m = re.match(r"(sm_\d+)", arch)
    return m.group(1) if m else arch


def _norm_model(name: str) -> str:
    n = re.sub(r"\[HYPOTHESIS\]$", "", str(name).strip())
    if ":" in n:
        n = n.rsplit(":", 1)[1]
    return n.lower()


def _model_names(model: Model | None) -> set[str]:
    """Every name a producer may have used for this model: its ``name``, its constant name in ``verity.ml.circuits.silicon.models``, aliases."""
    if model is None:
        return set()
    names = {model.name.lower()}
    for const, obj in vars(_models).items():
        if obj is model and const.isupper():
            names.add(const.lower())
    names.update(MODEL_ALIASES.get(model.name, ()))
    return names


def _truthy(v: Any) -> bool:
    return v is True or (isinstance(v, str) and v.strip().lower() in ("true", "1", "yes"))


def _as_int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _find_repo(start: Path | None = None) -> Path | None:
    """The repository root: the nearest ancestor with ``verity/ml/circuits/silicon/discrepancy_log.json``."""
    here = (start or Path(__file__)).resolve()
    for p in [here, *here.parents]:
        if (p / "verity" / "ml" / "circuits" / "silicon" / "discrepancy_log.json").is_file():
            return p
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / "verity" / "ml" / "circuits" / "silicon" / "discrepancy_log.json").is_file():
            return p
    return None


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --- subjects ----------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Subject:
    """What is being trusted: a Target (relation over an instruction) or a bare registry instruction."""

    name: str
    instruction: str
    model: Model | None
    anchor_sku: str
    target: Target | None = None

    @property
    def anchor_family(self) -> str:
        return sku_family(self.anchor_sku) or self.anchor_sku.lower()

    @property
    def registry(self) -> Instruction | None:
        return INSTRUCTIONS.get(self.instruction)

    @property
    def model_names(self) -> set[str]:
        names = _model_names(self.model)
        if self.target is not None and getattr(self.target.pipeline, "name", None):
            names.add(str(self.target.pipeline.name).lower())
        return names

    @property
    def names(self) -> set[str]:
        """Names a D-entry ``profile`` or a label may use for this subject."""
        out = {self.name, self.instruction}
        if self.target is not None:
            out.add(self.target.name)
        return out


def _target_aliases() -> dict[str, Target]:
    out: dict[str, Target] = dict(TARGETS)
    for const, obj in vars(_target).items():
        if isinstance(obj, Target) and const.isupper():
            out[const] = obj
    return out


def subjects(name: str) -> Subject:
    """Resolve a target name (``first-campaign-target/2026-09-21``), a target constant (``FIRST``) or an instruction id."""
    aliases = _target_aliases()
    if name in aliases:
        t = aliases[name]
        if t.instruction is None:
            raise KeyError(f"target {t.name} names no instruction")
        model = INSTRUCTIONS[t.instruction].model if t.instruction in INSTRUCTIONS else None
        if model is None and hasattr(t.pipeline, "step"):
            model = t.pipeline  # a Target whose pipeline is itself a Model and whose instruction the registry does not model yet
        return Subject(name=t.name, instruction=t.instruction, model=model, anchor_sku=t.anchor_device, target=t)
    if name in INSTRUCTIONS:
        ins = INSTRUCTIONS[name]
        wrapped = [t for t in TARGETS.values() if t.instruction == name]
        if wrapped:
            return subjects(wrapped[0].name)
        return Subject(name=ins.id, instruction=ins.id, model=ins.model, anchor_sku=ins.anchor_device or ANCHOR_SKU.get(ins.sm, ins.sm),
                       target=None)
    raise KeyError(f"unknown target or instruction {name!r}; targets: {sorted(TARGETS)}; instructions: {sorted(INSTRUCTIONS)}")


def all_subject_names() -> list[str]:
    names = list(TARGETS)
    covered = {t.instruction for t in TARGETS.values()}
    names += [i for i, ins in INSTRUCTIONS.items() if ins.model is not None and i not in covered]
    return names


# --- evidence collected from the store ------------------------------------------------------------------------------------------


@dataclass
class HardwareRecord:
    """One hardware run of one instruction, normalised from any of the three formats the store holds (tc-probe-fp8's nested
    ``tc_evidence``, fp4-5090's flat meta, the WP1 ``bench-result/v1`` of ``tc_probe`` whose counts live in ``probe_results.json``)."""

    id: str
    kind: str
    run_id: str | None
    instruction: str | None
    sku: str | None
    family: str | None
    arch: str | None
    driver: str | None
    nvcc: str | None
    host: str | None
    seed: int | None
    elements: int
    mismatches: dict[str, int]                    # normalised model name -> mismatches over asserted families
    families: dict[str, dict]                     # family -> {elements, asserted, mismatches: {model: n}}
    replays: list[dict]                           # {kind, file, records, mismatches}
    instances: dict | None                        # {dataset, manifest_sha256, tiers: {tier: {instances, mismatches}}}
    sass: dict | None                             # {single_instruction, count, expected}
    lowering: dict[str, str | None]
    labels: dict[str, list[str]]
    legacy: bool
    note: str = ""
    canary: bool = False                          # a `tc_probe --canary` run: a verdict on the anchor, not family coverage
    gpu_uuid: str | None = None                   # the physical device (label `device=`, attempt hw.gpu_uuid, payload gpu_uuid); never a hostname

    def mismatches_for(self, names: set[str]) -> int | None:
        hits = [v for k, v in self.mismatches.items() if k in names]
        return min(hits) if hits else None

    def replay_records(self) -> int:
        return sum(int(r.get("records") or 0) for r in self.replays)

    def agrees(self, names: set[str]) -> bool:
        """Every compared word reproduced by one of ``names``: a 0-mismatch sweep over > 0 asserted elements, or -- when nothing was
        swept against the model -- a 0-mismatch replay with > 0 records, or a frozen-instances record labelled ``mismatches=0``.
        Any nonzero sweep count for the model is a disagreement whatever the replays say."""
        m = self.mismatches_for(names)
        if m is not None and m > 0:
            return False
        if m == 0 and self.elements > 0:
            return True
        if m is None and self.mismatches:
            return False  # swept against other models only (e.g. a superseded hypothesis): not evidence about this one either way
        if self.replays and all(int(r.get("mismatches") or 0) == 0 for r in self.replays) and self.replay_records() > 0:
            # a replay compares against the pipeline named in the replay (legacy) or the record's model label
            pipes = {_norm_model(r.get("pipeline") or "") for r in self.replays if r.get("pipeline")}
            if not pipes or pipes & names:
                return True
        if self.labels.get("family") == ["frozen-instances"] and self.labels.get("mismatches", ["1"])[-1] == "0":
            return True
        tiers = (self.instances or {}).get("tiers") or {}
        if m is None and not self.replays and tiers and all(int(t.get("mismatches") or 0) == 0 for t in tiers.values()):
            return True  # an instances replay (incl. a negative tier: every claimed-wrong word detected) with no failure
        return False

    def partially_agrees(self, names: set[str]) -> str | None:
        """Some agreement (HYPOTHESIS): the reason, or None.  A full agreement, or one asserted family / one replay reproduced bit-for-bit."""
        if self.agrees(names):
            return "every compared word reproduced"
        for fam, f in self.families.items():
            hits = [v for k, v in f["mismatches"].items() if k in names]
            if f["asserted"] and f["elements"] > 0 and hits and min(hits) == 0:
                return f"family {fam} reproduced ({f['elements']:,} elements) while other families disagree"
        if self.mismatches and self.mismatches_for(names) is None:
            return None  # swept against other models only: the replays say nothing about this model
        for r in self.replays:
            if int(r.get("records") or 0) > 0 and int(r.get("mismatches") or 0) == 0:
                pipe = _norm_model(r.get("pipeline") or "")
                if not pipe or pipe in names:
                    return f"replay {r.get('file')} reproduced ({r.get('records')} records) while the sweep disagrees"
        return None

    def environment(self) -> tuple[str | None, str | None]:
        return (self.driver, (self.nvcc or "").split("/")[0].strip() or None)


@dataclass
class RelationRecord:
    """A ``bench-result/v1`` about a relation: the checker sweep (honest / negatives), a proof batch a verifier accepted, ..."""

    id: str
    run_id: str | None
    labels: dict[str, list[str]]
    fingerprint: dict
    validation: dict
    params: dict
    tiers: dict[str, tuple[int, int]] = field(default_factory=dict)   # tier -> (ok, total)

    @property
    def passed(self) -> bool:
        return (self.validation.get("status") == "passed")

    @property
    def proof_class(self) -> str | None:
        v = self.labels.get("proof_class") or ([self.fingerprint.get("proof_class")] if self.fingerprint.get("proof_class") else [])
        return v[-1] if v else None

    @property
    def independently_verified(self) -> bool:
        if "independently_verified" in self.labels:
            return _truthy(self.labels["independently_verified"][-1])
        ver = self.fingerprint.get("verifier")
        return isinstance(ver, dict) and "independent" in str(ver.get("name", "")).lower()

    @property
    def checker_text(self) -> str:
        return " ".join(str(x) for x in (self.fingerprint.get("checker"), self.validation.get("checker")) if x)

    def honest(self) -> int:
        if "honest" in self.labels:
            return _as_int(self.labels["honest"][-1]) or 0
        return sum(ok for t, (ok, tot) in self.tiers.items() if not t.endswith("-neg") and t.startswith("vu") and ok == tot)

    def negatives(self) -> int:
        if "negatives" in self.labels:
            return _as_int(self.labels["negatives"][-1]) or 0
        return sum(ok for t, (ok, tot) in self.tiers.items() if t.endswith("-neg") and ok == tot)


@dataclass
class DatasetIdentity:
    """A frozen input set in the repo (``fixtures/<name>/v<k>/manifest.json`` whose ``target.profile`` names the target)."""

    name: str
    path: str                 # repo-relative manifest path, or ``contract:<CONSTANT>`` for a recipe pinned by digest in the bench contract
    sha256: str               # the manifest's sha256, or the contract's recipe digest
    tiers: dict[str, dict]
    contract: bool = False    # True: a seeded recipe (no manifest, no composition) identified by its contract digest

    @property
    def honest_tiers(self) -> list[str]:
        return [t for t in self.tiers if not t.endswith("-neg")]

    @property
    def negative_tiers(self) -> list[str]:
        return [t for t in self.tiers if t.endswith("-neg")]

    def corner_families(self, tier: str) -> dict[str, int]:
        comp = (self.tiers.get(tier) or {}).get("composition") or {}
        return dict(comp.get("corner_families") or {})


class _StoreView:
    """The few LocalStore calls this module makes, with the labels folded into ``key -> [values in ts order]``."""

    def __init__(self, store: Any):
        self.store = store
        self._attempts: dict[str, dict | None] = {}

    def artifacts(self, kind: str) -> list[str]:
        rows = self.store.index.select_artifacts(kind=kind)
        return [r["id"] if isinstance(r, dict) else str(r) for r in rows]

    def manifest(self, art_id: str):
        return self.store.get_manifest(art_id)

    def labels(self, art_id: str) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for lab in self.store.labels(art_id):
            key, value = (lab.key, lab.value) if hasattr(lab, "key") else (lab["key"], lab["value"])
            out.setdefault(str(key), []).append(value if isinstance(value, str) else json.dumps(value, sort_keys=True))
        return out

    def attempt_ids(self, art_id: str) -> list[str]:
        try:
            return list(self.store.index.attempts_by_output(art_id))
        except Exception:  # noqa: BLE001 - an index without attempts is still a store
            return []

    def attempt(self, run_id: str) -> dict | None:
        if run_id not in self._attempts:
            try:
                self._attempts[run_id] = self.store.get_attempt(run_id)
            except Exception:  # noqa: BLE001
                self._attempts[run_id] = None
        return self._attempts[run_id]

    def payload_json(self, art_id: str | None, name: str) -> dict | None:
        """``name`` inside a locally held tree artifact; never fetches from a remote."""
        if not art_id:
            return None
        try:
            if not self.store.has_local(art_id):
                return None
            p = Path(self.store.fetch(art_id))
            f = p / name if p.is_dir() else p
            if not f.is_file():
                return None
            return json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            return None


def _attempt_conditions(view: _StoreView, art_id: str) -> tuple[str | None, dict]:
    for rid in view.attempt_ids(art_id):
        att = view.attempt(rid)
        if att:
            return rid, dict(att.get("conditions") or {})
    ids = view.attempt_ids(art_id)
    return (ids[0] if ids else None), {}


def _params(model: Model | None) -> tuple | None:
    """The accumulator-step parameters of a GroupSum-like model (two names for the same silicon behaviour compare equal)."""
    if model is None or not all(hasattr(model, a) for a in ("groups", "width", "zero_exponent", "dtype")):
        return None
    return (tuple(model.groups), model.width, model.zero_exponent, model.dtype)


def _legacy_instruction(arch: str | None, pipeline: str | None) -> tuple[str | None, set[str]]:
    """The registry instruction a WP1 ``tc_probe`` result (arch + pipeline name) measured, and every model name its mismatch count
    stands for: the pipeline's own name plus the instruction's model when that model has identical parameters (the RTX 4090 runs
    compared the Ampere pipeline, which is Ada's under another name)."""
    sm = _sm(arch)
    by_name = {m.name: m for m in vars(_models).values() if isinstance(getattr(m, "name", None), str) and hasattr(m, "step")}
    pipe_model = by_name.get(pipeline or "")
    names = {_norm_model(pipeline)} if pipeline else set()
    for ins in INSTRUCTIONS.values():
        if ins.sm != sm or ins.model is None or not pipeline:
            continue
        if ins.model.name == pipeline:
            return ins.id, names
        if pipe_model is not None and _params(ins.model) is not None and _params(ins.model) == _params(pipe_model):
            names.add(_norm_model(ins.model.name))
            return ins.id, names
    return None, names


def _hardware_record(view: _StoreView, art_id: str, kind: str, meta: dict, refs: dict, labels: dict[str, list[str]]) -> HardwareRecord | None:
    wf = meta.get("workload_fingerprint") or {}
    run_id, cond = _attempt_conditions(view, art_id)
    te = meta.get("tc_evidence")
    rec = HardwareRecord(id=art_id, kind=kind, run_id=run_id or meta.get("run_id"), instruction=None, sku=None, family=None, arch=None,
                         driver=None, nvcc=None, host=None, seed=None, elements=0, mismatches={}, families={}, replays=[], instances=None,
                         sass=None, lowering={}, labels=labels, legacy=False)
    if isinstance(te, dict):                                              # tc-probe-fp8: nested tc_evidence
        rec.instruction = te.get("instruction") or wf.get("instruction")
        rec.sku, rec.arch, rec.driver, rec.nvcc, rec.seed = te.get("sku"), te.get("arch"), te.get("driver"), te.get("nvcc"), _as_int(te.get("seed"))
        for name, f in (te.get("families") or {}).items():
            rec.families[name] = {"elements": int(f.get("elements") or 0), "asserted": bool(f.get("asserted", True)),
                                  "mismatches": {_norm_model(m): int(v) for m, v in (f.get("mismatches") or {}).items()}}
        rec.mismatches = {_norm_model(m): int(v) for m, v in (te.get("total_mismatches_by_model") or {}).items()}
        for k in ("golden_replay", "special_replay", "capture_replay"):
            r = te.get(k)
            if isinstance(r, dict):
                rec.replays.append(_replay_summary(k, r))
        if isinstance(te.get("instances_replay"), dict):
            rec.instances = _instances_summary(te["instances_replay"])
        if isinstance(te.get("sass"), dict):
            s = te["sass"]
            rec.sass = {"single_instruction": s.get("single_instruction", s.get("tile_kernel_mma_count") == 1), "count": s.get("tile_kernel_mma_count"),
                        "expected": s.get("expected_mma_count"), "ok": s.get("ok")}
        rec.lowering = {_norm_model(m): d.get("lowering") for m, d in (te.get("models") or {}).items() if isinstance(d, dict)}
        if te.get("instances_tier") and rec.instances is None:            # instances-hw lane (benchmarks/tc_probe/instances_hw.py): one tier per artifact
            tier = str(te["instances_tier"])
            negative = bool(te.get("negative")) or tier.endswith("-neg")
            if negative:  # the tier's expected words are the CLAIMED (wrong) ones: its failure count is the undetected negatives
                neg = te.get("negative") if isinstance(te.get("negative"), dict) else {}
                bad = _as_int(neg.get("undetected"))
                if bad is None:
                    bad = next((_as_int(m.get("value")) for m in meta.get("measurements") or [] if isinstance(m, dict) and m.get("name") == "negatives_undetected"), None)
                tier_mism = bad if bad is not None else (0 if (meta.get("validation") or {}).get("status") == "passed" else 1)
            else:
                tier_mism = int(te.get("mismatches") or 0) + int(te.get("other_positions_nonzero") or 0)
                if isinstance(te.get("model"), str) and te.get("mismatches") is not None:
                    rec.mismatches.setdefault(_norm_model(te["model"]), int(te["mismatches"]))
                    rec.elements = rec.elements or int(te.get("elements") or 0)
            rec.instances = {"dataset": te.get("instances_dataset") or (wf.get("instances") or {}).get("dataset"),
                             "manifest_sha256": te.get("manifest_sha256") or (wf.get("instances") or {}).get("manifest_sha256"),
                             "tiers": {tier: {"instances": _as_int(te.get("instances")), "mismatches": tier_mism, "negative": negative}}}
        cn = te.get("canary")
        if isinstance(cn, dict) and not rec.families:                     # tc_probe --canary: the counts are the verdict
            rec.canary = True
            for name, f in (cn.get("families") or {}).items():
                rec.families[name] = {"elements": int(f.get("elements") or 0), "asserted": bool(f.get("asserted", True)),
                                      "mismatches": {_norm_model(m): int(v) for m, v in (f.get("mismatches") or {}).items()}}
            rec.mismatches = {_norm_model(m): int(v) for m, v in (cn.get("total_mismatches_by_model") or {}).items()}
            for r in cn.get("replays") or []:
                if isinstance(r, dict):
                    rec.replays.append({"kind": "canary", "file": r.get("file"), "records": r.get("records"), "mismatches": r.get("mismatches"), "pipeline": r.get("pipeline")})
    elif "counts" in meta and "instruction" in meta:                       # fp4-5090: flat meta
        rec.instruction = meta.get("instruction")
        rec.sku, rec.arch, rec.driver, rec.nvcc, rec.seed = meta.get("sku"), meta.get("arch"), meta.get("driver"), meta.get("nvcc"), _as_int(meta.get("seed"))
        rec.host = meta.get("gpu_uuid")
        counts = meta.get("counts") or {}
        # tc_probe_fp4 scores several models per word and names the one it asserts for the instruction ``primary_model`` (fp4-re:
        # "verity" = the registry model, "model5" = autoproof's refuted control).  The primary key stands for the registry model.
        ins = INSTRUCTIONS.get(str(meta.get("instruction") or ""))
        primary = meta.get("primary_model")
        registry_name = ins.model.name if ins is not None and ins.model is not None else None

        def _key(m: str) -> str:
            return _norm_model(registry_name) if (primary and m == primary and registry_name) else _norm_model(m)

        rec.mismatches = {_key(m): int(v) for m, v in (counts.get("mismatches") or {}).items()}
        for name, f in (meta.get("families") or {}).items():
            rec.families[name] = {"elements": int(f.get("elements") or 0), "asserted": not bool(f.get("record_only")),
                                  "mismatches": {_key(m): int(v) for m, v in (f.get("mismatches") or {}).items()}}
        rec.elements = int(counts.get("elements") or 0)
        if counts.get("replay_cases"):
            rec.replays.append({"kind": "autoproof", "file": ",".join(wf.get("replays") or []), "records": int(counts["replay_cases"]),
                                "mismatches": int(counts["replay_cases"]) - int(counts.get("replay_matched") or 0), "pipeline": None})
        sm = meta.get("sass_mma_counts")
        if isinstance(sm, dict) and sm:
            vals = [v for v in sm.values() if isinstance(v, int)]
            if vals:
                rec.sass = {"single_instruction": max(vals) == 1, "count": max(vals), "expected": 1, "ok": None}
    elif wf.get("probe") == "benchmarks/tc_probe/tc_probe.py":                 # WP1 tc_probe result recorded as bench-result/v1
        rec.legacy = True
        rec.sku, rec.arch, rec.seed = wf.get("device"), wf.get("arch"), _as_int(wf.get("seed"))
        derived, model_keys = _legacy_instruction(wf.get("arch"), wf.get("pipeline"))
        rec.instruction = wf.get("instruction") or derived
        pr = view.payload_json((refs or {}).get("run_files"), "probe_results.json")
        if pr is None:
            rec.note = "payload probe_results.json not held locally: counts unavailable"
        else:
            ident = pr.get("identity") or {}
            smi = str(ident.get("nvidia_smi") or "")
            rec.driver = smi.split(",")[1].strip() if smi.count(",") >= 1 else None
            rec.nvcc, rec.host = ident.get("nvcc"), ident.get("hostname")
            pipeline = pr.get("pipeline") or wf.get("pipeline")
            model_keys = model_keys | ({_norm_model(pipeline)} if pipeline else set())
            sw = pr.get("sweep") or {}
            for name, f in (sw.get("families") or {}).items():
                mism = f.get("mismatches_vs_model")
                rec.families[name] = {"elements": int(f.get("elements") or 0), "asserted": name != "specials",
                                      "mismatches": {k: int(mism) for k in model_keys} if mism is not None else {}}
            if sw:
                rec.elements = int(sw.get("total_elements") or 0)
                if sw.get("total_mismatches_vs_model") is not None:
                    for k in model_keys:
                        rec.mismatches[k] = int(sw["total_mismatches_vs_model"])
            for k in ("golden_replay", "special_replay", "capture_replay"):
                r = pr.get(k)
                if isinstance(r, dict):
                    rec.replays.append(_replay_summary(k, r))
            if isinstance(pr.get("instances_replay"), dict):
                rec.instances = _instances_summary(pr["instances_replay"])
            if isinstance(pr.get("sass"), dict):
                s = pr["sass"]
                rec.sass = {"single_instruction": s.get("single_instruction", s.get("tile_kernel_mma_count") == 1), "count": s.get("tile_kernel_mma_count"),
                            "expected": s.get("expected_mma_count"), "ok": s.get("ok")}
    else:
        return None
    # the label contract wins over payload fields wherever a label exists
    if labels.get("instruction"):
        rec.instruction = labels["instruction"][-1]
    if rec.elements == 0:
        for m in meta.get("measurements") or []:
            if isinstance(m, dict) and m.get("name") == "elements" and isinstance(m.get("value"), (int, float)):
                rec.elements = int(m["value"])
        if rec.elements == 0 and rec.families:
            rec.elements = sum(f["elements"] for f in rec.families.values() if f["asserted"])
    rec.sku = rec.sku or cond.get("hw.sku") or wf.get("device")
    rec.driver = rec.driver or cond.get("rt.driver")
    rec.nvcc = rec.nvcc or cond.get("rt.nvcc")
    tev = te if isinstance(te, dict) else {}
    rec.gpu_uuid = (labels.get("device") or [None])[-1] or cond.get("hw.gpu_uuid") or tev.get("gpu_uuid") or wf.get("gpu_uuid") or meta.get("gpu_uuid")
    rec.host = rec.host or cond.get("hw.hostname") or tev.get("hostname") or wf.get("hostname") or meta.get("host") or rec.gpu_uuid
    rec.family = (labels.get("hardware") or [None])[-1] or cond.get("hw.family") or sku_family(rec.sku)
    if labels.get("family") == ["canary"] or wf.get("family") == "canary":
        rec.canary = True
    if labels.get("family") == ["frozen-instances"] and rec.instances is None:
        rec.instances = {"dataset": (labels.get("instances_dataset") or [None])[-1], "manifest_sha256": (labels.get("instances_manifest_sha256") or [None])[-1],
                         "tiers": {t: {"instances": None, "mismatches": _as_int((labels.get("mismatches") or ["1"])[-1])} for t in labels.get("instances_tier", [])}}
    return rec


def _instances_summary(ir: dict) -> dict:
    tiers: dict[str, dict] = {}
    for tier, r in ir.items():
        if not isinstance(r, dict) or tier in ("digests",):
            continue
        if "instances" in r:
            mism = int(r.get("mismatches") or 0) + int(r.get("accumulator_mismatches") or 0) + int(r.get("y_mismatches") or 0) + int(r.get("other_positions_nonzero") or 0)
            tiers[tier] = {"instances": int(r["instances"]), "mismatches": mism}
    return {"dataset": ir.get("dataset"), "manifest_sha256": ((ir.get("digests") or {}).get("manifest_sha256")), "tiers": tiers}


_TIER_RE = re.compile(r"(?P<tier>[A-Za-z0-9_.-]+)\s+(?P<ok>\d+)\s*/\s*(?P<tot>\d+)\s+ok")


def _relation_record(view: _StoreView, art_id: str, meta: dict, labels: dict[str, list[str]]) -> RelationRecord:
    wf = meta.get("workload_fingerprint") or {}
    val = meta.get("validation") or {}
    run_id, _ = _attempt_conditions(view, art_id)
    params: dict = {}
    if run_id:
        att = view.attempt(run_id) or {}
        params = dict(att.get("params") or {})
    rec = RelationRecord(id=art_id, run_id=run_id or meta.get("run_id"), labels=labels, fingerprint=wf, validation=val, params=params)
    for m in _TIER_RE.finditer(str(val.get("detail") or "")):
        rec.tiers[m.group("tier")] = (int(m.group("ok")), int(m.group("tot")))
    return rec


@dataclass
class Evidence:
    hardware: list[HardwareRecord]
    relation: list[RelationRecord]
    compile_checks: list[dict]                   # {id, counts: {instruction: n}, nvcc}
    dossiers: list[dict]                         # {id, target, level, labels}

    @classmethod
    def collect(cls, store: Any) -> "Evidence":
        view = _StoreView(store)
        hardware: list[HardwareRecord] = []
        relation: list[RelationRecord] = []
        checks: list[dict] = []
        dossiers: list[dict] = []
        for kind in ("tc-evidence/v1", "bench-result/v1", "tc-compile-check/v1", DOSSIER_KIND):
            try:
                ids = view.artifacts(kind)
            except Exception:  # noqa: BLE001
                ids = []
            for art_id in ids:
                m = view.manifest(art_id)
                meta, refs = dict(m.meta or {}), dict(m.refs or {})
                labels = view.labels(art_id)
                if kind == DOSSIER_KIND:
                    dossiers.append({"id": art_id, "target": meta.get("subject") or meta.get("target"), "level": meta.get("level"), "labels": labels})
                    continue
                if kind == "tc-compile-check/v1":
                    counts = {}
                    for x in meta.get("measurements") or []:
                        if isinstance(x, dict) and str(x.get("name", "")).endswith(".tile_kernel_mma_count"):
                            counts[str(x["name"])[: -len(".tile_kernel_mma_count")]] = x.get("value")
                    checks.append({"id": art_id, "counts": counts, "nvcc": (meta.get("workload_fingerprint") or {}).get("nvcc"), "labels": labels})
                    continue
                hw = _hardware_record(view, art_id, kind, meta, refs, labels)
                if hw is not None:
                    hardware.append(hw)
                elif kind == "bench-result/v1":
                    relation.append(_relation_record(view, art_id, meta, labels))
        return cls(hardware=hardware, relation=relation, compile_checks=checks, dossiers=dossiers)


# --- repository fixtures -------------------------------------------------------------------------------------------------------


def _load_discrepancies(repo: Path | None) -> list[dict]:
    if repo is None:
        return []
    p = repo / "verity" / "ml" / "circuits" / "silicon" / "discrepancy_log.json"
    if not p.is_file():
        return []
    doc = json.loads(p.read_text())
    return list(doc.get("entries") or [])


#: Discrepancy-log entry classes.  ``numerics``: the model's words vs the hardware's (gates TRUSTED through Tb and PINNED through
#: P5's regression tiles).  ``binding``: which kernels / shapes / accumulation orders realise the chain the circuit proves -- a
#: deployment caveat the dossier lists, not a gate on the circuit (D6).  An entry without the field is ``numerics``.
D_CLASSES = ("numerics", "binding")


def d_class(entry: dict) -> str:
    cls = entry.get("class") or "numerics"
    if cls not in D_CLASSES:
        raise ValueError(f"{entry.get('id')}: class must be one of {D_CLASSES}, got {cls!r}")
    return cls


#: Input sets identified by a recipe digest pinned in the bench contract rather than by a repo fixture manifest (the FP8 Ada
#: set is seeded: uniform finite E4M3 bytes from ``FP8_ADA_INSTANCES_SEED``).  target name -> the contract constant prefix; the
#: constants ``<PREFIX>_INSTANCES_{DATASET,TIER,DIGEST}`` must all exist for the set to count (the Hopper ones once b-hopper lands them).
CONTRACT_INSTANCE_SETS: dict[str, str] = {
    "fp8-ada-mma-draft/2026-09-22": "FP8_ADA",
    "fp8-hopper-wgmma-draft/2026-09-22": "FP8_HOPPER",
    "nvfp4-sm120-mma-draft/2026-09-22": "NVFP4",
}


def _contract_instance_sets(subject: Subject, contract: Any | None = None) -> list[DatasetIdentity]:
    """The subject's contract-pinned input sets (``CONTRACT_INSTANCE_SETS``); ``contract`` overrides the module (tests)."""
    prefixes = [p for name, p in CONTRACT_INSTANCE_SETS.items() if name in subject.names]
    if not prefixes:
        return []
    if contract is None:
        try:
            from verity_numerical.bench import contract as _contract  # a sibling workspace package; absent -> no contract sets
        except ImportError:
            return []
        contract = _contract
    out = []
    for prefix in prefixes:
        vals = {k: getattr(contract, f"{prefix}_INSTANCES_{k}", None) for k in ("DATASET", "TIER", "DIGEST")}
        if not all(isinstance(v, str) and v for v in vals.values()):
            continue
        # an optional ``<PREFIX>_INSTANCES_FAMILIES`` (family -> VU count at the frozen tier size) is the recipe's corner-family
        # composition -- what a fixture manifest's ``composition.corner_families`` carries for a materialised set (Tc's edge predicate)
        fams = getattr(contract, f"{prefix}_INSTANCES_FAMILIES", None)
        tier: dict = {}
        if isinstance(fams, dict) and fams and all(isinstance(k, str) and isinstance(v, int) for k, v in fams.items()):
            tier = {"composition": {"corner_families": {k: v for k, v in fams.items() if k != "random"}, "families": dict(fams)}}
        out.append(DatasetIdentity(name=vals["DATASET"], path=f"contract:{prefix}_INSTANCES_DIGEST", sha256=vals["DIGEST"],
                                   tiers={vals["TIER"]: tier}, contract=True))
    return out


def _datasets_for(subject: Subject, repo: Path | None) -> list[DatasetIdentity]:
    if repo is None or not (repo / "fixtures").is_dir():
        return []
    out = []
    for man in sorted((repo / "fixtures").glob("*/v*/manifest.json")):
        try:
            doc = json.loads(man.read_text())
        except (OSError, ValueError):
            continue
        tgt = doc.get("target") if isinstance(doc.get("target"), dict) else {}
        if tgt.get("profile") in subject.names or (subject.model is not None and tgt.get("pipeline") == subject.model.name):
            tiers = doc.get("tiers") if isinstance(doc.get("tiers"), dict) else {t: {} for t in (doc.get("tiers") or [])}
            out.append(DatasetIdentity(name=str(doc.get("dataset") or man.parent.name), path=man.relative_to(repo).as_posix(),
                                       sha256=_sha256_file(man), tiers=tiers))
    return out


def _replica(repo: Path | None, rel: str) -> dict | None:
    if repo is None:
        return None
    p = repo / rel
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except ValueError:
        return None


# --- the report ----------------------------------------------------------------------------------------------------------------


@dataclass
class EvidenceRef:
    id: str
    kind: str
    note: str = ""


@dataclass
class Criterion:
    id: str
    level: str
    title: str
    passed: bool
    evidence: list[EvidenceRef] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


@dataclass
class TrustReport:
    subject: str
    target: str | None
    instruction: str
    model: str | None
    anchor_sku: str
    anchor_family: str
    level: str
    registry_status: str | None
    criteria: list[Criterion]
    next_level: str | None
    missing_to_next: list[str]
    d_entries: list[dict]
    references: list[dict]
    sass: dict
    hardware_records: list[dict]
    relation_records: list[dict]
    invalidation_triggers: list[dict]
    relitigation_procedure: list[str]
    constants: dict
    generated_utc: str
    previous_dossiers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    canaries: list[dict] = field(default_factory=list)   # `tc_probe --canary` runs on the anchor: {id, run_id, passed, ...}
    deployment_caveats: list[dict] = field(default_factory=list)   # binding-class D-entries: listed, never a gate on the circuit

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrustReport":
        crit = [Criterion(id=c["id"], level=c["level"], title=c["title"], passed=c["passed"],
                          evidence=[EvidenceRef(**e) for e in c.get("evidence", [])], missing=list(c.get("missing", [])), details=dict(c.get("details", {})))
                for c in d["criteria"]]
        kw = {k: v for k, v in d.items() if k != "criteria" and k != "schema"}
        return cls(criteria=crit, **kw)

    def cited(self) -> list[str]:
        ids = {e.id for c in self.criteria for e in c.evidence if e.id.startswith("art:")}
        ids.update(r["id"] for r in self.hardware_records if str(r.get("id", "")).startswith("art:"))
        ids.update(r["id"] for r in self.relation_records if str(r.get("id", "")).startswith("art:"))
        return sorted(ids)

    def criterion(self, cid: str) -> Criterion:
        return next(c for c in self.criteria if c.id == cid)


# --- assess ------------------------------------------------------------------------------------------------------------------


def _ref(rec: HardwareRecord, note: str) -> EvidenceRef:
    return EvidenceRef(id=rec.id, kind=rec.kind + (" (legacy tc_probe result)" if rec.legacy else ""), note=note)


def _fmt_rec(rec: HardwareRecord, names: set[str]) -> str:
    m = rec.mismatches_for(names)
    parts = [f"{rec.family or rec.sku}", f"{rec.elements:,} elements"]
    parts.append("mismatches=" + ("n/a" if m is None else f"{m:,}"))
    if rec.replays:
        parts.append("replays " + ", ".join(f"{r['kind']}:{r.get('records')}/{r.get('mismatches')}" for r in rec.replays))
    if rec.run_id:
        parts.append(rec.run_id)
    return "; ".join(parts)


def _family_ok(rec: HardwareRecord, names: set[str], fam: str, *, asserted: bool = True, allow_canary: bool = False) -> bool:
    f = rec.families.get(fam)
    if f is None or f["elements"] <= 0 or (rec.canary and not allow_canary):  # a canary's few tiles are a verdict, not coverage
        return False
    if not asserted:
        return True
    hits = [v for k, v in f["mismatches"].items() if k in names]
    return bool(hits) and min(hits) == 0


def _replay_summary(key: str, r: dict) -> dict:
    """One replay of a tc_probe result: records, and mismatches against the recorded words -- plus, where the replay also compared
    the declared model (``capture_replay``), the model's mismatches and the other tile positions' unexpected words."""
    bad = int(r.get("mismatches") or 0) + int(r.get("model_mismatches") or 0) + int(r.get("other_positions_nonzero") or 0)
    return {"kind": key.split("_")[0], "file": r.get("file"), "records": r.get("records"), "mismatches": bad, "pipeline": r.get("pipeline")}


def _replay_ok(rec: HardwareRecord, fragment: str) -> dict | None:
    for r in rec.replays:
        if fragment in str(r.get("file") or "") and int(r.get("records") or 0) > 0 and int(r.get("mismatches") or 0) == 0:
            return r
    return None


def _tile_covered(recs: list[HardwareRecord], names: set[str], tile: str) -> EvidenceRef | None:
    kind, _, what = tile.partition(":")
    for rec in recs:
        if kind == "family" and _family_ok(rec, names, what, allow_canary=True):  # a regression tile is small by design
            return _ref(rec, f"family {what}: {rec.families[what]['elements']:,} elements, 0 mismatches")
        if kind == "replay":
            r = _replay_ok(rec, what)
            if r:
                return _ref(rec, f"replay {r.get('file')}: {r.get('records')} records, 0 mismatches")
    return None


def _relation_attribution(rec: RelationRecord, subject: Subject, datasets: list[DatasetIdentity]) -> str | None:
    """Why a bench-result is about this subject (the first rule that fires), or None."""
    labels, wf = rec.labels, rec.fingerprint
    if any(v in subject.names for v in labels.get("target", [])):
        return f"label target={labels['target'][-1]}"
    rel_labels = RELATION_LABELS.get(subject.name, ()) + (RELATION_LABELS.get(subject.target.name, ()) if subject.target else ())
    for v in labels.get("relation", []):
        if v in rel_labels:
            return f"label relation={v}"
    if labels.get("relation") or labels.get("target"):
        return None                                                     # labelled for another relation
    rel = str(wf.get("relation") or "")
    if subject.instruction in rel:
        return f"fingerprint relation names {subject.instruction}"
    if wf.get("instruction") == subject.instruction:
        return "fingerprint instruction"
    model = wf.get("model")
    if isinstance(model, str) and _norm_model(model) in subject.model_names:
        return f"fingerprint model={model}"
    if isinstance(rel, str) and rel and (any(name in rel.lower() for name in subject.model_names)):
        return "fingerprint relation names the model"
    other_relation = bool(rel) or bool(wf.get("instruction")) or bool(model)
    if other_relation:
        return None                                                     # a fingerprint that names some other relation
    inst = wf.get("instances") if isinstance(wf.get("instances"), dict) else {}
    for ds in datasets:
        if wf.get("dataset") == ds.name or inst.get("dataset") == ds.name:
            return f"fingerprint dataset {ds.name}"
        em = wf.get("export_manifest") if isinstance(wf.get("export_manifest"), dict) else {}
        if em.get("tier") in ds.tiers:
            return f"fingerprint export_manifest.tier={em.get('tier')} of {ds.name}"
        root = str(rec.params.get("root") or rec.params.get("dir") or "")
        if root.rstrip("/").endswith(ds.name):
            return f"attempt params root ends with {ds.name}"
        tiers = wf.get("tiers")
        if isinstance(tiers, list) and tiers and set(tiers) <= set(ds.tiers):
            return f"fingerprint tiers of {ds.name}"
    if wf.get("profile") in subject.names:
        return f"fingerprint profile={wf.get('profile')}"
    return None


def assess(target_name: str, store: Any, *, repo: str | os.PathLike[str] | None = None, evidence: Evidence | None = None) -> TrustReport:
    """Compute the trust level of ``target_name`` (a target name, target constant or instruction id) from ``store`` and the repo."""
    subject = subjects(target_name)
    repo_path = Path(repo).resolve() if repo is not None else _find_repo()
    ev = evidence if evidence is not None else Evidence.collect(store)
    names = subject.model_names
    anchor = subject.anchor_family
    registry = subject.registry
    criteria: list[Criterion] = []
    notes: list[str] = []
    if repo_path is None:
        notes.append("repository root not found: fixture-based criteria (D-entries, datasets, references) are evaluated as absent")

    mine = [r for r in ev.hardware if r.instruction == subject.instruction]
    on_anchor = [r for r in mine if r.family == anchor]
    agreeing_anchor = [r for r in on_anchor if r.agrees(names)]
    disagreeing_anchor = [r for r in on_anchor if not r.agrees(names) and (r.mismatches_for(names) or 0) > 0]
    siblings = [i for i, ins in INSTRUCTIONS.items() if ins.model is not None and subject.model is not None and ins.model is subject.model and i != subject.instruction]
    all_entries = [e for e in _load_discrepancies(repo_path) if e.get("profile") in subject.names]
    d_entries = [e for e in all_entries if d_class(e) == "numerics"]      # model vs hardware words: these gate the levels
    binding_entries = [e for e in all_entries if d_class(e) == "binding"]  # which kernels/shapes realise the chain: dossier caveats
    datasets = _datasets_for(subject, repo_path) + _contract_instance_sets(subject)

    # ---- HYPOTHESIS ----
    c = Criterion("H1", "HYPOTHESIS", "a model is registered for the instruction", subject.model is not None)
    if subject.model is not None:
        c.details = {"model": subject.model.name, "validated": getattr(subject.model, "validated", "")[:200]}
    else:
        c.missing.append(f"{subject.instruction}: no Model in the registry (status {registry.status if registry else 'absent'})")
    criteria.append(c)

    c = Criterion("H2", "HYPOTHESIS", "the model agrees with hardware somewhere (fully or on some family), or is inherited from a sibling instruction that does", False)
    agreeing_any = [(r, r.partially_agrees(names)) for r in mine if r.partially_agrees(names)]
    if agreeing_any:
        c.passed = True
        c.evidence = [_ref(r, f"{why}: " + _fmt_rec(r, names)) for r, why in agreeing_any]
    else:
        sib_recs = [r for r in ev.hardware if r.instruction in siblings and r.agrees(names)]
        if sib_recs:
            c.passed = True
            c.evidence = [_ref(r, f"inherited from {r.instruction}: " + _fmt_rec(r, names)) for r in sib_recs]
            c.details["inherited_from"] = sorted({r.instruction for r in sib_recs if r.instruction})
        else:
            c.missing.append(f"no hardware record on any SKU where {subject.model.name if subject.model else 'the model'} reproduces every word of "
                             f"{subject.instruction}" + (f" (siblings sharing the model: {', '.join(siblings)})" if siblings else ""))
    criteria.append(c)

    # ---- PINNED ----
    # A disagreeing anchor record that a *resolved* D-entry cites as its hardware evidence is superseded: it measured the model as it
    # stood, the entry's resolution changed the model, and the re-sweep (P1/P2 agreeing records, P5 tiles) speaks for the current one.
    superseded_by: dict[str, list[str]] = {}
    for e in d_entries:
        if e.get("status") == "resolved":
            refs = {str(x) for x in (e.get("evidence_refs") or [])}
            for r in disagreeing_anchor:
                if (r.run_id and r.run_id in refs) or r.id in refs:
                    superseded_by.setdefault(r.id, []).append(e["id"])
    live_disagreeing = [r for r in disagreeing_anchor if r.id not in superseded_by]
    c = Criterion("P1", "PINNED", f"hardware evidence on the anchor SKU ({anchor}) with 0 mismatches for the model", bool(agreeing_anchor) and not live_disagreeing)
    c.evidence = [_ref(r, _fmt_rec(r, names)) for r in agreeing_anchor]
    for r in disagreeing_anchor:
        if r.id in superseded_by:
            c.evidence.append(_ref(r, f"SUPERSEDED (cited by resolved {', '.join(superseded_by[r.id])}): " + _fmt_rec(r, names)))
    if superseded_by:
        c.details["superseded"] = superseded_by
    open_ds = [e["id"] for e in d_entries if e.get("status") == "open"]
    disagreeing_anchor = live_disagreeing
    for r in disagreeing_anchor:
        c.evidence.append(_ref(r, ("CANARY " if r.canary else "") + "DISAGREES: " + _fmt_rec(r, names)))
        c.missing.append(f"{r.id}: {r.mismatches_for(names):,} mismatches on the anchor ({r.run_id}" + (", canary" if r.canary else "") + "); "
                         + (f"resolve the open D-entries {', '.join(open_ds)} with hardware as arbiter, then re-sweep" if open_ds
                            else "open a D-entry with this hardware evidence (relitigation procedure step 3), resolve it, then re-sweep"))
    if not on_anchor:
        others = sorted({r.family or r.sku or "?" for r in mine})
        c.missing.append(f"no run of {subject.instruction} on the anchor family {anchor}" + (f" (runs exist on: {', '.join(others)})" if others else ""))
    elif not agreeing_anchor and not disagreeing_anchor:
        c.missing.append(f"runs on {anchor} exist but none compares {subject.model.name if subject.model else 'the model'} with a count ({', '.join(r.id[:20] for r in on_anchor)})")
    legacy = [r for r in agreeing_anchor if r.legacy]
    if legacy:
        c.details["legacy_records"] = [r.id for r in legacy]
        notes.append(f"P1 rests on {len(legacy)} WP1 tc_probe result(s) recorded as bench-result/v1 (no instruction/model_agrees labels; counts read from "
                     "probe_results.json): re-registering them as tc-evidence/v1 would make the contract uniform")
    criteria.append(c)

    total = sum(r.elements for r in agreeing_anchor if not r.canary)
    c = Criterion("P2", "PINNED", f">= N_min = {N_MIN:,} elements with 0 mismatches on the anchor (canaries excluded)", total >= N_MIN)
    c.details = {"elements": total, "n_min": N_MIN}
    c.evidence = [_ref(r, f"{r.elements:,} elements") for r in agreeing_anchor if r.elements and not r.canary]
    if total < N_MIN:
        c.missing.append(f"{N_MIN - total:,} more 0-mismatch elements on {anchor} ({total:,} recorded); one default `tc_probe --sweep` is ~2.5e7")
    criteria.append(c)

    c = Criterion("P3", "PINNED", "the canonical family set is covered on the anchor with 0 mismatches", True)
    for role, fams in CANONICAL_FAMILIES.items():
        hit = None
        for fam in fams:
            if fam.startswith("replay:"):
                for r in on_anchor:
                    rep = _replay_ok(r, fam.split(":", 1)[1])
                    if rep:
                        hit = _ref(r, f"{role}: replay {rep.get('file')} ({rep.get('records')} records, 0 mismatches)")
                        break
            else:
                for r in on_anchor:
                    if _family_ok(r, names, fam, asserted=role not in UNASSERTED_ROLES):
                        f = r.families[fam]
                        hit = _ref(r, f"{role}: family {fam} ({f['elements']:,} elements" + (", 0 mismatches)" if role not in UNASSERTED_ROLES else ", recorded)"))
                        break
            if hit:
                break
        c.details[role] = hit.id if hit else None
        if hit:
            c.evidence.append(hit)
        else:
            c.passed = False
            c.missing.append(f"family role {role} ({' | '.join(fams)}) not recorded on {anchor} for {subject.instruction}")
    criteria.append(c)

    sass_info: dict = {"single_instruction": None, "source": None}
    c = Criterion("P4", "PINNED", "SASS shows one hardware instruction per PTX instruction, or the lowering is documented", False)
    for r in agreeing_anchor + on_anchor:
        if r.sass and r.sass.get("single_instruction"):
            c.passed = True
            c.evidence.append(_ref(r, f"cuobjdump: {r.sass.get('count')} MMA-class instruction in the tile kernel (nvcc {(r.nvcc or '?')[:40]})"))
            sass_info = {"single_instruction": True, "source": r.id, "count": r.sass.get("count"), "nvcc": r.nvcc}
            break
        if r.sass and r.sass.get("ok") and any(r.lowering.get(n) for n in names):
            c.passed = True
            low = next(r.lowering[n] for n in names if r.lowering.get(n))
            c.evidence.append(_ref(r, f"documented lowering {low}: {r.sass.get('count')} MMA-class instructions, model agrees"))
            sass_info = {"single_instruction": False, "source": r.id, "count": r.sass.get("count"), "lowering": low, "nvcc": r.nvcc}
            break
    if not c.passed:
        for chk in ev.compile_checks:
            n = chk["counts"].get(subject.instruction)
            if n == 1:
                c.passed = True
                c.evidence.append(EvidenceRef(chk["id"], "tc-compile-check/v1", f"{subject.instruction}.tile_kernel_mma_count = 1 (nvcc {(chk.get('nvcc') or '?')[:40]}; SASS is a property of ptxas+arch, not of the device)"))
                sass_info = {"single_instruction": True, "source": chk["id"], "count": 1, "nvcc": chk.get("nvcc")}
                break
    if not c.passed:
        c.missing.append(f"no `--sass-check` on an anchor run and no tc-compile-check/v1 with {subject.instruction}.tile_kernel_mma_count == 1; "
                         "run `tc_probe --compile-only --sass-check` (any host) or add --sass-check to the anchor sweep")
    criteria.append(c)

    c = Criterion("P5", "PINNED", "regression tiles of every resolved D-entry naming the subject replayed on the anchor with 0 mismatches", True)
    for e in d_entries:
        did, status = e.get("id"), e.get("status")
        if status not in ("resolved", "excluded"):
            continue
        tiles = REGRESSION_TILES.get(did)
        if tiles is None:
            c.passed = False
            c.missing.append(f"{did} ({status}) has no entry in REGRESSION_TILES: name the tc_probe family or replay that reproduces it")
            continue
        if not tiles:
            c.details[did] = "no hardware tile (proof-level, excluded domain or kernel-level)"
            continue
        for t in tiles:
            hit = _tile_covered(on_anchor, names, t)
            if hit:
                hit.note = f"{did}: " + hit.note
                c.evidence.append(hit)
            else:
                c.passed = False
                c.missing.append(f"{did}: regression tile {t} not replayed on {anchor} with 0 mismatches")
    criteria.append(c)

    # ---- TRUSTED ----
    refs_out: list[dict] = []
    c = Criterion("Ta", "TRUSTED", "an independent reference agrees, or every disagreement with it is resolved with hardware as arbiter", False)
    for ref in INDEPENDENT_REFERENCES.get(subject.instruction, ()):
        present = repo_path is not None and all((repo_path / p).is_file() for p in ref.paths)
        entry = {"name": ref.name, "kind": ref.kind, "paths": list(ref.paths), "present": present, "verdict": None, "evidence": [], "note": ref.note}
        cites = [e for e in d_entries if set(e.get("evidence_refs") or []) & set(ref.paths)]
        if not present:
            entry["verdict"] = "missing: " + ", ".join(p for p in ref.paths if repo_path is None or not (repo_path / p).is_file())
        else:
            if ref.replay:
                for r in on_anchor:
                    rep = _replay_ok(r, ref.replay)
                    if rep:
                        entry["verdict"] = f"agrees: {rep.get('records')} records replayed on {anchor} with 0 mismatches"
                        entry["evidence"].append(r.id)
                        c.evidence.append(_ref(r, f"{ref.name}: {entry['verdict']}"))
                        break
            if entry["verdict"] is None and ref.replica:
                doc = _replica(repo_path, ref.replica) or {}
                conf = ((doc.get("files") or {}).get("summary.json") or {}).get("conformance") or {}
                same = doc.get("instruction") == subject.instruction and _norm_model(str(doc.get("model") or "")) in names
                if same and conf.get("mismatches") == 0 and int(conf.get("words_compared") or 0) > 0:
                    entry["verdict"] = f"agrees: replica records {int(conf['words_compared']):,} words, 0 mismatches for {doc.get('model')}"
                    entry["evidence"].append(ref.replica)
                    c.evidence.append(EvidenceRef(ref.replica, "fixture (replica)", f"{ref.name}: {entry['verdict']}"))
                elif same:
                    entry["verdict"] = f"replica disagrees or has no structured conformance ({conf.get('mismatches')!r})"
                else:
                    entry["verdict"] = f"replica is about {doc.get('instruction')} / {doc.get('model')}, not this subject"
            if entry["verdict"] is None:
                if cites and all(e.get("status") in ("resolved", "excluded") for e in cites):
                    entry["verdict"] = "disagreements resolved with hardware as arbiter: " + ", ".join(e["id"] for e in cites)
                    for e in cites:
                        c.evidence.append(EvidenceRef(e["id"], "discrepancy-log", f"{ref.name}: {e['status']}"))
                elif cites:
                    entry["verdict"] = "cited by open D-entries: " + ", ".join(e["id"] for e in cites if e.get("status") == "open")
                else:
                    entry["verdict"] = "unassessed: no replay on the anchor, no replica conformance and no D-entry cites it"
        entry["agrees"] = str(entry["verdict"]).startswith(("agrees", "disagreements resolved"))
        refs_out.append(entry)
    c.passed = any(e["agrees"] for e in refs_out)
    if not refs_out:
        c.missing.append(f"no independent reference catalogued for {subject.instruction} (INDEPENDENT_REFERENCES): add Hawkeye / autoproof / a second compiled path")
    elif not c.passed:
        for e in refs_out:
            c.missing.append(f"{e['name']}: {e['verdict']}")
    criteria.append(c)

    c = Criterion("Tb", "TRUSTED", "every numerics-class discrepancy-log entry naming the target or its instruction is resolved (or excluded from the domain); "
                  "binding-class entries are deployment caveats, not gates", True)
    for e in d_entries:
        status = e.get("status")
        c.evidence.append(EvidenceRef(e["id"], "discrepancy-log", f"{status}: {str(e.get('issue') or '')[:120]}"))
        if status not in ("resolved", "excluded"):
            c.passed = False
            c.missing.append(f"{e['id']} is {status} (owner {e.get('owner') or '?'}): {str(e.get('to_resolve') or e.get('issue') or '')[:200]}")
    c.details = {"entries": [e["id"] for e in d_entries], "open": [e["id"] for e in d_entries if e.get("status") == "open"],
                 "binding": [e["id"] for e in binding_entries]}
    criteria.append(c)

    attributed = [(r, why) for r in ev.relation for why in [_relation_attribution(r, subject, datasets)] if why]
    c = Criterion("Tc", "TRUSTED", f"relation == model: the prover's checker accepts >= {HONEST_MIN:,} honest VUs (incl. edge families) and rejects >= {NEGATIVES_MIN} mutations", False)
    best_h, best_n = 0, 0
    for r, why in attributed:
        if not r.passed:
            continue
        h, n = r.honest(), r.negatives()
        if h or n:
            c.evidence.append(EvidenceRef(r.id, "bench-result/v1", f"{why}; honest {h:,} / negatives {n} ({r.validation.get('checker')}: {str(r.validation.get('detail'))[:100]})"))
            best_h, best_n = max(best_h, h), max(best_n, n)
    edge = {}
    for ds in datasets:
        for t in ds.honest_tiers:
            fams = ds.corner_families(t)
            if fams:
                edge[t] = fams
    c.details = {"honest": best_h, "negatives": best_n, "edge_families": edge}
    c.passed = best_h >= HONEST_MIN and best_n >= NEGATIVES_MIN and (bool(edge) or not datasets)
    if best_h < HONEST_MIN:
        c.missing.append(f"honest set: {best_h:,} accepted VUs recorded (need {HONEST_MIN:,}) in a bench-result/v1 validation attributed to the target")
    if best_n < NEGATIVES_MIN:
        c.missing.append(f"negative set: {best_n} rejected mutations recorded (need {NEGATIVES_MIN}); extend the negative tier and re-run the checker sweep")
    if datasets and not edge:
        if all(ds.contract for ds in datasets):
            c.missing.append("the instance set is a seeded recipe pinned by contract digest with no corner families: freeze a fixture manifest with edge families "
                             "(or add a corner tier to the recipe) so the honest set includes them")
        else:
            c.missing.append("the frozen honest set has no edge (corner) families in its manifest")
    if not datasets and subject.target is not None:
        c.missing.append(f"no frozen instance set (fixtures/*/v*/manifest.json with target.profile = {subject.name}, or a contract-pinned recipe digest in "
                         f"CONTRACT_INSTANCE_SETS) whose edge families the honest set must include")
    criteria.append(c)

    c = Criterion("Td", "TRUSTED", "hardware-in-the-loop: the frozen instance set (a repo fixture manifest or a contract-pinned recipe digest) re-executed "
                  "on the anchor with 0 mismatches", False)
    for ds in datasets:
        covered: dict[str, EvidenceRef] = {}
        for r in ev.hardware:                                            # every replay of this dataset, on whatever silicon
            if r.instances is None:
                continue
            sha = r.instances.get("manifest_sha256")
            # a fixture set is named by its dataset; a contract-pinned set is its recipe digest, whatever the replay called the set
            # (instances-hw recorded the FP8 Ada set as `fp8-ada synthetic ...` / `vu-k1536-fp8` with the digest as manifest_sha256)
            by_digest = bool(ds.contract and sha and sha == ds.sha256)
            if r.instances.get("dataset") != ds.name and not by_digest:
                continue
            if r.family != anchor:
                c.missing.append(f"{ds.name}: replay {r.id[:24]} ({r.run_id}) ran on {r.family or r.sku} as {r.instruction or 'an unnamed instruction'}, not on the anchor {anchor}")
                continue
            if r.instruction not in (None, subject.instruction):
                continue
            if sha and sha != ds.sha256:
                c.missing.append(f"{ds.name}: replay {r.id[:24]} checked {'digest' if ds.contract else 'manifest'} {sha[:12]}..., "
                                 f"{ds.path} pins {ds.sha256[:12]}...; re-run against the frozen set")
                continue
            for tier, t in (r.instances.get("tiers") or {}).items():
                if t.get("mismatches") != 0:
                    continue
                # the digest identifies one recipe, i.e. one tier: a replay that matched it covers the contract tier under its own name
                key = tier if tier in ds.honest_tiers or not by_digest or len(ds.honest_tiers) != 1 else ds.honest_tiers[0]
                if key not in covered:
                    named = f" (recorded as {r.instances.get('dataset')}/{tier})" if key != tier else ""
                    covered[key] = _ref(r, f"{ds.name}/{key}: {t.get('instances') or '?'} instances, 0 mismatches on {anchor}"
                                           f"{'; recipe digest ' + sha[:12] + '...' if by_digest else ''}{named}")
        for tier in ds.honest_tiers:
            if tier in covered:
                c.evidence.append(covered[tier])
            elif ds.contract:
                c.missing.append(f"{ds.name}/{tier}: no tc-evidence/v1 on {anchor} labelled family=frozen-instances mismatches=0 whose manifest_sha256 is the "
                                 f"contract digest {ds.sha256[:12]}... ({ds.path}; or labelled instances_dataset={ds.name} instances_tier={tier})")
            else:
                c.missing.append(f"{ds.name}/{tier}: no tc-evidence/v1 on {anchor} labelled family=frozen-instances instances_dataset={ds.name} instances_tier={tier} mismatches=0")
        c.details[ds.name] = {"tiers": ds.honest_tiers, "covered": sorted(covered), "manifest_sha256": ds.sha256, "identified_by": ds.path,
                              "contract": ds.contract}
        if all(t in covered for t in ds.honest_tiers):
            c.passed = True
    if not datasets:
        c.missing.append(f"no frozen instance set to re-execute on {anchor}: freeze one (fixtures/<dataset>/v<k>/manifest.json with target.profile = {subject.name}, "
                         f"or pin a recipe digest in the bench contract and list it in CONTRACT_INSTANCE_SETS), then `tc_probe --replay-instances` on the anchor "
                         f"labelled family=frozen-instances")
    criteria.append(c)

    c = Criterion("Te", "TRUSTED", "an independent verifier accepted a full-contract proof of the relation", False)
    for r, why in attributed:
        if not (r.passed and r.independently_verified):
            continue
        if "RELATION-ONLY" in r.checker_text.upper():
            continue
        pc = r.proof_class
        if pc and pc not in FULL_CONTRACT_PROOF_CLASSES:
            continue
        c.passed = True
        ver = r.fingerprint.get("verifier") if isinstance(r.fingerprint.get("verifier"), dict) else {}
        c.evidence.append(EvidenceRef(r.id, "bench-result/v1", f"{why}; proof_class={pc}; verifier {ver.get('name') or r.validation.get('checker')}; {str(r.validation.get('detail'))[:80]}"))
    if not c.passed:
        seen = [r for r, _ in attributed if r.passed and ("prove" in r.checker_text.lower() or r.proof_class)]
        c.missing.append(f"no bench-result/v1 attributed to the target with independently_verified=true and a full-contract proof_class "
                         f"({', '.join(FULL_CONTRACT_PROOF_CLASSES)})" + (f"; {len(seen)} proof run(s) exist without independent verification" if seen else ""))
    criteria.append(c)

    c = Criterion("Tf", "TRUSTED", "no open trust_downgrade label (canary mismatch, verifier rejection, red-team finding) on the subject's evidence", True)
    scope: list[tuple[str, dict[str, list[str]]]] = [(r.id, r.labels) for r in mine] + [(r.id, r.labels) for r, _ in attributed]
    scope += [(d["id"], d["labels"]) for d in ev.dossiers if d.get("target") in subject.names]
    for art_id, labels in scope:
        vals = labels.get("trust_downgrade")
        if vals and not str(vals[-1]).startswith("resolved"):
            c.passed = False
            c.evidence.append(EvidenceRef(art_id, "label", f"trust_downgrade={vals[-1]}"))
            c.missing.append(f"{art_id}: trust_downgrade={vals[-1]} is open; follow RELITIGATION_PROCEDURE and close it with trust_downgrade=resolved:<D-id>")
    criteria.append(c)

    # ---- TRUSTED_STABLE ----
    c = Criterion("S1", "TRUSTED_STABLE", ">= 2 physical devices of the anchor SKU (GPU UUIDs) or >= 2 driver/ptxas environments reproduce the model's words "
                  f"(agreeing anchor runs of >= {STABLE_MIN_ELEMENTS:,} words: a canary suffices)", False)
    stable_recs = [r for r in agreeing_anchor if r.elements >= STABLE_MIN_ELEMENTS or r.replay_records() >= STABLE_MIN_ELEMENTS]
    devices = sorted({r.gpu_uuid for r in stable_recs if r.gpu_uuid})
    envs = sorted({r.environment() for r in stable_recs if any(r.environment())})
    c.details = {"devices": devices, "environments": [list(e) for e in envs],
                 "hosts_without_gpu_uuid": sorted({r.host for r in stable_recs if r.host and not r.gpu_uuid})}
    c.evidence = [_ref(r, f"device {r.gpu_uuid or '(no GPU UUID; host ' + (r.host or '?') + ')'}; driver {r.driver or '?'}; {(r.nvcc or '?')[:44]}") for r in stable_recs]
    c.passed = len(devices) >= 2 or len(envs) >= 2
    if not c.passed:
        c.missing.append(f"only {len(devices)} GPU UUID / {len(envs)} driver+nvcc environment among the agreeing anchor runs (hostnames are not device "
                         f"identities); run `tc_probe --canary` on a second {anchor} (label device=<GPU UUID>) or after a driver/ptxas change")
    criteria.append(c)

    # ---- level ----
    level = Level.UNPROBED
    for lv in (Level.HYPOTHESIS, Level.PINNED, Level.TRUSTED, Level.TRUSTED_STABLE):
        if all(cr.passed for cr in criteria if Level[cr.level] <= lv):
            level = lv
        else:
            break
    next_level = Level(level + 1) if level < Level.TRUSTED_STABLE else None
    missing_to_next = [m for cr in criteria if next_level is not None and Level[cr.level] == next_level and not cr.passed for m in cr.missing]
    if registry is not None and registry_level(registry.status) > level:
        notes.append(f"REGISTRY AHEAD OF THE EVIDENCE: registry asserts status={registry.status} but the computed level is {level.name}; "
                     f"downgrade Instruction.status to {level.name.lower()} citing this dossier (registry status <= computed level)")
    if registry is not None and registry.status != "pinned" and level >= Level.PINNED:
        notes.append(f"registry says status={registry.status} but the evidence pins the model: update the registry")
    if registry is not None and registry.dossier is None:
        notes.append("registry Instruction.dossier is unset: point it at the published dossier id")
    canaries = [r for r in on_anchor if r.canary]
    failing_canaries = [r for r in canaries if not r.agrees(names)]
    if failing_canaries:
        notes.append(f"INVALIDATION TRIGGER canary-mismatch: {len(failing_canaries)} canary run(s) on {anchor} disagree with the model "
                     f"({', '.join(r.id for r in failing_canaries)}); P1 fails until a D-entry resolves them (relitigation procedure)")

    return TrustReport(
        subject=subject.name, target=subject.target.name if subject.target else None, instruction=subject.instruction,
        model=subject.model.name if subject.model else None, anchor_sku=subject.anchor_sku, anchor_family=anchor,
        level=level.name, registry_status=registry.status if registry else None, criteria=criteria,
        next_level=next_level.name if next_level else None, missing_to_next=missing_to_next,
        d_entries=[_d_summary(e) for e in d_entries],
        deployment_caveats=[_d_summary(e) for e in binding_entries],
        references=refs_out, sass=sass_info,
        hardware_records=[{"id": r.id, "kind": r.kind, "run_id": r.run_id, "family": r.family, "sku": r.sku, "elements": r.elements,
                           "mismatches": r.mismatches_for(names), "agrees": r.agrees(names), "families": sorted(r.families), "legacy": r.legacy, "canary": r.canary,
                           "replays": [{k: v for k, v in rp.items() if k in ("kind", "file", "records", "mismatches")} for rp in r.replays],
                           "driver": r.driver, "nvcc": (r.nvcc or "")[:60] or None, "host": r.host, "labels": {k: v[-1] for k, v in r.labels.items()}}
                          for r in mine],
        relation_records=[{"id": r.id, "run_id": r.run_id, "attribution": why, "status": r.validation.get("status"), "checker": r.validation.get("checker"),
                           "proof_class": r.proof_class, "independently_verified": r.independently_verified, "honest": r.honest(), "negatives": r.negatives(),
                           "labels": {k: v[-1] for k, v in r.labels.items()}}
                          for r, why in attributed if r.tiers or r.proof_class or r.independently_verified or "honest" in r.labels],
        invalidation_triggers=[dict(t) for t in INVALIDATION_TRIGGERS], relitigation_procedure=list(RELITIGATION_PROCEDURE),
        constants={"N_MIN": N_MIN, "HONEST_MIN": HONEST_MIN, "NEGATIVES_MIN": NEGATIVES_MIN, "canonical_families": {k: list(v) for k, v in CANONICAL_FAMILIES.items()},
                   "full_contract_proof_classes": list(FULL_CONTRACT_PROOF_CLASSES)},
        generated_utc=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        previous_dossiers=[d["id"] for d in ev.dossiers if d.get("target") in subject.names], notes=notes,
        canaries=[{"id": r.id, "run_id": r.run_id, "passed": r.agrees(names), "elements": r.elements, "mismatches": r.mismatches_for(names),
                   "driver": r.driver, "nvcc": (r.nvcc or "")[:60] or None} for r in canaries],
    )


def _d_summary(e: dict) -> dict:
    return {"id": e["id"], "class": d_class(e), "status": e.get("status"), "issue": e.get("issue"), "resolution": e.get("resolution"), "owner": e.get("owner"),
            "evidence_refs": list(e.get("evidence_refs") or []), "regression_tiles": list(REGRESSION_TILES.get(e["id"], ()))}


# --- rendering ---------------------------------------------------------------------------------------------------------------


MD_EVIDENCE_PER_CRITERION = 8  # the markdown table shows this many evidence refs per criterion; the JSON dossier holds them all


def render_markdown(rep: TrustReport) -> str:
    out: list[str] = []
    out.append(f"# Trust dossier: {rep.subject}")
    out.append("")
    out.append(f"**Level: {rep.level}**" + (f" (next: {rep.next_level})" if rep.next_level else "") +
               f" -- instruction `{rep.instruction}`, model `{rep.model}`, anchor {rep.anchor_sku} (family `{rep.anchor_family}`), "
               f"registry status `{rep.registry_status}`, generated {rep.generated_utc}.")
    out.append("")
    out.append("Trust is computed from the evidence below, never asserted; it re-opens only through an invalidation trigger.")
    for n in rep.notes:
        out.append(f"- note: {n}")
    out.append("")
    out.append("## Criteria")
    out.append("")
    out.append("| id | level | criterion | result | evidence |")
    out.append("|---|---|---|---|---|")
    for c in rep.criteria:
        shown = c.evidence[:MD_EVIDENCE_PER_CRITERION]
        ev = "<br>".join(f"`{e.id}` {e.note}".replace("|", "\\|") for e in shown) or "-"
        if len(c.evidence) > len(shown):
            ev += f"<br>... +{len(c.evidence) - len(shown)} more (every id is in the JSON dossier)"
        out.append(f"| {c.id} | {c.level} | {c.title.replace('|', '/')} | {'PASS' if c.passed else 'FAIL'} | {ev} |")
    out.append("")
    if rep.missing_to_next:
        out.append(f"## Missing to reach {rep.next_level}")
        out.append("")
        for m in rep.missing_to_next:
            out.append(f"- {m}")
        out.append("")
    later = [m for c in rep.criteria if not c.passed and c.level != rep.next_level for m in c.missing]
    if later:
        out.append("## Also failing (later levels)")
        out.append("")
        for m in later:
            out.append(f"- {m}")
        out.append("")
    out.append("## Discrepancy-log entries naming the subject (numerics)")
    out.append("")
    if rep.d_entries:
        out.append("| entry | status | issue | resolution | regression tiles |")
        out.append("|---|---|---|---|---|")
        for e in rep.d_entries:
            out.append(f"| {e['id']} | {e['status']} | {str(e.get('issue') or '')[:140].replace('|', '/')} | {str(e.get('resolution') or '')[:160].replace('|', '/')} | {', '.join(e['regression_tiles']) or '-'} |")
    else:
        out.append("none")
    out.append("")
    out.append("## Deployment caveats (binding-class entries: which kernels / shapes realise the chain; not a gate on the circuit)")
    out.append("")
    if rep.deployment_caveats:
        out.append("| entry | status | owner | issue | resolution |")
        out.append("|---|---|---|---|---|")
        for e in rep.deployment_caveats:
            out.append(f"| {e['id']} | {e['status']} | {e.get('owner') or '-'} | {str(e.get('issue') or '')[:160].replace('|', '/')} | {str(e.get('resolution') or '')[:160].replace('|', '/')} |")
    else:
        out.append("none")
    out.append("")
    out.append("## Independent references")
    out.append("")
    if rep.references:
        for r in rep.references:
            out.append(f"- **{r['name']}** ({r['kind']}): {r['verdict']}" + (f" -- {r['note']}" if r.get("note") else "") + f"; paths: {', '.join(r['paths'])}")
    else:
        out.append("none catalogued")
    out.append("")
    out.append("## SASS / lowering")
    out.append("")
    out.append(f"`{json.dumps(rep.sass, sort_keys=True)}`")
    out.append("")
    out.append("## Hardware records for the instruction")
    out.append("")
    if rep.hardware_records:
        out.append("| artifact | kind | run | family | elements | mismatches | agrees | replays |")
        out.append("|---|---|---|---|---|---|---|---|")
        for r in rep.hardware_records:
            reps = ", ".join(f"{x['kind']} {x.get('records')}/{x.get('mismatches')}" for x in r["replays"]) or "-"
            tag = " (legacy)" if r["legacy"] else (" (canary)" if r.get("canary") else "")
            out.append(f"| `{r['id']}` | {r['kind']}{tag} | {r['run_id'] or '-'} | {r['family'] or r['sku']} | {r['elements']:,} | "
                       f"{'n/a' if r['mismatches'] is None else r['mismatches']} | {'yes' if r['agrees'] else 'no'} | {reps} |")
    else:
        out.append("none")
    out.append("")
    out.append("## Canaries on the anchor")
    out.append("")
    if rep.canaries:
        for cn in rep.canaries:
            out.append(f"- `{cn['id']}` ({cn['run_id'] or '-'}): {'PASS' if cn['passed'] else 'FAIL -- invalidation trigger'}; {cn['elements']:,} elements, "
                       f"mismatches {'n/a' if cn['mismatches'] is None else cn['mismatches']}; driver {cn.get('driver') or '?'}, nvcc {cn.get('nvcc') or '?'}")
    else:
        out.append("none recorded (`benchmarks/tc_probe/tc_probe.py --instruction <id> --canary` on the anchor; label the result `family=canary`)")
    out.append("")
    out.append("## Relation records attributed to the target")
    out.append("")
    if rep.relation_records:
        out.append("| artifact | run | attribution | status | proof_class | independently verified | honest | negatives |")
        out.append("|---|---|---|---|---|---|---|---|")
        for r in rep.relation_records:
            out.append(f"| `{r['id']}` | {r['run_id'] or '-'} | {r['attribution']} | {r['status']} | {r['proof_class'] or '-'} | {'yes' if r['independently_verified'] else 'no'} | {r['honest']:,} | {r['negatives']} |")
    else:
        out.append("none")
    out.append("")
    out.append("## Invalidation triggers")
    out.append("")
    for t in rep.invalidation_triggers:
        out.append(f"- **{t['id']}**: {t['event']} -- recorded as `{t['label']}`; detects {t['detects']}")
    out.append("")
    out.append("## Relitigation procedure")
    out.append("")
    for step in rep.relitigation_procedure:
        out.append(f"- {step}")
    out.append("")
    out.append(f"Constants: N_min = {rep.constants['N_MIN']:,}; honest >= {rep.constants['HONEST_MIN']:,}; negatives >= {rep.constants['NEGATIVES_MIN']}; "
               f"canonical families = {json.dumps(rep.constants['canonical_families'])}.")
    if rep.previous_dossiers:
        out.append("")
        out.append("Previous dossiers: " + ", ".join(f"`{d}`" for d in rep.previous_dossiers))
    return "\n".join(out) + "\n"


# --- publish -----------------------------------------------------------------------------------------------------------------


def publish(rep: TrustReport, store: Any, *, by: str = "trust", push: bool = True) -> dict:
    """Write the dossier as a ``trust-dossier/v1`` artifact (meta = the report, refs.cited = every cited artifact), label it, and push
    ``--preserve`` when the store has a remote.  Returns ``{"id", "pushed", "labels"}``."""
    meta = {"schema": DOSSIER_SCHEMA, **rep.to_dict()}
    refs: dict[str, Any] = {"cited": rep.cited()}
    if rep.previous_dossiers:
        refs["previous"] = list(rep.previous_dossiers)
    art_id = store.put_artifact(DOSSIER_KIND, meta, refs)
    labels = {"target": rep.subject, "instruction": rep.instruction, "level": rep.level, "anchor": rep.anchor_family}
    existing = {}
    for lab in store.labels(art_id):
        existing.setdefault(lab.key, []).append(lab.value)
    for k, v in labels.items():
        if v not in existing.get(k, []):
            store.add_label(art_id, k, v, by)
    pushed = None
    if push:
        remote = getattr(store, "remote", None)
        if remote is not None:
            pushed = store.push(art_id, preserve=True)
        else:
            pushed = {"skipped": "no remote configured"}
    return {"id": art_id, "pushed": pushed, "labels": labels}


# --- CLI ---------------------------------------------------------------------------------------------------------------------


DEFAULT_STORE_CLASS = "research.store.local:LocalStore"


def _open_store(root: str | None, store_class: str = DEFAULT_STORE_CLASS):
    """The evidence store for the CLI: the store class is named as data (``module:Class``) and resolved at run time; any
    object with the :class:`LocalStore` surface documented at the top of this module works."""
    import importlib

    mod, _, cls = store_class.partition(":")
    store_cls = getattr(importlib.import_module(mod), cls or "LocalStore")
    return store_cls(root) if root else store_cls()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python benchmarks/tc_probe/trust.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("target", nargs="*", help="target name, target constant (FIRST) or instruction id; or --all")
    ap.add_argument("--all", action="store_true", help="every Target plus every modelled registry instruction")
    ap.add_argument("--root", default=None, help="store root (default: the store class's default, ~/.research/store)")
    ap.add_argument("--store-class", default=DEFAULT_STORE_CLASS, help="module:Class of the store to open (default %(default)s)")
    ap.add_argument("--repo", default=None, help="repository root (default: found from this file / cwd)")
    ap.add_argument("--format", choices=("md", "json"), default="md")
    ap.add_argument("--publish", action="store_true", help="write trust-dossier/v1 artifacts and push --preserve")
    ap.add_argument("--no-push", action="store_true", help="with --publish: write locally, do not push")
    ap.add_argument("--out", default=None, help="directory to write <subject>.md/.json into (default: stdout)")
    args = ap.parse_args(argv)
    names = all_subject_names() if args.all else list(args.target)
    if not names:
        ap.error("give a target/instruction or --all")
    store = _open_store(args.root, args.store_class)
    evidence = Evidence.collect(store)
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    published: list[dict] = []
    ahead: list[str] = []
    for name in names:
        rep = assess(name, store, repo=args.repo, evidence=evidence)
        text = json.dumps({"schema": DOSSIER_SCHEMA, **rep.to_dict()}, indent=1, sort_keys=True) if args.format == "json" else render_markdown(rep)
        if out_dir:
            (out_dir / (re.sub(r"[^A-Za-z0-9._-]", "_", rep.subject) + ("." + args.format))).write_text(text)
        else:
            sys.stdout.write(text if args.format == "md" else text + "\n")
        if args.publish:
            res = publish(rep, store, push=not args.no_push)
            published.append({"subject": rep.subject, **res})
            print(f"published {rep.subject}: {res['id']} level={rep.level} pushed={json.dumps(res['pushed'])[:120]}", file=sys.stderr)
        if rep.target is None and registry_level(rep.registry_status) > Level[rep.level]:
            ahead.append(f"{rep.instruction}: registry status={rep.registry_status} > computed {rep.level}" + (f" (dossier {published[-1]['id']})" if published else ""))
    if published and args.format == "json" and not out_dir:
        sys.stdout.write(json.dumps(published, indent=1) + "\n")
    for line in ahead:
        print(f"REGISTRY AHEAD OF THE EVIDENCE  {line}: downgrade Instruction.status and cite the dossier", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
