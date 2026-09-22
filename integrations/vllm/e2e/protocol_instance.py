"""Protocol-instance description: the pre-challenge experiment, machine-readable (agreement 1, GUIDANCE §57.3; lane PROTO-A).

One JSON document fixes, BEFORE any challenge is drawn, everything the bound and the verifier's acceptance rule
depend on: the program (digest + identity + Input-inclusion convention), the numerical vocabulary/profile, the
VU/RU partition (:= the validated plan, with the population counts DERIVED from that plan), the designated
outputs (declared and vocabulary-implied domains, multiplicities, ``W_out``), the sampler law with its
seed-derivation rule and commitment -> challenge chronology, the mandatory vs diagnostic obligations, and the
security threshold + Bound algorithm version.  Prose and the field table: ``docs/vllm-poc/protocol-instance.md``.

    python -m e2e.protocol_instance validate INSTANCE.json                 # strict schema + semantic rules
    python -m e2e.protocol_instance check INSTANCE.json --record DIR [--validation ART.json]   # cross-check vs a record
    python -m e2e.protocol_instance digest INSTANCE.json

Strictness (agreement-1 rule): unknown fields and missing required fields are errors, everywhere in the document.
Every number-bearing field is a :class:`Quantity` ``{"value", "label", "source"}`` with ``label`` in
{MEASURED, MODELED, ANALYTICAL} and a non-empty ``source`` (path#field or a named computation).  Semantic
rules (``validate_instance``) enforce the brief's separations: population counts come from the validated plan
and equal the validation artifact's; realized sample sizes never appear as configured probabilities; a
fixed-size law cannot carry an attached Bound without the restricted-sample analysis reference; the chronology
has every commitment before every challenge that depends on it and records every challenge ever drawn
(``rerolls`` must be 0); diagnostic obligations are disjoint from mandatory ones; ``W_out`` equals the sum of
multiplicity x declared bits over the designated outputs.

The loader has no dependency on the plan machinery: it validates a document.  ``check_against_record`` reads a
``run_composed`` record directory and a compact-validation artifact and reports every mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, fields, is_dataclass
from fractions import Fraction
from pathlib import Path
from types import UnionType
from typing import Any, Sequence, Union, get_args, get_origin, get_type_hints

_POC = str(Path(__file__).resolve().parents[1])
if _POC not in sys.path:  # for `frontier.result_schema` (COST-D's stage vocabulary); e2e itself is imported as a package from vllm-poc
    sys.path.insert(0, _POC)

SCHEMA = "veritor/protocol-instance/v1"
LABELS = ("MEASURED", "MODELED", "ANALYTICAL")
LAWS = ("bernoulli", "fixed_size")
STATUSES = ("pre-challenge", "historical-reconstruction")
SERVE_FAMILIES: tuple[str, ...] = ("Serve_v1", "Serve_v4")  # [E7 D54] identity.family values of the (case, LP, STEPS) serving programs; == sorted(verity_vllm.registry.b1.SERVE_FAMILIES)
INPUT_INCLUSION = ("gates_include_input_gates", "operator_gates_only")
COUNT_BASES = ("validated_plan", "bucketed_upper", "physical_leaves", "analytical_census")
EVENT_KINDS = ("commitment", "challenge", "declaration", "message")
WEIGHT_VARIANTS = ("V1", "V2", "V3", "none")
PLAN_DIGEST_KINDS = ("plan-json/v1", "plan-rule/v1")  # veritor.plans.summary.PLAN_DIGEST_JSON / PLAN_DIGEST_RULE
PARTITION_DIGEST_KIND = "partition-intervals/v1"  # veritor.plans.summary.PARTITION_DIGEST (SUMMARY-C objection 1, draft 2)
FIRST_BOUNDARY_RULES = ("outgoing boundary runs of every CU", "designated outputs only")  # COST-D objection 1
ACQUISITION = ("request", "ru")  # COST-D objection 4
PROOF_STEPS = ("witness", "checker", "argument")  # COST-D objection 2
WEIGHT_CHECK_STAGE = "verify.weights_unconditional"  # COST-D agreement 3 draft 1a (objection 6)
CHRONOLOGY_KINDS = ("security-sample/v1", "adapter-transcript/v1", "partition-sample/v1")  # REC draft 3; [L1 RECORD] e2e.partition_sample over population/v2
WEIGHTS_VARIANT = "unconditional weight binding"  # GUIDANCE §57.10 S5
_STAGE_RE = re.compile(r"^[a-z]+\.[a-z_]+$")


def _known_stages() -> frozenset[str] | None:
    """COST-D's closed stage vocabulary (agreement 3) when the frontier package is importable; None otherwise (format check only)."""

    try:
        from frontier.result_schema import STAGE_NAMES  # vllm-poc/frontier

        return frozenset(STAGE_NAMES)
    except Exception:  # noqa: BLE001 -- the vocabulary is another lane's; absence must not make the instance unloadable
        return None
_HEX64 = frozenset("0123456789abcdef")


class InstanceError(ValueError):
    """A protocol-instance document is malformed or violates a semantic rule."""


# -- generic strict decoding --------------------------------------------------------------------------------------------


def _is_optional(tp: Any) -> bool:
    return get_origin(tp) is not None and type(None) in get_args(tp)


def _strip_optional(tp: Any) -> Any:
    if _is_optional(tp):
        args = [a for a in get_args(tp) if a is not type(None)]
        return args[0] if len(args) == 1 else args
    return tp


def _decode(tp: Any, value: Any, path: str) -> Any:
    """Decode ``value`` as ``tp`` strictly: exact types, no unknown keys, no missing required keys."""

    if _is_optional(tp):
        if value is None:
            return None
        tp = _strip_optional(tp)
    if is_dataclass(tp):
        if not isinstance(value, dict):
            raise InstanceError(f"{path}: expected an object for {tp.__name__}, got {type(value).__name__}")
        hints = get_type_hints(tp)
        names = {f.name for f in fields(tp)}
        unknown = sorted(set(value) - names)
        if unknown:
            raise InstanceError(f"{path}: unknown field(s) {unknown} (agreement-1 strictness: no unknown fields)")
        kwargs = {}
        for f in fields(tp):
            sub = f"{path}.{f.name}"
            if f.name not in value:
                # a field is required unless its type is Optional (then absent == null); dataclass defaults are not consulted
                if _is_optional(hints[f.name]):
                    kwargs[f.name] = None
                    continue
                raise InstanceError(f"{sub}: required field missing")
            kwargs[f.name] = _decode(hints[f.name], value[f.name], sub)
        return tp(**kwargs)
    origin = get_origin(tp)
    if origin in (Union, UnionType):
        for member in get_args(tp):
            if member is type(None):
                continue
            try:
                return _decode(member, value, path)
            except InstanceError:
                continue
        raise InstanceError(f"{path}: value {value!r} matches none of {[getattr(a, '__name__', str(a)) for a in get_args(tp)]}")
    if origin in (list, tuple):
        if not isinstance(value, list):
            raise InstanceError(f"{path}: expected a list, got {type(value).__name__}")
        (inner,) = get_args(tp)[:1]
        items = [_decode(inner, v, f"{path}[{i}]") for i, v in enumerate(value)]
        return tuple(items) if origin is tuple else items
    if origin is dict:
        if not isinstance(value, dict):
            raise InstanceError(f"{path}: expected an object, got {type(value).__name__}")
        kt, vt = get_args(tp)
        return {_decode(kt, k, f"{path}<key>"): _decode(vt, v, f"{path}[{k!r}]") for k, v in value.items()}
    if tp is Any:
        return value
    if tp is bool:
        if type(value) is not bool:
            raise InstanceError(f"{path}: expected a boolean, got {type(value).__name__}")
        return value
    if tp is int:
        if type(value) is not int:
            raise InstanceError(f"{path}: expected an integer, got {type(value).__name__}")
        return value
    if tp is float:
        if type(value) not in (int, float):
            raise InstanceError(f"{path}: expected a number, got {type(value).__name__}")
        return float(value)
    if tp is str:
        if type(value) is not str:
            raise InstanceError(f"{path}: expected a string, got {type(value).__name__}")
        return value
    raise InstanceError(f"{path}: unsupported schema type {tp!r}")


def _encode(obj: Any) -> Any:
    if is_dataclass(obj):
        # a field added AFTER the schema was frozen carries a `None` default and is OMITTED when unset, so the
        # digest of every instance written before it existed is unchanged (`proof`, GUEST S12)
        return {f.name: _encode(getattr(obj, f.name)) for f in fields(obj) if not (f.default is None and getattr(obj, f.name) is None)}
    if isinstance(obj, (list, tuple)):
        return [_encode(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    return obj


def _hex64(value: str, path: str, problems: list[str]) -> None:
    if len(value) != 64 or not set(value) <= _HEX64:
        problems.append(f"{path}: expected a 64-hex sha256 digest, got {value[:20]!r}")


def _fraction(value: str, path: str, problems: list[str]) -> Fraction | None:
    try:
        f = Fraction(value)
    except (ValueError, ZeroDivisionError):
        problems.append(f"{path}: not a rational {value!r}")
        return None
    if not 0 <= f <= 1:
        problems.append(f"{path}: rate {value} outside [0, 1]")
    return f


# -- the schema ----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Quantity:
    """A number with its evidence label and source (brief rule: every number carries MEASURED/MODELED/ANALYTICAL)."""

    value: int | float | str
    label: str
    source: str
    note: str | None = None


@dataclass(frozen=True)
class ProgramIdentity:
    """The registered program the instance is about.  ``Serve_v1`` (vLLM inference): ``case`` names the registry config, ``LP``/``STEPS``
    the shape.  [E6 2026-09-11] Any other registered ``verity.ir`` composite (e.g. E8's ``DemoCore_v1{N,R}``): ``family`` = the composite id
    ``Name_vN`` as registered in ``verity.ir.defs.REGISTRY``, ``params`` = its static bindings (``bind(defn, **params).id == function``),
    ``module`` = the importable module that registers it, ``LP``/``STEPS`` = null, ``case`` = a short label, ``config_sha12`` = the first
    12 hex of sha256(canonical params JSON).  ``function`` is always the bound ``SpecializedDefinition.id``."""

    family: str
    config_sha12: str
    case: str
    LP: int | None
    STEPS: int | None
    function: str
    params: dict[str, Any] | None = None  # generic programs: the static bindings (omitted for Serve_v1)
    module: str | None = None  # generic programs: module whose import registers ``family`` (omitted for Serve_v1)
    # [E7 D57 2026-09-13] the DECLARED attention kernel geometry of a `Serve_v4` identity: {"kernel": "flash_fwd_splitkv_kernel", "kBlockM": 64,
    # "kBlockN": 128, "num_splits": 1, ...} (the shape b1.fa2_kernel_geometry produces) -- threaded to the program as the static BN (`b1.serve_program`); absent for `Serve_v1`, whose
    # geometry is the legacy kBlockN 64 implicit in its body (every pre-D54 instance)
    attention: dict[str, Any] | None = None

    @property
    def is_serve(self) -> bool:
        # [E7 D54 2026-09-13] the serving programs pinned by (case, LP, STEPS): Serve_v1 (legacy geometry) and Serve_v4 (declared geometry)
        return self.family in SERVE_FAMILIES


@dataclass(frozen=True)
class ProgramBuilderPin:
    """[E6 2026-09-11, INTEGRATOR D1] How the VERIFIER rebuilds a generic registered ``verity.ir`` program in its own process, with no vLLM
    assumption: ``getattr(import_module(module), symbol)(**params)`` returns the ``verity.ir.Program`` (or a ``(Program, digest)`` tuple);
    the caller recomputes ``program_digest(encode_program(program))`` and refuses (fail-closed) unless it equals ``program_digest`` ==
    ``program.digest``.  E3's ``locator.build_pinned_program`` honours this pin; :func:`build_pinned_program` here is the reference."""

    module: str
    symbol: str
    program_digest: str
    params: dict[str, Any] | None = None
    source: str | None = None


PROGRAM_KINDS = ("serve", "registered")


@dataclass(frozen=True)
class ProgramSection:
    digest: str
    identity: ProgramIdentity
    input_inclusion: str
    gates: Quantity
    input_gates: Quantity
    source: str
    kind: str | None = None  # [D1] "registered" = a generic registered verity.ir program (driver skips vLLM-only steps BY NAME); "serve"/None = Serve_v1
    builder: ProgramBuilderPin | None = None  # [D1] required when kind == "registered"
    attention_provenance: dict[str, Any] | None = None  # [E6 D57] {"label", "value", "source", "rule"}: the MEASUREMENT identity.attention's value was pinned from (E10 §A); required with it


@dataclass(frozen=True)
class VocabularySection:
    digest: str
    schema: str
    profile_name: str
    profile_sha256: str
    source: str


@dataclass(frozen=True)
class ValidationRef:
    """The validation verdict artifact the partition is := to (digest-pinned to program and plan)."""

    artifact: str
    program_digest: str
    plan_digest: str
    status: str
    source: str
    partition_digest: str | None = None  # the same pair as the instance (draft 2): required when the instance carries partition_digest


@dataclass(frozen=True)
class FirstBoundary:
    """COST-D objection 1 (2026-09-09 17:10 UTC): WHICH values the first (pre-``r_q``) commitment binds.  Paper §5.2 fixes the first
    boundary by the RU partition -- the outgoing boundary runs of every CU -- so a two-stage Bound over a partition assumes exactly that
    value set was committed before the replay draw.  ``rule`` names the set; ``committed_in`` the chronology event that committed it."""

    rule: str  # "outgoing boundary runs of every CU" | "designated outputs only"
    rows: Quantity | None  # rows (or leaves) of the first boundary per program instance
    bytes: Quantity | None
    committed_in: str  # chronology event name
    source: str


@dataclass(frozen=True)
class WeightInputCheck:
    """THEORY-F 2026-09-09 16:52 UTC (exfiltration review §F4): how the weight ``Input`` class is checked, per plan.
    V1 = weight class at inner rate 1 INSIDE the two-stage sampler (not unconditional); V2 = an unconditional per-run obligation outside
    the sampler (claimed weight values == the registered commitment, authenticated); V3 = weights read as prescribed constants (a different
    circuit); none.  ``bound_excludes_weight_family`` is None until SEARCH-B states how the fold treats the family; a Bound cannot attach
    under V2 while it is None."""

    variant: str  # V1 | V2 | V3 | none
    obligation: str | None  # V2: the exact check
    commitment: str | None  # V2: what it is checked against
    per_run: bool | None  # V2: per run (per program instance) or reused
    reuse_rule: str | None  # V2 reused: the rule that justifies reuse
    stage: str | None  # COST-D stage name of the obligation: verify.weights_unconditional (agreement 3 draft 1a); prover side = fixed.input_registration
    bound_excludes_weight_family: bool | None
    source: str
    # draft 2 (COST-D objection 6 / lead): the record check that ESTABLISHES the variant (path#field), or None = claimed, unverified.
    # A pre-challenge instance has none yet; a historical instance claiming V2 without one is "V2 claimed, unverified".
    verified_by: str | None = None


@dataclass(frozen=True)
class PartitionSection:
    plan_name: str
    plan_descriptor_digest: str
    plan_digest_kind: str  # SUMMARY-C's convention (program-summary-format.md §2): plan-json/v1 | plan-rule/v1 (veritor.plans.summary.plan_digest)
    vu_level: str
    cu_strategy: str
    validation: ValidationRef
    vu_count: Quantity
    ru_count: Quantity
    count_basis: str
    vu_count_bucketed_upper: Quantity | None
    address_map: str | None
    source: str
    # program instances (requests) sharing this partition in the record: the RECORD population is instances x (vu_count, ru_count),
    # VU/RU indices tiled per instance (global VU = r * vu_count + i; global RU = r * ru_count + j).  1 = one program instance.
    instances: int
    weight_input_check: WeightInputCheck
    # draft 2 (reviewer objections 2026-09-09):
    # SUMMARY-C 1: the partition ITSELF, representation-independent (veritor.plans.summary.partition_digest, partition-intervals/v1);
    #   plan_descriptor_digest stays the constructor/representation pin (its kind in plan_digest_kind).
    partition_digest: str
    partition_digest_kind: str
    # SUMMARY-C 3 (optional): Summary.digest (veritor/plans/program-summary/v1) -- names the fold's inputs (Summary.appendix_a()).
    summary_digest: str | None
    # COST-D 1: which value set the first commitment binds, and in which event.
    first_boundary: FirstBoundary
    # COST-D 4: acquisition/replay of a selected RU is per RU or forces its whole request ("request" | "ru").
    acquisition_granularity: str


@dataclass(frozen=True)
class DesignatedOutput:
    name: str
    multiplicity: int
    domain_bits_declared: int
    domain_bits_vocabulary_implied: Quantity
    wire_bits: int
    enforcement: str
    source: str


@dataclass(frozen=True)
class OutputsSection:
    designated: list[DesignatedOutput]
    W_out_bits: Quantity
    scope: str
    source: str


@dataclass(frozen=True)
class SamplerLaw:
    kind: str
    q: str | None
    s: str | None
    k: int | None
    N: int | None
    coverage_size: int | None
    delta_note: str | None
    source: str


@dataclass(frozen=True)
class ChronologyEvent:
    order: int
    kind: str
    name: str
    digest: str | None
    utc: str | None
    derived_from: list[str]
    source: str


@dataclass(frozen=True)
class Chronology:
    seed_derivation: str
    seed_commitment: str | None
    events: list[ChronologyEvent]
    challenges_drawn: list[str]
    rerolls: int
    source: str
    # REC draft 3: which recorder wrote chronology.json -- "security-sample/v1" (e2e.security_sample, re-derived by its verify) or
    # "adapter-transcript/v1" (the vLLM adapter's selective protocol: T_0 statement with the committed randomness source, r_q, interiors,
    # r_s; re-derived by verity_vllm.bundle.rederive_transcript / selective.rederive).  Absent == security-sample/v1.
    chronology_kind: str | None = None
    # [A3 contract 2026-09-10, R7] who holds the seed between r_q and r_s: "verifier_process_memory" (e2e.partition_sample stage1/stage2
    # in one verifier process; the seed is written only into the seed_reveal event) or "record_dir_file" (legacy: seed.hex in the record
    # directory from stage 1 on -- readable by a co-located prover; labelled legacy).  Absent == legacy record_dir_file.
    seed_custody: str | None = None


SEED_CUSTODY = ("verifier_process_memory", "record_dir_file")
B_EX_UNITS = ("replay_units",)
BOUND_SCOPES = ("epoch",)


@dataclass(frozen=True)
class EpochSection:
    """[A3 contract 2026-09-10, R16] The audit is one explicitly declared *epoch*: its identifier, the exact requests / RUs it covers,
    its exemption budget (an RU count) and the scope of the error bound.  Both commitment stages are global within the epoch:
    ``r_q`` is derived only after every RU in ``ru_indices`` has its step-1 values committed (one boundary tree per request in
    ``request_ids``), ``r_s`` only after every RU of J has its step-3 values committed or is in F.  ``B_ex`` here is THE budget the
    verifier enforces at step 4 (``|F| <= B_ex``) and the allowance the Bound is computed with; never per request."""

    epoch_id: str
    request_ids: list[str]  # the exact served request ids covered; ``ru_indices`` tile them (RU r of instance i = i * ru_count + r)
    ru_indices: list[int]
    B_ex: int
    B_ex_unit: str  # "replay_units"
    B_ex_provenance: str  # who set the budget and why (declared before r_q; the record never changes it)
    delta: str  # the epoch's delta (Fraction as str); the integrity bound of this epoch, not of a request
    bound_scope: str  # "epoch"
    source: str
    # [A3, integrator 19:50Z / A10 WR-5, WR-6 -- 2026-09-10 20:xxZ] optional pins added after the schema was frozen (omitted when unset:
    # every earlier instance digests unchanged).  A9's runner sets all three before the first commitment.
    family: str | None = None  # the interface family of the fixed circuit's relations (e.g. "veritor.tc-ampere-bf16@2"); verifier_v3 refuses a proof object of another family
    approved_elf_sha256: list[str] | None = None  # the approved guest set for this epoch (sha256 of the ELF); verifier_v3 refuses a host guest outside it
    inputs: DeclaredInputs | None = None  # the declared x per request (WR-5): step 2 compares the committed prompt / seed / sampling-parameter leaves with it


@dataclass(frozen=True)
class DeclaredInput:
    """One request's declared input ``x`` as the auditor fixes it before the first commitment: the prompt (by digest of its token ids),
    the seed and the sampling parameters (by digest of their canonical JSON).  ``None`` for seed / sampling params = not pinned."""

    prompt_sha256: str  # declared_input_digest(prompt_token_ids)
    prompt_len: int
    seed: int | None = None
    sampling_params_sha256: str | None = None


@dataclass(frozen=True)
class PromptLayoutPin:
    """The prompt collection's layout: ``count`` root Input gates of ``width`` bits committed under boundary ``field`` at offset t."""

    field: str  # "prompt_token"
    count: int  # == program.identity.LP
    width: int  # 32 (int32 token-id leaves)


@dataclass(frozen=True)
class InputLayoutPin:
    """[A2 r1-input-layout.md §4, A3 R1] ``epoch.inputs.layout``: the digest of C's root Input-gate layout (``veritor.interface.input_layout``:
    ``layout_digest(prompt_layout(LP))``), pinned beside the registered weights root so step 2 checks the committed prompt leaves gate by
    gate (count == LP, width, scalar, no committed position C has no gate for)."""

    kind: str  # INPUT_LAYOUT_KIND
    prompt: PromptLayoutPin
    digest: str  # 64-hex, recomputed at validation
    weights_layout_digest_ref: str | None = None  # "instance.weights.layout_digest" (the weights root Input gate keeps its leaf-table digest)


@dataclass(frozen=True)
class DeclaredInputs:
    kind: str  # DECLARED_INPUTS_KIND
    per_request: dict[str, DeclaredInput]  # request id -> declared input; must cover exactly epoch.request_ids
    source: str
    layout: InputLayoutPin | None = None  # [R1] the input layout pin (count == LP); None = not pinned (labelled, like epoch.family)


DECLARED_INPUTS_KIND = "a3/declared-inputs/v1"
INPUT_LAYOUT_KIND = "veritor/input-layout/v1"


def input_layout_section(LP: int, *, width: int = 32) -> dict[str, Any]:
    """The ``epoch.inputs.layout`` pin for a ``prompt: Array(LP, I<width>)`` collection (A2's ``veritor.interface.input_layout``)."""

    from veritor.interface.input_layout import layout_digest, prompt_layout

    return {"kind": INPUT_LAYOUT_KIND, "prompt": {"field": "prompt_token", "count": int(LP), "width": int(width)},
            "digest": layout_digest(prompt_layout(int(LP), width=int(width))), "weights_layout_digest_ref": "instance.weights.layout_digest"}


def declared_input_digest(token_ids: list[int]) -> str:
    """sha256 of the canonical JSON list of the prompt token ids (the declared ``x`` of one request)."""

    return hashlib.sha256(json.dumps([int(t) for t in token_ids], separators=(",", ":")).encode()).hexdigest()


def sampling_params_digest(params: dict[str, Any]) -> str:
    """sha256 of the canonical JSON of the sampling parameters as the serving lane records them (``production.jsonl#sampling_params``)."""

    return hashlib.sha256(json.dumps(params, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def declared_inputs_section(rows: list[dict[str, Any]], *, source: str, pin_sampling_params: bool = True, LP: int | None = None) -> dict[str, Any]:
    """The ``epoch.inputs`` section from the auditor's request rows (``request_id``, ``prompt_token_ids``, optional ``seed`` /
    ``sampling_params``) -- built BEFORE the first commitment, from the auditor's own submission, never from the prover's record.
    ``LP`` (the program's prompt length): every prompt must have exactly LP tokens (C has one Input gate per position) and the
    ``layout`` pin (``input_layout_section``) is added."""

    per = {}
    for r in rows:
        ids = list(r["prompt_token_ids"])
        if not ids:
            raise InstanceError(f"request {r.get('request_id')}: empty prompt cannot be declared")
        if LP is not None and len(ids) != int(LP):
            raise InstanceError(f"request {r.get('request_id')}: prompt has {len(ids)} tokens, the program C(LP={LP}) has {LP} prompt Input gates")
        sp = r.get("sampling_params")
        per[str(r["request_id"])] = {"prompt_sha256": declared_input_digest(ids), "prompt_len": len(ids), "seed": (int(r["seed"]) if r.get("seed") is not None else None),
                                     "sampling_params_sha256": (sampling_params_digest(sp) if (pin_sampling_params and isinstance(sp, dict)) else None)}
    return {"kind": DECLARED_INPUTS_KIND, "per_request": per, "source": source, **({"layout": input_layout_section(int(LP))} if LP is not None else {})}


@dataclass(frozen=True)
class AuthorizedCommitment:
    """S5 (GUIDANCE §57.10): the weight object the AUDITOR authorizes, fixed pre-challenge from the auditor's own pins -- never the prover's manifest."""

    checkpoint_repo: str
    checkpoint_revision: str
    safetensors_sha256: dict[str, str]
    config_sha256: str
    expected_root: str  # weights Merkle root recomputed by the auditor from the pinned checkpoint under the layout rule (weights_authorized.json)
    leaf_count: int
    ids_digest: str
    layout_rule: str  # capture format 2, rows_per_leaf 1: rank >= 2 tensors one leaf per row, vectors / scalars whole; derived buffers per the profile
    layout_digest: str  # sha256 of the leaf table (param -> canonical id, dtype, shape, leaf ids)
    artifact: str  # weights_authorized.json (in the record root; digest-pinned in the manifest before READY_PHASE0)
    artifact_sha256: str
    computed_by: str
    source: str


@dataclass(frozen=True)
class WeightsSection:
    """S5 "unconditional weight binding" as an explicit, priced variant beside the sampled-input baseline."""

    variant: str  # "unconditional weight binding"
    claim: str
    authorized_commitment: AuthorizedCommitment
    leaf_map: str  # (ii) how program weight identities map to tree positions
    no_other_path: str  # (iii) why substituted values cannot enter by another path
    covers_ref: str  # the exact leaf-id set the check binds (weights_authorized.json#covers), consumed by the fold only with verified coverage (S19)
    price: dict[str, Any] | None  # MEASURED after the audit (verifier seconds, bytes) -- None pre-challenge
    check_result: dict[str, Any] | None  # filled after the audit: root comparison outcome; None pre-challenge
    source: str


@dataclass(frozen=True)
class RealizedSample:
    """Realized sizes, recorded SEPARATELY from configured rates; never substituted for them."""

    J_size: Quantity | None
    Q_size: Quantity | None
    coverage_checked: Quantity | None
    in_program: Quantity | None
    out_of_program: Quantity | None
    source: str


@dataclass(frozen=True)
class SamplerSection:
    law: SamplerLaw
    population_ref: str
    chronology: Chronology
    realized: RealizedSample | None
    source: str


@dataclass(frozen=True)
class Obligation:
    name: str
    role: str
    counts_as_security_sample: bool
    source: str
    stage: str | None  # COST-D stage name (cost-result-format.md §1) this obligation's work is charged to
    # COST-D objection 2: the sub-steps "proved" consists of, for the sampled-unit obligation (witness = openings + statement build,
    # checker = transparent pre-check, argument = the proof); None for obligations that are not proof-supported.
    proof_steps: list[str] | None = None


@dataclass(frozen=True)
class ObligationsSection:
    mandatory: list[Obligation]
    diagnostic: list[Obligation]
    # the protocol-work stage names this instance defines (COST-D agreement 3 §7 / R12: a record may charge a mandatory stage only if
    # the instance names it; `frontier.result_schema.stages_not_in_instance(point, set(instance.obligations.stages))`)
    stages: list[str]
    source: str


@dataclass(frozen=True)
class SecuritySection:
    delta: str
    sigma_bits: Quantity
    B_ex: int
    kappa: str | None
    bound_algorithm: str
    bound_attached: bool
    bound_value_bits: Quantity | None
    bound_not_attached_reason: str | None
    fixed_size_analysis_ref: str | None
    source: str
    # [A3 contract 2026-09-10, R8] B_ex is an RU count (exemptions are selected RUs F ⊆ J, |F| <= B_ex, one budget per epoch) with its
    # provenance.  Absent == legacy instance whose B_ex was written as 0 and whose certificate plugged "selected exempt VUs" (a VU
    # count realized after r_s) -- not an exemption in the contract's sense; verifier_v3 refuses such an instance.
    B_ex_unit: str | None = None
    B_ex_provenance: str | None = None


@dataclass(frozen=True)
class Discrepancy:
    field_path: str
    finding: str
    severity: str
    resolution: str
    source: str


@dataclass(frozen=True)
class ProofSection:
    """GUEST (S12 opt-in): how the proved statement is encoded.  ``statement_format`` is the statement wire format the
    guest is handed (2 = ``veritor/proofs/statement/v2``, the historical record; 3 = ranged positions, the same lowering);
    ``h_L`` in the proof's public values is the digest of the bytes in THAT format.  Absent section == format 2."""

    statement_format: int
    source: str


STATEMENT_FORMATS = (2, 3)

# [E6 2026-09-11] the interior layout the interior tree is built by (brief interface 5: E2's `layout_digest()`); pinned before the run.
INTERIOR_LAYOUT_FRAMINGS = ("chunks-4096",)
INTERIOR_LAYOUT_WORD_ORDERS = ("gate",)
INTERIOR_LAYOUT_PLACEHOLDER = "PLACEHOLDER"  # `status` prefix while E2's digest is not yet in hand; the verifier refuses to accept interior openings against it


@dataclass(frozen=True)
class InteriorLayoutPin:
    """``commitments.interior_layout``: the fixed rule by which interior words are framed into leaves and ordered (E2 owns the rule and its
    ``layout_digest()``; E6 pins it here).  ``digest`` is E2's ``layout_digest()`` (64 hex) or null while the pin is a named placeholder
    (``status`` starts with ``PLACEHOLDER``).  A run whose pin is a placeholder has NO interior tree the verifier will open against
    (``e2e.protocol_instance.interior_layout_status``)."""

    framing: str  # INTERIOR_LAYOUT_FRAMINGS
    word_order: str  # INTERIOR_LAYOUT_WORD_ORDERS
    tree: str  # tree name pattern, e.g. "interior/<request>/<step>" (one interior tree per (request, step) under the interior root)
    digest: str | None  # E2's layout_digest(); null = placeholder
    status: str  # "pinned" | "PLACEHOLDER: <what is awaited, from whom>"
    source: str
    rule: str | None = None  # E2's rule name, e.g. "interior-layout/chunks-4096/v2" (D13); None for pins written before E2 named it


@dataclass(frozen=True)
class CoverageTablePin:
    """``commitments.coverage_table``: the kind table the population identity was frozen against (which kinds are provable and by which
    checker/producer).  A row flip changes the table sha and therefore the ``a9/legacy-minus-inputs/v2`` partition digest (A3 §5)."""

    path: str  # repo-relative path of the table document (e2e.coverage_v1.kind_table build --write)
    sha256: str  # sha256 of that document
    kinds: int  # rows in the table
    source: str


@dataclass(frozen=True)
class TableSourcePin:
    """The fixture file a committed MUFU table was built from (E2 ``e2e.interior.tables.table_record``: ``source``)."""

    fixture: str
    file: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class TablePin:
    """[D23 2026-09-12] ``commitments.tables[<name>]``: the root of one committed MUFU table tree under E2's ``table-layout/v1`` -- the exact
    record ``e2e.interior.tables.table_record(name, tree, source)`` yields; ``e2e.interior.tables.check_tables(pins)`` rebuilds and compares."""

    tree: str  # "table/<name>"
    root: str
    leaf_count: int
    ids_digest: str
    rows: int
    row_width: int  # 64: (index << 32) | T[index]
    rows_per_leaf: int
    rule: str  # "table-layout/v1"
    layout_digest: str
    origin: str  # "table"
    domain: str
    index_rule: str
    readers: list[str]
    source: TableSourcePin


@dataclass(frozen=True)
class LiteralTreePin:
    """[D36 2026-09-12] one public literal tree ``literal/w=<w>`` of ``prescribed-literal/v1``: one leaf whose words are the program's ``Const<w>[0x…]_v1``
    values in sorted-id order; the verifier re-derives ``root`` / ``leaf_count`` / ``ids_digest`` from the pinned program (E9's ``steps.LiteralTable``)."""

    root: str
    leaf_count: int
    ids_digest: str


@dataclass(frozen=True)
class LiteralsPin:
    """[D36 2026-09-12, E7 draft / E9 co-signed] ``commitments.literals``: the ``prescribed-literal/v1`` table of the pinned program -- ``literals`` = every
    distinct ``Const<w>[0x…]_v1`` primitive id of the program (sorted), ``trees`` = one :class:`LiteralTreePin` per width.  A literal at an operand position is an
    ordinary incoming committed word opened from these trees; the verifier derives the same block from the program and refuses ``LITERAL_PIN_MISMATCH`` on any
    difference, ``LITERAL_UNPINNED`` when a selected VU reads a literal and the block is absent.  The block is E9's ``LiteralTable.from_program(program).pin()``
    verbatim (:func:`literals_pin` imports it; nothing is re-implemented here)."""

    rule: str
    literals: list[str]
    trees: dict[str, LiteralTreePin]


@dataclass(frozen=True)
class CommitmentsSection:
    """[E6 2026-09-11] pins that fix, before the first commitment, HOW the prover's trees are laid out (interior) and WHICH population
    identity the coverage table produced.  Optional so every instance written before it decodes with its digest unchanged.
    [D23] ``tables``: the committed MUFU table roots (instance-wide, no request prefix) the table-reading checkers open rows against.
    [D36] ``literals``: the public literal trees of ``prescribed-literal/v1`` (:class:`LiteralsPin`)."""

    interior_layout: InteriorLayoutPin | None
    coverage_table: CoverageTablePin | None
    source: str
    tables: dict[str, TablePin] | None = None
    literals: LiteralsPin | None = None


@dataclass(frozen=True)
class ProtocolInstance:
    schema: str
    instance_id: str
    status: str
    program: ProgramSection
    vocabulary: VocabularySection
    partition: PartitionSection
    outputs: OutputsSection
    sampler: SamplerSection
    obligations: ObligationsSection
    security: SecuritySection
    discrepancies: list[Discrepancy]
    notes: list[str]
    weights: WeightsSection | None = None  # REC draft 3: S5 unconditional weight binding (optional; absent in draft-2 instances)
    # optional (absent == null == format 2) so every existing instance still decodes; PROTO-A / REC to confirm placement
    proof: ProofSection | None = None
    # [A3 contract 2026-09-10, R16] the declared epoch (optional so every existing instance still decodes with its digest unchanged;
    # a record without it is LEGACY and refused by e2e.verifier_v3)
    epoch: EpochSection | None = None
    # [E6 2026-09-11] interior layout + coverage table pins (optional; absent in every instance written before today)
    commitments: CommitmentsSection | None = None

    @property
    def statement_format(self) -> int:
        return self.proof.statement_format if self.proof is not None else 2

    @property
    def exemption_budget(self) -> int:
        """The RU-count exemption budget the verifier enforces at step 4: the epoch's B_ex when declared, else security.B_ex."""

        return self.epoch.B_ex if self.epoch is not None else self.security.B_ex

    @property
    def is_contract_instance(self) -> bool:
        """True when the instance declares what the contract requires of an audit: an epoch and B_ex as an RU count."""

        return self.epoch is not None and self.security.B_ex_unit == "replay_units"


# -- loading / semantic validation -------------------------------------------------------------------------------------------


def load_instance(doc: dict[str, Any] | str | Path) -> ProtocolInstance:
    """Strictly decode a document (dict, JSON text, or path).  Raises :class:`InstanceError`."""

    if isinstance(doc, (str, Path)):
        p = Path(doc)
        text = p.read_text() if p.exists() else str(doc)
        try:
            doc = json.loads(text)
        except json.JSONDecodeError as e:
            raise InstanceError(f"not JSON: {e}") from e
    if not isinstance(doc, dict):
        raise InstanceError("instance must be a JSON object")
    if doc.get("schema") != SCHEMA:
        raise InstanceError(f"schema must be {SCHEMA!r}, got {doc.get('schema')!r}")
    inst = _decode(ProtocolInstance, doc, "instance")
    problems = validate_instance(inst)
    if problems:
        raise InstanceError("; ".join(problems))
    return inst


def _quantities(obj: Any, path: str, out: list[tuple[str, Quantity]]) -> None:
    if isinstance(obj, Quantity):
        out.append((path, obj))
    elif is_dataclass(obj):
        for f in fields(obj):
            _quantities(getattr(obj, f.name), f"{path}.{f.name}", out)
    elif isinstance(obj, (list, tuple)):
        for i, x in enumerate(obj):
            _quantities(x, f"{path}[{i}]", out)


def validate_instance(inst: ProtocolInstance) -> list[str]:
    """Semantic rules over a decoded instance.  Returns the list of violations (empty = valid)."""

    p: list[str] = []
    if inst.status not in STATUSES:
        p.append(f"instance.status must be one of {STATUSES}, got {inst.status!r}")
    if inst.proof is not None:
        if inst.proof.statement_format not in STATEMENT_FORMATS:
            p.append(f"instance.proof.statement_format must be one of {STATEMENT_FORMATS}, got {inst.proof.statement_format!r}")
        if not inst.proof.source.strip():
            p.append("instance.proof.source is empty (name where the format was chosen: run_composed --statement-format)")
    # every Quantity: label + non-empty source
    qs: list[tuple[str, Quantity]] = []
    _quantities(inst, "instance", qs)
    for path, q in qs:
        if q.label not in LABELS:
            p.append(f"{path}.label must be one of {LABELS}, got {q.label!r}")
        if not q.source.strip():
            p.append(f"{path}.source is empty (every number names its source)")
    # program
    _hex64(inst.program.digest, "instance.program.digest", p)
    if inst.program.input_inclusion not in INPUT_INCLUSION:
        p.append(f"instance.program.input_inclusion must be one of {INPUT_INCLUSION}")
    ident = inst.program.identity
    if len(ident.config_sha12) != 12:
        p.append("instance.program.identity.config_sha12 must be 12 hex characters")
    generic = not ident.is_serve
    if not generic:
        if ident.LP is None or ident.STEPS is None:
            p.append("instance.program.identity.LP / STEPS are required for Serve_v1")
        # [E7 D57] the geometry is part of the identity: Serve_v4 carries it (and `function` names its BN); Serve_v1 must not
        if ident.family == "Serve_v4":
            if not isinstance(ident.attention, dict):
                p.append("instance.program.identity.attention (kernel, kBlockM, kBlockN, num_splits) is required for Serve_v4")
            else:
                bn = ident.attention.get("kBlockN")
                if not isinstance(bn, int) or isinstance(bn, bool) or bn < 16 or bn % 16:
                    p.append(f"instance.program.identity.attention.kBlockN must be a positive multiple of 16, got {bn!r}")
                elif ident.function and f"BN={bn}" not in ident.function:
                    p.append(f"instance.program.identity.function does not name attention.kBlockN as BN={bn}")
                if ident.attention.get("kernel") not in ("flash_fwd_splitkv_kernel", "fa2-splitkv-align", "fa2-splitkv"):
                    p.append(f"instance.program.identity.attention.kernel {ident.attention.get('kernel')!r} is not the FA2 split-KV kernel the registry models")
        elif ident.attention is not None:
            p.append("instance.program.identity.attention must be absent for Serve_v1 (its geometry, kBlockN 64, is the body's; pin Serve_v4 to declare another)")
        # [E6 D57] the geometry VALUE carries its provenance: the pin records both (the value is E10's measurement, not a derivation the verifier trusts)
        prov = inst.program.attention_provenance
        if ident.attention is not None:
            if not isinstance(prov, dict) or not str(prov.get("source", "")).strip():
                p.append("instance.program.attention_provenance.source must name the measurement identity.attention was pinned from (D57)")
            elif any(prov.get("value", {}).get(k) != ident.attention.get(k) for k in ("kernel", "kBlockM", "kBlockN", "num_splits")):
                p.append("instance.program.attention_provenance.value != identity.attention (kernel, kBlockM, kBlockN, num_splits)")
        elif prov is not None:
            p.append("instance.program.attention_provenance without identity.attention (Serve_v1 declares no geometry)")
        if ident.function and f"LP={ident.LP}" not in ident.function:
            p.append("instance.program.identity.function does not name identity.LP")
        if ident.function and f"STEPS={ident.STEPS}" not in ident.function and "D_MAX=" not in ident.function:
            p.append("instance.program.identity.function does not name identity.STEPS (or D_MAX)")
    else:
        # [E6 2026-09-11] a generic registered composite: family = registry id, params = the static bindings that appear in `function`
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*_v[0-9]+", ident.family):
            p.append(f"instance.program.identity.family {ident.family!r} is not a registered composite id (Name_vN)")
        if not ident.params:
            p.append("instance.program.identity.params is required for a non-Serve_v1 program (the static bindings of the registered composite)")
        else:
            if not ident.function.startswith(ident.family + "{"):
                p.append(f"instance.program.identity.function must be {ident.family}{{...}} (bind(<family>, **params).id)")
            for k, v in ident.params.items():
                if f"{k}={v}" not in ident.function:
                    p.append(f"instance.program.identity.function does not name params.{k}={v}")
            if ident.config_sha12 != generic_config_sha12(ident.params):
                p.append("instance.program.identity.config_sha12 != sha256(canonical params)[:12] (e2e.protocol_instance.generic_config_sha12)")
        if not ident.module:
            p.append("instance.program.identity.module is required for a non-Serve_v1 program (the module whose import registers the family)")
        if ident.LP is not None or ident.STEPS is not None:
            p.append("instance.program.identity.LP / STEPS must be null for a non-Serve_v1 program (shape lives in params)")
    # [D1] program.kind / program.builder: the verifier's own rebuild of a generic registered program
    if inst.program.kind is not None and inst.program.kind not in PROGRAM_KINDS:
        p.append(f"instance.program.kind must be one of {PROGRAM_KINDS} (or absent = serve)")
    registered = inst.program.kind == "registered"
    if generic and not registered:
        p.append("instance.program.kind must be 'registered' for a non-Serve_v1 program (INTEGRATOR D1: the driver skips vLLM-only steps by name)")
    if registered and inst.program.builder is None:
        p.append("instance.program.builder {module, symbol, params, program_digest} is required when program.kind == 'registered' (INTEGRATOR D1)")
    if inst.program.builder is not None and not registered:
        p.append("instance.program.builder is pinned but program.kind != 'registered'")
    if (b := inst.program.builder) is not None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*", b.module):
            p.append(f"instance.program.builder.module {b.module!r} is not a dotted module name")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", b.symbol):
            p.append(f"instance.program.builder.symbol {b.symbol!r} is not an identifier")
        _hex64(b.program_digest, "instance.program.builder.program_digest", p)
        if b.program_digest != inst.program.digest:
            p.append("instance.program.builder.program_digest != instance.program.digest (the builder must rebuild exactly the pinned program)")
        if generic and ident.params and b.params is not None and dict(b.params) != dict(ident.params):
            p.append("instance.program.builder.params != instance.program.identity.params (one set of static bindings)")
    # vocabulary
    _hex64(inst.vocabulary.digest, "instance.vocabulary.digest", p)
    _hex64(inst.vocabulary.profile_sha256, "instance.vocabulary.profile_sha256", p)
    # partition := validated plan
    part = inst.partition
    _hex64(part.plan_descriptor_digest, "instance.partition.plan_descriptor_digest", p)
    _hex64(part.validation.program_digest, "instance.partition.validation.program_digest", p)
    _hex64(part.validation.plan_digest, "instance.partition.validation.plan_digest", p)
    if part.validation.program_digest != inst.program.digest:
        p.append("instance.partition.validation.program_digest != instance.program.digest (the partition is not a plan of THIS program)")
    if part.validation.plan_digest != part.plan_descriptor_digest:
        p.append("instance.partition.validation.plan_digest != instance.partition.plan_descriptor_digest")
    if part.plan_digest_kind not in PLAN_DIGEST_KINDS:
        p.append(f"instance.partition.plan_digest_kind must be one of {PLAN_DIGEST_KINDS} (SUMMARY-C agreement 2 §2)")
    if part.validation.status != "valid":
        # [L1 RECORD 2026-09-10] a partially validated partition (COMPACT-P1 whole B0: coverage-limited verdict) is admitted ONLY when the
        # instance cites the verdict verbatim as its validation scope (a `scope` discrepancy on this field); the population stays the
        # partition's VU identities (DIRECTIVE §1.5.5) and the certificate repeats the scope.  Anything else is still refused.
        cited = any(d.field_path == "partition.validation.status" and d.severity == "scope" and repr(part.validation.status) in d.finding for d in inst.discrepancies)
        if not cited:
            p.append(f"instance.partition.validation.status is {part.validation.status!r}, not 'valid' (an unvalidated partition is not a population; "
                     "a coverage-limited verdict must be cited as a 'scope' discrepancy on partition.validation.status)")
    if part.count_basis not in COUNT_BASES:
        p.append(f"instance.partition.count_basis must be one of {COUNT_BASES}")
    elif part.count_basis != "validated_plan":
        p.append(f"instance.partition.count_basis is {part.count_basis!r}: the sampling population must be derived from the validated plan "
                 "(physical leaves / analytical census are not populations; a bucketed upper count is an upper bound, not the population)")
    for name in ("vu_count", "ru_count"):
        q = getattr(part, name)
        if type(q.value) is not int or q.value < 0:
            p.append(f"instance.partition.{name}.value must be a nonnegative integer")
        if q.label != "MEASURED":
            p.append(f"instance.partition.{name}.label must be MEASURED (counted on the validated plan), got {q.label}")
    if type(part.instances) is not int or part.instances < 1:
        p.append("instance.partition.instances must be a positive integer (program instances = requests sharing the partition)")
    # draft 2: the partition itself (SUMMARY-C 1/3)
    _hex64(part.partition_digest, "instance.partition.partition_digest", p)
    if part.partition_digest_kind != PARTITION_DIGEST_KIND:
        # [E6 2026-09-11] a generic registered program declares its own partition rule by name (e.g. E8's demo-core/stage-ru-gate-vu/v1);
        # the format loader the CLI names re-derives that digest (e2e.partition_sample.register_instance_format).  Serve_v1 instances keep
        # the strict rule (record_v5's a9/legacy-minus-inputs/v1|v2 identities are admitted by record_v5.instance_v5's enumerated allowlist).
        if generic and part.partition_digest_kind.strip() and part.partition_digest_kind in part.source:
            pass
        else:
            p.append(f"instance.partition.partition_digest_kind must be {PARTITION_DIGEST_KIND!r} (veritor.plans.summary.partition_digest)")
    if part.validation.partition_digest is None:
        p.append("instance.partition.validation.partition_digest is required (the artifact carries the same pair as the instance)")
    elif part.validation.partition_digest != part.partition_digest:
        p.append("instance.partition.validation.partition_digest != instance.partition.partition_digest")
    if part.summary_digest is not None:
        _hex64(part.summary_digest, "instance.partition.summary_digest", p)
    # draft 2: first boundary (COST-D 1) and acquisition granularity (COST-D 4)
    fb = part.first_boundary
    if fb.rule not in FIRST_BOUNDARY_RULES:
        p.append(f"instance.partition.first_boundary.rule must be one of {FIRST_BOUNDARY_RULES}")
    event_names = {e.name for e in inst.sampler.chronology.events}
    if not fb.committed_in.strip():
        p.append("instance.partition.first_boundary.committed_in must name the chronology event that commits the first boundary")
    elif event_names and fb.committed_in not in event_names:
        p.append(f"instance.partition.first_boundary.committed_in {fb.committed_in!r} is not a chronology event")
    elif event_names:
        by_name = {e.name: e for e in inst.sampler.chronology.events}
        rq = [e for e in inst.sampler.chronology.events if e.kind == "challenge"]
        if rq and by_name[fb.committed_in].order > rq[0].order:
            p.append(f"instance.partition.first_boundary.committed_in {fb.committed_in!r} (order {by_name[fb.committed_in].order}) is AFTER the "
                     f"first challenge {rq[0].name!r} (order {rq[0].order}): the first boundary was not committed before r_q")
    if part.ru_count.value and type(part.ru_count.value) is int and part.ru_count.value > 1 and fb.rule != "outgoing boundary runs of every CU" \
            and inst.security.bound_attached:
        p.append("instance.security.bound_attached with first_boundary.rule != 'outgoing boundary runs of every CU' over a multi-RU partition: "
                 "the two-stage Bound assumes every CU's outgoing boundary was committed before r_q (paper §5.2)")
    if part.acquisition_granularity not in ACQUISITION:
        p.append(f"instance.partition.acquisition_granularity must be one of {ACQUISITION}")
    if part.vu_count_bucketed_upper is not None:
        u = part.vu_count_bucketed_upper
        if type(u.value) is not int or (type(part.vu_count.value) is int and u.value < part.vu_count.value):
            p.append("instance.partition.vu_count_bucketed_upper must be an integer >= vu_count (it is an UPPER bound)")
        if "upper" not in (u.note or "").lower():
            p.append("instance.partition.vu_count_bucketed_upper.note must say it is an upper bound")
    # outputs
    total = 0
    for i, o in enumerate(inst.outputs.designated):
        if o.multiplicity <= 0 or o.domain_bits_declared <= 0 or o.wire_bits <= 0:
            p.append(f"instance.outputs.designated[{i}]: multiplicity/domain/wire bits must be positive")
        if o.domain_bits_declared > o.wire_bits:
            p.append(f"instance.outputs.designated[{i}]: declared domain {o.domain_bits_declared} > wire width {o.wire_bits}")
        total += o.multiplicity * o.domain_bits_declared
    if not inst.outputs.designated:
        p.append("instance.outputs.designated is empty (no designated outputs)")
    if inst.outputs.W_out_bits.value != total:
        p.append(f"instance.outputs.W_out_bits.value {inst.outputs.W_out_bits.value} != sum(multiplicity x declared bits) = {total}")
    if inst.outputs.W_out_bits.label != "ANALYTICAL":
        p.append("instance.outputs.W_out_bits.label must be ANALYTICAL (a sum over the declaration)")
    # sampler law
    law = inst.sampler.law
    if law.kind not in LAWS:
        p.append(f"instance.sampler.law.kind must be one of {LAWS}, got {law.kind!r}")
    if law.kind == "bernoulli":
        if law.q is None or law.s is None:
            p.append("bernoulli law needs q and s")
        else:
            _fraction(law.q, "instance.sampler.law.q", p)
            _fraction(law.s, "instance.sampler.law.s", p)
        if law.k is not None:
            p.append("bernoulli law must not carry a fixed size k (a realized count is not a configured probability)")
    if law.kind == "fixed_size":
        if law.k is None or law.N is None or law.k <= 0 or law.N < law.k:
            p.append("fixed_size law needs 0 < k <= N")
        if law.s is None:
            p.append("fixed_size law needs the inner rate s")
        else:
            _fraction(law.s, "instance.sampler.law.s", p)
        if law.q is not None and law.k is not None and law.N is not None and Fraction(law.q) == Fraction(law.k, law.N) \
                and "k/N" not in (law.delta_note or ""):
            p.append("fixed_size law states q = k/N: a realized fraction is not a Bernoulli rate; drop q or mark delta_note 'q recorded as k/N, not a Bernoulli rate'")
    if inst.sampler.population_ref != "partition":
        p.append("instance.sampler.population_ref must be 'partition' (the population is the validated VU partition, nothing else)")
    # chronology
    ch = inst.sampler.chronology
    if ch.rerolls != 0:
        p.append(f"instance.sampler.chronology.rerolls = {ch.rerolls}: every challenge drawn is final; rerolls are forbidden")
    orders = [e.order for e in ch.events]
    if orders != sorted(orders) or len(set(orders)) != len(orders):
        p.append("instance.sampler.chronology.events must be in strictly increasing order")
    seen: dict[str, ChronologyEvent] = {}
    n_challenges = 0
    for e in ch.events:
        if e.kind not in EVENT_KINDS:
            p.append(f"chronology event {e.name!r}: kind must be one of {EVENT_KINDS}")
        if e.digest is not None:
            _hex64(e.digest, f"chronology event {e.name!r}.digest", p)
        if e.kind == "challenge":
            n_challenges += 1
            if not e.derived_from:
                p.append(f"challenge {e.name!r} names no commitment/message it is derived from")
            for d in e.derived_from:
                if d not in seen:
                    p.append(f"challenge {e.name!r} is derived from {d!r}, which does not precede it in the chronology")
                elif seen[d].kind == "challenge":
                    pass  # a later challenge may depend on an earlier one (two-stage protocol)
            if e.digest is not None and e.digest not in ch.challenges_drawn:
                p.append(f"challenge {e.name!r} digest is not listed in challenges_drawn (every challenge ever drawn is recorded)")
        seen[e.name] = e
    if n_challenges == 0 and inst.status == "historical-reconstruction":
        p.append("a historical reconstruction must record the challenges that were drawn")
    if len(ch.challenges_drawn) < n_challenges:
        p.append("chronology.challenges_drawn lists fewer digests than challenge events")
    for d in ch.challenges_drawn:
        _hex64(d, "chronology.challenges_drawn[]", p)
    # realized never as configured
    r = inst.sampler.realized
    if r is not None:
        for name in ("J_size", "Q_size", "coverage_checked", "in_program", "out_of_program"):
            q = getattr(r, name)
            if q is not None and (type(q.value) is not int or q.label != "MEASURED"):
                p.append(f"instance.sampler.realized.{name} must be an integer MEASURED count")
    # obligations
    mand = {o.name for o in inst.obligations.mandatory}
    diag = {o.name for o in inst.obligations.diagnostic}
    if mand & diag:
        p.append(f"obligations listed as both mandatory and diagnostic: {sorted(mand & diag)}")
    for o in inst.obligations.diagnostic:
        if o.counts_as_security_sample:
            p.append(f"diagnostic obligation {o.name!r} claims to count as a security sample (forbidden)")
        if o.role != "diagnostic":
            p.append(f"diagnostic obligation {o.name!r} has role {o.role!r}")
    for o in inst.obligations.mandatory:
        if o.role not in ("unconditional", "sampled"):
            p.append(f"mandatory obligation {o.name!r} role must be 'unconditional' or 'sampled'")
    lowered = " ".join(sorted(mand)).lower()
    if "output" not in lowered or ("typing" not in lowered and "domain" not in lowered):
        p.append("mandatory obligations must include the unconditional output typing/domain check")
    if "agreement" not in lowered and "linkage" not in lowered:
        p.append("mandatory obligations must include the unconditional output agreement/linkage check")
    # stages (COST-D agreement 3, R12): the instance's protocol-work stage names; closed vocabulary when result_schema is importable
    stages = inst.obligations.stages
    if len(set(stages)) != len(stages):
        p.append("instance.obligations.stages has duplicates")
    known = _known_stages()
    for st in stages:
        if st.startswith("diagnostic."):
            p.append(f"instance.obligations.stages names {st!r}: diagnostic stages are never protocol work")
        elif known is not None and st not in known:
            p.append(f"instance.obligations.stages names {st!r}, not in frontier.result_schema.STAGES (closed vocabulary; publish a mapping instead)")
        elif known is None and not _STAGE_RE.match(st):
            p.append(f"instance.obligations.stages entry {st!r} is not a '<layer>.<name>' stage name")
    for o in inst.obligations.mandatory:
        if o.stage is not None and o.stage not in stages:
            p.append(f"mandatory obligation {o.name!r} charges stage {o.stage!r} which instance.obligations.stages does not name")
        if o.proof_steps is not None:
            bad = [x for x in o.proof_steps if x not in PROOF_STEPS]
            if bad or len(set(o.proof_steps)) != len(o.proof_steps):
                p.append(f"mandatory obligation {o.name!r}.proof_steps must be distinct members of {PROOF_STEPS}, got {o.proof_steps}")
    sampled = [o for o in inst.obligations.mandatory if o.role == "sampled"]
    if sampled and not any(o.proof_steps for o in sampled):
        p.append("the sampled-unit obligation must name its proof_steps [witness, checker, argument] (COST-D objection 2: 'proved' is three "
                 "separately priced steps; name the ones the record performs)")
    for o in inst.obligations.diagnostic:
        if o.stage is not None and not o.stage.startswith("diagnostic."):
            p.append(f"diagnostic obligation {o.name!r} charges protocol stage {o.stage!r} (diagnostic work is 'diagnostic.*')")
    # weight Input class (THEORY-F 16:52 UTC): V1 / V2 / V3 / none, and what the Bound does with the weight family
    w = inst.partition.weight_input_check
    if w.variant not in WEIGHT_VARIANTS:
        p.append(f"instance.partition.weight_input_check.variant must be one of {WEIGHT_VARIANTS}")
    if w.variant == "V2":
        if not w.obligation or not w.commitment:
            p.append("weight_input_check V2 needs the exact obligation and the commitment it is checked against")
        if w.per_run is None:
            p.append("weight_input_check V2 needs per_run (true) or per_run false + reuse_rule")
        elif w.per_run is False and not w.reuse_rule:
            p.append("weight_input_check V2 reused across runs needs reuse_rule")
        if w.stage != WEIGHT_CHECK_STAGE:
            p.append(f"weight_input_check V2 stage must be {WEIGHT_CHECK_STAGE!r} (COST-D agreement 3 draft 1a), got {w.stage!r}")
        if w.stage not in stages:
            p.append("instance.obligations.stages must list weight_input_check.stage (COST-D prices it as required recurring work)")
        if inst.status == "historical-reconstruction" and not w.verified_by:
            p.append("historical instance claims weight_input_check V2 without verified_by (the record check that establishes it): mark it "
                     "'claimed, unverified' in source and set bound_excludes_weight_family = null")
    # security
    sec = inst.security
    if sec.bound_attached and w.variant == "V2" and w.bound_excludes_weight_family is None:
        p.append("bound_attached under weight_input_check V2 requires bound_excludes_weight_family to be stated (true/false with derivation; THEORY-F IMPL-CHECK)")
    if sec.bound_attached and w.variant in ("V1", "none") and w.bound_excludes_weight_family:
        p.append("bound_excludes_weight_family cannot be true under V1/none (the weight family is a sampled fault source; §6.3)")
    d = _fraction(sec.delta, "instance.security.delta", p)
    if d is not None and d == 0:
        p.append("instance.security.delta must be positive")
    if sec.B_ex < 0:
        p.append("instance.security.B_ex must be >= 0")
    # [A3 contract 2026-09-10] R8 (B_ex as an RU count with provenance), R16 (epoch declaration), R7 (seed custody)
    if sec.B_ex_unit is not None and sec.B_ex_unit not in B_EX_UNITS:
        p.append(f"instance.security.B_ex_unit must be one of {B_EX_UNITS}, got {sec.B_ex_unit!r} (exemptions are selected RUs, never VUs or families)")
    if sec.B_ex_unit is not None and not (sec.B_ex_provenance or "").strip():
        p.append("instance.security.B_ex_provenance is required when B_ex_unit is declared (who set the budget, before r_q)")
    if sec.B_ex > inst.partition.ru_count.value * inst.partition.instances:
        p.append(f"instance.security.B_ex = {sec.B_ex} exceeds N_R = {inst.partition.ru_count.value * inst.partition.instances} (the Bound's endpoint; a budget "
                 "at or above N_R exempts every RU)")
    sc = inst.sampler.chronology.seed_custody
    if sc is not None and sc not in SEED_CUSTODY:
        p.append(f"sampler.chronology.seed_custody must be one of {SEED_CUSTODY}, got {sc!r}")
    ep = inst.epoch
    if ep is not None:
        n_r = inst.partition.ru_count.value * inst.partition.instances
        if not ep.epoch_id.strip():
            p.append("instance.epoch.epoch_id must not be empty")
        if ep.B_ex_unit not in B_EX_UNITS:
            p.append(f"instance.epoch.B_ex_unit must be one of {B_EX_UNITS}, got {ep.B_ex_unit!r}")
        if ep.B_ex < 0:
            p.append("instance.epoch.B_ex must be >= 0")
        if ep.B_ex != sec.B_ex:
            p.append(f"instance.epoch.B_ex = {ep.B_ex} != instance.security.B_ex = {sec.B_ex}: one budget per epoch, stated once")
        if sec.B_ex_unit != "replay_units":
            p.append("an epoch is declared but instance.security.B_ex_unit is not 'replay_units' (legacy 'selected exempt VUs' is not an exemption budget)")
        if not ep.B_ex_provenance.strip():
            p.append("instance.epoch.B_ex_provenance must not be empty")
        if ep.bound_scope not in BOUND_SCOPES:
            p.append(f"instance.epoch.bound_scope must be one of {BOUND_SCOPES}, got {ep.bound_scope!r} (never per request)")
        if ep.delta != sec.delta:
            p.append(f"instance.epoch.delta = {ep.delta!r} != instance.security.delta = {sec.delta!r}: the bound is the epoch's, stated once")
        if len(ep.request_ids) != inst.partition.instances:
            p.append(f"instance.epoch.request_ids names {len(ep.request_ids)} requests but partition.instances = {inst.partition.instances}")
        if len(set(ep.request_ids)) != len(ep.request_ids):
            p.append("instance.epoch.request_ids has duplicates")
        if any(not str(r).strip() for r in ep.request_ids):
            p.append("instance.epoch.request_ids has an empty id")
        if sorted(ep.ru_indices) != list(range(n_r)):
            p.append(f"instance.epoch.ru_indices must be exactly [0, {n_r}) (every RU of every covered request; the epoch is the whole audit)")
        if ep.family is not None and not ep.family.strip():
            p.append("instance.epoch.family must not be empty when given")
        if ep.approved_elf_sha256 is not None:
            if not ep.approved_elf_sha256:
                p.append("instance.epoch.approved_elf_sha256 must name at least one approved guest when given")
            for h in ep.approved_elf_sha256:
                _hex64(h, "instance.epoch.approved_elf_sha256[]", p)
        if ep.inputs is not None:
            if ep.inputs.kind != DECLARED_INPUTS_KIND:
                p.append(f"instance.epoch.inputs.kind must be {DECLARED_INPUTS_KIND!r}, got {ep.inputs.kind!r}")
            if sorted(ep.inputs.per_request) != sorted(ep.request_ids):
                p.append("instance.epoch.inputs.per_request must cover exactly epoch.request_ids (the declared x of every request in the epoch)")
            LP = inst.program.identity.LP
            for rid, di in ep.inputs.per_request.items():
                _hex64(di.prompt_sha256, f"instance.epoch.inputs.per_request[{rid!r}].prompt_sha256", p)
                if di.prompt_len <= 0:
                    p.append(f"instance.epoch.inputs.per_request[{rid!r}].prompt_len must be > 0")
                # [R1, integrator 21:00Z] C(LP) has exactly LP prompt Input gates: a declared prompt of another length has no gate / leaves gates open
                if isinstance(LP, int) and di.prompt_len != LP:
                    p.append(f"instance.epoch.inputs.per_request[{rid!r}].prompt_len {di.prompt_len} != program.identity.LP {LP} (C has one Input gate per prompt position)")
            lay = ep.inputs.layout
            if lay is not None:
                if lay.kind != INPUT_LAYOUT_KIND:
                    p.append(f"instance.epoch.inputs.layout.kind must be {INPUT_LAYOUT_KIND!r}")
                if lay.prompt.field != "prompt_token" or lay.prompt.width != 32:
                    p.append("instance.epoch.inputs.layout.prompt must be {field: prompt_token, width: 32} (int32 token-id leaves)")
                if isinstance(LP, int) and lay.prompt.count != LP:
                    p.append(f"instance.epoch.inputs.layout.prompt.count {lay.prompt.count} != program.identity.LP {LP}")
                _hex64(lay.digest, "instance.epoch.inputs.layout.digest", p)
                try:
                    from veritor.interface.input_layout import layout_digest, prompt_layout

                    want = layout_digest(prompt_layout(int(lay.prompt.count), width=int(lay.prompt.width), field=lay.prompt.field))
                    if want != lay.digest:
                        p.append(f"instance.epoch.inputs.layout.digest {lay.digest[:16]} != layout_digest(prompt_layout({lay.prompt.count})) {want[:16]}")
                except Exception as e:  # noqa: BLE001
                    p.append(f"instance.epoch.inputs.layout: digest not recomputable: {type(e).__name__}: {e}")
                if di.sampling_params_sha256 is not None:
                    _hex64(di.sampling_params_sha256, f"instance.epoch.inputs.per_request[{rid!r}].sampling_params_sha256", p)
    if sec.bound_attached:
        if sec.bound_value_bits is None:
            p.append("bound_attached is true but bound_value_bits is missing")
        if law.kind == "fixed_size" and not sec.fixed_size_analysis_ref:
            p.append("a Bound is attached under a fixed_size law without fixed_size_analysis_ref (protocol-instance.md appendix A3): "
                     "the Bound is stated for Bernoulli(q); fixed-size needs its own analysis before attachment")
    else:
        if not sec.bound_not_attached_reason:
            p.append("bound_attached is false: bound_not_attached_reason is required")
    # REC draft 3: chronology kind and the S5 weights section
    ck = inst.sampler.chronology.chronology_kind
    if ck is not None and ck not in CHRONOLOGY_KINDS:
        p.append(f"sampler.chronology.chronology_kind must be one of {sorted(CHRONOLOGY_KINDS)}, got {ck!r}")
    w = inst.weights
    if w is not None:
        if w.variant != WEIGHTS_VARIANT:
            p.append(f"weights.variant must be {WEIGHTS_VARIANT!r}, got {w.variant!r}")
        ac = w.authorized_commitment
        for name, val in (("expected_root", ac.expected_root), ("ids_digest", ac.ids_digest), ("layout_digest", ac.layout_digest), ("artifact_sha256", ac.artifact_sha256),
                          ("config_sha256", ac.config_sha256)):
            _hex64(val, f"weights.authorized_commitment.{name}", p)
        for fname, sha in ac.safetensors_sha256.items():
            _hex64(sha, f"weights.authorized_commitment.safetensors_sha256[{fname!r}]", p)
        if not ac.safetensors_sha256:
            p.append("weights.authorized_commitment.safetensors_sha256 is empty: the authorized checkpoint must be pinned by file digests")
        if ac.leaf_count <= 0:
            p.append("weights.authorized_commitment.leaf_count must be positive")
        for name, val in (("leaf_map", w.leaf_map), ("no_other_path", w.no_other_path), ("covers_ref", w.covers_ref), ("claim", w.claim), ("computed_by", ac.computed_by)):
            if not str(val).strip():
                p.append(f"weights.{name} must not be empty (S5 requirements (i)-(iii) are stated, not implied)")
        if inst.status == "pre-challenge" and (w.price is not None or w.check_result is not None):
            p.append("weights.price / check_result must be null in a pre-challenge instance (they are MEASURED after the audit)")
        if inst.partition.weight_input_check.variant != "V2":
            p.append(f"weights section present but partition.weight_input_check.variant is {inst.partition.weight_input_check.variant!r}: unconditional weight "
                     "binding is the V2 check (root recomputed from the authorized object)")
    # [E6 2026-09-11] commitments: interior layout pin (E2's rule) and the coverage table the population identity was frozen against
    cm = inst.commitments
    if cm is not None:
        il = cm.interior_layout
        if il is not None:
            if il.framing not in INTERIOR_LAYOUT_FRAMINGS:
                p.append(f"instance.commitments.interior_layout.framing must be one of {INTERIOR_LAYOUT_FRAMINGS}, got {il.framing!r}")
            if il.word_order not in INTERIOR_LAYOUT_WORD_ORDERS:
                p.append(f"instance.commitments.interior_layout.word_order must be one of {INTERIOR_LAYOUT_WORD_ORDERS}, got {il.word_order!r}")
            if not il.tree.startswith("interior/"):
                p.append("instance.commitments.interior_layout.tree must name the interior tree pattern (interior/...)")
            if il.digest is None:
                if not il.status.startswith(INTERIOR_LAYOUT_PLACEHOLDER):
                    p.append(f"instance.commitments.interior_layout.digest is null: status must start with {INTERIOR_LAYOUT_PLACEHOLDER!r} and name what is awaited")
            else:
                _hex64(il.digest, "instance.commitments.interior_layout.digest", p)
                if il.status != "pinned":
                    p.append("instance.commitments.interior_layout.status must be 'pinned' when digest is present")
            if not il.source.strip():
                p.append("instance.commitments.interior_layout.source is empty")
        ct = cm.coverage_table
        if ct is not None:
            _hex64(ct.sha256, "instance.commitments.coverage_table.sha256", p)
            if ct.kinds <= 0 or not ct.path.strip():
                p.append("instance.commitments.coverage_table must name the table path and a positive row count")
        lt = cm.literals
        if lt is not None:  # [D36] the verifier compares the whole block by equality; here: well-formed, one tree per literal width, hex roots
            if not lt.rule.strip() or not lt.literals:
                p.append("instance.commitments.literals must name its rule and at least one Const<w>[0x…]_v1 literal id")
            if list(lt.literals) != sorted(set(lt.literals)):
                p.append("instance.commitments.literals.literals must be sorted and distinct (the trees' word order is sorted-id order)")
            if not all(re.match(_CONST_ID, i) for i in lt.literals):
                p.append("instance.commitments.literals.literals must all be Const<w>[0x…]_v<n> primitive ids")
            widths = {m.group(1) for i in lt.literals if (m := re.match(_CONST_ID, i))}
            for t, tree in lt.trees.items():
                _hex64(tree.root, f"instance.commitments.literals.trees[{t}].root", p)
                _hex64(tree.ids_digest, f"instance.commitments.literals.trees[{t}].ids_digest", p)
                if tree.leaf_count != 1 or not t.startswith("literal/"):
                    p.append(f"instance.commitments.literals.trees[{t}]: one leaf per width under 'literal/w=<w>' (prescribed-literal/v1)")
            if {t.split("w=", 1)[-1] for t in lt.trees} != widths:
                p.append(f"instance.commitments.literals.trees widths {sorted(t.split('w=', 1)[-1] for t in lt.trees)} != the literal ids' widths {sorted(widths)}")
    return p


def interior_layout_status(inst: ProtocolInstance) -> tuple[str, str | None]:
    """``("pinned", digest)`` when E2's layout digest is pinned; ``("placeholder", None)`` when the pin is named but the digest is not yet in
    hand; ``("absent", None)`` when the instance carries no interior layout pin.  Anything but ``pinned`` means: no interior openings are
    verifiable against this instance (the interior tree's layout is not fixed)."""

    cm = inst.commitments
    if cm is None or cm.interior_layout is None:
        return "absent", None
    il = cm.interior_layout
    return ("pinned", il.digest) if il.digest is not None else ("placeholder", None)


def generic_config_sha12(params: dict[str, Any]) -> str:
    """``config_sha12`` of a generic registered program: sha256 of the canonical JSON of its static bindings, first 12 hex."""

    return hashlib.sha256(json.dumps(params, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()[:12]


def generic_program_identity(defn_id: str, params: dict[str, Any], *, module: str, case: str | None = None) -> dict[str, Any]:
    """The ``program.identity`` of a registered ``verity.ir`` composite bound at ``params`` (no vLLM assumption): ``family`` = the registry
    id, ``function`` = ``bind(REGISTRY[defn_id], **params).id`` (import ``module`` first so the composite is registered)."""

    import importlib

    from verity.ir import bind
    from verity.ir.defs import REGISTRY

    importlib.import_module(module)
    if defn_id not in REGISTRY:
        raise InstanceError(f"{defn_id!r} is not registered after importing {module!r}")
    sp = bind(REGISTRY[defn_id], **params)
    return {"family": defn_id, "config_sha12": generic_config_sha12(params), "case": case or "".join(f"{k}{v}" for k, v in sorted(params.items())),
            "LP": None, "STEPS": None, "function": sp.id, "params": dict(params), "module": module}


def program_builder_pin(*, module: str, symbol: str, params: dict[str, Any] | None, program_digest: str, source: str | None = None) -> dict[str, Any]:
    """[D1] The ``program.builder`` document (see :class:`ProgramBuilderPin`)."""

    return {"module": module, "symbol": symbol, "params": dict(params) if params is not None else None, "program_digest": program_digest,
            "source": source or f"INTEGRATOR D1 2026-09-11: the verifier rebuilds the program as {module}.{symbol}(**params) and checks its digest"}


def build_pinned_program(builder: ProgramBuilderPin | dict[str, Any], *, expected_digest: str | None = None):
    """[D1] Rebuild the pinned program in THIS process from its ``program.builder`` pin and check its digest (fail-closed): returns
    ``(Program, program_digest)``.  The callable may return a ``verity.ir.Program`` or a tuple whose first element is one (E8's
    ``demo_core.circuit.build(N, R) -> (Program, digest)``).  Any mismatch -- import, symbol, return type, digest -- is an ``InstanceError``;
    nothing is read from a prover file.  E3's ``locator.build_pinned_program`` calls this for ``program.kind == "registered"``."""

    import importlib

    from verity.ir import Program, encode_program, program_digest

    b = builder if isinstance(builder, ProgramBuilderPin) else _decode(ProgramBuilderPin, builder, "program.builder")
    try:
        mod = importlib.import_module(b.module)
    except Exception as e:  # noqa: BLE001 -- the refusal names the failure
        raise InstanceError(f"program.builder: import {b.module!r} failed: {type(e).__name__}: {e}") from e
    fn = getattr(mod, b.symbol, None)
    if not callable(fn):
        raise InstanceError(f"program.builder: {b.module}.{b.symbol} is not a callable")
    built = fn(**(b.params or {}))
    prog = built[0] if isinstance(built, tuple) else built
    if not isinstance(prog, Program):
        raise InstanceError(f"program.builder: {b.module}.{b.symbol}(**params) returned {type(prog).__name__}, not verity.ir.Program")
    digest = program_digest(encode_program(prog))
    want = expected_digest or b.program_digest
    if digest != want:
        raise InstanceError(f"program.builder: rebuilt program digest {digest} != pinned {want} ({b.module}.{b.symbol}(**{b.params}))")
    if b.program_digest != want:
        raise InstanceError(f"program.builder.program_digest {b.program_digest} != program.digest {want}")
    return prog, digest


def instance_digest(inst: ProtocolInstance | dict[str, Any]) -> str:
    doc = _encode(inst) if is_dataclass(inst) else inst
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def to_json(inst: ProtocolInstance) -> dict[str, Any]:
    return _encode(inst)


# -- cross-checks against a record and a validation artifact -----------------------------------------------------------------


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_validation_artifact(path: Path) -> dict[str, Any]:
    """A compact-validation artifact (``veritor/compact-validation/v1``): the verdict of ``validate_plan`` on the
    compact whole program, pinned to ``program_digest`` and ``plan_digest`` with the counted ``vus``/``cus``."""

    doc = json.loads(Path(path).read_text())
    required = ("schema", "program_digest", "plan_digest", "status", "vus", "cus", "plan_name")
    missing = [k for k in required if k not in doc]
    if missing:
        raise InstanceError(f"{path}: validation artifact missing {missing}")
    if doc["schema"] != "veritor/compact-validation/v1":
        raise InstanceError(f"{path}: validation artifact schema {doc['schema']!r} != 'veritor/compact-validation/v1'")
    return doc


def check_against_validation(inst: ProtocolInstance, art: dict[str, Any]) -> list[str]:
    p: list[str] = []
    part = inst.partition
    if art["program_digest"] != inst.program.digest:
        p.append(f"validation artifact program_digest {art['program_digest'][:12]} != instance program digest {inst.program.digest[:12]}")
    if art["plan_digest"] != part.plan_descriptor_digest:
        p.append(f"validation artifact plan_digest {art['plan_digest'][:12]} != instance plan descriptor digest {part.plan_descriptor_digest[:12]}")
    if art.get("partition_digest") is None:
        p.append("validation artifact carries no partition_digest (draft 2: the artifact pins the partition itself, not only its representation)")
    elif art["partition_digest"] != part.partition_digest:
        p.append(f"validation artifact partition_digest {art['partition_digest'][:12]} != instance partition_digest {part.partition_digest[:12]}")
    if art["status"] != "valid":
        cited = any(d.field_path == "partition.validation.status" and repr(art["status"]) in d.finding for d in inst.discrepancies)
        if not cited or part.validation.status != art["status"]:
            p.append(f"validation artifact status {art['status']!r} != 'valid' and not cited as the instance's validation scope")
    if art["vus"] != part.vu_count.value:
        p.append(f"validation artifact vus {art['vus']} != instance vu_count {part.vu_count.value} (population must be the validated plan's)")
    if art["cus"] != part.ru_count.value:
        p.append(f"validation artifact cus {art['cus']} != instance ru_count {part.ru_count.value}")
    if art.get("plan_name") != part.plan_name:
        p.append(f"validation artifact plan_name {art.get('plan_name')!r} != instance plan_name {part.plan_name!r}")
    return p


def check_against_record(inst: ProtocolInstance, record: Path, validation: Path | None = None) -> list[str]:
    """Cross-check the instance against a ``run_composed`` record directory (manifest, sampling policy, verify/)."""

    p: list[str] = []
    record = Path(record)
    manifest = json.loads((record / "manifest.json").read_text()) if (record / "manifest.json").exists() else None
    if manifest is None:
        return [f"{record}: manifest.json missing"]
    inp = manifest.get("inputs") or {}
    if inp.get("program_digest") != inst.program.digest:
        p.append(f"record program_digest {str(inp.get('program_digest'))[:12]} != instance {inst.program.digest[:12]}")
    prof = (inp.get("semantic_profile") or {}).get("sha256")
    if prof and prof != inst.vocabulary.profile_sha256:
        p.append(f"record semantic_profile sha {prof[:12]} != instance profile_sha256 {inst.vocabulary.profile_sha256[:12]}")
    # GUEST (S12): the statement format the instance names is the one the record proved (absent == 2 on both sides)
    rec_fmt = inp.get("statement_format", 2)
    if rec_fmt != inst.statement_format:
        p.append(f"record statement_format {rec_fmt} != instance proof.statement_format {inst.statement_format}")
    pol_file = record / "sampling_policy.json"
    if not pol_file.exists():
        p.append("record has no sampling_policy.json (pre-challenge policy not bound)")
    else:
        pol = json.loads(pol_file.read_text())
        if inp.get("sampling_policy_sha256") and _sha256_file(pol_file) != inp["sampling_policy_sha256"]:
            p.append("sampling_policy.json sha != manifest pin")
        law = inst.sampler.law
        if pol.get("selector") != law.kind:
            p.append(f"record selector {pol.get('selector')!r} != instance law {law.kind!r} (mismatched policy/law)")
        if pol.get("s_declared") and law.s and Fraction(pol["s_declared"]) != Fraction(law.s):
            p.append(f"record s_declared {pol['s_declared']} != instance s {law.s}")
        if law.kind == "fixed_size" and pol.get("k") is not None and pol.get("k") != law.k:
            p.append(f"record k {pol.get('k')} != instance k {law.k}")
        if law.kind == "bernoulli" and pol.get("q_declared") and law.q and Fraction(pol["q_declared"]) != Fraction(law.q):
            p.append(f"record q_declared {pol['q_declared']} != instance q {law.q}")
    sv_file = record / "verify" / "selective_verify.json"
    if sv_file.exists():
        sv = json.loads(sv_file.read_text())
        pop = (sv.get("population") or {}).get("vus")
        # the record's inner population covers the selected program instances (J's requests); each carries the plan's VUs
        n_sel = len(sv.get("selected") or []) or (sv.get("k") if isinstance(sv.get("k"), int) else None) or inst.partition.instances
        expected = n_sel * inst.partition.vu_count.value
        if pop is not None and pop != expected:
            p.append(f"record population.vus {pop} (physical committed leaves of {n_sel} selected instance(s)) != {n_sel} x instance vu_count "
                     f"{inst.partition.vu_count.value} = {expected} (validated plan): mismatched population")
        if sv.get("requests_total") is not None and sv["requests_total"] != inst.partition.instances:
            p.append(f"record requests_total {sv['requests_total']} != instance partition.instances {inst.partition.instances}")
        doms = sv.get("output_domains") or []
        per_request = {o.name: o for o in inst.outputs.designated}
        slot = per_request.get("token_slot")
        if slot is not None and doms:
            checked = {r.get("checked") for r in doms}
            if checked != {slot.multiplicity}:
                p.append(f"record output_domains checked per request {sorted(checked)} != instance token_slot multiplicity {slot.multiplicity}")
    desc = None
    for rel in ("adapter/selective/description.json", "adapter/commit-v2/description.json"):
        if (record / rel).exists():
            desc = json.loads((record / rel).read_text())
            break
    if desc is not None:
        declared = None
        for node in desc.get("nodes") or []:
            if node.get("op") == "sampler.select":
                declared = node.get("declared_domain")
        slot = next((o for o in inst.outputs.designated if o.name == "token_slot"), None)
        if slot is not None and declared is not None and declared != slot.domain_bits_declared:
            p.append(f"record description declared_domain {declared} != instance token_slot domain {slot.domain_bits_declared} (mismatched output domain)")
    if validation is not None:
        try:
            p += check_against_validation(inst, load_validation_artifact(validation))
        except InstanceError as e:
            p.append(str(e))
    else:
        p.append("no compact-validation artifact given: the partition's validation verdict is not in evidence (missing evidence)")
    return p


# -- building a pre-challenge instance from a validated population directory ------------------------------------------------

DEFAULT_MANDATORY = [
    {"name": "output typing / domain on every designated output of every request", "role": "unconditional", "counts_as_security_sample": False,
     "source": "check_output_domains (declared bits per slot)", "stage": "verify.output_checks"},
    {"name": "output agreement / linkage on every request (claimed y == committed designated outputs; first boundary)", "role": "unconditional",
     "counts_as_security_sample": False, "source": "Protocol 2 step 2; verify_composed output_linkage on ALL requests", "stage": "verify.output_checks"},
    {"name": "served-token correspondence (served_tokens.json pinned; transcript == committed outputs)", "role": "unconditional",
     "counts_as_security_sample": False, "source": "certificate.py pin('served_tokens.json')", "stage": "verify.output_checks"},
    {"name": "weight Input class: committed weights tree recomputed from the registered checkpoint == committed root (unconditional, V2)",
     "role": "unconditional", "counts_as_security_sample": False, "source": "selective_verify bindings.model (303/303 leaves, root_matches)", "stage": "fixed.input_registration"},
    {"name": "commitment bindings: description regeneration, boundary roots", "role": "unconditional", "counts_as_security_sample": False,
     "source": "selective_verify bindings.{architecture,output}", "stage": "verify.coverage"},
    {"name": "first-boundary commitment of every RU + designated outputs before r_q", "role": "unconditional", "counts_as_security_sample": False,
     "source": "Protocol 2 step 1; statement phase digest", "stage": "boundary.commit"},
    {"name": "message structure: exactly the prescribed retained interiors of J, B_ex = 0", "role": "unconditional", "counts_as_security_sample": False,
     "source": "leaf_predicate; first_stage", "stage": "acquire.commit"},
    {"name": "security-sampled unit checks: r_s Bernoulli(s) over the VUs of J's RUs, each selected VU re-executed / proved", "role": "sampled",
     "counts_as_security_sample": True, "source": "e2e.security_sample chronology.json events[r_s].selection; proofs/", "stage": "prove.argument",
     "proof_steps": ["witness", "checker", "argument"]},
    {"name": "scheduler step trace (batch composition per step) committed in phase 0", "role": "unconditional", "counts_as_security_sample": False,
     "source": "manifest phase0 step records; classification: MANDATORY as part of fixed.output_collection -- under batch-invariant kernels the batch "
               "composition is not a value of the computation, but the step trace is what binds each served token to its (request, step) slot of the "
               "program instance (served_tokens.json binding, GUIDANCE §50-§51); without it output linkage is unanchored",
     "stage": "fixed.output_collection"},
]
DEFAULT_DIAGNOSTIC = [
    {"name": "coverage sample (diagnostic re-execution beyond the security sample)", "role": "diagnostic", "counts_as_security_sample": False,
     "source": "sampling_policy coverage_size", "stage": "diagnostic.other"},
    {"name": "control full replay / batch-invariance comparison", "role": "diagnostic", "counts_as_security_sample": False, "source": "summary.json stages.preparation",
     "stage": "diagnostic.control_replay"},
    {"name": "verifier timing / witness accounting", "role": "diagnostic", "counts_as_security_sample": False, "source": "selective_verify timing, witness", "stage": "diagnostic.other"},
]
# the protocol-work stages a Protocol-2 record charges (COST-D cost-result-format.md §1; every mandatory stage of the two-stage protocol)
DEFAULT_STAGES = ["fixed.setup", "fixed.input_registration", "fixed.output_collection", "boundary.collect", "boundary.commit", "challenge.replay_draw",
                  "acquire.replay", "acquire.commit", "exempt.declare", "challenge.inner_draw", "prove.witness", "prove.checker", "prove.argument",
                  "prove.output_linkage", "verify.descriptor", "verify.bound", "verify.sampling", "verify.coverage", "verify.proofs", "verify.output_checks",
                  "verify.weights_unconditional"]
# today's implementation of the weight Input class (THEORY-F V1/V2/V3): V2 by full recomputation of the weights tree, reused across the
# record's requests because the weights are the same prescribed constants for every request of the same checkpoint
DEFAULT_WEIGHT_CHECK = {
    "variant": "V2",
    "obligation": "weights Merkle tree (capture format v2, one leaf per weight row) recomputed from the registered checkpoint files (sha-pinned in "
                  "manifests/checkpoints.json) == the committed weights root c_x; every sampled unit's weight inputs are then opened against c_x (InputEq obligations)",
    "commitment": "c_x = weights root (selective_verify.json#bindings.model.root_matches; manifest inputs.weights_root_v2)",
    "per_run": False,
    "reuse_rule": "one weights tree per checkpoint per record: the weight Input values are identical for every request (same checkpoint, same values.bin), "
                  "so root equality binds every weight leaf of every program instance; the check is repeated per record, not per request",
    "stage": "verify.weights_unconditional",
    "bound_excludes_weight_family": None,
    "verified_by": None,
    "source": "THEORY-F 2026-09-09 16:52 UTC; selective_verify.json#bindings.model (303/303 leaves, 270,079,072 bytes recomputed, 6.5 s MEASURED on B0); "
              "verifier side = verify.weights_unconditional (COST-D agreement 3 draft 1a), prover side = fixed.input_registration (the weights tree); whether "
              "the fold excludes the weight family is SEARCH-B's derivation -- None until stated, so no Bound attaches under V2 meanwhile",
}
DEFAULT_FIRST_BOUNDARY = {
    "rule": "outgoing boundary runs of every CU",
    "rows": None, "bytes": None,
    "committed_in": "boundary",
    "source": "paper §5.2 (the first boundary is fixed by the RU partition); e2e.security_sample: the boundary commitment event precedes r_q; "
              "rows/bytes filled from ru_map/value_map at capture (COST-D first-stage-boundary.md §3.1 prices boundary.collect from them)",
}


def pre_challenge_instance(*, instance_id: str, population_dir: Path, identity: dict[str, Any], vocabulary: dict[str, Any],
                           outputs: list[dict[str, Any]], q: str, s: str, delta: str, input_gates: int, instances: int = 1, coverage_size: int | None = None,
                           bound_algorithm: str = "veritor.analysis.bound.bound via veritor.frontier.bound_adapter.FOLD_OPTIONS (frozen fold)",
                           seed_derivation: str = "e2e.security_sample: seed = 32 verifier bytes; seed_commitment = H(domain, seed); r_q = bernoulli_subset(seed, "
                                                  "'q/replay-unit', H(PHASE, 'q', instance_digest, boundary_digest), N_R, q); r_s = bernoulli_subset(seed, "
                                                  "'s/verification-unit', H(PHASE, 's', instance_digest, r_q_digest, interior_digest), |VUs of J|, s)",
                           notes: list[str] | None = None, vocabulary_V: int | None = None, weight_input_check: dict[str, Any] | None = None,
                           stages: list[str] | None = None, summary_digest: str | None = None, first_boundary: dict[str, Any] | None = None,
                           acquisition_granularity: str = "ru", weights_authorized: Path | None = None, chronology_kind: str | None = None,
                           seed_commitment_note: str | None = None, epoch_id: str | None = None, request_ids: list[str] | None = None,
                           B_ex: int = 0, B_ex_provenance: str | None = None, seed_custody: str | None = None,
                           family: str | None = None, approved_elf_sha256: list[str] | None = None, declared_inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    """A ``status = pre-challenge`` Bernoulli instance over a population directory written by ``e2e.population`` (maps.json,
    population.json, compact_validation.json).  Counts come from the validation artifact; the chronology is empty (nothing drawn yet);
    ``realized`` is null.  ``outputs`` items: ``{name, multiplicity, domain_bits_declared, wire_bits, enforcement, source}``."""

    population_dir = Path(population_dir)
    art = load_validation_artifact(population_dir / "compact_validation.json")
    maps = json.loads((population_dir / "maps.json").read_text())
    # [L1 RECORD 2026-09-10] the population may be the compact population/v2 file (e2e.population_v2; the only population whole B0 has)
    pop_file = population_dir / "population.json"
    if not pop_file.exists() and (population_dir / "population.compact.json").exists():
        pop_file = population_dir / "population.compact.json"
    pop = json.loads(pop_file.read_text())
    if art["vus"] != pop["vu_count"] or art["cus"] != pop["ru_count"] or art["plan_digest"] != pop["plan_digest"] or art["program_digest"] != pop["program_digest"]:
        raise InstanceError(f"{population_dir}: {pop_file.name} disagrees with compact_validation.json")
    if pop.get("partition_digest") and pop["partition_digest"] != art.get("partition_digest"):
        raise InstanceError(f"{population_dir}: {pop_file.name} partition_digest disagrees with compact_validation.json")
    discrepancies: list[dict[str, Any]] = []
    if art["status"] != "valid":
        # [L1 RECORD] the validation SCOPE is cited verbatim (brief: "cite the coverage as the validation scope"); the population stays the
        # partition's identities (DIRECTIVE §1.5.5 "SMALL uses P1 as validated"); never bridged, never relabelled
        discrepancies.append({"field_path": "partition.validation.status", "severity": "scope",
                              "finding": f"validation artifact status {art['status']!r}: {art.get('scope') or art.get('codes')}",
                              "resolution": "cited verbatim in the certificate as the validation scope; the population is the partition's VU identities unchanged",
                              "source": str(population_dir / "compact_validation.json")})
    for o in outputs:
        implied = math.log2(vocabulary_V) if vocabulary_V else None
        o.setdefault("domain_bits_vocabulary_implied", {"value": implied if implied is not None else "unknown", "label": "ANALYTICAL",
                                                        "source": f"log2(V), V = {vocabulary_V}" if vocabulary_V else "vocabulary size not supplied"})
    w_out = sum(o["multiplicity"] * o["domain_bits_declared"] for o in outputs)
    sigma = -math.log2(Fraction(delta))
    return {
        "schema": SCHEMA, "instance_id": instance_id, "status": "pre-challenge",
        "program": {"digest": art["program_digest"], "identity": identity, "input_inclusion": "gates_include_input_gates",
                    "gates": {"value": maps["program"]["gates"], "label": "MEASURED", "source": f"{population_dir}/maps.json#program.gates (built program, circuit.size)"},
                    "input_gates": {"value": input_gates, "label": "MEASURED", "source": "build(program).input_gates (e2e.population index.json#program.input_gates)"},
                    "source": f"{population_dir}/compact_validation.json#program_digest (program_from_verity_ir digest of the bridged program)"},
        "vocabulary": vocabulary,
        "partition": {"plan_name": art["plan_name"], "plan_descriptor_digest": art["plan_digest"], "plan_digest_kind": art.get("plan_digest_kind", "plan-json/v1"),
                      "vu_level": art.get("vu_level", ""), "cu_strategy": art.get("cu_strategy", ""),
                      "validation": {"artifact": str(population_dir / "compact_validation.json"), "program_digest": art["program_digest"], "plan_digest": art["plan_digest"],
                                     "partition_digest": art["partition_digest"],
                                     "status": art["status"], "source": art.get("verdict_source", "compact_validation.json")},
                      "vu_count": {"value": art["vus"], "label": "MEASURED", "source": f"{population_dir}/compact_validation.json#vus == len(plan.verification_units)"},
                      "ru_count": {"value": art["cus"], "label": "MEASURED", "source": f"{population_dir}/compact_validation.json#cus == len(plan.commitment_units)"},
                      "count_basis": "validated_plan", "vu_count_bucketed_upper": None, "address_map": str(population_dir / "value_map.tsv"),
                      "instances": instances, "weight_input_check": dict(weight_input_check or DEFAULT_WEIGHT_CHECK),
                      "partition_digest": art["partition_digest"], "partition_digest_kind": art.get("partition_digest_kind", PARTITION_DIGEST_KIND),
                      "summary_digest": summary_digest,
                      "first_boundary": dict(first_boundary or DEFAULT_FIRST_BOUNDARY), "acquisition_granularity": acquisition_granularity,
                      "source": "e2e.population (validated plan; RU membership in population.json; record population = instances x plan counts)"},
        "outputs": {"designated": outputs, "W_out_bits": {"value": w_out, "label": "ANALYTICAL", "source": "sum over designated outputs of multiplicity x declared bits"},
                    "scope": "per program instance (one request)", "source": "declaration fixed before the first commitment"},
        "sampler": {"law": {"kind": "bernoulli", "q": str(Fraction(q)), "s": str(Fraction(s)), "k": None, "N": None, "coverage_size": coverage_size, "delta_note": None,
                            "source": "declared here, before any commitment"},
                    "population_ref": "partition",
                    "chronology": {"seed_derivation": seed_derivation, "seed_commitment": None, "events": [], "challenges_drawn": [], "rerolls": 0,
                                   "source": seed_commitment_note or "empty before the audit; e2e.security_sample appends every event to chronology.json and the record copies it here",
                                   **({"chronology_kind": chronology_kind} if chronology_kind else {}),
                                   **({"seed_custody": seed_custody} if seed_custody else {})},
                    "realized": None, "source": "Protocol 2 with B_ex = 0 over the validated partition (paper §5.2, appendix B)"},
        "obligations": {"mandatory": [dict(o) for o in DEFAULT_MANDATORY], "diagnostic": [dict(o) for o in DEFAULT_DIAGNOSTIC],
                        "stages": list(stages or DEFAULT_STAGES),
                        "source": "brief A.3; diagnostic checks never count as security samples; stages = COST-D cost-result-format.md §1 protocol-work names (R12)"},
        "security": {"delta": str(Fraction(delta)), "sigma_bits": {"value": sigma, "label": "ANALYTICAL", "source": "-log2(delta)"}, "B_ex": int(B_ex), "kappa": None,
                     "bound_algorithm": bound_algorithm, "bound_attached": False, "bound_value_bits": None,
                     "bound_not_attached_reason": "pre-challenge: the Bound is computed after the audit from this instance's law and partition; none attached yet",
                     "fixed_size_analysis_ref": None, "source": "declared here",
                     # [A3 contract 2026-09-10, R8] the budget is an RU count with provenance whenever an epoch is declared
                     **({"B_ex_unit": "replay_units", "B_ex_provenance": B_ex_provenance or "declared by the operator before r_q"} if request_ids is not None else {})},
        "discrepancies": discrepancies,
        "notes": notes or ["pre-challenge instance written by e2e.protocol_instance.pre_challenge_instance; its digest is pinned in the record's manifest before READY_PHASE0"],
        **({"weights": weights_section_from_authorized(weights_authorized)} if weights_authorized else {}),
        # [A3 contract 2026-09-10, R16] the epoch: exactly the requests and RUs the audit covers, its budget and the scope of its bound
        **({"epoch": epoch_section(epoch_id or instance_id, list(request_ids), art["cus"] * instances, B_ex=B_ex, B_ex_provenance=B_ex_provenance,
                                   delta=str(Fraction(delta)), family=family, approved_elf_sha256=approved_elf_sha256, inputs=declared_inputs)} if request_ids is not None else {}),
    }


_CONST_ID = r"^Const(\d+)\[0x[0-9a-fA-F]+\]_v\d+$"


def literals_pin(literal_ids: Sequence[str]) -> dict[str, Any]:
    """[D36] ``commitments.literals`` for the pinned program's literal ids (``Const<w>[0x…]_v1``, the population's Const kinds): E9's
    ``verifier_v3.steps.LiteralTable(ids).pin()`` -- the SAME class the verifier derives its table with (``from_program`` walks the gates to the same id set;
    ``from_pin`` reads this block back), imported, never copied.  Raises InstanceError when that class is not on this checkout (the pin is then not
    written and the instance says so by name), when an id is not a Const primitive id, or when the pin does not round-trip through ``from_pin``."""

    ids = sorted({str(i) for i in literal_ids})
    bad = [i for i in ids if not re.match(_CONST_ID, i)]
    if bad or not ids:
        raise InstanceError(f"literal ids must be Const<w>[0x…]_v<n> primitive ids, at least one: bad {bad}")
    try:
        from e2e.verifier_v3.steps import LiteralTable
    except ImportError as e:
        raise InstanceError(f"verifier_v3.steps.LiteralTable (E9, prescribed-literal/v1) not on this checkout: {e}") from e
    table = LiteralTable(ids)
    pin = table.pin()
    if not isinstance(pin, dict) or list(pin.get("literals") or []) != ids or not isinstance(pin.get("trees"), dict) or not pin.get("rule"):
        raise InstanceError(f"LiteralTable.pin() is not a {{rule, literals, trees}} block over the given ids: {list(pin) if isinstance(pin, dict) else type(pin).__name__}")
    again = LiteralTable.from_pin(pin).pin()
    if again != pin:
        raise InstanceError("literal pin does not round-trip through LiteralTable.from_pin")
    return pin


def table_pins(names: tuple[str, ...] | None = None) -> dict[str, dict[str, Any]]:
    """[D23] ``commitments.tables`` from E2's rule: every table rebuilt in this process (``e2e.interior.tables.build_table_tree``, ~1 s each) and
    recorded with ``table_record``; ``check_tables`` then confirms the pins are what the rule yields (``[]``).  Raises InstanceError otherwise."""

    try:
        from e2e.interior import tables as T
    except ImportError as e:  # pragma: no cover - E2's module absent
        raise InstanceError(f"e2e.interior.tables not importable: {e}") from e
    pins: dict[str, dict[str, Any]] = {}
    for name in names or tuple(T.TABLES):
        tree, source = T.build_table_tree(name)
        pins[name] = T.table_record(name, tree, source)
    problems = T.check_tables(pins, names=names)
    if problems:
        raise InstanceError("table pins do not re-derive: " + "; ".join(problems))
    return pins


def frozen_instance(*, instance_id: str, program: dict[str, Any], partition: dict[str, Any], vocabulary: dict[str, Any], outputs: list[dict[str, Any]],
                    q: str, s: str, delta: str, request_ids: list[str], B_ex: int = 0, B_ex_provenance: str | None = None, family: str | None = None,
                    approved_elf_sha256: list[str] | None = None, declared_inputs: dict[str, Any] | None = None, commitments: dict[str, Any] | None = None,
                    notes: list[str] | None = None, stages: list[str] | None = None, seed_custody: str | None = None, chronology_kind: str | None = None,
                    weights_authorized: Path | None = None, vocabulary_V: int | None = None, discrepancies: list[dict[str, Any]] | None = None,
                    seed_derivation: str | None = None) -> dict[str, Any]:
    """[E6 2026-09-11] A ``status = pre-challenge`` instance from EXPLICIT program / partition sections (no population directory, no vLLM
    assumption): the caller derives the counts by rule (``e2e.population_v3.expected_population`` for Serve_v1, ``expected_population_generic``
    for any registered composite) and pins them here.  ``partition`` carries the PartitionSection fields the rule fixes (plan_name,
    plan_descriptor_digest, plan_digest_kind, vu_level, cu_strategy, validation, vu_count, ru_count, count_basis, partition_digest,
    partition_digest_kind, source; optional instances / weight_input_check / first_boundary / acquisition_granularity / address_map).
    Everything else (sampler law, empty chronology, obligations, security, epoch) is the same as :func:`pre_challenge_instance`."""

    if not request_ids:
        raise InstanceError("frozen_instance: request_ids must name the epoch's requests (N_R = ru_count x len(request_ids))")
    for o in outputs:
        implied = math.log2(vocabulary_V) if vocabulary_V else None
        o.setdefault("domain_bits_vocabulary_implied", {"value": implied if implied is not None else "unknown", "label": "ANALYTICAL",
                                                        "source": f"log2(V), V = {vocabulary_V}" if vocabulary_V else "vocabulary size not supplied"})
    w_out = sum(o["multiplicity"] * o["domain_bits_declared"] for o in outputs)
    sigma = -math.log2(Fraction(delta))
    instances = int(partition.get("instances", len(request_ids)))
    part = {"vu_count_bucketed_upper": None, "address_map": None, "summary_digest": None, "instances": instances,
            "weight_input_check": dict(DEFAULT_WEIGHT_CHECK), "first_boundary": dict(DEFAULT_FIRST_BOUNDARY), "acquisition_granularity": "ru", **partition}
    n_r = int(part["ru_count"]["value"]) * instances
    if seed_derivation is None:
        seed_derivation = ("e2e.security_sample: seed = 32 verifier bytes; seed_commitment = H(domain, seed); r_q = bernoulli_subset(seed, 'q/replay-unit', "
                           "H(PHASE, 'q', instance_digest, boundary_digest), N_R, q); r_s = bernoulli_subset(seed, 's/verification-unit', "
                           "H(PHASE, 's', instance_digest, r_q_digest, interior_digest), |VUs of J|, s)")
    doc = {
        "schema": SCHEMA, "instance_id": instance_id, "status": "pre-challenge",
        "program": program,
        "vocabulary": vocabulary,
        "partition": part,
        "outputs": {"designated": outputs, "W_out_bits": {"value": w_out, "label": "ANALYTICAL", "source": "sum over designated outputs of multiplicity x declared bits"},
                    "scope": "per program instance (one request)", "source": "declaration fixed before the first commitment"},
        "sampler": {"law": {"kind": "bernoulli", "q": str(Fraction(q)), "s": str(Fraction(s)), "k": None, "N": None, "coverage_size": None, "delta_note": None,
                            "source": "declared here, before any commitment"},
                    "population_ref": "partition",
                    "chronology": {"seed_derivation": seed_derivation, "seed_commitment": None, "events": [], "challenges_drawn": [], "rerolls": 0,
                                   "source": "empty before the audit; e2e.security_sample appends every event to chronology.json and the record copies it here",
                                   **({"chronology_kind": chronology_kind} if chronology_kind else {}), **({"seed_custody": seed_custody} if seed_custody else {})},
                    "realized": None, "source": "Protocol 2 over the declared epoch (paper §5.2, appendix B; docs/PROTOCOL.md)"},
        "obligations": {"mandatory": [dict(o) for o in DEFAULT_MANDATORY], "diagnostic": [dict(o) for o in DEFAULT_DIAGNOSTIC], "stages": list(stages or DEFAULT_STAGES),
                        "source": "brief A.3; diagnostic checks never count as security samples; stages = COST-D cost-result-format.md §1 protocol-work names (R12)"},
        "security": {"delta": str(Fraction(delta)), "sigma_bits": {"value": sigma, "label": "ANALYTICAL", "source": "-log2(delta)"}, "B_ex": int(B_ex), "kappa": None,
                     "bound_algorithm": "veritor.analysis.bound.bound via veritor.frontier.bound_adapter.FOLD_OPTIONS (frozen fold)", "bound_attached": False,
                     "bound_value_bits": None,
                     "bound_not_attached_reason": "pre-challenge: the Bound is computed after the audit from this instance's law and partition; none attached yet",
                     "fixed_size_analysis_ref": None, "source": "declared here",
                     "B_ex_unit": "replay_units", "B_ex_provenance": B_ex_provenance or "declared by the operator before r_q"},
        "discrepancies": list(discrepancies or []),
        "notes": notes or ["pre-challenge instance written by e2e.protocol_instance.frozen_instance; its digest is pinned in the record's manifest before READY_PHASE0"],
        **({"weights": weights_section_from_authorized(weights_authorized)} if weights_authorized else {}),
        "epoch": epoch_section(instance_id, list(request_ids), n_r, B_ex=B_ex, B_ex_provenance=B_ex_provenance, delta=str(Fraction(delta)), family=family,
                               approved_elf_sha256=approved_elf_sha256, inputs=declared_inputs),
        **({"commitments": commitments} if commitments is not None else {}),
    }
    # canonical form (defaulted-None fields dropped) so the written document's digest survives load -> encode
    return _encode(_decode(ProtocolInstance, doc, "frozen_instance"))


def e2_layout_pin() -> dict[str, Any] | None:
    """E2's ``interior-layout/chunks-4096/v1`` pin (``e2e.interior.layout.layout_pin()``: framing, word_order, tree, digest) when E2's module is
    importable in this process, else None.  The digest is COMPUTED by E2's rule here, never copied from a report."""

    try:
        from e2e.interior import layout as e2_layout  # E2's file (eod/e2); absent until merged
    except ImportError:
        return None
    pin = dict(e2_layout.layout_pin())
    if e2_layout.check_pin(pin):
        raise InstanceError(f"e2e.interior.layout.layout_pin() does not pass its own check_pin: {e2_layout.check_pin(pin)}")
    return pin


def interior_layout_pin(*, digest: str | None = None, tree: str = "interior/<req>/<step>", framing: str = "chunks-4096", word_order: str = "gate",
                        awaiting: str = "E2 layout_digest() (e2e.interior.layout, out/eod/reports/e2.md)", use_e2: bool = True, rule: str | None = None) -> dict[str, Any]:
    """The ``commitments.interior_layout`` document (brief interface 5 / E2's rule §6): the four fields E2's ``check_pin`` compares -- ``framing``,
    ``word_order``, ``tree``, ``digest`` -- plus ``status`` and ``source``.  ``digest`` given: pinned as given.  Otherwise, when E2's module is
    importable (``use_e2``), the pin is E2's ``layout_pin()`` verbatim (status ``pinned``); else the named PLACEHOLDER."""

    if digest is None and use_e2 and (e2 := e2_layout_pin()) is not None:
        framing, word_order, tree, digest = e2["framing"], e2["word_order"], e2["tree"], e2["digest"]
        rule = e2.get("rule") or rule or "interior-layout/chunks-4096/v1"  # E2's module names its rule from D13 on; earlier modules are the v1 rule
        src = f"e2e.interior.layout.layout_pin() (E2, {rule}): computed in the freezing process"
    else:
        src = "brief EOD-2026-09-11 interface 5 / E2 rule §6: 4096-word chunks per root call in gate order, one tree per (request, step); digest = e2e.interior.layout.layout_digest()" + (
            f" ({rule}, digest as published in out/eod/reports/e2.md)" if rule else "")
    return {"framing": framing, "word_order": word_order, "tree": tree, "digest": digest, **({"rule": rule} if rule else {}),
            "status": "pinned" if digest is not None else f"{INTERIOR_LAYOUT_PLACEHOLDER}: awaiting {awaiting}",
            "source": src + "; pinned by E6 before the first commitment"}


def epoch_section(epoch_id: str, request_ids: list[str], n_r: int, *, B_ex: int, B_ex_provenance: str | None, delta: str,
                  family: str | None = None, approved_elf_sha256: list[str] | None = None, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    """[A3 contract 2026-09-10, R16] The ``epoch`` section: one audit over exactly ``request_ids`` (one program instance each) whose RUs
    are ``[0, n_r)``; ``B_ex`` is an RU count, ``delta`` the epoch's bound.  Declared before r_q; pinned by the instance digest."""

    if len(set(request_ids)) != len(request_ids) or not request_ids:
        raise InstanceError("epoch.request_ids must be a non-empty list of distinct request ids")
    return {"epoch_id": epoch_id, "request_ids": list(request_ids), "ru_indices": list(range(n_r)), "B_ex": int(B_ex), "B_ex_unit": "replay_units",
            "B_ex_provenance": B_ex_provenance or "declared by the operator before r_q", "delta": str(Fraction(delta)), "bound_scope": "epoch",
            "source": "e2e.protocol_instance.epoch_section: both commitment stages are global within this epoch (docs/PROTOCOL.md reporting requirements, R16)",
            # [A3 20:xxZ] optional pins (omitted when None): the interface family, the approved guest set, the declared inputs (WR-5/WR-6)
            **({"family": family} if family is not None else {}), **({"approved_elf_sha256": list(approved_elf_sha256)} if approved_elf_sha256 is not None else {}),
            **({"inputs": inputs} if inputs is not None else {})}


def weights_section_from_authorized(path: Path) -> dict[str, Any]:
    """The S5 ``weights`` section of a pre-challenge instance from ``scripts/rec_weights_authorize.py``'s artifact."""

    path = Path(path)
    doc = json.loads(path.read_text())
    if doc.get("schema") != "verity-vllm/weights-authorized/v1":
        raise InstanceError(f"{path}: not a weights-authorized artifact")
    cp, lay = doc["checkpoint"], doc["layout"]
    return {
        "variant": WEIGHTS_VARIANT,
        "claim": doc["claim"],
        "authorized_commitment": {
            "checkpoint_repo": cp["repo"], "checkpoint_revision": cp["revision"], "safetensors_sha256": dict(cp["safetensors_sha256"]), "config_sha256": cp["config_sha256"],
            "expected_root": doc["expected_root"], "leaf_count": int(doc["leaf_count"]), "ids_digest": doc["ids_digest"],
            "layout_rule": f"capture format {lay['capture_format']}, rows_per_leaf {lay['rows_per_leaf']}: {lay['row_rule']}; weights dtype {lay['weights_dtype']}, "
                           f"rope cache {lay['rope_cache_dtype']}; {lay['params']} parameters from {lay['rules']} profile rules (profile sha {lay['semantic_profile_sha256'][:16]})",
            "layout_digest": lay["leaf_table_digest"], "artifact": "weights_authorized.json", "artifact_sha256": _sha256_file(path),
            "computed_by": "scripts/rec_weights_authorize.py (auditor side, from manifests/checkpoints.json pins + the local checkpoint re-hashed against them; no prover artefact read)",
            "source": f"{path} computed {doc['utc']}; authorized independently of any prover claim",
        },
        "leaf_map": "weights_authorized.json#leaf_table: param -> canonical id w=<param>[/c=<row>], dtype, shape, leaf ids (tree positions are the MerkleTree "
                    "tree_sorted order of these ids); program weight Input gates -> (param, row) is the derived correspondence (DERIVE-CORR L2/L3), not re-derived here",
        "no_other_path": doc["no_other_path"]["assertion"] + "; checked by " + doc["no_other_path"]["checked_by"],
        "covers_ref": f"weights_authorized.json#covers ({doc['covers']['kind']}, {doc['covers']['leaf_count']} leaf ids, digest {doc['covers']['digest'][:16]}); "
                      "consumed by the fold only with verified coverage of exactly these units (S19) -- otherwise the Bound is computed WITHOUT the exclusion",
        "price": None, "check_result": None,
        "source": "GUIDANCE §57.10 S5: explicit, priced variant beside the ordinary sampled-input baseline",
    }


# -- CLI --------------------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="strict schema + semantic rules")
    v.add_argument("instance")
    c = sub.add_parser("check", help="cross-check against a record (+ validation artifact)")
    c.add_argument("instance")
    c.add_argument("--record", required=True)
    c.add_argument("--validation", default=None)
    g = sub.add_parser("digest")
    g.add_argument("instance")
    mk = sub.add_parser("make", help="write a pre-challenge Bernoulli instance over an e2e.population directory")
    mk.add_argument("--population-dir", required=True, help="directory with maps.json, population.json, compact_validation.json")
    mk.add_argument("--instance-id", required=True)
    mk.add_argument("--identity", required=True, help="JSON: {family, config_sha12, case, LP, STEPS, function}")
    mk.add_argument("--vocabulary", required=True, help="JSON: {digest, schema, profile_name, profile_sha256, source}")
    mk.add_argument("--outputs", required=True, help="JSON list: [{name, multiplicity, domain_bits_declared, wire_bits, enforcement, source}]")
    mk.add_argument("--vocabulary-size", type=int, default=None, help="V, for the vocabulary-implied domain log2(V)")
    mk.add_argument("--q", required=True)
    mk.add_argument("--s", required=True)
    mk.add_argument("--delta", required=True)
    mk.add_argument("--instances", type=int, default=1, help="program instances (requests) in the record")
    mk.add_argument("--coverage-size", type=int, default=None, help="diagnostic coverage draw (never a security sample)")
    mk.add_argument("--first-boundary-rule", default=None, choices=list(FIRST_BOUNDARY_RULES), help="default: outgoing boundary runs of every CU")
    mk.add_argument("--first-boundary-committed-in", default=None, help="chronology event that commits the first boundary (default 'boundary')")
    mk.add_argument("--acquisition-granularity", default="ru", choices=list(ACQUISITION))
    mk.add_argument("--weights-authorized", default=None, help="[REC S5] weights_authorized.json from scripts/rec_weights_authorize.py -> instance.weights")
    mk.add_argument("--chronology-kind", default=None, choices=list(CHRONOLOGY_KINDS))
    mk.add_argument("--seed-derivation", default=None, help="override the sampler.chronology.seed_derivation text (the recorder's actual derivation)")
    mk.add_argument("--summary-digest", default=None)
    mk.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "make":
        pdir = Path(a.population_dir)
        index = json.loads((pdir.parent / "index.json").read_text()) if (pdir.parent / "index.json").exists() else json.loads((pdir / "index.json").read_text())
        try:
            doc = pre_challenge_instance(instance_id=a.instance_id, population_dir=pdir, identity=json.loads(a.identity), vocabulary=json.loads(a.vocabulary),
                                         outputs=json.loads(a.outputs), q=a.q, s=a.s, delta=a.delta, input_gates=index["program"]["input_gates"],
                                         instances=a.instances, coverage_size=a.coverage_size, vocabulary_V=a.vocabulary_size,
                                         first_boundary=({**DEFAULT_FIRST_BOUNDARY, **({"rule": a.first_boundary_rule} if a.first_boundary_rule else {}),
                                                          **({"committed_in": a.first_boundary_committed_in} if a.first_boundary_committed_in else {})}
                                                         if (a.first_boundary_rule or a.first_boundary_committed_in) else None),
                                         acquisition_granularity=a.acquisition_granularity, weights_authorized=Path(a.weights_authorized) if a.weights_authorized else None,
                                         chronology_kind=a.chronology_kind, summary_digest=a.summary_digest,
                                         **({"seed_derivation": a.seed_derivation} if a.seed_derivation else {}))
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
            inst = load_instance(Path(a.out))
        except InstanceError as e:
            print(f"[protocol_instance] INVALID: {e}")
            return 1
        print(f"[protocol_instance] wrote {a.out}: pre-challenge {inst.sampler.law.kind}(q={inst.sampler.law.q}, s={inst.sampler.law.s}) over "
              f"{inst.partition.instances} x (N_V {inst.partition.vu_count.value}, N_R {inst.partition.ru_count.value}); digest {instance_digest(inst)}")
        return 0
    try:
        inst = load_instance(Path(a.instance))
    except InstanceError as e:
        print(f"[protocol_instance] INVALID: {e}")
        return 1
    if a.cmd == "validate":
        print(f"[protocol_instance] VALID {a.instance} (digest {instance_digest(inst)[:16]}; law {inst.sampler.law.kind}; "
              f"N_V {inst.partition.vu_count.value}, N_R {inst.partition.ru_count.value}; {len(inst.discrepancies)} discrepancies flagged)")
        return 0
    if a.cmd == "digest":
        print(instance_digest(inst))
        return 0
    problems = check_against_record(inst, Path(a.record), Path(a.validation) if a.validation else None)
    for pr in problems:
        print(f"[protocol_instance]   problem: {pr}")
    print(f"[protocol_instance] {'CONSISTENT' if not problems else 'INCONSISTENT'} with {a.record} ({len(problems)} problem(s))")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
