"""The relation registry and the ``check`` runner (contract section 7).

Registry: ``relation_id -> Relation(checker, klass, op_type, includes_full_op, ...)`` where
``checker(instance, opened_inputs) -> predicted bytes``.  ``instance`` is a
:class:`verity_vllm.check.poc_description.NodeInstance` (the node bound to request/position/port) and
``opened_inputs`` maps the node's argument names to authenticated :class:`Value` objects (a list
of them for ``all_le``/``all_lt`` arguments, the raw value for ``const`` arguments).  The
checker returns the exact bytes it predicts for the committed output; the runner compares them
byte for byte.

Two classes are kept apart everywhere: ``same_runtime`` (re-run the registered runtime op) and
``independent`` (an explicit numerical model).  Unfilled slots are registered with
``implemented=False`` and raise :class:`RelationNotImplemented` so an unsupported region is never
silent: the runner records them as ``unimplemented`` and the verdict rule turns that into
``PARTIAL``.

The runner (:func:`check`) works from the description plus authenticated openings only.  It
verifies every opening it uses against the committed root before any checker sees it, never
touches a live process, and records ``includes_full_op`` per op family.

Pure-numpy ``independent`` checkers shipped here: ``embedding.row.v1``,
``residual_add.bf16.v1`` (float32 add, round-to-nearest-even to BF16),
``argmax.lowest_index.v1`` (ties -> lowest index), ``token.equals.v1`` (sampler output equals
the boundary record), ``const.fill.v1`` (a derived leaf equals its prescribed constant).
Slots for W1/W3/W4: ``gemm``, ``rmsnorm``, ``rope``, ``attention``, ``silu_mul``, ``lm_head``,
``sampler.select`` (see :data:`SLOTS`).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from verity_vllm.program import dtypes
from verity_vllm.commit import hashing, identity
from verity_vllm.observe.capture_v1 import CaptureError, CaptureReader, Value
from verity_vllm.commit.merkle import CommitError, Opening, TreeCache, verify_opening_in
from verity_vllm.check.poc_description import Description, DescriptionError, NodeInstance
from verity_vllm.check.replay import challenge as CH

SAME_RUNTIME = "same_runtime"
INDEPENDENT = "independent"
CLASSES = (SAME_RUNTIME, INDEPENDENT)

Inputs = Mapping[str, Any]
Checker = Callable[[NodeInstance, Inputs], bytes]


class RelationError(ValueError):
    """A checker could not run on the inputs it was given (wrong dtype, shape, missing arg)."""


class RelationNotImplemented(NotImplementedError):
    """The relation slot exists in the registry but nobody has filled it yet."""


class UnknownRelation(KeyError):
    pass


class ChallengeBindingError(ValueError):
    """The description, commitments, required set or sample changed after the challenge."""


class LazyOpenError(Exception):
    """A lazily opened input did not authenticate / is missing; carries the VU status."""

    def __init__(self, status: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


class LazyInput:
    """An input a checker opens (authenticated) only when it needs it (arg ``lazy: true``).

    ``get()`` returns the :class:`Value` (or list of values for list-valued args); each opening
    is authenticated against the commitments and counted in the VU result like an eager input.
    """

    def __init__(self, name: str, refs: str | list[str], opener: Callable[[str], Value]) -> None:
        self.name = name
        self.refs = refs
        self._opener = opener
        self._value: Value | list[Value] | None = None

    def get(self) -> Value | list[Value]:
        if self._value is None:
            if isinstance(self.refs, list):
                self._value = [self._opener(r) for r in self.refs]
            else:
                self._value = self._opener(self.refs)
        return self._value

    @property
    def opened(self) -> bool:
        return self._value is not None

    def ref_list(self) -> list[str]:
        return list(self.refs) if isinstance(self.refs, list) else [self.refs]


@dataclass(frozen=True, slots=True)
class Relation:
    relation_id: str
    klass: str
    op_type: str
    checker: Checker
    includes_full_op: bool
    version: str = "1"
    doc: str = ""
    implemented: bool = True
    owner: str = "W2"

    def to_json(self) -> dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "class": self.klass,
            "op_type": self.op_type,
            "includes_full_op": self.includes_full_op,
            "version": self.version,
            "implemented": self.implemented,
            "owner": self.owner,
            "doc": self.doc,
        }


REGISTRY: dict[str, Relation] = {}


def register(
    relation_id: str,
    *,
    klass: str,
    op_type: str,
    includes_full_op: bool,
    version: str = "1",
    doc: str = "",
    implemented: bool = True,
    owner: str = "W2",
    replace: bool = False,
) -> Callable[[Checker], Checker]:
    """Decorator registering a checker.  Filling an unimplemented slot needs no ``replace``;
    overriding an implemented relation does."""
    if klass not in CLASSES:
        raise ValueError(f"class must be one of {CLASSES}")

    def wrap(checker: Checker) -> Checker:
        existing = REGISTRY.get(relation_id)
        if existing is not None and existing.implemented and not replace:
            raise ValueError(f"{relation_id} is already registered; pass replace=True")
        REGISTRY[relation_id] = Relation(
            relation_id, klass, op_type, checker, includes_full_op, version, doc, implemented, owner
        )
        return checker

    return wrap


def get(relation_id: str) -> Relation:
    try:
        return REGISTRY[relation_id]
    except KeyError as error:
        raise UnknownRelation(relation_id) from error


def ids(*, implemented: bool | None = None) -> list[str]:
    return sorted(
        r.relation_id
        for r in REGISTRY.values()
        if implemented is None or r.implemented == implemented
    )


def registry_json(registry: Mapping[str, Relation] | None = None) -> list[dict[str, Any]]:
    registry = REGISTRY if registry is None else registry
    return [registry[k].to_json() for k in sorted(registry)]


# --- helpers for checkers ------------------------------------------------------------------------


def need(inputs: Inputs, name: str) -> Value:
    value = inputs.get(name)
    if not isinstance(value, Value):
        raise RelationError(f"missing or non-leaf input {name!r}")
    return value


def scalar_int(value: Value) -> int:
    if value.dtype not in (
        "int32",
        "int64",
        "int16",
        "int8",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    ):
        raise RelationError(f"{value.canonical_id}: expected an integer scalar, got {value.dtype}")
    if value.shape not in ((), (1,)):
        raise RelationError(f"{value.canonical_id}: expected a scalar, got shape {value.shape}")
    return int(value.array().reshape(()))


def encode_scalar(dtype: str, number: int | float) -> bytes:
    return dtypes.array_to_bytes(np.array(number, dtype=dtypes.view_dtype(dtype)))


def encode_like(inst: NodeInstance, array: np.ndarray) -> bytes:
    """Encode a float32 array into the instance's output dtype (bf16 via RNE)."""
    dtype = inst.dtype
    if dtype == "bfloat16":
        return dtypes.f32_to_bf16_bytes(array)
    if dtype in ("float16", "float32", "float64"):
        return dtypes.array_to_bytes(np.asarray(array).astype(dtypes.view_dtype(dtype)))
    raise RelationError(f"cannot encode float results as {dtype}")


# --- independent checkers ------------------------------------------------------------------------


@register(
    "embedding.row.v1",
    klass=INDEPENDENT,
    op_type="embedding",
    includes_full_op=False,
    doc="Row `token` of the weight `table`, sliced byte-exactly from the opened weight leaf.",
)
def embedding_row(inst: NodeInstance, inputs: Inputs) -> bytes:
    token = scalar_int(need(inputs, "token"))
    table = need(inputs, "table")
    if len(table.shape) != 2:
        raise RelationError(f"{table.canonical_id}: embedding table must be 2-D")
    rows, width = table.shape
    if not 0 <= token < rows:
        raise RelationError(f"token {token} outside the table's {rows} rows")
    size = dtypes.itemsize(table.dtype)
    assert size is not None
    row_bytes = width * size
    return table.data[token * row_bytes : (token + 1) * row_bytes]


@register(
    "residual_add.bf16.v1",
    klass=INDEPENDENT,
    op_type="residual_add",
    includes_full_op=False,
    doc="a + b with both BF16 inputs widened to float32, one IEEE add, RNE back to BF16.",
)
def residual_add_bf16(inst: NodeInstance, inputs: Inputs) -> bytes:
    a = need(inputs, "a")
    b = need(inputs, "b")
    if a.dtype != "bfloat16" or b.dtype != "bfloat16":
        raise RelationError("residual_add.bf16.v1 needs bfloat16 inputs")
    if a.shape != b.shape:
        raise RelationError(f"shape mismatch {a.shape} vs {b.shape}")
    total = a.float32() + b.float32()
    return dtypes.f32_to_bf16_bytes(total)


@register(
    "argmax.lowest_index.v1",
    klass=INDEPENDENT,
    op_type="argmax",
    includes_full_op=False,
    doc="Greedy token: index of the maximum logit, ties resolved to the lowest index; NaN "
    "logits are an error.  Output encoded in the node's integer dtype.",
)
def argmax_lowest_index(inst: NodeInstance, inputs: Inputs) -> bytes:
    logits = need(inputs, "logits").float32().reshape(-1)
    if logits.size == 0:
        raise RelationError("empty logits")
    if np.isnan(logits).any():
        raise RelationError("NaN in logits")
    # np.argmax returns the first occurrence of the maximum, i.e. the lowest index on ties.
    index = int(np.argmax(logits))
    if inst.dtype not in ("int32", "int64"):
        raise RelationError(f"argmax output dtype {inst.dtype} must be int32 or int64")
    return encode_scalar(inst.dtype, index)


@register(
    "sampler.select.greedy.v1",
    klass=INDEPENDENT,
    op_type="sampler",
    includes_full_op=False,
    doc="Greedy sampler.select as emitted by the adapter for temperature-0 requests: the token is "
    "the lowest index of the maximum logit (vLLM's argmax).  If a ``sampling_params`` or "
    "``params`` boundary leaf is wired in, it must say greedy (temperature 0); otherwise the "
    "description is trusted to use this relation only for greedy requests.  Stochastic "
    "sampling is W3's ``sampler.select.v1``.",
)
def sampler_select_greedy(inst: NodeInstance, inputs: Inputs) -> bytes:
    params = inputs.get("sampling_params", inputs.get("params"))
    if isinstance(params, Value):
        if params.dtype != "json":
            raise RelationError("sampling_params must be a json boundary leaf")
        doc = params.json() or {}
        if float(doc.get("temperature", 0.0)) != 0.0 and doc.get("greedy") is not True:
            raise RelationError(f"sampler.select.greedy.v1 on a non-greedy request: {doc}")
    return argmax_lowest_index(inst, inputs)


@register(
    "token.equals.v1",
    klass=INDEPENDENT,
    op_type="sampler",
    includes_full_op=False,
    doc="The committed sampler output equals the protected output record at this position "
    "(output linkage; the record comes from the boundary tree).",
)
def token_equals(inst: NodeInstance, inputs: Inputs) -> bytes:
    token = scalar_int(need(inputs, "token"))
    if inst.dtype not in ("int32", "int64"):
        raise RelationError(f"token output dtype {inst.dtype} must be int32 or int64")
    return encode_scalar(inst.dtype, token)


@register(
    "const.fill.v1",
    klass=INDEPENDENT,
    op_type="other",
    includes_full_op=False,
    doc="A derived leaf equals the prescribed constant (`value` const arg) broadcast over the "
    "node's dtype/shape: zeros at an inactive expert slot, a blank output slot, ...",
)
def const_fill(inst: NodeInstance, inputs: Inputs) -> bytes:
    if "value" not in inputs:
        raise RelationError("const.fill.v1 needs a `value` const arg")
    fill = inputs["value"]
    if isinstance(fill, Mapping) and "const" in fill:
        fill = fill["const"]
    if isinstance(fill, Value):
        raise RelationError("const.fill.v1's value must be a const arg, not a leaf")
    shape = inst.node.port_shape(inst.port)
    concrete = tuple(d if isinstance(d, int) else inputs["__symbols__"][d] for d in shape)
    dtype = inst.dtype
    if dtype == "bfloat16":
        return dtypes.f32_to_bf16_bytes(np.full(concrete, float(fill), np.float32))
    return dtypes.array_to_bytes(np.full(concrete, fill, dtype=dtypes.view_dtype(dtype)))


# --- MoE block (W9): independent numpy checkers ---------------------------------------------------


def topk_lowest_index(scores: np.ndarray, k: int) -> np.ndarray:
    """Top-k indices by descending score; ties resolved to the LOWEST expert index (the fused
    ``topk_softmax`` kernel's rule: each round keeps a strictly greater value or, on equality, the
    smaller expert id, both inside a thread and across the warp reduction)."""
    out = []
    s = np.asarray(scores, dtype=np.float32).copy()
    for _ in range(min(k, len(s))):
        best = int(np.flatnonzero(s == s.max())[0])  # first (lowest index) among the maxima
        out.append(best)
        s[best] = -np.inf
    return np.asarray(out, dtype=np.int64)


@register(
    "moe.topk.ids.v1",
    klass=INDEPENDENT,
    op_type="router",
    includes_full_op=True,
    owner="W9",
    doc="Router / top-k ids from the opened router-logits row (bf16 widened to float32): the top_k "
    "expert ids in descending logit order, ties to the LOWEST expert index, in route order (int32 "
    "[top_k]). Tie rule = the fused _moe_C.topk_softmax kernel's for non-power-of-two expert "
    "counts (csrc/libtorch_stable/moe/topk_softmax_kernels.cu: moeSoftmax in float32, then "
    "moeTopK with cub::ArgMax, which keeps the strictly greater probability or, on equality, "
    "the smaller expert index; a chosen expert is excluded from later rounds). Ranking by logits "
    "equals ranking by the exact softmax (monotone); it differs from the kernel only where two "
    "distinct bf16 logits give the same float32 softmax value (|logit| below ~2^-24 relative to "
    "the row maximum, not observed). NaN/inf logits are rejected (the kernel maps them to 0).",
)
def moe_topk_ids(inst: NodeInstance, inputs: Inputs) -> bytes:
    if inputs.get("e_score_correction_bias") is not None:
        raise RelationError("moe.topk.ids.v1: e_score_correction_bias (biased routing) is not supported")
    logits = need(inputs, "logits").float32().reshape(-1)
    if not np.all(np.isfinite(logits)):
        raise RelationError("moe.topk.ids.v1: non-finite router logit")
    k = int(inst.node.port_shape(0)[-1]) if inst.node.port_shape(0) else 1
    ids = topk_lowest_index(logits, k)
    return dtypes.array_to_bytes(ids.astype(dtypes.view_dtype(inst.dtype)))


@register(
    "moe.combine.v1",
    klass=INDEPENDENT,
    op_type="combine",
    includes_full_op=True,
    owner="W9",
    doc="Combination of the routed experts: the fixed population of expert-slot outputs at the "
    "token (every slot opened, inactive ones are derived zeros) summed over the top_k slots named "
    "by the checked router ids, in route order, in float32 (acc = 0; acc += float(slot_k), "
    "k = 0..top_k-1) with one bf16 rounding of the result: the _moe_C.moe_sum kernel of the pinned "
    "nightly (csrc/libtorch_stable/moe/moe_align_sum_kernels.cu, moe_sum_vec_kernel<bf16, TOPK>: "
    "float acc[VEC]; acc[j] += static_cast<float>(in); out = static_cast<bf16>(acc)). The "
    "slots not named by the router must be the derived zero contribution: a non-zero value at "
    "an inactive slot is rejected here (N11) even though the kernel never reads it.",
)
def moe_combine(inst: NodeInstance, inputs: Inputs) -> bytes:
    slots = inputs.get("slots")
    if not isinstance(slots, list) or not all(isinstance(v, Value) for v in slots):
        raise RelationError("moe.combine.v1 needs the list of expert-slot outputs (expert=all)")
    ids = need(inputs, "ids").array().reshape(-1).astype(np.int64)
    if inst.dtype != "bfloat16":
        raise RelationError(f"moe.combine.v1: unsupported dtype {inst.dtype}")
    by_slot = {identity.parse(v.canonical_id).expert: v for v in slots}
    shape = tuple(
        d if isinstance(d, int) else inputs["__symbols__"][d] for d in inst.node.port_shape(inst.port)
    )
    active = {int(e) for e in ids}
    for e, v in by_slot.items():
        if e not in active and any(v.array().reshape(-1) != 0):
            raise RelationError(f"moe.combine.v1: inactive slot {e} carries a non-zero value")
    acc = np.zeros(dtypes.numel(shape), dtype=np.float32)
    for e in ids:
        v = by_slot.get(int(e))
        if v is None:
            raise RelationError(f"moe.combine.v1: router names slot {e} which is not in the population")
        acc = (acc + v.float32().reshape(-1)).astype(np.float32)
    return dtypes.f32_to_bf16_bytes(acc)


@register(
    "cast.widen.v1",
    klass=INDEPENDENT,
    op_type="other",
    includes_full_op=True,
    owner="W9",
    doc="aten._to_copy to a wider float dtype (bf16 -> float32 of the router logits): exact.",
)
def cast_widen(inst: NodeInstance, inputs: Inputs) -> bytes:
    x = need(inputs, "x")
    if inst.dtype == "float32":
        return dtypes.array_to_bytes(x.float32().astype(np.float32))
    if inst.dtype == "bfloat16" and x.dtype == "bfloat16":
        return x.data
    raise RelationError(f"cast.widen.v1: {x.dtype} -> {inst.dtype} is not a widening cast")


# --- slots for the other workers -----------------------------------------------------------------


def _slot(relation_id: str, klass: str, op_type: str, owner: str, hint: str) -> None:
    message = (
        f"{relation_id} ({klass}) is a registered slot with no checker yet; {owner} fills it "
        f"({hint}). Until then every VU of this family is reported as `unimplemented`, "
        "never as passed."
    )

    def unimplemented(inst: NodeInstance, inputs: Inputs) -> bytes:
        raise RelationNotImplemented(message)

    register(
        relation_id,
        klass=klass,
        op_type=op_type,
        includes_full_op=(klass == SAME_RUNTIME),
        doc=hint,
        implemented=False,
        owner=owner,
    )(unimplemented)


SLOTS: tuple[tuple[str, str, str, str, str], ...] = (
    ("gemm.same_runtime.v1", SAME_RUNTIME, "gemm", "W1",
     "torch/vLLM linear op on the opened row and weight"),
    ("gemm.bf16.v1", INDEPENDENT, "gemm", "W4",
     "Hawkeye-derived MMA sequence + epilogue model"),
    ("rmsnorm.same_runtime.v1", SAME_RUNTIME, "rmsnorm", "W1",
     "vLLM fused rms_norm / fused_add_rms_norm"),
    ("rmsnorm.bf16.v1", INDEPENDENT, "rmsnorm", "W4",
     "numpy IEEE sequence with the kernel's reduction order"),
    ("rope.same_runtime.v1", SAME_RUNTIME, "rope", "W1",
     "vLLM rotary_embedding op"),
    ("rope.bf16.v1", INDEPENDENT, "rope", "W4",
     "numpy cos/sin table + rotation, pinned rounding"),
    ("attention.same_runtime.v1", SAME_RUNTIME, "attention", "W1",
     "vLLM attention backend on opened Q and KV rows"),
    ("attention.bf16.v1", INDEPENDENT, "attention", "W4",
     "explicit softmax(QK^T)V with the backend's accumulation order"),
    ("silu_mul.same_runtime.v1", SAME_RUNTIME, "silu_mul", "W1",
     "vLLM silu_and_mul op"),
    ("silu_mul.bf16.v1", INDEPENDENT, "silu_mul", "W4",
     "numpy silu(x)*y in float32, RNE to BF16"),
    ("lm_head.same_runtime.v1", SAME_RUNTIME, "lm_head", "W1",
     "torch lm_head GEMM on the opened hidden row"),
    ("lm_head.bf16.v1", INDEPENDENT, "lm_head", "W4",
     "same model as gemm.bf16.v1 at the vocab shape"),
    ("sampler.select.v1", INDEPENDENT, "sampler", "W3",
     "vLLM sampler path re-derivation: greedy tie rule, seeded top-p/top-k"),
)  # fmt: skip

for _rid, _klass, _op_type, _owner, _hint in SLOTS:
    _slot(_rid, _klass, _op_type, _owner, _hint)


# --- challenge -----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Challenge:
    """What the checker binds itself to before it opens anything (N13)."""

    commitments_digest: str
    description_digest: str
    required_digest: str | None
    sample: tuple[str, ...]

    def digest(self) -> bytes:
        return hashing.H(
            hashing.CHALLENGE_TAG,
            bytes.fromhex(self.commitments_digest),
            bytes.fromhex(self.description_digest),
            bytes.fromhex(self.required_digest) if self.required_digest else b"",
            hashing.canonical_json_bytes(list(self.sample)),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "commitments_digest": self.commitments_digest,
            "description_digest": self.description_digest,
            "required_digest": self.required_digest,
            "sample": list(self.sample),
            "digest": self.digest().hex(),
        }

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Challenge:
        ch = cls(
            str(doc["commitments_digest"]),
            str(doc["description_digest"]),
            doc.get("required_digest"),
            tuple(str(s) for s in doc["sample"]),
        )
        if "digest" in doc and ch.digest().hex() != doc["digest"]:
            raise ChallengeBindingError("challenge digest does not match its fields")
        return ch


def make_challenge(
    commitments: Mapping[str, Any],
    description: Description,
    sample: Sequence[str],
    required_doc: Mapping[str, Any] | None = None,
) -> Challenge:
    return Challenge(
        commitments_digest=hashing.json_digest(dict(commitments)).hex(),
        description_digest=description.digest().hex(),
        required_digest=hashing.json_digest(dict(required_doc)).hex() if required_doc else None,
        sample=tuple(sample),
    )


# --- results -------------------------------------------------------------------------------------

STATUSES = (
    "exact",  # predicted bytes == committed bytes
    "mismatch",  # relation executed, bytes differ
    "auth_failed",  # an opening (output or input) did not authenticate
    "missing",  # output or an input is not committed anywhere
    "schema",  # leaf dtype/shape disagree with the description (N04)
    "structural",  # the id is not an instance the description predicts (namespace)
    "unimplemented",  # relation slot has no checker
    "unresolved",  # description says relation "unresolved": explicit gap in the coverage map
    "unknown_relation",
    "error",  # checker raised RelationError
)

UNRESOLVED = "unresolved"


@dataclass(slots=True)
class VUResult:
    canonical_id: str
    op: str
    op_type: str
    relation_id: str
    klass: str
    status: str
    detail: str | None = None
    first_divergence: dict[str, Any] | None = None
    inputs_opened: int = 0
    input_bytes: int = 0
    output_bytes: int = 0
    checker_seconds: float = 0.0
    includes_full_op: bool = False
    # declared lazy inputs the checker did not open: the instance's value does not depend on them
    # (an inactive expert slot's per-expert weights); a mutation of one is a no-op for this VU
    inputs_unopened: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "canonical_id": self.canonical_id,
            "op": self.op,
            "op_type": self.op_type,
            "relation_id": self.relation_id,
            "class": self.klass,
            "status": self.status,
            "detail": self.detail,
            "first_divergence": self.first_divergence,
            "inputs_opened": self.inputs_opened,
            "inputs_unopened": list(self.inputs_unopened),
            "input_bytes": self.input_bytes,
            "output_bytes": self.output_bytes,
            "checker_seconds": self.checker_seconds,
            "includes_full_op": self.includes_full_op,
        }


@dataclass(slots=True)
class FamilyResult:
    op_type: str
    klass: str
    exact: int = 0
    attempted: int = 0
    mismatched: int = 0
    unimplemented: int = 0
    errors: int = 0
    auth_failures: int = 0
    seconds: float = 0.0
    bytes: int = 0
    includes_full_op: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "op_type": self.op_type,
            "class": self.klass,
            "exact": self.exact,
            "attempted": self.attempted,
            "mismatched": self.mismatched,
            "unimplemented": self.unimplemented,
            "errors": self.errors,
            "auth_failures": self.auth_failures,
            "checker_work": {
                "seconds": self.seconds,
                "flops_or_bytes": self.bytes,
                "includes_full_op": self.includes_full_op,
            },
        }


@dataclass(slots=True)
class CheckReport:
    challenge_digest: str | None
    results: list[VUResult] = field(default_factory=list)
    openings_verified: int = 0
    openings_attempted: int = 0

    def families(self) -> dict[str, dict[str, FamilyResult]]:
        out: dict[str, dict[str, FamilyResult]] = {}
        for r in self.results:
            fam = out.setdefault(r.op_type, {}).setdefault(
                r.klass, FamilyResult(r.op_type, r.klass)
            )
            if r.status in ("exact", "mismatch"):
                fam.attempted += 1
                fam.seconds += r.checker_seconds
                fam.bytes += r.input_bytes + r.output_bytes
                fam.includes_full_op = fam.includes_full_op or r.includes_full_op
            if r.status == "exact":
                fam.exact += 1
            elif r.status == "mismatch":
                fam.mismatched += 1
            elif r.status == "unimplemented":
                fam.unimplemented += 1
            elif r.status == "auth_failed":
                fam.auth_failures += 1
            else:
                fam.errors += 1
        return out

    def relation_checks(self) -> dict[str, dict[str, int]]:
        out = {k: {"exact": 0, "attempted": 0} for k in CLASSES}
        for fams in self.families().values():
            for klass, fam in fams.items():
                if klass not in out:
                    continue  # no relation resolved for these VUs: nothing was attempted
                out[klass]["exact"] += fam.exact
                out[klass]["attempted"] += fam.attempted
        return out

    def authentication(self) -> dict[str, int]:
        return {"valid": self.openings_verified, "attempted": self.openings_attempted}

    def statuses(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.status] = counts.get(r.status, 0) + 1
        return counts

    def all_exact(self) -> bool:
        return bool(self.results) and all(r.status == "exact" for r in self.results)

    def rejected(self) -> list[VUResult]:
        return [r for r in self.results if r.status != "exact"]

    def to_json(self) -> dict[str, Any]:
        return {
            "challenge_digest": self.challenge_digest,
            "authentication": self.authentication(),
            "relation_checks": self.relation_checks(),
            "statuses": self.statuses(),
            "families": {
                op_type: {klass: fam.to_json() for klass, fam in fams.items()}
                for op_type, fams in self.families().items()
            },
            "results": [r.to_json() for r in self.results],
        }


def first_divergence(expected: bytes, actual: bytes) -> dict[str, Any]:
    n = min(len(expected), len(actual))
    for i in range(n):
        if expected[i] != actual[i]:
            lo, hi = max(0, i - 4), min(n, i + 12)
            return {
                "byte": i,
                "expected_hex": expected[lo:hi].hex(),
                "actual_hex": actual[lo:hi].hex(),
                "window_start": lo,
            }
    return {"byte": n, "expected_len": len(expected), "actual_len": len(actual)}


# --- the runner ----------------------------------------------------------------------------------


def _open_verified(
    cache: TreeCache, commitments: Mapping[str, Any], canonical_id: str, report: CheckReport
) -> tuple[Opening | None, str | None]:
    try:
        opening = cache.open(canonical_id)
    except KeyError as error:
        return None, f"missing: {error}"
    except CaptureError as error:
        # review fix (W2c): a leaf whose bytes are not in the bundle (weights referenced by
        # checkpoint sha with no resolver, a truncated values.bin) is `missing`, not a crash
        return None, f"missing: {error}"
    except CommitError as error:
        # WB2 (audit-security-baseline finding 4, WA's M6 / M12): the stored bytes do not hash to the
        # leaf the tree holds (a tampered index leaf, a tampered stored value) -- the opening does
        # not authenticate; a rejection counted like any other, never a traceback.
        report.openings_attempted += 1
        return None, f"auth_failed: {error}"
    report.openings_attempted += 1
    if not verify_opening_in(commitments, opening):
        return None, f"auth_failed: {canonical_id} does not authenticate under {opening.tree}"
    report.openings_verified += 1
    return opening, None


def check(
    description: Description,
    reader: CaptureReader,
    commitments: Mapping[str, Any],
    sample: Sequence[str],
    *,
    challenge: Challenge | None = None,
    required_doc: Mapping[str, Any] | None = None,
    registry: Mapping[str, Relation] | None = None,
    cache: TreeCache | None = None,
) -> CheckReport:
    """Run the registered relation for every VU (output canonical id) in ``sample``.

    Every opening used, output and inputs alike, is authenticated against ``commitments`` before
    a checker runs.  ``challenge`` (if given) must match the digests of the description,
    commitments, required set and sample actually used, else :class:`ChallengeBindingError`.
    """
    if not isinstance(reader, CaptureReader):
        raise TypeError("check() reads a finished capture, never a live process")
    registry = REGISTRY if registry is None else registry
    if challenge is not None:
        actual = make_challenge(commitments, description, sample, required_doc)
        if actual != challenge:
            raise ChallengeBindingError(
                "challenge binding mismatch: "
                + ", ".join(
                    name
                    for name in (
                        "commitments_digest",
                        "description_digest",
                        "required_digest",
                        "sample",
                    )
                    if getattr(actual, name) != getattr(challenge, name)
                )
                + " changed after the challenge was fixed"
            )
    cache = cache or TreeCache(reader)
    report = CheckReport(challenge.digest().hex() if challenge else None)
    profile_id = reader.profile_id

    for canonical_id in sample:
        result = VUResult(canonical_id, "", "", "", "", "structural")
        report.results.append(result)
        try:
            cid = identity.parse(canonical_id)
            if cid.kind != "value":
                raise DescriptionError("sample entries must be value ids")
            boundary = reader.boundary(cid.request_id or "")
            inst = description.instance_of(canonical_id, boundary)
        except (identity.IdentityError, DescriptionError, KeyError, ValueError) as error:
            result.detail = str(error)
            continue
        node = inst.node
        result.op, result.op_type, result.relation_id = node.op, node.op_type, node.relation
        relation = registry.get(node.relation)
        if relation is None:
            # "unresolved" is the adapter's explicit marker for an op with no relation yet (15.9);
            # anything else is a relation id nobody registered
            result.status = "unresolved" if node.relation == UNRESOLVED else "unknown_relation"
            result.detail = node.relation
            continue
        result.klass = relation.klass
        result.includes_full_op = relation.includes_full_op

        opening, problem = _open_verified(cache, commitments, canonical_id, report)
        if opening is None:
            result.status, result.detail = (problem or "").split(":")[0], problem
            continue
        dtype, shape = description.leaf_spec(node, inst.port)
        if opening.dtype != dtype or opening.shape != shape:
            result.status = "schema"
            result.detail = (
                f"leaf is {opening.dtype}{list(opening.shape)}, "
                f"description says {dtype}{list(shape)}"
            )
            continue
        result.output_bytes = len(opening.value)

        inputs: dict[str, Any] = {"__symbols__": description.symbols}
        failed = False
        try:
            resolved = description.resolve_args(inst, boundary, profile_id)
        except DescriptionError as error:
            result.status, result.detail = "missing", str(error)
            continue
        lazy_names = {a.name for a in node.args if a.extra.get("lazy")}

        def open_input(r: str, result: VUResult = result) -> Value:
            o, problem = _open_verified(cache, commitments, r, report)
            if o is None:
                raise LazyOpenError((problem or "").split(":")[0], problem or "")
            result.inputs_opened += 1
            result.input_bytes += len(o.value)
            return o.as_value()

        for name, ref in resolved.items():
            if isinstance(ref, dict) and "const" in ref:
                inputs[name] = ref["const"]
                continue
            if name in lazy_names:
                # opened and authenticated only if the checker asks (e.g. per-expert weights that
                # an inactive slot's zero prescription never needs)
                inputs[name] = LazyInput(name, ref, open_input)
                continue
            refs = ref if isinstance(ref, list) else [ref]
            opened: list[Value] = []
            try:
                for r in refs:
                    opened.append(open_input(r))
            except LazyOpenError as error:
                result.status, result.detail = error.status, error.detail
                failed = True
                break
            inputs[name] = opened if isinstance(ref, list) else opened[0]
        if failed:
            continue

        started = time.perf_counter()
        try:
            predicted = relation.checker(inst, inputs)
        except RelationNotImplemented as error:
            result.status, result.detail = "unimplemented", str(error)
            continue
        except LazyOpenError as error:
            result.status, result.detail = error.status, error.detail
            continue
        except RelationError as error:
            result.status, result.detail = "error", str(error)
            continue
        except Exception as error:  # a checker bug is still a failed check, never a pass
            result.status, result.detail = "error", f"{type(error).__name__}: {error}"
            continue
        finally:
            result.inputs_unopened = [
                r for v in inputs.values() if isinstance(v, LazyInput) and not v.opened for r in v.ref_list()
            ]
        result.checker_seconds = time.perf_counter() - started
        if predicted == opening.value:
            result.status = "exact"
        else:
            result.status = "mismatch"
            result.first_divergence = first_divergence(predicted, opening.value)
            result.detail = f"predicted != committed at byte {result.first_divergence['byte']}"
    return report


# --- output linkage (N14) ------------------------------------------------------------------------


@dataclass(slots=True)
class LinkageReport:
    request_id: str
    checked: int = 0
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_json(self) -> dict[str, Any]:
        return {"request_id": self.request_id, "checked": self.checked, "problems": self.problems}


def domain_violation(value: Value, bits: int) -> str | None:
    """The GUIDANCE §32.5 unconditional output type check on one authenticated integer leaf.

    The committed container bytes, reinterpreted as one unsigned little-endian integer, must
    have every bit above position ``bits`` equal to zero — for the pinned B0 convention
    (int32 container, declared domain 16) this is exactly "the high 16-bit word == 0".  A
    negative int32 (two's complement: high word 0xFFFF) violates.  This links the declared
    mathematical domain to the SAME authenticated value the commitment binds; it never
    truncates (a claimed ``0x0001_00C6`` is rejected, not read as token 198).  Unused codes
    inside the domain (B0: 49,152..65,535) pass — exact vocabulary membership is not required
    for the W_out bound (§31.2/§32.5); vocabulary-level faults stay with the sampled argmax /
    embedding relations.
    """
    raw = int.from_bytes(value.data, "little", signed=False)
    if raw >> bits:
        return (
            f"{value.canonical_id}: committed value 0x{raw:0{2 * len(value.data)}x} outside the "
            f"declared {bits}-bit domain (container bits above {bits} are set)"
        )
    return None


def check_output_linkage(
    description: Description,
    reader: CaptureReader,
    commitments: Mapping[str, Any],
    request_id: str,
    *,
    sampler_op: str = "sampler.select",
    cache: TreeCache | None = None,
) -> LinkageReport:
    """The protected output records are what the replayed sampler produced (contract 7, N14).

    For every record slot ``t``: the boundary's ``output_presence`` leaf authenticates; a present
    slot's ``output_token`` leaf authenticates and equals the committed ``sampler.select`` leaf at
    ``t``; a blank slot has no sampler output.  Presence must be a prefix consistent with
    ``finish_reason`` (``length`` leaves no blank slot; ``stop`` requires an eos/stop token).

    When the sampler node declares an output domain (``declared_domain``, GUIDANCE §31.2/§32.5),
    every opened ``output_token`` leaf and every opened ``sampler.select`` leaf additionally
    passes :func:`domain_violation` — one unconditional comparison per opened output leaf, run
    on every audit this check runs on, independent of both sampling draws.  Without a
    declaration nothing changes (pre-declaration records verify exactly as before).
    """
    cache = cache or TreeCache(reader)
    report = LinkageReport(request_id)
    boundary = reader.boundary(request_id)
    report.problems.extend(boundary.validate())
    profile_id = reader.profile_id
    sampler = description.node(sampler_op) if sampler_op in description else None
    declared = sampler.declared_domain if sampler is not None else None
    # the boundary's steps and the committed replay units of this request must be the same set
    committed_steps = {
        int(rec["step"])
        for tid, rec in commitments.get("trees", {}).items()
        if rec.get("kind") == "ru" and rec.get("request_id") == request_id
    }
    declared_steps = {s.step for s in boundary.steps}
    if committed_steps != declared_steps:
        report.problems.append(
            f"replay units committed {sorted(committed_steps)} != boundary steps "
            f"{sorted(declared_steps)}"
        )

    def sampler_id_at(t: int) -> str | None:
        if sampler is None:
            return None
        step = boundary.step_of(t)
        phases = [boundary.phase_at(t)] if step is not None else list(identity.PHASES)
        # a tensor-parallel sampler is replicated: rank 0's copy is the one linked to the output
        # record (the replicas' equality is a separate check)
        shards: list[int | None] = list(range(sampler.shards)) if sampler.shards else [None]
        for phase in phases:
            for shard in shards:
                cid = str(
                    identity.value_id(profile_id, request_id, t, phase, sampler.layer, sampler.op, 0, shard=shard)
                )
                if reader.locate(cid) is not None:
                    return cid
        return None

    for record in boundary.output_records:
        t = record.t
        report.checked += 1
        presence_id = str(identity.boundary_id(profile_id, request_id, "output_presence", t))
        opening, problem = _open_verified(cache, commitments, presence_id, CheckReport(None))
        if opening is None:
            report.problems.append(problem or presence_id)
            continue
        if bool(opening.as_value().array()) != record.present:
            report.problems.append(f"t={t}: presence leaf disagrees with the record")
        sampler_id = sampler_id_at(t)
        if record.present:
            token_id = str(identity.boundary_id(profile_id, request_id, "output_token", t))
            token_opening, problem = _open_verified(cache, commitments, token_id, CheckReport(None))
            if token_opening is None:
                report.problems.append(problem or token_id)
                continue
            if declared is not None:
                # the §32.5 unconditional type check, before sampler resolution: one comparison
                # per opened output leaf, on every audit, whatever the sampling draws did
                violation = domain_violation(token_opening.as_value(), declared)
                if violation:
                    report.problems.append(f"t={t}: {violation}")
            if sampler_id is None:
                report.problems.append(f"t={t}: no sampler output can be located for this record")
                continue
            sampler_opening, problem = _open_verified(
                cache, commitments, sampler_id, CheckReport(None)
            )
            if sampler_opening is None:
                report.problems.append(problem or sampler_id)
                continue
            if declared is not None:
                # the committed interior sampler value carries the same declared domain — the
                # same-value link of §31.2 (silent truncation of either copy is not equivalent)
                violation = domain_violation(sampler_opening.as_value(), declared)
                if violation:
                    report.problems.append(f"t={t}: {violation}")
            if scalar_int(token_opening.as_value()) != scalar_int(sampler_opening.as_value()):
                report.problems.append(
                    f"t={t}: record token {scalar_int(token_opening.as_value())} != sampler output "
                    f"{scalar_int(sampler_opening.as_value())}"
                )
        elif sampler_id is not None:
            report.problems.append(f"t={t}: blank record but a sampler output is committed")
    if boundary.finish_reason == "length" and any(not r.present for r in boundary.output_records):
        report.problems.append("finish_reason=length with blank slots")
    if boundary.finish_reason == "stop" and boundary.output_token_ids:
        params = boundary.sampling_params
        stop_ids = set(params.get("stop_token_ids", []) or [])
        if params.get("eos_token_id") is not None:
            stop_ids.add(int(params["eos_token_id"]))
        if stop_ids and boundary.output_token_ids[-1] not in stop_ids:
            report.problems.append("finish_reason=stop but the last token is not a stop token")
    return report


def check_output_domains(
    description: Description,
    reader: CaptureReader,
    commitments: Mapping[str, Any],
    request_id: str,
    *,
    sampler_op: str = "sampler.select",
    cache: TreeCache | None = None,
) -> LinkageReport:
    """The GUIDANCE §32.5 unconditional output type check, boundary-only — every request, every audit.

    Opens every present ``output_token`` boundary leaf of the request against its committed root
    (bound at ``T_0``, before ``r_q``) and requires :func:`domain_violation` to pass under the
    sampler node's ``declared_domain`` (the pinned B0 convention: int32 container, declared 16 —
    the high 16-bit word == 0).  Unlike :func:`check_output_linkage` this needs no interior
    commitment, so the selective protocol can — and must — run it for EVERY request of the
    workload, including requests outside the replay-unit selection ``J``: a domain check run only
    for selected requests would leave out-of-domain claims in unselected requests surviving with
    probability ``1 - q`` (exactly the two-stage trap of GUIDANCE §31.3 / mandatory-proofs.md §1),
    and the 16-bit ``W_out`` accounting would be unsound.  With no declared domain the report is
    trivially ok with ``checked = 0``: pre-declaration records verify exactly as before.
    """
    cache = cache or TreeCache(reader)
    report = LinkageReport(request_id)
    node = description.node(sampler_op) if sampler_op in description else None
    declared = node.declared_domain if node is not None else None
    if declared is None:
        return report
    boundary = reader.boundary(request_id)
    profile_id = reader.profile_id
    for record in boundary.output_records:
        if not record.present:
            continue
        report.checked += 1
        token_id = str(identity.boundary_id(profile_id, request_id, "output_token", record.t))
        opening, problem = _open_verified(cache, commitments, token_id, CheckReport(None))
        if opening is None:
            report.problems.append(problem or token_id)
            continue
        violation = domain_violation(opening.as_value(), declared)
        if violation:
            report.problems.append(f"t={record.t}: {violation}")
    return report


# --- sample coverage (N07) -----------------------------------------------------------------------


@dataclass(slots=True)
class SampleCoverage:
    ok: bool
    missing: list[str]
    duplicated: list[str]
    extra: list[str]

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "missing": self.missing,
            "duplicated": self.duplicated,
            "extra": self.extra,
        }


def verify_sample_coverage(sample: Sequence[str], report: CheckReport) -> SampleCoverage:
    """Every requested VU appears exactly once in the report and nothing else does."""
    requested = list(sample)
    seen: dict[str, int] = {}
    for r in report.results:
        seen[r.canonical_id] = seen.get(r.canonical_id, 0) + 1
    wanted = set(requested)
    missing = sorted(cid for cid in wanted if cid not in seen)
    duplicated = sorted(cid for cid, n in seen.items() if n > 1)
    if len(requested) != len(wanted):
        duplicated = sorted(set(duplicated) | {c for c in requested if requested.count(c) > 1})
    extra = sorted(cid for cid in seen if cid not in wanted)
    return SampleCoverage(not (missing or duplicated or extra), missing, duplicated, extra)


# --- drawing a stratified sample -----------------------------------------------------------------


def draw_sample(
    description: Description,
    reader: CaptureReader,
    n: int,
    *,
    seed: int,
    request_ids: Iterable[str] | None = None,
    op_types: Iterable[str] | None = None,
) -> list[str]:
    """Up to ``n`` committed output ids, round-robin over (op_type, layer, phase) strata.

    Slotted nodes (``experts``) are stratified as two families kept apart, ``<op_type>:active``
    (slots the run captured) and ``<op_type>:inactive`` (derived slots of the fixed population);
    inactive slots are drawn per position without enumerating the whole population.
    """
    rng = CH.generator(seed)
    strata: dict[tuple[str, str, str], list[Any]] = {}
    wanted_types = set(op_types) if op_types is not None else None
    profile_id = reader.profile_id
    for rid in list(request_ids) if request_ids is not None else reader.request_ids():
        boundary = reader.boundary(rid)
        # slot families of this request, by family id -> (active slots, slot count)
        fams: dict[str, tuple[list[int], int]] = {}
        for _, step in reader.ru_keys(rid):
            for f in reader.ru(rid, step).index.families:
                fams[str(f["family"])] = ([int(e) for e in f["active"]], int(f["slots"]))
        for node in description.nodes:
            if wanted_types is not None and node.op_type not in wanted_types:
                continue
            if node.experts:
                for t, phase in description.positions_of(node, boundary):
                    for port in range(node.ports):
                        fid = str(
                            identity.value_id(profile_id, rid, t, phase, node.layer, node.op, port)
                        )
                        fam = fams.get(fid)
                        if fam is None:
                            continue
                        active, slots = fam
                        for e in active:
                            strata.setdefault((f"{node.op_type}:active", node.layer, phase), []).append(
                                f"{fid}/e={e}"
                            )
                        if len(active) < slots:
                            strata.setdefault((f"{node.op_type}:inactive", node.layer, phase), []).append(
                                (fid, tuple(active), slots)
                            )
                continue
            for inst in description.instances(boundary, ops=[node.op]):
                cid = inst.output_id(profile_id)
                if reader.locate(cid) is None:
                    continue
                strata.setdefault((node.op_type, node.layer, inst.phase), []).append(cid)
    # round-robin over op families first, then over (layer, phase) strata inside a family
    families: dict[str, list[list[Any]]] = {}
    for (op_type, _, _), ids in sorted(strata.items()):
        order = rng.permutation(len(ids))
        families.setdefault(op_type, []).append([ids[i] for i in order])
    cursors = dict.fromkeys(families, 0)
    out: list[str] = []
    seen: set[str] = set()

    def materialise(item: Any) -> str:
        if isinstance(item, str):
            return item
        fid, active, slots = item
        inactive = [e for e in range(slots) if e not in active]
        return f"{fid}/e={int(inactive[int(rng.integers(len(inactive)))])}"

    while len(out) < n and any(any(b) for b in families.values()):
        for op_type, buckets in families.items():
            if len(out) >= n or not any(buckets):
                continue
            for _ in range(len(buckets)):
                bucket = buckets[cursors[op_type] % len(buckets)]
                cursors[op_type] += 1
                if bucket:
                    cid = materialise(bucket.pop())
                    if cid not in seen:
                        seen.add(cid)
                        out.append(cid)
                    break
    return out


# --- adjacent VUs (15.1): a producer whose output is a consumer's authenticated input ------------


@dataclass(frozen=True, slots=True)
class AdjacentPair:
    producer: str  # canonical id of the shared value (the producer VU's output)
    consumer: str  # canonical id of the consumer VU's output
    arg: str  # the consumer's argument bound to the shared value

    def to_json(self) -> dict[str, str]:
        return {"producer": self.producer, "consumer": self.consumer, "arg": self.arg}


def _node_inputs(
    description: Description, reader: CaptureReader, cid: str
) -> list[tuple[str, str]]:
    """(arg name, canonical id) for the node-sourced inputs of VU ``cid`` (lists expanded)."""
    try:
        c = identity.parse(cid)
        boundary = reader.boundary(c.request_id or "")
        inst = description.instance_of(cid, boundary)
        resolved = description.resolve_args(inst, boundary, reader.profile_id)
    except (identity.IdentityError, DescriptionError, KeyError, ValueError):
        return []
    out: list[tuple[str, str]] = []
    for name, ref in resolved.items():
        if isinstance(ref, dict):
            continue
        for r in ref if isinstance(ref, list) else [ref]:
            if identity.kind_of(r) == "value":
                out.append((name, r))
    return out


def adjacent_pairs(
    description: Description, reader: CaptureReader, sample: Sequence[str]
) -> list[AdjacentPair]:
    """Every (producer, consumer) pair inside ``sample`` whose shared value is one committed id."""
    chosen = set(sample)
    pairs: list[AdjacentPair] = []
    for consumer in sample:
        for name, ref in _node_inputs(description, reader, consumer):
            if ref in chosen and ref != consumer:
                pairs.append(AdjacentPair(ref, consumer, name))
    return pairs


def ensure_adjacent_pair(
    description: Description,
    reader: CaptureReader,
    sample: Sequence[str],
    *,
    seed: int,
) -> tuple[list[str], list[AdjacentPair]]:
    """Return ``sample`` extended (deterministically in ``seed``) so that it contains at least one
    adjacent pair, plus the pairs found.  Preference: add the committed producer of a consumer
    already in the sample; else add a committed consumer of a producer already in the sample;
    else add a fresh consumer with its producer.  The sample is unchanged when it already
    contains a pair or when the capture has no wired pair at all.
    """
    sample = list(sample)
    if pairs := adjacent_pairs(description, reader, sample):
        return sample, pairs
    rng = CH.generator(seed)
    order = [sample[i] for i in rng.permutation(len(sample))]
    for consumer in order:
        for name, ref in _node_inputs(description, reader, consumer):
            if reader.locate(ref) is not None:
                sample.append(ref)
                return sample, [AdjacentPair(ref, consumer, name)]
    # every committed (producer -> consumer, arg) edge of the capture
    edges: list[tuple[str, str, str]] = []
    for rid in reader.request_ids():
        boundary = reader.boundary(rid)
        for cid in description.expected_ids(boundary, reader.profile_id):
            if reader.locate(cid) is None:
                continue
            for name, ref in _node_inputs(description, reader, cid):
                if reader.locate(ref) is not None:
                    edges.append((ref, cid, name))
    chosen = set(sample)
    for producer in order:
        consumers = [e for e in edges if e[0] == producer and e[1] not in chosen]
        if consumers:
            ref, cid, name = consumers[int(rng.integers(len(consumers)))]
            sample.append(cid)
            return sample, [AdjacentPair(ref, cid, name)]
    if not edges:
        return sample, []
    ref, cid, name = edges[int(rng.integers(len(edges)))]
    for x in (cid, ref):
        if x not in sample:
            sample.append(x)
    return sample, [AdjacentPair(ref, cid, name)]


def adjacent_check(pairs: Sequence[AdjacentPair], report: CheckReport) -> dict[str, Any]:
    """15.1 invariant on a finished report: for at least one pair both VUs authenticated, the
    consumer's relation ran on the shared value as an opened input and both came out exact."""
    by_id = {r.canonical_id: r for r in report.results}
    checked = []
    ok = False
    for pair in pairs:
        p, c = by_id.get(pair.producer), by_id.get(pair.consumer)
        row = {
            **pair.to_json(),
            "producer_status": p.status if p else "absent",
            "consumer_status": c.status if c else "absent",
            "consumer_inputs_opened": c.inputs_opened if c else 0,
        }
        row["ok"] = bool(
            p and c and p.status == "exact" and c.status == "exact" and c.inputs_opened > 0
        )
        ok = ok or row["ok"]
        checked.append(row)
    return {"ok": ok, "pairs": checked}
