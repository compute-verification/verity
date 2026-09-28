"""The ``typed-obligation/v0`` object: model, canonical encoding, validation.

An object is a *set* of obligations over one registered primitive family --
the obligation projection of the typed hierarchical circuit (IR §B.8, §D,
§E, §F): the authenticated roots, the committed leaves and their word layouts
(commitment positions), the canonical value references (one per ``Value<w>``,
each the identity of the one gate that produced it) and where each is
committed, the subcircuits (shared bodies of one-output gates over registered
primitives) and the obligations (one subcircuit instance each, its boundary
bound to references).

The invariant this module enforces at the Track A / Track B seam (IR §B, §C)::

    Gate -> exactly one Value<w>.
    Composite interfaces -> ordered collections of value references.
    Commitment positions -> physical authentication locations for those values,
                            not semantic components of gate outputs.

Two deliberately different namespaces: a reference (``references[].ref``) is
the logical identity of a value -- its producing gate; its authentication
location (``references[].leaf`` + ``offset`` into that leaf's ``layout``, under
a ``root``) is a separate record.  A word may sit inside a row-block leaf
without becoming "component 17 of gate X".

Vocabulary (IR §F, the ontology directive): everything in this module and in
the JSON is Verity-level -- ``Subcircuit``, ``Gate``, ``Value``, boundary,
``Region`` / ``FunctionCall`` targets, ``Obligation``.  The legacy SP1 backend's
representation of a lowered local relation (``KindProgram``: ports, ``port k``
/ ``local i`` arguments, opened offsets, slots) is **not** part of this object;
it is produced from it by :func:`veritor.interface.lowering.lower`, behind the
lowering boundary, and is never named above it.

Three related objects, not synonyms (IR §D.1, §D.6): a *subcircuit* is any
selected set of gates of the root circuit with the wiring among them and its
induced boundary (``S.in``: the distinct references entering from outside;
``S.out``: the distinct references produced inside and consumed outside or
protected); a *region* is the contiguous special case; a *function call* is a
hierarchy-defined subcircuit with an additional declared argument/return
interface.  An obligation targets a subcircuit (``target.encoding`` says how
it is encoded -- ``region`` or ``call`` for now); its graph boundary
(``boundary.in`` / ``boundary.out``, distinct references) is kept separate
from its optional declared interface (``declared_interface.args`` /
``returns``, ordered, may repeat; IR §D.2), which projects onto the boundary
through an optional explicit partial map (``declared_interface.projection``:
a boundary index per occurrence, ``null`` for a non-boundary occurrence such
as a dead declared return; memo 2026-09-07 decision 4).

Prescribed inputs are gates (IR §D.3, §E): ``Input[k] : () -> Value<w>`` with
correct value the prescribed ``x[k]`` is an ordinary one-output gate of the
fault model, sampled under the verification plan like any other gate.  In this
object an ``Input`` gate is a gate of a subcircuit body carrying its prescribed
value as static data; its produced value is on the subcircuit's outgoing
boundary (protected), and the *lowering* -- not this level -- turns it into the
backend's pinned position.  A subcircuit that does not contain ``Input[k]``
reads ``x[k]`` as an ordinary incoming boundary value: authenticated, not
pinned, not checked -- exactly as any other unsampled gate.  No field of this
object assumes a prescribed input correct globally.

The JSON layout is fixed by :meth:`TypedObligationSet.to_json`; the canonical
bytes are the canonical JSON (sorted keys, no whitespace, printable-ASCII
strings) and the object digest is ``tagged_sha256("veritor/typed-obligation/v0",
bytes)``.

:func:`validate` accepts an object or raises :class:`InterfaceReject` with
the first applicable :class:`Code`, in the fixed order documented in
veritor's ``docs/interface/typed-obligation-v0.md`` -- so that every implementation of
the rules reports the same code for the same object.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from verity.ml.commitments.identity import canonical_json_bytes, tagged_sha256

from .families import make_tc_ampere_bf16_gate_set, make_tensor_core_gate_set
from .gateset import GateSet, make_isa_gate_set, make_word_gate_set
from .elem_bf16 import ELEM_BF16_NAME, ELEM_BF16_VERSION, make_elem_bf16_gate_set
from .int8 import make_int8_q7_gate_set
from .reduce_mufu import REDUCE_MUFU_NAME, REDUCE_MUFU_VERSION, make_reduce_mufu_gate_set

__all__ = [
    "ALIASED_INCOMING_FORMATS",
    "DIGEST_TAG",
    "FORMAT",
    "GATE",
    "IN",
    "admits_aliased_incoming",
    "INPUT",
    "MERKLE_FRAMING",
    "SUBCIRCUIT_TAG",
    "SUPPORTED_FAMILIES",
    "TARGET_ENCODINGS",
    "VLLM_FRAMING",
    "Array",
    "Binding",
    "Bits",
    "Boundary",
    "Code",
    "DeclaredInterface",
    "Definition",
    "Family",
    "Gate",
    "InterfaceReject",
    "Leaf",
    "Named",
    "Obligation",
    "Product",
    "Projection",
    "Reference",
    "Root",
    "Selector",
    "Subcircuit",
    "Target",
    "Tuple",
    "Type",
    "TypedObligationSet",
    "ValueType",
    "canonical_bytes",
    "component",
    "family_gate_set",
    "gate_width",
    "object_digest",
    "scalar_count",
    "scalar_width",
    "subcircuit_digest",
    "subcircuit_well_formed",
    "type_from_json",
    "type_to_json",
    "validate",
]

FORMAT = "typed-obligation/v0"
#: [E9, D28/D30 2026-09-12] Obligation formats, BY NAME, under which one committed word may be cited by several INCOMING positions of one
#: obligation.  EMPTY by the protocol's ruling (integrator D30): the x*x class (RMSNorm's ``F32Mul(x, x)``) is expressed by E7's
#: ``aliased-body/v1`` (``vllm-poc/e2e/coverage_v1/elem_builders.py`` ``ALIAS_RULE``: the body lists each DISTINCT committed operand word
#: once as incoming and the gate reads that port at every operand position naming it), so an obligation NEVER lists a word twice and
#: ``typed-obligation/v0`` keeps its meaning -- ``E_DUPLICATE_BOUNDARY_VALUE`` in every format (fail closed).  The plumbing
#: (``validate(obj, obligation_format=...)`` through ``check_binding`` / ``check_expected_positions``) stays; it admits nothing.
ALIASED_INCOMING_FORMATS: tuple[str, ...] = ()


def admits_aliased_incoming(obligation_format: str) -> bool:
    """Whether ``obligation_format`` (by name) admits one reference on several incoming positions of one obligation."""

    return str(obligation_format) in ALIASED_INCOMING_FORMATS
DIGEST_TAG = "veritor/typed-obligation/v0"
SUBCIRCUIT_TAG = "veritor/typed-obligation/v0/subcircuit"

#: The guest's Merkle framing (``veritor/protocol/merkle.py``): the only one
#: under which a lowered statement's openings authenticate.
MERKLE_FRAMING = "veritor/protocol/merkle/frame/v3"
#: The serving adapter's tree framing (``vllm-poc/verity_vllm/commit/hashing.py``).
VLLM_FRAMING = "verity-vllm/leaf/v1"
FRAMINGS = (MERKLE_FRAMING, VLLM_FRAMING)

#: How an obligation's target subcircuit is encoded (IR §D.1).  ``region``: one
#: contiguous interval of the root circuit in canonical gate order; ``call``: one
#: function-call instance of the hierarchy.  These are the only succinct
#: encodings implemented initially; noncontiguous encodings are an explicit
#: later item, not something v0 invents.
TARGET_ENCODINGS = ("region", "call")

_U32 = 1 << 32
_U64 = 1 << 64
#: Gate argument spaces: ``["in", i]`` the subcircuit's ``i``-th incoming
#: boundary value, ``["gate", j]`` the value of the earlier gate ``j`` of the
#: same body.
IN = "in"
GATE = "gate"
#: The prescribed-input gate ``Input[k] : () -> Value<w>`` (IR §E).  Reserved by
#: the schema; registered families name their primitives in lower case.
INPUT = "Input"
#: The legacy backend's source primitives.  They are the *lowering* of an
#: incoming boundary value or of an ``Input`` gate, never gates of this object.
_BACKEND_SOURCE_OPS = frozenset({"in", "weight"})


class Code(StrEnum):
    """Rejection codes, in the order the validator tries them."""

    #: Not an object of this format: wrong ``format``, missing or mistyped
    #: field, bad hex, non-printable-ASCII string, unknown framing, bad type,
    #: a gate argument that is not ``[space, index]``, an unknown target
    #: encoding.
    E_FORMAT = "E_FORMAT"
    #: ``family.id`` is not one the interface defines.
    E_UNSUPPORTED_FAMILY = "E_UNSUPPORTED_FAMILY"
    #: ``family.digest`` or ``family.width`` is not the family's.
    E_FAMILY_DIGEST = "E_FAMILY_DIGEST"
    #: A table is not sorted by its key, or has a duplicate key.
    E_ORDER = "E_ORDER"
    #: Two subcircuits claim one digest, or a subcircuit's digest is not the
    #: digest of its body.
    E_SUBCIRCUIT_CONFLICT = "E_SUBCIRCUIT_CONFLICT"
    #: A gate names a function the family lacks -- or a backend source
    #: primitive (``in``, ``weight``), which is a lowering, not a gate.
    E_UNKNOWN_FUNCTION = "E_UNKNOWN_FUNCTION"
    #: A gate has not exactly as many arguments as its function's arity.
    E_ARITY = "E_ARITY"
    #: A reference to something that is not there or not yet there: a later
    #: or self gate, an incoming value beyond the boundary, an outgoing gate
    #: index beyond the gates, an unknown type name, root, leaf, reference,
    #: subcircuit.
    E_DANGLING_REFERENCE = "E_DANGLING_REFERENCE"
    #: A width does not match: a gate argument's producer against the
    #: function's declared argument width, an ``Input`` value against its
    #: width, or a boundary value against the incoming type or the producing
    #: gate.
    E_WIDTH_MISMATCH = "E_WIDTH_MISMATCH"
    #: An ``Input`` gate whose value is not on the outgoing boundary: a
    #: sampled prescribed input is protected (authenticated), or it is not in
    #: the subcircuit at all.
    E_INPUT_NOT_PROTECTED = "E_INPUT_NOT_PROTECTED"
    #: An offset that does not resolve in the leaf's layout to one word, or a
    #: leaf rank at or beyond its root's count.
    E_OUT_OF_RANGE_OFFSET = "E_OUT_OF_RANGE_OFFSET"
    #: One commitment position (root+rank, root+position, or leaf+offset)
    #: declared under two identities or named by two references.
    E_IDENTITY_CONFLICT = "E_IDENTITY_CONFLICT"
    #: An obligation's boundary does not match its subcircuit's in number,
    #: or its declared interface does not project onto its boundary.
    E_BOUNDARY_SHAPE = "E_BOUNDARY_SHAPE"
    #: One obligation lists one reference twice on its boundary.
    E_DUPLICATE_BOUNDARY_VALUE = "E_DUPLICATE_BOUNDARY_VALUE"
    #: One value on the outgoing boundary of two obligations: a value has one
    #: producing gate, in one sampled subcircuit.
    E_DUPLICATE_PRODUCER = "E_DUPLICATE_PRODUCER"
    #: Well-formed at this level, but the legacy backend cannot represent it:
    #: an ``Input`` gate whose width is not the family's source width.  A
    #: lowering-boundary code.
    E_NOT_LOWERABLE = "E_NOT_LOWERABLE"


class InterfaceReject(Exception):
    """An object the interface rejects, with its code."""

    def __init__(self, code: Code, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _reject(code: Code, detail: str) -> InterfaceReject:
    return InterfaceReject(code, detail)


# --- the type language ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Bits:
    """The IR's ``Value<w>``: a ``width``-bit word (1 to 64) with exactly ``2^w`` encodings,
    encoded big-endian in ``ceil(width/8)`` bytes.  No numerical meaning: that belongs to the
    function that consumes or produces it (IR §3.1).  JSON: ``{"value": w}``."""

    width: int


@dataclass(frozen=True, slots=True)
class Array:
    """The IR's ``Array<n, T>``: ``length`` elements of ``of``, encoded in index order.  JSON: ``{"array": [n, T]}``."""

    of: Type
    length: int


@dataclass(frozen=True, slots=True)
class Tuple:
    """The IR's anonymous product ``(T1, ..., Tk)``, encoded in order; components are selected by index.  JSON: ``{"tuple": [T1, ...]}``."""

    items: tuple[Type, ...]


@dataclass(frozen=True, slots=True)
class Product:
    """The IR's record (named product): fields encoded in declaration order, selected by name.  JSON: ``{"record": [[name, T], ...]}``."""

    fields: tuple[tuple[str, Type], ...]


@dataclass(frozen=True, slots=True)
class Named:
    """A type alias (IR §3.1 "type aliases share representation expressions"): ``name`` says
    what the encoding is used as; the representation is ``of``'s.  JSON: ``{"named": [name, T]}``."""

    name: str
    of: Type


type Type = Bits | Array | Tuple | Product | Named
type Selector = int | str

#: Alias in the IR's vocabulary; code identifiers follow in the next wave (GUIDANCE §10.7).
ValueType = Bits


def type_to_json(t: Type) -> Any:
    if isinstance(t, Bits):
        return {"value": t.width}
    if isinstance(t, Array):
        return {"array": [t.length, type_to_json(t.of)]}
    if isinstance(t, Tuple):
        return {"tuple": [type_to_json(item) for item in t.items]}
    if isinstance(t, Product):
        return {"record": [[name, type_to_json(of)] for name, of in t.fields]}
    if isinstance(t, Named):
        return {"named": [t.name, type_to_json(t.of)]}
    raise TypeError(f"not a type: {t!r}")


def type_from_json(
    value: Any, table: Mapping[str, Type] | None = None, depth: int = 0
) -> Type:
    """Parse a type description; a string names an entry of ``table``."""

    if depth > 64:
        raise _reject(Code.E_FORMAT, "type nesting too deep")
    if isinstance(value, str):
        if table is None or value not in table:
            raise _reject(Code.E_DANGLING_REFERENCE, f"unknown type name {value!r}")
        return table[value]
    if not isinstance(value, dict) or len(value) != 1:
        raise _reject(Code.E_FORMAT, "a type is a one-key object or a type name")
    ((constructor, body),) = value.items()
    if constructor == "value":
        if type(body) is not int or not 1 <= body <= 64:
            raise _reject(Code.E_FORMAT, "value width must be an integer in [1, 64]")
        return Bits(body)
    if constructor == "array":
        if (
            not isinstance(body, list)
            or len(body) != 2
            or type(body[0]) is not int
            or not 1 <= body[0] < _U32
        ):
            raise _reject(Code.E_FORMAT, "array is [length >= 1, element type]")
        return Array(type_from_json(body[1], table, depth + 1), body[0])
    if constructor == "tuple":
        if not isinstance(body, list) or not body:
            raise _reject(Code.E_FORMAT, "tuple is a nonempty list of types")
        return Tuple(tuple(type_from_json(item, table, depth + 1) for item in body))
    if constructor == "record":
        if not isinstance(body, list) or not body:
            raise _reject(Code.E_FORMAT, "record is a nonempty list of [name, type]")
        fields: list[tuple[str, Type]] = []
        names: set[str] = set()
        for item in body:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not _ascii(item[0])
                or item[0] in names
            ):
                raise _reject(Code.E_FORMAT, "record field is [distinct name, type]")
            names.add(item[0])
            fields.append((item[0], type_from_json(item[1], table, depth + 1)))
        return Product(tuple(fields))
    if constructor == "named":
        if not isinstance(body, list) or len(body) != 2 or not _ascii(body[0]):
            raise _reject(Code.E_FORMAT, "named is [name, type]")
        return Named(body[0], type_from_json(body[1], table, depth + 1))
    raise _reject(Code.E_FORMAT, f"unknown type constructor {constructor!r}")


def scalar_width(t: Type) -> int | None:
    """The width of a scalar type (through named wrappers), else ``None``."""

    while isinstance(t, Named):
        t = t.of
    return t.width if isinstance(t, Bits) else None


def component(t: Type, path: Sequence[Selector]) -> Type:
    """The type at ``path`` in ``t``; raises ``E_OUT_OF_RANGE_OFFSET``."""

    for step in path:
        while isinstance(t, Named):
            t = t.of
        if isinstance(t, Array):
            if type(step) is not int or not 0 <= step < t.length:
                raise _reject(
                    Code.E_OUT_OF_RANGE_OFFSET,
                    f"index {step!r} outside array of {t.length}",
                )
            t = t.of
        elif isinstance(t, Tuple):
            if type(step) is not int or not 0 <= step < len(t.items):
                raise _reject(
                    Code.E_OUT_OF_RANGE_OFFSET,
                    f"index {step!r} outside tuple of {len(t.items)}",
                )
            t = t.items[step]
        elif isinstance(t, Product):
            if type(step) is not str:
                raise _reject(
                    Code.E_OUT_OF_RANGE_OFFSET, f"record needs a field name, got {step!r}"
                )
            for name, of in t.fields:
                if name == step:
                    t = of
                    break
            else:
                raise _reject(Code.E_OUT_OF_RANGE_OFFSET, f"no field {step!r}")
        else:
            raise _reject(Code.E_OUT_OF_RANGE_OFFSET, "path goes below a scalar")
    return t


def scalar_count(t: Type) -> int:
    while isinstance(t, Named):
        t = t.of
    if isinstance(t, Bits):
        return 1
    if isinstance(t, Array):
        return t.length * scalar_count(t.of)
    if isinstance(t, Tuple):
        return sum(scalar_count(item) for item in t.items)
    return sum(scalar_count(of) for _, of in t.fields)


# --- the model ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Family:
    """The registered primitive family every gate's function comes from, with its
    static bindings pinned by digest (IR §3.3)."""

    id: str
    digest: str
    width: int


@dataclass(frozen=True, slots=True)
class Root:
    """One authenticated commitment root (a tree; Track B's ``CommitmentRef`` plus a framing)."""

    id: str
    owner: int
    domain: str
    root: str
    count: int
    framing: str = MERKLE_FRAMING


@dataclass(frozen=True, slots=True)
class Leaf:
    """One committed leaf: a commitment position (root, leaf rank, domain position)
    and the static ``layout`` of the ``Value<w>`` words packed into it.  A layout is
    a type in the IR's sense used as a *packing*: where each word lives in the leaf's
    bytes.  It says nothing about which gates produced the words; that is the
    reference table's business (IR §B.3, §6.3: logical values and commitment chunks
    are distinct things)."""

    id: str
    root: str
    leaf: int
    position: int
    layout: Type


@dataclass(frozen=True, slots=True)
class Reference:
    """One ``Value<w>`` of the circuit and where it is authenticated.

    ``ref`` is the value's logical identity -- the root-scoped canonical reference
    of the one gate that produced it (an ``Input`` gate for a prescribed input).
    Track B never parses it.  ``leaf`` + ``offset`` is its authentication
    location: the committed leaf and the path, through that leaf's layout, to
    the word -- a layout matter, never "a component of a gate output".
    """

    ref: str
    leaf: str
    offset: tuple[Selector, ...] = ()


@dataclass(frozen=True, slots=True)
class Gate:
    """One gate of a subcircuit body: one call of a registered primitive, one
    ``Value<w>`` result (IR §B.2) -- or a prescribed-input gate.

    A primitive gate names its function ``op`` and its arguments, each one
    value: ``("in", i)`` the subcircuit's ``i``-th incoming boundary value,
    ``("gate", j)`` the value of the earlier gate ``j`` of the same body.  There
    are no result slots and no components of a gate's output.

    An ``Input`` gate (``op == "Input"``, IR §E) has no arguments; its
    prescribed value ``x[k]`` is static data: ``width`` bits, ``value`` its
    big-endian hex.  Its relation is ``claimed value == x[k]``; it is sampled
    when the subcircuit containing it is.
    """

    op: str
    args: tuple[tuple[str, int], ...] = ()
    width: int | None = None
    value: str | None = None

    @property
    def is_input(self) -> bool:
        return self.op == INPUT


@dataclass(frozen=True, slots=True)
class Definition:
    """Annotations (IR §F vocabulary): what a subcircuit *is* -- the function
    definition it is the body of, its static bindings (IR §3.3) and, when it is
    a subcircuit of a larger registered composite (IR §B.7: composite
    definitions *are* the decompositions), which one (``within``).
    Informative: the subcircuit digest is over the body alone; two bodies that
    agree are one subcircuit whatever they are called.  Every string is
    printable ASCII; every static parameter an unsigned integer or such a
    string.
    """

    name: str
    static: tuple[tuple[str, int | str], ...] = ()
    within: str | None = None


@dataclass(frozen=True, slots=True)
class Subcircuit:
    """A shared subcircuit body (IR §D.1): the selected one-output gates over
    registered primitives, in canonical order, each argument naming an incoming
    boundary value or an earlier gate of the body.

    ``incoming`` are the widths of the distinct incoming boundary values
    ``S.in``, in canonical order; ``outgoing`` the indices of the gates whose
    values form ``S.out`` -- produced inside and consumed outside or protected
    (every ``Input`` gate is among them).  ``digest`` is the tagged digest of
    the body (``incoming``, ``gates``, ``outgoing``) and is the subcircuit's
    identity across obligations; ``definition`` is informative.
    """

    digest: str
    incoming: tuple[int, ...]
    gates: tuple[Gate, ...]
    outgoing: tuple[int, ...]
    definition: Definition | None = None


@dataclass(frozen=True, slots=True)
class Binding:
    """Track B's obligation key: which run, compiled program, unit and replay unit."""

    session: str
    compiled: str
    unit: int
    replay_unit: int


@dataclass(frozen=True, slots=True)
class Target:
    """Which subcircuit of the root circuit an obligation is, as the serving side
    names it (IR §D.1).  ``encoding`` is one of :data:`TARGET_ENCODINGS`; ``id`` is
    Track A's root-scoped name for it (a region's canonical id, a call instance's
    path): carried, never interpreted by Track B.
    """

    encoding: str
    id: str


@dataclass(frozen=True, slots=True)
class Boundary:
    """The graph boundary of one obligation (IR §D.1 ``S.in`` / ``S.out``): the
    distinct value references entering the subcircuit from outside, one per
    entry of the subcircuit's ``incoming``, and the distinct references produced
    inside and protected, one per entry of ``outgoing``.  Distinctness is what a
    boundary *means*; there is no ``distinct()`` step (IR §D.2).
    """

    incoming: tuple[str, ...]
    outgoing: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Projection:
    """The explicit partial map from the declared interface onto ``boundary(S)``
    (research memo 2026-09-07, decision 4).  ``boundary(S) = boundary.in ++
    boundary.out`` (the canonical distinct union of ``S.in`` and ``S.out``);
    ``args[i]`` / ``returns[j]`` is the index in it of the ``i``-th declared
    argument / ``j``-th declared return, or ``None`` for an occurrence that is
    *not* on the boundary -- a dead declared return (a value the enclosing
    circuit never uses and does not protect) or a dead argument.  Non-boundary
    occurrences are marked, never silently dropped and never forced into the
    authenticated boundary.
    """

    args: tuple[int | None, ...]
    returns: tuple[int | None, ...]


@dataclass(frozen=True, slots=True)
class DeclaredInterface:
    """The declared call interface (IR §D.2 ``call.args`` / ``call.returns``):
    ordered, repeats kept (``F(v, v)`` has ``args = [v, v]``).  Projects onto the
    boundary.  Without ``projection`` the map is total: the distinct ``args`` are
    exactly ``boundary.incoming`` and every ``returns`` entry is on the boundary.
    With ``projection`` the map is explicit and may be partial
    (:class:`Projection`).  Optional; a region has none.
    """

    args: tuple[str, ...]
    returns: tuple[str, ...]
    projection: Projection | None = None


@dataclass(frozen=True, slots=True)
class Obligation:
    """One sampled subcircuit (IR §B.4, §D): an instance of ``subcircuit``, its
    graph boundary bound to authenticated value references, the roots those
    are committed under, its optional declared interface, its target
    (region or function-call instance) and Track B's binding.
    """

    binding: Binding
    subcircuit: str
    roots: tuple[str, ...]
    boundary: Boundary
    declared_interface: DeclaredInterface | None = None
    target: Target | None = None


@dataclass(frozen=True, slots=True)
class TypedObligationSet:
    family: Family
    types: tuple[tuple[str, Type], ...] = ()
    roots: tuple[Root, ...] = ()
    leaves: tuple[Leaf, ...] = ()
    references: tuple[Reference, ...] = ()
    subcircuits: tuple[Subcircuit, ...] = ()
    obligations: tuple[Obligation, ...] = ()
    type_table: dict[str, Type] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        self.type_table.clear()
        self.type_table.update(dict(self.types))

    # -- JSON --

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "format": FORMAT,
            "family": {
                "id": self.family.id,
                "digest": self.family.digest,
                "width": self.family.width,
            },
            "roots": [
                {
                    "id": r.id,
                    "owner": r.owner,
                    "domain": r.domain,
                    "root": r.root,
                    "count": r.count,
                    "framing": r.framing,
                }
                for r in self.roots
            ],
            "leaves": [
                {
                    "id": v.id,
                    "root": v.root,
                    "leaf": v.leaf,
                    "position": v.position,
                    "layout": type_to_json(v.layout),
                }
                for v in self.leaves
            ],
            "references": [
                {"ref": ref.ref, "leaf": ref.leaf, "offset": list(ref.offset)}
                for ref in self.references
            ],
            "subcircuits": [_subcircuit_to_json(s) for s in self.subcircuits],
            "obligations": [_obligation_to_json(o) for o in self.obligations],
        }
        if self.types:
            out["types"] = {name: type_to_json(t) for name, t in self.types}
        return out

    @classmethod
    def from_json(cls, data: Any) -> TypedObligationSet:
        """Parse the JSON layout; shape errors are ``E_FORMAT`` (validation proper is :func:`validate`)."""

        if not isinstance(data, dict):
            raise _reject(Code.E_FORMAT, "object must be a JSON object")
        if data.get("format") != FORMAT:
            raise _reject(Code.E_FORMAT, f"format must be {FORMAT!r}")
        allowed = {
            "format", "family", "types", "roots", "leaves", "references", "subcircuits", "obligations",
        }
        extra = set(data) - allowed
        if extra:
            raise _reject(Code.E_FORMAT, f"unknown top-level keys {sorted(extra)}")
        fam = _obj(data, "family", {"id", "digest", "width"})
        family = Family(_ascii_str(fam, "id"), _hex(fam, "digest", 32), _int(fam, "width", _U32))
        types_json = data.get("types", {})
        if not isinstance(types_json, dict):
            raise _reject(Code.E_FORMAT, "types must be an object")
        table: dict[str, Type] = {}
        for name in sorted(types_json):
            if not _ascii(name):
                raise _reject(Code.E_FORMAT, "type names are printable ASCII")
            # a named type may refer only to names before it (no recursion)
            table[name] = type_from_json(types_json[name], table)
        types = tuple((name, table[name]) for name in sorted(table))
        roots = tuple(
            Root(
                _ascii_str(r, "id"),
                _owner(r),
                _hex(r, "domain", 32),
                _hex(r, "root", 32),
                _int(r, "count", _U64),
                _framing(r),
            )
            for r in _list(data, "roots", {"id", "owner", "domain", "root", "count", "framing"})
        )
        leaves = tuple(
            Leaf(
                _ascii_str(v, "id"),
                _ascii_str(v, "root"),
                _int(v, "leaf", _U64),
                _int(v, "position", _U64),
                type_from_json(v["layout"], table) if "layout" in v else _missing("layout"),
            )
            for v in _list(data, "leaves", {"id", "root", "leaf", "position", "layout"})
        )
        references = tuple(
            _reference_from_json(ref) for ref in _list(data, "references", {"ref", "leaf", "offset"})
        )
        subcircuits = tuple(
            _subcircuit_from_json(s)
            for s in _list(data, "subcircuits", {"digest", "in", "gates", "out", "definition"})
        )
        obligations = tuple(
            _obligation_from_json(o)
            for o in _list(
                data,
                "obligations",
                {"binding", "subcircuit", "roots", "boundary", "declared_interface", "target"},
            )
        )
        return cls(family, types, roots, leaves, references, subcircuits, obligations)


def _gate_to_json(g: Gate) -> dict[str, Any]:
    if g.is_input:
        return {"op": INPUT, "width": g.width, "value": g.value}
    return {"op": g.op, "args": [[space, index] for space, index in g.args]}


def _subcircuit_body_json(s: Subcircuit) -> dict[str, Any]:
    return {
        "in": list(s.incoming),
        "gates": [_gate_to_json(g) for g in s.gates],
        "out": list(s.outgoing),
    }


def _subcircuit_to_json(s: Subcircuit) -> dict[str, Any]:
    out: dict[str, Any] = {"digest": s.digest, **_subcircuit_body_json(s)}
    if s.definition is not None:
        d: dict[str, Any] = {"name": s.definition.name}
        if s.definition.static:
            d["static"] = {k: v for k, v in s.definition.static}
        if s.definition.within is not None:
            d["within"] = s.definition.within
        out["definition"] = d
    return out


def _obligation_to_json(o: Obligation) -> dict[str, Any]:
    out: dict[str, Any] = {
        "binding": {
            "session": o.binding.session,
            "compiled": o.binding.compiled,
            "unit": o.binding.unit,
            "replay_unit": o.binding.replay_unit,
        },
        "subcircuit": o.subcircuit,
        "roots": list(o.roots),
        "boundary": {"in": list(o.boundary.incoming), "out": list(o.boundary.outgoing)},
    }
    if o.declared_interface is not None:
        out["declared_interface"] = {
            "args": list(o.declared_interface.args),
            "returns": list(o.declared_interface.returns),
        }
        if o.declared_interface.projection is not None:
            out["declared_interface"]["projection"] = {
                "args": list(o.declared_interface.projection.args),
                "returns": list(o.declared_interface.projection.returns),
            }
    if o.target is not None:
        out["target"] = {"encoding": o.target.encoding, "id": o.target.id}
    return out


def _definition_from_json(d: Any) -> Definition:
    d = _obj({"definition": d}, "definition", {"name", "static", "within"})
    static: list[tuple[str, int | str]] = []
    if "static" in d:
        if not isinstance(d["static"], dict):
            raise _reject(Code.E_FORMAT, "definition.static is an object of parameters")
        for key in sorted(d["static"]):
            value = d["static"][key]
            if not _ascii(key) or not (_ascii(value) or (type(value) is int and 0 <= value < _U64)):
                raise _reject(
                    Code.E_FORMAT,
                    "a static parameter is an ASCII name with an unsigned integer or ASCII value",
                )
            static.append((key, value))
    return Definition(
        _ascii_str(d, "name"), tuple(static), _ascii_str(d, "within") if "within" in d else None
    )


def _gate_from_json(g: Mapping[str, Any]) -> Gate:
    op = _ascii_str(g, "op")
    if op == INPUT:
        g = _obj({"gate": g}, "gate", {"op", "width", "value"})
        width = _int(g, "width", 65)
        value = g.get("value")
        if not isinstance(value, str) or not value or len(value) % 2 or not _is_hex(value):
            raise _reject(Code.E_FORMAT, "Input value must be lowercase hex bytes")
        return Gate(INPUT, (), width, value)
    g = _obj({"gate": g}, "gate", {"op", "args"})
    args_json = g.get("args")
    if not isinstance(args_json, list):
        raise _reject(Code.E_FORMAT, "gate args must be a list")
    args: list[tuple[str, int]] = []
    for arg in args_json:
        if (
            not isinstance(arg, list)
            or len(arg) != 2
            or arg[0] not in (IN, GATE)
            or type(arg[1]) is not int
            or not 0 <= arg[1] < _U32
        ):
            raise _reject(
                Code.E_FORMAT,
                "gate arg is ['in' | 'gate', index]: one value, no result slot or component",
            )
        args.append((arg[0], arg[1]))
    return Gate(op, tuple(args))


def _subcircuit_from_json(s: Mapping[str, Any]) -> Subcircuit:
    incoming = _index_list(s, "in")
    gates_json = s.get("gates")
    if not isinstance(gates_json, list) or not all(isinstance(g, dict) for g in gates_json):
        raise _reject(Code.E_FORMAT, "gates must be a list of objects")
    return Subcircuit(
        _hex(s, "digest", 32),
        incoming,
        tuple(_gate_from_json(g) for g in gates_json),
        _index_list(s, "out"),
        _definition_from_json(s["definition"]) if "definition" in s else None,
    )


def _ref_list(data: Mapping[str, Any], name: str) -> tuple[str, ...]:
    value = data.get(name)
    if not isinstance(value, list) or not all(_ascii(x) and x for x in value):
        raise _reject(Code.E_FORMAT, f"{name} must be a list of value references")
    return tuple(value)


def _projection_list(data: Mapping[str, Any], name: str) -> tuple[int | None, ...]:
    value = data.get(name)
    if not isinstance(value, list) or not all(
        x is None or (type(x) is int and 0 <= x < _U32) for x in value
    ):
        raise _reject(Code.E_FORMAT, f"projection.{name} must be a list of boundary indices or null")
    return tuple(value)


def _obligation_from_json(o: Mapping[str, Any]) -> Obligation:
    b = _obj(o, "binding", {"session", "compiled", "unit", "replay_unit"})
    binding = Binding(
        _hex(b, "session", 32), _hex(b, "compiled", 32), _int(b, "unit", _U64), _int(b, "replay_unit", _U64)
    )
    bnd = _obj(o, "boundary", {"in", "out"})
    boundary = Boundary(_ref_list(bnd, "in"), _ref_list(bnd, "out"))
    declared = None
    if "declared_interface" in o:
        di = _obj(o, "declared_interface", {"args", "returns", "projection"})
        projection = None
        if "projection" in di:
            p = _obj(di, "projection", {"args", "returns"})
            projection = Projection(_projection_list(p, "args"), _projection_list(p, "returns"))
        declared = DeclaredInterface(_ref_list(di, "args"), _ref_list(di, "returns"), projection)
    target = None
    if "target" in o:
        t = _obj(o, "target", {"encoding", "id"})
        encoding = _ascii_str(t, "encoding")
        if encoding not in TARGET_ENCODINGS:
            raise _reject(Code.E_FORMAT, f"target encoding must be one of {list(TARGET_ENCODINGS)}")
        target = Target(encoding, _ascii_str(t, "id"))
    return Obligation(binding, _hex(o, "subcircuit", 32), _ref_list(o, "roots"), boundary, declared, target)


def _reference_from_json(ref: Mapping[str, Any]) -> Reference:
    offset = ref.get("offset")
    if not isinstance(offset, list) or not all(
        (type(s) is int and 0 <= s < _U32) or _ascii(s) for s in offset
    ):
        raise _reject(Code.E_FORMAT, "offset is a list of indices and field names")
    return Reference(_ascii_str(ref, "ref"), _ascii_str(ref, "leaf"), tuple(offset))


# -- field helpers --


def _ascii(value: object) -> bool:
    return type(value) is str and all(0x20 <= ord(c) <= 0x7E for c in value)


def _is_hex(text: str) -> bool:
    return all(c in "0123456789abcdef" for c in text)


def _missing(name: str) -> Any:
    raise _reject(Code.E_FORMAT, f"missing field {name!r}")


def _obj(data: Mapping[str, Any], name: str, keys: set[str]) -> Mapping[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise _reject(Code.E_FORMAT, f"{name} must be an object")
    extra = set(value) - keys
    if extra:
        raise _reject(Code.E_FORMAT, f"{name} has unknown keys {sorted(extra)}")
    return value


def _list(data: Mapping[str, Any], name: str, keys: set[str]) -> list[Mapping[str, Any]]:
    value = data.get(name)
    if not isinstance(value, list):
        raise _reject(Code.E_FORMAT, f"{name} must be a list")
    for item in value:
        if not isinstance(item, dict):
            raise _reject(Code.E_FORMAT, f"{name} entries must be objects")
        extra = set(item) - keys
        if extra:
            raise _reject(Code.E_FORMAT, f"{name} entry has unknown keys {sorted(extra)}")
    return value


def _ascii_str(data: Mapping[str, Any], name: str) -> str:
    value = data.get(name)
    if not _ascii(value) or not value:
        raise _reject(Code.E_FORMAT, f"{name} must be a nonempty printable-ASCII string")
    return value


def _hex(data: Mapping[str, Any], name: str, length: int) -> str:
    value = data.get(name)
    if not isinstance(value, str) or len(value) != 2 * length or not _is_hex(value):
        raise _reject(Code.E_FORMAT, f"{name} must be {length} bytes of lowercase hex")
    return value


def _int(data: Mapping[str, Any], name: str, bound: int) -> int:
    value = data.get(name)
    if type(value) is not int or not 0 <= value < bound:
        raise _reject(Code.E_FORMAT, f"{name} must be an integer in [0, {bound})")
    return value


def _owner(data: Mapping[str, Any]) -> int:
    value = data.get("owner")
    if type(value) is not int or not -2 <= value < _U64 - 2:
        raise _reject(Code.E_FORMAT, "owner must be -2, -1 or a replay unit index")
    return value


def _framing(data: Mapping[str, Any]) -> str:
    value = data.get("framing", MERKLE_FRAMING)
    if value not in FRAMINGS:
        raise _reject(Code.E_FORMAT, f"unknown framing {value!r}")
    return value


def _index_list(data: Mapping[str, Any], name: str) -> tuple[int, ...]:
    value = data.get(name)
    if not isinstance(value, list) or not all(type(x) is int and 0 <= x < _U32 for x in value):
        raise _reject(Code.E_FORMAT, f"{name} must be a list of indices")
    return tuple(value)


# --- canonical encoding ---------------------------------------------------------


def canonical_bytes(obj: TypedObligationSet | Mapping[str, Any]) -> bytes:
    """The canonical JSON bytes of an object (sorted keys, no whitespace, UTF-8)."""

    data = obj.to_json() if isinstance(obj, TypedObligationSet) else obj
    return canonical_json_bytes(data)


def object_digest(obj: TypedObligationSet | Mapping[str, Any]) -> str:
    """``tagged_sha256("veritor/typed-obligation/v0", canonical_bytes)`` as hex."""

    return str(tagged_sha256(DIGEST_TAG, canonical_bytes(obj)))


def subcircuit_digest(s: Subcircuit) -> str:
    """The subcircuit's identity: the tagged digest of its canonical body
    (``in``, ``gates``, ``out``) -- not its ``digest`` field, not its annotation."""

    return str(tagged_sha256(SUBCIRCUIT_TAG, canonical_json_bytes(_subcircuit_body_json(s))))


# --- families -------------------------------------------------------------------


def family_gate_set(family_id: str, width: int) -> GateSet | None:
    """The pinned gate set of ``family_id`` at statement ``width``, if this package carries it.

    The SP1 kernel (``archive/sp1/common/src/gateset.rs``) also knows ``elem-f32``, ``ftz-f32``, ``pad-*`` and
    ``embed-gather``; those were not promoted from veritor (nothing in this repository builds obligations over
    them), so an object naming one is rejected here with ``E_UNSUPPORTED_FAMILY`` even though the guest would check
    it.  Extend this table when a real consumer needs a family.
    """

    name, _, version = family_id.rpartition("@")
    if (name, version) == ("veritor.toy-isa", "1") and 1 <= width <= 64:
        return make_isa_gate_set(width)
    if (name, version) == ("veritor.word-arithmetic", "2") and 1 <= width <= 64:
        return make_word_gate_set(width)
    if (name, version) == ("veritor.int8-q7", "1") and width == 32:
        return make_int8_q7_gate_set()
    if (name, version) == ("veritor.tensor-core.ampere_bf16_m16n8k16", "2") and width == 32:
        return make_tensor_core_gate_set()
    if name == "veritor.tc-ampere-bf16" and version in ("1", "2") and width == 32:
        # "1": f32_to_bf16 canonicalises NaN to 0x7FC0; "2": the hardware cvt.rn.bf16.f32 word 0x7FFF (L8)
        return make_tc_ampere_bf16_gate_set(version)
    if (name, version) == (REDUCE_MUFU_NAME, REDUCE_MUFU_VERSION) and width == 32:
        return make_reduce_mufu_gate_set()
    if (name, version) == (ELEM_BF16_NAME, ELEM_BF16_VERSION) and width == 32:
        return make_elem_bf16_gate_set()
    return None


SUPPORTED_FAMILIES = (
    "veritor.toy-isa@1",
    "veritor.word-arithmetic@2",
    "veritor.int8-q7@1",
    "veritor.tensor-core.ampere_bf16_m16n8k16@2",
    "veritor.tc-ampere-bf16@1",
    "veritor.tc-ampere-bf16@2",
    "veritor.reduce-mufu@1",
    "veritor.elem-bf16@1",
)


def gate_width(g: Gate, gate_set: GateSet) -> int:
    """The width of the one ``Value<w>`` a gate produces."""

    if g.is_input:
        assert g.width is not None
        return g.width
    return gate_set[g.op].width


# --- validation -----------------------------------------------------------------


def validate(obj: TypedObligationSet, *, obligation_format: str | None = None) -> GateSet:
    """Accept ``obj`` (returning its gate set) or raise :class:`InterfaceReject`.

    The checks run in the order of :class:`Code`'s documentation, table by
    table: family; roots; leaves; references; subcircuits; obligations;
    cross-obligation; lowerability.

    ``obligation_format`` (by name; default: the object's own ``format`` attribute when it has one, else :data:`FORMAT`) selects
    the boundary rule: under a format in :data:`ALIASED_INCOMING_FORMATS` one reference may stand on several INCOMING positions
    of one obligation; under every other format that is ``E_DUPLICATE_BOUNDARY_VALUE`` [E9, D28].
    """
    fmt = str(obligation_format) if obligation_format is not None else str(getattr(obj, "format", FORMAT) or FORMAT)
    aliasing = admits_aliased_incoming(fmt)

    fam = obj.family
    if fam.id not in SUPPORTED_FAMILIES:
        raise _reject(Code.E_UNSUPPORTED_FAMILY, f"unknown family {fam.id!r}")
    gate_set = family_gate_set(fam.id, fam.width)
    if gate_set is None:
        raise _reject(Code.E_FAMILY_DIGEST, f"{fam.id} does not resolve at width {fam.width}")
    if gate_set.digest != fam.digest:
        raise _reject(Code.E_FAMILY_DIGEST, "family digest is not the gate set's")

    # roots
    root_ids = [r.id for r in obj.roots]
    if root_ids != sorted(set(root_ids)):
        raise _reject(Code.E_ORDER, "roots must be sorted by id and distinct")
    roots = {r.id: r for r in obj.roots}

    # leaves (commitment positions)
    leaf_ids = [v.id for v in obj.leaves]
    if leaf_ids != sorted(set(leaf_ids)):
        raise _reject(Code.E_ORDER, "leaves must be sorted by id and distinct")
    coordinates: dict[tuple[str, int], str] = {}
    positions_seen: dict[tuple[str, int], str] = {}
    for v in obj.leaves:
        root = roots.get(v.root)
        if root is None:
            raise _reject(Code.E_DANGLING_REFERENCE, f"leaf {v.id!r} names unknown root {v.root!r}")
        if v.leaf >= root.count:
            raise _reject(
                Code.E_OUT_OF_RANGE_OFFSET,
                f"leaf {v.id!r} rank {v.leaf} is beyond root {v.root!r} of {root.count}",
            )
        if (v.root, v.leaf) in coordinates:
            raise _reject(Code.E_IDENTITY_CONFLICT, f"leaf {v.leaf} of {v.root!r} is declared twice")
        coordinates[(v.root, v.leaf)] = v.id
        if (v.root, v.position) in positions_seen:
            raise _reject(
                Code.E_IDENTITY_CONFLICT, f"position {v.position} of {v.root!r} is declared twice"
            )
        positions_seen[(v.root, v.position)] = v.id
    leaves = {v.id: v for v in obj.leaves}

    # references (canonical value references and where they are committed)
    ref_ids = [r.ref for r in obj.references]
    if ref_ids != sorted(set(ref_ids)):
        raise _reject(Code.E_ORDER, "references must be sorted by ref and distinct")
    committed_at: dict[tuple[str, tuple[Selector, ...]], str] = {}
    ref_width: dict[str, int] = {}
    for r in obj.references:
        leaf = leaves.get(r.leaf)
        if leaf is None:
            raise _reject(Code.E_DANGLING_REFERENCE, f"reference {r.ref!r} names unknown leaf {r.leaf!r}")
        width = scalar_width(component(leaf.layout, r.offset))
        if width is None:
            raise _reject(
                Code.E_OUT_OF_RANGE_OFFSET,
                f"reference {r.ref!r} offset {list(r.offset)} does not select one word",
            )
        key = (r.leaf, r.offset)
        if key in committed_at:
            raise _reject(
                Code.E_IDENTITY_CONFLICT,
                f"references {committed_at[key]!r} and {r.ref!r} name one commitment position",
            )
        committed_at[key] = r.ref
        ref_width[r.ref] = width
    references = {r.ref: r for r in obj.references}

    # subcircuits
    digests = [s.digest for s in obj.subcircuits]
    if len(set(digests)) != len(digests):
        raise _reject(Code.E_SUBCIRCUIT_CONFLICT, "two subcircuits claim one digest")
    if digests != sorted(digests):
        raise _reject(Code.E_ORDER, "subcircuits must be sorted by digest")
    for s in obj.subcircuits:
        if subcircuit_digest(s) != s.digest:
            raise _reject(Code.E_SUBCIRCUIT_CONFLICT, f"subcircuit {s.digest[:12]}: digest is not its body's")
        _validate_subcircuit(s, gate_set)
    subcircuits = {s.digest: s for s in obj.subcircuits}

    # obligations
    keys = [(o.binding.session, o.binding.compiled, o.binding.unit) for o in obj.obligations]
    if keys != sorted(set(keys)):
        raise _reject(Code.E_ORDER, "obligations must be sorted by (session, compiled, unit) and distinct")
    producers: dict[str, int] = {}
    for index, o in enumerate(obj.obligations):
        s = subcircuits.get(o.subcircuit)
        if s is None:
            raise _reject(Code.E_DANGLING_REFERENCE, f"obligation {index} names unknown subcircuit")
        if len(set(o.roots)) != len(o.roots):
            raise _reject(Code.E_ORDER, f"obligation {index} lists a root twice")
        for rid in o.roots:
            if rid not in roots:
                raise _reject(Code.E_DANGLING_REFERENCE, f"obligation {index} names unknown root {rid!r}")
        if len(o.boundary.incoming) != len(s.incoming):
            raise _reject(
                Code.E_BOUNDARY_SHAPE,
                f"obligation {index} binds {len(o.boundary.incoming)} incoming values to a subcircuit with {len(s.incoming)}",
            )
        if len(o.boundary.outgoing) != len(s.outgoing):
            raise _reject(
                Code.E_BOUNDARY_SHAPE,
                f"obligation {index} binds {len(o.boundary.outgoing)} outgoing values to a subcircuit with {len(s.outgoing)}",
            )
        seen: set[str] = set()
        n_in = len(o.boundary.incoming)
        for k, ref_id in enumerate((*o.boundary.incoming, *o.boundary.outgoing)):
            ref = references.get(ref_id)
            if ref is None:
                raise _reject(
                    Code.E_DANGLING_REFERENCE, f"obligation {index} names unknown reference {ref_id!r}"
                )
            if leaves[ref.leaf].root not in o.roots:
                raise _reject(
                    Code.E_DANGLING_REFERENCE,
                    f"obligation {index} binds {ref_id!r}, committed under a root it does not list",
                )
            if ref_id in seen:
                # [E9, D28] an aliased INCOMING operand (the same word read at two input positions) is legal only under a format that
                # admits it by name; on the outgoing side, or across the two sides, it never is (one producer per value)
                if aliasing and k < n_in and ref_id in o.boundary.incoming[:k]:
                    continue
                if k < n_in:  # the aliased-operand class: named with the format that refused it and the formats that admit it
                    raise _reject(
                        Code.E_DUPLICATE_BOUNDARY_VALUE,
                        f"obligation {index} lists {ref_id!r} twice among its incoming values (format {fmt!r} admits no aliased incoming value; "
                        f"admitted formats: {list(ALIASED_INCOMING_FORMATS)})",
                    )
                where = "as incoming and as outgoing" if ref_id in o.boundary.incoming else "twice on its boundary"
                raise _reject(Code.E_DUPLICATE_BOUNDARY_VALUE, f"obligation {index} lists {ref_id!r} {where}" if aliasing
                              else f"obligation {index} lists {ref_id!r} twice on its boundary")  # v0: the pinned vector's detail, verbatim
            seen.add(ref_id)
        for i, ref_id in enumerate(o.boundary.incoming):
            if ref_width[ref_id] != s.incoming[i]:
                raise _reject(
                    Code.E_WIDTH_MISMATCH,
                    f"obligation {index} binds a {ref_width[ref_id]}-bit value as incoming value {i} of {s.incoming[i]} bits",
                )
        for m, ref_id in enumerate(o.boundary.outgoing):
            gate = s.gates[s.outgoing[m]]
            if ref_width[ref_id] != gate_width(gate, gate_set):
                raise _reject(
                    Code.E_WIDTH_MISMATCH,
                    f"obligation {index} binds a {ref_width[ref_id]}-bit value to gate {s.outgoing[m]} ({gate.op}, {gate_width(gate, gate_set)} bits)",
                )
        if o.declared_interface is not None:
            _validate_declared_interface(index, o)
        for ref_id in o.boundary.outgoing:
            if ref_id in producers:
                raise _reject(
                    Code.E_DUPLICATE_PRODUCER,
                    f"value {ref_id!r} is produced by obligations {producers[ref_id]} and {index}",
                )
            producers[ref_id] = index

    # lowerability (the one backend limitation the interface reports)
    source_width = gate_set["in"].width
    for s in obj.subcircuits:
        for j, gate in enumerate(s.gates):
            if gate.is_input and gate.width != source_width:
                raise _reject(
                    Code.E_NOT_LOWERABLE,
                    f"subcircuit {s.digest[:12]} gate {j}: an Input of {gate.width} bits cannot be lowered in a family whose source width is {source_width}",
                )
    return gate_set


def _validate_declared_interface(index: int, o: Obligation) -> None:
    """The declared interface projects onto ``boundary(S) = boundary.in ++ boundary.out``.

    Without ``projection`` the map is total (every declared reference is on the
    boundary, E_DANGLING_REFERENCE; the distinct args are exactly the incoming
    boundary, E_BOUNDARY_SHAPE).  With ``projection`` it is the explicit partial
    map: an index names the boundary entry the occurrence *is* (E_BOUNDARY_SHAPE
    if the names differ, E_DANGLING_REFERENCE if there is no such entry), ``null``
    marks an occurrence that is not on the boundary (E_BOUNDARY_SHAPE if it is);
    the mapped args are exactly the incoming boundary and never an outgoing entry
    (E_BOUNDARY_SHAPE).  A total projection means what the two legacy rules mean.
    """

    di = o.declared_interface
    assert di is not None
    boundary = (*o.boundary.incoming, *o.boundary.outgoing)
    on_boundary = set(boundary)
    if di.projection is None:
        for ref_id in (*di.args, *di.returns):
            if ref_id not in on_boundary:
                raise _reject(
                    Code.E_DANGLING_REFERENCE,
                    f"obligation {index} declares {ref_id!r}, which is not on its boundary",
                )
        if set(di.args) != set(o.boundary.incoming):
            raise _reject(
                Code.E_BOUNDARY_SHAPE,
                f"obligation {index}: the distinct declared args are not the incoming boundary",
            )
        return
    p = di.projection
    if len(p.args) != len(di.args) or len(p.returns) != len(di.returns):
        raise _reject(
            Code.E_FORMAT, f"obligation {index}: projection lists must have the declared interface's lengths"
        )
    for side, refs, indices in (("arg", di.args, p.args), ("return", di.returns, p.returns)):
        for k, (ref_id, target) in enumerate(zip(refs, indices)):
            if target is None:
                if ref_id in on_boundary:
                    raise _reject(
                        Code.E_BOUNDARY_SHAPE,
                        f"obligation {index}: declared {side} {k} ({ref_id!r}) is on the boundary but marked non-boundary",
                    )
                continue
            if target >= len(boundary):
                raise _reject(
                    Code.E_DANGLING_REFERENCE,
                    f"obligation {index}: declared {side} {k} projects onto boundary index {target}, which is not there",
                )
            if boundary[target] != ref_id:
                raise _reject(
                    Code.E_BOUNDARY_SHAPE,
                    f"obligation {index}: declared {side} {k} ({ref_id!r}) projects onto boundary entry {target} ({boundary[target]!r})",
                )
    mapped_args = {target for target in p.args if target is not None}
    if mapped_args != set(range(len(o.boundary.incoming))):
        raise _reject(
            Code.E_BOUNDARY_SHAPE,
            f"obligation {index}: the mapped declared args are not exactly the incoming boundary",
        )


def _validate_subcircuit(s: Subcircuit, gate_set: GateSet) -> None:
    for w in s.incoming:
        if not 1 <= w <= 64:
            raise _reject(Code.E_FORMAT, "an incoming value width must be in [1, 64]")
    _validate_body(s, gate_set)


def subcircuit_well_formed(s: Subcircuit, gate_set: GateSet) -> bool:
    """The part of a subcircuit's validity the Lean specification decides of its lowering.

    ``WellFormed P ∧ WellTyped P`` of ``lean/VeritorSpec/Program.lean`` over the
    body's functions, arities, earlier-only references, protected ``Input``
    gates, outgoing order and argument widths -- everything :func:`validate`
    checks of a subcircuit except the ``[1, 64]`` bound on an incoming width,
    which Lean's ``Program`` does not carry.  A body with an ``Input`` gate the
    backend cannot pin (its width is not the family's source width,
    ``E_NOT_LOWERABLE``) has no lowering and is recorded ``False``.  The
    conformance vectors record this per subcircuit so the Lean runner can
    compare.
    """

    try:
        _validate_body(s, gate_set)
    except InterfaceReject:
        return False
    source_width = gate_set["in"].width
    return all(not g.is_input or g.width == source_width for g in s.gates)


def _validate_body(s: Subcircuit, gate_set: GateSet) -> None:
    produced: list[int] = []
    outgoing = set(s.outgoing)
    tag = s.digest[:12]
    for j, gate in enumerate(s.gates):
        if gate.is_input:
            assert gate.width is not None and gate.value is not None
            if not 1 <= gate.width <= 64:
                raise _reject(Code.E_FORMAT, f"subcircuit {tag} gate {j}: Input width must be in [1, 64]")
            if len(gate.value) != 2 * ((gate.width + 7) // 8) or int(gate.value, 16) >= 1 << gate.width:
                raise _reject(
                    Code.E_WIDTH_MISMATCH,
                    f"subcircuit {tag} gate {j}: Input value {gate.value!r} is not a {gate.width}-bit word",
                )
            if j not in outgoing:
                raise _reject(
                    Code.E_INPUT_NOT_PROTECTED,
                    f"subcircuit {tag} gate {j}: an Input gate's value is on the outgoing boundary",
                )
            produced.append(gate.width)
            continue
        if gate.op in _BACKEND_SOURCE_OPS:
            raise _reject(
                Code.E_UNKNOWN_FUNCTION,
                f"subcircuit {tag} gate {j}: {gate.op!r} is a backend source, not a gate -- use an incoming value or an Input gate",
            )
        try:
            semantics = gate_set[gate.op]
        except Exception:
            raise _reject(
                Code.E_UNKNOWN_FUNCTION, f"subcircuit {tag} gate {j}: unknown function {gate.op!r}"
            ) from None
        if len(gate.args) != semantics.arity:
            raise _reject(
                Code.E_ARITY,
                f"subcircuit {tag} gate {j}: {gate.op} takes {semantics.arity} arguments, got {len(gate.args)}",
            )
        for i, (space, index) in enumerate(gate.args):
            if space == IN:
                if index >= len(s.incoming):
                    raise _reject(
                        Code.E_DANGLING_REFERENCE,
                        f"subcircuit {tag} gate {j} reads incoming value {index} of {len(s.incoming)}",
                    )
                width = s.incoming[index]
            else:
                if index >= j:
                    raise _reject(
                        Code.E_DANGLING_REFERENCE,
                        f"subcircuit {tag} gate {j} reads gate {index}, not earlier",
                    )
                width = produced[index]
            if width != semantics.arg_widths[i]:
                raise _reject(
                    Code.E_WIDTH_MISMATCH,
                    f"subcircuit {tag} gate {j} argument {i} is {width} bits, {gate.op} takes {semantics.arg_widths[i]}",
                )
        produced.append(semantics.width)
    if list(s.outgoing) != sorted(set(s.outgoing)):
        raise _reject(Code.E_ORDER, f"subcircuit {tag}: out must be strictly increasing")
    for o in s.outgoing:
        if o >= len(s.gates):
            raise _reject(
                Code.E_DANGLING_REFERENCE, f"subcircuit {tag}: out names gate {o} of {len(s.gates)}"
            )
