"""CK-FTZ: the scalar flush-to-zero family of the fast-math FA2 attention kernel -- host-side handler (E4, EOD 2026-09-11).

The TINY partition's V5 selection samples FTZ *scalar* kinds that no approved checker covers (attempt 6,
``out/records/V5-TINY-bex0-inc-elem-b8177939.md``: 19 VUs of ``F32MulFtz_v1|batch`` (15), ``F32MulFtz_v1|primitive``,
``F32AddFtz_v1|primitive``, ``F32FmaSubFtz_v1|primitive``, ``Fa2InvSum_v1|primitive`` and, per RU, ~207,360 members of
the batch kind; the same kernel's ``F32Max_v1``, ``GuardNegInfZero_v1``, ``MufuEx2Ftz_v1`` are siblings of the same
family).  The brief's fixed decision keeps the scalar partition: a whole-head fa2 proof does NOT cover a scalar FTZ
statement, so every one of these kinds needs a scalar checker of its own.  This module is that checker's host side,
in the shape CK-ELEM set (``ck_elem.py``):

* **the relation**, integer-exact, on IEEE-754 binary32 encodings, exactly as ``verity_vllm/registry/prims.py``
  registers it (the source line is in :data:`KINDS`); arithmetic between the flushes is the approved backend's
  soft-float (``veritor.core.reduce_gates``, the Python twin of ``zk/sp1/common/src/reduce.rs``) -- see the NaN note;
* **the gate set** ``veritor.ftz-f32@1`` (:func:`make_ftz_gate_set`), pinned by digest to the guest's
  ``zk/sp1/programs/ftz/core/src/family.rs`` (:data:`FTZ_DIGEST`);
* **the kinds** (:data:`KINDS`): ``fn|form`` -> registered primitive, gate, operand widths;
* **the builder** (:func:`resolve` / :func:`build_many` / :func:`build`): a located VU -> one typed obligation (one
  gate over committed 32-bit words, ``typed-obligation/v0`` format 3) and its **expectation**
  (:func:`expectation`).  Operands are words of the record's committed trees: E2's interior trees (the f32 words the
  kernel actually holds) and, for kinds whose operands are boundary values, the boundary/weights trees.  The resolver
  is pluggable (:class:`WordResolver`), so the builder is program-agnostic: it never reads a kernel, only the pinned
  program's gate signature and the record's committed words;
* **the transparent mirror** (:func:`verify_transparent`): the Python evaluation of the statement + witness at the
  pinned digest -- a local check of relation and openings, NOT the approved checker's proof;
* **conformance vectors** (:func:`encode_conformance`, :func:`check_vectors`): the ``veritor/ftz-conformance/v1``
  batch the guest proves over ``ftz_vectors/*.json`` (positives and the negatives it must reject).

**Widths.**  Every operand of these kinds is an f32 word; the family resolves at statement width 32 and a committed
f32 element is ONE 32-bit word (``Array(Bits(32), n)`` per leaf, packed schema ``u32le`` -- ``packed.rs`` supports
it; this differs from the boundary exporter's two-halves layout of int32, which these kinds never read).  The two
table gates take a 64-bit second operand: the opened table row ``(index << 32) | word`` (the ``veritor.reduce-mufu@1``
table-gate contract).

**NaN rule (stated).**  ``prims.F32{Add,Sub,Mul}Ftz`` evaluate through numpy, i.e. the host CPU's NaN payload
propagation.  The approved guest pins the x86-SSE rule (``reduce.rs``: a NaN operand is returned quieted, first
operand first; ``inf - inf`` / ``0 * inf`` give the indefinite ``0xFFC00000``); this family inherits that pin.
Payloads are not banned.  The FMA kinds canonicalise to ``0x7FC00000`` (``prims.f32_fma_bits``); the MUFU kinds to
``0x7FFFFFFF`` (``fa2_model.cpp``).  ``ftz_vectors/README.md`` marks the vectors where an AArch64 evaluator would differ.

E7 owns dispatch/registry (``builders.py``, ``coverage.py``, ``approved_guests.json``); this module registers nothing
by itself beyond the same-process ``ensure_families`` interim CK-ELEM uses.
"""

from __future__ import annotations

import json
import struct
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for _p in (str(_REPO), str(_REPO / "src"), str(_REPO / "vllm-poc"), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ck_elem import CommittedWord, Refusal, _binding, _owner_of, _packed_witness  # noqa: E402
from veritor.core import reduce_gates as _rg  # noqa: E402
from veritor.core.gates import INPUT_SOURCE, WEIGHT_SOURCE, Gate, GateSet  # noqa: E402
from veritor.interface import expected as _expected  # noqa: E402
from veritor.interface import typed_obligation as _to  # noqa: E402
from veritor.interface.expected import ExpectedObligation  # noqa: E402
from veritor.interface.programs import make_subcircuit  # noqa: E402
from veritor.interface.typed_obligation import (  # noqa: E402
    IN,
    Array,
    Bits,
    Boundary,
    DeclaredInterface,
    Definition,
    Family,
    Leaf,
    Obligation,
    Reference,
    Root,
    Subcircuit,
    Target,
    Type,
    TypedObligationSet,
    validate,
)

__all__ = [
    "FTZ_FAMILY",
    "FTZ_DIGEST",
    "KINDS",
    "KindSpec",
    "Refusal",
    "CommittedWord",
    "ResolvedUnit",
    "WordResolver",
    "Built",
    "make_ftz_gate_set",
    "ensure_families",
    "family_gate_set",
    "kind_subcircuit",
    "resolve",
    "build",
    "build_many",
    "expectation",
    "verify_transparent",
    "evaluate",
    "encode_conformance",
    "decode_conformance",
    "check_vectors",
    "load_vectors",
]

FTZ_FAMILY = "veritor.ftz-f32@1"
FTZ_NAME, FTZ_VERSION = "veritor.ftz-f32", "1"
#: ``family.rs`` ``FTZ_F32_DIGEST_HEX`` -- the guest's gate-set digest; the two spellings must agree or nothing builds.
FTZ_DIGEST = "21f8f7d2183cad76ae62486445eee48409b3727b236b12329bb8d9b6e2353fef"
STATEMENT_WIDTH = 32
WORD_BITS = 32  # a committed f32 element is one 32-bit word
ROW_BITS = 64  # a committed table row is one 64-bit word: (index << 32) | T[index]
VLLM_FRAMING = "verity-vllm/leaf/v1"

# --- the relations (integer-exact twins of zk/sp1/programs/ftz/core/src/ftz.rs) -----------------------------------

F32_NAN = 0x7FC00000
MUFU_NAN = 0x7FFFFFFF
F32_ONE = 0x3F800000
_SIGN, _EXP, _MANT = 0x80000000, 0x7F800000, 0x7FFFFF
_NEG_INF = 0xFF800000
_M32 = 0xFFFFFFFF


def _is_nan(u: int) -> bool:
    return (u & _EXP) == _EXP and (u & _MANT) != 0


def ftz(u: int) -> int:
    """``prims.ftz_bits``: a subnormal encoding -> its signed zero; every other word unchanged."""
    u &= _M32
    return u & _SIGN if (u & _EXP) == 0 and (u & _MANT) else u


def f32_add_ftz(a: int, b: int) -> int:
    """``F32AddFtz_v1`` (prims.py:236): ``ftz(RN(ftz(a) + ftz(b)))``."""
    return ftz(_rg.f32_add(ftz(a), ftz(b)))


def _f32_sub(a: int, b: int) -> int:
    if _is_nan(a):
        return _rg._quiet(a)
    if _is_nan(b):
        return _rg._quiet(b)
    return _rg.f32_add(a, b ^ _SIGN)


def f32_sub_ftz(a: int, b: int) -> int:
    """``F32SubFtz_v1`` (prims.py:242): ``ftz(RN(ftz(a) - ftz(b)))``."""
    return ftz(_f32_sub(ftz(a), ftz(b)))


def f32_mul_ftz(a: int, b: int) -> int:
    """``F32MulFtz_v1`` (prims.py:248): ``ftz(RN(ftz(a) * ftz(b)))``."""
    return ftz(_rg.f32_mul(ftz(a), ftz(b)))


def f32_fma_ftz(a: int, b: int, c: int) -> int:
    """``F32FmaFtz_v1`` (prims.py:254): ``ftz(RN(ftz(a) * ftz(b) + ftz(c)))``, one rounding."""
    return ftz(_rg.f32_fma(ftz(a), ftz(b), ftz(c)))


def f32_fma_sub_ftz(a: int, b: int, c: int) -> int:
    """``F32FmaSubFtz_v1`` (prims.py:260): ``ftz(RN(ftz(a) * ftz(b) - ftz(c)))`` -- ``c`` flushed, then negated."""
    return ftz(_rg.f32_fma(ftz(a), ftz(b), ftz(c) ^ _SIGN))


def _order_key(u: int) -> int:
    mag = u & 0x7FFFFFFF
    if mag == 0:
        return 0
    return -mag if u & _SIGN else mag


def f32_max_ftz(a: int, b: int) -> int:
    """``F32Max_v1`` (prims.py:266): ``f32_max_bits(ftz(a), ftz(b))`` = ``ftz(a) > ftz(b) ? ftz(a) : ftz(b)`` (IEEE
    ``>``: a NaN on either side compares false -> the flushed second operand; ``+0 > -0`` is false)."""
    a, b = ftz(a), ftz(b)
    if _is_nan(a) or _is_nan(b):
        return b
    return a if _order_key(a) > _order_key(b) else b


def guard_neg_inf_zero(m: int) -> int:
    """``GuardNegInfZero_v1`` (prims.py:272): ``m == -inf ? +0 : m``; no flush."""
    m &= _M32
    return 0 if m == _NEG_INF else m


def _ex2_reduce(x: int) -> tuple[int, int] | None:
    ax = x & 0x7FFFFFFF
    e = ax >> 23
    if e == 0xFF or e == 0:
        return None
    unb = e - 127
    if unb >= 30:
        return None
    mant = (ax & _MANT) | 0x800000
    k = (mant << unb) & 0xFFFFFFFFFFFFFFFF if unb >= 0 else mant >> ((-unb) & 63)  # the C model's mod-64 shift
    if x & _SIGN == 0:
        fixed = k
    elif k & _MANT == 0:
        fixed = -k
    else:
        fixed = -k - 1
    return fixed >> 23, fixed & _MANT


def mufu_ex2_index(x: int) -> int:
    """The EX2 table row ``x`` reads (``j`` of ``mufu_ex2_bits``'s reduction; row 0 when the rule needs no table)."""
    r = _ex2_reduce(x & _M32)
    return 0 if r is None else r[1]


def mufu_ex2_tab(x: int, row: int) -> int:
    """``MufuEx2Ftz_v1`` (prims.py:426; ``fa2_model.cpp`` ``mufu_ex2_bits``) with the table row opened as
    ``(index << 32) | T[index]``; raises :class:`TableRowMismatch` when the row is not the one ``x`` reads."""
    x &= _M32
    word = _rg._table_word(row, mufu_ex2_index(x))
    ax = x & 0x7FFFFFFF
    e = ax >> 23
    if e == 0xFF:
        return MUFU_NAN if ax & _MANT else (0 if x & _SIGN else _EXP)
    if e == 0:
        return F32_ONE
    if e - 127 >= 30:
        return 0 if x & _SIGN else _EXP
    n, _ = _ex2_reduce(x)  # type: ignore[misc]
    re = ((word >> 23) & 0xFF) + n
    if re >= 255:
        return _EXP
    if re <= 0:
        return 0
    return (re << 23) | (word & _MANT)


def fa2_inv_sum_index(s: int) -> int:
    """The RCP row ``Fa2InvSum`` pins for ``s``: its mantissa field (``reduce_gates.mufu_rcp_index``)."""
    return s & _MANT


def fa2_inv_sum_tab(s: int, row: int) -> int:
    """``Fa2InvSum_v1`` (prims.py:440): ``(sum == ±0 || NaN) ? 1.0 : rcp.approx.ftz.f32(sum)`` (``mufu_rcp_bits``)."""
    s &= _M32
    word = _rg._table_word(row, fa2_inv_sum_index(s))
    if (s & 0x7FFFFFFF) == 0 or _is_nan(s):
        return F32_ONE
    sign = s & _SIGN
    e = (s >> 23) & 0xFF
    if e == 0xFF:
        return sign
    if e == 0:
        return sign | _EXP
    re = ((word >> 23) & 0xFF) - (e - 127)
    if re <= 0:
        return sign
    if re >= 0xFF:
        return sign | _EXP
    return sign | (re << 23) | (word & _MANT)


TableRowMismatch = _rg.TableRowMismatch

# --- the gate set (family.rs, gate for gate) -----------------------------------------------------------------------

#: ``family.rs`` ``GATES``: (name, arity, arg widths, evaluator).  Order = ``Gate::id`` (name order); the two source
#: gates carry the ``in``/``weight`` sources of ``gateset.rs``.
_GATES: tuple[tuple[str, int, tuple[int, ...], Any], ...] = (
    ("f32_add_ftz", 2, (32, 32), lambda a: f32_add_ftz(a[0], a[1])),
    ("f32_fma_ftz", 3, (32, 32, 32), lambda a: f32_fma_ftz(a[0], a[1], a[2])),
    ("f32_fma_sub_ftz", 3, (32, 32, 32), lambda a: f32_fma_sub_ftz(a[0], a[1], a[2])),
    ("f32_max_ftz", 2, (32, 32), lambda a: f32_max_ftz(a[0], a[1])),
    ("f32_mul_ftz", 2, (32, 32), lambda a: f32_mul_ftz(a[0], a[1])),
    ("f32_sub_ftz", 2, (32, 32), lambda a: f32_sub_ftz(a[0], a[1])),
    ("fa2_inv_sum", 2, (32, 64), lambda a: fa2_inv_sum_tab(a[0], a[1])),
    ("guard_neg_inf_zero", 1, (32,), lambda a: guard_neg_inf_zero(a[0])),
    ("in", 0, (), None),
    ("mufu_ex2_ftz", 2, (32, 64), lambda a: mufu_ex2_tab(a[0], a[1])),
    ("weight", 0, (), None),
)
GATE_ID = {name: i for i, (name, _, _, _) in enumerate(_GATES)}
_GATE_BY_NAME = {name: (arity, widths, fn) for name, arity, widths, fn in _GATES}


def make_ftz_gate_set() -> GateSet:
    """``veritor.ftz-f32@1`` as ``family.rs`` spells it; ``.digest`` must equal :data:`FTZ_DIGEST`."""
    gates = []
    for name, arity, widths, fn in _GATES:
        if name == "in":
            gates.append(Gate(name, 0, 64, replay_cost=0, proof_cost=1, source=INPUT_SOURCE))
        elif name == "weight":
            gates.append(Gate(name, 0, 64, replay_cost=0, proof_cost=1, source=WEIGHT_SOURCE))
        elif widths == (32,) * arity:
            gates.append(Gate(name, arity, 32, replay_cost=1, proof_cost=1, evaluate=fn))
        else:
            gates.append(Gate(name, arity, 32, replay_cost=1, proof_cost=1, arg_widths=widths, evaluate=fn))
    gs = GateSet(gates, name=FTZ_NAME, version=FTZ_VERSION)
    if str(gs.digest) != FTZ_DIGEST:
        raise RuntimeError(f"ftz-f32 gate set digest {gs.digest} is not the pinned {FTZ_DIGEST} (family.rs)")
    return gs


def evaluate(gate: str, args: Sequence[int]) -> int | None:
    """``family::evaluate``: the gate's word on ``args``, ``None`` when the relation rejects the arguments (a table
    row of the wrong index, an out-of-range word, a source gate)."""
    spec = _GATE_BY_NAME.get(gate)
    if spec is None or spec[2] is None or len(args) != spec[0]:
        return None
    for a, w in zip(args, spec[1], strict=True):
        if not 0 <= a < (1 << w):
            return None
    try:
        return spec[2](list(args))
    except TableRowMismatch:
        return None


# --- kinds -----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KindSpec:
    """One FTZ kind: registered primitive, gate, per-operand widths, and where the registry states the rule."""

    kind: str  # ``fn|form`` as the selection spells it
    prim: str  # registered primitive name (``verity_vllm.registry.prims``)
    gate: str  # gate of ``veritor.ftz-f32@1``
    in_widths: tuple[int, ...]
    out_width: int = 32
    rule: str = ""  # the registered rule, quoted
    source: str = ""  # ``prims.py:<line>``
    tiny: bool = True  # sampled by attempt 6's TINY selection

    @property
    def family(self) -> str:
        return FTZ_FAMILY

    @property
    def arity(self) -> int:
        return len(self.in_widths)

    @property
    def table(self) -> str | None:
        return {"mufu_ex2_ftz": "mufu/ex2", "fa2_inv_sum": "mufu/rcp"}.get(self.gate)


def _k(fn: str, form: str, gate: str, widths: tuple[int, ...], rule: str, line: int, *, tiny: bool = True) -> KindSpec:
    return KindSpec(f"{fn}_v1|{form}", fn, gate, widths, 32, rule, f"verity_vllm/registry/prims.py:{line}", tiny)


_MUL_RULE = "ftz(RN(ftz(a) * ftz(b)))"
KINDS: dict[str, KindSpec] = {
    s.kind: s
    for s in (
        _k("F32MulFtz", "batch", "f32_mul_ftz", (32, 32), _MUL_RULE, 248),
        _k("F32MulFtz", "primitive", "f32_mul_ftz", (32, 32), _MUL_RULE, 248),
        _k("F32AddFtz", "primitive", "f32_add_ftz", (32, 32), "ftz(RN(ftz(a) + ftz(b)))", 236),
        _k("F32FmaSubFtz", "primitive", "f32_fma_sub_ftz", (32, 32, 32), "ftz(RN(ftz(a) * ftz(b) - ftz(c)))", 260),
        _k("F32Max", "primitive", "f32_max_ftz", (32, 32), "ftz(a) > ftz(b) ? ftz(a) : ftz(b)", 266),
        _k("GuardNegInfZero", "primitive", "guard_neg_inf_zero", (32,), "m == -inf ? +0 : m", 272),
        _k("MufuEx2Ftz", "primitive", "mufu_ex2_ftz", (32, 64), "ex2.approx.ftz.f32: T[j]*2^n (mufu_ex2_bits)", 426),
        _k("Fa2InvSum", "primitive", "fa2_inv_sum", (32, 64), "(sum==±0||NaN) ? 1.0 : rcp.approx.ftz.f32(sum)", 440),
        # registered siblings of the family that the TINY selection did not sample
        _k("F32SubFtz", "primitive", "f32_sub_ftz", (32, 32), "ftz(RN(ftz(a) - ftz(b)))", 242, tiny=False),
        _k("F32FmaFtz", "primitive", "f32_fma_ftz", (32, 32, 32), "ftz(RN(ftz(a) * ftz(b) + ftz(c)))", 254, tiny=False),
    )
}

# --- the family in this process ------------------------------------------------------------------------------------

_installed = False


def ensure_families() -> None:
    """INTERIM (the CK-ELEM shape): make ``veritor.interface`` resolve ``veritor.ftz-f32@1`` at width 32 in THIS
    process until E7 registers the family in ``typed_obligation.py``.  ``validate`` checks ``family.digest`` against
    the gate set's, so the tie to the guest is the pinned digest either way.  Idempotent."""

    global _installed
    if _installed:
        return
    if _to.family_gate_set(FTZ_FAMILY, STATEMENT_WIDTH) is None:
        original = _to.family_gate_set

        def patched(family_id: str, width: int):
            if family_id == FTZ_FAMILY and width == STATEMENT_WIDTH:
                return make_ftz_gate_set()
            return original(family_id, width)

        _to.family_gate_set = patched  # type: ignore[assignment]
        if FTZ_FAMILY not in _to.SUPPORTED_FAMILIES:
            _to.SUPPORTED_FAMILIES = tuple(_to.SUPPORTED_FAMILIES) + (FTZ_FAMILY,)  # type: ignore[assignment]
        import veritor.interface as _iface
        import veritor.interface.lowering as _lowering
        import veritor.interface.vectors as _vectors

        for mod in (_iface, _lowering, _vectors, _expected):
            if hasattr(mod, "family_gate_set"):
                mod.family_gate_set = patched  # type: ignore[attr-defined]
            if hasattr(mod, "SUPPORTED_FAMILIES"):
                mod.SUPPORTED_FAMILIES = _to.SUPPORTED_FAMILIES  # type: ignore[attr-defined]
    _installed = True


def family_gate_set() -> GateSet:
    ensure_families()
    gs = _to.family_gate_set(FTZ_FAMILY, STATEMENT_WIDTH)
    if gs is None:
        raise RuntimeError(f"{FTZ_FAMILY} does not resolve at width {STATEMENT_WIDTH}")
    return gs


def family_of() -> Family:
    return Family(FTZ_FAMILY, str(family_gate_set().digest), STATEMENT_WIDTH)


def kind_subcircuit(spec: KindSpec) -> Subcircuit:
    """The one-gate body of ``spec``: incoming values ``0 .. arity`` (the operands in the primitive's signature
    order, ``Value<w_i>`` each), gate 0 = ``spec.gate`` over them, outgoing = gate 0."""

    gates = [(spec.gate, [(IN, i) for i in range(spec.arity)])]
    return make_subcircuit(list(spec.in_widths), gates, [0], Definition(spec.prim, (), spec.kind))


# --- resolving a located VU ----------------------------------------------------------------------------------------


class WordResolver(Protocol):
    """Where the record commits a gate's word.  ``word(request_id, g)`` is the committed word of gate ``g`` of the
    pinned program under request ``request_id`` (boundary, weights, or E2's interior trees) or ``None``;
    ``table_row(table, index)`` is the committed 64-bit row ``index`` of the named MUFU table or ``None``."""

    def word(self, request_id: str, g: int) -> CommittedWord | None: ...

    def table_row(self, table: str, index: int) -> CommittedWord | None: ...


@dataclass(frozen=True, slots=True)
class ResolvedUnit:
    """A located VU whose operand and output gates are committed words (table kinds: the row as second operand)."""

    vu_index: int
    kind: str
    spec: KindSpec
    gate: int
    root: str
    request_id: str
    inputs: tuple[CommittedWord, ...]
    output: CommittedWord
    replay_unit: int

    @property
    def target(self) -> str:
        return f"{self.root}/{self.spec.prim}@{self.gate}"


def _describe_gate(program: Any, g: int) -> str:
    try:
        gate = program.circuit.gate(g)
        return f"gate {g} ({gate.prim.name}, operands {tuple(gate.operands)})"
    except Exception:  # noqa: BLE001
        return f"gate {g}"


def _words_of(opening: Any) -> np.ndarray:
    dtype = str(opening.dtype)
    if dtype in ("float32", "uint32", "int32"):
        return np.frombuffer(opening.value, dtype="<u4")
    if dtype in ("uint64", "int64"):
        return np.frombuffer(opening.value, dtype="<u8")
    if dtype in ("bfloat16", "float16", "uint16", "int16"):
        return np.frombuffer(opening.value, dtype="<u2")
    raise Refusal(f"leaf dtype {dtype!r}: no word layout for this family")


def _word_width(opening: Any) -> int:
    return int(_words_of(opening).dtype.itemsize) * 8


def resolve(located: Mapping[str, Any], record: Any, resolver: WordResolver | None = None) -> ResolvedUnit:
    """A located VU (``kind``, ``vu_index``, ``gate``, ``root``, ``request_id``, ``replay_unit``) -> its committed
    operand and output words, or :class:`Refusal` naming exactly what the record does not commit.

    ``record`` is the CK-ELEM ``RecordHandle`` (``program``, ``locator``); ``resolver`` defaults to
    ``record.ftz`` when the record offers one, else to :class:`ElemWords` over its boundary/weights trees."""

    kind = str(located["kind"])
    spec = KINDS.get(kind)
    if spec is None:
        raise Refusal(f"{kind}: not a kind of {FTZ_FAMILY}")
    g = int(located["gate"])
    program = record.program
    try:
        gate = program.circuit.gate(g)
    except Exception as e:  # noqa: BLE001
        raise Refusal(f"gate {g}: not in the pinned program ({e})") from e
    if gate.prim.name != spec.prim:
        raise Refusal(f"gate {g} is {gate.prim.name}, not {spec.prim} ({kind})")
    operands = tuple(int(o) for o in gate.operands)
    n_words = spec.arity - (1 if spec.table else 0)
    if len(operands) != n_words:
        raise Refusal(f"gate {g}: {len(operands)} operands, {spec.prim} takes {n_words}")
    request_id = str(located["request_id"])
    if resolver is None:
        resolver = getattr(record, "ftz", None) or ElemWords(record)
    words: list[CommittedWord] = []
    for o in operands:
        w = resolver.word(request_id, o)
        if w is None:
            raise Refusal(f"operand {_describe_gate(program, o)} of gate {g}: not a committed word of the record")
        if w.width != 32:
            raise Refusal(f"operand {_describe_gate(program, o)} of gate {g}: a {w.width}-bit word, {spec.prim} reads f32")
        words.append(w)
    out = resolver.word(request_id, g)
    if out is None:
        raise Refusal(f"output {_describe_gate(program, g)}: not a committed word of the record")
    if out.width != 32:
        raise Refusal(f"output {_describe_gate(program, g)}: a {out.width}-bit word, {spec.prim} yields f32")
    if spec.table:
        # the row index is a function of the operand's committed value: read it through the record
        value = int(_words_of(record.open_leaf(words[0].leaf))[words[0].flat])
        index = mufu_ex2_index(value) if spec.gate == "mufu_ex2_ftz" else fa2_inv_sum_index(value)
        row = resolver.table_row(spec.table, index)
        if row is None:
            raise Refusal(f"table {spec.table} row {index}: no committed table tree in the record (E6/E7: pin the MUFU table roots)")
        if row.width != 64:
            raise Refusal(f"table {spec.table} row {index}: a {row.width}-bit word, the row is (index << 32) | word")
        words.append(row)
    return ResolvedUnit(
        int(located.get("vu_index", -1)), kind, spec, g, str(located.get("root", "")), request_id,
        tuple(words), out, int(located.get("replay_unit", -1)),
    )


class ElemWords:
    """The default resolver: CK-ELEM's boundary/weights locator (16-bit BF16 words -- an FTZ kind will refuse them by
    width, which is the exact reason: these kinds read interior f32 words E2 commits) and no table trees."""

    def __init__(self, record: Any) -> None:
        self._record = record

    def word(self, request_id: str, g: int) -> CommittedWord | None:
        elem = getattr(self._record, "elem", None)
        return elem().word(request_id, g) if callable(elem) else None

    def table_row(self, table: str, index: int) -> CommittedWord | None:
        return None


def expectation(located: Mapping[str, Any], record: Any, resolver: WordResolver | None = None, *, label: str = "") -> ExpectedObligation:
    """E3's expected statement for a located VU: operand references, output reference, body digest."""

    u = resolve(located, record, resolver)
    sub = kind_subcircuit(u.spec)
    return ExpectedObligation(
        tuple(w.expected() for w in u.inputs), (u.output.expected(),), sub.digest, (None,), {},
        label or f"vu={u.vu_index} {u.kind}",
    )


# --- building the typed statement ----------------------------------------------------------------------------------


@dataclass
class Built:
    obj: TypedObligationSet
    units: tuple[ResolvedUnit, ...]
    openings: dict[str, Any]
    statement: Any
    statement_bytes: bytes
    witness_bytes: bytes
    record: dict[str, Any]
    family: str = FTZ_FAMILY

    @property
    def expectations(self) -> list[ExpectedObligation]:
        return [
            ExpectedObligation(tuple(w.expected() for w in u.inputs), (u.output.expected(),), kind_subcircuit(u.spec).digest, (None,), {}, f"vu={u.vu_index} {u.kind}")
            for u in self.units
        ]


def _type_of(opening: Any) -> Type:
    ty: Type = Bits(_word_width(opening))
    for extent in reversed(opening.shape):
        ty = Array(ty, int(extent))
    return ty


def build_many(units: Sequence[ResolvedUnit], record: Any, *, name: str = "ck-ftz", authenticate: bool = True) -> Built:
    """One typed statement over ``units``: leaves/references deduplicated, obligations keyed as the exporter keys
    them, openings authenticated against ``commitments.json`` before any word is read (the CK-ELEM shape)."""

    from veritor.interface.lowering import statement_bytes_v3, to_statement_v3
    from veritor.protocol.proofs.packed import encode_packed_witness

    from verity_vllm.typed_obligation_export import authenticate_opening

    if not units:
        raise Refusal("no units to build")
    family = family_of()
    openings: dict[str, Any] = {}
    roots: dict[str, Root] = {}
    references: dict[str, Reference] = {}
    subcircuits: dict[str, Subcircuit] = {}
    how: dict[str, str] = {}

    def take(cid: str):
        o = openings.get(cid)
        if o is None:
            o = record.open_leaf(cid)
            if authenticate:
                how[cid] = authenticate_opening(record.commitments, o)
            openings[cid] = o
            if o.tree not in roots:
                rec = record.commitments["trees"][o.tree]
                if bytes.fromhex(rec["root"]) != o.root or int(rec["leaf_count"]) != o.leaf_count:
                    raise Refusal(f"{o.tree}: opening root/count disagree with commitments.json")
                roots[o.tree] = Root(o.tree, _owner(o.tree), rec["ids_digest"], o.root.hex(), o.leaf_count, VLLM_FRAMING)
        return o

    def ref_of(w: CommittedWord) -> tuple[str, int]:
        o = take(w.leaf)
        if o.tree != w.tree:
            raise Refusal(f"{w.leaf}: committed in {o.tree}, located in {w.tree}")
        shape = tuple(int(x) for x in o.shape)
        if shape != w.shape:
            raise Refusal(f"{w.leaf}: opening shape {shape} != located shape {w.shape}")
        words = _words_of(o)
        if _word_width(o) != w.width:
            raise Refusal(f"{w.ref}: a {_word_width(o)}-bit leaf word, located as {w.width}-bit")
        if not 0 <= w.flat < len(words):
            raise Refusal(f"{w.ref}: word {w.flat} outside the leaf ({len(words)} words)")
        if w.ref not in references:
            references[w.ref] = Reference(w.ref, w.leaf, w.offset)
        return w.ref, int(words[w.flat])

    pending = []
    for u in units:
        sub = kind_subcircuit(u.spec)
        subcircuits[sub.digest] = sub
        inc, row = [], []
        for w in u.inputs:
            r, v = ref_of(w)
            inc.append(r)
            row.append(v)
        r_out, v_out = ref_of(u.output)
        row.append(v_out)
        trees = tuple(sorted({w.tree for w in u.inputs} | {u.output.tree}))
        pending.append((u.request_id, u.target, sub, tuple(inc), (r_out,), trees, row, max(u.replay_unit, 0)))
    ordinals: dict[str, int] = {}
    built: list[tuple[Obligation, list[int]]] = []
    for session, target, sub, inc, out, trees, row, ru in sorted(pending, key=lambda p: (p[0], p[1])):
        unit = ordinals.get(session, 0)
        ordinals[session] = unit + 1
        binding = _binding(session, record.profile_id, unit, ru)
        built.append((Obligation(binding, sub.digest, trees, Boundary(inc, out), DeclaredInterface(inc, out), Target("call", target)), row))
    built.sort(key=lambda b: (b[0].binding.session, b[0].binding.compiled, b[0].binding.unit))
    leaves = [Leaf(o.canonical_id, o.tree, o.index, o.index, _type_of(o)) for o in openings.values()]
    obj = TypedObligationSet(
        family, (),
        tuple(roots[t] for t in sorted(roots)),
        tuple(sorted(leaves, key=lambda v: v.id)),
        tuple(sorted(references.values(), key=lambda r: r.ref)),
        tuple(sorted(subcircuits.values(), key=lambda s: s.digest)),
        tuple(o for o, _ in built),
    )
    ensure_families()
    validate(obj)
    statement_v3 = to_statement_v3(obj)
    statement = statement_v3.expand()
    st_bytes = statement_bytes_v3(obj)
    witness = _packed_witness(openings.values(), statement)
    proof = encode_packed_witness(witness)
    record_json = _record_json(obj, [row for _, row in built], proof, openings, how, units, name)
    return Built(obj, tuple(units), openings, statement, st_bytes, proof, record_json)


def _owner(tree: str) -> int:
    """Tree owner: CK-ELEM's mapping plus E2's interior trees ``interior/<request>/<ru>`` (owner = the RU)."""
    parts = tree.split("/")
    if len(parts) == 3 and parts[0] in ("interior", "ru"):
        return int(parts[2])
    if parts[0] in ("table", "mufu", "literal"):
        return -2  # a pinned table / literal tree is request-independent, like the weights (owner -2; validate allows -2, -1, RU)
        # [E7, NIGHT G1, out-of-lane one-liner flagged to the integrator (E4 absent tonight): prescribed-literal/v1's literal/w=<w> trees]
    return _owner_of(tree)


def build(located: Mapping[str, Any], record: Any, resolver: WordResolver | None = None, *, name: str = "ck-ftz") -> Built:
    """``build(located_unit, record) -> statement`` for ONE located VU (the registry's builder signature)."""
    return build_many([resolve(located, record, resolver)], record, name=name)


def _record_json(obj, values, proof, openings, how, units, name) -> dict[str, Any]:
    from veritor.interface.lowering import statement_bytes_v3, to_statement
    from veritor.interface.typed_obligation import canonical_bytes, object_digest
    from veritor.protocol.proofs.wire import encode_statement, statement_digest

    st2 = to_statement(obj)
    raw2 = encode_statement(st2)
    raw3 = statement_bytes_v3(obj)
    kinds = sorted({u.kind for u in units})
    return {
        "name": name,
        "description": f"CK-FTZ scalar statement: {', '.join(kinds)} ({len(units)} units) over the record's committed f32 words",
        "tags": ["ck-ftz", "ftz", "scalar", "packed-leaf", "packed-word", "witness", "composed", *kinds],
        "object": obj.to_json(),
        "canonical": canonical_bytes(obj.to_json()).decode("utf-8"),
        "digest": object_digest(obj.to_json()),
        "verdict": "accept",
        "statement": raw2.hex(),
        # [E7, D25 2026-09-12 00:50Z] the digest of the statement this record EMITS and the proof binds (statement.bin = the v3 bytes below;
        # public h_L = sha256(statement.bin); reverify.py / steps.proof_record_consistency require statement_digest == sha256(statement.bin)).
        # The v2 column ("statement") stays for the format-2 readers (rec_p1_obligations / run_composed assert it); its digest is named as such.
        "statement_digest": statement_digest(raw3).hex(),
        "statement_v2_digest": statement_digest(raw2).hex(),
        "statement_v3": raw3.hex(),
        "statement_format": 3,
        "values": values,
        "witness": proof.hex(),
        "provenance": {
            "builder": "e2e.coverage_v1.ck_ftz",
            "family": FTZ_FAMILY,
            "family_digest": FTZ_DIGEST,
            "units": [{"vu_index": u.vu_index, "kind": u.kind, "gate": u.gate, "target": u.target,
                       "inputs": [w.ref for w in u.inputs], "output": u.output.ref} for u in units],
            "authentication": how,
            "openings": {cid: {"tree": o.tree, "index": o.index, "bytes": len(o.value)} for cid, o in openings.items()},
        },
    }


def verify_transparent(built: Built) -> str:
    """The transparent backend's verdict on statement + witness at the pinned gate set (LABEL: the guest's Python
    mirror, a local check of relation and openings -- NOT the approved checker's proof)."""

    from veritor.protocol.messages import Reject
    from veritor.protocol.proofs import TransparentBackend

    try:
        TransparentBackend(family_gate_set()).verify(built.statement, built.witness_bytes)
        return "accept"
    except Reject as e:
        return f"reject:{e.code.name}"


# --- conformance vectors (statement.rs: veritor/ftz-conformance/v1) -------------------------------------------------

CONFORMANCE_MAGIC = b"VFTZ"
CONFORMANCE_VERSION = 1


def _parse_word(x: Any) -> int:
    if isinstance(x, str):
        return int(x, 16) if x.lower().startswith("0x") else int(x, 0)
    return int(x)


@dataclass(frozen=True, slots=True)
class Vector:
    """One conformance vector as ``ftz_vectors/*.json`` spells it."""

    id: str
    gate: str
    args: tuple[int, ...]
    claimed: int
    expect: str  # "accept" | "reject"
    note: str = ""
    source: str = ""


def load_vectors(path: Path | str) -> list[Vector]:
    doc = json.loads(Path(path).read_text())
    if doc.get("family") != FTZ_FAMILY:
        raise ValueError(f"{path}: family {doc.get('family')!r} is not {FTZ_FAMILY}")
    if doc.get("family_digest") not in (None, FTZ_DIGEST):
        raise ValueError(f"{path}: digest {doc.get('family_digest')} is not the pinned {FTZ_DIGEST}")
    out = []
    for v in doc["vectors"]:
        out.append(Vector(str(v["id"]), str(v["gate"]), tuple(_parse_word(a) for a in v["args"]), _parse_word(v["claimed"]),
                          str(v.get("expect", "accept")), str(v.get("note", "")), str(v.get("source", ""))))
    return out


def encode_conformance(units: Iterable[tuple[str, Sequence[int], int]]) -> bytes:
    """``statement::encode``: ``VFTZ | 1 | digest | count u32 | (gate u8 | arity u8 | args u64.. | claimed u64)*``."""
    body = bytearray()
    n = 0
    for gate, args, claimed in units:
        gid = GATE_ID.get(gate)
        if gid is None or _GATE_BY_NAME[gate][2] is None:
            raise ValueError(f"{gate}: not a relation gate of {FTZ_FAMILY}")
        body += struct.pack("<BB", gid, len(args))
        for a in args:
            body += struct.pack("<Q", int(a))
        body += struct.pack("<Q", int(claimed))
        n += 1
    return CONFORMANCE_MAGIC + bytes([CONFORMANCE_VERSION]) + bytes.fromhex(FTZ_DIGEST) + struct.pack("<I", n) + bytes(body)


def decode_conformance(data: bytes) -> list[tuple[str, tuple[int, ...], int]] | None:
    """``statement::decode``: ``None`` when the bytes are not a well-formed statement of this family."""
    if len(data) < 41 or data[:4] != CONFORMANCE_MAGIC or data[4] != CONFORMANCE_VERSION or data[5:37].hex() != FTZ_DIGEST:
        return None
    (count,) = struct.unpack_from("<I", data, 37)
    pos = 41
    out = []
    names = [g[0] for g in _GATES]
    for _ in range(count):
        if pos + 2 > len(data):
            return None
        gid, arity = data[pos], data[pos + 1]
        pos += 2
        if gid >= len(names) or _GATE_BY_NAME[names[gid]][2] is None or arity != _GATE_BY_NAME[names[gid]][0]:
            return None
        if pos + 8 * (arity + 1) > len(data):
            return None
        words = struct.unpack_from(f"<{arity + 1}Q", data, pos)
        pos += 8 * (arity + 1)
        out.append((names[gid], tuple(words[:-1]), words[-1]))
    return out if pos == len(data) else None


def public_values(statement: bytes) -> bytes:
    """``statement::check(...).public_values()``: ``sha256(statement) || all || count u32 || verdicts``."""
    import hashlib

    units = decode_conformance(statement)
    verdicts = b"" if units is None else bytes(int(evaluate(g, a) == c) for g, a, c in units)
    all_ok = int(bool(verdicts) and all(verdicts))
    return hashlib.sha256(statement).digest() + bytes([all_ok]) + struct.pack("<I", len(verdicts)) + verdicts


def check_vectors(vectors: Sequence[Vector]) -> dict[str, Any]:
    """Every vector's checker verdict (``evaluate(gate, args) == claimed``) against its ``expect``; the summary the
    integration test and the report read."""
    rows, wrong = [], []
    for v in vectors:
        got = evaluate(v.gate, v.args)
        verdict = "accept" if got == v.claimed else "reject"
        rows.append({"id": v.id, "gate": v.gate, "verdict": verdict, "expect": v.expect, "evaluated": None if got is None else f"0x{got:08X}"})
        if verdict != v.expect:
            wrong.append(v.id)
    return {
        "count": len(vectors),
        "positives": sum(v.expect == "accept" for v in vectors),
        "negatives": sum(v.expect == "reject" for v in vectors),
        "mismatches": wrong,
        "rows": rows,
    }
