"""The per-kind one-gate obligation builders (elementwise + FP32 scalar kinds of the approved guest) -- a *contribution* to the one
registry ``e2e.coverage_v1.builders.BUILDERS`` -- and the program-agnostic record handle + batch packaging (E7, EOD 2026-09-11).

(Integrator 22:15Z: this module was ``coverage_v1/builders.py`` on ``contract/ck-elem``; renamed at merge because A5/6's
``builders.py`` landed first.  A3 22:20Z unified them: ``builders._registry()`` folds this module's ``BUILDERS`` into the
``_Registry`` the dispatch reads, so ``verifier_v3`` and the prover consult ``coverage_v1.builders.BUILDERS`` only.)

One signature for every kind (the directive's station 1): ``build(located_unit, record) -> statement``, where ``located_unit`` is the
verifier's located VU (``e2e/verifier_v3/locator.py``: ``vu_index``, ``request_id``, ``kind``, ``gate_interval``) and ``record`` a
:class:`~e2e.coverage_v1.ck_elem.RecordHandle` -- or, since EOD 2026-09-11, a :class:`GenericRecord`: the same handle with the value
positions supplied by **word oracles** (``word(request_id, gate) -> CommittedWord | None``) and the leaf openings by an **opener**
(``open(canonical_id) -> verity_vllm.commit.Opening``), so that any registered ``verity.ir`` program whose committed values are laid out
in ``verity-vllm/leaf/v1`` trees (E8's CORE circuit; E2's ``interior/<req>/<step>`` trees for TINY) reaches the same builder, the same
statement format (typed-obligation/v0 -> format 3) and the same approved guest.  A builder also derives the *expected statement* of a
located VU for the verifier (``expectation(located_unit, record) -> ExpectedObligation``; the prover's object is never read) and names
its family.

Registration is by kind (``fn|form``).  EVERY kind of ``ck_elem.KINDS`` is registered (11 kinds: the four boundary-resolvable
elementwise kinds of attempt 6 AND the seven whose operand or output word is an operator interior).  The registry says *a builder
exists*; whether a given VU builds is the oracles' answer at build time: a word no committed leaf holds is a named
:class:`~e2e.coverage_v1.ck_elem.Refusal` (never a silent skip, never a proof of a different unit).  W-A9-12: a kind with a registered
builder is never labelled unsupported by the verifier's derivation -- the deriver asks the builder, and the builder refuses by name.

Batch packaging (interface 6): :func:`package_batches` turns the located VUs into proof-sized statements, one family per statement,
``max_units`` (default 64) units each; a :class:`Batch` names the VU identities it covers, so ``proofs.chunks`` and ``per_vu`` are
two separate lists (never "|Q| = proofs + input equations").
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover
    from veritor.interface.expected import ExpectedObligation

    from e2e.coverage_v1.ck_elem import Built, CommittedWord, ResolvedUnit

__all__ = [
    "Builder", "BUILDERS", "INTERIOR_BUILDERS", "EXTRA_BUILDERS", "CALL_BUILDERS", "ALL_BUILDERS", "ELEM_FAMILIES", "EXTRA_KINDS", "EXTRA_FAMILIES", "ALL_FAMILIES",
    "BOUNDARY_KINDS", "INTERIOR_KINDS_REGISTERED", "CALL_KINDS", "CALL_FAMILIES", "CallSpec", "TC_FAMILY", "ZERO32_PRIM", "resolve_call_unit", "unit_subcircuit",
    "PV_KIND", "PV_PRIM", "PV_BUILDERS", "PvSpec", "E7_KINDS", "resolve_pv_unit",
    "TABLE_KINDS", "TABLE_FAMILIES", "TABLE_BUILDERS", "TableSpec", "TABLE_TREE_PREFIX", "TABLE_OWNER", "resolve_table_unit", "table_leaf_name", "table_words",
    "dispatch_expectation", "dispatch_kind_program", "families_of", "kinds_of_family", "kind_spec", "resolve_unit",
    "located_mapping", "expectation_unit", "ALIAS_RULE", "AliasSpec", "alias_unit", "alias_digests", "is_aliased_body",
    "alias_map", "expected_body", "PhantomRecord", "phantom_unit", "phantom_widths", "CENSUS_PROFILE", "CENSUS_REQUEST",
    "WordOracle", "OracleChain", "LocateOracle", "GenericRecord", "open_record", "tree_records", "owner_of", "leaf_word_width", "UNKNOWN_SHAPE",
    "build_batch", "package_batches", "Batch", "Packaged", "batches_document", "DEFAULT_MAX_UNITS", "MAX_UNITS_BY_FAMILY",
]

# [W-A9-11, integrator 23:45Z] This module is on the PROVER's import path (``selective_prover`` -> ``coverage_v1.builders`` -> here) and the
# prover runs in the pinned vLLM environment (Python 3.11, ``PYTHONPATH=.`` only).  Nothing under ``src/veritor`` may be imported at
# module level here: ``ck_elem`` (which imports ``veritor.interface.*`` at import time) and ``veritor.interface.expected`` are imported
# lazily, inside the verifier-facing methods (``build`` / ``expectation`` / ``subcircuit_digest``) only.  The prover needs only the kind
# names and families of the registered builders (R18: which selected kinds have a checker) -- ``ELEM_FAMILIES`` below, pinned here and
# asserted equal to ``ck_elem.KINDS[k].family`` by ``tests/contract/a3_sequence`` (verifier side).

ELEM_FAMILY = "veritor.elem-bf16@1"
REDUCE_FAMILY = "veritor.reduce-mufu@1"

#: the four kinds whose whole boundary today's record commits (attempt 6 proved them: 208 VUs / 4 proof objects)
BOUNDARY_KINDS: tuple[str, ...] = ("SiluMulBf16_v1|batch", "Bf16AddF2fp_v1|batch", "RopeOut_v1|batch", "RopeOutAdd_v1|batch")
#: the eight kinds with an approved checker whose operand/output word is an operator INTERIOR -- buildable exactly when a word oracle
#: resolves that word (E2's ``interior/<req>/<step>`` trees for TINY; E8's own trees for CORE)
INTERIOR_KINDS_REGISTERED: tuple[str, ...] = ("Bf16ToF32_v1|batch", "F2fpBf16_v1|batch", "F2fpBf16_v1|primitive", "F32Mul_v1|batch",
                                              "F32Mul_v1|primitive", "F32Add_v1|primitive", "F32Div_v1|primitive", "F32Fma_v1|primitive")

#: kind -> interface family of every one-gate kind of ``ck_elem.KINDS`` (asserted equal to ``ck_elem.KINDS[k].family`` by
#: ``tests/contract/a3_sequence``); the prover reads this table without importing ``ck_elem``
ELEM_FAMILIES: dict[str, str] = {
    "SiluMulBf16_v1|batch": ELEM_FAMILY,
    "Bf16AddF2fp_v1|batch": ELEM_FAMILY,
    "RopeOut_v1|batch": REDUCE_FAMILY,
    "RopeOutAdd_v1|batch": REDUCE_FAMILY,
    "Bf16ToF32_v1|batch": REDUCE_FAMILY,
    "F2fpBf16_v1|batch": REDUCE_FAMILY,
    "F32Mul_v1|batch": REDUCE_FAMILY,
    "F32Mul_v1|primitive": REDUCE_FAMILY,
    "F32Add_v1|primitive": REDUCE_FAMILY,
    "F32Div_v1|primitive": REDUCE_FAMILY,
    "F32Fma_v1|primitive": REDUCE_FAMILY,
}

#: [E7, EOD 2026-09-11] one-gate kinds the approved guest checks that ``ck_elem.KINDS`` (frozen at CK-ELEM) does not list, as
#: ``(kind, prim, gate, family, arity, in_width, out_width, note)``; resolved by :func:`resolve_unit` with the same rule as ``ck_elem.resolve``
#: (``ck_elem`` itself is not modified).  ``F2fpBf16_v1|primitive`` (TINY: the attention ``p -> pb`` cast, 9 selected in attempt 6) is the
#: primitive form of the batch kind: the same ``F2fpBf16`` gate, ``f2fp_bf16`` in ``veritor.reduce-mufu@1`` (gateset.rs:213).
EXTRA_KINDS: tuple[tuple[str, str, str, str, int, int, int, str], ...] = (
    ("F2fpBf16_v1|primitive", "F2fpBf16", "f2fp_bf16", REDUCE_FAMILY, 1, 32, 16, "attention p (FP32 interior) -> pb (bf16 interior)"),
)
EXTRA_FAMILIES: dict[str, str] = {k[0]: k[3] for k in EXTRA_KINDS}

#: [E7, EOD 2026-09-11 23:20Z] the tensor-core family the accepted records' GEMM coordinates prove in (``typed_obligation_export.TC_FAMILY``)
TC_FAMILY = "veritor.tc-ampere-bf16@2"


@dataclass(frozen=True)
class CallSpec:
    """A registered CALL kind whose body is A2's ``GemmAccumulator<K>`` (``veritor.interface.programs.gemm_coordinate(k, opened_accumulator=True)``):
    ``b1.DotBf16{K}`` = ``ZERO32`` literal + ``K/16`` chained ``AmpereBF16TcDot16`` gates, the FP32 accumulator outgoing -- the same body the guest
    evaluates as ``tc_dot16_0`` + ``tc_dot16`` x (K/16 - 1) (``tc_total.rs``).  Shaped like ``ck_elem.KindSpec`` so :func:`build_batch` and
    ``ck_elem._record_json`` read it unchanged; ``subcircuit`` is the multi-gate body :func:`build_batch` prefers over ``kind_subcircuit``."""

    kind: str
    prim: str
    gate: str
    family: str
    arity: int
    in_width: int
    out_width: int
    note: str = ""
    k: int = 64

    @property
    def steps(self) -> int:
        return self.k // 16

    @property
    def subcircuit(self) -> Any:
        from veritor.interface.programs import gemm_coordinate

        return gemm_coordinate(self.k, opened_accumulator=True)


#: ``DotBf16_v1{K=64}|call`` (TINY: the per-key attention score ``s[h, c] = DotBf16(q, k[c])``, `b1.py:346`; E6 inventory 21,060 VUs): q slice
#: (rope port 0, committed bf16) x K row (rope port 1 at position c, committed bf16) -> FP32 score (interior: E2's ``interior/<req>/<step>`` tree).
#: Incoming ``0..64`` = x words, ``64..128`` = w words in ascending k (A2's GemmAccumulator order); outgoing = the last accumulator.
CALL_KINDS: dict[str, CallSpec] = {
    "DotBf16_v1{K=64}|call": CallSpec("DotBf16_v1{K=64}|call", "AmpereBF16TcDot16", "tc_dot16_0 + tc_dot16 x3 (GemmAccumulator K=64)", TC_FAMILY,
                                      128, 16, 32, "attention score: q (rope:0) . k[c] (rope:1) -> scores[h, c] (FP32 interior)", k=64),
}
CALL_FAMILIES: dict[str, str] = {k: s.family for k, s in CALL_KINDS.items()}
#: the literal ``b1.DotBf16`` starts its chain from (``verity.ir`` spelling of ``const(32, 0)``; ``Gate.prim.name``)
ZERO32_PRIM = "Const32[0x00000000]"

#: [E7, 23:40Z] ``AmpereBF16TcDot16_v1|batch`` (TINY: the PV step of the FA2 block, `b1.py:385`; E6 inventory 207,360 VUs -- the kind that decides V5):
#: one tensor-core gate over ``(acc: f32, 16 pb: bf16, 16 V: bf16) -> f32``.  ``acc`` is the previous PV output / the rescaled ``o_prev`` (FP32
#: interior word) -- body ``tc_dot16`` -- or the ``ZERO32`` literal (first block, first k-step) -- body ``tc_dot16_0`` (the guest's ``+0``
#: accumulator gate, no literal needed); the pb words are ``F2fpBf16`` outputs (bf16 interior), the V words the value rows (qkv-proj port 0,
#: committed).  A ``ZERO16`` PADDING literal among the pb words (the block's last k-step past the visible columns; 3,240 of the 207,360 VUs) is a
#: REFUSAL BY NAME: the interface admits an ``Input`` gate in a body only on the outgoing boundary (``E_INPUT_NOT_PROTECTED``, IR §E) and the
#: tc gate set has no zero-word gate, so that unit's relation is not expressible under the approved family today.
PV_KIND = "AmpereBF16TcDot16_v1|batch"
PV_PRIM = "AmpereBF16TcDot16"


@dataclass(frozen=True)
class PvSpec:
    """The per-unit spec of :data:`PV_KIND`: ``acc_zero`` when operand 0 is the ``ZERO32`` literal (body ``tc_dot16_0`` over 32 incoming bf16
    words) else the body ``tc_dot16`` over ``acc`` (32-bit incoming) + 32 bf16 words; ``subcircuit`` is that one-gate body."""

    acc_zero: bool = False
    kind: str = PV_KIND
    prim: str = PV_PRIM
    family: str = TC_FAMILY
    in_width: int = 16
    out_width: int = 32
    note: str = "FA2 PV step: o[h,b,ks,d] = tc_dot16(o[h,b,ks-1,d], pb[16], V[16]); first k-step of the first block: tc_dot16_0(pb, V)"

    @property
    def gate(self) -> str:
        return "tc_dot16_0" if self.acc_zero else "tc_dot16"

    @property
    def arity(self) -> int:
        return 32 if self.acc_zero else 33

    @property
    def in_widths(self) -> tuple[int, ...]:
        return tuple(([] if self.acc_zero else [32]) + [self.in_width] * 32)

    @property
    def subcircuit(self) -> Any:
        from veritor.interface.programs import IN, Definition, make_subcircuit

        incoming = ([] if self.acc_zero else [32]) + [self.in_width] * 32
        gates = [(self.gate, [(IN, i) for i in range(len(incoming))])]
        return make_subcircuit(incoming, gates, [0], Definition(self.prim, (("acc_zero", int(self.acc_zero)),), self.kind))


def _const_literal(gate: Any) -> tuple[int, int] | None:
    """``(width, value)`` when ``gate`` is a ``verity.ir`` constant literal (``Const<w>[0x...]``, no operands), else ``None``."""

    if gate.operands:
        return None
    name = str(getattr(gate.prim, "name", gate.prim.id))
    if not name.startswith("Const") or "[" not in name:
        return None
    try:
        width = int(name[5:name.index("[")])
        value = int(name[name.index("[") + 1:name.index("]")], 16)
    except ValueError:
        return None
    return width, value


#: [E7, 2026-09-12 00:15Z, D23] the MUFU TABLE kinds (TINY: 765 VUs -- ``RsqrtApprox`` 717, ``DivFullRcp`` 24, ``DivFullScaleA`` 12, ``MufuSqrtFtz`` 12):
#: one ``veritor.reduce-mufu@1`` gate over ``(x: f32 word, row: 64-bit table row)`` -> f32, the row being ``(index << 32) | T[index]`` of E2's
#: ``table-layout/v1`` tree ``table/<name>`` (``e2e.interior.tables``; root pinned at ``commitments.tables[<name>]``), ``index`` the registered
#: index function of the FIRST operand's committed VALUE (``veritor.core.reduce_gates.*_index``) -- the guest gate refuses a row whose high word
#: is not that index (``TableRowMismatch``), so the position -> index binding needs no extra statement.  ``DivFullScaleA`` reads no table (two
#: f32 words).  A record whose instance pins no ``commitments.tables`` REFUSES BY NAME (``TABLE_UNPINNED``); legacy records are unchanged (D12).
TABLE_TREE_PREFIX = "table/"
#: ``Root.owner`` of a pinned table tree: request-independent like the weights (-2; ``validate`` admits -2, -1, RU) -- E4's ``ck_ftz`` says the same
TABLE_OWNER = -2


@dataclass(frozen=True)
class TableSpec:
    """A :data:`TABLE_KINDS` row, shaped like ``ck_elem.KindSpec`` (+ ``table`` / ``index_fn`` / ``in_widths``): ``arity`` is the PRIMITIVE's
    (program) arity; the body takes ``in_widths`` = the operand words (32) then, for a table kind, the 64-bit row."""

    kind: str
    prim: str
    gate: str
    table: str | None  # E2's table name (``mufu/rsq`` ...) or None (no table row)
    index_fn: str | None  # ``veritor.core.reduce_gates`` function of the first operand's value -> row index
    arity: int
    note: str = ""
    family: str = REDUCE_FAMILY
    in_width: int = 32
    out_width: int = 32

    @property
    def in_widths(self) -> tuple[int, ...]:
        return (self.in_width,) * self.arity + ((64,) if self.table else ())

    def index_of(self, value: int) -> int:
        from veritor.core import reduce_gates

        return int(getattr(reduce_gates, str(self.index_fn))(int(value)))

    @property
    def subcircuit(self) -> Any:
        from veritor.interface.programs import IN, Definition, make_subcircuit

        widths = list(self.in_widths)
        return make_subcircuit(widths, [(self.gate, [(IN, i) for i in range(len(widths))])], [0], Definition(self.prim, (), self.kind))


TABLE_KINDS: dict[str, TableSpec] = {
    s.kind: s
    for s in (
        TableSpec("RsqrtApprox_v1|primitive", "RsqrtApprox", "rsqrt_approx", "mufu/rsq", "rsqrt_approx_index", 1,
                  "rms_norm: rsqrt.approx.ftz.f32 of the mean square (FP32 interior) through the MUFU.RSQ table row"),
        TableSpec("MufuSqrtFtz_v1|primitive", "MufuSqrtFtz", "mufu_sqrt_tab", "mufu/sqrt", "mufu_sqrt_index", 1,
                  "sqrt.approx.ftz.f32 through the MUFU.SQRT table row"),
        TableSpec("DivFullRcp_v1|primitive", "DivFullRcp", "div_full_rcp", "mufu/rcp", "div_full_rcp_index", 1,
                  "div.full.f32: range-scaled denominator -> MUFU.RCP table row"),
        TableSpec("DivFullScaleA_v1|primitive", "DivFullScaleA", "div_full_scale_a", None, None, 2,
                  "div.full.f32: numerator scaling by the denominator's range (two FP32 interior words, no table)"),
    )
}
TABLE_FAMILIES: dict[str, str] = {k: s.family for k, s in TABLE_KINDS.items()}


def table_of_kind(kind: str) -> tuple[str, Callable[[int], int]] | None:
    """[D23 ``table-layout/v1``] ``(table name, index_fn)`` of a TABLE kind -- E7's :data:`TABLE_KINDS` (``mufu/rsq|sqrt|rcp``) and E4's
    ``ck_ftz`` kinds with a table (``MufuEx2Ftz`` -> ``mufu/ex2`` via ``mufu_ex2_index``, ``Fa2InvSum`` -> ``mufu/rcp`` via
    ``fa2_inv_sum_index``); ``None`` for every other kind.  ONE spelling of "which row this unit reads" for the builder (``resolve``) and
    the verifier's expectation (``expected.py``): the row index is a function of the FIRST operand's committed value."""

    s = TABLE_KINDS.get(str(kind))
    if s is not None:
        return (str(s.table), s.index_of) if s.table else None
    try:
        from . import ck_ftz
    except ImportError:  # pragma: no cover -- prover side without src/veritor
        return None
    spec = ck_ftz.KINDS.get(str(kind))
    if spec is None or not spec.table:
        return None
    fn = ck_ftz.mufu_ex2_index if spec.gate == "mufu_ex2_ftz" else ck_ftz.fa2_inv_sum_index
    return str(spec.table), fn


def table_port(kind: str, value: int, tables: Any):
    """[D23 ``table-layout/v1``; E1 s3 finding 2026-09-12 15:50Z] The committed TABLE-ROW word a table kind's statement reads as its LAST
    incoming port, as an ``ExpectedValue``: the row ``index_fn(value)`` of the kind's MUFU table at E2's pinned position (``TableWords.table_row``
    -> ``table/<name>/chunk=<k>`` [flat]); ``None`` for a kind without a table.  ``value`` = the first operand's committed 32-bit word (the
    verifier reads it from the record's authenticated leaf; the prover from its own).  The guest checks the row's index half against the
    operand inside the body (``(index << 32) | T[index]``), so the port is part of the statement -- an expectation that omits it names a
    different statement (s3: prover ``1edf3e22…`` vs verifier ``18b206f5…``, same subcircuit).  Refuses ``TABLE_UNPINNED`` when the instance pins
    no such table."""

    tk = table_of_kind(kind)
    if tk is None:
        return None
    table, index_fn = tk
    if tables is None:
        raise _ck_elem().Refusal(f"{kind}: TABLE_UNPINNED: the statement reads table {table} and the instance pins no commitments.tables")
    index = int(index_fn(int(value) & 0xFFFFFFFF))
    row = tables.table_row(table, index)
    if row is None:
        raise _ck_elem().Refusal(f"{kind}: TABLE_UNPINNED: table {table} row {index} has no pinned tree (commitments.tables)")
    return row.expected()


def leaf_word_value(record: Any, leaf: str, flat: int, tree: str | None = None) -> int:
    """The committed word ``flat`` of leaf ``leaf`` as an int: through ``record.open_leaf`` (the builder's opening / the verifier's
    authenticated ``TreeCache`` over the capture) or, for a leaf of E2's ``interior/<req>/<step>`` tree the capture reader does not
    address, from the tree's shipped leaf file (:func:`interior_leaf_words`); the leaf's dtype fixes the word width."""

    import numpy as np

    if tree is not None and str(tree).startswith("interior/"):
        words = interior_leaf_words(record, str(tree), str(leaf))
        if words is not None:
            return int(words[int(flat)])
    opening = record.open_leaf(str(leaf))
    dtype = str(opening.dtype)
    width = _WORD_WIDTH_OF_DTYPE.get(dtype)
    if width is None:
        raise _ck_elem().Refusal(f"{leaf}: dtype {dtype!r} has no word layout")
    return int(np.frombuffer(opening.value, dtype=f"<u{width // 8}")[int(flat)])


def interior_leaf_words(record: Any, tree: str, leaf: str):
    """The words of leaf ``leaf`` of E2's committed tree ``interior/<req>/<step>`` as a numpy array, read from the record's shipped
    ``interior/<req>/<step>/leaf-<rank>.bin`` (the directory the record's capture reader was opened on, ``record.reader.run_dir``) and
    AUTHENTICATED against the index row's framed leaf hash (``tree.leaf_hash_of_row``: the same hash V3.trees_rehashed folded into the root it
    compared with ``commitments.json#interior_roots``); ``None`` when the record has no such directory (a handle without a reader), a
    ``Refusal`` by name when the bytes do not hash to the row."""

    import numpy as np
    from pathlib import Path

    reader = getattr(record, "reader", None)
    run_dir = getattr(reader, "run_dir", None)
    if run_dir is None:
        return None
    parts = str(tree).split("/")
    if len(parts) != 3:
        return None
    d = Path(run_dir) / "interior" / parts[1] / parts[2]
    if not (d / "index.json").exists():
        return None
    from e2e.interior import tree as _itree

    index = _itree.load_index(d)
    rows = list(index.get("leaves") or [])
    row = next(((i, r) for i, r in enumerate(rows) if str(r.get("canonical_id")) == str(leaf)), None)
    if row is None:
        raise _ck_elem().Refusal(f"{tree}: leaf {leaf} is not a leaf of the shipped index")
    i, r = row
    value = (d / str(r.get("file", f"leaf-{i:06d}.bin"))).read_bytes()
    if _itree.leaf_hash_of_row(r, value).hex() != str(r.get("sha256", "")):
        raise _ck_elem().Refusal(f"{tree}: leaf {leaf}: the shipped bytes do not hash to the index row (LEAF_HASH_MISMATCH)")
    width = _WORD_WIDTH_OF_DTYPE.get(str(r.get("dtype")))
    if width is None:
        raise _ck_elem().Refusal(f"{tree}: leaf {leaf}: dtype {r.get('dtype')!r} has no word layout")
    return np.frombuffer(value, dtype=f"<u{width // 8}")


_INTERIOR_INDEX_CACHE: dict[tuple[str, str], dict[str, tuple[int, dict[str, Any]]]] = {}


def interior_opening(record: Any, canonical_id: str):
    """[E7 oracles hook] The opening of a leaf of one of the record's ``interior/<req>/<step>`` trees found by its canonical id alone (the
    ``open_leaf(leaf)`` call E4's resolver makes for a table kind's operand value): the trees ``commitments.json`` names for the id's request
    (``r=<req>`` in the id) are searched by their shipped ``index.json``; the bytes are AUTHENTICATED against the row's framed leaf hash
    (:func:`interior_leaf_words`).  ``None`` when no shipped interior tree holds the leaf (the caller then raises the reader's own error)."""

    reader = getattr(record, "reader", None)
    run_dir = getattr(reader, "run_dir", None)
    if run_dir is None:
        return None
    from pathlib import Path

    from e2e.interior import tree as _itree

    cid = str(canonical_id)
    m = re.search(r"/r=([^/]+)/", cid)
    rid = m.group(1) if m else None
    trees = [t for t in tree_records(getattr(record, "commitments", None) or {}) if t.startswith("interior/") and (rid is None or t.split("/")[1] == rid)]
    for tree in trees:
        parts = tree.split("/")
        d = Path(run_dir) / "interior" / parts[1] / parts[2]
        if not (d / "index.json").exists():
            continue
        key = (str(run_dir), tree)
        rows = _INTERIOR_INDEX_CACHE.get(key)
        if rows is None:
            rows = _INTERIOR_INDEX_CACHE[key] = {str(r.get("canonical_id")): (i, dict(r)) for i, r in enumerate(_itree.load_index(d).get("leaves") or [])}
        hit = rows.get(cid)
        if hit is None:
            continue
        words = interior_leaf_words(record, tree, cid)  # authenticated against the row's sha256
        i, r = hit
        return _InteriorOpening(tree=tree, canonical_id=cid, index=int(r.get("rank", i)), dtype=str(r.get("dtype")),
                                shape=tuple(int(s) for s in (r.get("shape") or [len(words)])), value=words.tobytes())
    return None


@dataclass(frozen=True)
class _InteriorOpening:
    """The fields of ``verity_vllm.commit.Opening`` a builder reads (``dtype``, ``shape``, ``value`` bytes; ``tree`` / ``canonical_id`` / ``index``)."""

    tree: str
    canonical_id: str
    index: int
    dtype: str
    shape: tuple[int, ...]
    value: bytes


def verifier_oracle_chain(handle: Any):
    """[E7 oracles hook, for ``ck_elem.RecordHandle.elem``] The verifier-side handle's word oracle once ``handle.oracles`` is set (E9's
    ``coverage_inputs`` appends E3's ``PositionOracle.word``): the SAME chain :func:`open_record` wires on the prover side --
    ``LiteralOracle`` first (a ``Const`` operand gate -> the pinned literal tree's word, ``LITERAL_UNPINNED`` by name without a pin), then
    ``ElemLocator`` (weights + operator-boundary rows), then the handle's oracles (interior words at E3's positions).  The literal table is
    ``handle.literals`` when set (E9's ``steps.LiteralTable`` -- taken through its ``pin()`` -- or ours), else the ``commitments.literals`` block."""

    ck = _ck_elem()
    lt = record_literal_table(handle)
    chain: list[WordOracle] = [LiteralOracle(handle.program, lt), ck.ElemLocator(handle)]
    chain.extend(o if hasattr(o, "word") else _CallableOracle(o) for o in handle.oracles)
    return OracleChain(chain, weights_tree=ck.WEIGHTS_TREE)


@dataclass(frozen=True)
class _CallableOracle:
    """A bare ``word(request_id, g)`` callable (E3's ``PositionOracle.word``) as a :class:`WordOracle`."""

    fn: Callable[[str, int], Any]
    name: str = "positions"

    def word(self, request_id: str, g: int):
        return self.fn(request_id, int(g))


def verifier_ftz_resolver(handle: Any):
    """[E7 oracles hook, for ``ck_elem.RecordHandle.ftz``] E4's resolver over the wired chain + E2's ``TableWords`` (``handle.tables`` when set,
    else the ``commitments.tables`` block, else the position-only table -- the row POSITION is a function of (table, index, rows-per-leaf)
    alone; whether the table tree is pinned is the verifier's committed-trees check, exactly as :func:`expected_table_port` reads it)."""

    tw = getattr(handle, "tables", None)
    if tw is None:
        tw = table_words(getattr(handle, "commitments", None) or {})
    if tw is None:
        from e2e.interior.tables import TableWords

        tw = TableWords(None)
    return _FtzWords(handle, tw)


def expected_table_port(kind: str, statement: Any, record: Any, tables: Any = None):
    """The verifier's table port for a table kind from E3's :class:`PositionStatement` ``statement`` (operand 0 = the committed word whose
    value selects the row) and the record: ``None`` for a kind without a table.  ``tables`` defaults to E2's ``TableWords`` over the record's
    ``commitments.tables`` (``record.tables`` when the handle carries one, else built from ``record.commitments``)."""

    if table_of_kind(kind) is None:
        return None
    if tables is None:
        tables = getattr(record, "tables", None)
    if tables is None:
        tables = table_words(getattr(record, "commitments", None) or {})
    if tables is None:
        # the verifier's handle carries `commitments.json` (no `tables` block: the pin is instance-wide, protocol_instance.json). The row
        # POSITION is a function of (table, index, E2's fixed rows-per-leaf) alone; whether the table tree is pinned/committed is the
        # verifier's committed-trees check (`steps.public_trees` -> "root 'table/…' is not a committed tree of J"), not this port's business.
        from e2e.interior.tables import TableWords

        tables = TableWords(None)
    op = statement.operands[0]
    flat = int(op.offset[0]) if op.offset else 0
    return table_port(kind, leaf_word_value(record, op.leaf, flat, tree=op.tree), tables)


#: [NIGHT G1 06:30Z, for G3/G4 -- PROPOSED ``composed-body/v1``] a primitive kind the approved gate sets have NO single gate for is proven by a
#: SHORT CANONICAL BODY of approved gates that computes the same function bit for bit, reading -- besides the program operands -- one or more
#: STRUCTURAL LITERALS: public words of the pinned literal tree (``prescribed-literal/v1``) the body needs as constants (a program that has no such
#: literal refuses ``LITERAL_UNPINNED``).  First row: ``Bf16Add_v1|batch`` (Qwen2.5-0.5B's QKV bias add, 331,776 VUs; E6 05:25Z) =
#: ``aten.add(bf16, bf16)`` = ``bf16_rn(f32(a) + f32(b))`` with torch's NaN word ``0x7FC0``.  The gate set's ``bf16_add_f2fp`` is the SAME
#: function with the CUDA NaN word ``0x7FFF`` (2,811 of 500,400 measured pairs differ: every NaN result) -- not the kind.  The body:
#: ``one = f2fp_bf16(Const32[0x3f800000])`` (= bf16 ``0x3F80``, exact), ``out = rope_out_add(a, b, one, one)`` = ``bf16_rn_torch(RN32(b*1 + a*1))``
#: -- ``x*1`` is exact for every x (IEEE, denormals kept), ``fma(b, 1, a)`` = ``RN(a + b)`` = ``f32_add`` in every case incl. signed zeros and
#: NaN/inf, and ``rope_out_add`` rounds with torch's NaN word (``reduce_gates.rope_out_add``); MEASURED bit-identical to ``prims.Bf16Add`` on
#: 500,400 pairs (all NaN/inf/denormal/zero edge pairs + 300k random + 100k near-cancelling + 100k denormal; ``/tmp/e7-night/bf16add_equiv.py``,
#: the test below re-measures a 20k sample).  Incoming ports: ``a``, ``b`` (program operands, first occurrence), then the structural literal.
@dataclass(frozen=True)
class ComposedSpec:
    """A :data:`COMPOSED_KINDS` row, shaped like ``ck_elem.KindSpec``: ``arity`` / ``in_width`` are the PRIMITIVE's (program) operands; the body's
    incoming = those operands then ``literals`` (``Const<w>[0x…]_v1`` ids, structural constants from the pinned literal tree); ``body`` = the gates
    over ``(IN, i)`` / ``(GATE, j)`` with ``IN`` indices into that incoming list; ``outgoing`` = the index of the gate whose value is the output."""

    kind: str
    prim: str
    gate: str
    family: str
    arity: int
    in_width: int
    out_width: int
    literals: tuple[str, ...]
    body: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    outgoing: int
    note: str = ""

    @property
    def in_widths(self) -> tuple[int, ...]:
        return (self.in_width,) * self.arity + tuple(const_id_literal(l)[0] for l in self.literals)  # type: ignore[index]

    @property
    def subcircuit(self) -> Any:
        return self.aliased_subcircuit(tuple(range(self.arity + len(self.literals))))

    def aliased_subcircuit(self, wiring: tuple[int, ...]) -> Any:
        """The body with composed incoming ``i`` read from distinct port ``wiring[i]`` (``wiring`` = identity, the canonical body; a wiring over the
        program operands alone gets the literal ports appended)."""

        from veritor.interface.programs import GATE, IN, Definition, make_subcircuit

        w = tuple(wiring)
        if len(w) == self.arity:
            n = len(set(w))
            w = w + tuple(range(n, n + len(self.literals)))
        if len(w) != self.arity + len(self.literals):
            raise ValueError(f"{self.kind}: wiring {w} does not cover the {self.arity} operands + {len(self.literals)} structural literals")
        widths_all = self.in_widths
        first: dict[int, int] = {}
        for i, j in enumerate(w):
            first.setdefault(j, widths_all[i])
        widths = [first[j] for j in sorted(first)]
        gates = [(op, [(IN, w[k]) if space == IN else (GATE, k) for space, k in args]) for op, args in self.body]
        params = (("alias", ",".join(map(str, w))),) if w != tuple(range(len(w))) else ()
        return make_subcircuit(widths, gates, [self.outgoing], Definition(self.prim, params, self.kind))


COMPOSED_KINDS: dict[str, ComposedSpec] = {
    s.kind: s
    for s in (
        ComposedSpec("Bf16Add_v1|batch", "Bf16Add", "f2fp_bf16(one) ; rope_out_add(a, b, one, one)", REDUCE_FAMILY, 2, 16, 16,
                     ("Const32[0x3f800000]_v1",), (("f2fp_bf16", (("in", 2),)), ("rope_out_add", (("in", 0), ("in", 1), ("gate", 0), ("gate", 0)))), 1,
                     "aten.add(bf16, bf16) = bf16_rn_torch(f32(a) + f32(b)); the QKV bias add (Qwen2.5-0.5B) -- composed-body/v1"),
    )
}
COMPOSED_FAMILIES: dict[str, str] = {k: s.family for k, s in COMPOSED_KINDS.items()}
COMPOSED_RULE = "composed-body/v1"


def structural_literals(kind: str) -> tuple[str, ...]:
    """The structural literal ids a kind's body reads besides its operands (``()`` for every non-composed kind)."""

    s = COMPOSED_KINDS.get(str(kind))
    return tuple(s.literals) if s is not None else ()

#: [E7, 2026-09-12 01:30Z, D28] ALIASED OPERANDS.  The rule, verbatim (``aliased-body/v1``):
#:
#:   "A one-gate kind whose gate reads ONE committed word through several operands (``F32Mul(x, x)`` -- RMSNorm's mean square, 28 of the 62 uncovered
#:   units of TINY attempt 8) is expressed as the body whose incoming boundary lists that word ONCE and whose gate reads that incoming value at every
#:   operand position that names it: incoming = the unit's DISTINCT committed operand words in first-occurrence order; gate 0 = the kind's gate with
#:   operand i wired to the incoming value of the word operand i names; outgoing = gate 0.  The obligation's boundary therefore lists each committed
#:   word once (``typed-obligation/v0`` E_DUPLICATE_BOUNDARY_VALUE / E_IDENTITY_CONFLICT keep their meaning: two boundary entries never name one
#:   position) and the statement opens the word once.  The body's digest differs from the kind's canonical body by construction (it is a different
#:   circuit); the verifier's expectation for the unit carries THAT digest, derived from the pinned program's operand gates exactly as the builder does."
#:
#: No statement-format change (``typed-obligation/v0`` records stay valid and are still refused by name when they list a word twice), no lowering
#: change (``to_statement_v3`` maps a body operand ``(IN, i)`` to ``Port(i)``; the guest's ``gather_args`` reads ``values[inputs[k]]`` for every ``Arg::Port(k)``
#: -- a gate reading one port twice is an ordinary program) and NO GUEST REBUILD: the approved ELFs evaluate any well-formed program of the gate set.
#: Multi-gate bodies (``CallSpec`` / ``PvSpec``) alias nothing today: their operands are distinct by construction (a repeated x / w word is a refusal).
ALIAS_RULE = "aliased-body/v1"


@dataclass(frozen=True)
class AliasSpec:
    """The per-unit spec of a one-gate kind whose operands repeat a committed word: ``base`` is the kind's spec (``ck_elem.KindSpec`` / ``TableSpec``),
    ``wiring[i]`` the index into the DISTINCT incoming words that operand ``i`` reads (``(0, 0)`` for ``F32Mul(x, x)``)."""

    base: Any
    wiring: tuple[int, ...]

    @property
    def kind(self) -> str:
        return self.base.kind

    @property
    def prim(self) -> str:
        return self.base.prim

    @property
    def gate(self) -> str:
        return self.base.gate

    @property
    def family(self) -> str:
        return self.base.family

    @property
    def in_width(self) -> int:
        return self.base.in_width

    @property
    def out_width(self) -> int:
        return self.base.out_width

    @property
    def table(self) -> str | None:
        return getattr(self.base, "table", None)

    @property
    def note(self) -> str:
        return f"{getattr(self.base, 'note', '')} [aliased operands {self.wiring}: one committed word read by several operands]"

    @property
    def arity(self) -> int:
        """The body's incoming count (the DISTINCT words); the primitive's own arity is ``len(wiring)``."""

        return len(set(self.wiring))

    @property
    def in_widths(self) -> tuple[int, ...]:
        base_widths = tuple(getattr(self.base, "in_widths", None) or (self.base.in_width,) * len(self.wiring))
        first = {}
        for i, j in enumerate(self.wiring):
            first.setdefault(j, base_widths[i])
        return tuple(first[j] for j in sorted(first))

    @property
    def subcircuit(self) -> Any:
        from veritor.interface.programs import IN, Definition, make_subcircuit

        own = getattr(self.base, "aliased_subcircuit", None)  # composed-body/v1: the multi-gate body re-wired
        if own is not None:
            return own(self.wiring)
        gates = [(self.gate, [(IN, j) for j in self.wiring])]
        return make_subcircuit(list(self.in_widths), gates, [0], Definition(self.prim, (("alias", ",".join(map(str, self.wiring))),), self.kind))


def alias_unit(u: ResolvedUnit) -> ResolvedUnit:
    """``u`` with repeated committed operand words folded per :data:`ALIAS_RULE`: when two operands name one position ``(tree, leaf, flat)``, the
    unit's ``inputs`` become the distinct words (first-occurrence order) and its ``spec`` an :class:`AliasSpec` wiring every operand to its word.
    A unit without repeats is returned unchanged.  The multi-gate ``CallSpec`` is never aliased (its resolver refuses repeats); the one-gate
    ``PvSpec`` is (``prescribed-literal/v1``: every ``ZERO16`` pad reads the one literal word)."""

    import dataclasses

    if isinstance(u.spec, CallSpec) or len(u.inputs) < 2:
        return u
    keys = [(w.tree, w.leaf, int(w.flat)) for w in u.inputs]
    if len(set(keys)) == len(keys):
        return u
    order: dict[tuple[str, str, int], int] = {}
    wiring = []
    distinct = []
    for w, k in zip(u.inputs, keys):
        if k not in order:
            order[k] = len(distinct)
            distinct.append(w)
        wiring.append(order[k])
    return dataclasses.replace(u, spec=AliasSpec(u.spec, tuple(wiring)), inputs=tuple(distinct))


def _identity_key(v: Any) -> tuple[str, str, tuple[int, ...]]:
    """The COMMITTED IDENTITY of an expected value / committed word: ``(tree, leaf id, offset)`` -- never its bits (D32.1)."""

    off = getattr(v, "offset", None)
    if off is None:
        off = (int(v.flat),)
    return (str(v.tree), str(v.leaf), tuple(int(x) for x in off))


def _literal_id_of(v: Any, origin: str | None) -> str | None:
    """The ``Const<w>[0x…]_v1`` id of a literal operand of an expected statement (E3's spelling: ``origins[i] == "literal"``, ``ExpectedValue`` with
    ``tree == "literal"`` and the Const id as its ``leaf``), else ``None``."""

    if origin == "literal" or str(getattr(v, "tree", "")) == "literal":
        leaf = str(getattr(v, "leaf", ""))
        lit = const_id_literal(leaf)
        return const_prim_id(*lit) if lit is not None else leaf
    return None


def _pv_acc_zero(statement: Any) -> bool:
    """Is operand 0 of a :data:`PV_KIND` statement the ``ZERO32`` literal (body ``tc_dot16_0``, the position dropped)?"""

    origins = tuple(getattr(statement, "origins", ()) or ())
    ops = tuple(statement.operands)
    if str(statement.kind) != PV_KIND or not ops:
        return False
    lid = _literal_id_of(ops[0], origins[0] if origins else None)
    return lid is not None and const_id_literal(lid) == (32, 0)


def alias_map(statement: Any) -> tuple[int, ...]:
    """[D32.1, the export E9/E3 compare against] ``operand position -> incoming port`` of the body the builder proves for an expected statement
    (``veritor.interface.expected.PositionStatement``: ``operands`` = ``ExpectedValue(tree, leaf, offset)`` per operand, ``origins`` per operand).
    Ports are numbered by FIRST OCCURRENCE of each distinct committed identity ``(tree, leaf, offset)``.  [``prescribed-literal/v1``] a ``literal``
    operand IS a committed identity -- the literal tree's word, keyed by its ``Const<w>[0x…]_v1`` id (two positions reading the same literal share
    one port; distinct literals are distinct ports); the ONE exception is the PV step's ``ZERO32`` accumulator, dropped by the ``tc_dot16_0`` body
    and mapped ``-1``.  Inputs: identities only -- no committed values, no bytes.  ``F32Mul(x, x)`` -> ``(0, 0)``; ``F32Mul(x, y)`` with
    ``bits(x) == bits(y)`` and distinct identities -> ``(0, 1)``; ``F32Mul(y, x)`` -> ``(0, 1)`` with port 0 = y; ``F32Add(x, c)`` -> ``(0, 1)``;
    ``F32Add(c, c)`` -> ``(0, 0)``."""

    origins = tuple(getattr(statement, "origins", ()) or ())
    ports: dict[tuple[str, str, tuple[int, ...]], int] = {}
    out = []
    acc_zero = _pv_acc_zero(statement)
    for i, v in enumerate(statement.operands):
        if i == 0 and acc_zero:
            out.append(-1)
            continue
        lid = _literal_id_of(v, origins[i] if i < len(origins) else None)
        key = ("literal", lid, ()) if lid is not None else _identity_key(v)
        out.append(ports.setdefault(key, len(ports)))
    return tuple(out)


def literal_ids(statement: Any) -> tuple[str, ...]:
    """The distinct literal ids an expected statement reads (sorted), for :class:`LiteralTable`; identities only."""

    origins = tuple(getattr(statement, "origins", ()) or ())
    ids = {_literal_id_of(v, origins[i] if i < len(origins) else None) for i, v in enumerate(statement.operands)}
    return tuple(sorted(x for x in ids if x is not None))


def expected_incoming(statement: Any, literals: Mapping[str, Any] | Sequence[str] | LiteralTable | None) -> tuple[Any, ...]:
    """[export for E9/E3] The incoming PORTS of the body :func:`expected_body` names, in first-occurrence order, as ``ExpectedValue`` s: committed
    operands by their identity, literal operands as the pinned literal tree's word (``LiteralTable.expected(id)``; the PV ``ZERO32`` accumulator
    dropped).  ``literals`` = the instance's ``commitments.literals`` pin (or ids / a table); a literal absent from it is a ``Refusal`` by name
    (``LITERAL_UNPINNED``).  Identity-only.  The verifier's ``ExpectedObligation.incoming`` for the unit is exactly this tuple."""

    lt = literal_table({}, literals) if literals is not None else None
    origins = tuple(getattr(statement, "origins", ()) or ())
    ports: list[Any] = []
    for i, (v, p) in enumerate(zip(statement.operands, alias_map(statement))):
        if p < 0 or p < len(ports):
            continue
        lid = _literal_id_of(v, origins[i] if i < len(origins) else None)
        if lid is None:
            ports.append(v)
            continue
        w = lt.expected(lid) if lt is not None else None
        if w is None:
            raise _ck_elem().Refusal(f"LITERAL_UNPINNED: operand {i} of {statement.kind} is the literal {lid}; the instance pins no literal tree holding it")
        ports.append(w)
    for lid in structural_literals(str(statement.kind)):  # composed-body/v1: the body's structural constants, after the operands
        w = lt.expected(lid) if lt is not None else None
        if w is None:
            raise _ck_elem().Refusal(f"LITERAL_UNPINNED: the {statement.kind} body reads the structural literal {lid}; the instance pins no literal tree holding it")
        ports.append(w)
    return tuple(ports)


def expected_body(statement: Any) -> str:
    """The body digest the builder proves for ``statement`` (one-gate kinds: the canonical body, or the :data:`ALIAS_RULE` body when
    :func:`alias_map` repeats a port -- the PV step over its ``acc_zero`` variant; multi-gate kinds: the builder's canonical digest).
    Identity-only.  ``KeyError`` for an unregistered kind."""

    kind = str(statement.kind)
    spec = kind_spec(kind)
    wiring = tuple(p for p in alias_map(statement) if p >= 0)
    if isinstance(spec, PvSpec):
        spec = PvSpec(_pv_acc_zero(statement))
        if wiring and len(set(wiring)) != len(wiring):
            return AliasSpec(spec, wiring).subcircuit.digest.lower()
        return spec.subcircuit.digest.lower()
    if isinstance(spec, ComposedSpec):  # composed-body/v1: the operands' wiring, the structural literal ports appended
        return spec.aliased_subcircuit(wiring).digest.lower()
    if isinstance(spec, CallSpec) or not wiring or len(set(wiring)) == len(wiring):
        if kind in ALL_BUILDERS:
            return str(ALL_BUILDERS[kind].subcircuit_digest()).lower()
        from . import ck_ftz as _ftz  # E4's one-gate kinds: E4's canonical body

        return _ftz.kind_subcircuit(spec).digest.lower()
    return AliasSpec(spec, wiring).subcircuit.digest.lower()


#: [D32.2] the profile id of a census (phantom) record -- 12 lowercase hex, as identity requires
CENSUS_PROFILE = "e7ce05000000"
CENSUS_REQUEST = "census"


class PhantomRecord:
    """[D32.2] A record with IDENTITIES but no VALUES: what :func:`can_express` composes over.  Built from expected statements
    (``PositionStatement``: positions + origins) -- every committed leaf they name exists with the shape the positions need and a value of
    ZERO bits; every tree they name has a nominal root.  Openings are never authenticated (there is no commitment); the relation is never
    checked (there is no truth) -- only whether the builder's code path can compose a body, an obligation and a format-3 statement."""

    def __init__(self, statements: Sequence[Any], *, widths: Mapping[str, int] | None = None, tables: Mapping[str, Mapping[str, Any]] | None = None,
                 literals: Mapping[str, Any] | Sequence[str] | LiteralTable | None = None) -> None:
        self.profile_id = CENSUS_PROFILE
        self.extent: dict[str, int] = {}
        self.tree_of: dict[str, str] = {}
        self.width_of: dict[str, int] = dict(widths or {})
        for st in statements:
            for v in (*st.operands, *st.outputs):
                if str(v.tree) == "literal":
                    continue
                key = _identity_key(v)
                flat = key[2][-1] if key[2] else 0
                self.extent[v.leaf] = max(self.extent.get(v.leaf, 0), int(flat) + 1)
                self.tree_of[v.leaf] = str(v.tree)
        trees = sorted(set(self.tree_of.values()))
        self.leaves_of = {t: sorted(l for l, tt in self.tree_of.items() if tt == t) for t in trees}
        self.commitments: dict[str, Any] = {"schema": "phantom", "profile_id": self.profile_id, "trees": {}, "interior_roots": {}}
        import hashlib

        for t in trees:  # a nominal root per tree, DISTINCT per tree id (the witness identifies trees by root)
            rec = {"root": hashlib.sha256(f"phantom-root:{t}".encode()).hexdigest(), "leaf_count": len(self.leaves_of[t]),
                   "ids_digest": hashlib.sha256(f"phantom-ids:{t}".encode()).hexdigest()}
            if t.startswith("interior/"):
                self.commitments["interior_roots"]["/".join(t.split("/")[1:])] = rec
            else:
                self.commitments["trees"][t] = rec
        if tables:
            self.commitments["tables"] = dict(tables)
        self.tables_pin = dict(tables or {})
        self.tables = table_words(self.commitments, self.tables_pin)  # E2's TableWords over the pin (public data; identity-level)
        self._open = _TableOpener(self._open_phantom, self.tables) if self.tables is not None else self._open_phantom
        # [prescribed-literal/v1] the literal tree: the instance pin when given, else rebuilt from the statements' own literal ids (identity-level:
        # the ids ARE the values; representability does not depend on which other literals the program has)
        if literals is None:
            ids = sorted({lid for st in statements for lid in (*literal_ids(st), *structural_literals(str(st.kind)))})
            literals = ids or None
        self.literals = literal_table(self.commitments, literals)
        if self.literals is not None:
            self.commitments["literals"] = self.literals.pin()
            self._open = _LiteralOpener(self._open, self.literals)

    def width(self, leaf: str, default: int = 32) -> int:
        return int(self.width_of.get(leaf, default))

    def open_leaf(self, cid: str):
        return self._open(cid)

    def _open_phantom(self, cid: str):
        from verity_vllm.commit import Opening

        if cid not in self.extent:
            raise _ck_elem().Refusal(f"{cid}: no such leaf in the phantom record (the statements name no position in it)")
        n, w = self.extent[cid], self.width(cid)
        t = self.tree_of[cid]
        root = bytes.fromhex((self.commitments["trees"].get(t) or self.commitments["interior_roots"]["/".join(t.split("/")[1:])])["root"])
        return Opening(tree=t, root=root, leaf_count=len(self.leaves_of[self.tree_of[cid]]), index=self.leaves_of[self.tree_of[cid]].index(cid),
                       canonical_id=cid, dtype={16: "uint16", 32: "uint32", 64: "uint64", 8: "uint8"}.get(w, f"u{w}"), shape=(n,), value=bytes(n * (w // 8)), path=())

    def oracle(self, statements: Sequence[Any]) -> "LocateOracle":
        """E3's position oracle over the statements' ``operand_gates`` / ``output_gates`` -> the 5-tuple spelling (tree, leaf, shape, flat, width)."""

        by_gate: dict[int, tuple[str, str, tuple[int, ...], int, int]] = {}
        for st in statements:
            for g, v in zip((*st.operand_gates, *st.output_gates), (*st.operands, *st.outputs)):
                if str(v.tree) == "literal":
                    continue
                flat = int(v.offset[-1]) if v.offset else 0
                by_gate[int(g)] = (str(v.tree), str(v.leaf), (self.extent[v.leaf],), flat, self.width(v.leaf))
        return LocateOracle(lambda _rid, g: by_gate.get(int(g)), name="census")


def phantom_widths(statement: Any) -> dict[str, int]:
    """Leaf -> word width from the kind's spec (operand i -> ``in_widths[i]``, output -> ``out_width``); identity-level, no values."""

    out: dict[str, int] = {}
    try:
        spec = kind_spec(str(statement.kind))
    except KeyError:
        return out
    in_widths = tuple(getattr(spec, "in_widths", None) or (getattr(spec, "in_width", 32),) * len(statement.operands))
    for i, v in enumerate(statement.operands):
        if str(v.tree) != "literal" and i < len(in_widths):
            out.setdefault(str(v.leaf), int(in_widths[i]))
    for v in statement.outputs:
        if str(v.tree) != "literal":
            out.setdefault(str(v.leaf), int(getattr(spec, "out_width", 32)))
    return out


def phantom_unit(statement: Any, record: Any) -> ResolvedUnit:
    """A ``ResolvedUnit`` from an expected statement alone (no program): the kind's spec, operands / output as identity-only
    ``CommittedWord`` s (zero values in the phantom leaves), then :func:`alias_unit` -- the composition path :func:`build_batch` runs.
    A ``literal`` operand is the literal tree's word (``prescribed-literal/v1``; the PV ``ZERO32`` accumulator is dropped by ``tc_dot16_0``).
    Refuses by name: a literal the record's literal tree does not hold (``LITERAL_UNPINNED``), a repeated operand of a multi-gate kind
    (``E_DUPLICATE_BOUNDARY_VALUE``), an operand that is also the output (idem)."""

    ck = _ck_elem()
    kind = str(statement.kind)
    spec = kind_spec(kind)
    origins = tuple(getattr(statement, "origins", ()) or ())
    words = []
    acc_zero = _pv_acc_zero(statement)
    if isinstance(spec, PvSpec):
        spec = PvSpec(acc_zero)
    for i, v in enumerate(statement.operands):
        lid = _literal_id_of(v, origins[i] if i < len(origins) else None)
        if lid is not None:
            if i == 0 and acc_zero:
                continue  # the tc_dot16_0 body: the guest's +0 accumulator, no position
            lt = getattr(record, "literals", None)
            w = lt.position(lid) if lt is not None else None
            if w is None:
                raise ck.Refusal(f"{kind}: LITERAL_UNPINNED: operand {i} is the literal {lid} and the instance pins no literal tree holding it ({LITERAL_RULE})")
            words.append(w)
            continue
        flat = int(v.offset[-1]) if v.offset else 0
        words.append(ck.CommittedWord(str(v.tree), str(v.leaf), (record.extent[v.leaf],), flat, record.width(v.leaf)))
    if len(statement.outputs) != 1:
        raise ck.Refusal(f"{kind}: {len(statement.outputs)} outputs; a registry unit has one")
    o = statement.outputs[0]
    out = ck.CommittedWord(str(o.tree), str(o.leaf), (record.extent[o.leaf],), int(o.offset[-1]) if o.offset else 0, record.width(o.leaf))
    if any(_identity_key(w) == _identity_key(out) for w in words):
        raise ck.Refusal(f"{kind}: the output word is also an operand: one reference on both sides of a boundary (E_DUPLICATE_BOUNDARY_VALUE)")
    keys = [_identity_key(w) for w in words]
    if isinstance(spec, CallSpec) and len(set(keys)) != len(keys):
        raise ck.Refusal(f"{kind}: a multi-gate body reads one committed word at two operands; not aliased (E_DUPLICATE_BOUNDARY_VALUE)")
    if isinstance(spec, TableSpec):
        if not record.tables_pin.get(spec.table):
            raise ck.Refusal(f"{kind}: TABLE_UNPINNED: the instance pins no commitments.tables[{spec.table!r}]")
        # the row word: identity-only -> row 0 of the pinned table tree (the phantom operand is zero bits); the leaf exists in E2's layout
        tw = getattr(record, "tables", None)
        row = tw.table_row(spec.table, 0) if tw is not None else None
        if row is None:
            raise ck.Refusal(f"{kind}: TABLE_UNPINNED: {spec.table} row 0 cannot be placed from the pin")
        words.append(row)
    if isinstance(spec, ComposedSpec):
        words.extend(_structural_words(kind, spec, record))
    ru = int(out.tree.split("/")[2]) if out.tree.startswith("ru/") else 0
    u = ck.ResolvedUnit(int(statement.vu_index), kind, spec, int(statement.output_gates[0]) if statement.output_gates else -1, "census", CENSUS_REQUEST, tuple(words), out, ru)
    return alias_unit(u)


def record_literal_table(record: Any) -> "LiteralTable | None":
    """[E7 SCALE 2026-09-12, E10 q2 refusal `Bf16Add_v1|batch: LITERAL_UNPINNED`] The record handle's pinned literal table as OUR
    :class:`LiteralTable` (``position(id) -> CommittedWord``), whatever spelling the handle carries: ours (the prover's ``open_record``),
    E9's ``steps.LiteralTable`` (the verifier's ``coverage_inputs`` sets ``record_handle.literals`` -- taken through its ``pin()``, the same
    ids, the same trees, CHECKED against the pinned roots), or none -- then the ``commitments.literals`` block when the record carries one.
    ``None`` = unpinned (a body that reads a literal refuses ``LITERAL_UNPINNED`` by name)."""

    lt = getattr(record, "literals", None)
    if lt is None or isinstance(lt, LiteralTable):
        if lt is None:
            lt = literal_table(getattr(record, "commitments", None) or {}, None)
        return lt
    pin = lt.pin() if hasattr(lt, "pin") else lt
    return literal_table(getattr(record, "commitments", None) or {}, pin)


def _structural_words(kind: str, spec: "ComposedSpec", record: Any) -> list[Any]:
    """[composed-body/v1] the pinned literal tree's words of ``spec.literals``; ``LITERAL_UNPINNED`` by name when the record pins none / not these."""

    lt = record_literal_table(record)
    out = []
    for lid in spec.literals:
        w = lt.position(lid) if lt is not None else None
        if w is None:
            raise _ck_elem().Refusal(f"{kind}: LITERAL_UNPINNED: the body reads the structural literal {lid} and the instance pins no literal tree "
                                     f"holding it ({LITERAL_RULE}; pinned: {tuple(lt.ids) if lt is not None else ()})")
        out.append(w)
    return out


#: every registered kind -> family (``ELEM_FAMILIES`` + ``EXTRA_FAMILIES`` + ``CALL_FAMILIES`` + the PV step + the table kinds)
ALL_FAMILIES: dict[str, str] = {**ELEM_FAMILIES, **EXTRA_FAMILIES, **CALL_FAMILIES, PV_KIND: TC_FAMILY, **TABLE_FAMILIES, **COMPOSED_FAMILIES}
#: the kinds this module resolves itself (not through ``ck_elem.resolve``): the call kinds, the PV step and the table kinds
E7_KINDS: frozenset[str] = frozenset(CALL_KINDS) | {PV_KIND} | frozenset(TABLE_KINDS) | frozenset(COMPOSED_KINDS)

#: proof-object size: units per statement.  MEASURED (attempt 6, RTX 5090, SP1 6.4.0 cuda): elem-bf16 50 units = 6.2 M cycles / 3.6 s,
#: reduce-mufu 13 units = 1.8 M cycles / 2.2 s; prices_n64 (RTX 4090): f2fp n=64 0.076 s/VU, 2^16 cycles/VU.  64 keeps every family
#: well under SP1's comfortable shard count; the guest itself has no batch limit (relations.rs walks the obligation list).
DEFAULT_MAX_UNITS = 64
MAX_UNITS_BY_FAMILY: dict[str, int] = {ELEM_FAMILY: 64, REDUCE_FAMILY: 64, "veritor.tc-ampere-bf16@2": 64}


def _ck_elem():
    from . import ck_elem  # verifier / obligation-stage side only (imports src/veritor)

    return ck_elem


_EXTRA_SPECS: dict[str, Any] = {}


def kind_spec(kind: str) -> Any:
    """``ck_elem.KindSpec`` of a registered kind: ``ck_elem.KINDS[kind]``, the :data:`EXTRA_KINDS` row or the :data:`CALL_KINDS` spec (``KeyError`` otherwise)."""

    if kind in CALL_KINDS:
        return CALL_KINDS[kind]
    if kind == PV_KIND:
        return PvSpec()  # the literal-free body; a unit's own spec (its literal pattern) comes from resolve_pv_unit
    if kind in TABLE_KINDS:
        return TABLE_KINDS[kind]
    if kind in COMPOSED_KINDS:
        return COMPOSED_KINDS[kind]
    if kind in ftz_kinds():
        return ftz_kinds()[kind]  # E4's KindSpec (same field shape; body via ck_ftz.kind_subcircuit)
    ck = _ck_elem()
    spec = ck.KINDS.get(kind)
    if spec is not None:
        return spec
    if not _EXTRA_SPECS:
        for k, prim, gate, fam, arity, w_in, w_out, note in EXTRA_KINDS:
            _EXTRA_SPECS[k] = ck._k(k, prim, gate, fam, arity, w_in, w_out, note)
    return _EXTRA_SPECS[kind]


def resolve_unit(located: Mapping[str, Any], record: Any) -> ResolvedUnit:
    """``ck_elem.resolve`` for every registered kind: kinds of ``ck_elem.KINDS`` go through it unchanged; an :data:`EXTRA_KINDS` kind is
    resolved by the same rule (one gate of the spec's primitive, operands and output through ``record.elem``, widths as the gate declares).
    Every one-gate unit then passes :func:`alias_unit` (D28): repeated committed operand words fold into one incoming value."""

    ck = _ck_elem()
    located = located_mapping(located)
    kind = str(located["kind"])
    if located.get("gate_interval") is None:
        raise ck.Refusal(f"{kind}: the located unit names no gate_interval in the pinned program; the one-gate builder cannot place it "
                         f"(the unit's operands and output are interior words -- E2's interior/<req>/<step> leaves -- located by E3)")
    if kind in CALL_KINDS:
        return resolve_call_unit(located, record)
    if kind == PV_KIND:
        return alias_unit(resolve_pv_unit(located, record))  # prescribed-literal/v1: the ZERO16 pads fold into one incoming port
    if kind in TABLE_KINDS:
        return alias_unit(resolve_table_unit(located, record))
    if kind in COMPOSED_KINDS:
        return alias_unit(resolve_composed_unit(located, record))
    if kind in ftz_kinds():
        return alias_unit(resolve_ftz_unit(located, record))
    if kind in ck.KINDS or kind not in EXTRA_FAMILIES:
        return alias_unit(ck.resolve(located, record))
    return alias_unit(_resolve_extra_unit(located, record, kind))


def resolve_composed_unit(located: Mapping[str, Any], record: Any) -> ResolvedUnit:
    """A :data:`COMPOSED_KINDS` VU (one gate of the spec's primitive) as a ``ck_elem.ResolvedUnit``: the operands committed ``in_width``-bit words
    (``record.elem``, a literal operand through the literal oracle), the output a committed ``out_width``-bit word, then the body's structural
    literals from the pinned literal tree (``LITERAL_UNPINNED`` by name otherwise).  Anything else is a refusal naming the gate."""

    ck = _ck_elem()
    located = located_mapping(located)
    kind = str(located["kind"])
    spec = COMPOSED_KINDS[kind]
    lo, hi = (int(x) for x in located["gate_interval"])
    if hi - lo != 1:
        raise ck.Refusal(f"{kind}: VU [{lo},{hi}) is not one gate")
    request_id = str(located["request_id"])
    gate = record.program.circuit.gate(lo)
    pname = getattr(gate.prim, "name", gate.prim.id)
    if pname != spec.prim:
        raise ck.Refusal(f"{kind}: gate {lo} is {gate.prim.id}, not {spec.prim}")
    if len(gate.operands) != spec.arity:
        raise ck.Refusal(f"{kind}: gate {lo} has {len(gate.operands)} operands, {spec.prim} takes {spec.arity}")
    name, _root = record.locator.root_of(lo)
    if name is None:
        raise ck.Refusal(f"{kind}: gate {lo} lies in no root call")
    elem = record.elem
    inputs = []
    for i, og in enumerate(gate.operands):
        w = elem.word(request_id, int(og))
        if w is None:
            raise ck.Refusal(f"{kind}: operand {i} = {ck._describe_gate(record.program, int(og))} is an interior value the record does not commit")
        if w.width != spec.in_width:
            raise ck.Refusal(f"{kind}: operand {i} = {w.ref} is a {w.width}-bit committed word, {spec.prim} takes {spec.in_width}-bit values")
        inputs.append(w)
    out = elem.row_word(request_id, lo)
    if out is None:
        raise ck.Refusal(f"{kind}: output {ck._describe_gate(record.program, lo)} is an interior value the record does not commit")
    if out.width != spec.out_width:
        raise ck.Refusal(f"{kind}: output {out.ref} is a {out.width}-bit committed word, {spec.prim} produces {spec.out_width} bits")
    inputs.extend(_structural_words(kind, spec, record))
    ru = int(out.tree.split("/")[2]) if out.tree.startswith("ru/") else 0
    return ck.ResolvedUnit(int(located.get("vu_index", -1)), kind, spec, lo, name, request_id, tuple(inputs), out, ru)


def ftz_kinds() -> Mapping[str, Any]:
    """E4's ``ck_ftz.KINDS`` (the ``veritor.ftz-f32@1`` one-gate kinds) when the module is in the checkout, else ``{}``."""

    try:
        from . import ck_ftz as _ftz
    except Exception:  # noqa: BLE001 -- E4's module absent: no FTZ kinds
        return {}
    return _ftz.KINDS


def resolve_ftz_unit(located: Mapping[str, Any], record: Any) -> ResolvedUnit:
    """An E4 (``ck_ftz``) one-gate unit through E4's OWN resolver (``record.ftz``: the oracle chain + E2's tables + the literal tree), in the
    verifier's located spelling (``gate_interval`` -> ``gate`` / ``root``); the result has ``ck_elem.ResolvedUnit``'s field shape, so
    :func:`alias_unit` / :func:`build_batch` (family ``veritor.ftz-f32@1``) compose it like any other one-gate unit."""

    from . import ck_ftz as _ftz

    u = dict(located_mapping(located))
    if "gate" not in u:
        lo, hi = (int(x) for x in u["gate_interval"])
        if hi - lo != 1:
            raise _ck_elem().Refusal(f"{u.get('kind')}: VU [{lo},{hi}) is not one gate")
        u["gate"] = lo
    if not u.get("root"):
        locator = getattr(record, "locator", None)
        name = locator.root_of(int(u["gate"]))[0] if locator is not None and hasattr(locator, "root_of") else None
        u["root"] = name or ""
    u.setdefault("replay_unit", -1)
    try:
        return _ftz.resolve(u, record, getattr(record, "ftz", None))
    except _ftz.Refusal as e:  # E4's own class -> the registry's, so package_batches / can_express name it the same way
        raise _ck_elem().Refusal(str(e)) from e


def _resolve_extra_unit(located: Mapping[str, Any], record: Any, kind: str) -> ResolvedUnit:
    ck = _ck_elem()
    spec = kind_spec(kind)
    lo, hi = (int(x) for x in located["gate_interval"])
    if hi - lo != 1:
        raise ck.Refusal(f"{kind}: VU [{lo},{hi}) is not one gate")
    request_id = str(located["request_id"])
    gate = record.program.circuit.gate(lo)
    pname = getattr(gate.prim, "name", gate.prim.id)
    if pname != spec.prim:
        raise ck.Refusal(f"{kind}: gate {lo} is {gate.prim.id}, not {spec.prim}")
    if len(gate.operands) != spec.arity:
        raise ck.Refusal(f"{kind}: gate {lo} has {len(gate.operands)} operands, the gate set's {spec.gate} takes {spec.arity}")
    name, _root = record.locator.root_of(lo)
    if name is None:
        raise ck.Refusal(f"{kind}: gate {lo} lies in no root call")
    elem = record.elem
    inputs = []
    for og in gate.operands:
        w = elem.word(request_id, int(og))
        if w is None:
            raise ck.Refusal(f"{kind}: operand {ck._describe_gate(record.program, int(og))} is an interior value the record does not commit")
        if w.width != spec.in_width:
            raise ck.Refusal(f"{kind}: operand {w.ref} is a {w.width}-bit committed word, the gate takes {spec.in_width}-bit values")
        inputs.append(w)
    out = elem.row_word(request_id, lo)
    if out is None:
        raise ck.Refusal(f"{kind}: output {ck._describe_gate(record.program, lo)} is an interior value the record does not commit")
    if out.width != spec.out_width:
        raise ck.Refusal(f"{kind}: output {out.ref} is a {out.width}-bit committed word, the gate produces {spec.out_width} bits")
    ru = int(out.tree.split("/")[2]) if out.tree.startswith("ru/") else 0
    return ck.ResolvedUnit(int(located.get("vu_index", -1)), kind, spec, lo, name, request_id, tuple(inputs), out, ru)


def resolve_call_unit(located: Mapping[str, Any], record: Any) -> ResolvedUnit:
    """A :data:`CALL_KINDS` VU (``gate_interval`` = the call's body in the pinned program) as a ``ck_elem.ResolvedUnit`` whose ``spec`` is the
    :class:`CallSpec`: the body must be ``b1.DotBf16{K}`` verbatim -- gate ``lo`` the ``ZERO32`` literal, gates ``lo+1 .. hi`` ``K/16``
    ``AmpereBF16TcDot16`` gates each ``(acc = the previous gate, 16 x words, 16 w words)`` -- and every x / w operand (outside the interval) a
    committed 16-bit word, the last accumulator a committed 32-bit word (E2's interior tree for TINY).  Anything else is a refusal naming the gate."""

    ck = _ck_elem()
    located = located_mapping(located)
    kind = str(located["kind"])
    spec = CALL_KINDS[kind]
    lo, hi = (int(x) for x in located["gate_interval"])
    if hi - lo != spec.steps + 1:
        raise ck.Refusal(f"{kind}: VU [{lo},{hi}) has {hi - lo} gates, the body ZERO32 + {spec.steps} x {spec.prim} has {spec.steps + 1}")
    request_id = str(located["request_id"])
    circuit = record.program.circuit
    zero = circuit.gate(lo)
    if zero.operands or getattr(zero.prim, "name", zero.prim.id) != ZERO32_PRIM:
        raise ck.Refusal(f"{kind}: gate {lo} is {zero.prim.id} with {len(zero.operands)} operands, not the {ZERO32_PRIM} literal the body starts from")
    name, _root = record.locator.root_of(hi - 1)
    if name is None:
        raise ck.Refusal(f"{kind}: gate {hi - 1} lies in no root call")
    xs: list[int] = []
    ws: list[int] = []
    for s in range(spec.steps):
        g = lo + 1 + s
        gate = circuit.gate(g)
        pname = getattr(gate.prim, "name", gate.prim.id)
        if pname != spec.prim:
            raise ck.Refusal(f"{kind}: gate {g} is {gate.prim.id}, not {spec.prim} (step {s} of the body)")
        ops = [int(o) for o in gate.operands]
        if len(ops) != 33:
            raise ck.Refusal(f"{kind}: gate {g} has {len(ops)} operands, {spec.prim} takes acc + 16 + 16")
        if ops[0] != g - 1:
            raise ck.Refusal(f"{kind}: gate {g}'s accumulator is gate {ops[0]}, not the previous step {g - 1} (the body is one chain from ZERO32)")
        if any(lo <= o < hi for o in ops[1:]):
            raise ck.Refusal(f"{kind}: gate {g} reads an x / w operand inside the body [{lo},{hi}); the body's only interior value is the accumulator")
        xs += ops[1:17]
        ws += ops[17:33]
    elem = record.elem
    inputs = []
    for og in xs + ws:  # A2's GemmAccumulator order: x[0..K) then w[0..K)
        w = elem.word(request_id, og)
        if w is None:
            raise ck.Refusal(f"{kind}: operand {ck._describe_gate(record.program, og)} is an interior value the record does not commit")
        if w.width != spec.in_width:
            raise ck.Refusal(f"{kind}: operand {w.ref} is a {w.width}-bit committed word, {spec.prim} takes {spec.in_width}-bit values")
        inputs.append(w)
    out = elem.row_word(request_id, hi - 1)
    if out is None:
        raise ck.Refusal(f"{kind}: output {ck._describe_gate(record.program, hi - 1)} (the FP32 accumulator after {spec.steps} steps) is an interior value "
                         f"the record does not commit")
    if out.width != spec.out_width:
        raise ck.Refusal(f"{kind}: output {out.ref} is a {out.width}-bit committed word, the accumulator is {spec.out_width} bits")
    ru = int(out.tree.split("/")[2]) if out.tree.startswith("ru/") else 0
    return ck.ResolvedUnit(int(located.get("vu_index", -1)), kind, spec, hi - 1, name, request_id, tuple(inputs), out, ru)


def resolve_pv_unit(located: Mapping[str, Any], record: Any) -> ResolvedUnit:
    """A :data:`PV_KIND` VU (one ``AmpereBF16TcDot16`` gate) as a ``ck_elem.ResolvedUnit`` whose ``spec`` is the :class:`PvSpec` of ITS literal
    body: operand 0 (``acc``) a committed 32-bit word (``tc_dot16``) or the ``ZERO32`` literal (``tc_dot16_0``); operands 1..17 (``pb``) and 17..33
    (``V``) committed 16-bit words -- a ``ZERO16`` padding literal is the pinned literal tree's word (``prescribed-literal/v1``; all pads ONE
    incoming port under ``aliased-body/v1``), a refusal by name ``LITERAL_UNPINNED`` when the instance pins none; the output a committed 32-bit
    word.  Anything else is a refusal naming the gate."""

    ck = _ck_elem()
    located = located_mapping(located)
    kind = str(located["kind"])
    lo, hi = (int(x) for x in located["gate_interval"])
    if hi - lo != 1:
        raise ck.Refusal(f"{kind}: VU [{lo},{hi}) is not one gate")
    request_id = str(located["request_id"])
    circuit = record.program.circuit
    gate = circuit.gate(lo)
    pname = getattr(gate.prim, "name", gate.prim.id)
    if pname != PV_PRIM:
        raise ck.Refusal(f"{kind}: gate {lo} is {gate.prim.id}, not {PV_PRIM}")
    ops = [int(o) for o in gate.operands]
    if len(ops) != 33:
        raise ck.Refusal(f"{kind}: gate {lo} has {len(ops)} operands, {PV_PRIM} takes acc + 16 + 16")
    name, _root = record.locator.root_of(lo)
    if name is None:
        raise ck.Refusal(f"{kind}: gate {lo} lies in no root call")
    elem = record.elem
    inputs: list[CommittedWord] = []
    acc_zero = False
    for i, og in enumerate(ops):
        want = 32 if i == 0 else 16
        lit = _const_literal(circuit.gate(og))
        if lit is not None:
            w_lit, v_lit = lit
            if i == 0 and (w_lit, v_lit) == (32, 0):
                acc_zero = True  # body tc_dot16_0: the guest's +0 accumulator, no literal in the statement
                continue
            # [prescribed-literal/v1] any other literal (the ZERO16 pads past the visible columns, 2*(16-m) positions reading ONE word) is the
            # pinned literal tree's word -- elem.word answers it through the LiteralOracle, or refuses LITERAL_UNPINNED by name
        w = elem.word(request_id, og)
        if w is None:
            raise ck.Refusal(f"{kind}: operand {i} = {ck._describe_gate(record.program, og)} is an interior value the record does not commit")
        if w.width != want:
            raise ck.Refusal(f"{kind}: operand {i} = {w.ref} is a {w.width}-bit committed word, {PV_PRIM} takes {want} bits there")
        inputs.append(w)
    out = elem.row_word(request_id, lo)
    if out is None:
        raise ck.Refusal(f"{kind}: output {ck._describe_gate(record.program, lo)} (the FP32 accumulator) is an interior value the record does not commit")
    if out.width != 32:
        raise ck.Refusal(f"{kind}: output {out.ref} is a {out.width}-bit committed word, the accumulator is 32 bits")
    ru = int(out.tree.split("/")[2]) if out.tree.startswith("ru/") else 0
    return ck.ResolvedUnit(int(located.get("vu_index", -1)), kind, PvSpec(acc_zero), lo, name, request_id, tuple(inputs), out, ru)


#: [NIGHT G1, PROPOSED] ``prescribed-literal/v1`` -- verbatim: the pinned program's distinct literal values form a public tree per width,
#: ``literal/w=<w>`` (one leaf ``literal/w=<w>/chunk=0``, dtype ``uint<w>``, words = the values in sorted order of their ``Const<w>[0x…]_v1`` ids,
#: ``verity-vllm/leaf/v1`` framing, owner -2 like the weights and the MUFU tables); its roots are DERIVED BY THE VERIFIER from the pinned program
#: and PINNED in the instance (``commitments.literals = {rule, literals: [ids], trees: {tree: {root, leaf_count, ids_digest}}}``, as
#: ``commitments.tables``); a literal at an operand position is that tree's word -- an ordinary incoming committed word of the obligation, opened
#: by the proof against the pinned root; the body is the kind's canonical body, aliased under ``aliased-body/v1`` when one literal word is read at
#: several positions; a record whose instance pins no literal tree refuses such a unit by name (``LITERAL_UNPINNED``); a unit whose operands are ALL
#: literals IS a proof obligation (its committed output is a claim about the program).  No statement-format, guest or interface change.
LITERAL_RULE = "prescribed-literal/v1"
LITERAL_TREE_PREFIX = "literal/"
LITERAL_OWNER = -2


def const_id_literal(prim_id: str) -> tuple[int, int] | None:
    """``(width, value)`` of a ``Const<w>[0x…]_v1`` primitive id (``|primitive`` kind suffix tolerated), else ``None``."""

    name = str(prim_id).split("|")[0]
    if name.endswith("_v1"):
        name = name[:-3]
    if not name.startswith("Const") or "[" not in name or not name.endswith("]"):
        return None
    try:
        return int(name[5:name.index("[")]), int(name[name.index("[") + 1:-1], 16)
    except ValueError:
        return None


def const_prim_id(width: int, value: int) -> str:
    return f"Const{int(width)}[0x{int(value):0{int(width) // 4}x}]_v1"


def program_literal_ids(program: Any) -> tuple[str, ...]:
    """[``prescribed-literal/v1``, the O(#definitions) derivation] Every distinct ``Const<w>[0x…]_v1`` primitive id of ``program``, from the
    program's DEFINITION DAG rather than its gates: a program is a tree of function-call instances whose bodies are shared
    ``SpecializedDefinition`` objects; the set of primitives that occur in the circuit is the union, over the definitions reachable from the
    root, of the primitives in each body (a body node materialises ``node.gates >= 1`` gates of ``node.fn``; a node with 0 gates -- an
    empty batch -- puts nothing in the circuit and is skipped).  Each definition is visited once (memo by object identity), so the walk
    costs O(sum of body lengths) -- thousands of nodes on TINY / SMOL360 / QWEN05 against 242.7 M+ gates for the scan.  Works on a
    ``verity.ir.program.Program`` and on E2's ``LazyProgram`` (its root's ``nodes`` are materialised one step-call at a time).  The literal
    checks are the scan's (``locator.literal_of``): zero-arity, no params, and the primitive evaluates to the value in its name -- a
    primitive that does not is a ``Refusal`` (``LITERAL_MISMATCH``), never silently dropped."""

    from verity.ir.defs import PrimitiveDefinition

    ids: set[str] = set()
    seen: set[int] = set()
    stack = [program.root]
    while stack:
        fn = stack.pop()
        if id(fn) in seen:
            continue
        seen.add(id(fn))
        for node in fn.body.nodes:
            if int(node.gates) <= 0:
                continue
            sub = node.fn
            if isinstance(sub, PrimitiveDefinition):
                name = str(getattr(sub, "name", sub.id))
                if getattr(sub, "family", None) != "Const" and not name.startswith("Const"):
                    continue
                if len(sub.params) or "[0x" not in name:
                    continue
                lit = const_id_literal(sub.id)
                if lit is None:
                    continue
                w, bits = lit
                got = int(sub.evaluate())
                if got != bits:
                    raise _ck_elem().Refusal(f"LITERAL_MISMATCH: {sub.id} evaluates to 0x{got:x}, not the literal in its name ({LITERAL_RULE})")
                ids.add(const_prim_id(w, bits))
            else:
                stack.append(sub)
    return tuple(sorted(ids))


class LiteralTable:
    """[``prescribed-literal/v1``] The pinned program's literal values as committed trees, rebuilt in memory by builder AND verifier from the
    literal ids alone (identity-level: the ids carry the values).  ``position(id)`` is the literal's committed word; ``open(cid)`` its leaf
    opening; ``pin()`` what E6 writes into ``commitments.literals``; ``check_pin`` refuses a pin the rebuilt trees do not reproduce."""

    def __init__(self, ids: Sequence[str]) -> None:
        import numpy as np

        from e2e.interior.layout import CallSpan, InteriorLayout, LeafSpec, dtype_for_width
        from e2e.interior.tree import InteriorTree

        parsed: dict[str, tuple[int, int]] = {}
        for raw in ids:
            lit = const_id_literal(raw)
            if lit is None:
                raise _ck_elem().Refusal(f"{LITERAL_RULE}: {raw!r} is not a Const<w>[0x…]_v1 literal id")
            parsed[const_prim_id(*lit)] = lit
        self.ids: tuple[str, ...] = tuple(sorted(parsed))
        self.words: dict[int, list[str]] = {}
        for pid in self.ids:
            self.words.setdefault(parsed[pid][0], []).append(pid)
        self.trees: dict[str, Any] = {}
        self.leaves: dict[str, dict[str, Any]] = {}
        self.index: dict[str, tuple[str, str, int, int]] = {}  # prim id -> (tree, leaf, flat, width)
        for w, pids in sorted(self.words.items()):
            tree, cid = self.tree_id(w), self.leaf_id(w)
            value = np.asarray([parsed[p][1] for p in pids], dtype=f"<u{w // 8}").tobytes()
            leaf = LeafSpec(0, cid, "literals", 0, w, np.arange(len(pids), dtype=np.int64), origin="literal")
            layout = InteriorLayout(tree, [leaf], [CallSpan("literals", 0, len(pids))], prefix="", chunk_words=len(pids))  # the root is over (id, dtype, shape, words) alone; the layout rule tag is informative
            self.leaves[cid] = {"tree": tree, "dtype": dtype_for_width(w), "shape": (len(pids),), "value": value}
            self.trees[tree] = InteriorTree(layout, [value])
            for k, p in enumerate(pids):
                self.index[p] = (tree, cid, k, w)

    @staticmethod
    def tree_id(width: int) -> str:
        return f"{LITERAL_TREE_PREFIX}w={int(width)}"

    @staticmethod
    def leaf_id(width: int) -> str:
        return f"{LITERAL_TREE_PREFIX}w={int(width)}/chunk=0"

    @classmethod
    def from_population(cls, pop_doc: Mapping[str, Any]) -> "LiteralTable":
        """From a v2 population's ``basis.excluded.literals_compiled`` (the kinds E6 compiled away)."""

        ids = ((pop_doc.get("basis") or {}).get("excluded") or {}).get("literals_compiled") or []
        return cls([str(k) for k in ids])

    @classmethod
    def from_program(cls, program: Any) -> "LiteralTable":
        """From the program's Const primitives, by the definition walk (:func:`program_literal_ids`: every specialised definition's body once,
        O(#definitions), not O(#gates)); :meth:`from_program_scan` is the gate-by-gate reference the walk is measured against."""

        return cls(program_literal_ids(program))

    @classmethod
    def from_program_scan(cls, program: Any) -> "LiteralTable":
        """The reference derivation: a scan of every gate (small programs / tests; 242.7 M gates on TINY -- minutes to hours in Python)."""

        circuit = program.circuit
        ids = set()
        for g in range(int(program.gates)):
            lit = _const_literal(circuit.gate(g))
            if lit is not None:
                ids.add(const_prim_id(*lit))
        return cls(sorted(ids))

    def position(self, prim_id: str):
        lit = const_id_literal(prim_id)
        if lit is None or const_prim_id(*lit) not in self.index:
            return None
        tree, cid, k, w = self.index[const_prim_id(*lit)]
        return _ck_elem().CommittedWord(tree, cid, (len(self.words[w]),), k, w)

    def expected(self, prim_id: str):
        from veritor.interface.expected import ExpectedValue

        w = self.position(prim_id)
        return None if w is None else ExpectedValue(w.tree, w.leaf, (int(w.flat),))

    def open(self, cid: str):
        lf = self.leaves.get(cid)
        if lf is None:
            raise _ck_elem().Refusal(f"{cid}: not a leaf of the literal trees ({sorted(self.leaves)})")
        return self.trees[lf["tree"]].open(0)

    def tree_records(self) -> dict[str, dict[str, Any]]:
        return {t: {"root": tr.root.hex(), "leaf_count": tr.leaf_count, "ids_digest": tr.ids_digest.hex()} for t, tr in self.trees.items()}

    def pin(self) -> dict[str, Any]:
        return {"rule": LITERAL_RULE, "literals": list(self.ids), "trees": self.tree_records()}

    def check_pin(self, pin: Mapping[str, Any]) -> None:
        if str(pin.get("rule")) != LITERAL_RULE:
            raise _ck_elem().Refusal(f"commitments.literals names rule {pin.get('rule')!r}, this builder implements {LITERAL_RULE}")
        mine = self.tree_records()
        theirs = {str(t): dict(r) for t, r in (pin.get("trees") or {}).items()}
        if set(mine) != set(theirs):
            raise _ck_elem().Refusal(f"commitments.literals trees {sorted(theirs)} != rebuilt {sorted(mine)}")
        for t, r in mine.items():
            if any(str(theirs[t].get(k)).lower() != str(v).lower() for k, v in r.items()):
                raise _ck_elem().Refusal(f"{t}: commitments.literals root/count/ids disagree with the tree rebuilt from the pinned literal ids")


_LITERAL_TABLE_CACHE: dict[str, LiteralTable] = {}


def literal_table(commitments: Mapping[str, Any], literals: Mapping[str, Any] | Sequence[str] | LiteralTable | None = None) -> LiteralTable | None:
    """The record's :class:`LiteralTable` from the instance pin (``literals`` = the ``commitments.literals`` block, or a list of literal ids, or a
    table; else the block carried in ``commitments``); rebuilt from the ids and CHECKED against the pinned roots; ``None`` when nothing is pinned."""

    if isinstance(literals, LiteralTable):
        return literals
    pin: Any = literals if literals is not None else commitments.get("literals")
    if not pin:
        return None
    if isinstance(pin, Mapping):
        ids = [str(x) for x in (pin.get("literals") or [])]
    else:
        ids, pin = [str(x) for x in pin], None
    import json as _json

    key = _json.dumps(sorted(ids))
    lt = _LITERAL_TABLE_CACHE.get(key)
    if lt is None:
        lt = _LITERAL_TABLE_CACHE[key] = LiteralTable(ids)
    if pin is not None and pin.get("trees"):
        lt.check_pin(pin)
    return lt


@dataclass
class LiteralOracle:
    """The FIRST oracle of a record's chain: a ``Const<w>[…]`` operand gate is the literal tree's word (``prescribed-literal/v1``); without a
    pinned literal tree such a gate is a refusal BY NAME (``LITERAL_UNPINNED``), never ``None`` (so no builder mistakes it for an uncommitted
    interior word).  Any other gate: ``None`` (the next oracle answers)."""

    program: Any
    table: LiteralTable | None
    interior: bool = False
    name: str = "literal"

    def word(self, request_id: str, g: int):
        try:
            gate = self.program.circuit.gate(int(g))
        except Exception:  # noqa: BLE001 -- not a gate of this program: not ours to answer
            return None
        lit = _const_literal(gate)
        if lit is None:
            return None
        if self.table is None:
            raise _ck_elem().Refusal(f"LITERAL_UNPINNED: operand gate {int(g)} is the program literal {gate.prim.id}; the instance pins no "
                                     f"commitments.literals ({LITERAL_RULE}) -- the literal's committed word cannot be named")
        w = self.table.position(gate.prim.id)
        if w is None:
            raise _ck_elem().Refusal(f"LITERAL_UNPINNED: {gate.prim.id} (gate {int(g)}) is not among the pinned literals {self.table.ids}")
        return w

    def row_word(self, request_id: str, g: int):
        return None  # a literal is never a VU's output


class _LiteralOpener:
    """The leaf opener with the literal trees' leaves routed to the :class:`LiteralTable` (rebuilt, pinned), everything else to ``inner``."""

    def __init__(self, inner: Callable[[str], Any] | None, table: LiteralTable) -> None:
        self.inner, self.table = inner, table

    def __call__(self, cid: str) -> Any:
        if str(cid).startswith(LITERAL_TREE_PREFIX):
            return self.table.open(str(cid))
        if self.inner is None:
            raise _ck_elem().Refusal(f"{cid}: this record has no leaf opener")
        return self.inner(cid)


def table_leaf_name(cid: str) -> tuple[str, int] | None:
    """``(table name, chunk)`` of an E2 table leaf id ``table/<name>/chunk=<k>`` (``e2e.interior.tables.leaf_id``), else ``None``."""

    if not cid.startswith(TABLE_TREE_PREFIX) or "/chunk=" not in cid:
        return None
    name, _, k = cid[len(TABLE_TREE_PREFIX):].rpartition("/chunk=")
    return (name, int(k)) if k.isdigit() else None


def resolve_table_unit(located: Mapping[str, Any], record: Any) -> ResolvedUnit:
    """A :data:`TABLE_KINDS` VU as a ``ck_elem.ResolvedUnit``: the one gate of the spec's primitive, its operand(s) committed 32-bit words
    (E2's interior tree), and -- for a table kind -- the 64-bit row ``index_fn(value of operand 0)`` of the pinned table tree as the LAST input
    (``record.tables``: E2's ``TableWords`` over ``commitments.tables``, attached by :func:`open_record`).  Refusals by name: ``TABLE_UNPINNED``
    (the record's instance pins no ``commitments.tables`` / not this table), no opener (the row index is a function of the operand's committed
    value: the verifier-side derivation needs the archive's leaves, as E4's ``ck_ftz`` does), width or primitive mismatch."""

    ck = _ck_elem()
    located = located_mapping(located)
    kind = str(located["kind"])
    spec = TABLE_KINDS[kind]
    lo, hi = (int(x) for x in located["gate_interval"])
    if hi - lo != 1:
        raise ck.Refusal(f"{kind}: VU [{lo},{hi}) is not one gate")
    request_id = str(located["request_id"])
    gate = record.program.circuit.gate(lo)
    pname = getattr(gate.prim, "name", gate.prim.id)
    if pname != spec.prim:
        raise ck.Refusal(f"{kind}: gate {lo} is {gate.prim.id}, not {spec.prim}")
    if len(gate.operands) != spec.arity:
        raise ck.Refusal(f"{kind}: gate {lo} has {len(gate.operands)} operands, {spec.prim} takes {spec.arity}")
    name, _root = record.locator.root_of(lo)
    if name is None:
        raise ck.Refusal(f"{kind}: gate {lo} lies in no root call")
    elem = record.elem
    inputs = []
    for og in gate.operands:
        w = elem.word(request_id, int(og))
        if w is None:
            raise ck.Refusal(f"{kind}: operand {ck._describe_gate(record.program, int(og))} is an interior value the record does not commit")
        if w.width != spec.in_width:
            raise ck.Refusal(f"{kind}: operand {w.ref} is a {w.width}-bit committed word, {spec.gate} takes {spec.in_width}-bit values")
        inputs.append(w)
    if spec.table:
        tables = getattr(record, "tables", None)
        if tables is None or tables.table_row(spec.table, 0) is None:
            raise ck.Refusal(f"{kind}: TABLE_UNPINNED: the record's instance pins no commitments.tables[{spec.table!r}] (E2's table-layout/v1 "
                             f"tree; E6 pins the roots) -- the {spec.gate} row operand is not a committed word of this record")
        open_leaf = getattr(record, "open_leaf", None)
        if open_leaf is None:
            raise ck.Refusal(f"{kind}: the {spec.table} row index is {spec.index_fn}(value of {inputs[0].ref}) and this record cannot open the "
                             f"leaf (expectation-only handle): derive with the archive's leaves")
        x = inputs[0]
        try:
            value = int(_leaf_words(open_leaf(x.leaf), x.width)[x.flat])
        except Exception as e:  # noqa: BLE001 -- the opener's own refusal, named
            raise ck.Refusal(f"{kind}: cannot open {x.ref} for the table index: {type(e).__name__}: {e}") from e
        index = spec.index_of(value)
        row = tables.table_row(spec.table, index)
        if row is None:
            raise ck.Refusal(f"{kind}: {spec.table} row {index} (from {x.ref} = 0x{value:08x}) lies outside the pinned table")
        if int(row.width) != 64:
            raise ck.Refusal(f"{kind}: {spec.table} row {index} is a {row.width}-bit word, the row is (index << 32) | T[index]")
        inputs.append(ck.CommittedWord(str(row.tree), str(row.leaf), tuple(int(s) for s in row.shape), int(row.flat), 64))
    out = elem.row_word(request_id, lo)
    if out is None:
        raise ck.Refusal(f"{kind}: output {ck._describe_gate(record.program, lo)} is an interior value the record does not commit")
    if out.width != spec.out_width:
        raise ck.Refusal(f"{kind}: output {out.ref} is a {out.width}-bit committed word, {spec.gate} produces {spec.out_width} bits")
    ru = int(out.tree.split("/")[2]) if out.tree.startswith("ru/") else 0
    return ck.ResolvedUnit(int(located.get("vu_index", -1)), kind, spec, lo, name, request_id, tuple(inputs), out, ru)


def unit_subcircuit(u: ResolvedUnit) -> Any:
    """The body a resolved unit's obligation pins: the :class:`CallSpec`'s multi-gate ``subcircuit`` or ``ck_elem.kind_subcircuit`` (one gate)."""

    sub = getattr(u.spec, "subcircuit", None)
    if sub is not None:
        return sub
    if str(u.kind) in ftz_kinds() and u.spec is ftz_kinds()[str(u.kind)]:
        from . import ck_ftz as _ftz  # E4's one-gate body (its KindSpec spells in_widths, not in_width)

        return _ftz.kind_subcircuit(u.spec)
    return _ck_elem().kind_subcircuit(u.spec)


def expectation_unit(located: Mapping[str, Any], record: Any, *, label: str = "") -> ExpectedObligation:
    """``ck_elem.expectation`` for every registered kind (same construction: operand positions, output position, the kind's body digest)."""

    # Every kind goes through :func:`resolve_unit` (``ck_elem.resolve`` + :func:`alias_unit` for ``ck_elem.KINDS``): the expectation's
    # positions and body digest are the aliased unit's when operands repeat a word (D28), exactly what :func:`build_batch` proves.
    from veritor.interface.expected import ExpectedObligation as _EO

    u = resolve_unit(located, record)
    sub = unit_subcircuit(u)
    return _EO(tuple(w.expected() for w in u.inputs), (u.output.expected(),), sub.digest, (None,), {}, label or f"vu={u.vu_index} {u.kind} {u.target}")


class Builder(Protocol):
    kind: str
    family: str
    owner: str

    def build(self, located_unit: Mapping[str, Any], record: Any) -> Any: ...

    def expectation(self, located_unit: Mapping[str, Any], record: Any) -> ExpectedObligation: ...


def located_mapping(unit: Any) -> Mapping[str, Any]:
    """The located VU as the builders read it (a mapping with ``kind``, ``request_id``, ``gate_interval`` / ``gate``, ``vu_index``).  The
    verifier's locator gives a mapping; A5/6's ``builders.LocatedUnit`` dataclass (``vu_id``, ``kind``, ``request_id``, ``t``, ``layer``,
    ``head``, ..., ``extra``) is read field by field -- its ``extra["gate_interval"]`` when the locator put one there.  A unit that names no
    gate is a refusal at ``resolve_unit`` (naming the kind), never an attribute error."""

    if isinstance(unit, Mapping):
        return unit
    from dataclasses import fields, is_dataclass

    if is_dataclass(unit):
        d: dict[str, Any] = {f.name: getattr(unit, f.name) for f in fields(unit)}
        extra = d.pop("extra", None) or {}
        d.update({k: v for k, v in dict(extra).items() if k not in d})
        d.setdefault("vu_index", d.get("vu_id"))
        return d
    raise _ck_elem().Refusal(f"located unit of type {type(unit).__name__}: neither a mapping nor a LocatedUnit")


@dataclass(frozen=True)
class _ElemBuilder:
    kind: str
    family: str
    owner: str = "ck-elem/e7"
    #: verifier-side dispatch marker (``coverage_v1.builders.derives_expectation``): this builder derives the expected statement
    derives_expectation: bool = True

    def build(self, located_unit: Mapping[str, Any], record: Any) -> Any:
        ck_elem = _ck_elem()
        located_unit = located_mapping(located_unit)
        if str(located_unit.get("kind")) != self.kind:
            raise ck_elem.Refusal(f"BUILDERS[{self.kind!r}] asked to build a {located_unit.get('kind')!r} unit")
        return build_batch([resolve_unit(located_unit, record)], record)

    def expectation(self, located_unit: Mapping[str, Any], record: Any) -> ExpectedObligation:
        located_unit = located_mapping(located_unit)
        if str(located_unit.get("kind")) != self.kind:
            raise ValueError(f"BUILDERS[{self.kind!r}] asked for the expectation of a {located_unit.get('kind')!r} unit")
        return expectation_unit(located_unit, record)

    def subcircuit_digest(self) -> str | None:
        """The kind's ONE canonical body digest -- ``None`` for :data:`PV_KIND`, which has two bodies (``tc_dot16`` / ``tc_dot16_0`` by the unit's
        accumulator; the expectation the builder derives carries the digest; ``dispatch_kind_program`` then reports that one)."""

        if self.kind == PV_KIND:
            return None
        spec = kind_spec(self.kind)
        sub = getattr(spec, "subcircuit", None)
        return (sub if sub is not None else _ck_elem().kind_subcircuit(spec)).digest


#: [D12, 2026-09-11 21:35Z] CK-ELEM's name keeps CK-ELEM's meaning: the builders of the :data:`BOUNDARY_KINDS` -- the kinds a record that
#: commits only operator boundaries (REC-P1-B0, attempt 6) can build.  The interior kinds are :data:`INTERIOR_BUILDERS`; classification of a
#: located unit is still by POSITION at resolve time (``resolve_unit``: an operand/output word the record does not commit is a refusal naming
#: the gate), so on a record without an ``interior_layout`` pin an interior-kind VU is ``uncovered`` with the old reason, never "unsupported".
BUILDERS: dict[str, Builder] = {k: _ElemBuilder(k, ELEM_FAMILIES[k]) for k in BOUNDARY_KINDS}
#: kind -> builder for the interior kinds of ``ck_elem.KINDS`` (7): buildable exactly when a word oracle resolves their interior words
#: (E2's ``interior/<req>/<step>`` trees for TINY, E8's own trees for CORE); refuse by name otherwise
INTERIOR_BUILDERS: dict[str, Builder] = {k: _ElemBuilder(k, fam) for k, fam in ELEM_FAMILIES.items() if k not in BOUNDARY_KINDS}
#: kind -> builder for the :data:`EXTRA_KINDS` (kinds ``ck_elem.KINDS`` does not list; same gate as a listed sibling form)
EXTRA_BUILDERS: dict[str, Builder] = {k: _ElemBuilder(k, fam) for k, fam in EXTRA_FAMILIES.items()}
#: [23:20Z] kind -> builder for the :data:`CALL_KINDS` (``DotBf16_v1{K=64}|call``: A2's GemmAccumulator body in ``veritor.tc-ampere-bf16@2``, the
#: approved guests' family; x / w words committed at the operator boundary, the FP32 score through E2's interior tree); refuses by name otherwise
CALL_BUILDERS: dict[str, Builder] = {k: _ElemBuilder(k, fam, owner="e7") for k, fam in CALL_FAMILIES.items()}
#: [23:40Z] the PV step (``AmpereBF16TcDot16_v1|batch``): one ``tc_dot16`` / ``tc_dot16_0`` gate, same family; padded units refuse by name
PV_BUILDERS: dict[str, Builder] = {PV_KIND: _ElemBuilder(PV_KIND, TC_FAMILY, owner="e7")}
#: [D23] the four MUFU table kinds (``veritor.reduce-mufu@1``; the row through E2's pinned table trees)
TABLE_BUILDERS: dict[str, Builder] = {k: _ElemBuilder(k, fam, owner="e7") for k, fam in TABLE_FAMILIES.items()}
#: [NIGHT G1] composed-body/v1 kinds (Bf16Add_v1|batch for Qwen2.5-0.5B)
COMPOSED_BUILDERS: dict[str, Builder] = {k: _ElemBuilder(k, fam, owner="e7") for k, fam in COMPOSED_FAMILIES.items()}
#: this module's whole contribution to the ONE registry ``coverage_v1.builders.BUILDERS`` (12 kinds; W-A9-12: every kind with an approved
#: checker has a registered builder, and the verifier's derivation never labels one unsupported)
ALL_BUILDERS: dict[str, Builder] = {**BUILDERS, **INTERIOR_BUILDERS, **EXTRA_BUILDERS, **CALL_BUILDERS, **PV_BUILDERS, **TABLE_BUILDERS, **COMPOSED_BUILDERS}


def dispatch_expectation(located_unit: Mapping[str, Any], record: Any) -> ExpectedObligation:
    """What ``verifier_v3.expected.RecordPartition.locate`` does for a kind in ``BUILDERS`` (dispatch patch): the
    kind's builder derives the expectation from the located unit and the record's committed id lists."""

    kind = str(located_unit.get("kind"))
    b = ALL_BUILDERS.get(kind)
    if b is None:
        raise KeyError(f"{kind}: no builder registered (BUILDERS)")
    return b.expectation(located_unit, record)


def dispatch_kind_program(kind: str, expectation: Any = None) -> tuple[str, str]:
    """What ``verifier_v3.expected.RecordCircuit.kind_program`` returns for a kind in ``BUILDERS``: the builder's family
    and the kind's canonical body digest (the expectation's own digest when it carries one; they must agree)."""

    b = ALL_BUILDERS.get(kind)
    if b is None:
        raise KeyError(f"{kind}: no builder registered (BUILDERS)")
    digest = getattr(b, "subcircuit_digest", lambda: None)()
    declared = getattr(expectation, "subcircuit", None)
    if declared is not None and digest is not None and str(declared).lower() != str(digest).lower():
        if is_aliased_body(kind, declared, expectation):
            return b.family, str(declared).lower()  # D28: the unit's aliased body (one committed word read by several operands)
        raise ValueError(f"{kind}: expectation subcircuit {str(declared)[:12]} != the builder's body {str(digest)[:12]}")
    return b.family, str(digest if digest is not None else declared)


def alias_digests(kind: str, incoming: int | None = None) -> dict[str, tuple[int, ...]]:
    """Every :data:`ALIAS_RULE` body digest of a one-gate ``kind`` -> its wiring: the wirings of the kind's arity onto ``1 .. arity-1``
    distinct words (``incoming`` narrows to bodies with that many incoming values).  Empty for multi-gate kinds and arity-1 kinds."""

    from itertools import product

    spec = kind_spec(kind)
    if isinstance(spec, CallSpec) or spec.arity < 2:
        return {}
    out: dict[str, tuple[int, ...]] = {}
    if isinstance(spec, PvSpec):
        # [prescribed-literal/v1] the PV step's structural wirings: acc (committed, or dropped by the tc_dot16_0 body), pb[0..16), pa[0..16);
        # the k-step past the visible columns pads the LAST 16-m pb and/or pa words with ONE literal word (ZERO16) -- enumerated, not searched
        for acc_zero in (True, False):
            base = PvSpec(acc_zero)
            for m in range(1, 16):
                for side in ("both", "pb", "pa"):
                    tokens: list[object] = ([] if acc_zero else ["acc"])
                    tokens += [("pb", i) if i < m or side == "pa" else "L" for i in range(16)]
                    tokens += [("pa", i) if i < m or side == "pb" else "L" for i in range(16)]
                    order: dict[object, int] = {}
                    wiring = tuple(order.setdefault(t, len(order)) for t in tokens)
                    if incoming is not None and len(order) != incoming:
                        continue
                    out[AliasSpec(base, wiring).subcircuit.digest.lower()] = wiring
        return out
    for wiring in product(range(spec.arity), repeat=spec.arity):
        # a canonical wiring: distinct words numbered by first occurrence, at least one repeat
        seen: list[int] = []
        ok = True
        for j in wiring:
            if j == len(seen):
                seen.append(j)
            elif j > len(seen):
                ok = False
                break
        if not ok or len(seen) == spec.arity or (incoming is not None and len(seen) != incoming):
            continue
        out[AliasSpec(spec, wiring).subcircuit.digest.lower()] = wiring
    return out


def is_aliased_body(kind: str, declared: Any, expectation: Any = None) -> bool:
    """Is ``declared`` the digest of one of ``kind``'s :data:`ALIAS_RULE` bodies (for the expectation's incoming count when it carries one)?"""

    n = None
    incoming = getattr(expectation, "incoming", None)
    if incoming is not None:
        n = len(incoming)
    try:
        return str(declared).lower() in alias_digests(kind, n)
    except (KeyError, ValueError):
        return False


def families_of(builders: Mapping[str, Builder] = ALL_BUILDERS) -> tuple[str, ...]:
    """The interface families the registered builders emit (the verifier accepts a proof object of any of them)."""

    return tuple(sorted({b.family for b in builders.values()}))


def kinds_of_family(family: str, builders: Mapping[str, Builder] = ALL_BUILDERS) -> tuple[str, ...]:
    return tuple(sorted(k for k, b in builders.items() if b.family == family))


# --- program-agnostic record handle (E7, EOD 2026-09-11) ---------------------------------------------------------------------------------


@runtime_checkable
class WordOracle(Protocol):
    """Gate index in the pinned program -> the committed position of its value, or ``None`` when this oracle does not hold it.

    The position is a ``ck_elem.CommittedWord(tree, leaf, shape, flat, width)``: tree id (as ``commitments.json`` names it), the leaf's
    canonical id, the leaf's shape, the word's row-major flat index in the leaf, and the word width (16 for a bf16 leaf, 32 for a
    float32/int32 leaf -- the leaf is addressed as ``packed/v1:verity-vllm/leaf/v1:u<width>le:<count>:<flat>``, so a leaf must be uniform
    in width).  Oracles are program-agnostic: E3's expected-statement derivation (interface 5) supplies one for TINY's boundary +
    interior leaves; E8's CORE adapter supplies its own.  ``interior`` marks an oracle over newly committed values (interface 3: a gate
    with a boundary position is never interior -- :class:`OracleChain` asks the oracles in order and refuses a second answer)."""

    def word(self, request_id: str, g: int) -> CommittedWord | None: ...


#: marker shape of a position whose leaf shape / word width the oracle does not know (E3's ``(tree, leaf id, flat)`` spelling); completed
#: from the leaf's opening by :class:`OracleChain` before ``ck_elem.resolve`` reads the width
UNKNOWN_SHAPE: tuple[int, ...] = ()


@dataclass
class LocateOracle:
    """A :class:`WordOracle` over a plain callable ``locate(request_id, g)`` returning ``None``, a ``CommittedWord``, a tuple
    ``(tree, leaf, flat)`` (E3's interface-5 spelling: shape and width are then taken from the leaf's opening / dtype) or
    ``(tree, leaf, shape, flat[, width])``, or a mapping with keys ``tree, leaf, flat`` and optionally ``shape`` (or ``count``), ``width``."""

    locate: Callable[[str, int], Any]
    name: str = "locate"
    default_width: int | None = None

    def word(self, request_id: str, g: int) -> CommittedWord | None:
        CW = _ck_elem().CommittedWord
        r = self.locate(request_id, g)
        if r is None:
            return None
        if isinstance(r, CW):
            return r
        width_default = int(self.default_width) if self.default_width is not None else 0  # 0 = unknown, completed from the leaf
        if isinstance(r, Mapping):
            shape = r.get("shape")
            if shape is None and "count" in r:
                shape = (int(r["count"]),)
            return CW(str(r["tree"]), str(r["leaf"]), tuple(int(x) for x in shape) if shape is not None else UNKNOWN_SHAPE, int(r["flat"]),
                      int(r.get("width", width_default)))
        if isinstance(r, (tuple, list)):
            if len(r) == 3:
                tree, leaf, flat = r
                return CW(str(tree), str(leaf), UNKNOWN_SHAPE, int(flat), width_default)
            tree, leaf, shape, flat, *rest = r
            return CW(str(tree), str(leaf), tuple(int(x) for x in shape) if shape is not None else UNKNOWN_SHAPE, int(flat), int(rest[0]) if rest else width_default)
        raise TypeError(f"{self.name}: locate returned {type(r).__name__}, not None / CommittedWord / tuple / mapping")


class OracleChain:
    """The ``record.elem`` interface ``ck_elem.resolve`` reads (``word`` for operands, ``row_word`` for the output), over an ordered list
    of oracles.  Interface 3 (no gate ever gets a second value): if two oracles answer for one gate with DIFFERENT positions the chain
    refuses; the same position twice (an oracle wrapping another) is tolerated.  ``row_word`` excludes the weights tree (a VU's output is
    never a weight).  A position with an unknown shape / width (:data:`UNKNOWN_SHAPE`, width 0) is completed through ``describe(leaf_id) ->
    (shape, dtype)`` (the opener, prover side; the verifier's committed leaf table otherwise) -- without it such a position is a refusal."""

    def __init__(self, oracles: Sequence[WordOracle], *, weights_tree: str = "weights",
                 describe: Callable[[str], tuple[Sequence[int], str]] | None = None) -> None:
        self.oracles = tuple(oracles)
        self.weights_tree = weights_tree
        self.describe = describe
        self._described: dict[str, tuple[tuple[int, ...], int]] = {}

    def _complete(self, w: CommittedWord, g: int) -> CommittedWord:
        if tuple(w.shape) != UNKNOWN_SHAPE and int(w.width) > 0:
            return w
        ck = _ck_elem()
        if self.describe is None:
            raise ck.Refusal(f"gate {g}: position {w.tree}:{w.leaf}[{w.flat}] came without shape/width and the record has no leaf describer")
        d = self._described.get(w.leaf)
        if d is None:
            shape, dtype = self.describe(w.leaf)
            d = (tuple(int(x) for x in shape), leaf_word_width(dtype))
            self._described[w.leaf] = d
        shape, width = d
        if int(w.width) > 0 and int(w.width) != width:
            raise ck.Refusal(f"gate {g}: {w.leaf} located as a {w.width}-bit word but the leaf packs {width}-bit words")
        return ck.CommittedWord(w.tree, w.leaf, shape if tuple(w.shape) == UNKNOWN_SHAPE else tuple(w.shape), int(w.flat), width)

    def word(self, request_id: str, g: int) -> CommittedWord | None:
        Refusal = _ck_elem().Refusal
        found: CommittedWord | None = None
        for o in self.oracles:
            w = o.word(request_id, g)
            if w is None:
                continue
            w = self._complete(w, g)
            if found is not None and (w.tree, w.leaf, w.flat) != (found.tree, found.leaf, found.flat):
                raise Refusal(f"gate {g}: two committed positions ({found.tree}:{found.ref} and {w.tree}:{w.ref}) -- a gate has one value (interface 3)")
            found = found or w
        return found

    def row_word(self, request_id: str, g: int) -> CommittedWord | None:
        w = self.word(request_id, g)
        if w is not None and w.tree == self.weights_tree:
            return None
        return w


class _PathLocator:
    """The slice of ``ProgramLocator`` ``ck_elem.resolve`` uses when the record supplies none: ``root_of(g)`` = the enclosing root call
    (its node name -- ``stage3`` for E8's CORE circuit, ``t=<t>/layers.<L>/<role>`` for Serve -- or ``call<k>``), the statement's ``Target``
    label; a gate in no root call gets the program's own name.  No input gates (a program-agnostic record resolves inputs through its
    oracles)."""

    input_gates = 0

    def __init__(self, program: Any) -> None:
        self.program = program
        self._calls: list[tuple[int, int, str, Any]] | None = None

    def _root_calls(self) -> list[tuple[int, int, str, Any]]:
        if self._calls is None:
            calls = []
            try:
                for k, c in enumerate(self.program.circuit.root_calls()):
                    name = getattr(getattr(c, "node", None), "name", None) or f"call{k}"
                    calls.append((int(c.lo), int(c.hi), str(name), c))
            except Exception:  # noqa: BLE001 -- a program without root calls (all gates at the root) has one region
                calls = []
            self._calls = sorted(calls)
        return self._calls

    def root_of(self, g: int) -> tuple[str | None, Any]:
        g = int(g)
        for lo, hi, name, call in self._root_calls():
            if lo <= g < hi:
                return name, call
        try:
            gate = self.program.circuit.gate(g)
            path = "/".join(str(p) for p in (getattr(gate, "path", None) or ()))
        except Exception:  # noqa: BLE001
            path = ""
        return (path or "program"), None

    def locate_input(self, g: int) -> dict[str, Any]:
        return {"kind": "none"}


def _generic_record_base():
    return _ck_elem().RecordHandle


class GenericRecord:
    """A program-agnostic ``ck_elem.RecordHandle``: the pinned program, ``commitments.json`` (``trees``: root / leaf_count / ids_digest per
    tree, ``profile_id``), the word oracles (:class:`OracleChain`) and the leaf opener.  Constructed through :func:`open_record`; this
    class exists so the prover's import graph (W-A9-11) sees no ``src/veritor`` import at module load."""

    def __new__(cls, *a: Any, **k: Any):  # pragma: no cover - documentation
        raise TypeError("use open_record(...)")


def tree_records(commitments: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Every committed tree ``commitments.json`` names, as ``tree id -> {root, leaf_count, ids_digest, ...}``: the phase-0 ``trees`` table
    plus E2's ``interior_roots["<req>/<step>"]`` (interface 4) under their tree id ``interior/<req>/<step>``.  A tree listed in both with
    different roots is a refusal (one commitment per tree)."""

    out: dict[str, dict[str, Any]] = {str(k): dict(v) for k, v in (commitments.get("trees") or {}).items()}
    for key, rec in (commitments.get("interior_roots") or {}).items():
        key = str(key)
        if isinstance(rec, str):  # E8's spelling: "<req>/<step>" -> root hex, the tree itself listed under trees["ru/<req>/<step>"]
            listed = [t for t in (f"ru/{key}", f"interior/{key}", key) if t in out]
            if listed and str(out[listed[0]].get("root")).lower() != rec.lower():
                raise _ck_elem().Refusal(f"{listed[0]}: commitments.json lists two different roots (trees vs interior_roots)")
            continue
        tid = key if "/" in key and key.split("/")[0] in ("interior", "ru") else f"interior/{key}"
        if tid in out and str(out[tid].get("root")).lower() != str(rec.get("root")).lower():
            raise _ck_elem().Refusal(f"{tid}: commitments.json lists two different roots (trees vs interior_roots)")
        out.setdefault(tid, dict(rec))
    for name, rec in (commitments.get("tables") or {}).items():  # [D23] E2's pinned MUFU table trees: instance-wide, tree id table/<name>
        tid = str(rec.get("tree") or f"{TABLE_TREE_PREFIX}{name}")
        if tid in out and str(out[tid].get("root")).lower() != str(rec.get("root")).lower():
            raise _ck_elem().Refusal(f"{tid}: commitments lists two different roots (trees vs tables)")
        out.setdefault(tid, dict(rec))
    for tid, rec in ((commitments.get("literals") or {}).get("trees") or {}).items():  # [prescribed-literal/v1] the pinned literal trees
        tid = str(tid)
        if tid in out and str(out[tid].get("root")).lower() != str(rec.get("root")).lower():
            raise _ck_elem().Refusal(f"{tid}: commitments lists two different roots (trees vs literals)")
        out.setdefault(tid, dict(rec))
    return out


def table_words(commitments: Mapping[str, Any], tables: Mapping[str, Mapping[str, Any]] | None = None) -> Any | None:
    """E2's ``TableWords`` over the pinned ``commitments.tables`` (``tables`` given explicitly -- the instance's block -- or carried in
    ``commitments``); ``None`` when nothing is pinned (a legacy record: the table kinds then refuse ``TABLE_UNPINNED``, D12)."""

    pins = dict(tables) if tables else dict(commitments.get("tables") or {})
    if not pins:
        return None
    import json as _json

    key = _json.dumps(pins, sort_keys=True)
    tw = _TABLE_WORDS_CACHE.get(key)
    if tw is None:
        from e2e.interior.tables import TableWords

        tw = _TABLE_WORDS_CACHE[key] = TableWords(pins)  # the rebuilt table trees are pure functions of the pin: ONE copy per pin (census: one per process)
    return tw


_TABLE_WORDS_CACHE: dict[str, Any] = {}


class _TableOpener:
    """The record's leaf opener with E2's table leaves routed to ``TableWords.open`` (``table/<name>/chunk=<k>`` -> the opening of leaf ``k``,
    rebuilt in memory and checked against the pinned root); everything else to the record's own opener."""

    def __init__(self, inner: Callable[[str], Any] | None, tables: Any) -> None:
        self.inner = inner
        self.tables = tables

    def __call__(self, cid: str) -> Any:
        named = table_leaf_name(str(cid))
        if named is not None and self.tables is not None:
            name, k = named
            opening, _flat, _word = self.tables.open(name, k * self.tables._rows_per_leaf(name))
            return opening
        if self.inner is None:
            raise _ck_elem().Refusal(f"{cid}: this record has no leaf opener")
        return self.inner(cid)


class _FtzWords:
    """E4's ``ck_ftz.WordResolver`` over this record: ``word`` through the oracle chain, ``table_row`` through E2's ``TableWords`` (D23: the
    FTZ table kinds ``MufuEx2Ftz`` / ``Fa2InvSum`` resolve their rows through the same pinned trees, in E4's own ``CommittedWord``)."""

    def __init__(self, handle: Any, tables: Any) -> None:
        self.handle = handle
        self.tables = tables

    def word(self, request_id: str, g: int) -> Any | None:
        return self.handle.elem.word(request_id, int(g))

    def table_row(self, table: str, index: int) -> Any | None:
        if self.tables is None:
            return None
        row = self.tables.table_row(str(table), int(index))
        if row is None:
            return None
        try:
            from .ck_ftz import CommittedWord as CW  # E4's type (same field order); loaded only when E4's builder runs (verifier side)
        except ImportError:
            return row
        return CW(str(row.tree), str(row.leaf), tuple(int(s) for s in row.shape), int(row.flat), int(row.width))


def _with_all_trees(commitments: Mapping[str, Any]) -> dict[str, Any]:
    view = dict(commitments)
    view["trees"] = tree_records(commitments)
    return view


class _CachedOpener:
    """``open(canonical_id) -> Opening`` asked once per leaf (the oracle chain's describer and the batch builder share the opening)."""

    def __init__(self, opener: Callable[[str], Any]) -> None:
        self.opener = opener
        self.openings: dict[str, Any] = {}

    def __call__(self, cid: str) -> Any:
        o = self.openings.get(cid)
        if o is None:
            o = self.opener(cid)
            self.openings[cid] = o
        return o

    def describe(self, cid: str) -> tuple[tuple[int, ...], str]:
        o = self(cid)
        return tuple(int(x) for x in o.shape), str(o.dtype)


def open_record(program: Any, commitments: Mapping[str, Any], *, oracles: Sequence[WordOracle], opener: Callable[[str], Any] | None,
                C: Mapping[str, Any] | None = None, locator: Any = None, committed_ids: Mapping[str, Sequence[str]] | None = None,
                profile_id: str | None = None, reader: Any = None, include_boundary_rows: bool = False,
                describe: Callable[[str], tuple[Sequence[int], str]] | None = None, tables: Mapping[str, Mapping[str, Any]] | None = None,
                literals: Mapping[str, Any] | Sequence[str] | LiteralTable | None = None):
    """The record as a registered builder reads it, from program-agnostic parts.

    * ``oracles``: :class:`WordOracle` objects asked in order for every operand / output gate (E8: one oracle over its trees; TINY: E3's
      locator over boundary + ``interior/<req>/<step>`` leaves).  With ``include_boundary_rows`` (TINY on a vLLM capture, ``reader`` given),
      ``ck_elem.ElemLocator`` (weights + operator-boundary rows via ``CircuitMap``) is asked FIRST, then ``oracles``.
    * ``opener(canonical_id) -> verity_vllm.commit.Opening`` -- the prover's opening of a committed leaf (``tree``, ``root``,
      ``leaf_count``, ``index``, ``canonical_id``, ``dtype``, ``shape``, ``value`` (raw little-endian bytes), ``path``).  On the verifier
      side (expectations only) pass ``None``: openings are then a refusal, positions still derive.
    * ``describe(canonical_id) -> (shape, dtype)`` completes positions given as ``(tree, leaf, flat)``; defaults to the opener.
    * ``committed_ids``: tree id -> canonical ids in tree order (the verifier's, checked against ``ids_digest``); the prover side may omit
      them when ``reader`` is given.
    * ``commitments``: ``commitments.json`` as written (``trees`` + E2's ``interior_roots``); the handle sees every tree under its id.
    * ``tables`` [D23]: the instance's ``commitments.tables`` (E2's ``table-layout/v1`` pins; instance-wide, so ``commitments.json`` need not
      repeat them -- when it does, they are read from there).  Attaches ``handle.tables`` (E2's ``TableWords``) and routes ``table/<name>/chunk=<k>``
      leaves through it in ``open_leaf``; also ``handle.ftz`` = E4's ``WordResolver`` over the chain + the tables.  Absent -> the table kinds
      refuse ``TABLE_UNPINNED``; nothing else changes (legacy records, D12).
    * ``literals`` [``prescribed-literal/v1``]: the instance's ``commitments.literals`` pin (or the literal ids; or a :class:`LiteralTable`),
      else the block in ``commitments``.  Attaches ``handle.literals`` (the trees rebuilt from the ids and checked against the pin), puts a
      :class:`LiteralOracle` FIRST in the chain (a ``Const`` operand gate -> the literal tree's word) and routes ``literal/w=<w>/chunk=0`` leaves
      through it.  Absent -> a unit with a literal operand refuses ``LITERAL_UNPINNED``; nothing else changes.
    """

    ck = _ck_elem()
    tw = table_words(commitments, tables)
    lt = literal_table(commitments, literals)
    view = _with_all_trees({**commitments, **({"tables": dict(tables)} if tables else {}), **({"literals": lt.pin()} if lt is not None else {})})
    handle = ck.RecordHandle(program, dict(C or {}), locator if locator is not None else _PathLocator(program), view, reader,
                             {t: list(ids) for t, ids in (committed_ids or {}).items()}, str(profile_id if profile_id is not None else commitments.get("profile_id", "")))
    if tw is not None:
        opener = _TableOpener(opener, tw)
    if lt is not None:
        opener = _LiteralOpener(opener, lt)
    handle.tables = tw
    handle.literals = lt
    cached = _CachedOpener(opener) if opener is not None else None
    chain: list[WordOracle] = [LiteralOracle(program, lt)]
    if include_boundary_rows:
        chain.append(ck.ElemLocator(handle))
    chain.extend(oracles)
    handle._elem = OracleChain(chain, weights_tree=ck.WEIGHTS_TREE, describe=describe if describe is not None else (cached.describe if cached else None))
    handle.ftz = _FtzWords(handle, tw)  # E4's builders read record.ftz (word + table_row); the tables half is E2's, the words the chain's
    if cached is not None:
        handle.open_leaf = cached  # type: ignore[method-assign]  # ck_elem.resolve reads record.elem.word / row_word; build reads open_leaf
    elif reader is not None:
        from verity_vllm.commit import TreeCache

        handle._cache = TreeCache(reader)
    return handle


# --- batch building (one statement per family, width-aware leaves, interior trees) --------------------------------------------------------

_OWNER_WEIGHTS, _OWNER_BOUNDARY = -2, -1


def owner_of(tree: str) -> int:
    """``Root.owner`` of a committed tree: weights -2, ``boundary/<req>`` -1, ``ru/<req>/<n>`` n, ``interior/<req>/<step>`` step (E2's sibling
    tree of the RU at that step, interface 4).  Any other tree id is a refusal (a builder must not invent a tree the verifier cannot name)."""

    if tree == "weights":
        return _OWNER_WEIGHTS
    if tree.startswith(TABLE_TREE_PREFIX):
        return TABLE_OWNER  # [D23] a pinned MUFU table tree: request-independent, like the weights
    if tree.startswith(LITERAL_TREE_PREFIX):
        return LITERAL_OWNER  # [prescribed-literal/v1] the pinned literal tree: request-independent, like the weights
    parts = tree.split("/")
    if parts[0] == "boundary":
        return _OWNER_BOUNDARY
    if len(parts) == 3 and parts[0] in ("ru", "interior") and parts[2].isdigit():
        return int(parts[2])
    raise _ck_elem().Refusal(f"tree {tree!r}: no owner mapping (weights | table/<name> | literal/w=<w> | boundary/<req> | ru/<req>/<n> | interior/<req>/<step>)")


_WORD_WIDTH_OF_DTYPE = {"bfloat16": 16, "float16": 16, "int16": 16, "uint16": 16, "float32": 32, "int32": 32, "uint32": 32,
                        "uint8": 8, "int8": 8, "uint64": 64, "int64": 64, "float64": 64}


def leaf_word_width(dtype: str) -> int:
    """Word width of a uniform leaf: the dtype's element width (E2's chunk dtypes ``uint8|uint16|uint32|uint64`` and ``u<w>`` for other
    ``Value<w>`` gates; the boundary trees' ``bfloat16`` / ``float32``).  The packed schema addresses the leaf as ``u<width>le`` words."""

    d = str(dtype)
    w = _WORD_WIDTH_OF_DTYPE.get(d)
    if w is None and d.startswith("u") and d[1:].isdigit():
        w = int(d[1:])
    if w is None or w <= 0 or w > 64:
        raise _ck_elem().Refusal(f"leaf dtype {dtype!r} is not laid out as uniform words of 8..64 bits")
    return w


def _leaf_type(opening: Any, width: int):
    from veritor.interface.typed_obligation import Array, Bits

    ty: Any = Bits(width)
    for extent in reversed(tuple(opening.shape)):
        ty = Array(ty, int(extent))
    return ty


def _leaf_words(opening: Any, width: int) -> Any:
    import numpy as np

    dt = {8: "<u1", 16: "<u2", 32: "<u4", 64: "<u8"}.get(int(width))
    if dt is None:
        nbytes = (int(width) + 7) // 8
        raw = bytes(opening.value)
        return [int.from_bytes(raw[i:i + nbytes], "little") for i in range(0, len(raw) - len(raw) % nbytes, nbytes)]
    return np.frombuffer(opening.value, dtype=dt)


def build_batch(units: Sequence[ResolvedUnit], record: Any, *, name: str = "e7-batch", authenticate: bool = True) -> Built:
    """One typed statement (typed-obligation/v0 -> format 3) over ``units`` of ONE family: the program-agnostic form of
    ``ck_elem.build_many`` -- leaves typed by their committed dtype (``Array<..., Value<16|32>>``, so a float32 interior leaf is addressed
    as 32-bit words), ``interior/<req>/<step>`` trees owned by their step, openings authenticated against ``commitments.json`` before any
    word is read.  Obligations are keyed ``(session = request, compiled = profile, unit = ordinal)`` as the serving exporter keys them."""

    ck = _ck_elem()
    from veritor.interface.lowering import statement_bytes_v3, to_statement_v3
    from veritor.interface.typed_obligation import (
        Boundary,
        DeclaredInterface,
        Leaf,
        Obligation,
        Reference,
        Root,
        Subcircuit,
        Target,
        TypedObligationSet,
        validate,
    )
    from veritor.protocol.proofs.packed import encode_packed_witness

    from verity_vllm.typed_obligation_export import authenticate_opening

    if not units:
        raise ck.Refusal("no units to build")
    families = {u.spec.family for u in units}
    if len(families) != 1:
        raise ck.Refusal(f"units of several families in one statement: {sorted(families)}")
    family_id = next(iter(families))
    family = ck.family_of(family_id)
    if getattr(record, "phantom", False):  # [D32.2] a census record has identities and zero bits, no commitment to authenticate against
        authenticate = False
    openings: dict[str, Any] = {}
    widths: dict[str, int] = {}
    roots: dict[str, Root] = {}
    references: dict[str, Reference] = {}
    subcircuits: dict[str, Subcircuit] = {}
    how: dict[str, str] = {}

    commitments_view = _with_all_trees(record.commitments)  # boundary/RU/weights trees + E2's interior_roots under their tree ids

    def take(cid: str):
        o = openings.get(cid)
        if o is None:
            o = record.open_leaf(cid)
            if authenticate:
                how[cid] = authenticate_opening(commitments_view, o)
            openings[cid] = o
            widths[cid] = leaf_word_width(o.dtype)
            if o.tree not in roots:
                rec = commitments_view["trees"].get(o.tree)
                if rec is None:
                    raise ck.Refusal(f"{o.tree}: the opening names a tree commitments.json does not list (trees / interior_roots)")
                if bytes.fromhex(rec["root"]) != o.root or int(rec["leaf_count"]) != o.leaf_count:
                    raise ck.Refusal(f"{o.tree}: opening root/count disagree with commitments.json")
                roots[o.tree] = Root(o.tree, owner_of(o.tree), rec["ids_digest"], o.root.hex(), o.leaf_count, ck.VLLM_FRAMING)
        return o

    def ref_of(w: CommittedWord) -> tuple[str, int]:
        o = take(w.leaf)
        if o.tree != w.tree:
            raise ck.Refusal(f"{w.leaf}: committed in {o.tree}, located in {w.tree}")
        shape = tuple(int(x) for x in o.shape)
        if shape != tuple(w.shape):
            raise ck.Refusal(f"{w.leaf}: opening shape {shape} != located shape {tuple(w.shape)}")
        if widths[w.leaf] != w.width:
            raise ck.Refusal(f"{w.ref}: located as a {w.width}-bit word, the leaf ({o.dtype}) packs {widths[w.leaf]}-bit words")
        words = _leaf_words(o, w.width)
        if not 0 <= w.flat < len(words):
            raise ck.Refusal(f"{w.ref}: word {w.flat} outside the leaf ({len(words)} words)")
        if w.ref not in references:
            references[w.ref] = Reference(w.ref, w.leaf, w.offset)
        return w.ref, int(words[w.flat])

    pending: list[tuple[str, str, Subcircuit, tuple[str, ...], tuple[str, ...], tuple[str, ...], list[int], int]] = []
    for u in units:
        sub = unit_subcircuit(u)  # the CallSpec's multi-gate body (GemmAccumulator) or ck_elem's one-gate body
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
    built: list[tuple[Obligation, list[int], str]] = []
    for session, target, sub, inc, out, trees, row, ru in sorted(pending, key=lambda p: (p[0], p[1])):
        unit = ordinals.get(session, 0)
        ordinals[session] = unit + 1
        binding = ck._binding(session, record.profile_id, unit, ru)
        built.append((Obligation(binding, sub.digest, trees, Boundary(inc, out), DeclaredInterface(inc, out), Target("call", target)), row, target))
    built.sort(key=lambda b: (b[0].binding.session, b[0].binding.compiled, b[0].binding.unit))
    leaves = [Leaf(o.canonical_id, o.tree, o.index, o.index, _leaf_type(o, widths[cid])) for cid, o in openings.items()]
    obj = TypedObligationSet(
        family, (),
        tuple(roots[t] for t in sorted(roots)),
        tuple(sorted(leaves, key=lambda v: v.id)),
        tuple(sorted(references.values(), key=lambda r: r.ref)),
        tuple(sorted(subcircuits.values(), key=lambda s: s.digest)),
        tuple(o for o, _, _ in built),
    )
    ck.ensure_families()
    validate(obj)
    statement_v3 = to_statement_v3(obj)
    statement = statement_v3.expand()
    st_bytes = statement_bytes_v3(obj)
    witness = ck._packed_witness(openings.values(), statement)
    proof = encode_packed_witness(witness)
    record_json = ck._record_json(obj, [row for _, row, _ in built], proof, openings, how, units, name)
    record_json["provenance"]["builder"] = "e2e.coverage_v1.elem_builders.build_batch"
    # [D25] the record describes the statement it emits: statement.bin = st_bytes, and its digest is what the proof's public h_L will be
    # (steps.proof_record_consistency / reverify.py V4 refuse otherwise) -- a builder that disagrees with itself never ships a record
    import hashlib

    if str(record_json.get("statement_digest")) != hashlib.sha256(st_bytes).hexdigest() or int(record_json.get("statement_format", 2)) != 3:
        raise ck.Refusal(f"{name}: composed.json#statement_digest {str(record_json.get('statement_digest'))[:16]} is not sha256 of the format-3 statement the "
                         f"record emits ({hashlib.sha256(st_bytes).hexdigest()[:16]}) -- record refused (D25)")
    return ck.Built(obj, tuple(units), openings, statement, st_bytes, proof, record_json, family_id)


@dataclass
class Batch:
    """One proof object to be: the family, the kinds in it, the VU identities it covers (in obligation order), the built statement."""

    index: int
    family: str
    kinds: tuple[str, ...]
    vu_indices: tuple[int, ...]
    built: Built
    name: str = ""

    def summary(self) -> dict[str, Any]:
        b = self.built
        return {"chunk": self.index, "name": self.name, "family": self.family, "kinds": list(self.kinds), "units": len(self.vu_indices),
                "covers_vu_indices": list(self.vu_indices), "obligations": len(b.obj.obligations), "leaves": len(b.obj.leaves),
                "roots": [r.id for r in b.obj.roots], "h_O": b.record["digest"], "statement_bytes": len(b.statement_bytes), "witness_bytes": len(b.witness_bytes),
                "statement_format": 3}


@dataclass
class Packaged:
    batches: list[Batch]
    refused: list[dict[str, Any]]  # {vu_index, kind, request_id, reason}: never dropped silently
    by_family: dict[str, int] = field(default_factory=dict)

    @property
    def covered_vu_indices(self) -> list[int]:
        return sorted(v for b in self.batches for v in b.vu_indices)


def package_batches(located_units: Iterable[Mapping[str, Any]], record: Any, *, max_units: int | Mapping[str, int] | None = None,
                    name: str = "e7", authenticate: bool = True, transparent_check: bool = True, builders: Mapping[str, Builder] | None = None) -> Packaged:
    """Every located VU whose kind has a registered builder -> proof-sized statements: one family per statement, at most ``max_units``
    units each (default :data:`MAX_UNITS_BY_FAMILY` / :data:`DEFAULT_MAX_UNITS`), VU identities kept per batch.  A unit the builder
    refuses (an unresolvable word, a kind absent from the registry) is a ``refused`` row with the reason -- never dropped, never proved
    as something else.  With ``transparent_check`` every batch is also run through the guest's Python mirror (``verify_transparent``;
    LABEL: a local relation check, NOT the approved checker's proof) and its verdict recorded in ``built.record['witness_verdict']``."""

    ck = _ck_elem()
    regs = builders if builders is not None else {**ALL_BUILDERS, **ftz_kinds()}  # E4's FTZ kinds package here too (family ftz-f32@1)
    resolved: dict[str, list[ResolvedUnit]] = {}
    refused: list[dict[str, Any]] = []
    for u in located_units:
        kind = str(u.get("kind"))
        vu = int(u.get("vu_index", -1))
        if kind not in regs:
            refused.append({"vu_index": vu, "kind": kind, "request_id": u.get("request_id"), "reason": f"{kind}: no builder registered"})
            continue
        try:
            r = resolve_unit(u, record)
        except ck.Refusal as e:
            refused.append({"vu_index": vu, "kind": kind, "request_id": u.get("request_id"), "reason": str(e)[:300]})
            continue
        resolved.setdefault(r.spec.family, []).append(r)
    batches: list[Batch] = []
    for fam in sorted(resolved):
        rs = sorted(resolved[fam], key=lambda r: (r.request_id, r.vu_index))
        n = (max_units.get(fam, DEFAULT_MAX_UNITS) if isinstance(max_units, Mapping) else max_units) if max_units is not None else MAX_UNITS_BY_FAMILY.get(fam, DEFAULT_MAX_UNITS)
        n = max(1, int(n))
        for c0 in range(0, len(rs), n):
            part = rs[c0:c0 + n]
            idx = len(batches)
            bname = f"{name}-chunk{idx:02d}-{fam}"
            built = build_batch(part, record, name=bname, authenticate=authenticate)
            if transparent_check:
                built.record["witness_verdict"] = ck.verify_transparent(built)
            batches.append(Batch(idx, fam, tuple(sorted({r.kind for r in part})), tuple(r.vu_index for r in part), built, bname))
    return Packaged(batches, refused, {fam: sum(1 for b in batches if b.family == fam) for fam in resolved})


def batches_document(p: Packaged, *, sampled_vu_indices: Iterable[int] | None = None, input_vu_indices: Iterable[int] = ()) -> dict[str, Any]:
    """Interface 6, prover side: ``proofs.chunks`` (one row per proof object, the VU identities it covers) and ``per_vu`` (covered /
    uncovered / input) as SEPARATE lists, plus the three numbers reported separately: sampled VUs, covered VUs, proof objects."""

    covered = set(p.covered_vu_indices)
    inputs = set(int(v) for v in input_vu_indices)
    sampled = set(int(v) for v in sampled_vu_indices) if sampled_vu_indices is not None else covered | {int(r["vu_index"]) for r in p.refused} | inputs
    per_vu = {str(v): ("input" if v in inputs else "covered" if v in covered else "uncovered") for v in sorted(sampled)}
    return {
        "schema": "veritor/e7/batches/v1",
        "proofs": {"chunks": [b.summary() for b in p.batches], "count": len(p.batches)},
        "per_vu": per_vu,
        "refused": list(p.refused),
        "counts": {"sampled_vus": len(sampled), "covered_vus": len(covered & sampled), "input_vus": len(inputs & sampled), "uncovered_vus": len([v for v in sampled if per_vu[str(v)] == "uncovered"]),
                   "proof_objects": len(p.batches)},
        "note": "sampled VUs, covered VUs and proof objects are three separate numbers; coverage is by VU identity (per_vu), never by proof count",
    }
