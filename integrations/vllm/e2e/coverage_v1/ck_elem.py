"""CK-ELEM: record-wired obligation builders for the elementwise kinds the approved guest evaluates.

The five stations of the directive (``docs/handoffs-in/cursor-directive-2026-09-10-1310.md`` section 4) for one kind
are builder, checker, dispatch, expected statement, integration test.  This module is the *builder* and the *expected
statement* of the elementwise kinds; ``builders.py`` registers them (``BUILDERS[kind]``) for the dispatch patch.

What a builder reads (vocabulary of ``docs/PROTOCOL.md``): a *located VU* -- the verifier's own derivation of a selected
VU (``e2e/verifier_v3/locator.py``: ``kind``, ``gate_interval`` in the pinned program) -- and the record's *committed
leaves* (the RU trees of ``commitments.json`` opened through ``verity_vllm.commit.TreeCache``; the registered weights
tree for the rope cos/sin table).  It resolves every operand gate and the output gate of the VU's primitive to a
committed word: a word of an operator-boundary row the replay commits (``verity_vllm.correspondence.core.CircuitMap``: capture
identity -> the producing call's return gates) or a word of a weights leaf (``ProgramLocator.locate_input``).  A gate
that is neither -- an interior value today's record does not commit (an FP32 running value of a norm, an attention
score, ...) -- is a **refusal** naming the gate and the enclosing call, never a statement over invented words.  The
refusal is the per-kind R18 evidence: the kind's checker exists, its inputs are not committed, so a selected VU of the
kind is an unsupported check until the commitment set changes (an integrator decision, not a builder's).

The typed statement (``typed-obligation/v0``, format 3 by ``veritor.interface.lowering.to_statement_v3``): one
obligation per VU whose subcircuit is the kind's one gate over its incoming words (widths of the committed words, 16
for BF16) with the output word outgoing; roots/leaves/references exactly as the serving exporter spells them
(``verity_vllm.typed_obligation_export``: ``word_reference``, ``_owner_of``, ``VLLM_FRAMING``), so the lowering's
``packed/v1`` positions and the guest's leaf hashing are unchanged.  The family is the approved ELF's gate set the
kind's gate lives in: ``veritor.elem-bf16@1`` (``silu_mul_bf16``, ``bf16_add_f2fp``) or ``veritor.reduce-mufu@1``
(``rope_out``, ``rope_out_add``, ``bf16_to_f32``, ``f32_*``); both resolve in guest ``9fd00e60``
(``zk/sp1/common/src/gateset.rs`` ``ELEM_BF16_ID`` / ``REDUCE_MUFU_ID`` at width 32) and neither is in this
checkout's ``veritor.interface.typed_obligation.SUPPORTED_FAMILIES`` -- see :func:`ensure_families` (INTERIM
registration in this process; the permanent form is ``data/contract/ck-elem/patches/expected.diff`` for A2).

The expected statement (station 4): :func:`expectation` derives, from the located VU and the committed id lists only
(never from the prover's object), the :class:`~veritor.interface.expected.ExpectedObligation` whose incoming/outgoing
positions ``check_expected_positions`` compares the proved object against; its subcircuit digest is the same
canonical body :func:`kind_subcircuit` builds, so a statement for a neighbouring word (another ``i``, another ``t``)
or another body is refused.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for _p in (str(_REPO), str(_REPO / "src"), str(_REPO / "vllm-poc")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from veritor.interface import expected as _expected  # noqa: E402
from veritor.interface import typed_obligation as _to  # noqa: E402
from veritor.interface.expected import ExpectedObligation, ExpectedValue  # noqa: E402
from veritor.interface.programs import make_subcircuit  # noqa: E402
from veritor.interface.typed_obligation import (  # noqa: E402
    IN,
    Array,
    Bits,
    Binding,
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
    "ELEM_FAMILY",
    "REDUCE_FAMILY",
    "KINDS",
    "KindSpec",
    "Refusal",
    "CommittedWord",
    "ResolvedUnit",
    "Built",
    "RecordHandle",
    "ensure_families",
    "family_gate_set",
    "kind_subcircuit",
    "ElemLocator",
    "resolve",
    "build",
    "build_many",
    "expectation",
    "verify_transparent",
]

ELEM_FAMILY = "veritor.elem-bf16@1"
REDUCE_FAMILY = "veritor.reduce-mufu@1"
STATEMENT_WIDTH = 32  # both families resolve at width 32 only (gateset.rs REDUCE_MUFU_ID / ELEM_BF16_ID: 32..=32)
WORD_BITS = 16  # a committed BF16 element is one 16-bit word (typed_obligation_export.WORD_BITS)
VLLM_FRAMING = "verity-vllm/leaf/v1"
WEIGHTS_TREE = "weights"
OWNER_WEIGHTS, OWNER_BOUNDARY = -2, -1


# --- kinds -----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KindSpec:
    """One elementwise kind: the registered primitive it is a batch/primitive VU of, the gate of the approved guest's
    gate set that evaluates it, the family that gate set is, and the operand/output widths the gate declares."""

    kind: str  # ``fn|form`` as the selection spells it
    prim: str  # ``verity_vllm.registry.prims`` name (``Gate.prim.name``)
    gate: str  # gate name in ``family``'s gate set
    family: str
    arity: int
    in_width: int  # width of the operand words the gate consumes
    out_width: int  # width of the value the gate produces
    note: str = ""


def _k(kind: str, prim: str, gate: str, family: str, arity: int, in_w: int, out_w: int, note: str = "") -> KindSpec:
    return KindSpec(kind, prim, gate, family, arity, in_w, out_w, note)


#: The kinds this lane owns that have a gate in the approved guest.  Kinds with 32-bit operands or outputs resolve
#: only if the record commits those words (it does not today: the refusal names the gate) -- they are listed so the
#: builder's refusal, not a missing table entry, is the evidence.
KINDS: dict[str, KindSpec] = {
    s.kind: s
    for s in (
        _k("SiluMulBf16_v1|batch", "SiluMulBf16", "silu_mul_bf16", ELEM_FAMILY, 2, 16, 16, "gate_up row -> act_mul row (both committed)"),
        _k("Bf16AddF2fp_v1|batch", "Bf16AddF2fp", "bf16_add_f2fp", ELEM_FAMILY, 2, 16, 16, "residual add: two committed rows -> committed row"),
        _k("RopeOut_v1|batch", "RopeOut", "rope_out", REDUCE_FAMILY, 4, 16, 16, "qkv gemm row words + cos_sin weights row -> rope row"),
        _k("RopeOutAdd_v1|batch", "RopeOutAdd", "rope_out_add", REDUCE_FAMILY, 4, 16, 16, "as RopeOut (second half)"),
        _k("Bf16ToF32_v1|batch", "Bf16ToF32", "bf16_to_f32", REDUCE_FAMILY, 1, 16, 32, "output FP32 word is a norm interior (not committed)"),
        _k("F2fpBf16_v1|batch", "F2fpBf16", "f2fp_bf16", REDUCE_FAMILY, 1, 32, 16, "operand FP32 word is a norm interior (not committed)"),
        _k("F32Mul_v1|batch", "F32Mul", "f32_mul", REDUCE_FAMILY, 2, 32, 32, "FP32 interior"),
        _k("F32Mul_v1|primitive", "F32Mul", "f32_mul", REDUCE_FAMILY, 2, 32, 32, "FP32 interior"),
        _k("F32Add_v1|primitive", "F32Add", "f32_add", REDUCE_FAMILY, 2, 32, 32, "FP32 interior"),
        _k("F32Div_v1|primitive", "F32Div", "f32_div", REDUCE_FAMILY, 2, 32, 32, "FP32 interior"),
        _k("F32Fma_v1|primitive", "F32Fma", "f32_fma", REDUCE_FAMILY, 3, 32, 32, "FP32 interior"),
    )
}

#: Owned kinds with NO gate in the approved guest (checker missing) or whose relation is not one gate over committed
#: words; ``resolve`` refuses them with this reason (the kind table's status is A5/6's; this is the builder's word).
NO_CHECKER: dict[str, str] = {
    "F32FmaFtz_v1|primitive": "no FTZ gate in any family of the approved guest (gateset.rs); attention/norm interior",
    "F32SubFtz_v1|primitive": "no FTZ gate in any family of the approved guest",
    "F32MulFtz_v1|batch": "no FTZ gate in any family of the approved guest",
    "GuardNegInfZero_v1|primitive": "no gate in any family of the approved guest",
    "DivFullRcp_v1|primitive": "table primitive: needs the committed reciprocal table tree (reduce_tables.py) the record does not commit",
    "DivFullScaleA_v1|primitive": "operands are FP32 interiors of a table-backed division",
    "ArgmaxStep_v1|scan": (
        "scan carry (best, best_index, i) is a TokenSelect interior the record does not commit (only the lm_head logits row "
        "and sampler.select are); the gates exist in veritor.reduce-mufu@1 (bf16_gt, select_bf16, select_i32, i32_add). "
        "W-A56-2: tie rule MEASURED lowest index (REC-P1-B0: 4 of 48 committed rows tie at the max, all 4 served the lowest "
        "index = the program's Bf16GtStrict scan); NaN rule UNAVAILABLE on the record (0 NaN rows) and the program "
        "(NaN never wins) disagrees with torch's GreaterOrNan (first NaN wins, MODELED from source): not built until measured"
    ),
}


def _is_const(kind: str) -> bool:
    return kind.startswith("Const16[") or kind.startswith("Const32[")


# --- families --------------------------------------------------------------------------------------------------------


def _elem_gate_set():
    bench = _REPO / "zk" / "bench"
    if str(bench) not in sys.path:
        sys.path.insert(0, str(bench))
    from elem_gates import ELEM_BF16_DIGEST, make_elem_bf16_gate_set  # type: ignore[import-not-found]

    gs = make_elem_bf16_gate_set()
    if str(gs.digest) != ELEM_BF16_DIGEST:
        raise RuntimeError(f"elem-bf16 gate set digest {gs.digest} is not the pinned {ELEM_BF16_DIGEST}")
    return gs


def _reduce_gate_set():
    from veritor.core.reduce_gates import make_reduce_mufu_gate_set

    return make_reduce_mufu_gate_set()


_LOCAL_FAMILIES = {ELEM_FAMILY: _elem_gate_set, REDUCE_FAMILY: _reduce_gate_set}
_installed = False


def ensure_families() -> None:
    """INTERIM: make ``veritor.interface`` resolve ``veritor.elem-bf16@1`` and ``veritor.reduce-mufu@1`` at width 32
    in THIS process when the checkout's ``typed_obligation.py`` does not yet (A2's file; the permanent change is
    ``data/contract/ck-elem/patches/expected.diff``).  ``validate`` checks ``family.digest`` against the gate set's,
    so the tie to the approved ELF's gate set is the pinned digest either way.  Idempotent; a no-op once the
    interface carries the families."""

    global _installed
    if _installed:
        return
    missing = [f for f in _LOCAL_FAMILIES if _to.family_gate_set(f, STATEMENT_WIDTH) is None]
    if missing:
        original = _to.family_gate_set

        def patched(family_id: str, width: int):
            if family_id in _LOCAL_FAMILIES and width == STATEMENT_WIDTH:
                return _LOCAL_FAMILIES[family_id]()
            return original(family_id, width)

        _to.family_gate_set = patched  # type: ignore[assignment]
        _to.SUPPORTED_FAMILIES = tuple(_to.SUPPORTED_FAMILIES) + tuple(f for f in missing if f not in _to.SUPPORTED_FAMILIES)  # type: ignore[assignment]
        # modules that bound the name at import time
        import veritor.interface as _iface
        import veritor.interface.lowering as _lowering
        import veritor.interface.vectors as _vectors

        for mod in (_iface, _lowering, _vectors, _expected):
            if hasattr(mod, "family_gate_set"):
                mod.family_gate_set = patched  # type: ignore[attr-defined]
            if hasattr(mod, "SUPPORTED_FAMILIES"):
                mod.SUPPORTED_FAMILIES = _to.SUPPORTED_FAMILIES  # type: ignore[attr-defined]
    _installed = True


def family_gate_set(family_id: str):
    ensure_families()
    gs = _to.family_gate_set(family_id, STATEMENT_WIDTH)
    if gs is None:
        raise RuntimeError(f"{family_id} does not resolve at width {STATEMENT_WIDTH}")
    return gs


def family_of(family_id: str) -> Family:
    return Family(family_id, str(family_gate_set(family_id).digest), STATEMENT_WIDTH)


# --- the canonical body per kind -----------------------------------------------------------------------------------


def kind_subcircuit(spec: KindSpec) -> Subcircuit:
    """The one-gate body of ``spec``: incoming values ``0 .. arity`` (the operands in the primitive's signature order,
    each ``Value<in_width>``), gate 0 = ``spec.gate`` over them, outgoing = gate 0.  The ``Definition`` annotation
    (``<prim>`` within the kind) does not enter the digest; the digest is what the expectation pins."""

    own = getattr(spec, "subcircuit", None)  # [E7 D28] a spec that carries its own body (elem_builders.CallSpec / PvSpec / TableSpec / AliasSpec)
    if own is not None:
        return own
    gates = [(spec.gate, [(IN, i) for i in range(spec.arity)])]
    return make_subcircuit([spec.in_width] * spec.arity, gates, [0], Definition(spec.prim, (), spec.kind))


# --- committed words -----------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CommittedWord:
    """One committed word: the tree, the leaf's canonical id, the leaf's shape, the word's row-major flat index in the
    leaf, and the path through the leaf's layout (the ``Reference.offset``)."""

    tree: str
    leaf: str
    shape: tuple[int, ...]
    flat: int
    width: int = WORD_BITS

    @property
    def offset(self) -> tuple[int, ...]:
        return tuple(int(i) for i in np.unravel_index(self.flat, self.shape))

    @property
    def ref(self) -> str:
        return self.leaf + "".join(f"[{k}]" for k in self.offset)

    def expected(self) -> ExpectedValue:
        return ExpectedValue(self.tree, self.leaf, self.offset)


class Refusal(ValueError):
    """The builder refuses: the VU is not one gate over committed words (the reason names the gate / call), or the kind
    has no gate in the approved guest.  Never a silently uncovered entry."""


@dataclass(frozen=True, slots=True)
class ResolvedUnit:
    """A located VU whose operand and output gates are all committed words."""

    vu_index: int
    kind: str
    spec: KindSpec
    gate: int  # the VU's gate in the pinned program
    root: str  # the enclosing root call name (``t=<t>/layers.<L>/<role>`` ...)
    request_id: str
    inputs: tuple[CommittedWord, ...]
    output: CommittedWord
    replay_unit: int  # RU ordinal (owner of the output's tree)

    @property
    def target(self) -> str:
        return f"{self.root}/{self.spec.prim}@{self.gate}"


# --- the record ----------------------------------------------------------------------------------------------------


@dataclass
class RecordHandle:
    """The record as the builder reads it: the pinned program, the verifier's ``ProgramLocator``, the committed trees
    (``commitments.json``) and the capture reader (RU stores, weights) for openings and id lists.

    ``reader`` may be ``None`` on the verifier side when only the expectation is derived (ids are then given by
    ``committed_ids``: tree id -> its committed canonical ids in tree order)."""

    program: Any
    C: dict[str, Any]
    locator: Any  # e2e.verifier_v3.locator.ProgramLocator
    commitments: dict[str, Any]
    reader: Any = None  # verity_vllm.capture.CaptureReader
    committed_ids: dict[str, list[str]] = field(default_factory=dict)
    profile_id: str = ""
    _cache: Any = None
    _elem: Any = None
    # [E7 SCALE 2026-09-12, day-list item "oracles hook" (E9 / E10 08:10Z; D45 follow-up)] the verifier's word oracles.  ``expected.coverage_inputs``
    # appends E3's ``PositionOracle.word`` here once the position oracle exists (`record_handle.oracles = (..., positions.word)`); with a non-empty
    # chain :attr:`elem` is ``elem_builders.OracleChain([LiteralOracle, ElemLocator(self), *oracles])`` and :attr:`ftz` E4's resolver over it + E2's
    # ``TableWords`` -- the SAME chain the prover's ``open_record`` wires -- so ``dispatch_expectation`` resolves interior words and the MUFU table
    # row port itself, and ``RecordPartition._from_positions`` (the fallback E3 wrote for "the builder refuses an interior word") is not consulted
    # for a kind whose builder derives.  Empty (the default): exactly the pre-hook handle (boundary + weights through ``ElemLocator`` only).
    _oracles: tuple = ()
    _ftz: Any = None

    @property
    def oracles(self) -> tuple:
        return self._oracles

    @oracles.setter
    def oracles(self, value) -> None:
        self._oracles = tuple(value or ())
        self._elem = None  # re-wired lazily by :attr:`elem`
        self._ftz = None

    @property
    def ftz(self) -> Any:
        """E4's ``ck_ftz.WordResolver`` (``word`` + ``table_row``) when the oracle chain is wired, else ``None`` (E4 falls back to ``ElemWords``)."""
        if self._ftz is None and self._oracles:
            from e2e.coverage_v1 import elem_builders as _eb

            self._ftz = _eb.verifier_ftz_resolver(self)
        return self._ftz

    @ftz.setter
    def ftz(self, value) -> None:
        self._ftz = value

    @classmethod
    def open(cls, sel_dir: Path, program: Any, C: dict[str, Any], locator: Any) -> RecordHandle:
        from verity_vllm.capture import CaptureReader
        from verity_vllm.commit import TreeCache, load_commitments

        reader = CaptureReader(sel_dir)
        comm = load_commitments(reader)
        h = cls(program, C, locator, comm, reader, profile_id=str(reader.profile_id))
        h._cache = TreeCache(reader)
        return h

    def ids_of(self, tree_id: str) -> list[str]:
        if tree_id in self.committed_ids:
            return self.committed_ids[tree_id]
        if self.reader is None:
            raise KeyError(f"{tree_id}: no committed id list and no reader")
        parts = tree_id.split("/")
        if parts[0] == "ru" and len(parts) == 3:
            ids = list(self.reader.ru(parts[1], int(parts[2])).ids())
        elif parts[0] == "boundary":
            ids = list(self.reader.boundary(parts[1]).leaf_ids(self.profile_id))
        else:
            raise KeyError(tree_id)
        self.committed_ids[tree_id] = ids
        return ids

    def ru_tree_ids(self, request_id: str) -> list[str]:
        return [t for t in self.commitments["trees"] if t.startswith(f"ru/{request_id}/")]

    def open_leaf(self, canonical_id: str):
        if self._cache is None:
            raise Refusal("no capture reader: openings unavailable (verifier side derives expectations only)")
        try:
            return self._cache.open(canonical_id)
        except Refusal:
            raise
        except Exception as ex:  # noqa: BLE001
            # [E7 oracles hook] a leaf of E2's ``interior/<req>/<step>`` tree: the capture reader does not address it (its canonical id is not a
            # boundary/weights identity) -- read it from the shipped tree directory, authenticated against the index row's framed leaf hash
            if self._oracles:
                from e2e.coverage_v1 import elem_builders as _eb

                o = _eb.interior_opening(self, canonical_id)
                if o is not None:
                    return o
                raise Refusal(f"INTERIOR_LEAF_UNREADABLE: {canonical_id}: not a leaf the capture reader addresses ({type(ex).__name__}: {str(ex)[:80]}) "
                              f"and a leaf of no shipped interior tree of the record (fail closed: the unit's value-dependent port cannot be named)") from ex
            raise

    @property
    def elem(self) -> ElemLocator:
        if self._elem is None:
            if self._oracles:
                from e2e.coverage_v1 import elem_builders as _eb

                self._elem = _eb.verifier_oracle_chain(self)
            else:
                self._elem = ElemLocator(self)
        return self._elem


# --- locating gates in committed leaves ------------------------------------------------------------------------------


class ElemLocator:
    """Gate index in the pinned program -> committed word, from the program alone plus the committed id lists.

    Operator-boundary rows: ``CircuitMap.locate(CaptureIdentity)`` gives the producing call's return gates of a
    committed value id (row-major tensor order) -- the inverse is a per-root-call index built lazily over the ids of
    the RU trees of the VU's request (a root call ``t=<t>/layers.<L>/<role>`` holds the rows of that t / layer whose
    op belongs to the role).  Weights: ``ProgramLocator.locate_input`` + the registered weights layout (rows -> block
    leaf, offset within the block)."""

    def __init__(self, record: RecordHandle) -> None:
        from verity_vllm.correspondence.core import CaptureIdentity, CircuitMap

        self.record = record
        self.cmap = CircuitMap(record.program)
        self._ident = CaptureIdentity
        self._by_root: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray, str, str, tuple[int, ...]]]] = {}
        self._indexed: set[str] = set()

    # -- rows ------------------------------------------------------------------------------------------------------
    def _index_request(self, request_id: str) -> None:
        """Index every committed value id of the request's RU trees by the root call its gates lie in."""
        if request_id in self._indexed:
            return
        self._indexed.add(request_id)
        loc = self.record.locator
        from verity_vllm.correspondence.core import NotInProgram

        for tree_id in self.record.ru_tree_ids(request_id):
            for cid in self.record.ids_of(tree_id):
                try:
                    ident = self._ident.parse(cid)
                except ValueError:
                    continue  # not a capture-v2 value id (boundary / other)
                try:
                    _call, gates = self.cmap.locate(ident)
                except (NotInProgram, KeyError, IndexError):
                    continue
                if not gates:
                    continue
                g = np.asarray(gates, dtype=np.int64)
                order = np.argsort(g, kind="stable")
                name, _root = loc.root_of(int(g[0]))
                if name is None:
                    continue
                self._by_root.setdefault((request_id, name), []).append((g[order], order.astype(np.int64), tree_id, cid, ()))

    def row_word(self, request_id: str, g: int) -> CommittedWord | None:
        """The committed row word at gate ``g`` (a return gate of an operator whose row the replay commits), or None."""
        self._index_request(request_id)
        name, _root = self.record.locator.root_of(g)
        if name is None:
            return None
        for sorted_gates, order, tree_id, cid, _ in self._by_root.get((request_id, name), ()):
            k = int(np.searchsorted(sorted_gates, g))
            if k < len(sorted_gates) and int(sorted_gates[k]) == g:
                flat = int(order[k])
                shape = self._shape_of(tree_id, cid, len(sorted_gates))
                return CommittedWord(tree_id, cid, shape, flat)
        return None

    def _shape_of(self, tree_id: str, cid: str, n_words: int) -> tuple[int, ...]:
        rec = self.record
        if rec.reader is not None:
            try:
                v = rec.reader.value(cid)
                return tuple(int(x) for x in v.shape)
            except Exception:  # noqa: BLE001
                pass
        return (n_words,)

    # -- weights ---------------------------------------------------------------------------------------------------
    def weight_word(self, g: int) -> CommittedWord | None:
        loc = self.record.locator
        if g >= loc.input_gates:
            return None
        w = loc.locate_input(g)
        if w.get("kind") != "weight":
            return None
        cid, offset = str(w["canonical_id"]), int(w["offset"])
        reader = self.record.reader
        if reader is None or not reader.has_weights():
            raise Refusal(f"gate {g}: weights word {cid}[{offset}] but the record has no weights store to resolve its leaf")
        entry = reader.weights.entry(cid)
        shape = tuple(int(d) for d in entry.shape)
        K = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        if entry.layout.kind == "rows":
            row, k = divmod(offset, K)
            start, end = entry.block_of(row)
            leaf = entry.block_id(start)
            block_shape = tuple(int(d) for d in entry.block_shape(start, end))
            flat = (row - start) * K + k
            return CommittedWord(WEIGHTS_TREE, leaf, block_shape, flat)
        return CommittedWord(WEIGHTS_TREE, cid, shape, offset)

    def word(self, request_id: str, g: int) -> CommittedWord | None:
        w = self.weight_word(g)
        if w is not None:
            return w
        return self.row_word(request_id, g)


# --- resolve a located VU -----------------------------------------------------------------------------------------


def _describe_gate(program: Any, g: int) -> str:
    try:
        gate = program.circuit.gate(g)
        return f"{gate.prim.id}@{g} in {'/'.join(gate.path)}"
    except Exception as e:  # noqa: BLE001
        return f"gate {g} ({type(e).__name__})"


def resolve(located: Mapping[str, Any], record: RecordHandle) -> ResolvedUnit:
    """The located VU as one gate over committed words, or a :class:`Refusal` with the reason."""

    kind = str(located["kind"])
    if kind in NO_CHECKER:
        raise Refusal(f"{kind}: {NO_CHECKER[kind]}")
    if _is_const(kind):
        raise Refusal(f"{kind}: a constant has no operand and its value is an interior word; not a relation over committed words")
    spec = KINDS.get(kind)
    if spec is None:
        raise Refusal(f"{kind}: not an elementwise kind this builder knows (BUILDERS[kind] absent)")
    lo, hi = (int(x) for x in located["gate_interval"])
    if hi - lo != 1:
        raise Refusal(f"{kind}: VU [{lo},{hi}) is not one gate")
    request_id = str(located["request_id"])
    circuit = record.program.circuit
    gate = circuit.gate(lo)
    pname = getattr(gate.prim, "name", gate.prim.id)
    if pname != spec.prim:
        raise Refusal(f"{kind}: gate {lo} is {gate.prim.id}, not {spec.prim}")
    if len(gate.operands) != spec.arity:
        raise Refusal(f"{kind}: gate {lo} has {len(gate.operands)} operands, the gate set's {spec.gate} takes {spec.arity}")
    name, _root = record.locator.root_of(lo)
    if name is None:
        raise Refusal(f"{kind}: gate {lo} lies in no root call")
    elem = record.elem
    inputs: list[CommittedWord] = []
    for og in gate.operands:
        w = elem.word(request_id, int(og))
        if w is None:
            raise Refusal(f"{kind}: operand {_describe_gate(record.program, int(og))} is an interior value the record does not commit")
        if w.width != spec.in_width:
            raise Refusal(f"{kind}: operand {w.ref} is a {w.width}-bit committed word, the gate takes {spec.in_width}-bit values")
        inputs.append(w)
    out = elem.row_word(request_id, lo)
    if out is None:
        raise Refusal(f"{kind}: output {_describe_gate(record.program, lo)} is an interior value the record does not commit")
    if out.width != spec.out_width:
        raise Refusal(f"{kind}: output {out.ref} is a {out.width}-bit committed word, the gate produces {spec.out_width} bits")
    ru = int(out.tree.split("/")[2]) if out.tree.startswith("ru/") else 0
    u = ResolvedUnit(int(located.get("vu_index", -1)), kind, spec, lo, name, request_id, tuple(inputs), out, ru)
    # [E7 D28, aliased-body/v1] operands that repeat one committed word (F32Mul(x, x)) fold into one incoming value read by every operand
    # position that names it: the obligation's boundary lists the word once (typed-obligation/v0 E_DUPLICATE_BOUNDARY_VALUE keeps its meaning).
    from .elem_builders import alias_unit

    return alias_unit(u)


# --- the expected statement (verifier side) -----------------------------------------------------------------------


def expectation(located: Mapping[str, Any], record: RecordHandle, *, label: str = "") -> ExpectedObligation:
    """The verifier's expectation for a located elementwise VU: incoming = the operand words' positions (tree, leaf,
    offset) in the primitive's signature order, outgoing = the output word's position, subcircuit = the kind's
    canonical body digest.  Derived from the pinned program and the committed id lists (and the registered weights
    layout for a weights word); refuses (``ValueError``) exactly when :func:`resolve` refuses."""

    u = resolve(located, record)
    sub = kind_subcircuit(u.spec)
    return ExpectedObligation(tuple(w.expected() for w in u.inputs), (u.output.expected(),), sub.digest, (None,), {},
                              label or f"vu={u.vu_index} {u.kind} {u.target}")


# --- the typed statement (prover side) ---------------------------------------------------------------------------


@dataclass
class Built:
    """One typed statement over one or more resolved units of one family, with everything the proving pipeline
    (``rec_p1_prove.py`` / ``veritor-zk-host prove``) and the verifier (``check_binding``, ``check_expected_positions``)
    read: the object, its format-3 statement bytes, the packed witness bytes, and the composed-record JSON."""

    obj: TypedObligationSet
    units: tuple[ResolvedUnit, ...]
    openings: dict[str, Any]
    statement: Any  # veritor.protocol.proofs StatementV3
    statement_bytes: bytes
    witness_bytes: bytes
    record: dict[str, Any]
    family: str

    @property
    def expectations(self) -> list[ExpectedObligation]:
        out = []
        for u in self.units:
            sub = kind_subcircuit(u.spec)
            out.append(ExpectedObligation(tuple(w.expected() for w in u.inputs), (u.output.expected(),), sub.digest, (None,), {}, f"vu={u.vu_index} {u.kind}"))
        return out


def _type_of(opening: Any) -> Type:
    from verity_vllm.typed_obligation_export import words_per_element

    ty: Type = Bits(WORD_BITS)
    n = words_per_element(opening.dtype)
    if n > 1:
        ty = Array(ty, n)
    for extent in reversed(opening.shape):
        ty = Array(ty, int(extent))
    return ty


def _owner_of(tree: str) -> int:
    if tree == WEIGHTS_TREE:
        return OWNER_WEIGHTS
    parts = tree.split("/")
    if len(parts) == 3 and parts[0] == "ru":
        return int(parts[2])
    if parts[0] == "boundary":
        return OWNER_BOUNDARY
    raise Refusal(f"tree {tree!r}: no owner mapping")


def _binding(session_text: str, profile_id: str, unit: int, replay_unit: int) -> Binding:
    from verity_vllm.typed_obligation_export import _h

    return Binding(_h("session", session_text), _h("compiled", profile_id), unit, replay_unit)


def build_many(units: Sequence[ResolvedUnit], record: RecordHandle, *, name: str = "ck-elem", authenticate: bool = True) -> Built:
    """One typed statement over ``units`` (all of one family).  Leaves and references are deduplicated by identity;
    obligations are keyed ``(session = request, compiled = profile, unit = ordinal in target order)`` as the serving
    exporter keys its coordinates.  The openings are authenticated against ``commitments.json``
    (``typed_obligation_export.authenticate_opening``) before any word is read."""

    from verity_vllm.typed_obligation_export import authenticate_opening, leaf_words
    from veritor.interface.lowering import statement_bytes_v3, to_statement_v3
    from veritor.protocol.proofs.packed import encode_packed_witness

    if not units:
        raise Refusal("no units to build")
    families = {u.spec.family for u in units}
    if len(families) != 1:
        raise Refusal(f"units of several families in one statement: {sorted(families)}")
    family_id = next(iter(families))
    family = family_of(family_id)
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
                roots[o.tree] = Root(o.tree, _owner_of(o.tree), rec["ids_digest"], o.root.hex(), o.leaf_count, VLLM_FRAMING)
        return o

    def ref_of(w: CommittedWord) -> tuple[str, int]:
        o = take(w.leaf)
        if o.tree != w.tree:
            raise Refusal(f"{w.leaf}: committed in {o.tree}, located in {w.tree}")
        shape = tuple(int(x) for x in o.shape)
        if shape != w.shape:
            raise Refusal(f"{w.leaf}: opening shape {shape} != located shape {w.shape}")
        words = leaf_words(o)
        if not 0 <= w.flat < len(words):
            raise Refusal(f"{w.ref}: word {w.flat} outside the leaf ({len(words)} words)")
        if w.ref not in references:
            references[w.ref] = Reference(w.ref, w.leaf, w.offset)
        return w.ref, int(words[w.flat])

    pending: list[tuple[str, str, Subcircuit, tuple[str, ...], tuple[str, ...], tuple[str, ...], list[int], int]] = []
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
    built: list[tuple[Obligation, list[int], str]] = []
    for session, target, sub, inc, out, trees, row, ru in sorted(pending, key=lambda p: (p[0], p[1])):
        unit = ordinals.get(session, 0)
        ordinals[session] = unit + 1
        binding = _binding(session, record.profile_id, unit, ru)
        built.append((Obligation(binding, sub.digest, trees, Boundary(inc, out), DeclaredInterface(inc, out), Target("call", target)), row, target))
    built.sort(key=lambda b: (b[0].binding.session, b[0].binding.compiled, b[0].binding.unit))
    leaves = [Leaf(o.canonical_id, o.tree, o.index, o.index, _type_of(o)) for o in openings.values()]
    obj = TypedObligationSet(
        family, (),
        tuple(roots[t] for t in sorted(roots)),
        tuple(sorted(leaves, key=lambda v: v.id)),
        tuple(sorted(references.values(), key=lambda r: r.ref)),
        tuple(sorted(subcircuits.values(), key=lambda s: s.digest)),
        tuple(o for o, _, _ in built),
    )
    ensure_families()
    validate(obj)
    statement_v3 = to_statement_v3(obj)
    statement = statement_v3.expand()  # the format-2 view the packed table / transparent backend read
    st_bytes = statement_bytes_v3(obj)
    witness = _packed_witness(openings.values(), statement)
    proof = encode_packed_witness(witness)
    record_json = _record_json(obj, [row for _, row, _ in built], proof, openings, how, units, name)
    return Built(obj, tuple(units), openings, statement, st_bytes, proof, record_json, family_id)


def build(located: Mapping[str, Any], record: RecordHandle, *, name: str = "ck-elem") -> Built:
    """``build(located_unit, record) -> statement`` for ONE located VU (the registry's signature)."""

    return build_many([resolve(located, record)], record, name=name)


def _packed_witness(openings: Iterable[Any], statement: Any):
    from verity_vllm import hashing
    from veritor.protocol.proofs.packed import PackedLeaf, PackedWitness, packed_table

    by_root_rank = {(o.root, o.index): o for o in openings}
    trees = []
    for tree in packed_table(statement).trees:
        leaves = []
        for rank in tree.ranks:
            o = by_root_rank.get((tree.commitment.root, rank))
            if o is None:
                raise Refusal(f"the statement opens rank {rank} of root {tree.commitment.root.hex()[:16]}..., which no opening covers")
            leaves.append(PackedLeaf(o.canonical_id, o.dtype, hashing.shape_text(o.shape), o.value, tuple(o.path)))
        trees.append(tuple(leaves))
    return PackedWitness(tuple(trees))


def _record_json(obj: TypedObligationSet, values: list[list[int]], proof: bytes, openings: dict[str, Any], how: dict[str, str],
                 units: Sequence[ResolvedUnit], name: str) -> dict[str, Any]:
    """The ``composed.json``-shaped record ``rec_p1_prove.py`` reads (``digest``, ``statement_digest``, ``object``,
    ``statement_format``) -- built without ``vectors.Vector`` (which lowers to format 2 as well and re-verifies)."""

    from veritor.interface.lowering import statement_bytes_v3, to_statement
    from veritor.interface.typed_obligation import canonical_bytes, object_digest
    from veritor.protocol.proofs.wire import encode_statement, statement_digest

    st2 = to_statement(obj)
    raw2 = encode_statement(st2)
    raw3 = statement_bytes_v3(obj)
    kinds = sorted({u.kind for u in units})
    return {
        "name": name,
        "description": f"CK-ELEM elementwise statement: {', '.join(kinds)} ({len(units)} units) over the record's committed rows",
        "tags": ["ck-elem", "elementwise", "packed-leaf", "packed-word", "witness", "composed", *kinds],
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
            "builder": "e2e.coverage_v1.ck_elem",
            "units": [{"vu_index": u.vu_index, "kind": u.kind, "gate": u.gate, "target": u.target,
                       "inputs": [w.ref for w in u.inputs], "output": u.output.ref} for u in units],
            "authentication": how,
            "openings": {cid: {"tree": o.tree, "index": o.index, "bytes": len(o.value)} for cid, o in openings.items()},
        },
    }


# --- local verdict -------------------------------------------------------------------------------------------------


def verify_transparent(built: Built) -> str:
    """The transparent backend's verdict on the statement + witness (LABEL: the guest's Python mirror at the pinned
    gate-set digest -- a local check of the relation and the openings, NOT the approved checker's proof; the
    checker station is ``veritor-zk-host prove`` / ``verify-many`` on a CPU pod)."""

    from veritor.protocol.messages import Reject
    from veritor.protocol.proofs import TransparentBackend

    gs = family_gate_set(built.family)
    try:
        TransparentBackend(gs).verify(built.statement, built.witness_bytes)
        return "accept"
    except Reject as e:
        return f"reject:{e.code.name}"
