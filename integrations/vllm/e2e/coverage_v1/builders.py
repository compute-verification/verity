"""The obligation-builder registry: ``BUILDERS[kind].build(located_unit, record) -> Built`` (A5/6, integrator 20:40Z).

A *builder* turns one selected VU -- located by the verifier's own derivation (``e2e.verifier_v3.locator`` for GemmCoordinate /
Input; :mod:`e2e.coverage_v1.attn_locator` for the attention body) -- and the record's committed leaves into the statement a
checker proves, the openings that authenticate it, and the A2-shaped expectation (``veritor.interface.expected.ExpectedObligation``)
the verifier derives independently.  Dispatch is ``BUILDERS[unit.kind]``; a kind without an entry, or an entry that cannot build the
unit from what the record commits, raises :class:`BuildRefusal` with the exact missing word -- never a silent skip, never a proof
of a different unit.

What today's records commit (checked on a real RU tree, ``MINREC-B0-20260909T1930Z/profile-hash/capture-0``; `out/record/report.md:45`):
operator boundaries only -- ``rotary_emb.rope/port=0`` (q rows), ``rope/port=1`` (K rows), ``qkv_proj.gemm/port=0`` (V column slices),
``self_attn.attn.attn/port=0`` (head outputs), the norms, ``act_mul``, ``embedding``, ``sampler.select``.  No attention-interior word
(score, m, l, p, P^bf16, O accumulator) exists in any committed tree.  Consequently every per-primitive attention kind
(``AmpereBF16TcDot16_v1|batch``, ``DotBf16_v1{K=64}|call``, ``F2fpBf16_v1|primitive``, the five FTZ scalars) is registered here as an
:class:`InteriorRefuser`: its ``build`` names the interior leaf it would open under the PROPOSED ``attn-interior/v0`` layout
(:mod:`e2e.coverage_v1.interior_layout`) and refuses because the record does not commit it.  The whole-head relation
(``AttentionHead_v1{T,D}|call``, :mod:`e2e.coverage_v1.attn_head_builder`) reads only committed rows and is the one attention path
buildable from a record today; it is a PARTITION CHANGE (S55) and is registered behind :data:`HEAD_VU_FLAG`.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = ["BUILDERS", "Built", "BuildRefusal", "Builder", "HEAD_VU_FLAG", "LocatedUnit", "Record", "Row", "base_kind", "head_vu_enabled", "dispatch",
           "derives_expectation", "expectation_builders", "dispatch_expectation", "dispatch_kind_program", "families_of", "ensure_families",
           "open_record", "package_batches", "batches_document", "registry_table", "can_express", "OK", "VERIFIER_COVERED", "verifier_covered_rule"]

#: Environment flag (``COVERAGE_V1_HEAD_VU=1``) that registers the whole-head builder.  Off by default: adopting the head as the
#: attention VU changes the partition (new kind digests, counts, certificates) and is PROPOSED to the user, not adopted here.
HEAD_VU_FLAG = "COVERAGE_V1_HEAD_VU"


def head_vu_enabled() -> bool:
    return os.environ.get(HEAD_VU_FLAG, "") not in ("", "0", "false", "no")


class BuildRefusal(ValueError):
    """The unit cannot be built from what the record commits.  ``missing`` names the words (tree, canonical id, index) that
    do not exist in any committed tree; ``reason`` is the verifier-readable sentence."""

    def __init__(self, reason: str, *, missing: tuple[tuple[str, str, int | None], ...] = ()) -> None:
        super().__init__(reason)
        self.reason = reason
        self.missing = missing


@dataclass(frozen=True)
class LocatedUnit:
    """A selected VU located by the verifier: which request / step / layer / head / block / column / primitive.  ``kind`` is the
    partition's ``fn|form`` id; ``ru_tree`` the committed tree of its RU step (``ru/<request>/<step>``) when the locator knows it.
    Fields the kind does not use are ``None``."""

    vu_id: str
    kind: str
    request_id: str
    t: int
    layer: int | None = None
    head: int | None = None
    block: int | None = None
    column: int | None = None
    index: int | None = None
    ru_index: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Row:
    """One committed operator row the builder reads: the tree it lives in, its canonical id and rank there, dtype, shape and words
    (flat ``uint16`` for bf16, ``uint32`` for f32)."""

    tree: str
    canonical_id: str
    rank: int
    dtype: str
    shape: tuple[int, ...]
    words: np.ndarray

    @property
    def count(self) -> int:
        return int(np.prod(self.shape))

    @property
    def width(self) -> int:
        return 32 if self.dtype == "float32" else 16


@runtime_checkable
class Record(Protocol):
    """What a builder may read of a record: committed rows (the prover holds the capture), committed tree metadata, openings.
    Everything a builder emits must be reconstructible by the verifier from ``commitments.json`` + the id lists + the Description --
    a builder must not invent a tree the verifier cannot name."""

    config: dict[str, Any]  # NH, KVH, D, BN (the model config the program was built with)
    session: bytes
    compiled: bytes

    def row(self, request_id: str, t: int, op: str, port: int) -> Row | None:
        """The committed row of operator ``op`` (``layers.<L>.self_attn.rotary_emb.rope``) at ``(request, t, port)``, or None."""

    def tree(self, tree_id: str) -> Any:
        """``elem_builder.CommittedTree``-shaped metadata (root, leaf_count, ids_digest, owner) of a committed tree, or None."""

    def opened(self, tree_id: str, rank: int) -> Any:
        """The ``PackedLeaf`` (with path) of rank ``rank`` in ``tree_id`` -- the prover's opening of a committed leaf."""


@dataclass(frozen=True)
class Built:
    """The builder's output for one unit: the statement (raw format 3 or a typed object, see ``kind``), the format-3 witness,
    the A2 expectation keyed by vu_id, the coverage views (``coverage.ObligationView``) the verifier must recover from the proof,
    programs (``KindProgram.kind`` hex -> (name, subcircuit digest)) and labelled notes."""

    kind: str
    statement: Any
    witness: Any
    expectations: dict[str, Any]
    views: tuple[Any, ...]
    programs: dict[str, tuple[str, str]]
    notes: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Builder(Protocol):
    kind: str

    def build(self, unit: LocatedUnit, record: Record) -> Built: ...


# --- registry ------------------------------------------------------------------------------------------------------------


def _registry() -> dict[str, Builder]:
    from .elem_builders import (
        ALL_BUILDERS as ELEM_BUILDERS,  # the one-gate kinds (CK-ELEM's 11 + E7's extra forms; expectation-capable)
    )
    from .interior_layout import INTERIOR_KINDS, InteriorRefuser

    reg: dict[str, Builder] = {k: InteriorRefuser(k) for k in INTERIOR_KINDS}

    def fold(contribution: Mapping[str, Builder], source: str) -> None:
        for k, b in contribution.items():  # ONE registry: the dispatch (verifier_v3 + prover) reads this dict only; every module is a contribution
            have = reg.get(k)
            if have is not None and not isinstance(have, InteriorRefuser):
                raise RuntimeError(f"builder registry: kind {k!r} registered twice ({type(have).__module__}.{type(have).__name__} and {source})")
            # [E7, W-A9-12] a refuser is a placeholder ("reads uncommitted interior words"); a real builder for the same kind replaces it -- the
            # builder refuses by name at build time when the words are not committed, and never labels the kind unsupported.
            reg[k] = b

    fold(ELEM_BUILDERS, "elem_builders")
    for k, b in _ftz_contribution().items():  # E4's CK-FTZ kinds (family veritor.ftz-f32@1), through the located-unit adapter below
        fold({k: b}, "ftz_builders")
    fold(_select_contribution(), "select_builders")  # E5's TokenSelect / ArgmaxStep kinds (family veritor.select-argmax@1)
    if head_vu_enabled():
        from .attn_head_builder import HEAD_KIND, AttentionHeadBuilder

        reg[HEAD_KIND] = AttentionHeadBuilder()
    return reg


class _LocatedAdapter:
    """A contributed builder whose ``build`` / ``expectation`` spell the located VU as (``gate``, ``root``, ``replay_unit``) -- E4's
    ``ftz_builders`` -- called with the verifier's spelling (``gate_interval`` in the pinned program, root and RU derived from the record).
    One-gate kinds only: a VU of several gates is a refusal naming the interval, never a mis-built statement."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.kind = str(inner.kind)
        self.family = getattr(inner, "family", None)
        self.owner = getattr(inner, "owner", None)
        self.derives_expectation = bool(getattr(inner, "derives_expectation", True))

    def _located(self, unit: Mapping[str, Any], record: Any) -> dict[str, Any]:
        from .elem_builders import located_mapping

        u = dict(located_mapping(unit))
        if "gate" not in u:
            if "gate_interval" not in u:
                raise BuildRefusal(f"{self.kind}: located unit names neither gate nor gate_interval")
            lo, hi = (int(x) for x in u["gate_interval"])
            if hi - lo != 1:
                raise BuildRefusal(f"{self.kind}: VU [{lo},{hi}) is not one gate")
            u["gate"] = lo
        g = int(u["gate"])
        if not u.get("root"):
            locator = getattr(record, "locator", None)
            name = None
            if locator is not None and hasattr(locator, "root_of"):
                name, _ = locator.root_of(g)
            u["root"] = name or ""
        u.setdefault("replay_unit", -1)
        return u

    def _aliased(self, unit: Mapping[str, Any], record: Any) -> Any | None:
        """[``aliased-body/v1`` for E4's one-gate kinds] the unit resolved through E4's resolver and folded by ``alias_unit`` when it reads one
        committed word (a literal word under ``prescribed-literal/v1``, e.g. ``F32AddFtz(0, 0)``) at several positions; ``None`` when no repeat."""

        from . import elem_builders as _elem

        if self.owner != "ck-ftz" or self.kind not in _elem.ftz_kinds():
            return None
        u = _elem.resolve_unit(dict(unit), record)
        return u if isinstance(u.spec, _elem.AliasSpec) else None

    def build(self, unit: Mapping[str, Any], record: Any) -> Any:
        from . import elem_builders as _elem

        a = self._aliased(unit, record)
        if a is not None:
            return _elem.build_batch([a], record, name=f"{self.kind}-aliased")
        return self.inner.build(self._located(unit, record), record)

    def expectation(self, unit: Mapping[str, Any], record: Any) -> Any:
        a = self._aliased(unit, record)
        if a is not None:
            from . import elem_builders as _elem
            from veritor.interface.expected import ExpectedObligation

            return ExpectedObligation(tuple(w.expected() for w in a.inputs), (a.output.expected(),), _elem.unit_subcircuit(a).digest, (None,), {},
                                      f"vu={a.vu_index} {a.kind}")
        return self.inner.expectation(self._located(unit, record), record)

    def subcircuit_digest(self) -> str:
        return self.inner.subcircuit_digest()


def _ftz_contribution() -> dict[str, Builder]:
    """E4's ``ftz_builders.BUILDERS`` (branch eod/e4; folded in here, never registered by E4 itself), adapted to the verifier's located-unit
    spelling.  A checkout without the module contributes nothing -- the FTZ kinds then keep their ``InteriorRefuser`` placeholders."""

    try:
        from . import ftz_builders
    except ImportError:
        return {}
    # NOT here: ``ck_ftz`` (src/veritor) -- the PROVER imports this registry under PYTHONPATH=. / Python 3.11 (W-A9-11); the family's
    # interface resolution is the verifier's business: :func:`ensure_families` (called by the verifier-side dispatch entry points below)
    return {k: _LocatedAdapter(b) for k, b in ftz_builders.BUILDERS.items()}


def _select_contribution() -> dict[str, Builder]:
    """E5's ``select_builders.BUILDERS`` (branch eod/e5): ``TokenSelect_v1|whole-scan`` and ``ArgmaxStep_v1|scan`` (resolved to its whole-scan
    unit), family ``veritor.select-argmax@1`` -- a purpose-built select guest whose statement is commitment-bound (two boundary leaves per VU)
    but not a typed obligation: its ``expectation`` is a ``ck_select.SelectStatement`` and its ``subcircuit_digest()`` is ``None``.  The
    verifier's coverage check must compare ``sha256(statement)`` for these kinds (E5's ``check_expected``), not per-position openings.
    Folded in here, never registered by E5; absent module -> nothing contributed (the kinds then have no builder and say so)."""

    try:
        from . import select_builders
    except ImportError:
        return {}
    return {k: _StatementAdapter(b) for k, b in select_builders.BUILDERS.items()}


class StatementExpectation:
    """[E7] A statement-bound builder's expectation in BOTH spellings: the family's statement object (``statement``; ``encode()`` /
    ``digest()`` delegate to it -- ``coverage.expected_statements`` keys the VU by ``sha256(statement)``) AND A2's shape so the verifier's
    position comparison can read it (``incoming`` / ``outgoing`` as ``ExpectedValue(tree, leaf, (rank,))`` at LEAF granularity -- the guest
    authenticates whole leaves, not words; ``pins`` all None; ``subcircuit`` None: the relation is the approved ELF of ``family``, checked by
    the guest gate (V4), not by a body digest)."""

    def __init__(self, statement: Any, family: str, kind: str) -> None:
        from veritor.interface.expected import ExpectedValue

        self.statement = statement
        self.family = str(family)
        self.label = str(kind)
        self.subcircuit = None
        tree = str(statement.tree)
        self.incoming = (ExpectedValue(tree, str(statement.logits_id), (int(statement.logits_rank),)),)
        self.outgoing = (ExpectedValue(tree, str(statement.select_id), (int(statement.select_rank),)),)
        self.pins: tuple[str | None, ...] = (None,)

    def encode(self) -> bytes:
        return bytes(self.statement.encode())

    def digest(self) -> bytes:
        return bytes(self.statement.digest())

    def __getattr__(self, name: str) -> Any:  # V, literals, root, leaf_count, *_id, *_rank, to_json, check_bounds ... read through
        return getattr(self.statement, name)


class _StatementAdapter:
    """E5's ``SelectBuilder`` with its ``expectation`` wrapped in :class:`StatementExpectation`; ``check_expected`` accepts either spelling."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.kind = str(inner.kind)
        self.family = getattr(inner, "family", None)
        self.owner = getattr(inner, "owner", None)
        self.derives_expectation = bool(getattr(inner, "derives_expectation", True))

    def build(self, unit: Any, record: Any) -> Any:
        from .elem_builders import located_mapping

        return self.inner.build(located_mapping(unit), record)

    def expectation(self, unit: Any, record: Any) -> StatementExpectation:
        from .elem_builders import located_mapping

        return StatementExpectation(self.inner.expectation(located_mapping(unit), record), str(self.family), self.kind)

    def check_expected(self, statement_bytes: bytes, expected: Any) -> list[str]:
        return list(self.inner.check_expected(statement_bytes, getattr(expected, "statement", expected)))

    def subcircuit_digest(self) -> None:
        return self.inner.subcircuit_digest()


def base_kind(kind: str) -> str:
    """``AttentionHead_v1{T=9,D=64}|call`` -> ``AttentionHead_v1|call``: the kind id without its parameter block.  Kinds whose
    builder depends on the parameters (``DotBf16_v1{K=64}|call``) are registered under the full id and found first."""

    return re.sub(r"\{[^}]*\}", "", kind)


class _Registry(dict):
    """``BUILDERS``: kind -> Builder.  A full kind id is looked up as given, then by :func:`base_kind`; ``refresh`` re-reads the head
    flag so tests can toggle it."""

    def __missing__(self, kind: str) -> Builder:
        base = base_kind(kind)
        if base != kind and dict.__contains__(self, base):
            return dict.__getitem__(self, base)
        raise BuildRefusal(f"kind {kind!r} has no builder in coverage_v1.BUILDERS (registered: {sorted(self)})")

    def __contains__(self, kind: object) -> bool:
        return dict.__contains__(self, kind) or (isinstance(kind, str) and dict.__contains__(self, base_kind(kind)))

    def refresh(self) -> None:
        self.clear()
        self.update(_registry())


BUILDERS: _Registry = _Registry()
BUILDERS.refresh()


def dispatch(unit: LocatedUnit | Mapping[str, Any], record: Record) -> Built:
    """``BUILDERS[unit.kind].build(unit, record)`` -- the one call A3's ``selective_prover`` makes per selected unit.  ``unit`` is the
    verifier's located mapping or A5/6's :class:`LocatedUnit`; a contributed builder's own refusal type (``ck_elem.Refusal``, ...) reaches
    the prover as :class:`BuildRefusal` with the same sentence, so one exception type means "refused, by name" at this boundary."""

    kind = unit["kind"] if isinstance(unit, Mapping) else unit.kind
    try:
        return BUILDERS[kind].build(unit, record)
    except BuildRefusal:
        raise
    except ValueError as e:
        raise BuildRefusal(str(e)) from e


# --- verifier-side dispatch (A3, integrator 22:20Z): the kinds whose builder derives an expectation from the record ---------------------
#
# A builder that can serve the verifier carries ``family`` and ``expectation(located_unit: Mapping, record) -> ExpectedObligation``
# (CK-ELEM's ``_ElemBuilder``).  An ``InteriorRefuser`` has neither (it refuses at build time and the unit stays an uncovered selected
# equation); the whole-head builder is dispatched by kind in ``verifier_v3.expected.RecordPartition.locate`` (A5/6 patch), not here.


def derives_expectation(kind: str) -> bool:
    """True when ``BUILDERS[kind]`` can derive the verifier's expected statement of a located unit from the record."""

    if kind not in BUILDERS:
        return False
    b = BUILDERS[kind]
    return callable(getattr(b, "expectation", None)) and bool(getattr(b, "family", None))


def expectation_builders() -> dict[str, Builder]:
    """kind -> builder, for every registered kind that :func:`derives_expectation`."""

    return {k: b for k, b in BUILDERS.items() if derives_expectation(k)}


_FAMILIES_ENSURED = False


def ensure_families() -> None:
    """VERIFIER side only (imports ``src/veritor``): make every family a registered builder emits resolvable in ``veritor.interface``
    (``family_gate_set`` / ``check_family_approved`` / ``validate``) -- CK-ELEM's ``ck_elem.ensure_families()`` (elem-bf16, reduce-mufu) and
    E4's ``ck_ftz.ensure_families()`` (ftz-f32), each only when its module is in the checkout.  INTERIM until ``typed_obligation.py`` carries
    the families itself; ``reverify.py`` may call this instead of ``ck_elem.ensure_families()``.  Idempotent; never imported by the prover
    (W-A9-11: the prover's registry import must stay free of ``veritor.*``)."""

    global _FAMILIES_ENSURED
    if _FAMILIES_ENSURED:
        return
    for mod in ("ck_elem", "ck_ftz"):
        try:
            m = __import__(f"{__package__}.{mod}", fromlist=["ensure_families"])
        except ImportError:
            continue
        m.ensure_families()
    _FAMILIES_ENSURED = True


def dispatch_expectation(located_unit: Any, record: Any):
    """``verifier_v3.expected.RecordPartition.locate`` for a kind in :func:`expectation_builders`: the builder derives the expectation
    from the verifier's located unit and the record's committed id lists (never the prover's object)."""

    kind = str(located_unit.get("kind"))
    if not derives_expectation(kind):
        raise BuildRefusal(f"{kind}: no builder derives an expectation (registered with expectation: {sorted(expectation_builders())})")
    ensure_families()
    return BUILDERS[kind].expectation(located_unit, record)


#: [D32.2] the verdict of :func:`can_express` when the statement is representable
OK = "ok"


#: [NIGHT G1, PROPOSED ``gather-equality/v1``] kinds with NO proof obligation: the VERIFIER checks them by opening committed words and comparing
#: (V6's shape).  ``GatherBf16x49152_v1|batch`` (``h[j] = table[tok][j]``): the verifier opens the weights word the committed token selects
#: (``operand_gates[tok]``, weights root pinned) and the VU's committed output and requires equality -- no arithmetic; the check is E9's entry
#: point (``steps`` / ``reverify.py``).  ``can_express`` answers ``ok`` for these; :func:`verifier_covered_rule` names the rule for the census.
VERIFIER_COVERED: dict[str, str] = {"GatherBf16x49152_v1|batch": "gather-equality/v1"}
#: the rule is V-generic (the gather's width is the vocabulary: 49,152 TINY / SmolLM2-360M, 151,936 Qwen2.5-0.5B -- E6 05:25Z); a kind matching
#: this pattern is covered by the same rule (the verifier's check reads V from the kind)
_GATHER_KIND = re.compile(r"^GatherBf16x(\d+)_v1\|batch$")


def verifier_covered_rule(kind: str) -> str | None:
    """The by-name verifier rule covering ``kind`` without a proof, or ``None``."""

    rule = VERIFIER_COVERED.get(str(kind))
    if rule is None and _GATHER_KIND.match(str(kind)):
        rule = "gather-equality/v1"
    return rule


def can_express(statement: Any, unit: Mapping[str, Any] | None = None, *, program: Any = None,
                tables: Mapping[str, Mapping[str, Any]] | None = None, literals: Any = None) -> str:
    """[D32.2, the builder-side expressibility predicate for E3's census] ``can_express(expected_statement) -> "ok" | "E_<reason>"``.

    Signature (coordinated with E3, ``verifier_v3/census.py``): ``statement`` is E3's expected statement of ONE VU as positions
    (``veritor.interface.expected.PositionStatement``: ``kind``, ``operands`` / ``outputs`` as ``ExpectedValue(tree, leaf, offset)``,
    ``operand_gates`` / ``output_gates``, ``origins`` ∈ {input, boundary, interior, literal}); optional ``unit`` = the located unit
    (``vu_index, kind, request_id, gate_interval``) and ``program`` = the pinned ``verity.ir.Program`` -- when BOTH are given the predicate
    runs the registered builder's OWN ``build`` over a phantom record (identities, zero bits), i.e. the exact prover code path (resolvers,
    aliasing, padding literals, prescribed constants, table rows, family resolution, ``validate``, format-3 lowering); without them it runs
    the composition path from the statement alone (:func:`elem_builders.phantom_unit` -> ``build_batch``: same body construction, same
    ``validate``, same lowering).  ``tables`` = the instance's ``commitments.tables`` pin (E6), needed for the MUFU table kinds; ``literals`` =
    the instance's ``commitments.literals`` pin (``prescribed-literal/v1``; absent -> the statement's own literal ids stand in: representability
    does not depend on the rest of the program's literals).  A :data:`VERIFIER_COVERED` kind is ``ok`` (no proof: the verifier's equality check).

    IDENTITY-ONLY: no committed value is read or needed (the phantom leaves are zero bits); the predicate decides REPRESENTABILITY, never
    truth -- the relation is not checked.  Verdicts, by name: ``ok``; ``E_UNSUPPORTED_KIND`` (no registered builder); ``E_UNSUPPORTED_FAMILY``
    (the builder's family resolves in no approved interface gate set); ``E_NO_APPROVED_GUEST`` (no approved ELF implements the family);
    ``E_LITERAL_UNPINNED`` (a literal operand whose word the pinned literal tree does not hold); ``E_INPUT_NOT_PROTECTED`` (an Input gate off
    the outgoing boundary); ``E_DUPLICATE_BOUNDARY_VALUE`` (a repeated word a multi-gate body cannot alias, or a word on both sides); ``E_TABLE_UNPINNED`` (a MUFU
    table kind without ``commitments.tables[<name>]``); any other ``InterfaceReject`` code verbatim; ``E_BUILDER_REFUSED:<text>`` otherwise."""

    from . import elem_builders as _elem

    if isinstance(statement, Mapping):  # E3's census hands a WIDE unit (> 64 operands: the whole-scan select, GEMM) as a summary dict, not positions
        kind = str(statement.get("kind"))
        if _approved_checker_kind(kind) or verifier_covered_rule(kind):
            return OK
        b = BUILDERS.get(kind)
        if b is None:
            return "E_UNSUPPORTED_KIND"
        fam = str(b.family)
        from e2e import approved_guests as _AG

        if any(fam in (g.get("statement_families") or {}) for g in _AG.load_policy().get("guests") or [] if g.get("approval") == "reproduced"):
            return OK  # a statement-bound family (E5's select guest): the relation is the approved ELF; positions are the builder's
        return "E_NO_POSITION_STATEMENT"  # a typed body needs the operand positions to compose; the census gave a summary
    kind = str(statement.kind)
    if kind not in BUILDERS and _approved_checker_kind(kind):
        return OK  # the approved checker's exporter (GemmCoordinate<K>: typed_obligation_export) builds these; not a registry kind
    if verifier_covered_rule(kind):
        return OK  # no proof obligation: the verifier's by-name equality check (gather-equality/v1)
    b = BUILDERS.get(kind)
    if b is None:
        return "E_UNSUPPORTED_KIND"
    ensure_families()
    family = str(b.family)
    from e2e import approved_guests as AG
    from veritor.interface import typed_obligation as TO

    approved = [g for g in AG.load_policy().get("guests") or [] if g.get("approval") == "reproduced"]  # the rule registry_table applies
    if any(TO.family_gate_set(family, w) is not None for w in (16, 32)):  # a typed (gate-set) family
        if not any(family in (g.get("families") or {}) for g in approved):
            return "E_NO_APPROVED_GUEST"
    elif any(family in (g.get("statement_families") or {}) for g in approved):  # a statement-bound family (E5's select guest)
        return OK  # its relation IS the approved ELF; the statement is the builder's (no typed body to compose)
    else:
        return "E_UNSUPPORTED_FAMILY"
    ck = _elem._ck_elem()
    try:
        rec = _elem.PhantomRecord([statement], widths=_elem.phantom_widths(statement), tables=tables, literals=literals)
        if program is not None and unit is not None:
            handle = _elem.open_record(program, rec.commitments, oracles=[rec.oracle([statement])], opener=rec.open_leaf, tables=tables,
                                       literals=rec.literals, profile_id=rec.profile_id)
            handle.phantom = True  # build_batch: no authentication (there is no commitment); the relation is never checked
            if getattr(b, "owner", "") == "ck-ftz":  # E4's writer: the same resolve -> build_many path, with E4's own authenticate flag off
                from . import ck_ftz as _ftz

                a = b._aliased(unit, handle) if hasattr(b, "_aliased") else None
                if a is not None:  # one committed word at several positions: the aliased body through build_batch (as dispatch builds it)
                    _elem.build_batch([a], handle, name="census", authenticate=False)
                else:
                    located = b._located(unit, handle) if hasattr(b, "_located") else unit  # the registry's adaptation (gate / root), as dispatch does
                    _ftz.build_many([_ftz.resolve(located, handle, getattr(handle, "ftz", None))], handle, name="census", authenticate=False)
            else:
                b.build(unit, handle)
        else:
            if kind not in _elem.ALL_BUILDERS:
                return "E_NO_PROGRAM_FOR_KIND"  # E4/E5 builders compose only through their own resolvers: give the census `unit` + `program`
            _elem.build_batch([_elem.phantom_unit(statement, rec)], rec, name="census", authenticate=False)
    except TO.InterfaceReject as e:
        return str(getattr(e, "code", None) or "E_INTERFACE_REJECT")
    except Exception as e:  # noqa: BLE001 -- every refusal / crash is a NAMED verdict, never a pass
        if type(e).__name__ != "Refusal":  # ck_elem.Refusal, ck_ftz.Refusal (E4's own class), ...
            return f"E_BUILDER_CRASH:{type(e).__name__}:{str(e)[:120]}"
        return _refusal_code(str(e))
    return OK


def _refusal_code(text: str) -> str:
    """A builder refusal text -> the interface code it stands for (the builders spell the code, or the class, in their message)."""

    for code in ("E_DUPLICATE_BOUNDARY_VALUE", "E_INPUT_NOT_PROTECTED", "E_IDENTITY_CONFLICT", "E_WIDTH_MISMATCH", "E_ARITY", "E_BOUNDARY_SHAPE"):
        if code in text:
            return code
    if "TABLE_UNPINNED" in text or "no committed table tree" in text:  # E7's and E4's spellings of the unpinned MUFU table
        return "E_TABLE_UNPINNED"
    if "LITERAL_UNPINNED" in text:  # prescribed-literal/v1: the instance pins no literal tree (or not this literal)
        return "E_LITERAL_UNPINNED"
    if "literal" in text or ("Const" in text and ("not a committed" in text or "does not commit" in text or "operands ()" in text or "@" in text)):
        return "E_INPUT_NOT_PROTECTED"  # a prescribed constant / padding literal at an operand: an Input gate off the outgoing boundary
    return f"E_BUILDER_REFUSED:{text[:160]}"


def _approved_checker_kind(kind: str) -> bool:
    """A kind the approved CHECKER covers at VU granularity with an existing exporter (kind_table: GemmCoordinate<K>), outside the registry."""

    try:
        from .kind_table import row_of

        row = row_of(kind)
    except Exception:  # noqa: BLE001
        return False
    return bool(row is not None and row.checker.exists and row.builder.exists and row.checker.granularity == "same" and row.builder.granularity == "same")


def dispatch_kind_program(kind: str, expectation: Any = None) -> tuple[str, str]:
    """``verifier_v3.expected.RecordCircuit.kind_program`` for a kind in :func:`expectation_builders`: (family, canonical body digest);
    an expectation that declares another body is a refusal."""

    if not derives_expectation(kind):
        raise BuildRefusal(f"{kind}: no builder derives an expectation")
    ensure_families()
    b = BUILDERS[kind]
    digest = getattr(b, "subcircuit_digest", lambda: None)()
    declared = getattr(expectation, "subcircuit", None)
    if declared is not None and digest is not None and str(declared).lower() != str(digest).lower():
        from . import elem_builders as _elem

        if _elem.is_aliased_body(kind, declared, expectation):  # registry kinds and E4's one-gate FTZ kinds alike
            return str(b.family), str(declared).lower()  # D28 aliased-body/v1: one committed word read by several operands
        raise ValueError(f"{kind}: expectation subcircuit {str(declared)[:12]} != the builder's body {str(digest)[:12]}")
    if digest is None and declared is None:  # a statement-bound family (E5's select guest): the relation IS the approved ELF, no body digest
        return str(b.family), ""
    return str(b.family), str(digest if digest is not None else declared)


def families_of() -> tuple[str, ...]:
    """The interface families the expectation-capable builders emit (a record may carry proof objects of these besides the audit's)."""

    return tuple(sorted({str(b.family) for b in expectation_builders().values()}))


# --- E7 (EOD 2026-09-11): program-agnostic record + batch packaging over the ONE registry -----------------------------------------------
#
# ``open_record(program, commitments, oracles=..., opener=...)`` -> the record handle any registered one-gate builder reads (E8's CORE
# circuit, TINY's boundary + ``interior/<req>/<step>`` leaves); ``package_batches(located_units, record)`` -> proof-sized statements, one
# family each, n <= 64 units, with the VU identities each covers; ``batches_document`` -> ``proofs.chunks`` + ``per_vu`` (interface 6).
# Implemented in ``elem_builders`` (lazy: no src/veritor import at module load, W-A9-11); re-exported here so the dispatch has one name.


def open_record(*args: Any, **kwargs: Any):
    from .elem_builders import open_record as _open

    return _open(*args, **kwargs)


def package_batches(*args: Any, **kwargs: Any):
    from .elem_builders import package_batches as _pack

    return _pack(*args, **kwargs)


def batches_document(*args: Any, **kwargs: Any):
    from .elem_builders import batches_document as _doc

    return _doc(*args, **kwargs)


def registry_table() -> list[dict[str, Any]]:
    """The dispatch registry as a table: kind -> builder (module.class, owner) -> guest family -> whether the verifier derives its expectation.
    ``guest`` is the approved guest entry (``approved_guests.json``) that lists the family; ``None`` when no approved ELF lists it."""

    import json
    from pathlib import Path

    policy = json.loads((Path(__file__).resolve().parents[1] / "approved_guests.json").read_text())
    by_family: dict[str, list[str]] = {}
    for g in policy.get("guests") or []:
        if g.get("approval") != "reproduced":
            continue  # the table names the guests a proof may be accepted under today: `provisional-legacy` entries are not among them
        for fam in {*(g.get("families") or {}), *(g.get("statement_families") or {})}:  # gate-set families + statement-bound ones (E5's select guest)
            by_family.setdefault(fam, []).append(str(g["elf_sha256"])[:12])
    rows = []
    for kind in sorted(BUILDERS):
        b = BUILDERS[kind]
        fam = getattr(b, "family", None)
        rows.append({"kind": kind, "builder": f"{type(b).__module__}.{type(b).__name__}", "owner": getattr(b, "owner", None), "family": fam,
                     "derives_expectation": derives_expectation(kind), "approved_guests": by_family.get(str(fam), []) if fam else [],
                     "refuser": type(b).__name__ == "InteriorRefuser"})
    return rows
