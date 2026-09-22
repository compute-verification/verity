"""CK-FTZ's per-kind scalar obligation builders -- a *contribution* to the one registry ``e2e.coverage_v1.builders.BUILDERS``
(E4, EOD 2026-09-11; E7 owns ``builders.py`` and folds this dict in as ``elem_builders.BUILDERS`` is folded in today).

One signature for every kind (station 1): ``build(located_unit, record) -> Built`` and
``expectation(located_unit, record) -> ExpectedObligation``, where ``located_unit`` is the verifier's located VU
(``vu_index``, ``request_id``, ``kind``, ``gate``, ``root``, ``replay_unit``) and ``record`` the CK-ELEM
``RecordHandle`` (pinned program, ``commitments.json``, capture reader) extended with the FTZ **word resolver**.

Program-agnostic.  The builder never reads a kernel: it takes the VU's gate in the pinned program, the gate's operand
gates from the program's signature, and asks the record where those gates' words are committed.  That question is
the :class:`ck_ftz.WordResolver` protocol -- ``word(request_id, gate) -> CommittedWord | None`` (boundary, weights
or E2's interior trees) and ``table_row(table, index) -> CommittedWord | None`` (the pinned MUFU table trees).  The
resolver is ``record.ftz`` when the record offers one (E2/E3 attach it: :func:`attach_resolver`), else CK-ELEM's
boundary/weights locator, whose 16-bit BF16 words an f32 kind refuses by width -- the exact reason today's records
cannot build these kinds (their operands are interior f32 words no record commits yet).  :class:`MappingWords` is
the simplest resolver: explicit ``(request_id, gate) -> CommittedWord`` and ``(table, index) -> CommittedWord``
dictionaries, which is what E2's interior index reduces to per VU and what the integration test uses.

[W-A9-11] This module is on the PROVER's import path (``selective_prover`` -> ``coverage_v1.builders`` -> here) and
the prover runs in the pinned vLLM environment (Python 3.11, ``PYTHONPATH=.`` only): nothing under
``src/veritor`` is imported at module level; ``ck_ftz`` (which imports ``veritor.interface.*``) is imported lazily
inside the verifier-facing methods.  ``FTZ_FAMILIES`` is pinned here and asserted equal to ``ck_ftz.KINDS[k].family``
by ``tests/contract/e4_ftz_test.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # pragma: no cover
    from veritor.interface.expected import ExpectedObligation

__all__ = [
    "Builder",
    "BUILDERS",
    "FTZ_FAMILY",
    "FTZ_FAMILIES",
    "TINY_KINDS",
    "MappingWords",
    "attach_resolver",
    "dispatch_expectation",
    "dispatch_kind_program",
    "families_of",
    "kinds_of_family",
]

FTZ_FAMILY = "veritor.ftz-f32@1"

#: The FTZ kinds attempt 6's TINY selection sampled (``out/records/V5-TINY-bex0-inc-elem-b8177939.md``: 19 VUs of the
#: first four; the other three are the same kernel's scalar epilogue/guard kinds, sampled in other RUs of the partition).
TINY_KINDS: tuple[str, ...] = (
    "F32MulFtz_v1|batch",
    "F32MulFtz_v1|primitive",
    "F32AddFtz_v1|primitive",
    "F32FmaSubFtz_v1|primitive",
    "Fa2InvSum_v1|primitive",
    "F32Max_v1|primitive",
    "GuardNegInfZero_v1|primitive",
    "MufuEx2Ftz_v1|primitive",
)

#: kind -> interface family (== ``ck_ftz.KINDS[k].family`` for every kind); the two siblings are registered too so a
#: partition that samples them dispatches here rather than to "no checker".
FTZ_FAMILIES: dict[str, str] = {k: FTZ_FAMILY for k in TINY_KINDS + ("F32SubFtz_v1|primitive", "F32FmaFtz_v1|primitive")}


def _ck_ftz():
    try:
        from . import ck_ftz  # verifier / obligation-stage side only (imports src/veritor)
    except ImportError:  # loaded as a top-level module (tests, scripts)
        import ck_ftz  # type: ignore[no-redef]

    return ck_ftz


class Builder(Protocol):
    kind: str
    family: str
    owner: str

    def build(self, located_unit: Mapping[str, Any], record: Any) -> Any: ...

    def expectation(self, located_unit: Mapping[str, Any], record: Any) -> ExpectedObligation: ...


# --- resolvers -------------------------------------------------------------------------------------------------------


@dataclass
class MappingWords:
    """A :class:`ck_ftz.WordResolver` over explicit dictionaries: ``words[(request_id, gate)] -> CommittedWord`` and
    ``rows[(table, index)] -> CommittedWord``.  E2's interior index yields exactly these entries for the gates a VU
    touches; the test builds them from a synthetic record."""

    words: dict[tuple[str, int], Any] = field(default_factory=dict)
    rows: dict[tuple[str, int], Any] = field(default_factory=dict)

    def word(self, request_id: str, g: int) -> Any | None:
        return self.words.get((str(request_id), int(g)))

    def table_row(self, table: str, index: int) -> Any | None:
        return self.rows.get((str(table), int(index)))


def attach_resolver(record: Any, resolver: Any) -> Any:
    """Give ``record`` the FTZ word resolver the builders read (``record.ftz``); returns ``record``."""

    record.ftz = resolver
    return record


# --- builders --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _FtzBuilder:
    kind: str
    family: str
    owner: str = "ck-ftz"
    #: verifier-side dispatch marker (``coverage_v1.builders.derives_expectation``)
    derives_expectation: bool = True

    def build(self, located_unit: Mapping[str, Any], record: Any) -> Any:
        ck = _ck_ftz()
        if str(located_unit.get("kind")) != self.kind:
            raise ck.Refusal(f"BUILDERS[{self.kind!r}] asked to build a {located_unit.get('kind')!r} unit")
        return ck.build(located_unit, record, getattr(record, "ftz", None))

    def expectation(self, located_unit: Mapping[str, Any], record: Any) -> ExpectedObligation:
        if str(located_unit.get("kind")) != self.kind:
            raise ValueError(f"BUILDERS[{self.kind!r}] asked for the expectation of a {located_unit.get('kind')!r} unit")
        ck = _ck_ftz()
        return ck.expectation(located_unit, record, getattr(record, "ftz", None))

    def subcircuit_digest(self) -> str:
        ck = _ck_ftz()
        return ck.kind_subcircuit(ck.KINDS[self.kind]).digest


#: kind -> builder.  Every kind of the family is registered: the builder REFUSES (never silently skips) a unit whose
#: operands the record does not commit, naming the gate and the missing tree -- the honest per-unit answer until E2's
#: interior trees (and E6/E7's pinned MUFU table trees for the two table kinds) are in the commitment set.
BUILDERS: dict[str, Builder] = {k: _FtzBuilder(k, fam) for k, fam in FTZ_FAMILIES.items()}


def dispatch_expectation(located_unit: Mapping[str, Any], record: Any) -> ExpectedObligation:
    kind = str(located_unit.get("kind"))
    b = BUILDERS.get(kind)
    if b is None:
        raise KeyError(f"{kind}: no builder registered (ftz BUILDERS)")
    return b.expectation(located_unit, record)


def dispatch_kind_program(kind: str, expectation: Any = None) -> tuple[str, str]:
    b = BUILDERS.get(kind)
    if b is None:
        raise KeyError(f"{kind}: no builder registered (ftz BUILDERS)")
    digest = getattr(b, "subcircuit_digest", lambda: None)()
    declared = getattr(expectation, "subcircuit", None)
    if declared is not None and digest is not None and str(declared).lower() != str(digest).lower():
        raise ValueError(f"{kind}: expectation subcircuit {str(declared)[:12]} != the builder's body {str(digest)[:12]}")
    return b.family, str(digest if digest is not None else declared)


def families_of(builders: Mapping[str, Builder] = BUILDERS) -> tuple[str, ...]:
    return tuple(sorted({b.family for b in builders.values()}))


def kinds_of_family(family: str, builders: Mapping[str, Builder] = BUILDERS) -> tuple[str, ...]:
    return tuple(sorted(k for k, b in builders.items() if b.family == family))
