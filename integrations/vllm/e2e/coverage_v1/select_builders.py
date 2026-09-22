"""CK-SELECT's builder -- a *contribution* to the one registry ``e2e.coverage_v1.builders.BUILDERS`` (E7 registers it).

One signature (station 1): ``build(located_unit, record) -> ck_select.Built`` where ``located_unit`` is the verifier's
located VU (``vu_index``, ``request_id``, ``kind``, ``gate_interval``) and ``record`` a ``ck_elem.RecordHandle``; the
verifier side calls ``expectation(located_unit, record) -> ck_select.SelectStatement`` (from the program text and the
committed id lists only) and compares the proof's public digest with ``sha256(expectation.encode())``
(``ck_select.check_expected`` names every differing field).

Program-agnostic: the builder never assumes V, the literals or the gate layout -- ``ck_select.pin_function`` reads them from
the located ``TokenSelect`` call, and refuses a body that is not the registered one.

Kinds: ``TokenSelect_v1|whole-scan`` (Amendment A2 decision 4 / population_v3.TOKEN_SELECT_KIND) and -- for records whose
selection still names the scan steps -- ``ArgmaxStep_v1|scan``, which resolves to its enclosing whole-scan unit (the
``Built.unit.regrouped_step`` says which step was selected).  E6 owns whether a step VU counts as covered by the whole-scan
proof; this module only builds the whole-scan statement either way.

[W-A9-11] This module is on the PROVER's import path: nothing under ``src/veritor`` is imported here or in ``ck_select``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["BUILDERS", "SELECT_FAMILIES", "SelectBuilder"]

SELECT_FAMILY = "veritor.select-argmax@1"

#: kind -> interface family (== ``ck_select.FAMILY``)
SELECT_FAMILIES: dict[str, str] = {
    "TokenSelect_v1|whole-scan": SELECT_FAMILY,
    "ArgmaxStep_v1|scan": SELECT_FAMILY,
}


def _ck():
    from . import ck_select

    return ck_select


@dataclass(frozen=True)
class SelectBuilder:
    kind: str
    family: str = SELECT_FAMILY
    owner: str = "ck-select (E5)"
    #: verifier-side dispatch marker (``coverage_v1.builders.derives_expectation``): the expected STATEMENT is derived here.
    #: NOTE for E7: the expectation is a ``ck_select.SelectStatement`` (family ``veritor.select-argmax@1``), not a
    #: typed-obligation ``ExpectedObligation`` -- the verifier checks ``sha256(statement)`` against the proof's public
    #: values (``verify_transparent`` / ``veritor-select-host verify --statement``), not per-position openings.
    derives_expectation: bool = True

    def build(self, located_unit: Mapping[str, Any], record: Any) -> Any:
        ck = _ck()
        if str(located_unit.get("kind")) != self.kind:
            raise ck.Refusal(f"BUILDERS[{self.kind!r}] asked to build a {located_unit.get('kind')!r} unit")
        return ck.build(located_unit, record)

    def expectation(self, located_unit: Mapping[str, Any], record: Any) -> Any:
        ck = _ck()
        if str(located_unit.get("kind")) != self.kind:
            raise ValueError(f"BUILDERS[{self.kind!r}] asked for the expectation of a {located_unit.get('kind')!r} unit")
        return ck.expectation(located_unit, record)

    def check_expected(self, statement_bytes: bytes, expected: Any) -> list[str]:
        return _ck().check_expected(statement_bytes, expected)

    def subcircuit_digest(self) -> None:
        """No typed-obligation body: the relation is the guest's (family digest = the approved ELF); see the class note."""

        return None


BUILDERS: dict[str, SelectBuilder] = {k: SelectBuilder(k, fam) for k, fam in SELECT_FAMILIES.items()}
