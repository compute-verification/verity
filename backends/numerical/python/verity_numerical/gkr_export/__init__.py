"""Export of the WP2 checker as a layered arithmetic circuit for Candidate A (``backends/gkr``).

The gadgets of ``verity_numerical.checker`` are executed once on symbolic values (:class:`CircuitCtx`) so that the
five predicates become: quadratic gates (``mul``, ``boolean``, the ``lt_pow`` products), linear assertions
(``eq``, range-chunk splits, query-wire bindings) and lookup queries (16-bit range chunks into ``R16``, rows of
``POW``/``ALIGN``/``LEAD``/``NORM``).  Every value a lookup reads is an input column (the committed layer): a query
on a computed wire is materialised as a column bound by an assertion (PROTOCOL.md 4.3, ``commit_query_wires``).
Witnesses come from the WP2 ``gen_step`` (the base columns) plus the derived columns this module defines
(range chunks, looked-up rows, materialised query wires).  Nothing here changes the accepted (x, y) set: every
predicate of the checker is emitted, the non-power-of-two ranges (e.g. ``t < 255``) as two one-sided checks.
"""
from .circuit import (
    CircuitCtx,
    build_unit_circuit,
    export_circuit,
    random_finite_unit,
    unit_witness,
    write_witness,
)

__all__ = ["CircuitCtx", "build_unit_circuit", "export_circuit", "random_finite_unit", "unit_witness", "write_witness"]
