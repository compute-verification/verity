"""Pearl-C4's committed rows (`pc4`, `pc4_rowk`) against `leaves`, on `test_circuit_leaves`' helpers: with PoUW's
other units' rows they are exactly the rows with a schema, each unit's statement rows are its ports' schemas, and a row
unit fixes its ok word's bits 1-15 in gates."""
from __future__ import annotations

import pytest

import test_circuit_leaves as T
from verity.commitments.rowleaf import ROLE_W, ROLE_X
from verity.protocols.resource_accounting.compute.pouw.prover.circuit import leaves as L
from verity.protocols.resource_accounting.compute.pouw.prover.circuit import inert_bound, pc8, rowk
from verity.protocols.experimental.pearl_c4.prover import pc4, pc4_rowk

#: each unit's statement rows: its input rows' (row, role) by read, then each output row's (row, role) and run of outputs
STATEMENTS = {
    "row_a": ([("bf16", ROLE_X)], [(("nvfp4-p", ROLE_X), 3), (("ok", ROLE_X), 1)]),
    "row_b": ([("bf16", ROLE_W)], [(("nvfp4-p", ROLE_W), 3), (("ok", ROLE_W), 1)]),
    "tile": ([("nvfp4-p", ROLE_X)] * 2 + [("nvfp4-p", ROLE_W)] * 3, [(("tile", ROLE_X), 1)]),
}


def test_the_units_rows_are_exactly_the_rows_with_a_schema():
    """`pc4.PORTS` and the row_k cut's `pc4_rowk.PORTS4`, with PoUW's other units' rows, are exactly the rows with a
    schema."""
    used = T.rows_used(pc4.PORTS, pc8.PORTS, {"side": pc8.SIDE_PORTS}, inert_bound.PORTS, pc4_rowk.PORTS4, rowk.PORTS8)
    assert used == set(L.ROW_SCHEMA)


@pytest.mark.parametrize("role", list(STATEMENTS))
def test_a_units_statement_rows_are_its_ports_schemas(role):
    """`test_circuit_leaves`' on Pearl-C4's units: the row a row unit writes as its codes, scales and P (A′_i ‖ P_A,i,
    B̃_j ‖ P_B,j) is one run of output ports."""
    T.statement_rows_are_ports_schemas(pc4.PORTS[role], role, *STATEMENTS[role])


@pytest.mark.parametrize("fn", [pc4.templates(64)[r] for r in ("row_a", "row_b")], ids=lambda fn: fn.id)
def test_a_row_unit_fixes_its_ok_words_bits_1_to_15_in_gates(fn):
    T.test_a_row_unit_fixes_its_ok_words_bits_1_to_15_in_gates(fn)
