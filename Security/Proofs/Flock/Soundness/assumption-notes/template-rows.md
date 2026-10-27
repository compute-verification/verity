# A template's rows compute its gates: the evidence, and the general proof

This is not a hypothesis of the end-to-end theorem: the theorem is about the circuit of
the statement's rows, and what a correct unit computes is its certificate's (`IsRowsUnit.computes_of_cert`,
`../Certificate.lean`; `../ASSUMPTIONS.md`, The circuit and the certificate). The certificate's premises
are that a unit's rows are `GateRows.rows G L` for its gate circuit `G`, and that `G` computes the unit's definition. This
file is the evidence for those premises.

- **Evidence for every template:** the IR comparison tests in `verity/ml/flock/tests/test_ir_lowering.py`, on sampled
  inputs.
  - They evaluate the pinned rows against the IR's primitives:
    - RoPE on 8000 inputs (`test_rope_unit_matches_its_primitives`);
    - SiLU on every gate word and 8000 more (`test_silu_unit_covers_the_whole_table`);
    - both on the captured slices (`test_captured_slices`).
  - They evaluate the pieces the rows are built from, as gate circuits, against the IR's primitives, on 6000 inputs
    each (`test_pieces_match_the_ir_primitives`, `test_bf16_adds_match_the_ir_primitives`).
  - A test in the same file mutates one row and checks that the comparison catches it.

  They sample. They don't prove it.
- **The general proof** is the typed layout's. For every layout `deriveChecked` accepts, rows satisfied with the
  constant at 1 carry `evalT` of the layout's type on its outputs (`Types.Dag.layout_sound`, `../Types/Dag.lean`), and
  a template Definition whose rows are a checked layout's computes that type (`TmplValue.defComputes_of_layout`,
  `../TmplComputes.lean`). A template's certificate reaches it once its rows are a checked layout's; until then its
  premises rest on the tests above. `GateRows.rows_sound` (`../GateRows.lean`) proves the same for rows derived from any
  gate circuit and layout by the v2 lowering rule.
