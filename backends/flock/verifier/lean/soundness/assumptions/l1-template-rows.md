# L1. Each template's rows compute its gates

A named hypothesis, outside Lean: the rows' outputs are the gates' outputs, in the program's Boolean circuit, on every
input. The theorems are about the rows the verifier parses (`Prog.circuit`, `../README.md` §1.5), so the audit certifies
drawn units against their rows. That says something about the program's Boolean circuit only through L1.

- **Evidence today:** the IR comparison tests in `backends/flock/tests/test_ir_lowering.py`, on sampled inputs.
  - They evaluate the pinned rows against the IR's primitives:
    - RoPE on 8000 inputs (`test_rope_unit_matches_its_primitives`);
    - SiLU on every gate word and 8000 more (`test_silu_unit_covers_the_whole_table`);
    - both on the captured slices (`test_captured_slices`).
  - They evaluate the pieces the rows are built from, as gate circuits, against the IR's primitives, on 6000 inputs
    each (`test_pieces_match_the_ir_primitives`, `test_bf16_adds_match_the_ir_primitives`).
  - A test in the same file mutates one row and checks that the comparison catches it.

  They sample. They don't prove it.
- **The path to discharge it:** a verified lowering. That means gadget proofs, with Lean re-deriving the rows from the
  gate circuit, so that the pinned rows equal the rows Lean derives. This matches the plan of record for V[B], which
  Lean generates with an equivalence proof.
