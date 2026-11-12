"""verity.definitions.ops.boolean: word arithmetic as Boolean circuits.

Each module's docstring says what it holds, most of them the next version of an `ops` module's word Definitions, the
same function on bits.

The builders live elsewhere: `forms`, `gather` and `trace` in `verity.circuits.boolean`; `fp`, `fp4` and `tc_step` in
`verity.definitions.silicon`; the universal unit in `verity.definitions.private_circuits.universal`; `sha512` and `merkle` in
`verity.commitments.gates`. Flock's `verity_flock.gf2`, `.fp`, `.unit` and `.unit_fp4` are `forms`, `fp`, `tc_step`
and `fp4` under their Flock names, `verity_flock.tail_pieces` imports `softmax`'s and `scalar`'s, and
`verity_flock.sha512_circuit` imports the gates' `sha512`.
"""
