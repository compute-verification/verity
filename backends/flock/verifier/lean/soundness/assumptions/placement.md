# `Placement`: each drawn unit's rows sit at its place in its table

`Placement` and `UnitPlace` (`FlockSoundness/Lowering.lean`), taken by `lowering_sound`, by `loweringSoundB_of_place` and
`loweringSoundC_of_place`, and by the `_placed` forms of the composed audit (`flock_batched_count_placed`,
`flock_batched_drawn_placed`).

- **What it says.** A statement places a unit's rows in a block when the block matrices' rows at the unit's computed
  columns read exactly its rows' columns, and the rows at its constant read the pin, as every slot's constant is wired to
  it (spec §16.1). `UnitPlace` carries a drawn unit's template rows, its instance in the circuit (`IsRowsUnit`), and the
  block and columns where its table's statement holds them.
- **To be discharged by** the statement builder. Proving it for the executable's statement (the slots with Δ XORed in,
  spec §16.1) is part of the level-3 refinement, with lane `flock-verifier`.
