# The lowering: a drawn wrong unit leaves its table unsatisfied

`Partition.LoweringSound` and `Partition.InputsLoweringSound` (`FlockSoundness/Audit/Flock.lean`) at the oracle layer, and
`Partition.LoweringSoundB` and `Partition.LoweringSoundC` (`Audit/FlockBatched.lean`, `Audit/FlockCompiled.lean`) at the
compiled layer, taken by the Flock audit theorems (`flock_session_sound`, `flock_inputs_sound`, `flock_batched_count`,
`flock_compiled_count` and their siblings).

- **What they say.** At the oracle layer, a drawn wrong unit lies in a session table whose statement no committed table
  satisfies, and inputs other than the anchors' leave the always-proved input statement unsatisfied. At the compiled
  layer, a message packing a witness that satisfies the statement of a drawn unit's table decodes, for that unit, to
  values on which it is correct.
- **Discharged** at the compiled layer by `loweringSoundB_of_place` and `loweringSoundC_of_place`, from each drawn
  unit's `UnitPlace` (`lowering_sound`, `../README.md` §1.5), given the statement builder's placement (`placement.md`)
  and L1 (`l1-template-rows.md`). At the oracle layer, by the lowering theorem (level 3, another lane).
