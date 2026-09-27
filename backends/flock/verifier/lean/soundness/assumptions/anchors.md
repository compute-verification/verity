# `AnchorsSound`: the committed inputs are the anchors' values

`Partition.AnchorsSound` (`FlockSoundness/Audit/OneStage.lean`), the term `δ_in` of the audit's cone theorems
(`Partition.audit_cone`, `audit_cone_count`, `audit_covered`).

- **What it says.** The session accepts while the committed inputs differ from the anchors' values with probability at
  most `δ_in`.
- **Discharged** at the oracle layer by `InputsSound` (`oracle_anchorsSound`). For Flock that is `flock_inputs_sound`,
  `2^-205`, given `InputsLoweringSound` (`lowering.md`). At the compiled layer it awaits `registered_weights`
  (`../README.md` §1.3).
