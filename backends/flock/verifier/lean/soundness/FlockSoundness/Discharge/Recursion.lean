import FlockSoundness.Discharge.Recursion.Defs
import FlockSoundness.Discharge.Recursion.Compose
import FlockSoundness.Discharge.Recursion.Fork
import FlockSoundness.Discharge.Recursion.Sound

/-!
# Recursion

A hidden developer circuit proved inside a public verifier's `--zk` session, with an interactive inner protocol whose
messages are committed before live coins (`Recursion.Defs`); its soundness from outer soundness, the commitments'
binding to reference openings fixed before each coin, and inner soundness (`commit_and_prove`, `stage_bind`,
`recursive_sound`).
-/
