import FlockSoundness.Discharge.Recursion.Defs
import FlockSoundness.Discharge.Recursion.Compose
import FlockSoundness.Discharge.Recursion.Fork
import FlockSoundness.Discharge.Recursion.Sound
import FlockSoundness.Discharge.Recursion.Flock
import FlockSoundness.Discharge.Recursion.Target
import FlockSoundness.Discharge.Recursion.Karatsuba

/-!
# Recursion

A hidden developer circuit proved inside a public verifier's `--zk` session, with an interactive inner protocol whose
messages are committed before live coins (`Recursion.Defs`); its soundness from outer soundness, the commitments'
binding to reference openings fixed before each coin, and inner soundness (`commit_and_prove`, `stage_bind`,
`recursive_sound`), and its instance with the outer proof a `--zk --registered` session of a fixed public verifier
(`zk_session_recursive`). What V* computes: the compiled table's acceptance of a hidden block statement
(`Recursion.Target`: `InnerAccepts`, `hidden_sound`), and its gadgets, starting with GF(2^128) multiplication
(`Recursion.Karatsuba`: `sound_mul128`).
-/
