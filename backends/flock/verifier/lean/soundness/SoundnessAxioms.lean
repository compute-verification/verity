/- `lake env lean SoundnessAxioms.lean`: the axioms #89's headline theorems depend on (the coordinator's audit allows only Lean's
standard three; BCHKS25 is a hypothesis, not an axiom). -/
import FlockSoundness
#print axioms FlockSoundness.rep_sound
#print axioms FlockSoundness.table_sound
#print axioms FlockSoundness.table_sound_fast100
#print axioms FlockSoundness.table_sound_fast100_34_35
