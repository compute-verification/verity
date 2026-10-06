import Specs.Flock.Guarantees.Table
import Proofs.Flock.Soundness.Soundness

/-!
# Proofs of one table's soundness

One theorem `Flock.SecurityProofs.Table.G : Flock.Guarantees.Table.G` per guarantee, which is the theorem in
`Proofs.Flock.Soundness.Soundness` that proves the same statement under its old name.
-/

namespace Flock.SecurityProofs.Table

theorem Sound : Flock.Guarantees.Table.Sound := @FlockSoundness.table_sound
theorem SoundFast100 : Flock.Guarantees.Table.SoundFast100 := @FlockSoundness.table_sound_fast100
theorem SoundFast100_34_35 : Flock.Guarantees.Table.SoundFast100_34_35 := @FlockSoundness.table_sound_fast100_34_35

end Flock.SecurityProofs.Table
