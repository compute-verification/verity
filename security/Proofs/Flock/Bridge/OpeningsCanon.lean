module

public import Proofs.Flock.Bridge.Canon

/-!
# The length of `bytesAt`

`bytesAt_length`: `Binding.bytesAt z a` has `N` bytes, one per position of `a`.
-/

@[expose] public section

namespace Flock.Bridge

/-- A witness's bytes at `N` addresses are `N` bytes. -/
theorem bytesAt_length {m N : ℕ} (z) (a : Fin N → Fin 8 → Fin (2 ^ m)) :
    (FlockSoundness.Binding.bytesAt (m := m) z a).length = N := by
  simp [FlockSoundness.Binding.bytesAt]

end Flock.Bridge

