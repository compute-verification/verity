module

public import Proofs.Flock.Soundness.Discharge.Exec.Event
import Proofs.Flock.Soundness.Game
import Proofs.Flock.Soundness.Registered.Binding
public import Proofs.Flock.Soundness.Registered.ReadsX
import Mathlib.Combinatorics.Matroid.Init

/-! # What V*'s bridge obligation reads of `Recursive.ZkReg.Off`, without its recursion imports. -/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion

/-- **A commit string that is a row of a unit of `S`.** -/
def rowIn {n : ℕ} {Q G : Type} (S : Finset (Fin n)) : (Fin n × Q) ⊕ G → Prop
  | .inl (u, _) => u ∈ S
  | .inr _ => False

end FlockSoundness.Discharge.Recursion
