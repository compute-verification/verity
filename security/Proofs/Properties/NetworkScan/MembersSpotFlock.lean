module

public import Properties.NetworkScan.Members
public import Proofs.Flock.SpotFlock.Code

/-!
# SPOT[Flock]'s statements are their members side by side

SPOT[Flock]'s parse (`SpotFlock.parse h`) gives every statement the auditor's held descriptor (`Held.desc`) and the
held members' programs side by side (`bitsProgs h.ms`), whatever the run's message. So at every compiled sampled
verifier whose parse is SPOT[Flock]'s, `StmtMembers` at the held members is what the auditor's held copies are: the
descriptor decodes, and the members are its templates (`MembersOf`). Both are of the copies the auditor fixes before
the run, and the auditor can evaluate both.
-/

@[expose] public section

namespace Properties.NetworkScan.MembersProof

/-- **SPOT[Flock]'s statements are their held members side by side**, where the held descriptor decodes and the held
members are its templates. -/
theorem stmtMembers_spotFlock (h : SpotFlock.Held) {Coin : Type} {A : Claims.SamplingGeneric.Arg Coin}
    (D : Claims.SamplingGeneric.Compiled Coin A) (hD : D.parse = SpotFlock.parse h) {P : Flock.Program.Prog}
    (hP : Flock.Program.decode ⟨h.desc.toArray⟩ = .ok P) (hM : MembersOf P h.ms) :
    StmtMembers D (fun _ => h.ms) := by
  intro x s hx
  rw [hD] at hx
  unfold SpotFlock.parse at hx
  obtain ⟨st, -, rfl⟩ := Option.map_eq_some_iff.1 hx
  exact ⟨P, hP, rfl, hM⟩

end Properties.NetworkScan.MembersProof
