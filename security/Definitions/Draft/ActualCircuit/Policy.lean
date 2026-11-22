module

public import Definitions.Draft.ActualCircuit.Epoch

/-!
# A trivial policy's statement (draft)

The policy accepts when compute accounting's number, read off the verifier's view, is within a limit `lim`; it draws no
challenge. By itself it says nothing about the actual circuit: its set is all actual circuits when the number is within
the limit, and it never accepts otherwise. Its content is about the view, and it bounds the actual circuit only beside
compute accounting's statement, whose bound the number is (`Draft.ActualCircuit.FourFromParts`).
-/

@[expose] public section

namespace Draft.ActualCircuit

open ResourceAccounting

/-- **The policy in one epoch.** If the verifier at `V` accepts (the number `num` its view carries is at most `lim`)
then, except with probability `0`, the actual circuit lies in the set of all actual circuits for which that number is
at most `lim`: all of them on any view it accepts. Not vacuous because: I cannot name one; its conclusion holds of
every strategy, the set saying nothing of the actual circuit and its property being the view's. -/
def PolicyEpoch : Prop :=
  ∀ {L : Type*} [LinearOrder L] (Ω : Type) [Fintype Ω] (𝓕 : Set (Actual.Run L)) (V : Finset L)
    (num : VerifierView L → ℕ) (lim : ℕ),
    Sound Ω 𝓕 V (fun τ => num τ ≤ lim) (Lies V fun τ => if num τ ≤ lim then Set.univ else ∅) 0

end Draft.ActualCircuit
