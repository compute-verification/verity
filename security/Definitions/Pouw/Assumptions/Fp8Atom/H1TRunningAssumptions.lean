module

public import Definitions.Pouw.Fp8Atom.H1TLemmas

/-! `H1TRunningWords`, the named assumption that `DistinctLiveH1T` holds on the pairs of words the one-flip argument does not cover. -/

@[expose] public section

namespace Pouw.Fp8Atom.H1T

/-- Equivalent to `DistinctLiveH1T` given `H1TOneFlip` (`h1tRunningOfDistinct`, `h1tDistinctLive`). -/
def H1TRunningWords : Prop :=
  ∀ U : Unit, U.Legal → ∀ (i : Fin U.m) (a a' : Fin U.n × Pos U.T),
    (a ≠ a' → Hard a.1.val a'.1.val a.2 a'.2 → U.word (i, a) ≠ U.word (i, a')) ∧
      (Stuck a.1.val a.2 → ∀ q : ℚ, U.word (i, a) ≠ fun _ => q)

end Pouw.Fp8Atom.H1T
