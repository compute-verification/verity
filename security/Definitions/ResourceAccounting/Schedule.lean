module

public import Mathlib.Logic.Embedding.Basic

/-! A deterministic schedule, `ScheduleDeterministic`, which counts the `a` that `SteeredLayout` reads. -/

@[expose] public section

namespace ResourceAccounting

/-- `canon` and `enc` are the verifier's, fixed before the coins: the prover chooses only the schedule. `a` is the cap
committed before the coins, never the bits of the one schedule published after them. -/
def ScheduleDeterministic {Pfx Y S : Type} {N T M : ℕ} (canon : Pfx → S → Fin N ↪ Fin T × Fin M)
    (enc : S → List Bool) (a : ℕ) (layout : Pfx → Y → Fin N ↪ Fin T × Fin M) : Prop :=
  Function.Injective enc ∧ ∀ c y, ∃ s, (enc s).length ≤ a ∧ layout c y = canon c s

end ResourceAccounting
