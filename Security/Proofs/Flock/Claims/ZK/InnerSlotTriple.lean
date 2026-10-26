import Proofs.Flock.Claims.ZK.SplitInner

/-!
# `InnerSlot` and `InnerTriple`

Two of `SplitInner`'s statements, each from its `SimInner` lemma at `InnerAt`'s run:

* **`innerSlot_holds`**: the shared phases end at the rep's `ε` slot (`innerPre_slot`).
* **`innerTriple_holds`**: the cut's `ρ_in`, `σ_in` are the session's at the honest pads (`innerTripleE`).
-/

namespace Claims.ZKMal

open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

theorem innerSlot_holds : InnerSlot := by
  rintro x w τ c n j r cons L ⟨-, -, -, s, spec, hs, dom, cm0, ic, extras, he, l, hl, o, t', l', ho, -, rfl⟩
  exact innerPre_slot hs w τ c n j r dom cm0 ic extras he hl ho

theorem innerTriple_holds : InnerTriple := fun x w τ c _ j r _ _ _ _ => innerTripleE x c w τ j r rfl

end Masked

end Claims.ZKMal
