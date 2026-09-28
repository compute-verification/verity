import Definitions.Pouw.PearlC.Instance

/-!
# Pearl-C's named protocol (proofs)

`pearlCProtocol sem c ρ`'s costs are Pearl-C's accounting at `(Params.pi, c)` with cap `ρ`
(`pearlCProtocol_accounting`).
-/

namespace Pouw.PearlC

variable {Q R S : Type}

theorem pearlCProtocol_accounting (sem : PearlCSem Q R S) (c : Costs) (ρ : ℚ) :
    PearlCAccounting (pearlCProtocol sem c ρ) Params.pi c ρ := by
  refine ⟨fun _ _ => rfl, fun H s U u act h => ?_⟩
  show (((1 - ρ) * creditOf Params.pi c (U.layout.shape u) : ℚ) : ℝ) ≤
    ((creditOf Params.pi c (U.layout.shape u) - debitOf Params.pi c (sem.unitDebit H s U u act) : ℚ) : ℝ)
  have h' : debitOf Params.pi c (sem.unitDebit H s U u act) ≤ ρ * creditOf Params.pi c (U.layout.shape u) := h
  exact_mod_cast (by linarith)

end Pouw.PearlC
