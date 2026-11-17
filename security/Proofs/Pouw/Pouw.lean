module

public import Definitions.Pouw.Guarantees.Pouw
import Mathlib.Tactic.Positivity.Finset
import Proofs.Registration.Device
import Proofs.Pouw.Game
import Proofs.Pouw.Prob

/-! PoUW's guarantee is a joint run's compute guarantee. -/

@[expose] public section

namespace ResourceAccounting.SecurityProofs

open ResourceAccounting Pouw

theorem ComputeOfPouw : ResourceAccounting.Guarantees.ComputeOfPouw := by
  intro Q R S _ _ _ _ CM P D γ₀ Ωmax ε hγ hΩ hWref hTT U hU t εs hεs hne ι α ΩQ ΩS _ _ _ d r p hq real o T hT
    hwork
  obtain ⟨⟨_, _, tix⟩⟩ := hne
  have hA : (real.adv o).Bounded U T (d.rate .oracle * (r.win .compute).len) := fun H s =>
    ⟨(real.cost_le (H, s, tix) o).trans (hwork (H, s, tix)),
      (real.calls_le (H, s, tix) o).trans (Schedule.ops_le (r.fits _) _ _ _)⟩
  exact (pr_mono fun x hx => ⟨(real.accepts_iff x o).1 hx.1, hx.2⟩).trans
    (Pouw.SecurityProofs.EndToEnd Q R S CM P D γ₀ Ωmax ε hγ hΩ hWref hTT U hU T hT _ hq (real.adv o) hA t εs hεs)

end ResourceAccounting.SecurityProofs
