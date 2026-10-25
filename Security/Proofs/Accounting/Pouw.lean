import Definitions.Accounting.Guarantees.Pouw
import Proofs.Accounting.Device
import Proofs.Pouw.Game

/-!
# Proof that PoUW's guarantee is a joint run's compute guarantee

At a value `o` of the other audits' coins, a bound `T` on the compute account's work on every compute coin bounds the
adversary `adv o` within `(T, q)` (`q` the device's oracle queries over the compute window): its cost on an oracle and
salt is at most the compute account's work at those and any tickets, and its oracle calls at most the compute account's
oracle queries, which the device's limits bound. PoUW's EndToEnd then bounds its audit, which is the compute audit.
-/

namespace Accounting.SecurityProofs

open Accounting Pouw

theorem ComputeOfPouw : Accounting.Guarantees.ComputeOfPouw := by
  intro Q R S _ _ _ _ CM P D γ₀ Ωmax ε hγ hΩ hWref hTT U hU t εs hεs hne ι α ΩQ ΩS _ _ _ d r p hq real o T hT
    hwork
  obtain ⟨⟨_, _, tix⟩⟩ := hne
  have hA : (real.adv o).Bounded U T (d.rate .oracle * (r.win .compute).len) := fun H s =>
    ⟨(real.cost_le (H, s, tix) o).trans (hwork (H, s, tix)),
      (real.calls_le (H, s, tix) o).trans (Schedule.ops_le (r.fits _) _ _ _)⟩
  exact (pr_mono fun x hx => ⟨(real.accepts_iff x o).1 hx.1, hx.2⟩).trans
    (Pouw.SecurityProofs.EndToEnd Q R S CM P D γ₀ Ωmax ε hγ hΩ hWref hTT U hU T hT _ hq (real.adv o) hA t εs hεs)

end Accounting.SecurityProofs
