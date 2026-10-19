import Proofs.Accounting.Headline.RtLogic3Fork
import Models.Accounting.Assumptions.TS

/-!
RT3-LOGIC-1 at #2791's premises (red team, not for landing; builds on #2791 @ 8ce15bb6f5, not on main): the honest
coin layer meets no `BindingForkAt` below `½`, so over the sm_120 headlines' 48,000 rounds their binding premise and
budget `48,000·(√δ_fork + n_π·δ_ZK) ≤ 2^-41` can't hold together for it. A faithful proof layer of a real window,
whose round-`w` proofs read round `w`'s keys and draw, is in this case; one that meets the premises opens at round `w`
only what was fixed before σ_w.
-/

namespace Accounting.SecurityProofs.TS.RtLogic3

open Pouw.RelativeSecurity Accounting.TS Accounting.Assumptions.TS

theorem coinLayer_not_bindingForkAt (n : ℕ) (δ : ℝ) (hδ : δ < 1 / 2) : ¬ BindingForkAt (coinLayer n) δ :=
  coinLayer_not_fork_le n δ hδ

/-- **The sm_120 headlines' binding premise and budget fail together for the honest coin layer** over 48,000 rounds. -/
theorem coinLayer_budget_unsat (δfork δzk : ℝ) (hzk : 0 ≤ δzk) (hB : BindingForkAt (coinLayer 47999) δfork) :
    ¬ (48000 : ℝ) * (Real.sqrt δfork + ((1 : ℕ) : ℝ) * δzk) ≤ (1 / 2) ^ 41 := by
  intro hb
  have hδ : 1 / 2 ≤ δfork := by
    by_contra h
    exact coinLayer_not_bindingForkAt 47999 δfork (not_le.mp h) hB
  have hs := Real.sq_sqrt (show (0 : ℝ) ≤ δfork by linarith)
  have hs0 := Real.sqrt_nonneg δfork
  have hc : Real.sqrt δfork ≤ 1 / 1000 := by
    norm_num at hb
    linarith
  nlinarith [mul_le_mul_of_nonneg_left hc hs0]

end Accounting.SecurityProofs.TS.RtLogic3
