module

public import Definitions.Poub.RateAuditPartial
public import Definitions.Assumptions.ResourceAccounting
import Mathlib.Basic.ENNReal.Real

/-! The rate audit's named assumptions, `WorkHardness` and `RateComputeAdditive`. -/

@[expose] public section

namespace ResourceAccounting.Assumptions

open ResourceAccounting ResourceAccounting.Rate

/-- **W, compute additivity, for a joint run's bandwidth game** (the actual-circuit model's obligation W:
compute adds by a direct-sum conjecture). At every
value of the audits' coins, the compute account's work over the compute window and the compute of the run's bandwidth
game (its tags at `c_tag` and its completions at `cost x` for `x` bits), which falls in the compute window, together
fit in the device's priced work over that window:

  `work + game compute ≤ F·T`.

So what the compute audit certifies is not available to complete words, and the adversary's idle compute, `F·T − W`
when the compute account did `W`, is all a window's completions get. In the device model it holds whenever the game's
ops are charged to the accounts other than the compute account (`Step.Fits` charges each op once), which is
`RealizesRate`'s compute field; on the actual circuit, whose gates carry no account, it is the conjecture that PoUW's
work and the completions share no gate. -/
def RateComputeAdditive (p : Price) (cTag : ℕ) (cost : ℕ → ℕ) {N b h ρ s : ℕ} {ι : Type} [Fintype ι] {d : Device ι}
    {α ΩW ΩS : Type} [Fintype α] {r : Run d α ΩW (RateCoins N b h ρ s) ΩS} (real : RealizesRatePartial r) : Prop :=
  ∀ x y s', r.work p (x, y, s') + (real.dev (x, s')).compute cTag cost y.1 y.2.1.1 y.2.1.2 ≤
    d.priced p * (r.win .compute).len

end ResourceAccounting.Assumptions
