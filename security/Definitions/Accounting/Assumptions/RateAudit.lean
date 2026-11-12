module

public import Definitions.Accounting.RateAuditPartial
public import Definitions.Assumptions.Accounting
import Mathlib.Basic.ENNReal.Real

/-!
# The rate audit's named assumptions (trusted)

What the rate audit with partial completion (`Definitions/Accounting/RateAuditPartial.lean`) assumes, each a named
`Prop`, taken as a hypothesis by the guarantees that use it, never an `axiom`:

* `WorkHardness b rounds c_root cost` (declared in `Definitions.Assumptions.Accounting`, which this module re-exports):
  completing `x` of a word's `b` bits costs at least `x/b` of a word's price
  `rounds · c_root` (claim `rate-codec-work`, `WorkHardness(p, k)` in partial-completion form, at `k = rounds`). The
  price is a parameter. The remote audit's route `Accounting.Remote.Run.Rebuild` (`rebuilt · c_word ≤ b · rebuildWork`)
  is what it gives a run (`Accounting.Guarantees.RatePartialRebuild`).
* `RateComputeAdditive`: a joint run's compute account and its bandwidth game share no work (the actual-circuit
  model's compute additivity, W), the remote audit's `Accounting.Remote.Assumptions.ComputeAdditive` for the device
  model's joint run.
-/

@[expose] public section

namespace Accounting.Assumptions

open Accounting Accounting.Rate

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

end Accounting.Assumptions
