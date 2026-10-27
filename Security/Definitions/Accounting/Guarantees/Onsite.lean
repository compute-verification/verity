import Definitions.Accounting.Assumptions.Onsite
import Definitions.Nci.Assumptions.Threads

/-!
# The onsite end-to-end overhead guarantee (trusted)

The end-to-end overhead proof's onsite spine (note:training-20261027T0134Z-draft-e2e-overhead-proof):

* `OnsiteTimeLowerBound` (step 5): on the onsite class (`Onsite.Class`) under `Nci.Assumptions.WeightsIncompressible`,
  `Accounting.Guarantees.Actual.DampenerWindowsCap`'s `b_a M ≤ n ⌈t/ω⌉₊ K Λ(K)` inverted: `M ≥ N` takes
  `t ≥ t_min = ω (b_a N/(n K Λ(K)) − 1) + 1` ticks (`Onsite.tMin`), the last window partial. In size, the actual
  circuit's gates on the servers over `[0, t)` number at most `Φ_total t` (`Onsite.computeTotal`), so a computation in
  the class that does `N` counted multiplications holds the servers' `Φ_total` gates a tick for at least `t_min` ticks.
* `OnsiteEndToEnd` (steps 3 to 7): at every strategy of the period's causal game, the probability that every subprotocol
  accepts and the run succeeds at the task in fewer than `t_min` ticks, with no two openings of one position, is at most
  `∑ δ + δ_b + η`: the subprotocols' errors (`Accounting.Guarantees.JointPeriodLive`'s hypotheses), the bridge's
  (`Assumptions.OnsiteBridge`) and coverage's (`Assumptions.OnsiteCoverage`).

The proofs are `Accounting.SecurityProofs.OnsiteTimeLowerBound` and `OnsiteEndToEnd`.
-/

namespace Accounting.Guarantees

open Accounting Verity Finset

/-- **The onsite time floor** (step 5). For an actual circuit `A` in the onsite class at the parameters `P`, the horizon
`t` and `M = M_r + M_d` counted multiplications, with the reference circuit's weights incompressible at
`(Γ, K = R + α F ω)`: when `M ≥ N`, the horizon is at least `t_min = ω (b_a N/(n K Λ(K)) − 1) + 1` ticks
(`Onsite.tMin`), and the gates `A` places on the servers during `[0, t)` number at most `Φ_total t`. -/
def OnsiteTimeLowerBound : Prop :=
  ∀ {L E : Type*} [Fintype E] [DecidableEq L] (P : Onsite L E) (A : Placed L) (t M N : ℕ),
    P.Class A t M → Nci.Assumptions.WeightsIncompressible P.c P.T P.Γ P.K P.b_w P.b_a → N ≤ M →
      P.tMin N ≤ t ∧ (A.region (univ.biUnion P.X) (Set.Iio t)).card ≤ P.computeTotal * t

/-- **The onsite end-to-end guarantee** (steps 3 to 7). On any device and any period read off the run of a causal game
`G`, whose five subprotocols have their per-period guarantees over `G` at certificates `c`, price `p` and errors `δ`
(`JointPeriodLive`'s hypotheses), at the onsite parameters `P`, for the world each strategy `s` and run `o` give (the
actual circuit `A s o`, its horizon `t s o`, its counted multiplications `M s o` and whether it succeeds at the task,
`success s o`): when the subprotocols' claims put the actual circuit in the onsite class but with probability `δ_b`
(`OnsiteBridge`), succeeding in the class takes `N` counted multiplications but with probability `η` (`OnsiteCoverage`),
and the reference circuit's weights are incompressible, then at every strategy the probability that every subprotocol
accepts and the run succeeds with a horizon below `t_min` (`Onsite.tMin N`), with no two openings of one position, is at
most `∑ δ + δ_b + η`. -/
def OnsiteEndToEnd : Prop :=
  ∀ (ι α : Type) [Fintype α] (d : Device ι) (C : Verity.Circuit) (Cm O : Type) (r : PeriodLive d α C Cm O)
    (G : Verity.Game O) (c : Certificates) (p : Price) (δ : Audit → ℝ) {L E : Type*} [Fintype E] [DecidableEq L]
    (P : Onsite L E) (A : Verity.Game.Strategy G → O → Placed L) (t M : Verity.Game.Strategy G → O → ℕ)
    (success : Verity.Game.Strategy G → O → Prop) (N : ℕ) (δb η : ℝ),
    (∀ a, r.Sound G c p a (δ a)) →
    Assumptions.OnsiteBridge r G c p P A t M δb →
    Assumptions.OnsiteCoverage G P A t M success N η →
    Nci.Assumptions.WeightsIncompressible P.c P.T P.Γ P.K P.b_w P.b_a →
    ∀ s : Verity.Game.Strategy G,
      Verity.Game.prob (fun o => (∀ a, r.accepts a o) ∧ success s o ∧ (t s o : ℝ) < P.tMin N ∧ ¬ r.Break G s o)
          G s ≤
        ∑ a, ENNReal.ofReal (δ a) + ENNReal.ofReal δb + ENNReal.ofReal η

end Accounting.Guarantees
