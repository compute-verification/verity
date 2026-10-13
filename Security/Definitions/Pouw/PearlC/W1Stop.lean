import Definitions.Pouw.PearlC.W1

/-!
# The machine stops a run at its budget (trusted; definitions only)

Daniel's item 9: the machine, not the program, stops a run at its budget. `CostModel.withStop CM` gives every program of
`CM` an optional budget `b`. A run whose cost would exceed `b` is stopped: it commits nothing and is charged its whole
budget (at least 0), as a device that preempts at `b` has spent it. A run within `b`, and a program with no budget, is
`CM`'s run at `CM`'s cost. Oracle calls are `CM`'s either way.

`w1Stop dv salt` is the priced W1 (`w1Priced`, compute plus `fetch` per word read) on that machine. Its charge is known as
the run goes, since it reads neither the oracle nor the outcome, so the device can stop at the budget.

`GγPerCoin` is the γ-gap property with the budget per coin: an adversary is bounded by `T` only on the coins where its
cost is at most `T`, and wins there. The γ-gap property over `CM.withStop` is equivalent to it
(`Proofs.Pouw.PearlC.W1Stop.gγ_withStop_iff`).
-/

namespace Pouw.PearlC

variable {Q R S : Type}

/-- **`CM` on a machine that stops a run at its budget**: a program is `CM`'s with an optional budget. A run that would
cost more than its budget `b` commits nothing and is charged `max b 0`. -/
noncomputable def CostModel.withStop (CM : CostModel Q R S) : CostModel Q R S where
  Prog := CM.Prog × Option ℝ
  run := fun p H U act σ s => match p.2 with
    | none => CM.run p.1 H U act σ s
    | some b => if CM.cost p.1 H U act σ s ≤ b then CM.run p.1 H U act σ s else fun _ => none
  cost := fun p H U act σ s => match p.2 with
    | none => CM.cost p.1 H U act σ s
    | some b => if CM.cost p.1 H U act σ s ≤ b then CM.cost p.1 H U act σ s else max b 0
  calls := fun p H U act σ s => CM.calls p.1 H U act σ s

/-- **The γ-gap property with the budget per coin**: for every adversary within `q` oracle calls, the probability that
it costs at most `T` and its correct units' honest work exceeds `T / (1 − γ)` is at most `η(q, N)`. -/
def GγPerCoin [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (D : Layout → Prop) (γ : ℝ) (η : ℕ → ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : Adversary CM,
    (∀ (H : Q → R) (s : S), A.pre.calls H + CM.calls A.online H U A.act (A.pre.eval H) s ≤ q) →
      pr (fun x : (Q → R) × S => CM.cost A.online x.1 U A.act (A.pre.eval x.1) x.2 ≤ T ∧
        Wins P U γ T (A.transcript U x.1 x.2) x.1 x.2) ≤ η q U.layout.N

/-- **The priced W1 on the stopping machine.** -/
noncomputable abbrev W1.w1Stop (dv : W1.W1Device) (salt : S → List ℕ) : CostModel Q R S :=
  (W1.w1Priced dv salt : CostModel Q R S).withStop

end Pouw.PearlC
