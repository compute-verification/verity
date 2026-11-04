module

public import Proofs.Flock.RecGeneric.Plans

/-!
# The plans' expected cost from one run's

`RunFinder.MadeLe` bounds a finder's expected cost in models whose runs, and every fork of one at each first run, cost
at most `c` on average. The second half is not needed: a model's fork offset `cut j` is a number, not a function of the
run, so a fork of a uniform run with a uniform run is uniform (`Claims.Recursion.Ex_fork`), and a plan's forks average
to one run's cost. `RunFinder.CostLe F e` is `MadeLe` with that half dropped: in every model whose runs cost at most `c`
calls on average, `E[F's cost] ≤ e(x)·c`.

Each plan names, in expectation: `forkPlan`, two runs (`u₀` and its fork); `capPlan`, two (`u₀`, and where `u₀` is
accepted and opens the cap position, the loop's forks up to the first that is, which Wald's bound holds to one more on
average, since the gate on `u₀` is the loop's predicate on a run with `u₀`'s prefix); `conflictPlan`, `k + 1` (`u₀`, and
where it is accepted, `k` waits of fresh runs each stopping at its first accepted one); `linkPlan`, two (as `capPlan`,
gated by its first fork).

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims

/-- **`F` names at most `e` runs in expectation, at one run's average cost**: in every model whose runs cost at most `c`
calls on average, `E[F's cost] ≤ e(x)·c`. -/
def RunFinder.CostLe {Coin : Type} [Fintype Coin] [Nonempty Coin] (F : RunFinder Coin) (e : Msg → ℕ) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (n : ℕ) (C : Type) [Fintype C] [Nonempty C] (m : RunModel Coin Ω n C)
    (x : Msg) (c : ℝ), (∀ ω, Ex (fun u : (Fin n → Coin) × C => (m.cost ω u : ℝ)) ≤ c) → F.ExCost m x ≤ e x * c

/-- **`B`'s plan names two runs**: `u₀` and its fork at the round guess. -/
def ForkPlanCost (Coin : Type) [Fintype Coin] [Nonempty Coin] : Prop :=
  ∀ k : Msg → ℕ, (forkPlan (Coin := Coin) k).CostLe fun _ => 2

/-- **`FC`'s plan names two runs in expectation**: `u₀`, and its loop of forks gated by `u₀`, which Wald's bound
counts once. -/
def CapPlanCost (Coin : Type) [Fintype Coin] [Nonempty Coin] : Prop :=
  ∀ (Cp : Claims.Recursion.Caps Coin) (M : ℕ), (capPlan Cp M).CostLe fun _ => 2

/-- **`F0`'s plan names `k + 1` runs in expectation**: `u₀`, and each of its `k` waits gated by it, which stops at its
first accepted run. -/
def ConflictPlanCost (Coin : Type) [Fintype Coin] [Nonempty Coin] : Prop :=
  ∀ (k : Msg → ℕ) (R : ℕ), (conflictPlan (Coin := Coin) k R).CostLe fun x => k x + 1

/-- **The link's plan names two runs in expectation**: its first fork, and its loop of forks gated by that one, which
Wald's bound counts once. -/
def LinkPlanCost (Coin : Type) [Fintype Coin] [Nonempty Coin] : Prop :=
  ∀ (k : Msg → ℕ) (M : ℕ), (linkPlan (Coin := Coin) k M).CostLe fun _ => 2

end Claims.RecGeneric
