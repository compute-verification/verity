module

public import Proofs.Flock.RecGeneric.Plans

/-! The plans' expected cost from one run's average (`RunFinder.CostLe`); definitions, no proofs. -/

@[expose] public section

namespace Claims.RecGeneric

open Claims

def RunFinder.CostLe {Coin : Type} [Fintype Coin] [Nonempty Coin] (F : RunFinder Coin) (e : Msg → ℕ) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (n : ℕ) (C : Type) [Fintype C] [Nonempty C] (m : RunModel Coin Ω n C)
    (x : Msg) (c : ℝ), (∀ ω, Ex (fun u : (Fin n → Coin) × C => (m.cost ω u : ℝ)) ≤ c) → F.ExCost m x ≤ e x * c

def ForkPlanCost (Coin : Type) [Fintype Coin] [Nonempty Coin] : Prop :=
  ∀ k : Msg → ℕ, (forkPlan (Coin := Coin) k).CostLe fun _ => 2

def CapPlanCost (Coin : Type) [Fintype Coin] [Nonempty Coin] : Prop :=
  ∀ (Cp : Claims.Recursion.Caps Coin) (M : ℕ), (capPlan Cp M).CostLe fun _ => 2

def ConflictPlanCost (Coin : Type) [Fintype Coin] [Nonempty Coin] : Prop :=
  ∀ (k : Msg → ℕ) (R : ℕ), (conflictPlan (Coin := Coin) k R).CostLe fun x => k x + 1

def LinkPlanCost (Coin : Type) [Fintype Coin] [Nonempty Coin] : Prop :=
  ∀ (k : Msg → ℕ) (M : ℕ), (linkPlan (Coin := Coin) k M).CostLe fun _ => 2

end Claims.RecGeneric
