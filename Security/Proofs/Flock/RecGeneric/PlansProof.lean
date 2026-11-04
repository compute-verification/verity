module

public import Proofs.Flock.RecGeneric.AtFlock
public import Proofs.Flock.Claims.Recursion.KSLin2

/-!
# The plans' reads (P1–P3)

How often each of Flock's plans asks about a bad accepted run, against how often one run is bad (`RunFinder.ReadsLe`):

* `forkPlan` asks about no run (`forkPlanReads`, `ForkPlanReads`).
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

variable (Coin : Type) [Fintype Coin] [Nonempty Coin]

/-- **P1: `B`'s plan asks about no run**, so it meets a bad run with probability `0`. -/
theorem forkPlanReads : ForkPlanReads Coin := by
  intro k Ω _ _ n C _ _ m x bad
  unfold RunFinder.PrAsk
  split_ifs
  · calc Pr _ ≤ Pr (fun _ : Fin ((forkPlan (Coin := Coin) k).sel x) × Ω ×
            (Fin ((forkPlan (Coin := Coin) k).slots x + 1) → (Fin n → Coin) × C) => False) :=
          Pr_mono fun s ⟨e, he, _⟩ => by simp [forkPlan, Plan.exec] at he
      _ = 0 := Pr_false
      _ ≤ _ := by simp
  · simp

end Claims.RecGeneric
