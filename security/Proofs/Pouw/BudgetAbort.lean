module

public import Definitions.Pouw.Game.Defs

/-! Cost models that truncate (`Accounting.BudgetAbort` over the game's `CostModel`); W1 does not truncate
(`Pouw.PearlC.W1BudgetAbort.w1_not_budgetAbort`). -/

@[expose] public section

namespace Pouw.BudgetAbort

variable {Q R S : Type}

def BudgetAbortG (CM : CostModel Q R S) : Prop :=
  (∀ (prog : CM.Prog) (H : Q → R) (U : Workload) (act : ℕ → Mat) (σ : State) (s : S),
      0 ≤ CM.cost prog H U act σ s) ∧
    ∀ (prog : CM.Prog) (U : Workload) (act : ℕ → Mat) (b : ℝ), 0 ≤ b → ∃ prog' : CM.Prog,
      ∀ (H : Q → R) (σ : State) (s : S),
        CM.cost prog' H U act σ s ≤ b ∧ CM.calls prog' H U act σ s ≤ CM.calls prog H U act σ s ∧
          (CM.cost prog H U act σ s ≤ b → CM.run prog' H U act σ s = CM.run prog H U act σ s ∧
            CM.hashed prog' H U act σ s = CM.hashed prog H U act σ s)

theorem budgetAbortG_of_constCost (CM : CostModel Q R S)
    (hnn : ∀ (prog : CM.Prog) (H : Q → R) (U : Workload) (act : ℕ → Mat) (σ : State) (s : S),
      0 ≤ CM.cost prog H U act σ s)
    (hconst : ∀ (prog : CM.Prog) (U : Workload) (act : ℕ → Mat) (H H' : Q → R) (σ σ' : State) (s s' : S),
      CM.cost prog H U act σ s = CM.cost prog H' U act σ' s')
    (halt : CM.Prog)
    (hhalt : ∀ (H : Q → R) (U : Workload) (act : ℕ → Mat) (σ : State) (s : S),
      CM.cost halt H U act σ s ≤ 0 ∧ CM.calls halt H U act σ s = 0) :
    BudgetAbortG CM := by
  refine ⟨hnn, fun prog U act b hb => ?_⟩
  by_cases h : ∃ (H : Q → R) (σ : State) (s : S), CM.cost prog H U act σ s ≤ b
  · obtain ⟨H₀, σ₀, s₀, h₀⟩ := h
    exact ⟨prog, fun H σ s => ⟨(hconst prog U act H H₀ σ σ₀ s s₀).trans_le h₀, le_rfl, fun _ => ⟨rfl, rfl⟩⟩⟩
  · refine ⟨halt, fun H σ s => ⟨(hhalt H U act σ s).1.trans hb, ?_, fun hc => (h ⟨H, σ, s, hc⟩).elim⟩⟩
    rw [(hhalt H U act σ s).2]
    exact Nat.zero_le _

/-- A program is a price and what it commits, or nothing, which commits blanks at no cost. -/
def fixedPrice (Q R S : Type) : CostModel Q R S where
  Prog := Option (ℕ × (Workload → (ℕ → Mat) → State → S → Trace))
  run := fun p _ U act σ s => match p with
    | some c => c.2 U act σ s
    | none => fun _ => none
  cost := fun p _ _ _ _ _ => match p with
    | some c => (c.1 : ℝ)
    | none => 0
  calls := fun _ _ _ _ _ _ => 0

theorem fixedPrice_budgetAbort : BudgetAbortG (fixedPrice Q R S) := by
  refine budgetAbortG_of_constCost _ ?_ ?_ none ?_
  · intro p _ _ _ _ _
    cases p with
    | none => exact le_rfl
    | some c => exact Nat.cast_nonneg _
  · intro p _ _ _ _ _ _ _ _
    cases p <;> rfl
  · intro _ _ _ _ _
    exact ⟨le_rfl, rfl⟩

end Pouw.BudgetAbort
