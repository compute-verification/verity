module

public import Definitions.Pouw.PearlC.W1Stop
public import Proofs.Pouw.PearlC.W1BudgetAbort
import Mathlib.Tactic.Positivity.Finset
import Proofs.Pouw.PearlC.W1Priced
import Proofs.Pouw.Prob

/-!
# The stopping machine truncates

`Accounting.BudgetAbort` fails for W1 under any refund rule (the static-refund three-load example),
and for the outcome refund by `W1BudgetAbort.w1_not_budgetAbort`. On a
machine that stops a run at its budget (`CostModel.withStop`, by ruling) it holds:

- `withStop_budgetAbort`: any model with nonnegative costs truncates on the stopping machine. The copy of a program at
  budget `b` is the same program with budget `b` (or the smaller of `b` and its own).
- `w1Stop_budgetAbort`: so does the priced W1 at positive prices.
- `gGap_withStop_iff`: the gap property over the stopping machine is the gap property with the budget per coin
  (`GGapPerCoin`), for `g ≤ 1`. A stopped run commits nothing, so it never wins.
- `gGap_of_withStop`: it implies the property over the model itself, since a program with no budget runs as before.

So the stop does not make the per-run bound free. `GGapPerCoin` is stronger than `GGap`, whose budget holds on every coin, and
it is the conjecture a per-run bound over W1 rests on.
-/

@[expose] public section

namespace Pouw.PearlC.W1Stop

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.W1 Pouw.PearlC.W1BudgetAbort

variable {Q R S : Type}

/-- **A model with nonnegative costs truncates on the stopping machine.** -/
theorem withStop_budgetAbort (CM : CostModel Q R S)
    (hnn : ∀ (p : CM.Prog) (H : Q → R) (U : Workload) (act : ℕ → Codes) (σ : State) (s : S),
      0 ≤ CM.cost p H U act σ s) :
    BudgetAbortPearlC CM.withStop := by
  refine ⟨?_, ?_⟩
  · rintro ⟨p, _ | b⟩ H U act σ s
    · exact hnn p H U act σ s
    · simp only [CostModel.withStop]
      split_ifs
      · exact hnn p H U act σ s
      · exact le_max_right _ _
  · rintro ⟨p, _ | b₀⟩ U act b hb
    · refine ⟨(p, some b), fun H σ s => ⟨?_, le_rfl, fun hc => ?_⟩⟩
      · simp only [CostModel.withStop]
        split_ifs with h
        · exact h
        · exact max_le le_rfl hb
      · simp only [CostModel.withStop] at hc ⊢
        rw [if_pos hc]
    · refine ⟨(p, some (min b b₀)), fun H σ s => ⟨?_, le_rfl, fun hc => ?_⟩⟩
      · simp only [CostModel.withStop]
        split_ifs with h
        · exact h.trans (min_le_left _ _)
        · exact max_le (min_le_left _ _) hb
      · simp only [CostModel.withStop] at hc ⊢
        by_cases h : CM.cost p H U act σ s ≤ b₀
        · rw [if_pos h] at hc
          rw [if_pos (le_min hc h), if_pos h]
        · rw [if_neg h, if_neg fun h' => h (h'.trans (min_le_right _ _))]

/-- The priced W1 charges nothing negative at positive prices. -/
theorem w1Priced_nonneg (dv : W1Device) (hdv : Positive dv) (salt : S → List ℕ) (p : CodeImage) (H : Q → R)
    (U : Workload) (act : ℕ → Codes) (σ : State) (s : S) : 0 ≤ (w1Priced dv salt).cost p H U act σ s := by
  have h0 : (0 : ℚ) ≤ (final dv (salt s) p U act σ).cost := (exec_mono dv hdv _ false _).1
  simp only [w1Priced]
  exact_mod_cast add_nonneg h0 (mul_nonneg hdv.2.2.2.2.1.le (Nat.cast_nonneg _))

/-- **The priced W1 truncates on the stopping machine**, at positive prices. -/
theorem w1Stop_budgetAbort (dv : W1Device) (hdv : Positive dv) (salt : S → List ℕ) :
    BudgetAbortPearlC (w1Stop dv salt : CostModel Q R S) :=
  withStop_budgetAbort _ (w1Priced_nonneg dv hdv salt)

/-- A run that commits nothing never wins, for `g ≤ 1`. -/
theorem not_wins_blank (P : Protocol Q R S) (U : Workload) {g T : ℝ} (hg : g ≤ 1) (hT : 0 ≤ T) (H : Q → R) (s : S) :
    ¬ Wins P U g T (fun _ => none) H s := by
  intro h
  have hK : correctSet P U H s (fun _ => none) = ∅ := by
    ext u
    simp [correctSet]
  rw [Wins, hK, Finset.sum_empty] at h
  have : 0 ≤ T / (1 - g) := div_nonneg hT (by linarith)
  linarith

/-- **The gap property over the stopping machine is the gap property with the budget per coin.** -/
theorem gGap_withStop_iff [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (P : Protocol Q R S) (D : Layout → Prop) {g : ℝ} (hg : g ≤ 1) (δ : ℕ → ℕ → ℝ) :
    GGap CM.withStop P D g δ ↔ GGapPerCoin CM P D g δ := by
  constructor
  · intro h U hU T hT q hq A hA
    let A' : Adversary CM.withStop := ⟨A.pre, (A.online, some T), A.act⟩
    have hb : A'.Bounded U T q := fun H s => by
      refine ⟨?_, hA H s⟩
      show (if CM.cost A.online H U A.act (A.pre.eval H) s ≤ T then CM.cost A.online H U A.act (A.pre.eval H) s
        else max T 0) ≤ T
      split_ifs with hc
      · exact hc
      · exact max_le le_rfl hT
    refine le_trans (pr_mono fun x hx => ?_) (h U hU T hT q hq A' hb)
    obtain ⟨hc, hw⟩ := hx
    show Wins P U g T (if CM.cost A.online x.1 U A.act (A.pre.eval x.1) x.2 ≤ T then
      CM.run A.online x.1 U A.act (A.pre.eval x.1) x.2 else fun _ => none) x.1 x.2
    rw [if_pos hc]
    exact hw
  · intro h U hU T hT q hq A hA
    obtain ⟨pre, ⟨p, o⟩, act⟩ := A
    let A₀ : Adversary CM := ⟨pre, p, act⟩
    have hq₀ : ∀ (H : Q → R) (s : S), A₀.pre.calls H + CM.calls A₀.online H U A₀.act (A₀.pre.eval H) s ≤ q :=
      fun H s => (hA H s).2
    refine le_trans (pr_mono fun x hx => ?_) (h U hU T hT q hq A₀ hq₀)
    have hcost := (hA x.1 x.2).1
    cases o with
    | none => exact ⟨hcost, hx⟩
    | some b =>
      change (if CM.cost p x.1 U act (pre.eval x.1) x.2 ≤ b then CM.cost p x.1 U act (pre.eval x.1) x.2
        else max b 0) ≤ T at hcost
      change Wins P U g T (if CM.cost p x.1 U act (pre.eval x.1) x.2 ≤ b then
        CM.run p x.1 U act (pre.eval x.1) x.2 else fun _ => none) x.1 x.2 at hx
      by_cases hc : CM.cost p x.1 U act (pre.eval x.1) x.2 ≤ b
      · rw [if_pos hc] at hcost hx
        exact ⟨hcost, hx⟩
      · rw [if_neg hc] at hx
        exact (not_wins_blank P U hg hT x.1 x.2 hx).elim

/-- **The stopping machine's gap-gap property gives the model's**: a program with no budget runs as before. -/
theorem gGap_of_withStop [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (P : Protocol Q R S) (D : Layout → Prop) (g : ℝ) (δ : ℕ → ℕ → ℝ) (h : GGap CM.withStop P D g δ) :
    GGap CM P D g δ :=
  W1Priced.gGap_of_cost_le CM.withStop CM (fun p => (p, none)) (fun _ _ _ _ _ _ => rfl) (fun _ _ _ _ _ _ => le_rfl)
    (fun _ _ _ _ _ _ => le_rfl) P D g δ h

end Pouw.PearlC.W1Stop
