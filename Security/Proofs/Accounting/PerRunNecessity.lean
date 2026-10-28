import Proofs.Accounting.PerRunPouw

/-!
# PoUW's per-run guarantee needs the cost model to truncate

`Pouw.Assumptions.TT` and EndToEnd's other hypotheses do not give `Accounting.Guarantees.PouwEndToEndPerRun`'s
conclusion without `Accounting.BudgetAbort`. In this cost model a workload of one 1×1×1 unit has one program, which
commits a correct unit on every salt, costs nothing on salt `true` and the unit's full work on salt `false`. TT holds
with error 0, since every adversary within budget `T` has `T` at least the whole workload's work, but the audit accepts
on every coin, so the probability of acceptance with realized cost below `(1 − γ)(1 − ε_s)·W` is `1/2`, above
`(1 − ε_s)^t + ε = 1/4` at `ε_s = 1/2` and `t = 2` (`tt_not_perRun`). The cost reads the salt, which the program
cannot branch on, so the class has no aborted program and `BudgetAbort` fails (`not_budgetAbort`).
-/

namespace Accounting.PerRunNecessity

open Pouw Finset

/-- One unit of shape `1 × 1 × 1`. -/
def layout : UnitLayout := ⟨1, fun _ => ⟨1, 1, 1⟩, fun _ => 0⟩

/-- The workload: `layout` with zero weights. -/
def U : Workload := ⟨layout, fun _ _ _ => 0⟩

/-- The zero commitment, correct for `U`'s unit under `P`. -/
def commit : Commit := ⟨fun _ _ => 0, fun _ _ _ => 0, fun _ _ => 0⟩

/-- One program, committing `commit` for every unit; free on salt `true`, the workload's plain work on salt `false`. -/
def CM : CostModel Unit Unit Bool where
  Prog := Unit
  run _ _ _ _ _ _ := fun _ => some commit
  cost _ _ U _ _ s := if s then 0 else ∑ u ∈ range U.layout.N, ((U.layout.shape u).Wmm : ℝ)
  calls _ _ _ _ _ _ := 0

/-- A protocol that compares no checked value and charges one per unit. -/
def P : Protocol Unit Unit Bool where
  Y _ _ _ _ _ := fun _ _ _ => 0
  checkIdx _ := ∅
  refChecked _ _ _ _ _ := fun _ _ _ => 0
  refOut _ _ _ _ _ := fun _ _ => 0
  Wref _ _ := 1

/-- Layouts whose units each have positive work, so `W_ref ≤ Ω_max · W_mm` at `Ω_max = 1`. -/
def D (L : UnitLayout) : Prop := ∀ u < L.N, 1 ≤ (L.shape u).Wmm

/-- The adversary with no preprocessing that runs the one program. -/
def A : Adversary CM := ⟨.done [], (), fun _ _ => 0⟩

theorem tt : Assumptions.TT CM P D 0 (fun _ _ => 0) := by
  refine ⟨fun _ _ _ _ _ => ⟨.done fun _ => some commit, fun _ => ⟨rfl, rfl⟩⟩, ?_⟩
  intro U' _ T _ q _ A' hA'
  classical
  refine (pr_eq_zero fun x hx => ?_).le
  have hT : ∑ u ∈ range U'.layout.N, ((U'.layout.shape u).Wmm : ℝ) ≤ T := by
    simpa [CM] using (hA' x.1 false).1
  have hK : ∑ u ∈ checkedSet P U' x.1 x.2 (A'.transcript U' x.1 x.2), ((U'.layout.shape u).Wmm : ℝ) ≤
      ∑ u ∈ range U'.layout.N, ((U'.layout.shape u).Wmm : ℝ) :=
    sum_le_sum_of_subset_of_nonneg (filter_subset _ _) fun _ _ _ => Nat.cast_nonneg _
  simp only [sub_zero, div_one] at hx
  exact absurd (hK.trans hT) (not_le.2 hx)

theorem wref_le (U' : Workload) (hU : U'.InDomain D) (u : ℕ) (hu : u < U'.layout.N) :
    (P.Wref U'.layout u : ℝ) ≤ 1 * ((U'.layout.shape u).Wmm : ℝ) := by
  simpa [P] using hU.2.1 u hu

theorem inDomain : U.InDomain D :=
  ⟨by norm_num [U, layout], fun u hu => by simp [U, layout, Shape.Wmm],
    fun _ _ _ _ _ _ => by norm_num [U, InInt7]⟩

theorem callsWithin : CallsWithin A U 0 := fun _ _ => by simp [A, CM, QTree.calls]

theorem accepts {t : ℕ} (H : Unit → Unit) (s : Bool) (c : Fin t → Ticket (P.Wref U.layout) U.layout.N) :
    AuditAccepts (correctSet P U H s (A.transcript U H s)) c := by
  classical
  intro i
  have hi : ((c i).1 : ℕ) = 0 := Nat.lt_one_iff.1 (c i).1.2
  simp only [hi, correctSet, mem_filter]
  refine ⟨by simp [U, layout], commit, rfl, ⟨fun _ _ _ _ => by norm_num [commit, InInt7], by simp [P]⟩, ?_⟩
  intro _ _ _ _
  simp [commit, mm]

theorem onlineCost_eq (H : Unit → Unit) (s : Bool) : onlineCost A U H s = if s then 0 else 1 := by
  simp [onlineCost, A, CM, U, layout, Shape.Wmm]

theorem pr_mid_true {α γ : Type} [Fintype α] [Fintype γ] [Nonempty α] [Nonempty γ] :
    pr (fun x : α × Bool × γ => x.2.1 = true) = 1 / 2 := by
  rw [pr_eq_card _ (univ ×ˢ ({true} ×ˢ univ)) (by simp)]
  have hα : (0 : ℝ) < Fintype.card α := Nat.cast_pos.2 Fintype.card_pos
  have hγ : (0 : ℝ) < Fintype.card γ := Nat.cast_pos.2 Fintype.card_pos
  simp only [card_product, card_univ, card_singleton, Fintype.card_prod, Fintype.card_bool, Nat.cast_mul,
    Nat.cast_one, Nat.cast_ofNat]
  field_simp

/-- **TT alone does not give PoUW's per-run guarantee.** The cost model `CM`, protocol `P` and domain `D` satisfy
EndToEnd's hypotheses at `γ₀ = 0`, `Ω_max = 1` and error 0, and the adversary `A` on the in-domain workload `U` makes no
oracle call; yet with `ε_s = 1/2` and `t = 2` the audit accepts with realized online cost below
`(1 − γ)(1 − ε_s)·W` with probability `1/2 > (1 − ε_s)^t + ε = 1/4`. -/
theorem tt_not_perRun :
    Assumptions.TT CM P D 0 (fun _ _ => 0) ∧
      (∀ U' : Workload, U'.InDomain D → ∀ u < U'.layout.N,
        (P.Wref U'.layout u : ℝ) ≤ 1 * ((U'.layout.shape u).Wmm : ℝ)) ∧
      U.InDomain D ∧ CallsWithin A U 0 ∧
      (1 - 1 / 2 : ℝ) ^ 2 + 0 <
        pr (fun x : (Unit → Unit) × Bool × (Fin 2 → Ticket (P.Wref U.layout) U.layout.N) =>
          AuditAccepts (correctSet P U x.1 x.2.1 (A.transcript U x.1 x.2.1)) x.2.2 ∧
          onlineCost A U x.1 x.2.1 < (1 - reductionGamma 0 1) * (1 - 1 / 2) * totalWref P U.layout) := by
  refine ⟨tt, wref_le, inDomain, callsWithin, ?_⟩
  have hW : (1 - reductionGamma 0 1) * (1 - 1 / 2) * totalWref P U.layout = 1 / 2 := by
    norm_num [reductionGamma, totalWref, P, U, layout]
  have : Nonempty (Ticket (P.Wref U.layout) U.layout.N) := ⟨⟨⟨0, by simp [U, layout]⟩, ⟨0, by simp [P]⟩⟩⟩
  have hev : ∀ x : (Unit → Unit) × Bool × (Fin 2 → Ticket (P.Wref U.layout) U.layout.N),
      (AuditAccepts (correctSet P U x.1 x.2.1 (A.transcript U x.1 x.2.1)) x.2.2 ∧
        onlineCost A U x.1 x.2.1 < (1 - reductionGamma 0 1) * (1 - 1 / 2) * totalWref P U.layout) ↔
        x.2.1 = true := by
    rintro ⟨H, s, c⟩
    rw [hW, onlineCost_eq]
    cases s <;> norm_num [accepts]
  rw [show (fun x : (Unit → Unit) × Bool × (Fin 2 → Ticket (P.Wref U.layout) U.layout.N) =>
      AuditAccepts (correctSet P U x.1 x.2.1 (A.transcript U x.1 x.2.1)) x.2.2 ∧
        onlineCost A U x.1 x.2.1 < (1 - reductionGamma 0 1) * (1 - 1 / 2) * totalWref P U.layout) =
      fun x => x.2.1 = true from funext fun x => propext (hev x), pr_mid_true]
  norm_num

/-- `CM` does not truncate: by `PouwEndToEndPerRun`, a truncating cost model would bound the probability in
`tt_not_perRun` by `1/4`. -/
theorem not_budgetAbort : ¬ BudgetAbort CM := by
  intro h
  obtain ⟨hTT, hW, hU, hA, hlt⟩ := tt_not_perRun
  exact absurd (SecurityProofs.PouwEndToEndPerRun Unit Unit Bool CM P D 0 1 (fun _ _ => 0) one_pos one_pos hW hTT h
    U hU 0 (by norm_num) A hA 2 (1 / 2) (by norm_num)) (not_le.2 hlt)

end Accounting.PerRunNecessity
