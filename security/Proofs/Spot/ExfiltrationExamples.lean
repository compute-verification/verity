module

public import Definitions.Spot.Dataflow
public import Definitions.Spot.Exfiltration
meta import Lean.Meta.Tactic.NormCast
import Mathlib.Algebra.Order.BigOperators.Expect
import Mathlib.Algebra.Order.Field.Power
import Mathlib.Analysis.Complex.Order
import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Basic.ENNReal.Real
import Mathlib.Data.EReal.Inv
import Mathlib.Data.Fintype.BigOperators
import Proofs.Spot.Determinacy
import Proofs.Spot.Exfiltration

/-! Exfiltration before detection: tightness, and what fails without each assumption. -/

@[expose] public section

namespace Population.Exfil

open Finset

variable {Ω : Type} [Fintype Ω] [DecidableEq Ω] {μ : PMF Ω}

def constStrategy (ℓ : ℝ) (s : Finset Ω) : Strategy Ω := ⟨fun _ => ℓ, fun _ => s⟩

omit [Fintype Ω] [DecidableEq Ω] in
@[simp] theorem const_leak (ℓ : ℝ) (s : Finset Ω) (h : List Ω) : (constStrategy ℓ s).leak h = ℓ := rfl

omit [Fintype Ω] [DecidableEq Ω] in
@[simp] theorem const_esc (ℓ : ℝ) (s : Finset Ω) (h : List Ω) : (constStrategy ℓ s).esc h = s := rfl

omit [Fintype Ω] [DecidableEq Ω] in
theorem const_leakBounded {ℓ : ℝ} (hℓ : 0 ≤ ℓ) (s : Finset Ω) : LeakBounded ℓ (constStrategy ℓ s) :=
  fun _ => ⟨hℓ, le_rfl⟩

omit [DecidableEq Ω] in
theorem prob_true (n : ℕ) : prob μ n (fun _ => True) = 1 :=
  Verity.Game.prPMFReal_true _

theorem tight_cum {ℓ : ℝ} (hℓ : 0 < ℓ) (s : Finset Ω) :
    ∀ (n j : ℕ) (h : List Ω), j < n →
      prob μ n (fun cs => ((j : ℝ) + 1) * ℓ ≤ (constStrategy ℓ s).cum n h cs) = Verity.Game.prPMFReal μ (· ∈ s) ^ j
  | 0, _, _, hj => absurd hj (Nat.not_lt_zero _)
  | n + 1, 0, h, _ => by
    rw [pow_zero]
    refine (prob_congr (E' := fun _ => True) fun cs => ?_).trans (prob_true (n + 1))
    simp only [Nat.cast_zero, zero_add, one_mul, iff_true]
    rw [← Fin.cons_self_tail cs, cum_cons, const_leak, const_esc]
    have := cum_nonneg (const_leakBounded hℓ.le s) n (cs 0 :: h) (Fin.tail cs)
    split_ifs <;> linarith
  | n + 1, j + 1, h, hj => by
    have key : ∀ c : Ω, prob μ n (fun cs => (((j + 1 : ℕ) : ℝ) + 1) * ℓ ≤
        (constStrategy ℓ s).cum (n + 1) h (Fin.cons c cs)) = if c ∈ s then Verity.Game.prPMFReal μ (· ∈ s) ^ j else 0 := by
      intro c
      by_cases hc : c ∈ s
      · rw [ite_eq_left hc, ← tight_cum hℓ s n j (c :: h) (by omega)]
        refine prob_congr fun cs => ?_
        rw [cum_cons, const_leak, const_esc, ite_eq_left hc]
        push_cast
        constructor <;> intro <;> linarith
      · rw [ite_eq_right hc, ← prob_false μ n]
        refine prob_congr fun cs => ?_
        rw [cum_cons, const_leak, const_esc, ite_eq_right hc]
        push_cast
        simp only [add_zero, iff_false, not_le]
        nlinarith
    rw [prob_succ]
    simp only [key]
    rw [sum_ite_esc, pow_succ']

theorem tight_tail {κ ℓ : ℝ} (hℓ : 0 < ℓ) {s : Finset Ω} (hs : Verity.Game.prPMFReal μ (· ∈ s) = Real.exp (-(κ * ℓ)))
    {n j : ℕ} (h : List Ω) (hj : j < n) :
    prob μ n (fun cs => ((j : ℝ) + 1) * ℓ ≤ (constStrategy ℓ s).cum n h cs) =
      Real.exp (-(κ * (((j : ℝ) + 1) * ℓ - ℓ))) := by
  rw [tight_cum hℓ s n j h hj, hs, ← Real.exp_nat_mul]
  ring_nf

theorem tight_exists {κ ℓ : ℝ} (hκ : 0 ≤ κ) (hℓ : 0 < ℓ) :
    ∃ (μ : PMF Bool) (A : Strategy Bool), DetectionPressure μ κ A ∧ LeakBounded ℓ A ∧
      ∀ (n j : ℕ), j < n → prob μ n (fun cs => ((j : ℝ) + 1) * ℓ ≤ A.cum n [] cs) =
        Real.exp (-(κ * (((j : ℝ) + 1) * ℓ - ℓ))) := by
  have h0 : 0 ≤ Real.exp (-(κ * ℓ)) := (Real.exp_pos _).le
  have h1 : Real.exp (-(κ * ℓ)) ≤ 1 := Real.exp_le_one_iff.2 (by nlinarith)
  let μ : PMF Bool :=
    PMF.ofFintype (fun b => ENNReal.ofReal (if b then Real.exp (-(κ * ℓ)) else 1 - Real.exp (-(κ * ℓ)))) (by
      rw [Fintype.sum_bool]
      exact (ENNReal.ofReal_add h0 (sub_nonneg.2 h1)).symm.trans (by rw [add_sub_cancel, ENNReal.ofReal_one]))
  have hs : Verity.Game.prPMFReal μ (· ∈ ({true} : Finset Bool)) = Real.exp (-(κ * ℓ)) := by
    rw [Verity.Game.prPMFReal_mem, sum_singleton]
    exact ENNReal.toReal_ofReal h0
  refine ⟨μ, constStrategy ℓ {true}, fun _ => hs.le, const_leakBounded hℓ.le _, fun n j hj => ?_⟩
  exact tight_tail hℓ hs [] hj

theorem no_pressure_cum (ℓ : ℝ) : ∀ (n : ℕ) (h : List Ω) (cs : Fin n → Ω),
    (constStrategy ℓ univ).cum n h cs = n * ℓ
  | 0, _, _ => by simp [Strategy.cum]
  | n + 1, h, cs => by
    rw [← Fin.cons_self_tail cs, cum_cons, const_leak, const_esc, ite_eq_left (mem_univ _), no_pressure_cum ℓ n]
    push_cast
    ring

omit [DecidableEq Ω] in
theorem not_pressure_univ {κ ℓ : ℝ} (hκ : 0 < κ) (hℓ : 0 < ℓ) :
    ¬ DetectionPressure μ κ (constStrategy ℓ univ) := by
  intro hP
  have h := hP []
  simp only [constStrategy, mem_univ, Verity.Game.prPMFReal_true] at h
  have : Real.exp (-(κ * ℓ)) < 1 := Real.exp_lt_one_iff.2 (by nlinarith)
  linarith

omit [Fintype Ω] in
theorem no_halt_cumUpTo (ℓ : ℝ) : ∀ (n : ℕ) (h : List Ω) (cs : Fin n → Ω),
    (constStrategy ℓ ∅).cumUpTo n n h cs = n * ℓ
  | 0, _, _ => by simp [Strategy.cumUpTo]
  | n + 1, h, cs => by
    rw [← Fin.cons_self_tail cs, cumUpTo_cons, const_leak, const_esc, ite_eq_right (notMem_empty _),
      no_halt_cumUpTo ℓ n]
    push_cast
    ring

omit [Fintype Ω] [DecidableEq Ω] in
theorem pressure_empty (κ ℓ : ℝ) : DetectionPressure μ κ (constStrategy ℓ (∅ : Finset Ω)) :=
  fun _ => by simpa [constStrategy, Verity.Game.prPMFReal_false] using (Real.exp_pos _).le

omit [Fintype Ω] in
theorem halt_cum (ℓ : ℝ) (n : ℕ) (h : List Ω) (cs : Fin (n + 1) → Ω) :
    (constStrategy ℓ (∅ : Finset Ω)).cum (n + 1) h cs = ℓ := by
  rw [← Fin.cons_self_tail cs, cum_cons]
  simp [constStrategy]

theorem not_trustedModel_of_reveal {W H T : Type} [Fintype W] {ρ : PMF W} {S : W → T}
    {hon : W → H} (dec : H → T) (hdec : ∀ w, dec (hon w) = S w) {g : ℝ} (hg : g < 1) :
    ¬ TrustedModel ρ S hon g := by
  intro hT
  have := hT dec
  simp only [hdec, Verity.Game.prPMFReal_true] at this
  linarith

end Population.Exfil

namespace Population

open Finset

def seedCopy (k : ℕ) : Dataflow (Fin k) (Fin k ⊕ Fin k) where
  input := univ.image Sum.inl
  reads i := {Sum.inl i}
  writes i := {Sum.inr i}

def seedFns (k : ℕ) : (seedCopy k).Fns Bool := fun i r _ => r ⟨Sum.inl i, mem_singleton_self _⟩

theorem seedCopy_checked (k : ℕ) : (seedCopy k).Checked := by
  refine ⟨fun s => ⟨fun hs u hu => ?_, fun hs => ?_⟩, fun u hu => ?_⟩
  · simp only [seedCopy, mem_image, mem_univ, true_and, mem_singleton] at hs hu
    obtain ⟨i, rfl⟩ := hs
    cases hu
  · cases s with
    | inl i => exact absurd (by simp [seedCopy]) hs
    | inr i => exact ⟨i, by simp [seedCopy], fun j hj => by simpa [seedCopy, eq_comm] using hj⟩
  · obtain ⟨_, ⟨s, hsw, hsr⟩, -⟩ := Relation.TransGen.head'_iff.1 hu
    simp only [seedCopy, mem_singleton] at hsw hsr
    rw [hsw] at hsr
    cases hsr

theorem seed_egress (k : ℕ) :
    2 ^ k ≤ ((seedCopy k).egress (seedFns k) (fun _ => false) (seedCopy k).input (univ.image Sum.inr) ∅).card := by
  classical
  have hmem : ∀ w : Fin k → Bool, (fun o : ↥(univ.image Sum.inr : Finset (Fin k ⊕ Fin k)) => Sum.elim w w o.1) ∈
      (seedCopy k).egress (seedFns k) (fun _ => false) (seedCopy k).input (univ.image Sum.inr) ∅ := by
    intro w
    simp only [Dataflow.egress, mem_image, mem_filter, mem_univ, true_and, subset_empty]
    refine ⟨Sum.elim w w, ⟨fun s hs hsX => absurd hs hsX, ?_⟩, rfl⟩
    refine (seedCopy k).incorrect_eq_empty fun i s => ?_
    obtain ⟨s, hs⟩ := s
    simp only [seedCopy, mem_singleton] at hs
    subst hs
    rfl
  have hinj : Function.Injective fun (w : Fin k → Bool)
      (o : ↥(univ.image Sum.inr : Finset (Fin k ⊕ Fin k))) => Sum.elim w w o.1 := by
    intro w w' h
    funext i
    exact congrFun h ⟨Sum.inr i, mem_image_of_mem _ (mem_univ i)⟩
  have := card_le_card_of_injOn _ (s := univ) (fun w _ => hmem w) hinj.injOn
  simpa using this

end Population
