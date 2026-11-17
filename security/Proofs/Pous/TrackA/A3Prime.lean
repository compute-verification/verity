module

public import Definitions.Pous.Model.Inc
public import Mathlib.Algebra.BigOperators.Group.Finset.Defs
public import Proofs.Pous.Thm1.Theorem1iiiTimed
import Mathlib.Combinatorics.SimpleGraph.Init
import Mathlib.Data.Sym.Sym2.Init
import Proofs.Pous.Sanity
import Proofs.Pous.Thm1.Theorem1iTimed
import Proofs.Pous.Thm1.Theorem1iiSeq

/-! A3′, the sequential audit from a per-run Lemma A, which enters as the named hypothesis `LemmaA`. -/

@[expose] public section

namespace PousThm1

open Pous
open scoped ENNReal

section progMap

variable {M : IdealModel} {α β : Type} (g : α → β) (f : M.Qry → M.Ans)

end progMap

end PousThm1

namespace PousTrackA

open Pous PousThm1
open scoped ENNReal

def LemmaA {M : IdealModel} (sch : Scheme M) (S D Q g : ℕ) (Γ : ℝ≥0∞)
    (G : Bits sch.n → M.Ω × sch.Coins → Prop) : Prop :=
  ∀ (W : Bits sch.n) (A₁ : Preprocessor sch S) (A₂ : TimedProdResponder sch S D Q),
    pr (fun x : M.Ω × sch.Coins =>
      let pp := (sch.enc W x.1 x.2).1
      let C := (sch.enc W x.1 x.2).2
      G W x ∧ g ≤ (Finset.univ.filter fun i => (A₂.prog W pp (A₁ W pp C x.1) i).run (M.answer x.1) = C i).card)
      ≤ Γ

theorem a3_prime {M : IdealModel} (sch : Scheme M) (S D Q k g : ℕ) (Γ : ℝ≥0∞)
    (G : Bits sch.n → M.Ω × sch.Coins → Prop) (hA : LemmaA sch S D Q g Γ G)
    (W : Bits sch.n) (A₁ : Preprocessor sch S) (A₂ : SeqResponder sch S D Q) :
    pr (seqAccepts sch k W A₁ A₂) ≤
      (((g - 1 : ℕ) : ℝ≥0∞) / sch.B) ^ k + (∑ j ∈ Finset.range k, (sch.B : ℝ≥0∞) ^ j) * Γ +
        pr (fun a => ¬ G W a) := by
  classical
  let st : List (Fin sch.B) → Bits sch.n → Bits sch.t → sch.Code → M.Ω → Bits S :=
    fun h W pp C ω => seqState (M.answer ω) (A₂.step W pp) 0 h (A₁ W pp C ω)
  let pred : List (Fin sch.B) → TimedProdResponder sch S D Q := fun h =>
    ⟨fun W pp σ i => progMap Prod.fst (A₂.step W pp h.length σ i),
      fun W pp σ i => bounded_progMap _ (A₂.bounded _ _ _ _ _)⟩
  let good : List (Fin sch.B) → M.Ω × sch.Coins → Fin sch.B → Prop := fun h a i =>
    ((pred h).prog W (sch.enc W a.1 a.2).1 (st h W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1) i).run
      (M.answer a.1) = (sch.enc W a.1 a.2).2 i
  let E : M.Ω × sch.Coins → Prop := fun a => ¬ G W a ∨
    ∃ j ∈ Finset.range k, ∃ v ∈ (Finset.univ : Finset (Fin j → Fin sch.B)),
      G W a ∧ g ≤ (Finset.univ.filter fun i => good (List.ofFn v) a i).card
  have hre : pr (seqAccepts sch k W A₁ A₂) = pr (fun x : (M.Ω × sch.Coins) × (Fin k → Fin sch.B) =>
      seqAnswers (M.answer x.1.1) (A₂.step W (sch.enc W x.1.1 x.1.2).1) 0 (List.ofFn x.2)
        (A₁ W (sch.enc W x.1.1 x.1.2).1 (sch.enc W x.1.1 x.1.2).2 x.1.1) =
      List.ofFn fun t => (sch.enc W x.1.1 x.1.2).2 (x.2 t)) :=
    pr_equiv (Equiv.prodAssoc _ _ _).symm (fun x : (M.Ω × sch.Coins) × (Fin k → Fin sch.B) =>
      seqAnswers (M.answer x.1.1) (A₂.step W (sch.enc W x.1.1 x.1.2).1) 0 (List.ofFn x.2)
        (A₁ W (sch.enc W x.1.1 x.1.2).1 (sch.enc W x.1.1 x.1.2).2 x.1.1) =
      List.ofFn fun t => (sch.enc W x.1.1 x.1.2).2 (x.2 t))
  rw [hre]
  have : Nonempty (Fin sch.B) := ⟨⟨0, sch.B_pos⟩⟩
  have hB0 : (sch.B : ℝ≥0∞) ≠ 0 := by exact_mod_cast sch.B_pos.ne'
  refine (pr_prod_le (α := M.Ω × sch.Coins) (β := Fin k → Fin sch.B)
    (fun a I => seqAnswers (M.answer a.1) (A₂.step W (sch.enc W a.1 a.2).1) 0 (List.ofFn I)
        (A₁ W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1) =
      List.ofFn fun t => (sch.enc W a.1 a.2).2 (I t))
    E ((((g - 1 : ℕ) : ℝ≥0∞) / sch.B) ^ k) fun a ha => ?_).trans ?_
  ·
    refine pr_seq_le _ _ _ _ k 0 _ fun h hh => ?_
    have hG : G W a := by by_contra hG; exact ha (Or.inl hG)
    have hlt : (Finset.univ.filter fun i => good h a i).card ≤ g - 1 := by
      have : ¬ g ≤ (Finset.univ.filter fun i => good h a i).card := by
        intro hg
        refine ha (Or.inr ⟨h.length, Finset.mem_range.2 hh, h.get, Finset.mem_univ _, hG, ?_⟩)
        rwa [List.ofFn_get]
      omega
    have hpr : pr (good h a) ≤ ((g - 1 : ℕ) : ℝ≥0∞) / sch.B := by
      rw [pr_def, Fintype.card_fin]
      exact ENNReal.div_le_div_right (by exact_mod_cast hlt) _
    simpa only [good, pred, st, run_progMap, Nat.zero_add] using hpr
  ·
    have hE : pr E ≤ pr (fun a => ¬ G W a) + (∑ j ∈ Finset.range k, (sch.B : ℝ≥0∞) ^ j) * Γ := by
      refine (pr_or_le _ _).trans (add_le_add le_rfl ?_)
      calc pr (fun a => ∃ j ∈ Finset.range k, ∃ v ∈ (Finset.univ : Finset (Fin j → Fin sch.B)),
            G W a ∧ g ≤ (Finset.univ.filter fun i => good (List.ofFn v) a i).card)
          ≤ ∑ j ∈ Finset.range k, pr (fun a => ∃ v ∈ (Finset.univ : Finset (Fin j → Fin sch.B)),
            G W a ∧ g ≤ (Finset.univ.filter fun i => good (List.ofFn v) a i).card) := pr_exists_le _ _
        _ ≤ ∑ j ∈ Finset.range k, ∑ _v : Fin j → Fin sch.B, Γ := by
            refine Finset.sum_le_sum fun j _ => (pr_exists_le _ _).trans (Finset.sum_le_sum fun v _ => ?_)
            have := hA W (st (List.ofFn v)) (pred (List.ofFn v))
            convert this using 3
        _ = (∑ j ∈ Finset.range k, (sch.B : ℝ≥0∞) ^ j) * Γ := by
            rw [Finset.sum_mul]
            refine Finset.sum_congr rfl fun j _ => ?_
            rw [Finset.sum_const, Finset.card_univ, Fintype.card_fun, Fintype.card_fin, Fintype.card_fin,
              nsmul_eq_mul, Nat.cast_pow]
    calc pr E + (((g - 1 : ℕ) : ℝ≥0∞) / sch.B) ^ k
        ≤ pr (fun a => ¬ G W a) + (∑ j ∈ Finset.range k, (sch.B : ℝ≥0∞) ^ j) * Γ +
          (((g - 1 : ℕ) : ℝ≥0∞) / sch.B) ^ k := add_le_add hE le_rfl
      _ = _ := by ring

theorem a3_prime_lam {M : IdealModel} (sch : Scheme M) (S D Q k g lam : ℕ) (Γ : ℝ≥0∞)
    (G : Bits sch.n → M.Ω × sch.Coins → Prop) (hA : LemmaA sch S D Q g Γ G)
    (hΓ : (∑ j ∈ Finset.range k, (sch.B : ℝ≥0∞) ^ j) * Γ ≤ (2⁻¹ : ℝ≥0∞) ^ lam)
    (W : Bits sch.n) (A₁ : Preprocessor sch S) (A₂ : SeqResponder sch S D Q) :
    pr (seqAccepts sch k W A₁ A₂) ≤
      (((g - 1 : ℕ) : ℝ≥0∞) / sch.B) ^ k + (2⁻¹ : ℝ≥0∞) ^ lam + pr (fun a => ¬ G W a) :=
  (a3_prime sch S D Q k g Γ G hA W A₁ A₂).trans (by gcongr)

end PousTrackA
