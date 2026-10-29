import Mathlib
import Proofs.Accounting.RateAudit.Audit

/-!
# The rate audit over `n` windows: the proof

A sum of footprints short of `n + 1 − m` floors, in windows that were all accepted within the compute budget, leaves at
least `m` of them short of the floor (`le_card_short`). The probability that window `i` is accepted short is at most
`RateAudit`'s error: at each value of the other windows' coins, window `i`'s device is one fixed device of `RateAudit`'s
class, played on the store and window `i`'s own coins (`pr_short_le`). Markov's inequality over the count of short
windows (`mul_pr_le_sum`) gives the guarantee.
-/

namespace Accounting.Rate

open Pous (Bits pr)
open scoped ENNReal

/-- A probability bound that holds at every value of the second coordinate holds for the pair. -/
theorem pr_le_of_fiber {X Y : Type} [Fintype X] [Fintype Y] (P : X × Y → Prop) (ε : ℝ≥0∞)
    (h : ∀ y, pr (fun x => P (x, y)) ≤ ε) : pr P ≤ ε := by
  classical
  unfold pr at h ⊢
  refine ENNReal.div_le_of_le_mul ?_
  have hy : ∀ y, ((Finset.univ.filter (fun x => P (x, y))).card : ℝ≥0∞) ≤ ε * Fintype.card X := by
    intro y
    by_cases hX : Fintype.card X = 0
    · have : (Finset.univ.filter (fun x => P (x, y))).card = 0 :=
        Nat.eq_zero_of_le_zero (hX ▸ Finset.card_le_univ _)
      simp [this]
    · exact (ENNReal.div_le_iff_le_mul (Or.inl (by exact_mod_cast hX)) (Or.inl (ENNReal.natCast_ne_top _))).1
        (by convert h y)
  have e : (Finset.univ.filter P).card = ∑ y, (Finset.univ.filter (fun x => P (x, y))).card := by
    rw [Finset.card_filter, Fintype.sum_prod_type_right]
    simp only [Finset.card_filter]
  rw [e, Nat.cast_sum, Fintype.card_prod, Nat.cast_mul]
  calc ∑ y, ((Finset.univ.filter (fun x => P (x, y))).card : ℝ≥0∞) ≤ ∑ _y : Y, ε * Fintype.card X :=
        Finset.sum_le_sum (fun y _ => hy y)
    _ = ε * (Fintype.card X * Fintype.card Y) := by
        rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
        ring

/-- A probability does not depend on how the type is labelled. -/
theorem pr_equiv {X Y : Type} [Fintype X] [Fintype Y] (e : X ≃ Y) (P : X → Prop) :
    pr P = pr (fun y => P (e.symm y)) := by
  classical
  unfold pr
  rw [card_filter_equiv e P, Fintype.card_congr e]

open Classical in
/-- Markov's inequality for a count of events: `m` times the probability that at least `m` of them hold is at most
the sum of their probabilities. -/
theorem mul_pr_le_sum {Ω ι : Type} [Fintype Ω] [Fintype ι] (P : ι → Ω → Prop) (m : ℕ) :
    (m : ℝ≥0∞) * pr (fun x => m ≤ (Finset.univ.filter (fun i => P i x)).card) ≤ ∑ i, pr (P i) := by
  unfold pr
  rw [← ENNReal.sum_div, ← mul_div_assoc]
  refine ENNReal.div_le_div_right ?_ _
  have key : ∀ T : Finset Ω, (∀ x ∈ T, m ≤ (Finset.univ.filter (fun i => P i x)).card) →
      (m : ℝ≥0∞) * T.card ≤ ∑ i, ((Finset.univ.filter (P i)).card : ℝ≥0∞) := by
    intro T hT
    have hc : m * T.card ≤ ∑ i, (Finset.univ.filter (P i)).card :=
      calc m * T.card ≤ ∑ x ∈ T, (Finset.univ.filter (fun i => P i x)).card := by
            rw [mul_comm, ← smul_eq_mul]
            exact Finset.card_nsmul_le_sum _ _ _ hT
        _ ≤ ∑ x, (Finset.univ.filter (fun i => P i x)).card :=
            Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _) (fun _ _ _ => Nat.zero_le _)
        _ = ∑ i, (Finset.univ.filter (P i)).card := by
            simp only [Finset.card_filter]
            exact Finset.sum_comm
    exact_mod_cast hc
  exact key _ (fun x hx => by simpa using hx)

variable {n N b h ρ S L q s : ℕ}

/-- Window `i` was accepted within the compute budget with its footprint short of the floor `fl`. -/
def Short (A : Windows n N b h ρ S L q s) (cTag cEnc F fl : ℕ) (i : Fin n) (x : Store N b × (Fin n → Coins N b h ρ s)) :
    Prop :=
  (A.dev i x.2).Accepts x.1 (x.2 i).1.1 (x.2 i).1.2 (x.2 i).2 ∧
    (A.dev i x.2).compute cTag cEnc x.1 (x.2 i).1.1 (x.2 i).1.2 ≤ F ∧
    (A.dev i x.2).footprint x.1 (x.2 i).1.1 (x.2 i).1.2 * cEnc < fl

open Classical in
/-- Footprints summing short of `n + 1 − m` floors, in windows all accepted within the budget, leave at least `m`
windows short. -/
theorem le_card_short (A : Windows n N b h ρ S L q s) (cTag cEnc F fl m : ℕ)
    (x : Store N b × (Fin n → Coins N b h ρ s))
    (hall : ∀ i, (A.dev i x.2).Accepts x.1 (x.2 i).1.1 (x.2 i).1.2 (x.2 i).2 ∧
      (A.dev i x.2).compute cTag cEnc x.1 (x.2 i).1.1 (x.2 i).1.2 ≤ F)
    (hsum : ∑ i, (A.dev i x.2).footprint x.1 (x.2 i).1.1 (x.2 i).1.2 * cEnc < (n + 1 - m) * fl) :
    m ≤ (Finset.univ.filter (fun i => Short A cTag cEnc F fl i x)).card := by
  classical
  by_contra hlt
  push Not at hlt
  set T := Finset.univ.filter (fun i => ¬ Short A cTag cEnc F fl i x)
  have hT : (Finset.univ.filter (fun i => Short A cTag cEnc F fl i x)).card + T.card = n := by
    rw [Finset.card_filter_add_card_filter_not, Finset.card_univ, Fintype.card_fin]
  have hle : T.card * fl ≤ ∑ i, (A.dev i x.2).footprint x.1 (x.2 i).1.1 (x.2 i).1.2 * cEnc :=
    calc T.card * fl = ∑ _i ∈ T, fl := by rw [Finset.sum_const, smul_eq_mul]
      _ ≤ ∑ i ∈ T, (A.dev i x.2).footprint x.1 (x.2 i).1.1 (x.2 i).1.2 * cEnc := by
          refine Finset.sum_le_sum (fun i hi => ?_)
          have hi := (Finset.mem_filter.1 hi).2
          simp only [Short, not_and, not_lt] at hi
          exact hi (hall i).1 (hall i).2
      _ ≤ ∑ i, (A.dev i x.2).footprint x.1 (x.2 i).1.1 (x.2 i).1.2 * cEnc :=
          Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _) (fun _ _ _ => Nat.zero_le _)
  have : (n + 1 - m) * fl ≤ T.card * fl := Nat.mul_le_mul_right _ (by omega)
  omega

/-- Window `i` is accepted short with at most `RateAudit`'s error, at every device the other windows' coins pick. -/
theorem pr_short_le (A : Windows n N b h ρ S L q s) (cTag cEnc F k lam : ℕ) (i : Fin n) :
    pr (Short A cTag cEnc F (floor b ρ S L F cTag cEnc k lam) i) ≤
      (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + (q.choose k : ℝ≥0∞) / 2 ^ lam := by
  classical
  let e : Store N b × (Fin n → Coins N b h ρ s) ≃
      (Store N b × Coins N b h ρ s) × ({j : Fin n // j ≠ i} → Coins N b h ρ s) :=
    ((Equiv.refl _).prodCongr (Equiv.funSplitAt i _)).trans (Equiv.prodAssoc _ _ _).symm
  rw [pr_equiv e]
  refine pr_le_of_fiber _ _ (fun ω' => ?_)
  refine (Bandwidth.pr_mono (fun x hx => ?_)).trans
    (SecurityProofs.RateAudit N b h ρ S L q s k F cTag cEnc lam (A i ω'))
  have hω : ((Equiv.funSplitAt i (Coins N b h ρ s)).symm (x.2, ω')) i = x.2 := by simp
  have hd : A.dev i ((Equiv.funSplitAt i (Coins N b h ρ s)).symm (x.2, ω')) = A i ω' := by
    simp only [Windows.dev]
    congr 1
    funext j
    simp [j.2]
  simpa [e, Short, hω, hd] using hx

end Accounting.Rate

namespace Accounting.SecurityProofs

open Accounting.Rate
open Pous (pr)
open scoped ENNReal

/-- **The rate audit over `n` windows** (`Accounting.Guarantees.RateAuditWindows`). -/
theorem RateAuditWindows : Accounting.Guarantees.RateAuditWindows := by
  classical
  intro n N b h ρ S L q s k F cTag cEnc lam m A
  calc (m : ℝ≥0∞) * pr (fun x : Store N b × (Fin n → Coins N b h ρ s) =>
          (∀ i, (A.dev i x.2).Accepts x.1 (x.2 i).1.1 (x.2 i).1.2 (x.2 i).2 ∧
            (A.dev i x.2).compute cTag cEnc x.1 (x.2 i).1.1 (x.2 i).1.2 ≤ F) ∧
          ∑ i, (A.dev i x.2).footprint x.1 (x.2 i).1.1 (x.2 i).1.2 * cEnc <
            (n + 1 - m) * floor b ρ S L F cTag cEnc k lam)
      ≤ (m : ℝ≥0∞) * pr (fun x : Store N b × (Fin n → Coins N b h ρ s) =>
          m ≤ (Finset.univ.filter (fun i => Short A cTag cEnc F (floor b ρ S L F cTag cEnc k lam) i x)).card) := by
        gcongr
        exact Bandwidth.pr_mono (fun x hx => le_card_short A cTag cEnc F _ m x hx.1 hx.2)
    _ ≤ ∑ i, pr (Short A cTag cEnc F (floor b ρ S L F cTag cEnc k lam) i) := mul_pr_le_sum _ m
    _ ≤ ∑ _i : Fin n, ((((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + (q.choose k : ℝ≥0∞) / 2 ^ lam) :=
        Finset.sum_le_sum (fun i _ => pr_short_le A cTag cEnc F k lam i)
    _ = (n : ℝ≥0∞) * ((((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + (q.choose k : ℝ≥0∞) / 2 ^ lam) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

end Accounting.SecurityProofs
