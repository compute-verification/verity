import Definitions.Nci.Guarantees.Capability.Budgets
import Proofs.Nci.Capability.Items
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Probability.ProbabilityMassFunction.Constructions
import Mathlib.Tactic.Linarith

/-!
# Local bounds: proofs of the budgets

* `HeldBudget`: Lemma 1 (`Verity.SecurityProofs.BlockDescription`) on the extraction that reads the held selection in
  one proof block and nothing in the others, at the registered run, bounds the selection's description length by
  `k + ⌈log₂ B⌉ + a_blk`, and description hardness bounds it below by its price minus `Λ`;
* `RecoveredBudget`: the proof block's program followed by the recovery's complement is a description of the source, of
  at most `a_blk + ⌈log₂ B⌉ + k + ℓz − βv J + amb` bits, and source hardness bounds it below by `ℓz − δ`;
* `StatHeldBudget`: in an honest run a proof block's held values are a function of its unregistered input bits
  (`Committed.recompute_congr`), at most `2^k` views, so one view, and the held value it gives, has probability at
  least `2^-k`, which min-entropy hardness bounds by `2^(Λ − price)`.
-/

namespace Nci.SecurityProofs

open Verity Finset Nci.Capability
open scoped ENNReal

theorem HeldBudget : Guarantees.HeldBudget := by
  intro R c k hCK ι X ρ M r t aBlk Λ hev hD u U σ hsel hσ
  classical
  let U' : Fin R.d.n → Finset ι := fun v => if v = u then U else ∅
  have hU' : ∀ v, Selects X (U' v) σ ∧ ∀ i ∈ U' v, σ i ⊆ holds R.d.D (R.d.𝓘.gates v) := by
    intro v
    by_cases hv : v = u
    · subst hv
      simpa [U'] using ⟨hsel, hσ⟩
    · simp [U', hv, Selects]
  obtain ⟨e, he, hE⟩ := hev U' σ hU'
  have hL := Verity.SecurityProofs.BlockDescription R c k hCK ρ M r t aBlk e _ he hE R.val (fun _ _ => rfl) u
  simp only [U', ↓reduceIte] at hL
  have h1 : ENNReal.ofReal (∑ i ∈ U, X.price i - Λ) ≤ ENNReal.ofReal ((k + Nat.clog 2 R.d.n + aBlk : ℕ) : ℝ) := by
    refine (hD R.val (fun _ _ => rfl) U σ hsel).trans ?_
    rw [ENNReal.ofReal_natCast]
    exact_mod_cast ENat.toENNReal_le.2 hL
  rw [ENNReal.ofReal_le_ofReal_iff (Nat.cast_nonneg _)] at h1
  linarith

theorem RecoveredBudget : Guarantees.RecoveredBudget := by
  intro R c k hCK ρ M r τ aBlk e z ℓz δ βv amb J he hS hRec u
  obtain ⟨s, hs, hrun⟩ := hRec R.val (fun _ _ => rfl) u
  have hL : LDL M τ r (z (R.d.D.eval R.val)) ≤ ((blockProgram R e (R.d.D.eval R.val) u ++ s).length : ℕ∞) := by
    unfold LDL
    exact iInf₂_le (blockProgram R e (R.d.D.eval R.val) u ++ s) hrun
  rw [List.length_append, length_blockProgram R hCK.boolean] at hL
  have h1 := (hS R.val (fun _ _ => rfl)).trans (ENat.toENNReal_le.2 hL)
  rw [ENat.toENNReal_coe, ← ENNReal.ofReal_natCast, ENNReal.ofReal_le_ofReal_iff (Nat.cast_nonneg _)] at h1
  push_cast at h1
  have he' : (e.length : ℝ) ≤ aBlk := by exact_mod_cast he
  have hk : ((R.d.unregistered (R.d.𝓘.gates u)).card : ℝ) ≤ k := by exact_mod_cast hCK.reads u
  push_cast
  linarith

/-- In an honest run, what a proof block holds is determined by its unregistered inputs. -/
theorem eval_eq_of_unregistered (R : Committed) {u : Fin R.d.n} {a a' : R.d.D.Transcript} (ha : R.Honest a)
    (ha' : R.Honest a') (h : ∀ g ∈ R.d.unregistered (R.d.𝓘.gates u), R.d.D.eval a g = R.d.D.eval a' g) :
    ∀ g ∈ holds R.d.D (R.d.𝓘.gates u), R.d.D.eval a g = R.d.D.eval a' g := by
  intro g hg
  rw [← R.recompute_eval ha u, ← R.recompute_eval ha' u]
  exact R.recompute_congr h g hg

theorem StatHeldBudget : Guarantees.StatHeldBudget := by
  intro R c k hCK ι X μ Λ hμ hH u U σ hsel hσ
  classical
  set S := R.d.unregistered (R.d.𝓘.gates u)
  let φ : R.d.D.Transcript → (S → Bool) := fun a g => (R.d.D.eval a g.1).getLsbD 0
  let f : R.d.D.Transcript → List Bool := fun a => selectionBits (R.d.D.eval a) U σ
  have hf : ∀ a ∈ μ.support, ∀ a' ∈ μ.support, φ a = φ a' → f a = f a' := by
    intro a ha a' ha' hφ
    have heq := eval_eq_of_unregistered R (hμ a ha) (hμ a' ha') fun x hx => by
      apply BitVec.eq_of_getLsbD_eq
      intro i hi
      have hi0 : i = 0 := by
        have := hCK.boolean.one_bit x
        omega
      subst hi0
      exact congrFun hφ ⟨x, hx⟩
    refine R.d.D.bits_congr fun g hg => heq g ?_
    obtain ⟨i, hi, hgi⟩ := mem_biUnion.1 ((mem_sort _).1 hg)
    exact hσ i hi hgi
  set ν := μ.map φ
  set n := Fintype.card (S → Bool)
  have hn : (n : ℝ≥0∞) = 2 ^ S.card := by simp [n]
  obtain ⟨b, hb⟩ : ∃ b, (n : ℝ≥0∞)⁻¹ ≤ ν b := by
    by_contra hcon
    simp only [not_exists, not_le] at hcon
    have hlt := ENNReal.sum_lt_sum_of_nonempty univ_nonempty fun b (_ : b ∈ univ) => hcon b
    rw [← tsum_fintype (L := SummationFilter.unconditional _), ν.tsum_coe, sum_const, card_univ, nsmul_eq_mul,
      ENNReal.mul_inv_cancel] at hlt
    · exact lt_irrefl _ hlt
    · simp
    · simp
  obtain ⟨a₀, ha₀, hφa₀⟩ := (PMF.mem_support_map_iff φ μ b).1
    ((PMF.mem_support_iff ν b).2 (ne_of_gt (lt_of_lt_of_le (by simp [hn]) hb)))
  have hv : ν b ≤ μ.toOuterMeasure {a | selectionBits (R.d.D.eval a) U σ = f a₀} := by
    rw [← PMF.toOuterMeasure_apply_singleton, PMF.toOuterMeasure_map_apply]
    refine PMF.toOuterMeasure_mono μ fun a ⟨haφ, ha⟩ => ?_
    exact hf a ha a₀ ha₀ (haφ.trans hφa₀.symm)
  have key := (hb.trans hv).trans (hH U σ (f a₀) hsel)
  rw [hn] at key
  have key' := ENNReal.toReal_mono ENNReal.ofReal_ne_top key
  rw [ENNReal.toReal_ofReal (Real.rpow_nonneg zero_le_two _), ENNReal.toReal_inv, ENNReal.toReal_pow,
    ENNReal.toReal_ofNat, ← Real.rpow_natCast, ← Real.rpow_neg zero_le_two,
    Real.rpow_le_rpow_left_iff one_lt_two] at key'
  have hk : (S.card : ℝ) ≤ k := by exact_mod_cast hCK.reads u
  linarith

end Nci.SecurityProofs
