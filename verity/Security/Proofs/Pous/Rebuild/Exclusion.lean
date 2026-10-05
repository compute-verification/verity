import Proofs.Pous.Rebuild.Clock
import Proofs.Pous.Rebuild.Window

/-!
# The background-rebuild exclusion

`Pous.Guarantees.BackgroundRebuildExcluded`. The counts are `Window.lean`'s. The bound is the ex post facto bound on any
DAG (`PousSpongeChain.chainExPostFactoG_v6`) at the band with no preprocessor state (`m = 0`) and loss `s = 1`, where
the band is `(1, D, 2^9)`-hard for every `D ≤ 1505` (`band_hard_one`), which is `auditCap`. At `s = 1` the
freshness term is `2 · (nPos²·2^9·2)² / 2^524288` and the collision term `nPos² / 2^524800`, each at most `2^-129`
for `Q ≤ 2^128`.
-/

namespace PousRebuild

open Pous Pous.Labelling Pous.Sponge Pous.BandChain Pous.Rebuild
open scoped ENNReal

/-- The ex post facto bound at `m = 0`, `s = 1`, `B = 2^9`, `ℓ = 524288`, `c = 512` is at most `2^-128` for
`Q ≤ 2^128`. -/
theorem rebuild_term_le (Q : ℕ) (hQ : Q ≤ 2 ^ 128) :
    2 ^ 0 * 2 * ((((nPos (2 ^ 9) Q) ^ 2 * 2 ^ 9 * 2) ^ (1 + 1) : ℕ) : ℝ≥0∞) / 2 ^ (524288 * 1) +
      (((nPos (2 ^ 9) Q) ^ 2 : ℕ) : ℝ≥0∞) / 2 ^ (524288 + 512) ≤ (2⁻¹ : ℝ≥0∞) ^ 128 := by
  have hnp : nPos (2 ^ 9) Q ≤ 2 ^ 138 := by
    unfold nPos
    norm_num
    omega
  have hb1 : (((nPos (2 ^ 9) Q) ^ 2 * 2 ^ 9 * 2) ^ (1 + 1) : ℕ) ≤ 2 ^ 572 :=
    calc (((nPos (2 ^ 9) Q) ^ 2 * 2 ^ 9 * 2) ^ (1 + 1) : ℕ) ≤ (((2 ^ 138) ^ 2 * 2 ^ 9 * 2) ^ (1 + 1) : ℕ) := by
          gcongr
      _ = 2 ^ 572 := by rw [← pow_mul, ← pow_add, ← pow_succ, ← pow_mul]
  have hb2 : ((nPos (2 ^ 9) Q) ^ 2 : ℕ) ≤ 2 ^ 276 :=
    calc ((nPos (2 ^ 9) Q) ^ 2 : ℕ) ≤ (2 ^ 138) ^ 2 := by gcongr
      _ = 2 ^ 276 := by rw [← pow_mul]
  have hc1 : ((((nPos (2 ^ 9) Q) ^ 2 * 2 ^ 9 * 2) ^ (1 + 1) : ℕ) : ℝ≥0∞) ≤ (2 : ℝ≥0∞) ^ 572 := by
    have h : (((2 ^ 572 : ℕ)) : ℝ≥0∞) = (2 : ℝ≥0∞) ^ 572 := by rw [Nat.cast_pow, Nat.cast_ofNat]
    exact h ▸ (by exact_mod_cast hb1)
  have hc2 : (((nPos (2 ^ 9) Q) ^ 2 : ℕ) : ℝ≥0∞) ≤ (2 : ℝ≥0∞) ^ 276 := by
    have h : (((2 ^ 276 : ℕ)) : ℝ≥0∞) = (2 : ℝ≥0∞) ^ 276 := by rw [Nat.cast_pow, Nat.cast_ofNat]
    exact h ▸ (by exact_mod_cast hb2)
  have ht1 : 2 ^ 0 * 2 * ((((nPos (2 ^ 9) Q) ^ 2 * 2 ^ 9 * 2) ^ (1 + 1) : ℕ) : ℝ≥0∞) / 2 ^ (524288 * 1)
      ≤ (2⁻¹ : ℝ≥0∞) ^ 129 := by
    rw [Nat.mul_one, pow_zero, one_mul]
    calc 2 * ((((nPos (2 ^ 9) Q) ^ 2 * 2 ^ 9 * 2) ^ (1 + 1) : ℕ) : ℝ≥0∞) / 2 ^ 524288
        ≤ 2 * (2 : ℝ≥0∞) ^ 572 / 2 ^ 524288 := by gcongr
      _ = (2⁻¹ : ℝ≥0∞) ^ (524288 - 573) := by
          rw [div_eq_mul_inv, ENNReal.inv_pow, ← pow_succ']
          exact PousDenseChain.two_pow_mul_inv_pow _ _ (by norm_num)
      _ ≤ (2⁻¹ : ℝ≥0∞) ^ 129 := pow_le_pow_right_of_le_one' (by norm_num) (by norm_num)
  have ht2 : (((nPos (2 ^ 9) Q) ^ 2 : ℕ) : ℝ≥0∞) / 2 ^ (524288 + 512) ≤ (2⁻¹ : ℝ≥0∞) ^ 129 := by
    calc (((nPos (2 ^ 9) Q) ^ 2 : ℕ) : ℝ≥0∞) / 2 ^ (524288 + 512)
        ≤ (2 : ℝ≥0∞) ^ 276 / 2 ^ (524288 + 512) := by gcongr
      _ = (2⁻¹ : ℝ≥0∞) ^ (524288 + 512 - 276) := by
          rw [div_eq_mul_inv, ENNReal.inv_pow]
          exact PousDenseChain.two_pow_mul_inv_pow _ _ (by norm_num)
      _ ≤ (2⁻¹ : ℝ≥0∞) ^ 129 := pow_le_pow_right_of_le_one' (by norm_num) (by norm_num)
  have hhalf : (2⁻¹ : ℝ≥0∞) ^ 129 + (2⁻¹ : ℝ≥0∞) ^ 129 = (2⁻¹ : ℝ≥0∞) ^ 128 := by
    rw [← two_mul, pow_succ', ← mul_assoc, ENNReal.mul_inv_cancel (by norm_num) (by norm_num), one_mul]
  exact (add_le_add ht1 ht2).trans hhalf.le

/-- **No re-encoding of a band segment within `1505` rounds**: programs that know `W` and the domains but hold no state
about `Π₂`, each within `D ≤ 1505` rounds and `Q ≤ 2^128` queries, output all `2^9` labels with probability at most
`2^-128`. -/
theorem rebuild_pr_le (d : ℕ) (hd : 5 ≤ d) (dom : ℕ → Bits 512)
    (hdom : Function.Injective (fun i : Fin (2 ^ 9) => dom i)) (W : Bits (2 ^ 9 * 524288)) (D Q : ℕ)
    (hD : D ≤ 1505) (hQ : Q ≤ 2 ^ 128)
    (A : Fin (2 ^ 9) → Prog (idealPerm (524288 + (524288 + 512))) (Bits 524288)) (hA : ∀ i, (A i).Bounded D Q) :
    pr (fun π : Equiv.Perm (Bits (524288 + (524288 + 512))) =>
        ∀ i, (A i).run ((idealPerm (524288 + (524288 + 512))).answer π) =
          ((chainBand (2 ^ 9) d 524288 512 dom).enc W π ()).2 i) ≤ (2⁻¹ : ℝ≥0∞) ^ 128 := by
  classical
  have hE := PousSpongeChain.chainExPostFactoG_v6 (2 ^ 9) 524288 512 0 1 D Q (2 ^ 9) (bandDAG (2 ^ 9) d) dom
    (Dense.blocks (2 ^ 9) 524288 W) hdom (band_hard_one d hd 512 524288 D hD)
    (fun _ => fun i => i.elim0) (fun _ i => A i) (fun _ i => hA i)
  refine le_trans (pr_mono fun π hH => ?_) (hE.trans (rebuild_term_le Q hQ))
  refine le_trans (le_of_eq (by rw [Finset.card_univ, Fintype.card_fin]))
    (Finset.card_le_card (s := Finset.univ) fun i _ => Finset.mem_filter.2 ⟨Finset.mem_univ _, hH i⟩)

/-- **`BackgroundRebuildExcluded`, proved.** -/
theorem background_rebuild_excluded : Pous.Guarantees.BackgroundRebuildExcluded := by
  refine ⟨rebuildCalls_512_5, rebuildCalls_512_12, rebuildCalls_ge, gradedRoundSpan_le, ?_⟩
  intro d hd dom hdom W Q hQ T hT A hA
  exact rebuild_pr_le d hd dom hdom W T Q hT hQ A hA

end PousRebuild
