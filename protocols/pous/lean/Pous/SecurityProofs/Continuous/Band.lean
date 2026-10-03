import Pous.SecurityProofs.Chain.Family
import Pous.SecurityProofs.Continuous.PerAnswer

/-!
# The continuous audit on the 64 KiB band at `D = 5` (DRAFT: an instance of the unreviewed game)

`ContinuousAudit` (`Continuous/Game.lean`, a draft) at the scheme of `Pous.Guarantees.BandChainMeets64`: 512 blocks of
64 KiB labels, in-degree `d ≥ 5`, any 256-bit salt, `D = 5`, `Q = 2^20`. The compression certificate is the one
`BandChainMeets64` uses, `TimedINC` at `β = 265253616` from the unconditional `chainExPostFactoG_v6`, and its `π` is at
most `2·2^-1128` (`chainPiSeg_fam_le`, as `seg_union_term_one` uses it). So `ε_proof = 1 − (β − B)/|C|`
`= 3182352/2^28 ≈ 1.1855%` (`band64_epsProof`), and the stopped union costs `n·(β + 1)·π ≤ n·2^-1098`.
-/

namespace PousContinuous

open Pous Pous.BandChain Pous.BandMulti PousSpongeChain
open scoped ENNReal

/-- The 64 KiB band of `Pous.Guarantees.BandChainMeets64` at in-degree `d` and salt `salt`. -/
abbrev band64 (d : ℕ) (salt : Bits 256) := chainBand (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)

/-- The band's certificate at `D = 5`, `β = 265253616`: `π ≤ 2·2^-1128`. -/
theorem band64_pi_le :
    chainPi (2 ^ 9) 524288 512 5 (2 ^ 20) 265253616 ≤ (2⁻¹ : ℝ≥0∞) ^ 1128 + (2⁻¹ : ℝ≥0∞) ^ 1128 := by
  rw [← chainPiSeg_one]
  exact chainPiSeg_fam_le 1 _ 69 59 1128 le_rfl (by norm_num [Pous.Sponge.nPos]) (by norm_num [Pous.Sponge.nPos])
    (by norm_num) (by norm_num)

/-- **Draft.** The band's `ε_proof` at `D = 5`: `1 − (265253616 − 512)/2^28 = 3182352/2^28 ≈ 1.1855%`. -/
theorem band64_epsProof (d : ℕ) (salt : Bits 256) : epsProof (band64 d salt) 265253616 = 3182352 / 2 ^ 28 := by
  show (1 : ℝ) - (((265253616 : ℕ) : ℝ) - ((2 ^ 9 : ℕ) : ℝ)) / (((2 ^ 9 : ℕ) : ℝ) * ((524288 : ℕ) : ℝ)) = _
  norm_num

open Classical in
/-- **Draft: the continuous audit on the 64 KiB band at `D = 5`.** For every `f` and `βe > 0`, over `n` challenges:
the misses are at most `f` and `Σ_j max(ε_j − ε_proof, 0)` exceeds `G(f)` with probability at most `βe + n·2^-1098`,
where `ε_proof = epsProof (band64 d salt) 265253616 = 3182352/2^28` (`band64_epsProof`). -/
theorem band64_continuous (d : ℕ) (hd : 5 ≤ d) (salt : Bits 256) (n : ℕ) (W : Bits (band64 d salt).n)
    (A₁ : VarPreprocessor (band64 d salt)) (R : VarResponder (band64 d salt) 5 (2 ^ 20)) (f βe : ℝ) (hβe : 0 < βe) :
    pr (fun x : Outcome (band64 d salt) n =>
        ((Finset.univ.filter fun j : Fin n =>
            ¬ hit (band64 d salt) W A₁ R (x.1, x.2.1) ((List.ofFn x.2.2).take j) (x.2.2 j)).card : ℝ) ≤ f ∧
          (⨅ θ : Set.Ioi (0 : ℝ), ((θ : ℝ) * f + Real.log (1 / βe)) / (1 - Real.exp (-(θ : ℝ)))) <
            ∑ j : Fin n, max (freed (band64 d salt) W A₁ R (x.1, x.2.1) ((List.ofFn x.2.2).take j) -
              epsProof (band64 d salt) 265253616) 0)
      ≤ ENNReal.ofReal βe + n * (2⁻¹ : ℝ≥0∞) ^ 1098 := by
  have hinc := band_timedINC chainExPostFactoG_v6 (2 ^ 9) d 524288 512 5 (2 ^ 20) 265253616 (Pous.Sponge.dom salt)
    hd (by norm_num) (dom_inj salt (by norm_num))
  have hβ : ((265253616 : ℕ) : ℝ≥0∞) + 1 ≤ 2 ^ 29 := by
    exact_mod_cast (by norm_num : (265253616 + 1 : ℕ) ≤ 2 ^ 29)
  have key : (((265253616 : ℕ) : ℝ≥0∞) + 1) * chainPi (2 ^ 9) 524288 512 5 (2 ^ 20) 265253616 ≤
      (2⁻¹ : ℝ≥0∞) ^ 1098 :=
    calc _ ≤ 2 ^ 29 * ((2⁻¹ : ℝ≥0∞) ^ 1128 + (2⁻¹ : ℝ≥0∞) ^ 1128) := mul_le_mul' hβ band64_pi_le
      _ = (2⁻¹ : ℝ≥0∞) ^ 1098 := two_pow_mul_two_inv _ _ _ (by norm_num)
  refine (continuous_audit (band64 d salt) 5 (2 ^ 20) 265253616 n _ hinc W A₁ R f βe hβe).trans ?_
  rw [mul_assoc]
  exact add_le_add le_rfl (mul_le_mul' le_rfl key)

end PousContinuous
