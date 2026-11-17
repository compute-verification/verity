module

public import Definitions.Pous.Accounting.PubMeets
public import Definitions.Pous.Model.BandChain
import Mathlib.Algebra.Order.Algebra
import Mathlib.Tactic.Positivity.Finset
import Proofs.Pous.Chain.FamilyRounds

/-! The band store with a public encoder at `D = 2` and `D = 1` rounds. -/

@[expose] public section

namespace PousPub

open Pous
open scoped ENNReal

/-- Holds under `band_pub_meets_family_final`'s four deployment conditions. -/
theorem band_pub_meets_family_d2 : ∀ (N : ℕ) [NeZero N], N ≤ 2 ^ 64 → ∀ d : ℕ, 2 ≤ d →
    (∀ salt : Bits 192,
      PubMeets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.BandMulti.segDom salt (2 ^ 9)))
        (exactCheck _) .sequential 2 (2 ^ 20) 97 ((2⁻¹ : ℝ≥0∞) ^ 128)) ∧
    (∀ salt : Bits 256,
      PubMeets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) (exactCheck _)
        .sequential 2 (2 ^ 20) 97 ((2⁻¹ : ℝ≥0∞) ^ 128)) := fun N _ hN d hd =>
  have h := Pous.BandMulti.Rounds.band_meets_family_d2 N hN d hd
  ⟨fun salt => (pubMeets_exact_iff _ (chainBandSeg_encDec _ _ _ _ _ _)).2 (h.1 salt),
    fun salt => (pubMeets_exact_iff _ (chainBandSeg_encDec _ _ _ _ _ _)).2 (h.2 salt)⟩

/-- Holds under `band_pub_meets_family_final`'s four deployment conditions. -/
theorem band_pub_meets_family_d1 : ∀ (N : ℕ) [NeZero N], N ≤ 2 ^ 64 → ∀ d : ℕ, 1 ≤ d →
    (∀ salt : Bits 192,
      PubMeets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.BandMulti.segDom salt (2 ^ 9)))
        (exactCheck _) .sequential 1 (2 ^ 20) 93 ((2⁻¹ : ℝ≥0∞) ^ 128)) ∧
    (∀ salt : Bits 256,
      PubMeets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) (exactCheck _)
        .sequential 1 (2 ^ 20) 93 ((2⁻¹ : ℝ≥0∞) ^ 128)) := fun N _ hN d hd =>
  have h := Pous.BandMulti.Rounds.band_meets_family_d1 N hN d hd
  ⟨fun salt => (pubMeets_exact_iff _ (chainBandSeg_encDec _ _ _ _ _ _)).2 (h.1 salt),
    fun salt => (pubMeets_exact_iff _ (chainBandSeg_encDec _ _ _ _ _ _)).2 (h.2 salt)⟩

end PousPub
