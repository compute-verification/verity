import Proofs.Pous.Chain.FamilyRounds

/-!
# The band store with a public encoder at `D = 2` and `D = 1` rounds

`PubFamily.lean` at fewer rounds: `chainBandSeg N` has no coins and encode-after-decode (`chainBandSeg_encDec`), so
`pubMeets_exact_iff` makes `PubMeets` with the exact check the same statement as `Meets` at any `D` and `k`, and
`FamilyRounds.lean` proves the latter at `D = 2`, `k = 97` and at `D = 1`, `k = 93`. `Pous.SecurityProofs` proves the
guarantees `BandPubMeetsFamilyD2` and `BandPubMeetsFamilyD1` from them.
-/

namespace PousPub

open Pous
open scoped ENNReal

/-- **The band store with a public encoder at `D = 2`, `k = 97`**, every `1 ≤ N ≤ 2^64` and `d ≥ 2`, both domain
layouts, unconditional, under `band_pub_meets_family_final`'s four deployment conditions. -/
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

/-- **The band store with a public encoder at `D = 1`, `k = 93`**, every `1 ≤ N ≤ 2^64` and `d ≥ 1`, both domain
layouts, unconditional, under the same four deployment conditions. -/
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
