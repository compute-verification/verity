import Proofs.Pous.Chain.Family

/-!
# The band store with a public encoder at every segment count from 1 to 2^64

`band_pub_iff_14` and `band_pub_iff_14_global` (`Flat.lean`, `PubEncoderBand`) fix `N = 448`. Here `N` is any count:
`chainBandSeg N` has no coins and encode-after-decode (`chainBandSeg_encDec`), so `pubMeets_exact_iff` makes `PubMeets`
with the exact check the same statement as `Meets`, and `band_meets_family_final` proves the latter for
`1 ≤ N ≤ 2^64` and `d ≥ 5`. `Pous.SecurityProofs.BandPubMeetsFamily` proves the guarantee `BandPubMeetsFamily` from it.
-/

namespace PousPub

open Pous
open scoped ENNReal

/-- The `N`-segment band under the implementation's per-segment tags, with a public encoder and the exact check,
`k = 111`. At `N = 448` it is `ChainBandPubMeets14`. -/
def ChainBandPubMeetsSeg (N : ℕ) [NeZero N] (d : ℕ) : Prop :=
  ∀ salt : Bits 192,
    PubMeets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.BandMulti.segDom salt (2 ^ 9)))
      (exactCheck _) .sequential 5 (2 ^ 20) 111 ((2⁻¹ : ℝ≥0∞) ^ 128)

/-- The same with one global salt and global indices. At `N = 448` it is `ChainBandPubMeets14Global`. -/
def ChainBandPubMeetsSegGlobal (N : ℕ) [NeZero N] (d : ℕ) : Prop :=
  ∀ salt : Bits 256,
    PubMeets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) (exactCheck _)
      .sequential 5 (2 ^ 20) 111 ((2⁻¹ : ℝ≥0∞) ^ 128)

theorem band_pub_iff_seg (N : ℕ) [NeZero N] (d : ℕ) :
    ChainBandPubMeetsSeg N d ↔ Pous.BandMulti.ChainBandMeetsSeg N d :=
  forall_congr' fun _ => pubMeets_exact_iff _ (chainBandSeg_encDec _ _ _ _ _ _)

theorem band_pub_iff_seg_global (N : ℕ) [NeZero N] (d : ℕ) :
    ChainBandPubMeetsSegGlobal N d ↔ Pous.BandMulti.ChainBandMeetsSegGlobal N d :=
  forall_congr' fun _ => pubMeets_exact_iff _ (chainBandSeg_encDec _ _ _ _ _ _)

/-- **The band store with a public encoder at every segment count from 1 to 2^64**, unconditional, both domain
layouts, every `d ≥ 5`. Deployment conditions (review §48): (1) the check is the exact, full-decode check, and a
sampled or spot check is not covered; (2) the salt is fixed by the verifier or the protocol, not the server; (3) `W`
is fixed before setup, and the primitive is independent of `W` and the salt; (4) the commitment to `W` and the
binding of `C*` are idealized. -/
theorem band_pub_meets_family_final : ∀ (N : ℕ) [NeZero N], N ≤ 2 ^ 64 → ∀ d : ℕ, 5 ≤ d →
    ChainBandPubMeetsSeg N d ∧ ChainBandPubMeetsSegGlobal N d := fun N _ hN d hd =>
  have h := Pous.BandMulti.band_meets_family_final N hN d hd
  ⟨(band_pub_iff_seg N d).2 h.1, (band_pub_iff_seg_global N d).2 h.2⟩

/-- At `N = 448` the family's two layouts are `ChainBandPubMeets14` and `ChainBandPubMeets14Global`. -/
theorem band_pub_meets_family_14 (d : ℕ) (hd : 5 ≤ d) : ChainBandPubMeets14 d ∧ ChainBandPubMeets14Global d :=
  band_pub_meets_family_final 448 (by norm_num) d hd

end PousPub
