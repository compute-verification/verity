module

public import Proofs.Pous.Chain.Family
import Mathlib.Algebra.Order.Algebra
import Mathlib.Tactic.Positivity.Finset

/-! The band store with a public encoder, at every segment count from 1 to `2^64`. -/

@[expose] public section

namespace PousPub

open Pous
open scoped ENNReal

def ChainBandPubMeetsSeg (N : ℕ) [NeZero N] (d : ℕ) : Prop :=
  ∀ salt : Bits 192,
    PubMeets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.BandMulti.segDom salt (2 ^ 9)))
      (exactCheck _) .sequential 5 (2 ^ 20) 111 ((2⁻¹ : ℝ≥0∞) ^ 128)

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

/-- Holds under four deployment conditions the statement does not enforce: the check decodes fully and exactly, not
by sampling or spot check; the verifier or the protocol fixes the salt; `W` is fixed before setup, the primitive
independent of `W` and the salt; and the commitment to `W` and the binding of `C*` are idealized. -/
theorem band_pub_meets_family_final : ∀ (N : ℕ) [NeZero N], N ≤ 2 ^ 64 → ∀ d : ℕ, 5 ≤ d →
    ChainBandPubMeetsSeg N d ∧ ChainBandPubMeetsSegGlobal N d := fun N _ hN d hd =>
  have h := Pous.BandMulti.band_meets_family_final N hN d hd
  ⟨(band_pub_iff_seg N d).2 h.1, (band_pub_iff_seg_global N d).2 h.2⟩

theorem band_pub_meets_family_14 (d : ℕ) (hd : 5 ≤ d) : ChainBandPubMeets14 d ∧ ChainBandPubMeets14Global d :=
  band_pub_meets_family_final 448 (by norm_num) d hd

end PousPub
