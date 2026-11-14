module

public import Definitions.Pouw.Guarantees.PearlCDev
public import Proofs.Pouw.PearlC.ServedMixedRev1Gap
public import Proofs.Pouw.PearlC.TTOutW1
public import Proofs.Pouw.PearlC.FlockTile.Gap

/-!
# Pearl-C's served form at every device record: the proofs

The guarantees of `Definitions.Pouw.Guarantees.PearlCDev`, each `Pouw.SecurityProofs.PearlC.G :
Pouw.Guarantees.PearlC.G`:

* `GapDevRev1K`: `pearlCGapDevRev1KOn` at W1, through `TTOutPearlCDevRev1W1`'s reading at the W1 cost model.
* `SampledDevRev1K`: `pearlCSampledDevRev1KOn_gen` at `g₀ = 1/400` at W1.
* `HiddenFlockDevRev1K`: `SampledDevRev1K` under the hidden audit (`gapHidden_of_sampled_all`), with the hidden
  audit's obligation `TileProofSoundAll` from T3 and the binding (`FlockTile.tileProofSoundAll_flock`).
-/

@[expose] public section

namespace Pouw.SecurityProofs.PearlC

open Pouw.PearlC Pouw.PearlC.Assumptions

theorem GapDevRev1K : Pouw.Guarantees.PearlC.GapDevRev1K := by
  intro Q R S _ _ _ _ dv salt d sem ρ c Ss ω hK hTT
  exact pearlCGapDevRev1KOn _ d sem ρ c Ss ω hK.1 (fun s hs => (hK.2 s hs).2)
    (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem SampledDevRev1K : Pouw.Guarantees.PearlC.SampledDevRev1K := by
  intro Q R S _ _ _ _ dv salt d sem ρ c Ss ω hK hTT
  have key := pearlCSampledDevRev1KOn_gen _ d sem ρ c Ss ω (1 / 400) (by norm_num) hK.1
    (fun s hs => (hK.2 s hs).1) (fun s hs => (hK.2 s hs).2) (Pouw.PearlC.TTOutW1.ttOutTilePearlCDevRev1_of_w1 hTT)
  have e : (((1 - (1 - 1 / 400) / ω : ℚ)) : ℝ) = 1 - (1 - 1 / 400) / (ω : ℝ) := by push_cast; ring
  rw [e]
  exact key

theorem HiddenFlockDevRev1K : Pouw.Guarantees.PearlC.HiddenFlockDevRev1K := by
  intro Q R S _ _ _ _ dv salt d sem ρ c Ss ω F δzk hK h0 hTT hT3 hbind
  exact gapHidden_of_sampled_all _ _ _ _ F.audit _ _ δzk (SampledDevRev1K dv salt d sem ρ c Ss ω hK hTT)
    (FlockTile.tileProofSoundAll_flock _ _ _ F δzk h0 hT3 hbind)

end Pouw.SecurityProofs.PearlC
