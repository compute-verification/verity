module

public import Definitions.Pouw.Guarantees.PearlCDev
public import Definitions.Pouw.Assumptions.PearlC.TTOutS8

/-!
# The H100 headlines under rev1 per audit tile, on the card's own cost model (trusted)

`Pouw.Guarantees.PearlC.SampledH100Rev1_8192` and its kin are stated at W1 on `W1Device.h100 sha fetch`, a setting
without the card's integer tensor core and at prices per SM clock. These are the gaps at the card's cost model `CM` on
`devH100Tensor`, the H100 record with its SM pipes priced per second at the tensor clock (`Prices.h100Tensor`), so
that they hold whichever clock each pipe holds under a mixed load. They are conditional on the two open premises of
`Definitions.Pouw.Assumptions.PearlC.TTOutS8`, which together imply TT_OUT at `CM` on that record: the device model `W1CompleteH100`
(`w1-complete-h100`, W1 on `W1Device.s8H100Best` dominates `CM` within every bound on rev1's audit tiles) and the conjecture
`TTOutTilePearlCH100Rev1S8` (`tt-out-tile-pearl-c-h100-rev1`, TT_OUT over W1 there). The first is falsified on the
H200 and TT_OUT rev1 is refuted by the proofs red team's rating, so the two are implications from refuted premises, kept
for the record.
-/

@[expose] public section

namespace Pouw.PearlC
open Pouw.PearlC.Assumptions

/-- **The H100 headline under rev1 per audit tile, 8192³, on the card's cost model**: `GGapSampled` at `CM` on
`devH100Tensor` at `g = 216371/40211200` (0.53809%), from W1's completeness on the H100 with INT8 and TT_OUT there. -/
def _root_.Pouw.Guarantees.PearlC.SampledH100Rev1S8_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (salt : S → List ℕ)
    (sem : PearlCSem Q R S) (hW1 : W1CompleteH100 salt sem (1 / 400) CM) (hTT : TTOutTilePearlCH100Rev1S8 salt sem (1 / 400)),
    GGapSampled CM (pearlCProtocolDevRev1 devH100Tensor sem (1 / 400)) (pearlCTilesDevRev1 devH100Tensor sem (1 / 400))
      (pearlCDomainDevAt devH100Tensor sh8192) ((216371 / 40211200 : ℚ) : ℝ) δPearlC

/-- **The H100 record under rev1 per audit tile on Llama 3.2 1B's served layout, on the card's cost model**: its four
linears `(n, k)` = (3072, 2048) qkv, (2048, 2048) o, (16384, 2048) gate_up and (2048, 8192) down, at any batch size, in
one layout: `GGapSampled` at `CM` on `devH100Tensor` at `g = 5040227/775520000` (0.64992%), the down projection's. -/
def _root_.Pouw.Guarantees.PearlC.SampledH100Rev1S8Llama32_1b : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (salt : S → List ℕ)
    (sem : PearlCSem Q R S) (hW1 : W1CompleteH100 salt sem (1 / 400) CM) (hTT : TTOutTilePearlCH100Rev1S8 salt sem (1 / 400)),
    GGapSampled CM (pearlCProtocolDevRev1 devH100Tensor sem (1 / 400)) (pearlCTilesDevRev1 devH100Tensor sem (1 / 400))
      (pearlCDomainDevShapes devH100Tensor {(3072, 2048), (2048, 2048), (16384, 2048), (2048, 8192)})
      ((5040227 / 775520000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC
