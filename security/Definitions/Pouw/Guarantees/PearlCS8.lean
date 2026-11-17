module

public import Definitions.Pouw.Guarantees.PearlCDev
public import Definitions.Pouw.Assumptions.PearlC.TTOutS8

/-! The H100 headlines under rev1 per audit tile at the card's own cost model, conditional on `W1CompleteH100`, which is falsified on the H200, and `TTOutTilePearlCH100Rev1S8`, which is refuted. -/

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
