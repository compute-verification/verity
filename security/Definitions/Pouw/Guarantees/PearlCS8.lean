module

public import Definitions.Pouw.Guarantees.PearlC
public import Definitions.Assumptions.PearlC

/-!
# The H100 headlines under rev1 per audit tile, on the card's own cost model (trusted)

`Pouw.Guarantees.PearlC.SampledH100Rev1_8192` and its kin are stated at W1 on `W1Device.h100 sha fetch`, a setting
without the card's integer tensor core and at prices the H200 was measured at. These are the same gaps at the card's
cost model `CM`, under the two named assumptions `Definitions.Assumptions.PearlC` splits TT_OUT into: the device
model `W1CompleteH100` (`w1-complete-h100`) and the conjecture `TTOutTilePearlCH100Rev1S8`
(`tt-out-tile-pearl-c-h100-rev1`), on `W1Device.s8H100Best`. It is the one PoUW statement module that reads a named
assumptions module (the `Definitions.Pouw` layer's exception in `lean-audit.json`).
-/

@[expose] public section

namespace Pouw.PearlC
open Pouw.PearlC.Assumptions

/-- **The H100 headline under rev1 per audit tile, 8192³, on the card's cost model**: `GGapSampled` at `CM` at
`g = 140237/26090000`, from W1's completeness on the H100 with INT8 and TT_OUT there. -/
def _root_.Pouw.Guarantees.PearlC.SampledH100Rev1S8_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (salt : S → List ℕ)
    (sem : PearlCSem Q R S) (hW1 : W1CompleteH100 salt CM) (hTT : TTOutTilePearlCH100Rev1S8 salt sem (1 / 400)),
    GGapSampled CM (pearlCProtocolDevRev1 devH100 sem (1 / 400)) (pearlCTilesDevRev1 devH100 sem (1 / 400))
      (pearlCDomainDevAt devH100 sh8192) ((140237 / 26090000 : ℚ) : ℝ) δPearlC

/-- **The H100 record under rev1 per audit tile on Llama 3.2 1B's served layout, on the card's cost model**: its four
linears `(n, k)` = (3072, 2048) qkv, (2048, 2048) o, (16384, 2048) gate_up and (2048, 8192) down, at any batch size, in
one layout: `GGapSampled` at `CM` at `g = 69533/10736000` (0.64766%), the down projection's. -/
def _root_.Pouw.Guarantees.PearlC.SampledH100Rev1S8Llama32_1b : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (salt : S → List ℕ)
    (sem : PearlCSem Q R S) (hW1 : W1CompleteH100 salt CM) (hTT : TTOutTilePearlCH100Rev1S8 salt sem (1 / 400)),
    GGapSampled CM (pearlCProtocolDevRev1 devH100 sem (1 / 400)) (pearlCTilesDevRev1 devH100 sem (1 / 400))
      (pearlCDomainDevShapes devH100 {(3072, 2048), (2048, 2048), (16384, 2048), (2048, 8192)})
      ((69533 / 10736000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC
