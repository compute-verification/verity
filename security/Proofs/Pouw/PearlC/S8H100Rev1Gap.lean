module

public import Proofs.Pouw.PearlC.W1Best
public import Proofs.Pouw.PearlC.DeviceRev1Gap
public import Proofs.Pouw.PearlC.ServedH100Rev1Gap
public import Definitions.Pouw.Guarantees.PearlCS8

/-!
# The H100 headlines on the card's cost model, from the device model and the conjecture

`ttOutTilePearlCH100Rev1_of_s8` is the abstract form `TTOutTilePearlCH100Rev1 CM` from its two named parts: W1's
completeness on the H100 with INT8 (`W1CompleteH100`, a simulation of `CM` by W1 on `W1Device.s8H100Best`) and TT_OUT
there (`TTOutTilePearlCH100Rev1S8`), by `Pouw.PearlC.ttOutTile_of_simulatedBy`. The guarantees of
`Definitions.Pouw.Guarantees.PearlCS8` are the per-tile gaps at 8192³ (`pearlCSampledH100Rev1_8192`) and on Llama 3.2
1B's served layout (`pearlCSampledH100Rev1_llama32_1b`) at that `CM`.
-/

@[expose] public section

namespace Pouw.PearlC

open Pouw.PearlC.Assumptions

variable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]

/-- **TT_OUT per tile on the card's cost model** from W1's completeness on the H100 with INT8 and TT_OUT there. -/
theorem ttOutTilePearlCH100Rev1_of_s8 {CM : CostModel Q R S} {salt : S → List ℕ} {sem : PearlCSem Q R S} {ρ : ℚ}
    (hW1 : W1CompleteH100 salt CM) (hTT : TTOutTilePearlCH100Rev1S8 salt sem ρ) :
    TTOutTilePearlCH100Rev1 CM sem ρ :=
  ttOutTile_of_simulatedBy hW1.1 hW1.2 ((Pouw.PearlC.TTOutW1.ttOutTileW1_iff _ _ _ _ _ _ _).1 hTT)

end Pouw.PearlC

theorem Pouw.SecurityProofs.PearlC.SampledH100Rev1S8_8192 : Pouw.Guarantees.PearlC.SampledH100Rev1S8_8192 :=
  fun CM salt sem hW1 hTT =>
    Pouw.PearlC.pearlCSampledH100Rev1_8192 CM sem (Pouw.PearlC.ttOutTilePearlCH100Rev1_of_s8 hW1 hTT)

theorem Pouw.SecurityProofs.PearlC.SampledH100Rev1S8Llama32_1b :
    Pouw.Guarantees.PearlC.SampledH100Rev1S8Llama32_1b :=
  fun CM salt sem hW1 hTT =>
    Pouw.PearlC.pearlCSampledH100Rev1_llama32_1b CM sem (Pouw.PearlC.ttOutTilePearlCH100Rev1_of_s8 hW1 hTT)
