import Definitions.Pouw.PearlC.TileCheck8
import Definitions.Pouw.PearlC.TileCheck4
import Definitions.Pouw.PearlC.Device
import Definitions.Pouw.PearlC.DeviceFp4

/-!
# Pearl-C's tile checks compute the words (trusted)

L3's words (`TileCheck.ComputesWords`) for the plain `pc8` and `pc4` Programs at core's own instance of its primitives
(`RowCircuit.coreOps`), with no hypothesis on core's float primitives or on the noise: `pc8`'s check refuses a noise
line with a NaN code (`tileCheck8`'s `anchorsOK`). Each proof is the theorem
`Pouw.SecurityProofs.PearlC.G : Pouw.Guarantees.PearlC.G`.
-/

namespace Pouw.PearlC

open Pouw.PearlC.WholeCircuit Pouw.PearlC.RowCircuit

/-- **`pc8`'s tile check computes the words at core's instance**: at a step `⟨[32], w, z⟩` with `w ≥ 24` (`sm120E4m3`
among them), on `D_dev`'s layouts whose every unit's tile divides its shape, and the audit's tiles, an accepted tile's
ticket and useful words have the reference values on every credited row. -/
def _root_.Pouw.Guarantees.PearlC.TileCheck8WordsCore : Prop :=
  ∀ {Q R S : Type} {w : ℕ}, 24 ≤ w → ∀ (zf : ℤ) (pr : Prices) (c : Costs) (ρ : ℚ)
    (noise : (Q → R) → S → (U : Workload) → (u : ℕ) → Codes →
      Noise (U.layout.shape u).m (U.layout.shape u).n (U.layout.shape u).k Params.pi.r),
    (tileCheck8 coreOps ⟨[32], w, zf⟩ noise).ComputesWords
      (pearlCProtocolDev (devAt ⟨[32], w, zf⟩ 4 pr) (pc8Sem coreOps ⟨[32], w, zf⟩ noise) ρ)
      (pearlCTiles (pc8Sem coreOps ⟨[32], w, zf⟩ noise) c ρ) (fun L => pearlCDomainDev (devAt ⟨[32], w, zf⟩ 4 pr) L ∧
        ∀ u < L.N, tileSide (L.shape u).m ∣ (L.shape u).m ∧ tileSide (L.shape u).n ∣ (L.shape u).n)

/-- **`pc4`'s tile check computes the words at core's instance**, on D₄ and the audit's tiles: an accepted tile's
ticket and useful words have the reference values on every credited row. -/
def _root_.Pouw.Guarantees.PearlC.TileCheck4WordsCore : Prop :=
  ∀ {Q R S : Type} (ρ : ℚ)
    (lines : (Q → R) → S → (U : Workload) → (u : ℕ) → Codes →
      Lines4 (U.layout.shape u).m (U.layout.shape u).n (U.layout.shape u).k),
    (tileCheck4 coreOps lines).ComputesWords (pearlCProtocolFp4 devFp4Sm120 (pc4Sem coreOps lines) ρ)
      (pearlCTilesFp4 devFp4Sm120 (pc4Sem coreOps lines) ρ) (pearlCDomainFp4 (pc4Sem coreOps lines))

end Pouw.PearlC
