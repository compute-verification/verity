import Definitions.Pouw.PearlC.TileCheck8
import Definitions.Pouw.PearlC.TileCheck4
import Definitions.Pouw.PearlC.Device
import Definitions.Pouw.PearlC.DeviceFp4

/-!
# Pearl-C's tile checks compute the words (trusted)

L3's words (`TileCheck.ComputesWords`, opened) for the plain `pc8` and `pc4` Programs at core's own instance of its
primitives (`RowCircuit.coreOps`): when `tileCheck8` or `tileCheck4` accepts a tile, its words are good. Each is stated
on its accept outcome first (`TileCheck.Accepts`), then the premises on the inputs, then the conclusion
(`TileWordsGood`). `Collides` is `False` for both Programs, so the conclusion has no collision disjunct. There is no
hypothesis on core's float primitives or on the noise: `pc8`'s check refuses a noise line with a NaN code
(`tileCheck8`'s `anchorsOK`). Each proof is the theorem `Pouw.SecurityProofs.PearlC.G : Pouw.Guarantees.PearlC.G`.
-/

namespace Pouw.PearlC

open Pouw.PearlC.WholeCircuit Pouw.PearlC.RowCircuit

/-- **`pc8`'s tile check computes the words at core's instance**: when `tileCheck8` at `coreOps` accepts a witness for
tile `g` and commitment `cm`, at a step of one group of 32 and width `≥ 24` (`sm120E4m3` among them), on a workload of
`D_dev`'s layouts whose every unit's tile divides its shape, and the audit's tiles, the tile's ticket and useful words
have the reference values on every credited row. -/
def _root_.Pouw.Guarantees.PearlC.TileCheck8WordsCore : Prop :=
  ∀ {Q R S : Type} (p : Pouw.Fp8Atom.Pipeline) (pr : Prices) (c : Costs) (ρ : ℚ)
    (noise : (Q → R) → S → (U : Workload) → (u : ℕ) → Codes →
      Noise (U.layout.shape u).m (U.layout.shape u).n (U.layout.shape u).k Params.pi.r)
    (U : Workload) (H : Q → R) (s : S) (g : ℕ) (cm : Commit) X,
    (tileCheck8 coreOps p noise).Accepts (pearlCProtocolDev (devAt p 4 pr) (pc8Sem coreOps p noise) ρ)
        (pearlCTiles (pc8Sem coreOps p noise) c ρ) H s U g cm X →
      p.groups = [32] → 24 ≤ p.width →
      U.InDomain (pearlCProtocolDev (devAt p 4 pr) (pc8Sem coreOps p noise) ρ)
        (fun L => pearlCDomainDev (devAt p 4 pr) L ∧
          ∀ u < L.N, tileSide (L.shape u).m ∣ (L.shape u).m ∧ tileSide (L.shape u).n ∣ (L.shape u).n) →
      g < ((pearlCTiles (pc8Sem coreOps p noise) c ρ).tiling U.layout).NT →
      TileWordsGood (pearlCProtocolDev (devAt p 4 pr) (pc8Sem coreOps p noise) ρ)
        (pearlCTiles (pc8Sem coreOps p noise) c ρ) U H s g cm

/-- **`pc4`'s tile check computes the words at core's instance**: when `tileCheck4` at `coreOps` accepts a witness for
tile `g` and commitment `cm`, on a workload of D₄ and the audit's tiles, the tile's ticket and useful words have the
reference values on every credited row. -/
def _root_.Pouw.Guarantees.PearlC.TileCheck4WordsCore : Prop :=
  ∀ {Q R S : Type} (ρ : ℚ)
    (lines : (Q → R) → S → (U : Workload) → (u : ℕ) → Codes →
      Lines4 (U.layout.shape u).m (U.layout.shape u).n (U.layout.shape u).k)
    (U : Workload) (H : Q → R) (s : S) (g : ℕ) (cm : Commit) X,
    (tileCheck4 coreOps lines).Accepts (pearlCProtocolFp4 devFp4Sm120 (pc4Sem coreOps lines) ρ)
        (pearlCTilesFp4 devFp4Sm120 (pc4Sem coreOps lines) ρ) H s U g cm X →
      U.InDomain (pearlCProtocolFp4 devFp4Sm120 (pc4Sem coreOps lines) ρ) (pearlCDomainFp4 (pc4Sem coreOps lines)) →
      g < ((pearlCTilesFp4 devFp4Sm120 (pc4Sem coreOps lines) ρ).tiling U.layout).NT →
      TileWordsGood (pearlCProtocolFp4 devFp4Sm120 (pc4Sem coreOps lines) ρ)
        (pearlCTilesFp4 devFp4Sm120 (pc4Sem coreOps lines) ρ) U H s g cm

end Pouw.PearlC
