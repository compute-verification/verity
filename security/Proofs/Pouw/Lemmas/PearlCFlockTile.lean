module

public import Definitions.Pouw.PearlC.Device
public import Definitions.Pouw.PearlC.DeviceKernelWref
public import Proofs.Pouw.PearlC.Records
public import Definitions.Pouw.PearlC.Instance
public import Definitions.Pouw.Assumptions.PearlC.TTOutW1
public import Definitions.Pouw.PearlC.FlockTile.Audit

/-! Pearl-C's gap under Flock's hidden audit at the served record, with `TileProofSoundAll` replaced by T3 at the
TileRead session and the binding, each an open hypothesis. -/

@[expose] public section

namespace Pouw.Guarantees.PearlC

open Pouw.PearlC Pouw.PearlC.Assumptions Pouw.PearlC.FlockTile

/-- `TileReadSound` and `TileReadBinds` are open obligations. It says something about Flock only at a session
whose `game` is TileRead's live session and whose `Cls` is T3's CR hypotheses: at an empty `Cls` both obligations
hold with `Bad` everything. -/
def HiddenFlockSm120v1LoopCast8p72Rev1InertShare1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S) (F : TileReadSession Q R S) (δzk : ℕ → ℝ),
    (∀ q, 0 ≤ δzk q) →
    TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    TileReadSound (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) F δzk →
    TileReadBinds (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) F →
    Pouw.PearlC.GGapHidden (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt)
      (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) F.audit
      ((1648180439 / 446072750000 : ℚ) : ℝ) δPearlC δzk

end Pouw.Guarantees.PearlC
