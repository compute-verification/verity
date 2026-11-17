module

public import Proofs.Pouw.Lemmas.PearlCFlockTile
public import Proofs.Pouw.PearlC.FlockTile.Audit
public import Proofs.Pouw.PearlC.HiddenTileGame

/-! Pearl-C's gap under Flock's hidden audit, conditional on T3 and the binding, whose TileRead Definition, Flock
soundness and honest prover are not in Lean. -/

@[expose] public section

namespace Pouw.PearlC.FlockTile

open Pouw.PearlC.Assumptions

variable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]

theorem tileProofSoundAll_flock (P : Protocol Q R S) (TR : TileRules Q R S) (D : Layout → Prop)
    (F : TileReadSession Q R S) (δzk : ℕ → ℝ) (h0 : ∀ q, 0 ≤ δzk q) (hT3 : TileReadSound P D F δzk)
    (hbind : TileReadBinds P TR D F) : TileProofSoundAll P TR D F.audit δzk :=
  tileProofSoundAll_of_tileGame P TR D F.audit F.cls δzk h0 (fun _ _ _ _ _ h => h)
    (fun U hU τ o H s g => F.tileGame P TR D δzk h0 hT3 hbind U hU τ o H s g)

/-- Conditional on T3 (`TileReadSound`) and the binding (`TileReadBinds`), neither proved, and cited only beside the
honest prover's `ProverBounded` at `F.audit`. It says something about Flock only where the session's `game` is
TileRead's live session and its `Cls` T3's CR hypotheses: at an empty `Cls` both obligations hold, `Bad` everything. -/
theorem pearlCHiddenFlockSm120v1LoopCast8p72Rev1InertShare1000_8192 (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S) (F : TileReadSession Q R S) (δzk : ℕ → ℝ) (h0 : ∀ q, 0 ≤ δzk q)
    (hTT : TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop)
      sem (1 / 1000))
    (hT3 : TileReadSound (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) F δzk)
    (hbind : TileReadBinds (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) F) :
    GGapHidden (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt)
      (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) F.audit
      ((1648180439 / 446072750000 : ℚ) : ℝ) δPearlC δzk :=
  pearlCHiddenSm120v1LoopCast8p72Rev1InertShare1000_8192 sha fetch salt sem F.audit δzk hTT
    (tileProofSoundAll_flock _ _ _ F δzk h0 hT3 hbind)

end Pouw.PearlC.FlockTile

namespace Pouw.SecurityProofs.PearlC

theorem HiddenFlockSm120v1LoopCast8p72Rev1InertShare1000_8192 :
    Pouw.Guarantees.PearlC.HiddenFlockSm120v1LoopCast8p72Rev1InertShare1000_8192 :=
  fun sha fetch salt sem F δzk h0 hTT hT3 hbind =>
    Pouw.PearlC.FlockTile.pearlCHiddenFlockSm120v1LoopCast8p72Rev1InertShare1000_8192 sha fetch salt sem F δzk h0 hTT hT3
      hbind

end Pouw.SecurityProofs.PearlC
