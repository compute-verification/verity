import Definitions.Pouw.PearlC.Device
import Definitions.Pouw.PearlC.DeviceKernelWref
import Definitions.Pouw.PearlC.DevicePricesLoop
import Definitions.Pouw.PearlC.Instance
import Definitions.Pouw.Assumptions.PearlC.TTOutW1
import Proofs.Pouw.PearlC.FlockTile.Audit

/-!
# Pearl-C's γ under C-Flock's hidden audit (trusted)

`HiddenFlockSm120v1LoopCast8p72Rev1Cap1000_8192` is `pearlCHiddenSm120v1LoopCast8p72Rev1Cap1000_8192`'s statement at
C-Flock's hidden audit (`Pouw.PearlC.FlockTile.TileReadSession.audit`), with the hidden audit's open obligation
`TileProofSoundAll` replaced by the two it reduces to there, T3 at the TileRead session and the binding, each a
hypothesis. Its proof is `Pouw.SecurityProofs.PearlC.HiddenFlockSm120v1LoopCast8p72Rev1Cap1000_8192`.

C-Flock's definitions, and the session's (`FlockTile.Audit`), are still in the proofs package, so this module imports
`Proofs.Pouw.PearlC.FlockTile.Audit`, as `Flock.Guarantees.EndToEndHidden`'s statement does C-Flock's; `Models.lean`,
which the `security` package builds without `security_proofs`, does not import it.
-/

namespace Pouw.Guarantees.PearlC

open Pouw.PearlC Pouw.PearlC.Assumptions Pouw.PearlC.FlockTile

/-- **v1 under rev1 per audit tile at the cap 1/1,000, 8192³, under C-Flock's hidden audit**, FP32 at 8.376, the honest
cast at 8.72: `γ = 1648180439/446072750000` (0.36949%), at `t·ηzk(q)` more error than the sampled audit, for C-Flock's
hidden audit over a TileRead session `F` (`F.audit`: the window root, each drawn tile's TileRead `--zk` session through
the bridge to core's games, and the class of batch provers whose every induced strategy meets the session's `Cls`).

Its hypotheses, at the cost model `W1.w1Priced` on `W1Device.sm120Loop sha fetch`:
* `TTOutTilePearlCDevRev1W1` (the device's timing at the cap at W1, a named assumption, as in the sampled audit);
* `TileReadSound` (T3 at the session, an open obligation to be discharged by C-Flock's soundness of the TileRead
  statement, whose CR hypotheses, `TableCRZ` at `cr/sha-512` and `LinkCRZC` at `ecr/sha-512`, are `F.Cls`);
* `TileReadBinds` (the binding, an open obligation to be proved from TileRead's gates, L4's registered reads,
  hm96-sha512's binding, L3 and L8);
* `0 ≤ ηzk`.

The session is a parameter until TileRead's Definition lands. The statement says something about C-Flock only at the
session whose `game` is TileRead's live session and whose `Cls` is T3's CR hypotheses: at a session with an empty `Cls`,
`F.audit`'s class is empty and both obligations hold with `Bad` everything. It is cited beside the honest prover's
`HiddenAudit.ProverBounded` at `F.audit` (`HiddenAudit`'s docstring). -/
def HiddenFlockSm120v1LoopCast8p72Rev1Cap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S) (F : TileReadSession Q R S) (ηzk : ℕ → ℝ),
    (∀ q, 0 ≤ ηzk q) →
    TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    TileReadSound (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) F ηzk →
    TileReadBinds (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) F →
    Pouw.PearlC.GγHidden (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt)
      (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) F.audit
      ((1648180439 / 446072750000 : ℚ) : ℝ) εPearlC ηzk

end Pouw.Guarantees.PearlC
