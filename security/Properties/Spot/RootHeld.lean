module

public import Spot.OneStageZk

/-!
# The per-member variant tests its record against the root its coins verify under (`MembersRootHeld`)

Everything the epoch's record registers hangs from the epoch's open root R: the registration check takes every
registered value's domain under it (`Docket.check`, through `Docket.ownEntries`), and the members' check holds each
run input node's one tree to the domain under it (`OneStageZk.runInputRoots`, `OneStageZk.runTreeHeld`). The verifier
reads R from its own parameters (`Docket.Verifier.windowRoot`), never from the record or the evidence.
`MembersRootHeld` says that where the audit accepts, that R is the root of the challenger session whose signed coins it
drew its sample from (`Challenger.sampleCoins`) and whose coin trees it held the sessions to
(`Challenger.trustedTreesOf`): the record's tests ran against the root the session's coins were drawn after.
-/

@[expose] public section

open Lean (Json)

namespace Properties.Spot

/-- **Where the audit accepts, the root its record's tests read is the root its coins verify under**: the check on the
blobs (`OneStageZk.checkBlobs`) accepts only at coins `ξ` that the challenger session the auditor's ledger opens on
the verifier's epoch root R released, under the verifier's key and family, after the registration's digest; and on
parameters that name members, the registration check passed with R as the epoch root, the challenger's coin trees
read under R, and every member's run input trees held to their domains under R. -/
def MembersRootHeld : Prop :=
  ∀ (p : _root_.Spot.OneStageZk.Params) (c : _root_.Spot.OneStage.Commitment) (e : _root_.Spot.OneStageZk.Evidence)
    (bs : Core.Blobs) (pub : _root_.Spot.OneStage.Public) (prof : Core.Integrity),
    _root_.Spot.OneStageZk.checkBlobs p c e bs = .accept pub prof →
      ∃ key label R ξ, p.base.challenger = some (key, label) ∧ p.base.verifier.windowRoot = some R ∧
        _root_.Spot.Challenger.sampleCoins key R label c.digest e.base.coins = .ok ξ ∧
        _root_.Spot.OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof ∧
        ∀ ms, p.members = some ms →
          _root_.Spot.Docket.check c.record { p.base.verifier with windowRoot := some R } = #[] ∧
          (∃ ts, _root_.Spot.Challenger.trustedTreesOf key R e.base.coins = .ok ts) ∧
          ∃ P rs roots₀ roots, _root_.Spot.Program.parse p.own.program = .ok P ∧
            _root_.Spot.Partition.evaluate P (_root_.Spot.getD p.own.partition "query") = .ok rs ∧
            _root_.Spot.OneStageZk.runInputRoots { p.base.verifier with windowRoot := some R } P.prog rs ms
              c.record roots₀ = .ok roots

end Properties.Spot
