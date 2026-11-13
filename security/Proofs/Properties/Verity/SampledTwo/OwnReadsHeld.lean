module

public import Proofs.Flock.SpotFlock.RunsAs.OwnReadsOut
public import Proofs.Flock.SpotFlock.RunsAs.OfCheck

/-!
# At the form's accept, the reads the interior record's checks leave out are reads the statement check held

The two-stage form's check, accepting, accepted a two-stage run of record (`RunsAs.acceptedTwo_of_check`). That run
hands the interior record's checks (`Spot.Record.checks`) the reads the verdicts of record report for the
registration's own values, less the reads of its own ports at their own roots (`Spot.Record.lessOwnReads`), and each
read so left out is one the run's second-stage statement check held (`RunsAs.accepted_leftOut`). The form's own last
step calls `Spot.Record.checks` at the same reads of the interior record's canonical text and of the verdicts before
`stamped` names their sessions; this file states the fact at the run of record's input.
-/

@[expose] public section

namespace VStar.Recursive.SampledTwo

open Lean (Json)

/-- **At the form's accept, each read the interior record's checks leave out is one the second-stage statement check
held**: the run of record `inputAt` built the input of is an accepted two-stage run, and each read
`Record.lessOwnReads` takes out of the verdicts' own for its interior record names one of that record's ports at that
port's root and is `RunsAs.HeldRead`, unless that port's root prints as null (`RunsAs.accepted_leftOut`). -/
theorem accepted_ownReads_form {form : Bool} {p : Params} {c : Spot.OneStage.Commitment} {ξ : ByteArray}
    {e : Evidence} {bs pub prof} (hp : ∃ grain replay k, p.rule = .twoStage grain replay k)
    (h : checkBlobsAtWith form p c ξ e bs = .accept pub prof) :
    ∃ c₂ ξ₂ r i it, inputAt p c ξ e bs = .ok (c₂, ξ₂, r, i) ∧ i.interior = some it ∧
      ∀ opened v root,
        Spot.Record.ownRegistered it.record (Spot.Record.sessionRegistered i.verdicts) = some opened →
        (v, root) ∈ opened → (v, root) ∉ (Spot.Record.lessOwnReads it.record (some opened)).getD #[] →
        ∃ x, (v, x) ∈ Spot.Record.byPort it.record ∧ Spot.jeq (Spot.getD x "root") root = true ∧
          (Spot.jeq (Spot.getD x "root") .null = true ∨ SpotFlock.RunsAs.HeldRead i it v root) := by
  obtain ⟨-, c₂, ξ₂, r, i, it, run, hin, ha, -⟩ := SpotFlock.RunsAs.acceptedTwo_of_check hp h
  exact ⟨c₂, ξ₂, r, i, it, hin, ha.interior, fun _ _ _ ho hv hno =>
    SpotFlock.RunsAs.accepted_leftOut ha ho hv hno⟩

end VStar.Recursive.SampledTwo
