module

public import Proofs.Flock.SpotFlock.RunsAs.OwnReadsOut
public import Proofs.Flock.SpotFlock.RunsAs.OfCheck

/-!
# At the form's accept, the reads the interior record's checks leave out are reads the statement check held

The two-stage form's check, accepting, accepted a two-stage run of record (`RunsAs.acceptedTwo_of_check`). That run
hands the interior record's checks (`Spot.Record.checks`) the reads the verdicts of record report for the
registration's own values, less the reads of its own ports at their own roots (`Spot.Record.lessOwnReads`), and each
read the verdicts report for it is one the run's second-stage statement check held (`RunsAs.accepted_ownReads`). The
form's own last step calls `Spot.Record.checks` at the same reads of the interior record's canonical text and of the
verdicts before `stamped` names their sessions; this file states the fact at the run of record's input.
-/

@[expose] public section

namespace VStar.Recursive.SampledTwo

open Lean (Json)

/-- **At the form's accept, each read the verdicts report for the interior record is one the second-stage statement
check held**: the run of record `inputAt` built the input of is an accepted two-stage run, and each read among those
its interior record's checks are handed before `Record.lessOwnReads` (the verdicts' own for the record's own values)
is `RunsAs.HeldRead`, or has a root that prints as null. So is each read `Record.lessOwnReads` then leaves out, which
names one of the record's ports at that port's root (`RunsAs.lessOwnReads_out`). -/
theorem accepted_ownReads_form {form : Bool} {p : Params} {c : Spot.OneStage.Commitment} {ξ : ByteArray}
    {e : Evidence} {bs pub prof} (hp : ∃ grain replay k, p.rule = .twoStage grain replay k)
    (h : checkBlobsAtWith form p c ξ e bs = .accept pub prof) :
    ∃ c₂ ξ₂ r i it, inputAt p c ξ e bs = .ok (c₂, ξ₂, r, i) ∧ i.interior = some it ∧
      ∀ opened v root,
        Spot.Record.ownRegistered it.record (Spot.Record.sessionRegistered i.verdicts) = some opened →
        (v, root) ∈ opened → Spot.jeq root .null = true ∨ SpotFlock.RunsAs.HeldRead i it v root := by
  obtain ⟨-, c₂, ξ₂, r, i, it, run, hin, ha, -⟩ := SpotFlock.RunsAs.acceptedTwo_of_check hp h
  exact ⟨c₂, ξ₂, r, i, it, hin, ha.interior, fun _ _ _ ho hv => SpotFlock.RunsAs.accepted_ownReads ha ho hv⟩

end VStar.Recursive.SampledTwo
