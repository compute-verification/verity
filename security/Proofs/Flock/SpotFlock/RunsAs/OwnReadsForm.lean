module

public import Proofs.Flock.SpotFlock.RunsAs.OwnReadsOfCheck

/-!
# At the form's accept, the reads its last step's record checks leave out, at that step's own reading

`leftOut_of_check` states the fact at the run of record's input: its interior record `it.record` and its verdicts
`i.verdicts`. The form's last step (`VStar.Recursive.SampledTwo.secondChecked`) calls `Spot.Record.checks` at another
reading of the same two things: the commitment `c₂` of the interior record's canonical text, and the sessions' own
verdicts `r.out`, before `stamped` names their sessions. `accepted_ownReads_form` restates the fact at that reading,
from two premises, one for each:

* `RecordReadSame`: the interior record the run of record reads is `c₂.record`. Open: the run of record reads
  `readBack`'s value, `Flock.JsonParse.parse` of the canonical text, and `c₂.record` is `Lean.Json.parse` of the same
  text (`Core.canonicalJson`); no lemma joins the two parsers.
* `VerdictsReadSame`: `Spot.Record.sessionRegistered` reads the same of the run of record's verdicts and of `r.out`.
  It holds (`verdictsReadSame`, in `StampedReads`), which leaves the corollary on the record's premise alone
  (`ownReads_form_of_record`).
-/

@[expose] public section

namespace SpotFlock.RunsAs

open Lean (Json)

/-- **The interior record the run of record reads is the record the form parsed.** An open premise: where `inputAt`
returns, the interior record of the input it built (`readBack`'s reading of the evidence's record) is the record of
the commitment the form's last step reads (`Spot.OneStage.commitmentOf` of the same record's canonical text). -/
def RecordReadSame : Prop :=
  ∀ (p : VStar.Recursive.SampledTwo.Params) (c : Spot.OneStage.Commitment) (ξ : ByteArray)
    (e : VStar.Recursive.SampledTwo.Evidence) (bs : Core.Blobs) (c₂ : Spot.OneStage.Commitment) (ξ₂ : ByteArray)
    (r : Spot.OneStageZk.Run) (i : Spot.Run.Input) (it : Spot.Run.Interior),
    VStar.Recursive.SampledTwo.inputAt p c ξ e bs = .ok (c₂, ξ₂, r, i) → i.interior = some it →
      it.record = c₂.record

/-- **The run of record's verdicts report the reads the sessions' own verdicts report.** Where `inputAt` returns,
`Spot.Record.sessionRegistered` reads the same of the verdicts of the input it built (`stamped`'s) and of the
sessions' own (`r.out`), which the form's last step reads. It holds: `verdictsReadSame`. -/
def VerdictsReadSame : Prop :=
  ∀ (p : VStar.Recursive.SampledTwo.Params) (c : Spot.OneStage.Commitment) (ξ : ByteArray)
    (e : VStar.Recursive.SampledTwo.Evidence) (bs : Core.Blobs) (c₂ : Spot.OneStage.Commitment) (ξ₂ : ByteArray)
    (r : Spot.OneStageZk.Run) (i : Spot.Run.Input),
    VStar.Recursive.SampledTwo.inputAt p c ξ e bs = .ok (c₂, ξ₂, r, i) →
      Spot.Record.sessionRegistered i.verdicts = Spot.Record.sessionRegistered r.out

/-- **At the form's accept, each read its last step's record checks leave out is one the second-stage statement
check held**, at the last step's own reading: of the reads the sessions' own verdicts `r.out` report for the interior
commitment `c₂`'s own values, each one `Record.lessOwnReads` takes out names one of `c₂.record`'s ports at that port's
root and is `HeldRead` at the run of record's input, unless that port's root prints as null. From `leftOut_of_check`
by the two premises. -/
theorem accepted_ownReads_form (hrec : RecordReadSame) (hver : VerdictsReadSame) {form : Bool}
    {p : VStar.Recursive.SampledTwo.Params} {c : Spot.OneStage.Commitment} {ξ : ByteArray}
    {e : VStar.Recursive.SampledTwo.Evidence} {bs pub prof}
    (hp : ∃ grain replay k, p.rule = .twoStage grain replay k)
    (h : VStar.Recursive.SampledTwo.checkBlobsAtWith form p c ξ e bs = .accept pub prof) :
    ∃ c₂ ξ₂ r i it, VStar.Recursive.SampledTwo.inputAt p c ξ e bs = .ok (c₂, ξ₂, r, i) ∧ i.interior = some it ∧
      ∀ opened v root,
        Spot.Record.ownRegistered c₂.record (Spot.Record.sessionRegistered r.out) = some opened →
        (v, root) ∈ opened → (v, root) ∉ (Spot.Record.lessOwnReads c₂.record (some opened)).getD #[] →
        ∃ x, (v, x) ∈ Spot.Record.byPort c₂.record ∧ Spot.jeq (Spot.getD x "root") root = true ∧
          (Spot.jeq (Spot.getD x "root") .null = true ∨ HeldRead i it v root) := by
  obtain ⟨c₂, ξ₂, r, i, it, hin, hint, H⟩ := leftOut_of_check hp h
  have hr : it.record = c₂.record := hrec p c ξ e bs c₂ ξ₂ r i it hin hint
  have hs : Spot.Record.sessionRegistered i.verdicts = Spot.Record.sessionRegistered r.out :=
    hver p c ξ e bs c₂ ξ₂ r i hin
  refine ⟨c₂, ξ₂, r, i, it, hin, hint, fun opened v root ho hv hno => ?_⟩
  rw [← hr, ← hs] at ho
  rw [← hr] at hno ⊢
  exact H opened v root ho hv hno

end SpotFlock.RunsAs
