module

public import Proofs.Flock.SpotFlock.RunsAs.OwnReadsOut
public import Proofs.Flock.SpotFlock.RunsAs.OfCheck

/-! At the form's accept, the reads the interior record's checks leave out are reads the statement check held
(`leftOut_of_check`). -/

@[expose] public section

namespace SpotFlock.RunsAs

open Lean (Json)

theorem leftOut_of_check {form : Bool} {p : VStar.Recursive.SampledTwo.Params} {c : Spot.OneStage.Commitment}
    {ξ : ByteArray} {e : VStar.Recursive.SampledTwo.Evidence} {bs pub prof}
    (hp : ∃ grain replay k, p.rule = .twoStage grain replay k)
    (hsn : SampledNoneOf p c ξ e bs)
    (h : VStar.Recursive.SampledTwo.checkBlobsAtWith form p c ξ e bs = .accept pub prof) :
    ∃ c₂ ξ₂ r i it, VStar.Recursive.SampledTwo.inputAt p c ξ e bs = .ok (c₂, ξ₂, r, i) ∧ i.interior = some it ∧
      ∀ opened v root,
        Spot.Record.ownRegistered it.record (Spot.Record.sessionRegistered i.verdicts) = some opened →
        (v, root) ∈ opened → (v, root) ∉ (Spot.Record.lessOwnReads it.record (some opened)).getD #[] →
        ∃ x, (v, x) ∈ Spot.Record.byPort it.record ∧ Spot.jeq (Spot.getD x "root") root = true ∧
          (Spot.jeq (Spot.getD x "root") .null = true ∨ HeldRead i it v root) := by
  obtain ⟨-, c₂, ξ₂, r, i, it, run, hin, ha, -⟩ := acceptedTwo_of_check hp hsn h
  exact ⟨c₂, ξ₂, r, i, it, hin, ha.interior, fun _ _ _ ho hv hno =>
    accepted_leftOut ha ho hv hno⟩

end SpotFlock.RunsAs
