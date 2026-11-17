module

public import Proofs.Flock.SpotFlock.RunsAs.OwnReadsOfCheck

/-! At the form's accept, the reads its last step's record checks leave out, at that step's own reading
(`accepted_ownReads_form`); its premise `RecordReadSame` is open, because no lemma joins the two parsers. -/

@[expose] public section

namespace SpotFlock.RunsAs

open Lean (Json)

/-- An open premise: no lemma joins `readBack`'s parse of the record's canonical text with the commitment's. -/
def RecordReadSame : Prop :=
  ∀ (p : VStar.Recursive.SampledTwo.Params) (c : Spot.OneStage.Commitment) (ξ : ByteArray)
    (e : VStar.Recursive.SampledTwo.Evidence) (bs : Core.Blobs) (c₂ : Spot.OneStage.Commitment) (ξ₂ : ByteArray)
    (r : Spot.OneStageZk.Run) (i : Spot.Run.Input) (it : Spot.Run.Interior),
    VStar.Recursive.SampledTwo.inputAt p c ξ e bs = .ok (c₂, ξ₂, r, i) → i.interior = some it →
      it.record = c₂.record

def VerdictsReadSame : Prop :=
  ∀ (p : VStar.Recursive.SampledTwo.Params) (c : Spot.OneStage.Commitment) (ξ : ByteArray)
    (e : VStar.Recursive.SampledTwo.Evidence) (bs : Core.Blobs) (c₂ : Spot.OneStage.Commitment) (ξ₂ : ByteArray)
    (r : Spot.OneStageZk.Run) (i : Spot.Run.Input),
    VStar.Recursive.SampledTwo.inputAt p c ξ e bs = .ok (c₂, ξ₂, r, i) →
      Spot.Record.sessionRegistered i.verdicts = Spot.Record.sessionRegistered r.out

theorem accepted_ownReads_form (hrec : RecordReadSame) (hver : VerdictsReadSame) {form : Bool}
    {p : VStar.Recursive.SampledTwo.Params} {c : Spot.OneStage.Commitment} {ξ : ByteArray}
    {e : VStar.Recursive.SampledTwo.Evidence} {bs pub prof}
    (hp : ∃ grain replay k, p.rule = .twoStage grain replay k)
    (hsn : SampledNoneOf p c ξ e bs)
    (h : VStar.Recursive.SampledTwo.checkBlobsAtWith form p c ξ e bs = .accept pub prof) :
    ∃ c₂ ξ₂ r i it, VStar.Recursive.SampledTwo.inputAt p c ξ e bs = .ok (c₂, ξ₂, r, i) ∧ i.interior = some it ∧
      ∀ opened v root,
        Spot.Record.ownRegistered c₂.record (Spot.Record.sessionRegistered r.out) = some opened →
        (v, root) ∈ opened → (v, root) ∉ (Spot.Record.lessOwnReads c₂.record (some opened)).getD #[] →
        ∃ x, (v, x) ∈ Spot.Record.byPort c₂.record ∧ Spot.jeq (Spot.getD x "root") root = true ∧
          (Spot.jeq (Spot.getD x "root") .null = true ∨ HeldRead i it v root) := by
  obtain ⟨c₂, ξ₂, r, i, it, hin, hint, H⟩ := leftOut_of_check hp hsn h
  have hr : it.record = c₂.record := hrec p c ξ e bs c₂ ξ₂ r i it hin hint
  have hs : Spot.Record.sessionRegistered i.verdicts = Spot.Record.sessionRegistered r.out :=
    hver p c ξ e bs c₂ ξ₂ r i hin
  refine ⟨c₂, ξ₂, r, i, it, hin, hint, fun opened v root ho hv hno => ?_⟩
  rw [← hr, ← hs] at ho
  rw [← hr] at hno ⊢
  exact H opened v root ho hv hno

end SpotFlock.RunsAs
