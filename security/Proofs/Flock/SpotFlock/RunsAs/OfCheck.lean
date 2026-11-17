module

public import Proofs.Flock.SpotFlock.RunsAs.Statement
public import Proofs.Flock.SpotFlock.Jeq
public import Proofs.Properties.Verity.SampledTwo.Steps
public import VStarRecursive.SampledTwo

/-! The two-stage form's check, accepting, accepted a two-stage run (`acceptedTwo_of_check`). -/

@[expose] public section

namespace SpotFlock.RunsAs

open Lean (Json)

def SampledNoneOf (p : VStar.Recursive.SampledTwo.Params) (c : Spot.OneStage.Commitment) (ξ : ByteArray)
    (e : VStar.Recursive.SampledTwo.Evidence) (bs : _root_.Core.Blobs) : Prop :=
  ∀ c₂ ξ₂ r i it, VStar.Recursive.SampledTwo.inputAt p c ξ e bs = .ok (c₂, ξ₂, r, i) → i.interior = some it →
    NoSampledOnly i it

theorem acceptedTwo_of_check {form : Bool} {p : VStar.Recursive.SampledTwo.Params} {c : Spot.OneStage.Commitment}
    {ξ : ByteArray} {e : VStar.Recursive.SampledTwo.Evidence} {bs pub prof}
    (hp : ∃ grain replay k, p.rule = .twoStage grain replay k)
    (hsn : SampledNoneOf p c ξ e bs)
    (h : VStar.Recursive.SampledTwo.checkBlobsAtWith form p c ξ e bs = .accept pub prof) :
    form = true ∧ ∃ c₂ ξ₂ r i it run, VStar.Recursive.SampledTwo.inputAt p c ξ e bs = .ok (c₂, ξ₂, r, i) ∧
      AcceptedTwo i it run ∧ VStar.Recursive.SampledTwo.sourceRead ξ run = .ok () := by
  obtain ⟨hf, c₂, ξ₂, r, i, run, hin, hrun, hsr, -, hj, -⟩ := VStar.Recursive.SampledTwo.accepted_steps h
  obtain ⟨-, -, -, -, vs, received₂, j₁, j₂, -, rfl, -, -⟩ := VStar.Recursive.SampledTwo.inputAt_ok hin
  exact ⟨hf, c₂, ξ₂, r, _, _, run, hin, ⟨hrun, str_of_jeq (jeq_symm hj), hp, rfl, hsn _ _ _ _ _ hin rfl⟩, hsr⟩

end SpotFlock.RunsAs
