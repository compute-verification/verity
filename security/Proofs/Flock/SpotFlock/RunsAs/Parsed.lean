module

public import Proofs.Flock.SpotFlock.RunsAs.OfCheck
import Proofs.Flock.Claims.Sampling.Select.Take
public import VStarRecursive.SampledTwo

/-!
# The two-stage form's parameters are two-stage

The form's parameter parser (`VStar.Recursive.SampledTwo.paramsOfJson`, the form's `BlobProtocol.params`) throws on a
rule that is not two-stage, so the parameters it returns carry a two-stage rule (`paramsOfJson_twoStage`). The proof
walks the parser's own code, binding by binding, and reads nothing of its text but that the rule it returns is the one it
matched as two-stage. With it, an accept of the form's check at parsed parameters is an accepted two-stage run with no
hypothesis on the rule (`acceptedTwo_of_parsed`).
-/

@[expose] public section

namespace SpotFlock.RunsAs

open Lean (Json)
open Properties.Spot.SecurityProofs (bind_eq_ok)

/-- **The form's parameters have a two-stage rule**: parameters `paramsOfJson` returns carry `Select.Rule.twoStage`. -/
theorem paramsOfJson_twoStage {j : Json} {p : VStar.Recursive.SampledTwo.Params}
    (h : VStar.Recursive.SampledTwo.paramsOfJson j = .ok p) : ∃ grain replay k, p.rule = .twoStage grain replay k := by
  replace h := (VStar.Recursive.SampledTwo.paramsOfJson_ok h).1
  unfold VStar.Recursive.SampledTwo.paramsParsed at h
  repeat'
    first
    | (cases h; done)
    | (obtain ⟨_, _, h⟩ := bind_eq_ok.1 h)
    | (simp only [pure, Except.pure, Except.ok.injEq] at h
       subst h
       exact ⟨_, _, _, rfl⟩)
    | split at h
    | dsimp only at h

/-- **The form's check, accepting at parsed parameters, accepted a two-stage run**: `acceptedTwo_of_check` at
parameters the form's parser returned, which are two-stage by `paramsOfJson_twoStage`. -/
theorem acceptedTwo_of_parsed {form : Bool} {j : Json} {p : VStar.Recursive.SampledTwo.Params}
    {c : Spot.OneStage.Commitment} {ξ : ByteArray} {e : VStar.Recursive.SampledTwo.Evidence} {bs pub prof}
    (hj : VStar.Recursive.SampledTwo.paramsOfJson j = .ok p)
    (h : VStar.Recursive.SampledTwo.checkBlobsAtWith form p c ξ e bs = .accept pub prof) :
    form = true ∧ ∃ c₂ ξ₂ r i it run, VStar.Recursive.SampledTwo.inputAt p c ξ e bs = .ok (c₂, ξ₂, r, i) ∧
      AcceptedTwo i it run ∧ VStar.Recursive.SampledTwo.sourceRead ξ run = .ok () :=
  acceptedTwo_of_check (paramsOfJson_twoStage hj) h

end SpotFlock.RunsAs
