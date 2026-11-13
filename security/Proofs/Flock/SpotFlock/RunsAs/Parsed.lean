module

public import Proofs.Flock.SpotFlock.RunsAs.OfCheck
import Proofs.Flock.Claims.Sampling.Select.Take
import Proofs.Properties.Challenger.CanonRead
public import VStarRecursive.SampledTwo

/-!
# The two-stage form's parameters are two-stage

The form's parameter parser (`VStar.Recursive.SampledTwo.paramsOfJson`, the form's `BlobProtocol.params`) throws on a
rule that is not two-stage, so the parameters it returns carry a two-stage rule (`paramsOfJson_twoStage`). The proof
walks the parser's own code, binding by binding, and reads nothing of its text but that the rule it returns is the one it
matched as two-stage. With it, an accept of the form's check at parsed parameters is an accepted two-stage run with no
hypothesis on the rule (`acceptedTwo_of_parsed`).

Each registration's parameters name each port once (`paramsOfJson_portsNodup`): the form reads both registrations'
parameters with `OneStage.paramsParsed` (`paramsOfJson_parsed`), which refuses ports whose keys are not strictly
increasing (`paramsParsed_portsNodup`).
-/

@[expose] public section

namespace SpotFlock.RunsAs

open Lean (Json)
open Properties.Spot.SecurityProofs (bind_eq_ok)

/-- **The form's parameters have a two-stage rule**: parameters `paramsOfJson` returns carry `Select.Rule.twoStage`. -/
theorem paramsOfJson_twoStage {j : Json} {p : VStar.Recursive.SampledTwo.Params}
    (h : VStar.Recursive.SampledTwo.paramsOfJson j = .ok p) : ∃ grain replay k, p.rule = .twoStage grain replay k := by
  unfold VStar.Recursive.SampledTwo.paramsOfJson at h
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

/-- **Parsed parameters name each port once**: `paramsParsed` refuses ports whose keys are not strictly increasing
(`OneStage.PORTS_ORDER`). -/
theorem paramsParsed_portsNodup {j : Json} {p : Spot.OneStage.Params} (h : Spot.OneStage.paramsParsed j = .ok p) :
    (p.verifier.ports.map (·.1)).toList.Nodup := by
  unfold Spot.OneStage.paramsParsed at h
  obtain ⟨q, -, h⟩ := bind_eq_ok.1 h
  obtain ⟨u, hu, h⟩ := bind_eq_ok.1 h
  obtain rfl : q = p := by simpa only [pure, Except.pure, Except.ok.injEq] using h
  split at hu
  · cases hu
  · rename_i hinc
    rw [Array.toList_map]
    refine List.Pairwise.imp (fun hab he => ?_) (Properties.Challenger.CanonRead.increasing_iff.1 ?_)
    · subst he
      exact String.lt_irrefl _ hab
    · simpa using hinc

/-- **The form's parameter reader is `paramsParsed`**: `paramsAt` returns what `paramsParsed` returns at some value. -/
theorem paramsAt_parsed {j : Json} {q : Spot.OneStage.Params} (h : VStar.Recursive.SampledTwo.paramsAt j = .ok q) :
    ∃ j', Spot.OneStage.paramsParsed j' = .ok q := by
  unfold VStar.Recursive.SampledTwo.paramsAt at h
  obtain ⟨_, -, h⟩ := bind_eq_ok.1 h
  obtain ⟨j', -, h⟩ := bind_eq_ok.1 h
  exact ⟨j', h⟩

/-- **The form's two registrations' parameters are `paramsParsed`'s**: the boundary's and the interior's. -/
theorem paramsOfJson_parsed {j : Json} {p : VStar.Recursive.SampledTwo.Params}
    (h : VStar.Recursive.SampledTwo.paramsOfJson j = .ok p) :
    (∃ jb, Spot.OneStage.paramsParsed jb = .ok p.boundary) ∧ ∃ ji, Spot.OneStage.paramsParsed ji = .ok p.interior := by
  unfold VStar.Recursive.SampledTwo.paramsOfJson at h
  repeat'
    first
    | (cases h; done)
    | (obtain ⟨_, _, h⟩ := bind_eq_ok.1 h)
    | (simp only [pure, Except.pure, Except.ok.injEq] at h
       subst h
       exact ⟨paramsAt_parsed (by assumption), paramsAt_parsed (by assumption)⟩)
    | split at h
    | dsimp only at h

/-- **The form's two registrations name each port once**: the boundary's parameters and the interior's
(`paramsOfJson_parsed`, `paramsParsed_portsNodup`). -/
theorem paramsOfJson_portsNodup {j : Json} {p : VStar.Recursive.SampledTwo.Params}
    (h : VStar.Recursive.SampledTwo.paramsOfJson j = .ok p) :
    (p.boundary.verifier.ports.map (·.1)).toList.Nodup ∧ (p.interior.verifier.ports.map (·.1)).toList.Nodup := by
  obtain ⟨⟨_, hb⟩, _, hi⟩ := paramsOfJson_parsed h
  exact ⟨paramsParsed_portsNodup hb, paramsParsed_portsNodup hi⟩

end SpotFlock.RunsAs
