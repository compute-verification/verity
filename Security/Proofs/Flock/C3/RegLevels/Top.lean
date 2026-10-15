import Proofs.Flock.Recursive.HoloExec
import Proofs.Flock.C3.RegLevels.TopHold

/-!
# Ruling 1: R's registered top from the verifier's hold of the inner transcript

The event carries the verifier's hold of the inner transcript (`innerCommitsAt`, in `AcceptsC3`). `RecVStar.candidate3`
holds R's stream only with a round 0 (its count check), walks its rounds in round order so round 0's tops are pushed
first, and refuses a binding round whose first top does not make, canonically, the `stacked` entry of the registration
META binds (`TopHold.innerCommits_top`). So R's stream's first top is the registered top's commit strings, and a cap whose
commit strings they are is R's registered top (`StackedTopAt`).
-/

namespace FlockC3.RegLevels

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Model Accounting Exec StrictCR

/-- **The binding round's top is R's registered top** (gap t0): where the verifier holds the inner transcript, R's
stream's first top `bcs` makes the registered `stacked` entry, so a cap whose commit strings under `salt` are `bcs` is
R's registered top under `salt`. -/
theorem stackedTop_of_innerCommits {tr : Lean.Json} {record : String} {st : Flock.Stmt} {reg : Lean.Json}
    {cs : Flock.RecVStar.InnerCommits} {s : Flock.RecVStar.InnerStream} {bcs : Array ByteArray} {lv : Level}
    {cap : Cap Refine.Dig} {salt : ℕ → ByteArray} (h : innerCommitsAt tr record st reg = .ok cs)
    (hs : cs.stream? "stacked/rep0" = some s) (htop : s.tops[0]? = some bcs) (hcap : TopOf lv cap salt bcs) :
    StackedTopAt lv st reg cap salt bcs := by
  unfold innerCommitsAt at h
  obtain ⟨_, -, h⟩ := TopHold.eok_bind h
  obtain ⟨prog, e, hreg, hfields, -⟩ := TopHold.innerCommits_top h hs htop
  exact ⟨prog, e, hreg, hfields, hcap⟩

end FlockC3.RegLevels

#print axioms FlockC3.RegLevels.stackedTop_of_innerCommits
