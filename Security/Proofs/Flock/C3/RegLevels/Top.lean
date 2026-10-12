import Proofs.Flock.Recursive.HoloExec

/-!
# Ruling 1: R's registered top from the verifier's hold of the inner transcript

The event carries the verifier's hold of the inner transcript (`innerCommitsAt`, in `AcceptsC3`). `RecVStar.candidate3`
holds R's stream with every round from 0 (`checkStackedShape`'s count), pushes round 0's tops first, and refuses a
binding round whose first top does not make, canonically, the `stacked` entry of the registration META binds. So R's
stream's first top is the registered top's commit strings, and a cap whose commit strings they are is R's registered top
(`StackedTop`). A stub: gap t0.
-/

namespace FlockC3.RegLevels

open FlockSoundness FlockSoundness.Discharge.Recursion
open Game Model Accounting Exec StrictCR

/-- **The binding round's top is R's registered top** (gap t0): where the verifier holds the inner transcript, R's
stream's first top `bcs` makes the registered `stacked` entry, so a cap whose commit strings are `bcs` is R's
registered top. -/
theorem stackedTop_of_innerCommits {tr : Lean.Json} {record : String} {st : Flock.Stmt} {reg : Lean.Json}
    {cs : Flock.RecVStar.InnerCommits} {s : Flock.RecVStar.InnerStream} {bcs : Array ByteArray} {lv : Level}
    {cap : Cap Refine.Dig} (h : innerCommitsAt tr record st reg = .ok cs)
    (hs : cs.stream? "stacked/rep0" = some s) (htop : s.tops[0]? = some bcs) (hcap : TopOf lv cap bcs) :
    StackedTop lv st reg cap := by
  sorry

end FlockC3.RegLevels
