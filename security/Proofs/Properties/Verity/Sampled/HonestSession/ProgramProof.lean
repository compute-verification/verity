module

public import Proofs.Properties.Verity.Sampled.HonestSession.Data.Program
public import Proofs.Properties.Verity.Sampled
public import Proofs.Flock.Bridge.DrawnLaw
public import Proofs.Flock.SessionLoop.Concrete.Inputs
public import Proofs.Flock.SessionLoop.Concrete.Fwd
public import Proofs.Pouw.Period.Sort
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic
import all Std.Data.TreeMap.Raw.Basic
import all Std.Data.DTreeMap.Raw.Basic
import all Std.Data.DTreeMap.Internal.Operations
import all Std.Data.DTreeMap.Internal.Queries
import all Std.Data.DTreeMap.Internal.Balancing
import all Std.Data.DTreeMap.Internal.Def
import all Init.Data.Ord.String
import all Init.Data.Ord.Basic
import all Init.Data.String.Basic
import all Init.Data.String.Defs
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.List.Control

/-!
# `ProgramFacts`' fields that hold without the descriptor's parse

* `partitionDigest_ok`: the partition object's digest is `partitionDigestB` by that constant's definition: both are
  `Sha512.hash` of the one text `canon partitionObj`, which exists since the object holds only integers
  (`FlockDeploy.Concrete.canon_eq`). No hash is evaluated.
* `lawAt_subset`: the law is `subset`, so `Spot.Draw.deriveLaw` returns `none` without reading the program, and a verifier
  holding the session's query, law and no work table derives the subset of one (`flockLaw_law`). The sort of the law's
  keys is read by `qsort_of_sorted` and `qsort_eq_of_perm`, never run.
* `design_eq_of_parse`, `lawAt_of_parse`: `ProgramFacts.design_eq` and `lawAt_of` at any descriptor that parses.

`program_parse`, `program_sha` and `popMatches` read `Flock.Program.decode`'s walk over `programFile`, which is not here.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession.ProgramProof

open Flock
open Lean (Json)

theorem intJ_partitionObj : FlockDeploy.Concrete.IntJ partitionObj = true := by
  unfold partitionObj Spot.Partition.build
  split <;> decide +kernel

/-- **`ProgramFacts.partitionDigest_ok`**. -/
theorem partitionDigest_ok : HmRow.partitionDigest partitionObj = .ok partitionDigestB := by
  unfold HmRow.partitionDigest
  rw [FlockDeploy.Concrete.canon_eq intJ_partitionObj]
  rfl

theorem keys_law : Spot.keys? law = some (Spot.sortStrings #["k", "law"]) := by kernel_rfl'

theorem sortStrings_kl : Spot.sortStrings #["k", "law"] = #["k", "law"] := by
  apply Array.toList_inj.1
  exact Properties.Pouw.Period.SecurityProofs.Sort.qsort_of_sorted (as := #["k", "law"]) (by decide +kernel)

theorem sortStrings_lawk : Spot.sortStrings #["law", "k"] = #["k", "law"] := by
  have h := Properties.Pouw.Period.SecurityProofs.Sort.qsort_eq_of_perm (as := #["law", "k"]) (bs := #["k", "law"])
    (List.Perm.swap "k" "law" [])
  apply Array.toList_inj.1
  exact h.trans (congrArg Array.toList sortStrings_kl)

theorem flockLaw_law : Spot.Draw.flockLaw law nUnits = .ok (.subset 1) := by
  unfold Spot.Draw.flockLaw
  rw [keys_law, sortStrings_kl, sortStrings_lawk]
  kernel_rfl'

theorem deriveLaw_law (p : Spot.Program) : Spot.Draw.deriveLaw p query law none = .ok none := by
  kernel_rfl'

theorem derive_law (p : Spot.Program) : Spot.Draw.derive p query law none = .ok law := by
  unfold Spot.Draw.derive
  rw [deriveLaw_law]
  rfl

/-- A verifier holding the session's query, law and no work table derives the subset of one, whatever its program. -/
theorem lawAt_subset (v : Spot.Registration.Verifier) (hq : v.query = query) (hl : v.law = law)
    (ht : v.workTable = none) : lawAt v nUnits = .ok (.subset 1) := by
  unfold lawAt
  rw [hq, hl, ht, derive_law, deriveLaw_law]
  simp only [Fwd.bind_ok]
  rw [flockLaw_law]
  rfl

theorem lawObj_none (l : Json) : Properties.Spot.SecurityProofs.lawObj l none = l := rfl

theorem drawnLaw_of_parse {p : Spot.Program} (h : Spot.Program.parse ⟨desc.toArray⟩ = .ok p) :
    Flock.Bridge.drawnLaw desc query law none nUnits = .ok (.subset 1) := by
  unfold Flock.Bridge.drawnLaw
  simp only [h, Fwd.bind_ok, deriveLaw_law, lawObj_none, flockLaw_law]
  rfl

/-- **`ProgramFacts.design_eq`** at a descriptor that parses. -/
theorem design_eq_of_parse {p : Spot.Program} (h : Spot.Program.parse ⟨desc.toArray⟩ = .ok p) :
    Flock.Bridge.designAt desc query law none nUnits = .subset 1 := by
  unfold Flock.Bridge.designAt
  rw [drawnLaw_of_parse h]

/-- **`ProgramFacts.lawAt_of`** at a descriptor that parses. -/
theorem lawAt_of_parse {p : Spot.Program} (h : Spot.Program.parse ⟨desc.toArray⟩ = .ok p)
    (v : Spot.Registration.Verifier) (_ : v.program = program) (hq : v.query = query) (hl : v.law = law)
    (ht : v.workTable = none) : lawAt v nUnits = .ok (Flock.Bridge.designAt desc query law none nUnits) := by
  rw [lawAt_subset v hq hl ht, design_eq_of_parse h]

end Verity.Sampled.Instance.HonestSession.ProgramProof
