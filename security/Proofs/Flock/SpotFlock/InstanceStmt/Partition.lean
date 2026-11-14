module

public import Proofs.Flock.SpotFlock.InstanceStmt.Digests
public import Proofs.Flock.Claims.Sampling.Select.SecondWitness
import Flock.HmRow
import Flock.TemplateQuery
import Flock.Canon
import all Init.Data.Array.QSort.Basic
import all Init.Data.String.Defs
import all Init.Data.String.Basic
import all Lean.Data.Json.Basic
import all Init.Data.Ord.String
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.List.Control
import all Init.Data.Range.Basic
import all Init.Data.Int.Repr
import all Init.Data.Repr
import all Lean.Data.Json.FromToJson.Basic
import all Std.Data.TreeMap.Raw.Basic
import all Std.Data.DTreeMap.Raw.Basic
import all Std.Data.DTreeMap.Internal.Operations
import all Std.Data.DTreeMap.Internal.Queries
import all Std.Data.DTreeMap.Internal.Balancing
import all Std.Data.DTreeMap.Internal.Def
import all Init.Data.Ord.Basic

/-! # The two-instance statement's partition, as `HmRow.checkPartition` reads it

`checkPartition` reads the partition object `partObj` twice: `checkPartitionObject` holds its keys, its format, its
program's digest and its query (`cpo_part`), and `partitionDigest` hashes its canonical text (`canon_part`). Each sorts
keys with `Array.qsort` or writes text with `canon`, well-founded both, which the kernel does not unfold: each is its
fuelled twin (`qsort_fuel`, `canon_fuel`). The rewrite leaves matches alone (`iota := false`): `templatesOf`'s match
on an array literal has no equation lemmas.
-/

@[expose] public section
open Flock
open Properties.Spot.SecurityProofs (canon_fuel canonFuel qsort_fuel)
open Lean (Json)

namespace SpotFlock.InstanceStmt

theorem program_hex : HmRow.isSha512Hex ((partObj.getObjValAs? String "program").toOption.getD "") = true :=
  sha512Hex_of _ (by decide +kernel)

/-- **The partition object checks**: its keys, its format, a SHA-512 digest for its program, and the query
`Q_template_instance` at version 0 with the one parameter `template`. -/
theorem cpo_part : HmRow.checkPartitionObject partObj = .ok ("Q_template_instance", 0) := by
  unfold HmRow.checkPartitionObject TemplateQuery.templatesOf
  simp (config := { iota := false }) only [qsort_fuel, program_hex]
  kernel_rfl'

/-- The partition object's canonical text, which `partitionDigest` hashes. -/
def partText : String := (canonFuel 8 partObj).toOption.getD ""

theorem canon_part : canon partObj = .ok partText := by
  rw [canon_fuel 8]
  kernel_rfl'

end SpotFlock.InstanceStmt
