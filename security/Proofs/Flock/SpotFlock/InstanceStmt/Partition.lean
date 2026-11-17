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

/-! The two-instance statement's partition as `HmRow.checkPartition` reads it (`cpo_part`, `canon_part`). -/

@[expose] public section
open Flock
open Properties.Spot.SecurityProofs (canon_fuel canonFuel qsort_fuel)
open Lean (Json)

namespace SpotFlock.InstanceStmt

theorem program_hex : HmRow.isSha512Hex ((partObj.getObjValAs? String "program").toOption.getD "") = true :=
  sha512Hex_of _ (by decide +kernel)

theorem cpo_part : HmRow.checkPartitionObject partObj = .ok ("Q_template_instance", 0) := by
  unfold HmRow.checkPartitionObject TemplateQuery.templatesOf
  simp (config := { iota := false }) only [qsort_fuel, program_hex]
  kernel_rfl'

def partText : String := (canonFuel 8 partObj).toOption.getD ""

theorem canon_part : canon partObj = .ok partText := by
  rw [canon_fuel 8]
  kernel_rfl'

end SpotFlock.InstanceStmt
