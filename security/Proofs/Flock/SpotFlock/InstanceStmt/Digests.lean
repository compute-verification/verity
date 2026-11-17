module

public import Proofs.Flock.SpotFlock.InstanceStmt.Header
public import Proofs.Flock.SpotFlock.InstanceStmt.Plain
import Proofs.Flock.Claims.Sampling.Select.SecondWitness
import Flock.HmRow
import Flock.TemplateQuery
import Flock.Canon
import Flock.JsonParse
import all Init.Data.Array.QSort.Basic
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.String.Defs
import all Init.Data.String.Basic
import all Init.Data.Ord.String
import all Init.Data.Ord.Basic
import all Init.Data.Range.Basic
import all Init.Data.List.Control
import all Init.Data.Repr
import all Init.Data.Int.Repr
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic
import all Std.Data.TreeMap.Raw.Basic
import all Std.Data.DTreeMap.Raw.Basic
import all Std.Data.DTreeMap.Internal.Operations
import all Std.Data.DTreeMap.Internal.Queries
import all Std.Data.DTreeMap.Internal.Balancing
import all Std.Data.DTreeMap.Internal.Def

/-! The literals `progHex` and `partHex` are the digests Flock computes (`progHex_eq`, `partHex_eq`). -/

@[expose] public section

namespace SpotFlock.InstanceStmt

open Flock Flock.ZkServer
open FlockSoundness.Discharge.Recursion.ZkServerJson (parse_printJson)
open FlockSoundness.Refine
open Properties.Spot.SecurityProofs (canon_fuel)
open Lean (Json)

theorem plain_progJ : plain progJ = true := by
  simp only [progJ]
  plain_json
theorem fjnorm_progJ : FlockSoundness.Discharge.Recursion.ZkServerJson.norm progJ = progJ := by kernel_rfl

theorem parse_prog : JsonParse.parse progText = .ok progJ := by
  rw [progText, parse_printJson plain_progJ, fjnorm_progJ]

theorem progHex_eq : ((fun d => hexOf d.sha512) <$> TemplateQuery.Desc.parse progBytes).toOption = some progHex := by
  unfold TemplateQuery.Desc.parse TemplateQuery.Desc.parseWith progBytes
  simp only [fromUTF8?_toUTF8, parse_prog, canon_fuel 12]
  decide +kernel

theorem partHex_eq : (hexOf <$> HmRow.partitionDigest partObj).toOption = some partHex := by
  unfold HmRow.partitionDigest
  simp only [canon_fuel 8]
  decide +kernel

/-- The kernel does not run `String.all`, so no walk decides `isSha512Hex` on a digest by itself. -/
theorem sha512Hex_of (s : String)
    (h : (s.toList.length == 128 && s.toList.all fun c => c.isDigit || ('a' ≤ c && c ≤ 'f')) = true) :
    HmRow.isSha512Hex s = true := by
  rw [HmRow.isSha512Hex, String.all_bool_eq, ← String.length_toList]
  exact h

end SpotFlock.InstanceStmt
