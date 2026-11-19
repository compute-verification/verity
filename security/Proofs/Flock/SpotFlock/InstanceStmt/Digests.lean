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

/-! # The program's and the partition's digests

`progHex` and `partHex` are literals, so that the walks over META read them; here the partition's is what Flock computes,
as `HmRow.partitionDigest` writes it (`partHex_eq`). The program's, `TemplateQuery.Desc.parse`'s program digest, is
`Decode.progHex_eq`, after the walk that reads it. `canon` recurses by well-founded recursion, which the kernel does not
unfold, so each proof evaluates its fuelled twin (`canon_fuel`).
-/

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

/-- **The partition's digest is `partHex`.** -/
theorem partHex_eq : (hexOf <$> HmRow.partitionDigest partObj).toOption = some partHex := by
  unfold HmRow.partitionDigest
  simp only [canon_fuel 8]
  decide +kernel

/-- `isSha512Hex` through `toList`: the kernel does not run `String.all`, so no walk decides it on a digest by itself
(META's two, `prog_hex` and `part_hex`, and the partition's program, `program_hex`). -/
theorem sha512Hex_of (s : String)
    (h : (s.toList.length == 128 && s.toList.all fun c => c.isDigit || ('a' ≤ c && c ≤ 'f')) = true) :
    HmRow.isSha512Hex s = true := by
  rw [HmRow.isSha512Hex, String.all_bool_eq, ← String.length_toList]
  exact h

end SpotFlock.InstanceStmt
