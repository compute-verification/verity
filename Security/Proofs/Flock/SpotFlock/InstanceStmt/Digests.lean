module

public import Proofs.Flock.SpotFlock.InstanceStmt.Header
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

`progHex` and `partHex` are literals, so that the walks over META read them; here they are what Flock computes: the
program's SHA-512 as `TemplateQuery.Desc.parse` reads the program (`progHex_eq`), and the partition's as
`HmRow.partitionDigest` writes it (`partHex_eq`). `canon` recurses by well-founded recursion, which the kernel does not
unfold, so each proof evaluates its fuelled twin (`canon_fuel`).
-/

@[expose] public section

namespace SpotFlock.InstanceStmt

open Flock Flock.Firewall
open FlockSoundness.Discharge.Recursion.FirewallJson (parse_printJson)
open FlockSoundness.Refine
open Properties.Spot.SecurityProofs (canon_fuel)
open Lean (Json)

theorem plain_progJ : plain progJ = true := by kernel_rfl
theorem fjnorm_progJ : FlockSoundness.Discharge.Recursion.FirewallJson.norm progJ = progJ := by kernel_rfl

theorem parse_prog : JsonParse.parse progText = .ok progJ := by
  rw [progText, parse_printJson plain_progJ, fjnorm_progJ]

/-- **The program's digest is `progHex`.** -/
theorem progHex_eq : ((fun d => hexOf d.sha512) <$> TemplateQuery.Desc.parse progBytes).toOption = some progHex := by
  unfold TemplateQuery.Desc.parse TemplateQuery.Desc.parseWith progBytes
  simp only [fromUTF8?_toUTF8, parse_prog, canon_fuel 12]
  decide +kernel

/-- **The partition's digest is `partHex`.** -/
theorem partHex_eq : (hexOf <$> HmRow.partitionDigest partObj).toOption = some partHex := by
  unfold HmRow.partitionDigest
  simp only [canon_fuel 8]
  decide +kernel

end SpotFlock.InstanceStmt
