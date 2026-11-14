module

public import Proofs.Flock.SpotFlock.InstanceStmt.Digests
public import Proofs.Flock.SpotFlock.InstanceStmt.Sizes
public import Proofs.Flock.SessionLoop.Concrete.Inputs
public import Proofs.Flock.SessionLoop.Concrete.Stmt
public import Proofs.Flock.SpotFlock.Code
import Flock.HmRow
import Flock.HmNets
import Flock.Canon
import Flock.JsonParse
import Flock.Net
import Flock.ZkServer
import Flock.Norm
import Flock.Tags
import Flock.Zk
import Flock.Registered
import Flock.RecLinksCmd
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
import all Init.Data.String.TakeDrop
import all Init.Data.String.Slice
import all Init.Data.String.Pattern.Basic
import all Init.Data.String.Defs
import all Init.Data.String.Basic
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.List.Control
import all Init.Data.Range.Basic
import all Init.Data.Int.Repr
import all Init.Data.Repr

/-! # The two-instance statement's inputs: `Raw.ins` and its reading

`ins_x` walks `Raw.ins` on `rawOf r ls lh H C` to `insX r ls lh H C`: the tags `Tags.circuit`, tree coins for one table,
the sample `sampleJ` (rebuilt by `norm`, which leaves it as it is), the partition `partObj`, the program's bytes and the
registered reads `regPorts` (`Registered.ofJson regJ`). At the fixed nets' layouts and a run whose strings are plain,
`x r` decodes to that `Raw` (`decode_x`), so the verifier reads it as `rX` (`readOf_x`), under the masked tags `tz`
(`tags_rX`), whose outputs are hidden (`hiddenOf_x`).
-/

@[expose] public section

namespace SpotFlock.InstanceStmt

open Flock Flock.ZkServer RecDeploy.VStarNets FlockVBridge.Names
open Flock.Readable (Read)
open FlockDeploy (Raw Ins jsonOf)
open FlockDeploy.Concrete (noNl utf8_noTen)
open FlockSoundness.Refine
open Properties.Flock.Nets (FixedNetsBuild)
open Lean (Json)

set_option maxHeartbeats 100000000 -- the walk below steps through `Raw.ins` and leaves closed checks to the kernel
set_option maxRecDepth 200000

/-! ## The masked tags -/

theorem bytes_split (ls lh : HmNets.Laid) : bytesOf ls lh =
    ("flock-circuit " ++ "copy16").toByteArray ++ ("\n".toByteArray ++ (("META " ++ metaText).toByteArray ++
      ("\n".toByteArray ++ (joinNl (secs ls lh)).toByteArray))) := by
  simp only [bytesOf, lines, fileLs, secs, joinNl, List.cons_append, List.nil_append, List.append_assoc,
    List.map_cons, String.join_cons, String.toUTF8_eq_toByteArray, String.toByteArray_append, ByteArray.append_assoc]

theorem meta_noTen : (10 : UInt8) ∉ ("META " ++ metaText).toByteArray.data.toList := by
  rw [String.toByteArray_append, ByteArray.data_append, Array.toList_append, List.mem_append, not_or]
  refine ⟨by decide +kernel, ?_⟩
  have h := utf8_noTen (jsonChars metaJ) (noNl metaJ plain_metaJ)
  rw [String.toUTF8_eq_toByteArray] at h
  exact h

theorem mask_meta : (metaJ.getObjValAs? Nat "mask_log").toOption.getD 14 = 14 := by decide +kernel

theorem maskLog_bytes (ls lh : HmNets.Laid) : Zk.maskLogOf (bytesOf ls lh) = 14 := by
  rw [bytes_split, FlockDeploy.Concrete.maskLog_of _ _ _ FlockDeploy.Concrete.first_noTen meta_noTen
    (by rw [drop_meta]; exact parse_meta)]
  exact mask_meta

/-! ## `Raw.ins` -/

theorem byName_t0 : Tags.byName? NAME = some t0 := by kernel_rfl
theorem plain_t0 : t0.plainLeaves = false := by decide +kernel

theorem part_plain : plain partObj = true := by
  simp only [partObj, progHex, HmRow.PARTITION_V1]
  plain_json
theorem fjnorm_part : FlockSoundness.Discharge.Recursion.ZkServerJson.norm partObj = partObj := by kernel_rfl

/-- **The partition file reads back** to `partObj`. -/
theorem jsonOf_part (what : String) : jsonOf what (printJson partObj).toUTF8 = .ok partObj :=
  jsonOf_print part_plain fjnorm_part what

/-- The registered reads' ports, as `Registered.ofJson` reads `regJ`. -/
def regPorts : Array Registered.Port := (Registered.ofJson regJ).toOption.getD #[]

theorem ofJson_reg : Registered.ofJson regJ = .ok regPorts := by kernel_rfl

/-- `Option.mapM` of a file that is there, as `Raw.ins` reads the sample, the partition and the registered reads:
unfolded, it is a `Functor.map` with a projection at its head, which no rule matches. Its side condition is walked. -/
theorem optMapM_ok {α β ε : Type} {f : α → Except ε β} {a : α} {b : β} (h : f a = .ok b) :
    (some a).mapM f = .ok (some b) := by
  rw [Option.mapM_some, h]
  rfl

theorem map_norm_sample : (some sampleJ).map Flock.norm = some sampleJ := congrArg some norm_sample

/-- What the verifier reads of the statement. -/
def rX (r : Run) (ls lh : HmNets.Laid) (H C : ByteArray) : Read :=
  { I := { tags := t0, circuitFile := bytesOf ls lh, publicFile := pubOf r (hexOf H) (hexOf C),
           coins := .tree (Zk.coinSpec (t0.tableNames 1)), count := 1, tables := [], partition := some partObj,
           program := some progBytes, work := none, stratified := none, closure := none },
    zk := true, draw := some sampleJ, publicInputs := none, registered := true }

/-- The statement's inputs: its tags, the registered reads and the reading. -/
def insX (r : Run) (ls lh : HmNets.Laid) (H C : ByteArray) : Ins :=
  { stmtTags := t0, registered := some regPorts, read := rX r ls lh H C }

theorem ins_x (r : Run) (ls lh : HmNets.Laid) (H C : ByteArray) :
    (rawOf r ls lh H C).ins = .ok (insX r ls lh H C) := by
  fwd_walk [byName_t0, plain_t0, optMapM_ok, jsonOf_sample, map_norm_sample, norm_sample, jsonOf_part, jsonOf_reg,
      ofJson_reg]
    block [HmNets.check, HmNets.shaLaid, HmNets.hmLaid, canon, canonUtf8, Net.parse, Registered.ofJson, Flock.norm,
      FlockDeploy.jsonOf, printJson]
    shallow [metaText]

theorem tags_rX (r : Run) (ls lh : HmNets.Laid) (H C : ByteArray) : Read.tags (rX r ls lh H C) = tz := by
  show Zk.tags t0 (Zk.maskLogOf (bytesOf ls lh)) 1 = tz
  rw [maskLog_bytes]
  rfl

/-! ## The verifier's reading of `x r` -/

/-- The statement's inputs at the fixed nets' layouts and the circuit file's two digests. -/
def ins (r : Run) : Ins := insX r (HmNets.shaLaid ()) (HmNets.hmLaid ()) circuitHash classHash

theorem insOf_x (hn : FixedNetsBuild) {r : Run} (hr : r.Plain) : FlockDeploy.insOf true (x r) = some (ins r) :=
  FlockDeploy.insOf_of (decode_x hn hr) rfl (ins_x r _ _ _ _)

theorem readOf_x (hn : FixedNetsBuild) {r : Run} (hr : r.Plain) :
    FlockDeploy.readOf true (x r) = some (ins r).read :=
  (congrArg (Option.map Ins.read) (insOf_x hn hr)).trans rfl

/-- **The statement's outputs are hidden**: it reads under the masked tags of `Tags.circuit`. -/
theorem hiddenOf_x (hn : FixedNetsBuild) {r : Run} (hr : r.Plain) : SpotFlock.hiddenOf (x r) = true := by
  unfold SpotFlock.hiddenOf
  rw [readOf_x hn hr]
  dsimp only [ins, insX]
  rw [tags_rX]
  decide +kernel

end SpotFlock.InstanceStmt
