module

public import Proofs.Flock.SpotFlock.InstanceStmt.Header
public import Proofs.Flock.SessionLoop.Concrete.File
public import Proofs.Flock.SessionLoop.Concrete.Inputs
import Flock.HmRow
import Flock.Canon
import Flock.JsonParse
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
import all Init.Data.String.TakeDrop
import all Init.Data.String.Slice
import all Init.Data.String.Pattern.Basic
import all Init.Data.String.Pattern.String
import all Init.Data.String.Pattern.Char
import all Init.Data.String.Search
import all Init.Data.String.Iterate
import all Init.Data.String.Decode
import all Init.Data.String.Termination
import all Init.Data.String.Extra

/-! # The two-instance statement's META line, and its circuit's class

META (`metaJ`) is a plain JSON object, so its line prints as it parses (`parse_meta`), ends in `}` and holds no
newline. The circuit's class (`HmRow.classOf`, SHA-512 of the file with META's bindings removed) is `classOfB ls lh`
at any layouts of the fixed nets (`classOf_eq`), a SHA-512 output (`classOfB_hash`): the file's first two newlines
end its head and META's line (`findByte_head`, `findByte_meta`), and META without its bindings is canonical JSON.
-/

@[expose] public section
open Flock Flock.Firewall RecDeploy.VStarNets
open FlockVBridge.Names (NoNl)
open FlockDeploy.Concrete (trim_of startsWith_append drop_append objT noNl findByte_eq IntJ canonUtf8_eq)
open FlockSoundness.Discharge.Recursion.FirewallJson (parse_printJson)
open FlockSoundness.Refine
open Lean (Json)

set_option maxHeartbeats 0 -- the `kernel_rfl` steps below evaluate META's tree
set_option maxRecDepth 200000

namespace SpotFlock.InstanceStmt

/-! ## META -/

theorem metaJ_obj : metaJ = .obj ⟨⟨objT metaJ⟩⟩ := by kernel_rfl
theorem plain_metaJ : plain metaJ = true := by kernel_rfl
theorem fjnorm_metaJ : FlockSoundness.Discharge.Recursion.FirewallJson.norm metaJ = metaJ := by kernel_rfl

theorem parse_meta : JsonParse.parse metaText = .ok metaJ := by
  rw [metaText, parse_printJson plain_metaJ, fjnorm_metaJ]

theorem metaText_last : metaText.toList.getLast? = some '}' := by
  rw [metaText, printJson, String.toList_ofList, metaJ_obj]
  simp [jsonChars]

theorem trim_meta : ("META " ++ metaText).trimAsciiEnd.toString = "META " ++ metaText := by
  apply trim_of
  simp only [String.toList_append, List.getLast?_append, metaText_last]
  rfl

theorem starts_meta : ("META " ++ metaText).startsWith "META " = true := startsWith_append _ _
theorem drop_meta : (("META " ++ metaText).drop 5).toString = metaText := drop_append "META " metaText

theorem noNl_meta : NoNl metaText := by
  unfold NoNl metaText printJson
  rw [String.toList_ofList]
  exact noNl metaJ plain_metaJ

/-! ## The circuit's class -/

/-- The file's sections after META. -/
def secs (ls lh : HmNets.Laid) : List String :=
  secLs "unit" "copy16" FlockDeploy.Concrete.unitLaid ++ secLs VStar.SHA_NET VStar.SHA_NET ls ++
    secLs VStar.HM_NET VStar.HM_NET lh

theorem nl_bytes : "\n".toByteArray.data.toList = [10] := by decide +kernel
theorem head_len : ("flock-circuit " ++ "copy16").toByteArray.data.toList.length = 20 := by decide +kernel
theorem head_none : ("flock-circuit " ++ "copy16").toByteArray.data.toList.findIdx? (· == (10 : UInt8)) = none := by
  decide +kernel

theorem bytes_cons (a : String) (L : List String) :
    (joinNl (a :: L)).toByteArray.data.toList = a.toByteArray.data.toList ++ 10 :: (joinNl L).toByteArray.data.toList := by
  rw [joinNl_cons, String.toByteArray_append, String.toByteArray_append, ByteArray.data_append, ByteArray.data_append,
    Array.toList_append, Array.toList_append, nl_bytes, List.append_assoc, List.singleton_append]

/-- The file's bytes: the head's, a newline, META's line, a newline, then the sections'. -/
theorem bytesOf_list (ls lh : HmNets.Laid) : (bytesOf ls lh).data.toList =
    ("flock-circuit " ++ "copy16").toByteArray.data.toList ++
      10 :: (("META " ++ metaText).toByteArray.data.toList ++ 10 :: (joinNl (secs ls lh)).toByteArray.data.toList) := by
  rw [bytesOf, String.toUTF8_eq_toByteArray]
  show (joinNl (("flock-circuit " ++ "copy16") :: ("META " ++ metaText) :: secs ls lh)).toByteArray.data.toList = _
  rw [bytes_cons, bytes_cons]

theorem findByte_head (ls lh : HmNets.Laid) : findByte (bytesOf ls lh) 10 = some 20 := by
  rw [findByte_eq, bytesOf_list, List.drop_zero, List.findIdx?_append, head_none, Option.none_or, List.findIdx?_cons]
  simp [head_len]

theorem findByte_meta (ls lh : HmNets.Laid) : ∃ n2, findByte (bytesOf ls lh) 10 21 = some n2 := by
  rw [findByte_eq, bytesOf_list]
  rw [← List.singleton_append, ← List.append_assoc, List.drop_left' (by simp [head_len])]
  cases hx : (("META " ++ metaText).toByteArray.data.toList ++
      10 :: (joinNl (secs ls lh)).toByteArray.data.toList).findIdx? (· == (10 : UInt8)) with
  | some i => exact ⟨i + 21, rfl⟩
  | none => exact absurd (List.findIdx?_eq_none_iff.1 hx 10 (by simp)) (by simp)

theorem bind_canon_ok {f : String → ByteArray} {j : Json} (h : IntJ j = true) :
    ∃ m, (canonUtf8 j >>= fun s => pure (Sha512.hash (f s)) : V ByteArray) = .ok (Sha512.hash m) :=
  ⟨_, by rw [canonUtf8_eq h]; rfl⟩

theorem classOf_hash (ls lh : HmNets.Laid) :
    ∃ m, HmRow.classOf (bytesOf ls lh) "flock-circuit copy16" metaJ = .ok (Sha512.hash m) := by
  obtain ⟨n2, h2⟩ := findByte_meta ls lh
  rw [metaJ_obj]
  unfold HmRow.classOf
  simp only [findByte_head ls lh, h2]
  exact bind_canon_ok (by kernel_rfl)

/-- **The circuit's class** is `classOfB ls lh`, at any layouts of the fixed nets. -/
theorem classOf_eq (ls lh : HmNets.Laid) :
    HmRow.classOf (bytesOf ls lh) "flock-circuit copy16" metaJ = .ok (classOfB ls lh) := by
  obtain ⟨m, hm⟩ := classOf_hash ls lh
  rw [classOfB, hm]
  rfl

theorem classOfB_hash (ls lh : HmNets.Laid) : ∃ m, classOfB ls lh = Sha512.hash m := by
  obtain ⟨m, hm⟩ := classOf_hash ls lh
  exact ⟨m, by rw [classOfB, hm]; rfl⟩

end SpotFlock.InstanceStmt
