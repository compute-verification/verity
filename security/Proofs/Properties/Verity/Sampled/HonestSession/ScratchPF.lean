module

public import Proofs.Properties.Verity.Sampled.HonestSession.PublicFile
public import Proofs.Flock.SessionLoop.Concrete.Public
public import Proofs.Flock.SessionLoop.Concrete.Inputs
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
import all Init.Data.String.Defs
import all Init.Data.String.Basic
import all Init.Data.String.Extra
import all Init.Data.Array.QSort.Basic
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.List.Control
import all Init.Data.Range.Basic
import all Init.Data.Int.Repr
import all Init.Data.Repr
import all Init.Data.String.TakeDrop
import all Init.Data.String.Slice
import all Init.Data.String.Pattern.Basic
import all Init.Data.String.Pattern.String
import all Init.Data.String.Pattern.Char
import all Init.Data.String.Pattern.Pred
import all Init.Data.String.Subslice
import all Init.Data.String.Search
import all Init.Data.String.Iterate
import all Init.Data.String.Decode
import all Init.Data.String.PosRaw
import all Init.Data.String.FindPos
import all Init.Data.String.Termination
import all Init.Data.String.ToSlice
import all Init.Data.Array.Attach
import all Init.WF
import all Init.Data.Nat.Fold

/-! # Scratch: the public-file walk for `PublicFileFacts`. -/

@[expose] public section

open Flock Flock.Firewall RecDeploy.VStarNets FlockVBridge.Names
open Lean (Json)
open Std.DTreeMap.Internal (Impl)
open FlockSoundness.Discharge.Recursion.FirewallJson (parse_printJson)
open FlockDeploy.Concrete (objT plain_obj_of plainStr_hexOf noNl_hexOf noNl utf8_noTen findByte_eq
  canon_eq canonUtf8_eq canon_obj_congr ok_of_isOk objM IntJ)

set_option maxHeartbeats 0
set_option maxRecDepth 200000

namespace Verity.Sampled.Instance.HonestSession

/-! ## Tags: `Zk.tags` touches only `identity`, so the public-file fields project through to `Tags.circuit`. -/

theorem tagsZ_fmt : tagsZ.publicFormat = Flock.Tags.circuit.publicFormat := rfl
theorem tagsZ_stmt : tagsZ.statement = Flock.Tags.circuit.statement := rfl
theorem tagsZ_pck : tagsZ.publicCircuitKey = "circuit_sha512" := by decide +kernel

/-! ## The class SHA-512 is 64 bytes, so not empty. -/

theorem classSha_ne : classSha.isEmpty = false := by
  unfold classSha
  simp only [ByteArray.isEmpty, Sha512.hash_size]
  decide

/-! ## Parameterized header: the one tainted field (`classes = hexOf classSha`) is abstracted to `sCls`.
The walk runs over `pubF sCls` with `sCls` symbolic (so `printJson (hdrF sCls)` is untainted and the fwd tactic's
constructor recognition survives); the concrete file is `pubF (hexOf classSha)`, recovered by `pubF_eq`. -/

def unitsF (sCls : String) : Json := Json.mkObj
  [("blocks", .arr #[.arr #[jn 0]]), ("classes", .arr #[.str sCls]), ("indices", .arr #[jn 0]),
   ("units_per_vu", jn 1), ("vus_per_block", jn 1)]
def hdrF (sSha sCls : String) : Json := Json.mkObj
  [("circuit_sha512", .str sSha), ("format", .str Flock.Tags.circuit.publicFormat),
   ("frame_v3", fv3J), ("instances", jn nUnits), ("rows", .bool false),
   ("statement", .str Flock.Tags.circuit.statement), ("units", unitsF sCls)]
def pubF (sSha sCls : String) : ByteArray := (printJson (hdrF sSha sCls)).toUTF8 ++ "\n".toUTF8 ++ rowBc

theorem unitsF_eq : unitsF (hexOf classSha) = unitsJ := rfl
theorem hdrF_eq : hdrF (hexOf circuitSha) (hexOf classSha) = hdrJ := rfl
theorem pubF_eq : pubF (hexOf circuitSha) (hexOf classSha) = publicFile := rfl

/-! ## Plainness of the header JSON. -/

theorem objY_plain {a : String} (ha : plainStr a = true) : plain (Json.mkObj [("y", .str a)]) = true := by
  rw [show Json.mkObj [("y", .str a)] = .obj ⟨⟨objT (Json.mkObj [("y", .str a)])⟩⟩ from by kernel_rfl']
  refine plain_obj_of (show (objT (Json.mkObj [("y", .str a)])).toListModel = [⟨"y", .str a⟩] from by kernel_rfl')
    (by kernel_rfl') ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq]
  exact ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], ha⟩

theorem fmt_plain : plain (.str Flock.Tags.circuit.publicFormat) = true := by
  rw [show Flock.Tags.circuit.publicFormat = "flock-circuit-inputs" from by kernel_rfl']
  simp [plain, plainStr, String.all_bool_eq]
theorem stmt_plain : plain (.str Flock.Tags.circuit.statement) = true := by
  rw [show Flock.Tags.circuit.statement = "verity/flock-circuit" from by kernel_rfl']
  simp [plain, plainStr, String.all_bool_eq]

theorem fv3_plain : plain fv3J = true := by
  rw [show fv3J = .obj ⟨⟨objT fv3J⟩⟩ from by kernel_rfl']
  refine plain_obj_of (show (objT fv3J).toListModel =
    [⟨"domain_ids", Json.mkObj [("y", .str (hexOf domY))]⟩, ⟨"frame", .str "frame-v3-sha512"⟩,
     ⟨"roots", Json.mkObj [("y", .str (hexOf rootY))]⟩,
     ⟨"row_key_sha512", .str (hexOf (Sha512.hash Hm96.Default512.key))⟩,
     ⟨"schemas", Json.mkObj [("y", .str HmRow.ROW_SCHEMA_V2)]⟩] from by kernel_rfl') (by kernel_rfl') ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
  refine ⟨⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], objY_plain (plainStr_hexOf _)⟩,
    ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], by simp [plain, plainList, plainStr, String.all_bool_eq, jn]⟩,
    ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], objY_plain (plainStr_hexOf _)⟩,
    ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], plainStr_hexOf _⟩,
    ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn],
      objY_plain (by simp [HmRow.ROW_SCHEMA_V2, plainStr, String.all_bool_eq])⟩⟩

theorem units_plainF (sCls : String) (hs : plainStr sCls = true) : plain (unitsF sCls) = true := by
  rw [show unitsF sCls = .obj ⟨⟨objT (unitsF sCls)⟩⟩ from by kernel_rfl']
  refine plain_obj_of (show (objT (unitsF sCls)).toListModel =
    [⟨"blocks", .arr #[.arr #[jn 0]]⟩, ⟨"classes", .arr #[.str sCls]⟩, ⟨"indices", .arr #[jn 0]⟩,
     ⟨"units_per_vu", jn 1⟩, ⟨"vus_per_block", jn 1⟩] from by kernel_rfl') (by kernel_rfl') ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
  refine ⟨⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], by simp [plain, plainList, plainStr, String.all_bool_eq, jn]⟩,
    ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], by simp only [plain, plainList, Bool.and_true]; exact hs⟩,
    ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], by simp [plain, plainList, plainStr, String.all_bool_eq, jn]⟩,
    ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], by simp [plain, plainStr, String.all_bool_eq, jn]⟩,
    ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], by simp [plain, plainStr, String.all_bool_eq, jn]⟩⟩

theorem hdr_plainF (sSha sCls : String) (hpsSha : plainStr sSha = true) (hs : plainStr sCls = true) :
    plain (hdrF sSha sCls) = true := by
  rw [show hdrF sSha sCls = .obj ⟨⟨objT (hdrF sSha sCls)⟩⟩ from by kernel_rfl']
  refine plain_obj_of (show (objT (hdrF sSha sCls)).toListModel =
    [⟨"circuit_sha512", .str sSha⟩, ⟨"format", .str Flock.Tags.circuit.publicFormat⟩,
     ⟨"frame_v3", fv3J⟩, ⟨"instances", jn nUnits⟩, ⟨"rows", .bool false⟩,
     ⟨"statement", .str Flock.Tags.circuit.statement⟩, ⟨"units", unitsF sCls⟩] from by kernel_rfl') (by kernel_rfl') ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
  refine ⟨⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], hpsSha⟩,
    ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], fmt_plain⟩,
    ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], fv3_plain⟩,
    ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], by simp [plain, plainStr, String.all_bool_eq, jn]⟩,
    ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], by simp [plain, plainList, plainStr, String.all_bool_eq, jn]⟩,
    ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], stmt_plain⟩,
    ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], units_plainF sCls hs⟩⟩

/-! ## The header text reads back (parameterized over the abstract class string). -/

theorem fj_normF (sSha sCls : String) :
    FlockSoundness.Discharge.Recursion.FirewallJson.norm (hdrF sSha sCls) = hdrF sSha sCls := by kernel_rfl'
theorem RNF (sSha sCls : String) : Flock.norm (hdrF sSha sCls) = hdrF sSha sCls := by kernel_rfl'

theorem hdr_noTenF (sSha sCls : String) (hpsSha : plainStr sSha = true) (hs : plainStr sCls = true) :
    (10 : UInt8) ∉ (printJson (hdrF sSha sCls)).toUTF8.data.toList :=
  utf8_noTen _ (noNl _ (hdr_plainF sSha sCls hpsSha hs))

/-- `pubF` re-associated to match the reference's `A ++ (nl ++ body)`. -/
def tailB : ByteArray := "\n".toUTF8 ++ rowBc
theorem pfF_eq (sSha sCls : String) : pubF sSha sCls = (printJson (hdrF sSha sCls)).toUTF8 ++ tailB := by
  unfold pubF tailB; rw [ByteArray.append_assoc]

theorem R1F (sSha sCls : String) (hpsSha : plainStr sSha = true) (hs : plainStr sCls = true) :
    findByte (pubF sSha sCls) 10 = some (printJson (hdrF sSha sCls)).toUTF8.size := by
  have hn : List.findIdx? (· == (10 : UInt8)) (printJson (hdrF sSha sCls)).toUTF8.data.toList = none :=
    List.findIdx?_eq_none_iff.2 fun x hx => by
      cases hx10 : (x == 10)
      · rfl
      · exact absurd ((beq_iff_eq.1 hx10) ▸ hx) (hdr_noTenF sSha sCls hpsSha hs)
  have h10 : "\n".toUTF8.data.toList = [10] := by decide +kernel
  rw [findByte_eq, pfF_eq, tailB, ByteArray.data_append, ByteArray.data_append, Array.toList_append,
    Array.toList_append, h10, List.drop_zero, List.findIdx?_append, hn, Option.none_or, List.cons_append,
    List.findIdx?_cons, beq_self_eq_true, if_pos rfl, Option.map_some, Option.map_some, Nat.zero_add, Nat.add_zero,
    Array.length_toList]
  rfl

theorem R2F (sSha sCls : String) :
    (pubF sSha sCls).extract 0 (printJson (hdrF sSha sCls)).toUTF8.size = (printJson (hdrF sSha sCls)).toUTF8 := by
  rw [pfF_eq]; exact ByteArray.extract_append_eq_left rfl

theorem R4F (sSha sCls : String) (hpsSha : plainStr sSha = true) (hs : plainStr sCls = true) :
    JsonParse.parse (printJson (hdrF sSha sCls)) = .ok (Flock.norm (hdrF sSha sCls)) := by
  rw [parse_printJson (hdr_plainF sSha sCls hpsSha hs), fj_normF, RNF]

theorem R5F (sSha sCls : String) :
    (pubF sSha sCls).extract ((printJson (hdrF sSha sCls)).toUTF8.size + 1) (pubF sSha sCls).size = rowBc := by
  rw [pfF_eq, tailB, ← ByteArray.append_assoc]
  exact ByteArray.extract_append_eq_right (by rw [ByteArray.size_append]; rfl) ByteArray.size_append

/-! ## The classes check and the units-entry canonicalisation (handled through the blocked `canon`). -/

theorem hnunits : nUnits = 1 := rfl
/-- String `!=` reflexivity (the generic `bne_self_eq_false` needs a `ReflBEq` the rewriter cannot discharge here). -/
theorem bneStr (s : String) : (s != s) = false := by simp
/-- The parsed `circuit_sha512` string `sSha` equals `hexOf c.sha`, so the circuit-key check passes -- a targeted
rewrite on the full check term so that `sSha` is *not* rewritten inside the (tainted) header text. -/
theorem hShaChkF (c : Flock.Circuit) (sSha : String) (h : sSha = hexOf c.sha) :
    (sSha != hexOf c.sha) = false := by subst h; simp
/-- The classes array as `mapM asStr` builds it (an `emptyWithCapacity … |>.push`). -/
theorem hclsSizeF (sCls : String) :
    ((Array.emptyWithCapacity #[Json.str sCls].size).push sCls).size = 1 := rfl
/-- The indices array as `mapM asNat` builds it, normalised to `#[0]` so the (concrete) ascending check reduces. -/
theorem idxArr : (Array.emptyWithCapacity #[jn 0].size).push (0 : Nat) = #[0] := rfl
/-- The one parsed class string `sCls` equals `hexOf c.classSha`, so the class check passes -- stated as a targeted
rewrite on the full check term so that `sCls` is *not* rewritten inside the (tainted) header text. -/
theorem hclsAnyF (c : Flock.Circuit) (sCls : String) (h : sCls = hexOf c.classSha) :
    Array.any ((Array.emptyWithCapacity #[Json.str sCls].size).push sCls) (· != hexOf c.classSha) 0 1 = false := by
  subst h; simp

/-! ## The walk. -/

fwd_def pubPF load_pf (c : Flock.Circuit) (sSha sCls s3' s4' : String)
    (hpsSha : plainStr sSha = true) (hps : plainStr sCls = true)
    (hsha : sSha = hexOf c.sha) (hsCls : sCls = hexOf c.classSha)
    (hports : c.ports = #[yPort]) (hg : c.g = 1) (hupv : c.upv = 1) (hout : c.outWords = 0)
    (hce : c.classSha.isEmpty = false)
    (hu1' : canon (unitsF sCls) = .ok s3') (hu2' : canon (HmRow.unitsEntry c #[0]) = .ok s3')
    (hd' : canonUtf8 (hdrF sSha sCls) = .ok s4') :
    HmRow.loadPublic tagsZ c (pubF sSha sCls) by
  [R1F sSha sCls hpsSha hps, R2F sSha sCls, fromUTF8?_toUTF8, R4F sSha sCls hpsSha hps, R5F sSha sCls, RNF sSha sCls,
    bneStr, tagsZ_fmt, tagsZ_stmt, tagsZ_pck, hShaChkF c sSha hsha, hports, hg, hupv, hout, hce,
    hnunits, hclsSizeF sCls, hclsAnyF c sCls hsCls, idxArr, hu1', hu2', hd']
  block [canon, canonUtf8]
  shallow []

end Verity.Sampled.Instance.HonestSession
