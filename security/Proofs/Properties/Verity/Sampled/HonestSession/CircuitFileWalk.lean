module

public import Proofs.Properties.Verity.Sampled.HonestSession.CircuitFileMeta
public import Proofs.Properties.Verity.Sampled.HonestSession.Hex
public import Proofs.Flock.SessionLoop.Concrete.File
public import Proofs.Flock.SessionLoop.Concrete.Inputs
public import Proofs.Flock.Soundness.ExecStrataOf
import Flock.HmRow
import Flock.Net
import Flock.HmNets
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
import Flock.Canon
import Flock.JsonParse
import Flock.RecOpen
import Flock.RecResiduals
import Flock.RecSparse
import Flock.RecOpening
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
import all Init.Data.String.Pattern.Pred
import all Init.Data.String.Subslice
import all Init.Data.String.Search
import all Init.Data.String.Iterate
import all Init.Data.String.Decode
import all Init.Data.String.PosRaw
import all Init.Data.String.FindPos
import all Init.Data.String.Termination
import all Init.Data.String.ToSlice
import all Init.Data.String.Bootstrap
import all Init.Data.String.Iter
import all Init.Data.String.Iterator
import all Init.Data.String.Substring
import all Init.Data.String.Modify
import all Init.Data.String.Extra
import all Init.Data.String.Legacy

/-!
# `HmRow.parse` on the circuit file

The walk (`Concrete/Fwd.lean`'s `fwd_def`) over the circuit file with its four digests that only a check reads as
variables (`metaPairsG`: the dummy row's leaf string, the partition, template and program digests), so that no step asks
the kernel for a digest.
-/

@[expose] public section
open Flock RecDeploy.VStarNets FlockVBridge.Names
open FlockDeploy.Concrete (trim_of startsWith_append drop_append getD_toArray getElem!_toArray lines_if tokens3 getD3
  toNat?_toString forIn_yield forIn_done bne_self unit_eq splitOnAux_eq sparse9 getD_mid extract_mid circuit_unit
  circuit_sha circuit_hm netLs_length unit_lines)
open Properties.Flock.Nets (built sparseOf)
open Lean (Json)
open FlockSoundness.Discharge.Recursion

set_option maxHeartbeats 100000000 -- each `fwd_def` walks the circuit file's parse over its literal lines, minutes apiece
set_option maxRecDepth 200000

namespace Verity.Sampled.Instance.HonestSession.CircuitFileProof

/-- `metaPairs` with the dummy row's leaf string `bD` and the partition, template and program digests `bP`, `bT`, `bS`. -/
def metaPairsG (bD bP bT bS : ByteArray) : List (String × Json) := [
  ("dummy", Json.mkObj [("bc", .arr #[.str (hexOf bD)]), ("outputs", .arr #[jn 0])]),
  ("format", .str "flock-circuit"),
  ("input_ports", jn 0),
  ("k_log", jn 22),
  ("leaf_scheme", Flock.Tags.leafSchemeHm96Os),
  ("leaves_in", .arr #[.arr #[.num ⟨-1, 0⟩]]),
  ("leaves_out", .arr #[.arr #[jn 0]]),
  ("mask_log", jn 14),
  ("out_ports", .arr #[.arr #[.str "y", jn 1]]),
  ("partition", Json.mkObj [("digest", .str (hexOf bP)), ("rule", .str HmRow.PARTITION_V1)]),
  ("per_vu", Json.mkObj [("hm96", jn 1), ("sha512x3", jn 1), ("unit", jn 1)]),
  ("ports", .arr #[.arr #[.str "y", jn 64, jn 1]]),
  ("program_digests", Json.mkObj [(templateId, .str (hexOf bT))]),
  ("program_sha512", .str (hexOf bS)),
  ("ranges", .arr #[
    Json.mkObj [("count", jn 1), ("net", .str "hm96"), ("pos0", jn 0), ("slot_log", jn 18), ("type", .str "unit")],
    Json.mkObj [("count", jn 1), ("net", .str "sha512x3"), ("pos0", jn 1), ("slot_log", jn 18), ("type", .str "unit")],
    Json.mkObj [("count", jn 1), ("net", .str "unit"), ("pos0", jn 1024), ("slot_log", jn 9), ("type", .str "unit")],
    Json.mkObj [("count", jn 1), ("pos0", jn 64), ("slot_log", jn 14), ("type", .str "mask")]]),
  ("row_key_sha512", .str (hexOf (Sha512.hash Hm96.Default512.key))),
  ("row_schemas", .arr #[.arr #[.str HmRow.ROW_SCHEMA_V2, jn 1024]]),
  ("rows", .arr #[Json.mkObj [("midstate", .arr ((HmRow.midstateOf yPort).map fun w => Json.str (HmRow.hex16 w))),
    ("pad", .str (hexOf (HmRow.tailOf yPort))), ("prefix", .str (hexOf (HmRow.prefixOf yPort)))]]),
  ("scheme", .str "frame-v3-sha512/hm96-sha512"),
  ("statement", .str Flock.Tags.circuit.name),
  ("template", .str templateId),
  ("units_per_vu", jn 1),
  ("vus_per_block", jn 1),
  ("wires", .arr #[.arr #[.str "hm96", jn 0, jn 0, .str "sha512x3", jn 0, jn 0]])]

def metaJG (bD bP bT bS : ByteArray) : Json := Json.mkObj (metaPairsG bD bP bT bS)
def metaTextG (bD bP bT bS : ByteArray) : String := String.ofList (jsonCharsS (metaJG bD bP bT bS))
def linesG (ls lh : HmNets.Laid) (bD bP bT bS : ByteArray) : List String :=
  fileLs unitName (metaTextG bD bP bT bS) FlockDeploy.Concrete.unitLaid ls lh
def bytesG (ls lh : HmNets.Laid) (bD bP bT bS : ByteArray) : ByteArray := (joinNl (linesG ls lh bD bP bT bS)).toUTF8

theorem metaPairs_eq : metaPairs = metaPairsG dummyBc partitionDigestB templateDigest programSha := rfl
theorem metaJ_eq : metaJ = metaJG dummyBc partitionDigestB templateDigest programSha := rfl
theorem circuitFile_eqG :
    circuitFile = bytesG (HmNets.shaLaid ()) (HmNets.hmLaid ()) dummyBc partitionDigestB templateDigest programSha := rfl

/-- `verify --zk`'s tags at the circuit file. -/
def tZ : Tags := Zk.tags Flock.Tags.circuit 14 1

/-! ## The META line -/

section
variable (bD bP bT bS : ByteArray)

theorem plainS_metaJG : SignedJson.plainS (metaJG bD bP bT bS) = true := by
  unfold metaJG metaPairsG
  refine plainS_mkObj ?pw ?_
  case pw => first | decide +kernel | simp
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
  and_intros
  all_goals (first | exact plainS_leafOs | plainT)

theorem parse_metaG : JsonParse.parse (metaTextG bD bP bT bS) = .ok (FirewallJson.norm (metaJG bD bP bT bS)) :=
  SignedJson.parse_jsonCharsS (plainS_metaJG bD bP bT bS)

theorem last_brace (l : List Char) : ('{' :: (l ++ ['}'])).getLast? = some '}' := by
  rw [← List.cons_append]
  exact List.getLast?_concat

theorem metaG_last : (jsonCharsS (metaJG bD bP bT bS)).getLast? = some '}' := by
  rw [metaJG, FirewallObj.mkObj_eq]
  exact last_brace _

theorem trim_metaG : ("META " ++ metaTextG bD bP bT bS).trimAsciiEnd.toString = "META " ++ metaTextG bD bP bT bS := by
  apply trim_of
  simp only [String.toList_append, metaTextG, String.toList_ofList, List.getLast?_append, metaG_last]
  rfl

theorem starts_metaG : ("META " ++ metaTextG bD bP bT bS).startsWith "META " = true := startsWith_append _ _
theorem drop_metaG : (("META " ++ metaTextG bD bP bT bS).drop 5).toString = metaTextG bD bP bT bS :=
  drop_append "META " _

theorem noNl_metaG : NoNl (metaTextG bD bP bT bS) := by
  unfold NoNl metaTextG
  rw [String.toList_ofList]
  exact noNlS _ (plainS_metaJG bD bP bT bS)

end

/-! ## The file's lines -/

theorem noNl_unitName : NoNl unitName := by unfold NoNl unitName; decide

theorem lines1G (ls lh : HmNets.Laid) (bD bP bT bS : ByteArray) :
    (linesG ls lh bD bP bT bS).getD 1 "" = "META " ++ metaTextG bD bP bT bS := rfl

theorem unit_netLs_length : (netLs unitName FlockDeploy.Concrete.unitLaid).length = 258 := by
  rw [netLs_length]; rfl

section
variable (ls lh : HmNets.Laid) (bD bP bT bS : ByteArray) (hls : VStar.netLines ls = 256002)
  (hlh : VStar.netLines lh = 246786)
include hls hlh

theorem linesG_eq : linesG ls lh bD bP bT bS =
    ["flock-circuit " ++ unitName, "META " ++ metaTextG bD bP bT bS] ++
      (("CIRCUIT unit " ++ toString 258) :: netLs unitName FlockDeploy.Concrete.unitLaid) ++
      (("CIRCUIT sha512x3 " ++ toString 256002) :: netLs "sha512x3" ls) ++
      (("CIRCUIT hm96 " ++ toString 246786) :: netLs "hm96" lh) := by
  simp only [linesG, fileLs, secLs, VStar.SHA_NET, VStar.HM_NET, unit_lines, hls, hlh]
  simp

theorem size_linesG : (linesG ls lh bD bP bT bS).toArray.size = 503051 := by
  rw [linesG_eq ls lh bD bP bT bS hls hlh]
  simp only [List.size_toArray, List.length_append, List.length_cons, netLs_length, unit_lines, hls, hlh]
  rfl

theorem line_unitG (i : Nat) (hi : i = 2) :
    (linesG ls lh bD bP bT bS).getD i "" = "CIRCUIT" ++ " " ++ "unit" ++ " " ++ toString 258 := by
  rw [linesG_eq ls lh bD bP bT bS hls hlh]
  rw [show ["flock-circuit " ++ unitName, "META " ++ metaTextG bD bP bT bS] ++
      (("CIRCUIT unit " ++ toString 258) :: netLs unitName FlockDeploy.Concrete.unitLaid) ++
      (("CIRCUIT sha512x3 " ++ toString 256002) :: netLs "sha512x3" ls) ++
      (("CIRCUIT hm96 " ++ toString 246786) :: netLs "hm96" lh) =
      ["flock-circuit " ++ unitName, "META " ++ metaTextG bD bP bT bS] ++
      (("CIRCUIT unit " ++ toString 258) :: netLs unitName FlockDeploy.Concrete.unitLaid) ++
      ((("CIRCUIT sha512x3 " ++ toString 256002) :: netLs "sha512x3" ls) ++
      (("CIRCUIT hm96 " ++ toString 246786) :: netLs "hm96" lh)) by simp]
  exact (getD_mid _ _ _ _ _ i (by simp [hi])).trans circuit_unit

theorem line_shaG (i : Nat) (hi : i = 261) :
    (linesG ls lh bD bP bT bS).getD i "" = "CIRCUIT" ++ " " ++ "sha512x3" ++ " " ++ toString 256002 := by
  rw [linesG_eq ls lh bD bP bT bS hls hlh]
  exact (getD_mid _ _ _ _ _ i (by simp [hi, unit_netLs_length, unit_lines])).trans circuit_sha

theorem line_hmG (i : Nat) (hi : i = 256264) :
    (linesG ls lh bD bP bT bS).getD i "" = "CIRCUIT" ++ " " ++ "hm96" ++ " " ++ toString 246786 := by
  rw [linesG_eq ls lh bD bP bT bS hls hlh, ← List.append_nil (_ ++ _ :: netLs "hm96" lh)]
  exact (getD_mid _ _ _ _ _ i (by simp [hi, unit_netLs_length, netLs_length, unit_lines, hls])).trans circuit_hm

theorem ext_unitG (a b : Nat) (ha : a = 3) (hb : b = 261) :
    (linesG ls lh bD bP bT bS).toArray.extract a b = (netLs unitName FlockDeploy.Concrete.unitLaid).toArray := by
  rw [linesG_eq ls lh bD bP bT bS hls hlh]
  rw [show ["flock-circuit " ++ unitName, "META " ++ metaTextG bD bP bT bS] ++
      (("CIRCUIT unit " ++ toString 258) :: netLs unitName FlockDeploy.Concrete.unitLaid) ++
      (("CIRCUIT sha512x3 " ++ toString 256002) :: netLs "sha512x3" ls) ++
      (("CIRCUIT hm96 " ++ toString 246786) :: netLs "hm96" lh) =
      (["flock-circuit " ++ unitName, "META " ++ metaTextG bD bP bT bS] ++ ["CIRCUIT unit " ++ toString 258]) ++
      netLs unitName FlockDeploy.Concrete.unitLaid ++
      ((("CIRCUIT sha512x3 " ++ toString 256002) :: netLs "sha512x3" ls) ++
      (("CIRCUIT hm96 " ++ toString 246786) :: netLs "hm96" lh)) by simp]
  exact extract_mid _ _ _ a b (by simp [ha]) (by simp [hb, unit_netLs_length, unit_lines])

theorem ext_shaG (a b : Nat) (ha : a = 262) (hb : b = 256264) :
    (linesG ls lh bD bP bT bS).toArray.extract a b = (netLs "sha512x3" ls).toArray := by
  rw [linesG_eq ls lh bD bP bT bS hls hlh]
  rw [show ["flock-circuit " ++ unitName, "META " ++ metaTextG bD bP bT bS] ++
      (("CIRCUIT unit " ++ toString 258) :: netLs unitName FlockDeploy.Concrete.unitLaid) ++
      (("CIRCUIT sha512x3 " ++ toString 256002) :: netLs "sha512x3" ls) ++
      (("CIRCUIT hm96 " ++ toString 246786) :: netLs "hm96" lh) =
      (["flock-circuit " ++ unitName, "META " ++ metaTextG bD bP bT bS] ++
        (("CIRCUIT unit " ++ toString 258) :: netLs unitName FlockDeploy.Concrete.unitLaid) ++
        ["CIRCUIT sha512x3 " ++ toString 256002]) ++ netLs "sha512x3" ls ++
      (("CIRCUIT hm96 " ++ toString 246786) :: netLs "hm96" lh) by simp]
  exact extract_mid _ _ _ a b (by simp [ha, unit_netLs_length, unit_lines]) (by simp [hb, unit_netLs_length, netLs_length, unit_lines, hls])

theorem ext_hmG (a b : Nat) (ha : a = 256265) (hb : b = 503051) :
    (linesG ls lh bD bP bT bS).toArray.extract a b = (netLs "hm96" lh).toArray := by
  rw [linesG_eq ls lh bD bP bT bS hls hlh]
  rw [show ["flock-circuit " ++ unitName, "META " ++ metaTextG bD bP bT bS] ++
      (("CIRCUIT unit " ++ toString 258) :: netLs unitName FlockDeploy.Concrete.unitLaid) ++
      (("CIRCUIT sha512x3 " ++ toString 256002) :: netLs "sha512x3" ls) ++
      (("CIRCUIT hm96 " ++ toString 246786) :: netLs "hm96" lh) =
      (["flock-circuit " ++ unitName, "META " ++ metaTextG bD bP bT bS] ++
        (("CIRCUIT unit " ++ toString 258) :: netLs unitName FlockDeploy.Concrete.unitLaid) ++
        (("CIRCUIT sha512x3 " ++ toString 256002) :: netLs "sha512x3" ls) ++ ["CIRCUIT hm96 " ++ toString 246786]) ++
      netLs "hm96" lh ++ [] by simp]
  exact extract_mid _ _ _ a b (by simp [ha, unit_netLs_length, netLs_length, unit_lines, hls])
    (by simp [hb, unit_netLs_length, netLs_length, unit_lines, hls, hlh])
end

theorem hutfG (ls lh : HmNets.Laid) (bD bP bT bS : ByteArray) :
    String.fromUTF8? (bytesG ls lh bD bP bT bS) = some (joinNl (linesG ls lh bD bP bT bS)) :=
  fromUTF8?_toUTF8 _
theorem hsplitG (ls lh : HmNets.Laid) (bD bP bT bS : ByteArray) :
    (joinNl (linesG ls lh bD bP bT bS)).splitOn "\n" = linesG ls lh bD bP bT bS ++ [""] :=
  splitOn_join _ (fileLs_noNl _ _ _ _ _ noNl_unitName (noNl_metaG bD bP bT bS))

/-! ## The unit net -/

def UNH : Net :=
  { built unitName 257 ⟨0, 1, #[16]⟩ ⟨1, 1, #[16]⟩ 1 (sparse9 FlockDeploy.Concrete.unitLaid.ra)
      (sparse9 FlockDeploy.Concrete.unitLaid.rb) with unitLog := 9 }

theorem unit_netOfH : HmNets.netOf unitName FlockDeploy.Concrete.unitLaid = .ok UNH := by kernel_rfl'

theorem tok_unitName : FlockDeploy.ConcreteNet.Tok unitName := ⟨by decide, by simp [NoWs, unitName]⟩

theorem parse_unitH : Net.parse (netLs unitName FlockDeploy.Concrete.unitLaid).toArray = .ok UNH :=
  FlockDeploy.ConcreteNet.parse_netLs tok_unitName (by simp [FlockDeploy.Concrete.unitLaid])
    (by simp [FlockDeploy.Concrete.unitLaid]) unit_netOfH

theorem split_lH : unitName.splitOn "/l" = [unitName] := by
  unfold unitName String.splitOn
  simp (config := {decide := true}) only [ite_false]
  repeat (rw [splitOnAux_eq]; simp (config := {decide := true}) only [ite_false])
theorem split_mH : unitName.splitOn "/m" = [unitName] := by
  unfold unitName String.splitOn
  simp (config := {decide := true}) only [ite_false]
  repeat (rw [splitOnAux_eq]; simp (config := {decide := true}) only [ite_false])
theorem split_nH : unitName.splitOn "/n" = [unitName] := by
  unfold unitName String.splitOn
  simp (config := {decide := true}) only [ite_false]
  repeat (rw [splitOnAux_eq]; simp (config := {decide := true}) only [ite_false])

theorem open_noneH : RecOpen.params unitName = none := by unfold RecOpen.params; rw [split_lH]
theorem res_noneH : RecResiduals.params unitName = none := by unfold RecResiduals.params; rw [split_mH]
theorem sparse_noneH : RecSparse.params unitName = none := by unfold RecSparse.params; rw [split_nH]
theorem opening_noneH : RecOpening.params unitName = none := by unfold RecOpening.params; rw [split_mH]

/-! ## Reading META's digests -/

theorem optD (x d : String) : (Except.ok x : Except String String).toOption.getD d = x := rfl

theorem lk_prog (bD bP bT bS : ByteArray) :
    (FirewallJson.norm (metaJG bD bP bT bS)).getObjValAs? String "program_sha512" = .ok (hexOf bS) := by kernel_rfl'

theorem lk_part (bD bP bT bS : ByteArray) :
    (((FirewallJson.norm (metaJG bD bP bT bS)).getObjVal? "partition").toOption.getD Json.null).getObjValAs? String
      "digest" = .ok (hexOf bP) := by kernel_rfl'

theorem sha_hex {b : ByteArray} (h : b.size = 64) : HmRow.isSha512Hex (hexOf b) = true := Hex.isSha512Hex_hexOf h

def lsJG (bD bP bT bS : ByteArray) : Json :=
  match J.get (FirewallJson.norm (metaJG bD bP bT bS)) "leaf_scheme" "" with | .ok j => j | .error _ => .null
def pinJZ : Json := tZ.pinnedLeafScheme.getD .null

theorem qsort_le1 {α : Type} (as : Array α) (lt : α → α → Bool) (h : as.size ≤ 1) : as.qsort lt = as := by
  have hp := FlockSoundness.Audit.StrataExec.qsort_perm lt as
  apply Array.ext'
  rcases as with ⟨l⟩
  match l, h, hp with
  | [], _, hp => exact hp.eq_nil
  | [_], _, hp => exact List.perm_singleton.1 hp
  | _ :: _ :: _, h, _ => simp at h

abbrev shaB (A B : Sparse) : Net := built "sha512x3" 256001 ⟨0, 64, HmNets.SHA_IN⟩ ⟨1984, 16, HmNets.SHA_OUT⟩ 64 A B
abbrev hmB (A B : Sparse) : Net := built "hm96" 246785 ⟨0, 32, HmNets.HM_IN⟩ ⟨1920, 8, HmNets.HM_OUT⟩ 32 A B

/-- `HmRow.parse` reads the dummy row's digests with `Array.mapM`, whose loop the walk does not step. -/
theorem mapM_one {α β ε : Type} (f : α → Except ε β) (a : Array α) (h : a.size = 1) :
    a.mapM f = (f (a[0]'(by rw [h]; exact Nat.one_pos)) >>= fun y => pure #[y]) := by
  obtain ⟨l⟩ := a
  match l, h with
  | [x], _ =>
    show Array.mapM f #[x] = (f x >>= fun y => pure #[y])
    simp [Array.mapM_eq_mapM_toList]

fwd_def cG parse_cG (ls lh : HmNets.Laid) (bD bP bT bS H C : ByteArray) (A B A' B' : Sparse) (s1 : String)
    (hH : Sha512.hash (bytesG ls lh bD bP bT bS) = H)
    (hC : HmRow.classOf (bytesG ls lh bD bP bT bS) ("flock-circuit " ++ unitName)
      (FirewallJson.norm (metaJG bD bP bT bS)) = .ok C)
    (hD : bD.size = 128) (hP : bP.size = 64) (hS : bS.size = 64)
    (hls : VStar.netLines ls = 256002) (hlh : VStar.netLines lh = 246786)
    (hps : Net.parse (netLs "sha512x3" ls).toArray = .ok (shaB A B))
    (hph : Net.parse (netLs "hm96" lh).toArray = .ok (hmB A' B'))
    (hc1 : canon (lsJG bD bP bT bS) = .ok s1) (hc2 : canon pinJZ = .ok s1)
    (hck : ∀ c : Flock.Circuit, (c.nets.getD c.shaNet default).2 = shaB A B → (c.nets.getD c.hmNet default).2 = hmB A' B' →
      HmNets.check c = .ok ()) :
    HmOut.parse tZ (bytesG ls lh bD bP bT bS) [] by
  [hH, hC, hutfG ls lh bD bP bT bS, hsplitG ls lh bD bP bT bS, lines_if _, getD_toArray, getElem!_toArray,
    lines1G ls lh bD bP bT bS, trim_metaG bD bP bT bS, starts_metaG bD bP bT bS, drop_metaG bD bP bT bS,
    parse_metaG bD bP bT bS, trim_of, size_linesG ls lh bD bP bT bS hls hlh, line_unitG ls lh bD bP bT bS hls hlh,
    line_shaG ls lh bD bP bT bS hls hlh, line_hmG ls lh bD bP bT bS hls hlh, tokens3, getD3, toNat?_toString,
    ext_unitG ls lh bD bP bT bS hls hlh, ext_shaG ls lh bD bP bT bS hls hlh, ext_hmG ls lh bD bP bT bS hls hlh,
    parse_unitH, hps, hph, forIn_yield, forIn_done, hc1, hc2, bne_self, hck, unit_eq, open_noneH, res_noneH,
    sparse_noneH, opening_noneH, lk_prog bD bP bT bS, lk_part bD bP bT bS, optD, sha_hex hS, sha_hex hP,
    qsort_le1, Flock.Hex.unhex_hexOf, hD, mapM_one]
  block [HmNets.check, HmNets.shaLaid, HmNets.hmLaid, HmNets.sha512x3Net, HmNets.hm96Net, canon, canonUtf8, Net.parse,
    HmRow.classOf]
  shallow [metaTextG]

end Verity.Sampled.Instance.HonestSession.CircuitFileProof
