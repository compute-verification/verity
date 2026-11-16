module

public import Proofs.Properties.Verity.Sampled.HonestSession.Data.Circuit
public import Proofs.Properties.Verity.Sampled.HonestSession.SignedJsonRead
public import Proofs.Flock.SessionLoop.Concrete.Public
public import Proofs.Flock.Recursive.ZkServerObj
import Flock.Zk
import Flock.Bytes
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
import all Init.Data.Int.Repr
import all Init.Data.Repr

/-!
# The circuit file's META line

The META line's value is plain (`plainS_metaJ`), so it parses to `norm metaJ` (`parse_meta`) and holds no newline; the
file's first two lines are its head and META lines, which `Zk.maskLogOf` reads to `14` (`maskLog_eq`). No proof here
evaluates a digest: each is a `hexOf`, plain whatever its bytes.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession.CircuitFileProof

open Lean (Json)
open Flock Flock.ZkServer
open FlockSoundness.Discharge.Recursion
open Std.DTreeMap.Internal (Impl)
open SignedJson

/-! ## Plain -/

theorem plainS_mkObj {L : List (String × Json)} (h : L.Pairwise fun a b => a.1 ≠ b.1)
    (hp : ∀ e ∈ L, plainStr e.1 = true ∧ plainS e.2 = true) : plainS (Json.mkObj L) = true := by
  rw [ZkServerObj.mkObj_eq]
  show (plainSObj _ && ascending _) = true
  rw [ZkServerObj.ascending_of_ordered (ZkServerObj.ordered_ofList L), Bool.and_true, plainSObj_iff]
  intro e he
  obtain ⟨e', he', rfl⟩ := List.mem_map.mp ((ZkServerObj.perm_ofList h).mem_iff.mp he)
  exact hp e' he'

theorem plainS_hex (b : ByteArray) : plainS (.str (hexOf b)) = true := FlockDeploy.Concrete.plainStr_hexOf b

theorem plainS_arr1 {j : Json} (h : plainS j = true) : plainS (.arr #[j]) = true := by
  show (plainS j && true) = true
  rw [h]
  rfl

theorem plainS_hexArr (A : Array UInt64) : plainS (.arr (A.map fun w => Json.str (HmRow.hex16 w))) = true := by
  show plainSList (A.map fun w => Json.str (HmRow.hex16 w)).toList = true
  rw [plainSList_iff]
  intro j hj
  rw [Array.toList_map, List.mem_map] at hj
  obtain ⟨w, -, rfl⟩ := hj
  show plainStr (HmRow.hex16 w) = true
  unfold HmRow.hex16
  exact FlockDeploy.Concrete.plainStr_hexOf _

theorem plainS_arr {l : List Json} (h : ∀ j ∈ l, plainS j = true) : plainS (.arr l.toArray) = true :=
  (plainSList_iff l).2 h

/-- Closes `plainS j = true` for a literal `j` whose only non-literal strings are `hexOf`s. -/
syntax "plainT" : tactic
macro_rules
  | `(tactic| plainT) => `(tactic| first
    | exact plainS_hex _
    | exact plainS_hexArr _
    | exact plainS_leafOs
    | (refine plainS_mkObj (by first | decide +kernel | simp) ?_
       simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
       and_intros <;> plainT)
    | (refine plainS_arr ?_
       simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
       and_intros <;> plainT)
    | simp [plainS, plainSList, plainStr, String.all_bool_eq, jn, HmRow.ROW_SCHEMA_V2, HmRow.PARTITION_V1, templateId,
        Tags.circuit])

theorem plainS_leafOs : plainS Tags.leafSchemeHm96Os = true := by
  show plainS (Json.mkObj [("id", .str "hm96-sha512/v1"), ("inner", .str "sha512 column digest"),
    ("salt_bytes", jn 192),
    ("salt_source", .str "os bytes (getrandom), 192 per leaf, each tree's read once, never reused or stored"),
    ("key", .str "default_key, a statement constant for every tree"),
    ("key_sha512", .str (hexOf (Sha512.hash Hm96.Default512.key)))]) = true
  plainT

theorem plainS_metaJ : plainS metaJ = true := by
  unfold metaJ metaPairs
  refine plainS_mkObj ?pw ?_
  case pw => first | decide +kernel | simp
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
  and_intros
  all_goals (first | exact plainS_leafOs | plainT)

/-- **The META line parses** to the parser's reading of `metaJ`. -/
theorem parse_meta : JsonParse.parse metaText = .ok (ZkServerJson.norm metaJ) := parse_jsonCharsS plainS_metaJ

/-! ## No newline -/

theorem repr_noNl (m : Int) : '\n' ∉ (Int.repr m).toList := by
  cases m with
  | ofNat n =>
    rw [repr_ofNat]
    intro hn
    have := ZkServerJson.toDigits_digits n '\n' hn
    revert this
    decide
  | negSucc k =>
    rw [repr_negSucc]
    intro hn
    rcases List.mem_cons.1 hn with h | h
    · exact absurd h (by decide)
    · have := ZkServerJson.toDigits_digits (k + 1) '\n' h
      revert this
      decide

mutual
theorem noNlS : ∀ j : Json, plainS j = true → '\n' ∉ jsonCharsS j
  | .null, _ => by decide
  | .bool true, _ => by decide
  | .bool false, _ => by decide
  | .num ⟨m, _⟩, _ => repr_noNl m
  | .str s, h => by
    simp only [jsonCharsS, List.mem_cons, List.mem_append, List.not_mem_nil, or_false, not_or]
    and_intros
    all_goals first | decide | exact FlockDeploy.Concrete.str_noNl h
  | .arr ⟨a⟩, h => by
    simp only [jsonCharsS, List.mem_cons, List.mem_append, List.not_mem_nil, or_false, not_or]
    have hi : '\n' ∉ [','].intercalate (jsonCharsSList a) := fun hn => by
      rcases FlockDeploy.Concrete.mem_intercalate _ hn with hc | ⟨x, hx, hc⟩
      · exact absurd hc (by decide)
      · exact noNlSList a h x hx hc
    and_intros
    all_goals first | decide | exact hi
  | .obj ⟨⟨t⟩⟩, h => by
    simp only [plainS, Bool.and_eq_true] at h
    simp only [jsonCharsS, List.mem_cons, List.mem_append, List.not_mem_nil, or_false, not_or]
    have hi : '\n' ∉ [','].intercalate (jsonCharsSObj t) := fun hn => by
      rcases FlockDeploy.Concrete.mem_intercalate _ hn with hc | ⟨x, hx, hc⟩
      · exact absurd hc (by decide)
      · exact noNlSObj t h.1 x hx hc
    and_intros
    all_goals first | decide | exact hi
theorem noNlSList : ∀ a : List Json, plainSList a = true → ∀ x ∈ jsonCharsSList a, '\n' ∉ x
  | [], _ => by simp [jsonCharsSList]
  | j :: js, h => by
    simp only [plainSList, Bool.and_eq_true] at h
    simp only [jsonCharsSList, List.mem_cons]
    rintro x (rfl | hx)
    · exact noNlS j h.1
    · exact noNlSList js h.2 x hx
theorem noNlSObj : ∀ t : Impl String (fun _ => Json), plainSObj t = true → ∀ x ∈ jsonCharsSObj t, '\n' ∉ x
  | .leaf, _ => by simp [jsonCharsSObj]
  | .inner _ k v l r, h => by
    simp only [plainSObj, Bool.and_eq_true] at h
    simp only [jsonCharsSObj, List.mem_append, List.mem_cons, List.not_mem_nil, or_false]
    rintro x ((hx | rfl) | hx)
    · exact noNlSObj l h.1.1.1 x hx
    · simp only [List.mem_cons, List.mem_append, List.not_mem_nil, or_false, not_or]
      and_intros
      all_goals first | decide | exact FlockDeploy.Concrete.str_noNl h.1.1.2 | exact noNlS v h.1.2
    · exact noNlSObj r h.2 x hx
end

theorem headLine_noNl : '\n' ∉ headLine.toList := by decide +kernel

theorem metaText_noNl : '\n' ∉ metaText.toList := by
  rw [metaText, String.toList_ofList]
  exact noNlS _ plainS_metaJ

theorem metaLine_noNl : '\n' ∉ ("META " ++ metaText).toList := by
  rw [String.toList_append, List.mem_append, not_or]
  exact ⟨by decide +kernel, metaText_noNl⟩

/-! ## The file's first two lines -/

theorem noTen (a : String) (h : '\n' ∉ a.toList) : (10 : UInt8) ∉ a.toUTF8.data.toList := by
  have := FlockDeploy.Concrete.utf8_noTen a.toList h
  rwa [String.ofList_toList] at this

theorem nl_data : "\n".toUTF8.data.toList = [10] := by decide +kernel

theorem toUTF8_app (s t : String) : (s ++ t).toUTF8 = s.toUTF8 ++ t.toUTF8 := by
  simp only [String.toUTF8_eq_toByteArray, String.toByteArray_append]

theorem size_eq (a : String) : a.toUTF8.size = a.toUTF8.data.toList.length := by
  rw [Array.length_toList]
  rfl

theorem findIdx_nl (A R : List UInt8) (hA : (10 : UInt8) ∉ A) :
    (A ++ (10 : UInt8) :: R).findIdx? (· == 10) = some A.length := by
  have hn : A.findIdx? (· == (10 : UInt8)) = none :=
    List.findIdx?_eq_none_iff.2 fun x hx => by
      cases hx10 : (x == 10)
      · rfl
      · exact absurd ((beq_iff_eq.1 hx10) ▸ hx) hA
  rw [List.findIdx?_append, hn, Option.none_or, List.findIdx?_cons, beq_self_eq_true, if_pos rfl, Option.map_some,
    Nat.zero_add]

theorem data_lines (a b R : String) : (a ++ "\n" ++ (b ++ "\n" ++ R)).toUTF8.data.toList =
    a.toUTF8.data.toList ++ (10 : UInt8) :: (b.toUTF8.data.toList ++ (10 : UInt8) :: R.toUTF8.data.toList) := by
  simp only [toUTF8_app, ByteArray.data_append, Array.toList_append, nl_data, List.append_assoc,
    List.cons_append, List.nil_append]

theorem findByte_head (a b R : String) (ha : '\n' ∉ a.toList) :
    findByte (a ++ "\n" ++ (b ++ "\n" ++ R)).toUTF8 10 = some a.toUTF8.size := by
  rw [FlockDeploy.Concrete.findByte_eq, data_lines, List.drop_zero, findIdx_nl _ _ (noTen a ha), Option.map_some,
    Nat.add_zero, size_eq]

theorem findByte_meta (a b R : String) (ha : '\n' ∉ a.toList) (hb : '\n' ∉ b.toList) :
    findByte (a ++ "\n" ++ (b ++ "\n" ++ R)).toUTF8 10 (a.toUTF8.size + 1) =
      some (a.toUTF8.size + 1 + b.toUTF8.size) := by
  rw [FlockDeploy.Concrete.findByte_eq, data_lines, size_eq a,
    show a.toUTF8.data.toList ++ (10 : UInt8) :: (b.toUTF8.data.toList ++ (10 : UInt8) :: R.toUTF8.data.toList) =
      (a.toUTF8.data.toList ++ [10]) ++ (b.toUTF8.data.toList ++ (10 : UInt8) :: R.toUTF8.data.toList) by simp,
    List.drop_left' (by simp), findIdx_nl _ _ (noTen b hb), Option.map_some, size_eq b]
  congr 1
  omega

theorem extract_meta (a b R : String) :
    (a ++ "\n" ++ (b ++ "\n" ++ R)).toUTF8.extract (a.toUTF8.size + 1) (a.toUTF8.size + 1 + b.toUTF8.size) =
      b.toUTF8 := by
  rw [show (a ++ "\n" ++ (b ++ "\n" ++ R)).toUTF8 = (a ++ "\n").toUTF8 ++ (b.toUTF8 ++ ("\n" ++ R).toUTF8) by
      simp only [toUTF8_app, ByteArray.append_assoc],
    show a.toUTF8.size + 1 = (a ++ "\n").toUTF8.size + 0 by simp [toUTF8_app, ByteArray.size_append]; rfl,
    show (a ++ "\n").toUTF8.size + 0 + b.toUTF8.size = (a ++ "\n").toUTF8.size + b.toUTF8.size by omega,
    ByteArray.extract_append_size_add]
  exact ByteArray.extract_append_eq_left rfl

/-- The circuit file: its head line, its META line, and the rest. -/
theorem circuitFile_eq : ∃ R : String, circuitFile = (headLine ++ "\n" ++ (("META " ++ metaText) ++ "\n" ++ R)).toUTF8 := by
  refine ⟨RecDeploy.VStarNets.joinNl (RecDeploy.VStarNets.secLs "unit" unitName FlockDeploy.Concrete.unitLaid ++
    RecDeploy.VStarNets.secLs VStar.SHA_NET VStar.SHA_NET (HmNets.shaLaid ()) ++
    RecDeploy.VStarNets.secLs VStar.HM_NET VStar.HM_NET (HmNets.hmLaid ())), ?_⟩
  show (RecDeploy.VStarNets.joinNl (["flock-circuit " ++ unitName, "META " ++ metaText] ++
    (RecDeploy.VStarNets.secLs "unit" unitName FlockDeploy.Concrete.unitLaid ++
    RecDeploy.VStarNets.secLs VStar.SHA_NET VStar.SHA_NET (HmNets.shaLaid ()) ++
    RecDeploy.VStarNets.secLs VStar.HM_NET VStar.HM_NET (HmNets.hmLaid ())))).toUTF8 = _
  rw [List.cons_append, List.cons_append, List.nil_append, RecDeploy.VStarNets.joinNl_cons,
    RecDeploy.VStarNets.joinNl_cons]
  rfl

theorem mask_lookup : (ZkServerJson.norm metaJ).getObjValAs? Nat "mask_log" = .ok 14 := by kernel_rfl'

/-- **The META line's `mask_log` is `14`.** -/
theorem maskLog_eq : Zk.maskLogOf circuitFile = 14 := by
  obtain ⟨R, hR⟩ := circuitFile_eq
  rw [hR]
  unfold Zk.maskLogOf
  rw [findByte_head _ _ _ headLine_noNl]
  simp only [findByte_meta _ _ _ headLine_noNl metaLine_noNl, Option.getD_some, extract_meta,
    FlockVBridge.Names.fromUTF8?_toUTF8]
  rw [show ("META " ++ metaText).drop 5 = ("META " ++ metaText).drop "META ".length from rfl,
    FlockDeploy.Concrete.drop_append, parse_meta]
  simp only [mask_lookup]
  rfl

end Verity.Sampled.Instance.HonestSession.CircuitFileProof
