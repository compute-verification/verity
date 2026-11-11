module

public import Proofs.Flock.SessionLoop.Concrete.File
public import Proofs.Flock.Recursive.FirewallJson
import Flock.HmRow
import Flock.HmNets
import Flock.Canon
import Flock.JsonParse
import Flock.Net
import Flock.Firewall
import Flock.Norm
import Flock.Public
import Flock.Statement
import Flock.Comp
import Flock.Bytes
import Flock.FV3
import Flock.Hm96
import Flock.Hash
import Flock.RowLeaf
import Flock.Circuit
import Flock.Tags
import Flock.Zk
import Flock.CoinTree
import Flock.Merkle
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

/-! # The concrete statement's public file, and what `HmRow.loadPublic` and `HmRow.checkPublic` make of it

The public file (`pubOf`) is the header `hdrJ` (`circuit_sha512`, the frame-v3-sha512 roots of the one input row and the
one output word, the units entry), printed by `printJson`, then a newline and the input row's digest `bc ‖ c` with the
output word. `load_pubV` walks `HmRow.loadPublic` on it to `pubV`; the header's text reads back (`R1`–`R5`) because
`printJson` of a plain JSON value holds no newline and parses back (`FirewallJson.parse_printJson`). `checkPublic_cW`
holds its roots to the rows: `checkPublicP` is `checkPublic` with the key's digest, the tree leaf and the zero row as
parameters, so that the kernel derives the key and its SHA-512 once (`keyHex_eq`, `lp_eq`, `sd_eq`) rather than in every
leaf.
-/

@[expose] public section
open Flock Flock.Firewall RecDeploy.VStarNets FlockVBridge.Names
open Lean (Json JsonNumber)
open Std.DTreeMap.Internal (Impl)

set_option maxHeartbeats 0 -- the walk below steps through `HmRow.loadPublic` and leaves closed checks to the kernel
set_option maxRecDepth 200000

namespace FlockDeploy.Concrete

def jn (n : Nat) : Json := .num ⟨n, 0⟩
def xPort : Port := { name := "x", words := 64, chunks := 1, nb := 0 }
def bc : ByteArray := HmRow.zeroRowBc xPort
def keyHex : String := hexOf (Sha512.hash Hm96.Default512.key)
def rootX : ByteArray := FV3.root (ByteArray.mk #[0]) #[FV3.leaf (ByteArray.mk #[0]) 0 HmRow.ROW_SCHEMA (Hm96.Default512.treeLeaf bc)]
def rootO : ByteArray := FV3.root (ByteArray.mk #[1]) #[FV3.leaf (ByteArray.mk #[1]) 0 "u16" (ByteArray.mk #[0, 0])]

def unitsJ : Json := Json.mkObj [("blocks", .arr #[.arr #[jn 0]]), ("indices", .arr #[jn 0]), ("units_per_vu", jn 1),
  ("vus_per_block", jn 1)]
def fv3J : Json := Json.mkObj [("domain_ids", Json.mkObj [("out", .str "01"), ("x", .str "00")]),
  ("frame", .str "frame-v3-sha512"), ("roots", Json.mkObj [("out", .str (hexOf rootO)), ("x", .str (hexOf rootX))]),
  ("row_key_sha512", .str keyHex), ("schemas", Json.mkObj [("x", .str "hm96-sha512/row/v1")])]
def restL : List (String × Json) := [("format", .str "flock-circuit-inputs"), ("frame_v3", fv3J), ("instances", jn 1),
  ("rows", .bool false), ("statement", .str "verity/flock-circuit"), ("units", unitsJ)]
/-- The public header: `circuit_sha512` and the closed rest, in key order. -/
def hdrJ (s : String) : Json := Json.mkObj (("circuit_sha512", .str s) :: restL)
def bodyB : ByteArray := bc ++ ByteArray.mk #[0, 0]
def tailB : ByteArray := "\n".toUTF8 ++ bodyB
def pubOf (s : String) : ByteArray := (printJson (hdrJ s)).toUTF8 ++ tailB

theorem get!_mk (l : List UInt8) (k : Nat) (h : k < l.length) : (ByteArray.mk ⟨l⟩).get! k = l[k] := by
  show (⟨l⟩ : Array UInt8)[k]! = l[k]
  simp [h]

theorem findByte_eq (b : ByteArray) (x : UInt8) (f : Nat) :
    findByte b x f = ((b.data.toList.drop f).findIdx? (· == x)).map (· + f) := by
  rcases b with ⟨⟨l⟩⟩
  have key : ∀ n k, l.length - k = n →
      (forIn (m := Id) (List.range' k n 1) ((none : Option (Option Nat)), ()) fun i __s =>
        if ((ByteArray.mk ⟨l⟩).get! i == x) = true then pure (ForInStep.done (some (some i), ()))
        else pure (ForInStep.yield (none, ()))).fst =
        (((l.drop k).findIdx? (· == x)).map (· + k)).map some := by
    intro n
    induction n with
    | zero =>
      intro k hk
      rw [List.drop_eq_nil_of_le (by omega)]
      rfl
    | succ n ih =>
      intro k hk
      have hk' : k < l.length := by omega
      rw [List.range'_succ, List.forIn_cons, List.drop_eq_getElem_cons hk', List.findIdx?_cons, get!_mk l k hk']
      by_cases hx : (l[k] == x) = true
      · rw [if_pos hx, if_pos hx]
        simp only [Option.map_some, Nat.zero_add]
        rfl
      · rw [if_neg hx, if_neg hx]
        show (forIn (m := Id) (List.range' (k + 1) n 1) ((none : Option (Option Nat)), ()) _).fst = _
        rw [ih (k + 1) (by omega)]
        cases List.findIdx? (· == x) (List.drop (k + 1) l) with
        | none => rfl
        | some j => simp only [Option.map_some]; congr 2; omega
  unfold findByte
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size]
  have h1 : ((ByteArray.mk ⟨l⟩).size - f + 1 - 1) / 1 = l.length - f := by
    show (l.length - f + 1 - 1) / 1 = l.length - f
    omega
  rw [h1]
  have := key (l.length - f) f rfl
  revert this
  generalize (forIn (m := Id) (List.range' f (l.length - f) 1) ((none : Option (Option Nat)), ()) fun i __s =>
        if ((ByteArray.mk ⟨l⟩).get! i == x) = true then pure (ForInStep.done (some (some i), ()))
        else pure (ForInStep.yield (none, ()))) = r
  intro h
  rcases r with ⟨o, u⟩
  simp only at h
  subst h
  cases List.findIdx? (· == x) (List.drop f l) <;> rfl

open FlockSoundness.Discharge.Recursion.FirewallJson (toDigits_digits plainStr_chars)

theorem mem_intercalate {c : Char} : ∀ (xs : List (List Char)), c ∈ [','].intercalate xs → c = ',' ∨ ∃ x ∈ xs, c ∈ x
  | [], h => by simp at h
  | [x], h => .inr ⟨x, by simp, by simpa using h⟩
  | x :: y :: ys, h => by
    rw [List.intercalate_cons_cons] at h
    simp only [List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with (h | h) | h
    · exact .inr ⟨x, by simp, h⟩
    · exact .inl h
    · rcases mem_intercalate (y :: ys) h with h | ⟨z, hz, hc⟩
      · exact .inl h
      · exact .inr ⟨z, List.mem_cons_of_mem _ hz, hc⟩

theorem str_noNl {s : String} (h : plainStr s = true) : '\n' ∉ s.toList := fun hn => by
  have := plainStr_chars h '\n' hn
  revert this
  decide

mutual
theorem noNl : ∀ j : Json, plain j = true → '\n' ∉ jsonChars j
  | .null, _ => by decide
  | .bool true, _ => by decide
  | .bool false, _ => by decide
  | .num ⟨m, _⟩, _ => fun hn => by
    have := toDigits_digits m.toNat '\n' hn
    revert this
    decide
  | .str s, h => by
    simp only [jsonChars, List.mem_cons, List.mem_append, List.not_mem_nil, or_false, not_or]
    and_intros
    all_goals first | decide | exact str_noNl h
  | .arr ⟨a⟩, h => by
    simp only [jsonChars, List.mem_cons, List.mem_append, List.not_mem_nil, or_false, not_or]
    have hi : '\n' ∉ [','].intercalate (jsonCharsList a) := fun hn => by
      rcases mem_intercalate _ hn with hc | ⟨x, hx, hc⟩
      · exact absurd hc (by decide)
      · exact noNlList a h x hx hc
    and_intros
    all_goals first | decide | exact hi
  | .obj ⟨⟨t⟩⟩, h => by
    simp only [plain, Bool.and_eq_true] at h
    simp only [jsonChars, List.mem_cons, List.mem_append, List.not_mem_nil, or_false, not_or]
    have hi : '\n' ∉ [','].intercalate (jsonCharsObj t) := fun hn => by
      rcases mem_intercalate _ hn with hc | ⟨x, hx, hc⟩
      · exact absurd hc (by decide)
      · exact noNlObj t h.1 x hx hc
    and_intros
    all_goals first | decide | exact hi
theorem noNlList : ∀ a : List Json, plainList a = true → ∀ x ∈ jsonCharsList a, '\n' ∉ x
  | [], _ => by simp [jsonCharsList]
  | j :: js, h => by
    simp only [plainList, Bool.and_eq_true] at h
    simp only [jsonCharsList, List.mem_cons]
    rintro x (rfl | hx)
    · exact noNl j h.1
    · exact noNlList js h.2 x hx
theorem noNlObj : ∀ t : Impl String (fun _ => Json), plainObj t = true → ∀ x ∈ jsonCharsObj t, '\n' ∉ x
  | .leaf, _ => by simp [jsonCharsObj]
  | .inner _ k v l r, h => by
    simp only [plainObj, Bool.and_eq_true] at h
    simp only [jsonCharsObj, List.mem_append, List.mem_cons, List.not_mem_nil, or_false]
    rintro x ((hx | rfl) | hx)
    · exact noNlObj l h.1.1.1 x hx
    · simp only [List.mem_cons, List.mem_append, List.not_mem_nil, or_false, not_or]
      and_intros
      all_goals first | decide | exact str_noNl h.1.1.2 | exact noNl v h.1.2
    · exact noNlObj r h.2 x hx
end

def hdrT (s : String) : Impl String (fun _ => Json) :=
  match hdrJ s with
  | .obj ⟨⟨t⟩⟩ => t
  | _ => .leaf

theorem hdr_obj (s : String) : hdrJ s = .obj ⟨⟨hdrT s⟩⟩ := by kernel_rfl
theorem hdr_model (s : String) : (hdrT s).toListModel =
    [⟨"circuit_sha512", .str s⟩, ⟨"format", .str "flock-circuit-inputs"⟩, ⟨"frame_v3", fv3J⟩, ⟨"instances", jn 1⟩,
     ⟨"rows", .bool false⟩, ⟨"statement", .str "verity/flock-circuit"⟩, ⟨"units", unitsJ⟩] := by kernel_rfl
theorem hdr_asc (s : String) : ascending (hdrT s) = true := by kernel_rfl
theorem fj_norm (s : String) : FlockSoundness.Discharge.Recursion.FirewallJson.norm (hdrJ s) = hdrJ s := by kernel_rfl
theorem RN' (s : String) : Flock.norm (hdrJ s) = hdrJ s := by kernel_rfl

/-! ### The hash's hex string -/

theorem loop_eq {β : Type} (f : β → UInt8 → β) (as : ByteArray) (stop : Nat) (h : stop ≤ as.size) :
    ∀ (i j : Nat) (b : β), i + j = stop →
      ByteArray.foldlM.loop (m := Id) (fun b a => pure (f b a)) as stop h i j b =
        ((as.data.toList.drop j).take i).foldl f b := by
  intro i
  induction i with
  | zero => intro j b _; unfold ByteArray.foldlM.loop; split <;> rfl
  | succ i ih =>
    intro j b hij
    unfold ByteArray.foldlM.loop
    have hj : j < stop := by omega
    rw [dite_eq_left_of_eq_true (eq_true hj)]
    have hjs : j < as.data.toList.length := by simp; omega
    rw [List.drop_eq_getElem_cons hjs]
    simp only [List.take_succ_cons, List.foldl_cons]
    rw [← ih (j + 1) _ (by omega)]
    simp [Array.getElem_toList]
    rfl

theorem foldl_eq {β : Type} (f : β → UInt8 → β) (init : β) (as : ByteArray) :
    as.foldl f init = as.data.toList.foldl f init := by
  unfold ByteArray.foldl ByteArray.foldlM
  rw [dite_eq_left_of_eq_true (eq_true (Nat.le_refl _))]
  dsimp only
  rw [loop_eq f as as.size (Nat.le_refl _) (as.size - 0) 0 init (by omega)]
  simp only [List.drop_zero, Nat.sub_zero]
  rw [List.take_of_length_le (by simp)]
  rfl

theorem foldl_push (d₁ d₂ : UInt8 → Char) : ∀ (l : List UInt8) (s : String),
    (l.foldl (fun s x => (s.push (d₁ x)).push (d₂ x)) s).toList = s.toList ++ l.flatMap fun x => [d₁ x, d₂ x]
  | [], s => by simp
  | x :: l, s => by
    rw [List.foldl_cons, foldl_push d₁ d₂ l]
    simp [String.toList_push]

/-- A character the hex string may hold: plain in a JSON string, and no newline. -/
def HexC (c : Char) : Prop := (c != '"' && c != '\\' && 0x20 ≤ c.val) = true ∧ c ≠ '\n'

theorem hexOf_toList (b : ByteArray) :
    (hexOf b).toList = b.data.toList.flatMap fun x => (hexOf ⟨#[x]⟩).toList := by
  unfold hexOf
  rw [foldl_eq, foldl_push]
  simp only [String.toList_empty, List.nil_append]
  congr 1
  funext x
  rw [foldl_eq, foldl_push]
  simp

/-- Decided as a `Bool`: with Batteries in scope, `decide` on `∀ c ∈ l, _` picks
`Char.instDecidableForallOfDecidablePred_batteries`, which runs through every `Char`. -/
theorem hexByte_hexC : ∀ k, k < 256 → ∀ c ∈ (hexOf ⟨#[k.toUInt8]⟩).toList, HexC c := by
  have hb : (List.range 256).all (fun k => (hexOf ⟨#[k.toUInt8]⟩).toList.all fun c =>
      c != '"' && c != '\\' && 0x20 ≤ c.val && c != '\n') = true := by decide +kernel
  intro k hk c hc
  have h := Bool.and_eq_true _ _ ▸ List.all_eq_true.1 (List.all_eq_true.1 hb k (List.mem_range.2 hk)) c hc
  exact ⟨h.1, bne_iff_ne.1 h.2⟩

theorem hexOf_hexC (b : ByteArray) : ∀ c ∈ (hexOf b).toList, HexC c := by
  rw [hexOf_toList]
  simp only [List.mem_flatMap]
  rintro c ⟨x, -, hc⟩
  have := hexByte_hexC x.toNat x.toNat_lt c
  have e : x.toNat.toUInt8 = x := by simp
  rw [e] at this
  exact this hc

theorem plainStr_hexOf (b : ByteArray) : plainStr (hexOf b) = true := by
  rw [plainStr, String.all_bool_eq, List.all_eq_true]
  exact fun c hc => (hexOf_hexC b c hc).1

theorem noNl_hexOf (b : ByteArray) : '\n' ∉ (hexOf b).toList := fun h => (hexOf_hexC b _ h).2 rfl

/-! ### UTF-8: a newline byte is a newline -/

theorem loop_data : ∀ (bs : List UInt8) (r : ByteArray),
    (List.toByteArray.loop bs r).data.toList = r.data.toList ++ bs
  | [], r => by simp [List.toByteArray.loop]
  | b :: bs, r => by
    rw [List.toByteArray.loop, loop_data bs (r.push b)]
    simp [ByteArray.push]

theorem toByteArray_data (bs : List UInt8) : bs.toByteArray.data.toList = bs := by
  rw [List.toByteArray, loop_data]
  rfl

theorem ofNat_ten {n : Nat} (h : (10 : UInt8) = UInt8.ofNat n) : n % 256 = 10 := by
  have := congrArg UInt8.toNat h
  simp [UInt8.toNat_ofNat] at this
  omega

theorem utf8_nl (c : Char) (h : (10 : UInt8) ∈ String.utf8EncodeChar c) : c = '\n' := by
  unfold String.utf8EncodeChar at h
  dsimp only at h
  split at h
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at h
    have := ofNat_ten h
    apply Char.ext
    apply UInt32.toNat_inj.1
    show c.val.toNat = 10
    omega
  · split at h
    · simp only [List.mem_cons, List.not_mem_nil, or_false] at h
      rcases h with h | h <;> have := ofNat_ten h <;> omega
    · split at h
      · simp only [List.mem_cons, List.not_mem_nil, or_false] at h
        rcases h with h | h | h <;> have := ofNat_ten h <;> omega
      · simp only [List.mem_cons, List.not_mem_nil, or_false] at h
        rcases h with h | h | h | h <;> have := ofNat_ten h <;> omega

theorem utf8_noTen (l : List Char) (h : '\n' ∉ l) : (10 : UInt8) ∉ (String.ofList l).toUTF8.data.toList := by
  rw [String.toUTF8_eq_toByteArray, String.toByteArray_ofList, List.utf8Encode, toByteArray_data]
  intro hm
  rcases List.mem_flatMap.1 hm with ⟨c, hc, hc10⟩
  exact h (utf8_nl c hc10 ▸ hc)

/-! ### The header's text -/

open FlockSoundness.Discharge.Recursion.FirewallJson (plainObj_iff parse_printJson)


theorem plain_obj_of {t : Impl String (fun _ => Json)} {L : List ((_ : String) × Json)} (hm : t.toListModel = L)
    (ha : ascending t = true) (h : ∀ e ∈ L, plainStr e.1 = true ∧ plain e.2 = true) : plain (.obj ⟨⟨t⟩⟩) = true := by
  show (plainObj t && ascending t) = true
  rw [ha, Bool.and_true, plainObj_iff, hm]
  exact h

def objT (j : Json) : Impl String (fun _ => Json) :=
  match j with
  | .obj ⟨⟨t⟩⟩ => t
  | _ => .leaf

theorem obj2_obj (a b : String) : Json.mkObj [("out", .str a), ("x", .str b)] =
    .obj ⟨⟨objT (Json.mkObj [("out", .str a), ("x", .str b)])⟩⟩ := by kernel_rfl
theorem obj2_model (a b : String) : (objT (Json.mkObj [("out", .str a), ("x", .str b)])).toListModel =
    [⟨"out", .str a⟩, ⟨"x", .str b⟩] := by kernel_rfl
theorem obj2_asc (a b : String) : ascending (objT (Json.mkObj [("out", .str a), ("x", .str b)])) = true := by kernel_rfl

theorem obj2_plain {a b : String} (ha : plainStr a = true) (hb : plainStr b = true) :
    plain (Json.mkObj [("out", .str a), ("x", .str b)]) = true := by
  rw [obj2_obj]
  refine plain_obj_of (obj2_model a b) (obj2_asc a b) ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
  exact ⟨⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], ha⟩, ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], hb⟩⟩

theorem obj1_obj (a : String) : Json.mkObj [("x", .str a)] = .obj ⟨⟨objT (Json.mkObj [("x", .str a)])⟩⟩ := by kernel_rfl
theorem obj1_model (a : String) : (objT (Json.mkObj [("x", .str a)])).toListModel = [⟨"x", .str a⟩] := by kernel_rfl
theorem obj1_asc (a : String) : ascending (objT (Json.mkObj [("x", .str a)])) = true := by kernel_rfl

theorem obj1_plain {a : String} (ha : plainStr a = true) : plain (Json.mkObj [("x", .str a)]) = true := by
  rw [obj1_obj]
  refine plain_obj_of (obj1_model a) (obj1_asc a) ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq]
  exact ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], ha⟩

theorem units_obj : unitsJ = .obj ⟨⟨objT unitsJ⟩⟩ := by kernel_rfl
theorem units_model : (objT unitsJ).toListModel = [⟨"blocks", .arr #[.arr #[jn 0]]⟩, ⟨"indices", .arr #[jn 0]⟩,
    ⟨"units_per_vu", jn 1⟩, ⟨"vus_per_block", jn 1⟩] := by kernel_rfl
theorem units_asc : ascending (objT unitsJ) = true := by kernel_rfl

theorem units_plain : plain unitsJ = true := by
  rw [units_obj]
  refine plain_obj_of units_model units_asc ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
  refine ⟨⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], by simp [plain, plainList, plainStr, String.all_bool_eq, jn]⟩, ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], by simp [plain, plainList, plainStr, String.all_bool_eq, jn]⟩, ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], by simp [plain, plainList, plainStr, String.all_bool_eq, jn]⟩, ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], by simp [plain, plainList, plainStr, String.all_bool_eq, jn]⟩⟩

theorem fv3_obj : fv3J = .obj ⟨⟨objT fv3J⟩⟩ := by kernel_rfl
theorem fv3_model : (objT fv3J).toListModel =
    [⟨"domain_ids", Json.mkObj [("out", .str "01"), ("x", .str "00")]⟩, ⟨"frame", .str "frame-v3-sha512"⟩,
     ⟨"roots", Json.mkObj [("out", .str (hexOf rootO)), ("x", .str (hexOf rootX))]⟩, ⟨"row_key_sha512", .str keyHex⟩,
     ⟨"schemas", Json.mkObj [("x", .str "hm96-sha512/row/v1")]⟩] := by kernel_rfl
theorem fv3_asc : ascending (objT fv3J) = true := by kernel_rfl

theorem fv3_plain : plain fv3J = true := by
  rw [fv3_obj]
  refine plain_obj_of fv3_model fv3_asc ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
  exact ⟨⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], obj2_plain (by simp [plain, plainList, plainStr, String.all_bool_eq, jn]) (by simp [plain, plainList, plainStr, String.all_bool_eq, jn])⟩, ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], by simp [plain, plainList, plainStr, String.all_bool_eq, jn]⟩,
    ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], obj2_plain (plainStr_hexOf _) (plainStr_hexOf _)⟩, ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], plainStr_hexOf _⟩,
    ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], obj1_plain (by simp [plain, plainList, plainStr, String.all_bool_eq, jn])⟩⟩

theorem hdr_plain (s : String) (hs : plainStr s = true) : plain (hdrJ s) = true := by
  rw [hdr_obj]
  refine plain_obj_of (hdr_model s) (hdr_asc s) ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
  exact ⟨⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], hs⟩, ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], by simp [plain, plainList, plainStr, String.all_bool_eq, jn]⟩, ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], fv3_plain⟩, ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], by simp [plain, plainList, plainStr, String.all_bool_eq, jn]⟩, ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], by simp [plain, plainList, plainStr, String.all_bool_eq, jn]⟩, ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], by simp [plain, plainList, plainStr, String.all_bool_eq, jn]⟩,
    ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], units_plain⟩⟩


theorem hdr_noTen (b : ByteArray) : (10 : UInt8) ∉ (printJson (hdrJ (hexOf b))).toUTF8.data.toList :=
  utf8_noTen _ (noNl _ (hdr_plain _ (plainStr_hexOf b)))

theorem R1 (b : ByteArray) : findByte (pubOf (hexOf b)) 10 = some (printJson (hdrJ (hexOf b))).toUTF8.size := by
  have hn : List.findIdx? (· == (10 : UInt8)) (printJson (hdrJ (hexOf b))).toUTF8.data.toList = none :=
    List.findIdx?_eq_none_iff.2 fun x hx => by
      cases hx10 : (x == 10)
      · rfl
      · exact absurd ((beq_iff_eq.1 hx10) ▸ hx) (hdr_noTen b)
  have h10 : "\n".toUTF8.data.toList = [10] := by decide +kernel
  rw [findByte_eq, pubOf, tailB, ByteArray.data_append, ByteArray.data_append, Array.toList_append,
    Array.toList_append, h10, List.drop_zero, List.findIdx?_append, hn, Option.none_or, List.cons_append,
    List.findIdx?_cons, beq_self_eq_true, if_pos rfl, Option.map_some, Option.map_some, Nat.zero_add, Nat.add_zero,
    Array.length_toList]
  rfl


theorem R2 (b : ByteArray) :
    (pubOf (hexOf b)).extract 0 (printJson (hdrJ (hexOf b))).toUTF8.size = (printJson (hdrJ (hexOf b))).toUTF8 :=
  ByteArray.extract_append_eq_left rfl

theorem R4 (b : ByteArray) : JsonParse.parse (printJson (hdrJ (hexOf b))) = .ok (Flock.norm (hdrJ (hexOf b))) := by
  rw [parse_printJson (hdr_plain _ (plainStr_hexOf b)), fj_norm, RN']

theorem R5 (b : ByteArray) :
    (pubOf (hexOf b)).extract ((printJson (hdrJ (hexOf b))).toUTF8.size + 1) (pubOf (hexOf b)).size = bodyB := by
  rw [pubOf, tailB, ← ByteArray.append_assoc]
  exact ByteArray.extract_append_eq_right (by rw [ByteArray.size_append]; rfl) ByteArray.size_append

theorem RN (b : ByteArray) : Flock.norm (hdrJ (hexOf b)) = hdrJ (hexOf b) := RN' _

fwd_def pubV load_pubV (ls lh : HmNets.Laid) (H : ByteArray) (A B A' B' : Sparse) (s1 s3 s4 : String)
    (hu1 : canon unitsJ = .ok s3) (hu2 : canon (HmRow.unitsEntry (cW ls lh H A B A' B' s1) #[0]) = .ok s3)
    (hd : canonUtf8 (hdrJ (hexOf H)) = .ok s4) :
    HmRow.loadPublic tz (cW ls lh H A B A' B' s1) (pubOf (hexOf H)) by
  [R1 H, R2 H, fromUTF8?_toUTF8, R4 H, R5 H, RN H, bne_self, hu1, hu2, hd]
  block [HmNets.check, HmNets.shaLaid, HmNets.hmLaid, HmNets.sha512x3Net, HmNets.hm96Net, canon, canonUtf8, Net.parse,
    HmRow.parse]
  shallow [metaText]

def LP : ByteArray := ⟨#[118, 101, 114, 105, 116, 121, 47, 104, 109, 57, 54, 45, 115, 104, 97, 53,
  49, 50, 47, 108, 101, 97, 102, 47, 118, 49, 0, 175, 197, 166, 12, 133,
  216, 170, 117, 203, 76, 127, 114, 101, 136, 210, 153, 161, 113, 46, 28, 106,
  98, 31, 9, 202, 100, 128, 154, 95, 237, 215, 68, 160, 246, 176, 32, 29,
  5, 232, 43, 41, 205, 91, 39, 78, 108, 41, 103, 12, 212, 228, 171, 16,
  131, 25, 17, 251, 31, 158, 12, 112, 25, 223, 34, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]⟩
def SD : ByteArray := ⟨#[135, 164, 250, 176, 96, 71, 77, 236, 78, 131, 6, 34, 230, 135, 77, 52,
  114, 54, 39, 64, 219, 217, 124, 21, 84, 185, 74, 159, 87, 38, 223, 136,
  117, 132, 52, 75, 23, 90, 56, 180, 174, 165, 123, 135, 54, 187, 17, 254,
  196, 216, 117, 92, 66, 66, 192, 199, 185, 155, 35, 156, 124, 22, 211, 60]⟩
def BC : ByteArray := ⟨#[14, 144, 57, 56, 27, 187, 106, 34, 209, 161, 124, 204, 211, 120, 136, 123,
  18, 68, 233, 198, 142, 97, 119, 136, 131, 8, 37, 246, 88, 177, 96, 190,
  18, 79, 113, 108, 72, 48, 90, 116, 13, 101, 255, 194, 80, 77, 125, 116,
  0, 9, 158, 120, 108, 200, 139, 153, 83, 5, 51, 74, 144, 17, 144, 223,
  135, 164, 250, 176, 96, 71, 77, 236, 78, 131, 6, 34, 230, 135, 77, 52,
  114, 54, 39, 64, 219, 217, 124, 21, 84, 185, 74, 159, 87, 38, 223, 136,
  117, 132, 52, 75, 23, 90, 56, 180, 174, 165, 123, 135, 54, 187, 17, 254,
  196, 216, 117, 92, 66, 66, 192, 199, 185, 155, 35, 156, 124, 22, 211, 60]⟩

def KH : String :=
  "afc5a60c85d8aa75cb4c7f726588d299a1712e1c6a621f09ca64809a5fedd744a0f6b0201d05e82b29cd5b274e6c29670cd4e4ab10831911fb1f9e0c7019df22"
def RXH : String :=
  "9966711cb66ab7d8ac0105deebcb092f8e16e05b246bad9f3eed7985c4370b1fb10cc879e4fb980d120657df6ef9c6f5a9f74262d8f17bdb5b3fa51ef91d6292"
def ROH : String :=
  "b4310c92cfb83afda323d689bf9ab67c468b3f7c86c7677f8b220ae50b34e993f1f5d1ba7ec9017ea27caf36dbb34f3ff267ef4503de480799e8e739b4fdc31d"

theorem keyHex_eq : hexOf (Sha512.hash Hm96.Default512.key) = KH := by decide +kernel
theorem lp_eq : Hm96.Default512.leafPrefix = LP := by decide +kernel
theorem sd_eq : Sha512.hash (Hm96.Default512.saltPrefix ++ ByteArray.mk (Array.replicate HmRow.SALT_BYTES 0)) = SD := by
  decide +kernel
theorem tl_eq : Hm96.Default512.treeLeaf = fun b => Sha512.hash (LP ++ b) := by
  funext b; unfold Hm96.Default512.treeLeaf; rw [lp_eq]
theorem zr_eq : HmRow.zeroRowBc = fun p =>
    (if p.segs.isSome then HmRow.segDigest p.bits (ByteArray.mk (Array.replicate p.bytes 0))
     else Sha512.hash (HmRow.prefixOf p ++ ByteArray.mk (Array.replicate p.bytes 0))) ++ SD := by
  funext p; unfold HmRow.zeroRowBc; rw [sd_eq]
theorem bc_eq : bc = BC := by
  unfold bc; rw [zr_eq]; decide +kernel
theorem rootX_eq : hexOf rootX = RXH := by
  unfold rootX; rw [tl_eq, bc_eq]; decide +kernel
theorem rootO_eq : hexOf rootO = ROH := by
  unfold rootO; decide +kernel

end FlockDeploy.Concrete

namespace Flock.HmRow
/-- `checkPublic` with the default key's digest in hex, the tree leaf and the zero row as parameters. -/
def checkPublicP (kh : String) (tl : ByteArray → ByteArray) (zr : Port → ByteArray) (c : Circuit) (pub : Public) :
    V Unit := do
  if J.has c.metaJ "links" then throw "META links: a statement with public outputs has no output row to link (links are a hidden-output statement's: `HmOut`)"
  let fv ← pub.frameV3
  if (← J.str fv "frame" "public file") != "frame-v3-sha512"
      || (← J.str fv "row_key_sha512" "public file") != kh then
    throw "the instance file's rows are not frame-v3-sha512 hm96 leaves under the default key"
  let np := c.ports.size
  let (n, bcs, outputs) := pub.population.getD (pub.n, pub.digests, pub.outputs)
  if let .ok d := pub.header.getObjVal? "unit_draw" then
    let u ← Draw.UnitDraw.ofJson d
    if (← J.nat pub.header "instances" "public file") != u.proved.size || pub.n != u.proved.size then
      throw "U3: the statement's instances are not the drawn units and their closure"
  let H := Sha512.hash
  for p in [0:np] do
    let port := c.ports[p]!
    let dom ← pub.fv3Bytes "domain_ids" port.name
    let schema := if port.segs.isSome then ROW_SCHEMA_SEG else if port.v2 then ROW_SCHEMA_V2 else ROW_SCHEMA
    if (← J.str (← J.get fv "schemas" "public file") port.name "public file") != schema then
      throw s!"input port {port.name}: schema is not {schema}"
    let (count, row) : Nat × (Nat → Nat) := match pub.tables with
      | some t => (t.counts[p]!, fun r => t.starts[p]! + r)
      | none => (n, fun i => i * np + p)
    let leaves := (Array.range count).map fun r => FV3.leaf dom r schema (tl (bcs.getD (row r) ByteArray.empty)) H
    if FV3.root dom leaves H != (← pub.fv3Bytes "roots" port.name) then
      throw s!"input port {port.name}: the rows' b || c do not hash to the header's frame-v3-sha512 root"
    if zr port != c.dummyDigests.getD p ByteArray.empty then
      throw s!"the pinned dummy b || c of port {port.name} is not the zero row's"
  let nOut := c.outWords
  let mut at_ := 0
  for (oname, w) in c.outPorts do
    let dom ← pub.fv3Bytes "domain_ids" oname
    let words := (Array.range n).foldl (fun acc i => acc ++ (Array.range w).map fun k => outputs.getD (i * nOut + at_ + k) 0) #[]
    let leaves := (Array.range words.size).map fun r =>
      FV3.leaf dom r "u16" (ByteArray.mk #[(words[r]! / 256).toUInt8, (words[r]! % 256).toUInt8]) H
    if FV3.root dom leaves H != (← pub.fv3Bytes "roots" oname) then
      throw s!"output port {oname}: the output words do not hash to the header's frame-v3-sha512 root"
    at_ := at_ + w

theorem checkPublic_eqP (c : Circuit) (pub : Public) :
    checkPublic c pub = checkPublicP (hexOf (Sha512.hash Hm96.Default512.key)) Hm96.Default512.treeLeaf zeroRowBc c pub :=
  rfl
end Flock.HmRow

namespace FlockDeploy.Concrete
open Flock.HmRow (checkPublicP checkPublic_eqP)

def fv3L : Json := Json.mkObj [("domain_ids", Json.mkObj [("out", .str "01"), ("x", .str "00")]),
  ("frame", .str "frame-v3-sha512"), ("roots", Json.mkObj [("out", .str ROH), ("x", .str RXH)]),
  ("row_key_sha512", .str KH), ("schemas", Json.mkObj [("x", .str "hm96-sha512/row/v1")])]
theorem fv3_eq : fv3J = fv3L := by
  unfold fv3J keyHex fv3L; rw [keyHex_eq, rootX_eq, rootO_eq]
theorem checkPublic_cW (ls lh : HmNets.Laid) (H : ByteArray) (A B A' B' : Sparse) (s1 s3 s4 : String) :
    HmRow.checkPublic (cW ls lh H A B A' B' s1) (pubV ls lh H A B A' B' s1 s3 s4) = .ok () := by
  rw [HmRow.checkPublic_eqP, keyHex_eq, tl_eq, zr_eq]
  unfold pubV
  simp only [bodyB, bc_eq, hdrJ, restL, fv3_eq]
  kernel_rfl

end FlockDeploy.Concrete
