module

public import Circuits.Program
import Proofs.Flock.SessionLoop.Concrete.Fwd
import Std.Data.HashMap.Lemmas
import Std.Data.HashSet.Lemmas
import Std.Data.String.ToNat
import all Circuits.Program
import all Base.Canon
import all Base.JsonParse
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
# The decoder's answer on the Program of no node

`decode_empty`: `Flock.Program.decode` reads `programEmpty`, a descriptor whose root has no node, as the Program
`progEmpty`, a term, with no hypothesis. `fwd_def` (`Concrete/Fwd.lean`) walks `decode` on its bytes and adds both.

The kernel runs most of `decode` on a closed text, and the walk's rules are the steps it does not:
* the descriptor's parse, through `JsonParse.parseList` on its characters (`parse_empty`);
* the definitions by well-founded recursion on `Json` or `Ty` (`domainError`, by the fuelled check `cleanF`;
  `Ty.ofJson` and `Ty.leaves`, by one unfolding each);
* the lookups in hash maps keyed by strings, whose hash the kernel does not compute (`ddefs_rule`, `index_rule`,
  `busy_rule`);
* the string calls `contains`, `all`, `splitOn` and `toNat!`;
* the decoder's state monad applied to its state (`st_bind` and the rest).
-/

namespace Flock.Program.Closed

open Lean (Json JsonNumber)

/-- The characters of `programEmpty`. -/
@[expose] public def emptyChars : List Char :=
  ['{', '"', 'a', 'n', 'n', 'o', 't', 'a', 't', 'i', 'o', 'n', 's', '"', ':', '{', '}', ',', '"', 'c', 'a', 'l', 'l',
   'e', 'e', 's', '"', ':', '[', ']', ',', '"', 'd', 'e', 'f', 'i', 'n', 'i', 't', 'i', 'o', 'n', 's', '"', ':', '{',
   '"', 'R', 'o', 'o', 't', '_', 'v', '1', '"', ':', '{', '"', 'b', 'i', 'n', 'd', 'i', 'n', 'g', 's', '"', ':', '{',
   '}', ',', '"', 'b', 'o', 'd', 'y', '"', ':', '{', '"', 'n', 'o', 'd', 'e', 's', '"', ':', '[', ']', ',', '"', 'p',
   'a', 'r', 'a', 'm', 's', '"', ':', '[', ']', ',', '"', 'r', 'e', 't', '"', ':', '{', '"', 'c', 'o', 'n', 'c', 'a',
   't', '"', ':', '[', ']', '}', ',', '"', 'r', 'e', 't', '_', 't', 'y', 'p', 'e', '"', ':', '[', '"', 'T', 'u', 'p',
   'l', 'e', '"', ',', '[', ']', ']', '}', ',', '"', 'f', 'o', 'r', 'm', '"', ':', '"', 'c', 'o', 'm', 'p', 'o', 's',
   'i', 't', 'e', '"', ',', '"', 'i', 'd', '"', ':', '"', 'R', 'o', 'o', 't', '_', 'v', '1', '"', '}', '}', ',', '"',
   'p', 'r', 'o', 'g', 'r', 'a', 'm', '_', 'f', 'n', '"', ':', '"', 'R', 'o', 'o', 't', '_', 'v', '1', '"', ',', '"',
   'r', 'o', 'o', 't', '"', ':', '"', 'R', 'o', 'o', 't', '_', 'v', '1', '"', ',', '"', 's', 'c', 'h', 'e', 'm', 'a',
   '"', ':', '"', 'v', 'e', 'r', 'i', 't', 'y', '-', 'i', 'r', '/', 'd', 'e', 's', 'c', 'r', 'i', 'p', 't', 'o', 'r',
   '/', 'v', '1', '"', ',', '"', 't', 'y', 'p', 'e', 's', '"', ':', '[', ']', '}']

/-- A descriptor whose root has no node. -/
@[expose] public def programEmpty : String := String.ofList emptyChars

/-- The parse of `programEmpty`. -/
def jE : Json := (JsonParse.parseList emptyChars).toOption.getD .null

theorem parseList_empty : JsonParse.parseList emptyChars = .ok jE := by kernel_rfl'

theorem parse_empty : JsonParse.parse programEmpty = .ok jE := by
  rw [JsonParse.parse_eq, programEmpty, String.toList_ofList]; exact parseList_empty

theorem fromUTF8?_toUTF8 (s : String) : String.fromUTF8? s.toUTF8 = some s := by
  simp only [String.fromUTF8?, String.toUTF8_eq_toByteArray, s.isValidUTF8, ↓reduceDIte]; rfl

theorem contains_root : "Root_v1".contains '{' = false := by
  rw [String.contains_char_eq]; decide +kernel

theorem toNat_one : "1".toNat! = 1 := by
  have r : Nat.repr 1 = "1" := by apply String.toList_inj.1; rw [Nat.toList_repr]; decide +kernel
  have h : "1".toNat? = some 1 := r ▸ Nat.toNat?_repr 1
  simp only [String.toNat!, String.Slice.toNat!, String.toNat?, String.Slice.toNat?] at h ⊢
  split at h <;> simp_all

theorem splitOnAux_eq (s sep : String) (b i j : String.Pos.Raw) (r : List String) :
    s.splitOnAux sep b i j r =
      if i.atEnd s = true then (b.extract s i :: r).reverse
      else if (i.get s == j.get sep) = true then
        if (j.next sep).atEnd sep = true then
          s.splitOnAux sep (i.next s) (i.next s) 0 (b.extract s ((i.next s).unoffsetBy (j.next sep)) :: r)
        else s.splitOnAux sep b (i.next s) (j.next sep) r
      else s.splitOnAux sep b ((i.unoffsetBy j).next s) 0 r := String.splitOnAux.eq_1 s sep b i j r

theorem split_root : "Root_v1".splitOn "_v" = ["Root", "1"] := by
  unfold String.splitOn
  simp (config := {decide := true}) only [ite_false]
  repeat (rw [splitOnAux_eq]; simp (config := {decide := true}) only [ite_false, ite_true])

/-- Where `domainError` finds nothing, read with fuel: a check the kernel runs. -/
def cleanF : Nat → Json → Bool
  | 0, _ => false
  | n + 1, j => match j with
    | .null | .bool _ => true
    | .num m => m.exponent == 0 && !(m.mantissa.natAbs > MAX_SAFE_INT)
    | .str s => s.toList.all (fun c : Char => c.toNat < 0x7f)
    | .arr a => a.toList.all (cleanF n)
    | .obj kvs => kvs.toArray.toList.all fun x => x.1.toList.all (fun c : Char => c.toNat < 0x7f) && cleanF n x.2

theorem cleanF_sound : ∀ (n : Nat) (j : Json) (p : String), cleanF n j = true → domainError j p = none
  | 0, _, _, h => by simp [cleanF] at h
  | n + 1, j, p, h => by
    cases j with
    | null => rw [domainError]
    | bool b => rw [domainError]
    | num m =>
      simp only [cleanF, Bool.and_eq_true, beq_iff_eq, Bool.not_eq_true', decide_eq_false_iff_not] at h
      rw [domainError]; simp [h.1, h.2]
    | str s =>
      simp only [cleanF] at h
      rw [domainError]; simp [h]
    | arr a =>
      simp only [cleanF, List.all_eq_true] at h
      rw [domainError, Array.findSome?_eq_none_iff]
      intro i _
      exact cleanF_sound n _ _ (h _ (Array.getElem_mem_toList _))
    | obj kvs =>
      simp only [cleanF, List.all_eq_true, Bool.and_eq_true] at h
      rw [domainError, Array.findSome?_eq_none_iff]
      intro x _
      have hx := h x.1 (Array.mem_toList_iff.2 x.2)
      simp only [List.all_eq_true.2 hx.1, ↓reduceIte]
      exact cleanF_sound n _ _ hx.2

theorem ofJson_tuple0 (j : Json) (h : j = .arr #[.str "Tuple", .arr #[]]) : Ty.ofJson j = .ok (.tuple #[]) := by
  subst h; rw [Ty.ofJson]; kernel_rfl'

theorem leaves_rule (t : Ty) (h : t = .tuple #[]) : t.leaves = 0 := by subst h; rw [Ty.leaves]; rfl

theorem domain_none (j : Json) (p : String) (h : cleanF 64 j = true) : domainError j p = none := cleanF_sound 64 j p h

theorem split_rule (s : String) (h : s = "Root_v1") : s.splitOn "_v" = ["Root", "1"] := h ▸ split_root

theorem toNat_rule (s : String) (h : s = "1") : s.toNat! = 1 := h ▸ toNat_one

theorem rfi_rule (s : String) (h : s = "Root_v1") : recordFunctionIds s = #[] := by
  subst h; unfold recordFunctionIds recordPairs; rw [contains_root]; rfl

/-- The root's definition in `programEmpty`. -/
def rootE : Json :=
  ((jE.getObjVal? "definitions").toOption.bind fun d => (d.getObjVal? "Root_v1").toOption).getD .null

theorem ddefs_rule (m : Std.HashMap String Json) (hm : m = (∅ : Std.HashMap String Json).insert "Root_v1" rootE)
    (k : String) (hk : k = "Root_v1") : m[k]? = some rootE := by
  subst hm hk; exact Std.HashMap.getElem?_insert_self

theorem index_rule (k : String) : (∅ : Std.HashMap String Nat)[k]? = none := Std.HashMap.getElem?_empty

theorem busy_rule (k : String) : (∅ : Std.HashSet String).contains k = false := Std.HashSet.contains_empty

/-- The decoder's monad: its state over its refusals. -/
abbrev DM := StateT DecState (Except String)

theorem st_run {α} (x : DM α) (s : DecState) : StateT.run x s = x s := rfl
theorem st_bind {α β} (x : DM α) (f : α → DM β) (s : DecState) : (x >>= f) s = (x s >>= fun p => f p.1 p.2) := rfl
theorem st_get (s : DecState) : (get : DM DecState) s = .ok (s, s) := rfl
theorem st_set (s s' : DecState) : (set s' : DM PUnit) s = .ok (⟨⟩, s') := rfl
theorem st_modify (f : DecState → DecState) (s : DecState) : (modify f : DM PUnit) s = .ok (⟨⟩, f s) := rfl
theorem st_pure {α} (a : α) (s : DecState) : (pure a : DM α) s = .ok (a, s) := rfl
theorem st_ite {α} (c : Prop) [Decidable c] (t e : DM α) (s : DecState) :
    (if c then t else e) s = if c then t s else e s := by split <;> rfl
theorem st_forIn_nil {α β} (xs : Array α) (h : xs = #[]) (b : β) (f : α → β → DM (ForInStep β)) (s : DecState) :
    (forIn xs b f : DM β) s = .ok (b, s) := by subst h; rfl
theorem st_forIn_range0 {β} (m : Nat) (h : m = 0) (b : β) (f : Nat → β → DM (ForInStep β)) (s : DecState) :
    (forIn [0:m] b f : DM β) s = .ok (b, s) := by
  subst h; rw [Std.Legacy.Range.forIn_eq_forIn_range']; rfl

set_option maxHeartbeats 0 in -- the walk steps through `decode` and leaves closed checks to the kernel
fwd_def progEmpty decode_empty : decode programEmpty.toUTF8 by
  [fromUTF8?_toUTF8, parse_empty, domain_none, ofJson_tuple0, split_rule, toNat_rule, rfi_rule, ddefs_rule,
    index_rule, busy_rule, String.contains_char_eq, String.all_bool_eq, Array.mapM_empty,
    st_run, st_bind, st_get, st_set, st_modify, st_pure, st_ite,
    st_forIn_nil, st_forIn_range0, leaves_rule]
  block [domainError, Ty.ofJson]
  shallow [programEmpty]

end Flock.Program.Closed
