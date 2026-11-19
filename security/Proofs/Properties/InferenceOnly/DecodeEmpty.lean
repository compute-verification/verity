module

public import Circuits.Program
import Batteries.Tactic.OpenPrivate
import Proofs.Core.QSort
import Proofs.Flock.SessionLoop.Concrete.Fwd
import Std.Data.HashMap.Lemmas
import Std.Data.HashSet.Lemmas
import Std.Data.String.ToNat
import Base.Canon
import Base.JsonParse
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
  `canon`, by `intsF`, its value left as `canonOf`; `fnRefs` and `subFns` on the empty object; `Ty.ofJson` and
  `Ty.leaves`, by one unfolding each);
* the lookups in hash maps keyed by strings, whose hash the kernel does not compute (`ddefs_rule`, `index_rule`,
  `busy_rule`, `memo_none`, `memo_one`), and the duplicate-name checks on empty lists (`dup_rule`, `names_rule`);
* the string calls `contains`, `all`, `splitOn` and `toNat!`;
* the state monads applied to their state, the decoder's and the digests' memo (`st_bind` and the rest).
-/

namespace Flock.Program.Closed

open Lean (Json JsonNumber)
open private DecState fnRefs from Circuits.Program

/-- The characters of `programEmpty`. -/
@[expose] public def emptyChars : List Char :=
  ['{', '"', 'a', 'n', 'n', 'o', 't', 'a', 't', 'i', 'o', 'n', 's', '"', ':', '{', '}', ',', '"', 'b', 'o', 'o', 'l',
   'e', 'a', 'n', '"', ':', '{', '}', ',', '"', 'c', 'a', 'l', 'l', 'e', 'e', 's', '"', ':', '[', ']', ',', '"', 'd',
   'e', 'f', 'i', 'n', 'i', 't', 'i', 'o', 'n', 's', '"', ':', '{', '"', 'R', 'o', 'o', 't', '_', 'v', '1', '"', ':',
   '{', '"', 'b', 'i', 'n', 'd', 'i', 'n', 'g', 's', '"', ':', '{', '}', ',', '"', 'b', 'o', 'd', 'y', '"', ':', '{',
   '"', 'n', 'o', 'd', 'e', 's', '"', ':', '[', ']', ',', '"', 'p', 'a', 'r', 'a', 'm', 's', '"', ':', '[', ']', ',',
   '"', 'r', 'e', 't', '"', ':', '{', '"', 'c', 'o', 'n', 'c', 'a', 't', '"', ':', '[', ']', '}', ',', '"', 'r', 'e',
   't', '_', 't', 'y', 'p', 'e', '"', ':', '[', '"', 'T', 'u', 'p', 'l', 'e', '"', ',', '[', ']', ']', '}', ',', '"',
   'f', 'o', 'r', 'm', '"', ':', '"', 'c', 'o', 'm', 'p', 'o', 's', 'i', 't', 'e', '"', ',', '"', 'i', 'd', '"', ':',
   '"', 'R', 'o', 'o', 't', '_', 'v', '1', '"', '}', '}', ',', '"', 'i', 'n', 'p', 'u', 't', 's', '"', ':', '[', ']',
   ',', '"', 'p', 'r', 'o', 'g', 'r', 'a', 'm', '_', 'f', 'n',
   '"', ':', '"', 'R', 'o', 'o', 't', '_', 'v', '1', '"', ',', '"', 'r', 'o', 'o', 't', '"', ':', '"', 'R', 'o', 'o',
   't', '_', 'v', '1', '"', ',', '"', 's', 'c', 'h', 'e', 'm', 'a', '"', ':', '"', 'v', 'e', 'r', 'i', 't', 'y', '-',
   'i', 'r', '/', 'd', 'e', 's', 'c', 'r', 'i', 'p', 't', 'o', 'r', '/', 'v', '2', '"', ',', '"', 't', 'y', 'p', 'e',
   's', '"', ':', '[', ']', '}']

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

theorem dup_rule (ps : Array (String × Ty)) (h : ps = #[]) :
    ((ps.foldl (fun s p => s.insert p.1) (∅ : Std.HashSet String)).size != ps.size) = false := by
  subst h; simp

theorem names_rule (names : Array String) (h : names = #[]) :
    (names.contains "" ||
      (names.foldl (fun s x => s.insert x) (∅ : Std.HashSet String)).size != names.size) = false := by
  subst h; simp

theorem index_rule (k : String) : (∅ : Std.HashMap String Nat)[k]? = none := Std.HashMap.getElem?_empty

theorem busy_rule (k : String) : (∅ : Std.HashSet String).contains k = false := Std.HashSet.contains_empty

theorem toArray_nil : (∅ : Std.TreeMap.Raw String Json compare).toArray = #[] := by
  first | rfl | decide | simp [Std.TreeMap.Raw.toArray, Std.DTreeMap.Raw.toArray]

theorem fnRefs_leaf (acc : Array String) : fnRefs acc (.obj ∅) = acc := by
  rw [fnRefs]
  have hq : ∀ (lt : _ → _ → Bool), (∅ : Std.TreeMap.Raw String Json compare).toArray.attach.qsort lt = #[] :=
    fun lt => Array.eq_empty_of_size_eq_zero (by rw [Verity.QSort.size_qsort, Array.size_attach, toArray_nil]; rfl)
  simp only [hq]
  rfl

theorem refs_rule (c : Array Json) (e : Json) (hb : e.getObjValD "bindings" = .obj ∅)
    (hn : (e.getObjValD "body").getObjValD "nodes" = .arr #[]) : entryRefs c e = #[] := by
  simp only [entryRefs]
  rw [hb, hn, fnRefs_leaf]
  split <;> rfl

theorem refs_root : entryRefs #[] rootE = #[] := refs_rule #[] rootE (by kernel_rfl') (by kernel_rfl')

theorem subFns_rule (d : String → Option String) (j : Json) (h : j = .obj ∅) :
    subFns d j = .ok (Json.mkObj []) := by
  subst h
  have ha : (∅ : Std.TreeMap.Raw String Json compare).toArray.attach = #[] :=
    Array.eq_empty_of_size_eq_zero (by rw [Array.size_attach, toArray_nil]; rfl)
  rw [subFns]
  simp only [ha, toArray_nil, Array.size_empty, Nat.reduceBEq, Bool.false_and, Bool.false_eq_true, ↓reduceIte,
    Array.mapM_empty]
  first | rfl | simp

theorem list_mapM_ok {α β : Type} (f : α → V β) :
    ∀ (l : List α), (∀ x ∈ l, ∃ t, f x = .ok t) → ∃ r, l.mapM f = .ok r
  | [], _ => ⟨[], rfl⟩
  | x :: l, h => by
    obtain ⟨t, ht⟩ := h x List.mem_cons_self
    obtain ⟨r, hr⟩ := list_mapM_ok f l (fun y hy => h y (List.mem_cons_of_mem _ hy))
    exact ⟨t :: r, by rw [List.mapM_cons, ht]; show (do let as ← l.mapM f; pure (t :: as)) = _; rw [hr]; rfl⟩

theorem array_mapM_ok {α β : Type} (f : α → V β) (xs : Array α) (h : ∀ x ∈ xs, ∃ t, f x = .ok t) :
    ∃ r, xs.mapM f = .ok r := by
  rw [Array.mapM_eq_mapM_toList]
  obtain ⟨r, hr⟩ := list_mapM_ok f xs.toList (fun x hx => h x (Array.mem_toList_iff.1 hx))
  exact ⟨r.toArray, by rw [hr]; rfl⟩

/-- Where `canon` succeeds, read with fuel: every number an integer, the strings never read. -/
def intsF : Nat → Json → Bool
  | 0, _ => false
  | n + 1, j => match j with
    | .num m => m.exponent == 0
    | .arr a => a.toList.all (intsF n)
    | .obj kvs => kvs.toArray.toList.all fun x => intsF n x.2
    | _ => true

/-- `canon` succeeds where `intsF` holds. -/
theorem canon_ok : ∀ (n : Nat) (j : Json), intsF n j = true → ∃ s, canon j = .ok s
  | 0, _, h => by simp [intsF] at h
  | n + 1, j, h => by
    cases j with
    | null => rw [canon]; exact ⟨_, rfl⟩
    | bool b => rw [canon]; exact ⟨_, rfl⟩
    | str s => rw [canon]; exact ⟨_, rfl⟩
    | num m =>
      simp only [intsF, beq_iff_eq] at h
      rw [canon]; simp only [h, beq_self_eq_true, ↓reduceIte]; exact ⟨_, rfl⟩
    | arr a =>
      simp only [intsF, List.all_eq_true] at h
      rw [canon]
      obtain ⟨parts, hp⟩ := array_mapM_ok (fun x : {x // x ∈ a} => canon x.1) a.attach
        (fun x _ => canon_ok n x.1 (h x.1 (Array.mem_toList_iff.2 x.2)))
      exact ⟨_, by rw [hp]; rfl⟩
    | obj kvs =>
      simp only [intsF, List.all_eq_true] at h
      rw [canon]
      obtain ⟨vals, hv⟩ := array_mapM_ok (fun p : {x // x ∈ kvs.toArray} => do pure (p.1.1, ← canon p.1.2))
        kvs.toArray.attach (fun p _ => by
          obtain ⟨t, ht⟩ := canon_ok n p.1.2 (h p.1 (Array.mem_toList_iff.2 p.2))
          exact ⟨(p.1.1, t), by simp only [ht]; rfl⟩)
      exact ⟨_, by rw [hv]; rfl⟩

/-- `canon j`'s text, where it has one. -/
def canonOf (j : Json) : String := (canon j).toOption.getD ""

theorem canon_rule (j : Json) (h : intsF 64 j = true) : canon j = .ok (canonOf j) := by
  obtain ⟨s, hs⟩ := canon_ok 64 j h
  rw [canonOf, hs]; rfl

theorem reached_rule (defs : Array (String × Json)) (callees : Array Json) (roots : Array String)
    (hd : defs = #[("Root_v1", rootE)]) (hc : callees = #[]) (hr : roots = #["Root_v1"]) :
    checkReached defs callees roots = .ok () := by
  subst hd hc hr
  unfold checkReached
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, ← Array.foldl_toList, List.foldl_cons,
    List.foldl_nil, refs_root]
  simp (config := {decide := true}) [List.range', List.forIn_cons,
    Std.HashMap.getD_insert_self, refs_root]
  rfl

/-- A state monad over the decoder's refusals: the decoder's (`DecState`) and the digests' memo. -/
abbrev DM (σ : Type) := StateT σ (Except String)

theorem st_run {σ α} (x : DM σ α) (s : σ) : StateT.run x s = x s := rfl
theorem st_bind {σ α β} (x : DM σ α) (f : α → DM σ β) (s : σ) :
    (x >>= f) s = (x s >>= fun p => f p.1 p.2) := rfl
theorem st_get {σ} (s : σ) : (get : DM σ σ) s = .ok (s, s) := rfl
theorem st_set {σ} (s s' : σ) : (set s' : DM σ PUnit) s = .ok (⟨⟩, s') := rfl
theorem st_modify {σ} (f : σ → σ) (s : σ) : (modify f : DM σ PUnit) s = .ok (⟨⟩, f s) := rfl
theorem st_pure {σ α} (a : α) (s : σ) : (pure a : DM σ α) s = .ok (a, s) := rfl
theorem st_ite {σ α} (c : Prop) [Decidable c] (t e : DM σ α) (s : σ) :
    (if c then t else e) s = if c then t s else e s := by split <;> rfl
theorem st_mapConst {σ α β} (b : β) (x : DM σ α) (s : σ) :
    (Functor.mapConst b x) s = (x s >>= fun p => .ok (b, p.2)) := rfl
theorem st_lift {σ α} (x : Except String α) (s : σ) :
    (monadLift x : DM σ α) s = (x >>= fun a => .ok (a, s)) := rfl
theorem st_forIn_nil {σ α β} (xs : Array α) (h : xs = #[]) (b : β) (f : α → β → DM σ (ForInStep β))
    (s : σ) :
    (forIn xs b f : DM σ β) s = .ok (b, s) := by subst h; rfl
theorem st_forIn_range0 {σ β} (m : Nat) (h : m = 0) (b : β) (f : Nat → β → DM σ (ForInStep β)) (s : σ) :
    (forIn [0:m] b f : DM σ β) s = .ok (b, s) := by
  subst h; rw [Std.Legacy.Range.forIn_eq_forIn_range']; rfl
theorem st_forIn_root {σ β} (xs : Array (String × Json)) (h : xs = #[("Root_v1", rootE)]) (b : β)
    (f : String × Json → β → DM σ (ForInStep β)) (s : σ) :
    (forIn xs b f : DM σ β) s = (f ("Root_v1", rootE) b s >>= fun p => .ok (p.1.value, p.2)) := by
  subst h
  rw [← Array.forIn_toList]
  show (do let r ← f ("Root_v1", rootE) b s; _) = _
  apply congrArg; funext p; rcases p with ⟨_ | _, _⟩ <;> rfl

theorem lift_self {α} (x : Except String α) : (monadLift x : Except String α) = x := rfl

theorem memo_none (m : Std.HashMap String String) (h : m = ∅) (k : String) : m[k]? = none := by
  subst h; exact Std.HashMap.getElem?_empty
theorem memo_one (k k' v : String) (h : k = k') : ((∅ : Std.HashMap String String).insert k v)[k']? = some v := by
  subst h; exact Std.HashMap.getElem?_insert_self

set_option maxHeartbeats 4000000 in -- the walk steps through `decode` and leaves closed checks to the kernel
fwd_def progEmpty decode_empty : decode programEmpty.toUTF8 by
  [fromUTF8?_toUTF8, parse_empty, domain_none, ofJson_tuple0, split_rule, toNat_rule, rfi_rule, ddefs_rule,
    index_rule, busy_rule, String.contains_char_eq, String.all_bool_eq, Array.mapM_empty,
    st_run, st_bind, st_get, st_set, st_modify, st_pure, st_ite, st_mapConst, st_lift,
    st_forIn_nil, st_forIn_range0, st_forIn_root, memo_one, memo_none, lift_self, leaves_rule, dup_rule,
    names_rule, reached_rule, refs_rule, subFns_rule, canon_rule]
  block [domainError, Ty.ofJson, checkReached]
  shallow [programEmpty]

end Flock.Program.Closed
