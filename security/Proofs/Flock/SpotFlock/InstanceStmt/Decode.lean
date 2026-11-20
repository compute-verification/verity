module

public import Proofs.Flock.SpotFlock.InstanceStmt.Digests
public import Proofs.Flock.SessionLoop.Concrete.Fwd
import Proofs.Flock.Claims.Sampling.Select.SecondWitness
import Batteries.Tactic.OpenPrivate
import Flock.Program
import Flock.TemplateQuery
import Flock.Canon
import Flock.JsonParse
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
import all Init.Data.String.Pattern.String
import all Init.Data.String.Pattern.Char
import all Init.Data.String.Pattern.Pred
import all Init.Data.String.Defs
import all Init.Data.String.Basic
import all Init.Data.String.Extra
import all Init.Data.String.Termination
import all Init.Data.String.Search
import all Init.Data.String.Iterate
import all Init.Data.String.Decode
import all Init.Data.String.ToSlice
import all Init.Data.String.Substring
import all Init.Data.String.Subslice
import all Init.Data.String.PosRaw
import all Init.Data.String.Modify
import all Init.Data.String.Legacy
import all Init.Data.String.Iterator
import all Init.Data.String.Iter
import all Init.Data.String.FindPos
import all Init.Data.String.Bootstrap
import all Init.Data.Array.Basic
import all Init.Data.Array.Attach
import all Init.Data.Array.DecidableEq
import all Init.Data.Array.QSort.Basic
import all Init.Data.Array.Subarray
import all Init.Data.Stream
import all Init.Data.List.Control
import all Init.Data.Range.Basic
import all Init.Data.Nat.Fold
import all Init.Data.Int.Repr
import all Init.Data.Repr
import all Init.WF

/-! # The statement's program decoded: `Program.decode progBytes`

`decode_prog` walks `Program.decode` on the program's bytes to `progX`: the definitions `T_v1`, a primitive of one
16-bit output, and the root `Root[Two_v1]_v1`, a batch of two of its members returning both. Its Boolean map sends
`T_v1` to `StandIn[T]_v1`, a batch of sixteen `Const1[0x0]_v1` gates, which the decoder holds to `T_v1`'s parameters
and return at one bit a leaf (`booleanKeys_prog`, `extendReach_prog`, `checkBody_prog`). The decoder's tables are
`Std.HashMap`s and `Std.HashSet`s keyed by `String.hash`, which the kernel does not evaluate, so each lookup is read off
the inserts that built its table (`hm_get_insert`, `hs_contains_insert`, `hm_get_foldl`, `hm_contains_foldl`), and its
state is a `StateT` stepped by its monad's equations (`st_bind` and its siblings). Each well-founded function it calls
is a fact at the statement's own values (`Ty.ofJson`, `Ty.leaves`, `Ty.beq`, `String.splitOn`, `Widths.bodyBad`) or
its fuelled twin (`domainError_fuel`, `canon_fuel`). The program's SHA-512 is of `bareText`, its canonical text without
`annotations`, a literal as `progHex` is (`canon_bare`). `descParse_X` is the template queries' reading of the same
bytes (`TemplateQuery.Desc.parse`), `descX`, whose digest is `progHex` (`descX_hex`).
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

open private Flock.Program.DecState.mk Flock.Program.DecState.defs Flock.Program.DecState.index
  Flock.Program.DecState.busy Flock.Program.DecState.walked from Circuits.Descriptor

namespace SpotFlock.InstanceStmt.Decode

open Flock Flock.Program
open private widthAfter? extendReach inputOutsideRoot from Circuits.Descriptor
open Lean (Json)
open Properties.Spot.SecurityProofs (canon_fuel canonFuel qsort_fuel)

/-! ## The decoder's tables -/

theorem hm_get_insert {β : Type} (m : Std.HashMap String β) (k a : String) (v : β) :
    (m.insert k v)[a]? = if k == a then some v else m[a]? := by
  rw [Std.HashMap.getElem?_insert]
theorem hm_get_empty {β : Type} (a : String) : (({} : Std.HashMap String β))[a]? = none :=
  Std.HashMap.getElem?_empty
theorem hs_contains_insert (m : Std.HashSet String) (k a : String) :
    (m.insert k).contains a = (k == a || m.contains a) := Std.HashSet.contains_insert
theorem hs_contains_erase (m : Std.HashSet String) (k a : String) :
    (m.erase k).contains a = (!(k == a) && m.contains a) := Std.HashSet.contains_erase
theorem hs_contains_empty (a : String) : ({} : Std.HashSet String).contains a = false :=
  Std.HashSet.contains_empty

theorem hm_get_foldl_from {β : Type} (l : List (String × β)) (m : Std.HashMap String β) (a : String) :
    (l.foldl (fun m (k, v) => m.insert k v) m)[a]? = ((l.reverse.find? (·.1 == a)).map (·.2)).or m[a]? := by
  induction l generalizing m with
  | nil => simp
  | cons kv l ih =>
    obtain ⟨k, v⟩ := kv
    rw [List.foldl_cons, ih, Std.HashMap.getElem?_insert, List.reverse_cons, List.find?_append]
    by_cases h : k = a
    · subst h; cases l.reverse.find? (·.1 == k) <;> simp
    · have : (k == a) = false := by simpa using h
      cases l.reverse.find? (·.1 == a) <;> simp [this]

theorem hm_get_foldl {β : Type} (arr : Array (String × β)) (a : String) :
    (arr.foldl (fun m (k, v) => m.insert k v) ({} : Std.HashMap String β))[a]? =
      (arr.toList.reverse.find? (·.1 == a)).map (·.2) := by
  rw [← Array.foldl_toList, hm_get_foldl_from, Std.HashMap.getElem?_empty, Option.or_none]

theorem hm_contains_foldl_from {β : Type} (l : List (String × β)) (m : Std.HashMap String β) (a : String) :
    (l.foldl (fun m (k, v) => m.insert k v) m).contains a = (l.any (·.1 == a) || m.contains a) := by
  induction l generalizing m with
  | nil => simp
  | cons kv l ih =>
    obtain ⟨k, v⟩ := kv
    rw [List.foldl_cons, ih, Std.HashMap.contains_insert, List.any_cons]
    cases l.any (·.1 == a) <;> cases k == a <;> simp

theorem hm_contains_foldl {β : Type} (arr : Array (String × β)) (a : String) :
    (arr.foldl (fun m (k, v) => m.insert k v) ({} : Std.HashMap String β)).contains a =
      arr.toList.any (·.1 == a) := by
  rw [← Array.foldl_toList, hm_contains_foldl_from, Std.HashMap.contains_empty, Bool.or_false]

theorem hs_contains_foldl_from {α : Type} (g : α → String) (l : List α) (m : Std.HashSet String) (a : String) :
    (l.foldl (fun s p => s.insert (g p)) m).contains a = (l.any (g · == a) || m.contains a) := by
  induction l generalizing m with
  | nil => simp
  | cons p l ih =>
    rw [List.foldl_cons, ih, Std.HashSet.contains_insert, List.any_cons]
    cases l.any (g · == a) <;> cases g p == a <;> simp

theorem hs_contains_foldl {α : Type} (g : α → String) (arr : Array α) (a : String) :
    (arr.foldl (fun s p => s.insert (g p)) ({} : Std.HashSet String)).contains a = arr.toList.any (g · == a) := by
  rw [← Array.foldl_toList, hs_contains_foldl_from, Std.HashSet.contains_empty, Bool.or_false]

/-! ## The decoder's state and loops -/

theorem st_bind {σ ε α β : Type} (x : StateT σ (Except ε) α) (f : α → StateT σ (Except ε) β) (s : σ) :
    (x >>= f) s = (x s >>= fun p => f p.1 p.2) := rfl
theorem st_get {σ ε : Type} (s : σ) : (get : StateT σ (Except ε) σ) s = .ok (s, s) := rfl
theorem st_set {σ ε : Type} (s s' : σ) : (set s' : StateT σ (Except ε) PUnit) s = .ok (⟨⟩, s') := rfl
theorem st_modify {σ ε : Type} (f : σ → σ) (s : σ) : (modify f : StateT σ (Except ε) PUnit) s = .ok (⟨⟩, f s) := rfl
theorem st_pure {σ ε α : Type} (a : α) (s : σ) : (pure a : StateT σ (Except ε) α) s = .ok (a, s) := rfl
theorem st_throw {σ ε α : Type} (e : ε) (s : σ) : (throw e : StateT σ (Except ε) α) s = .error e := rfl
theorem st_ite {σ ε α : Type} (c : Prop) [Decidable c] (a b : StateT σ (Except ε) α) (s : σ) :
    (ite c a b) s = ite c (a s) (b s) := by split <;> rfl
theorem st_dite {σ ε α : Type} (c : Prop) [Decidable c] (a : c → StateT σ (Except ε) α) (b : ¬c → StateT σ (Except ε) α)
    (s : σ) : (dite c a b) s = dite c (fun h => a h s) (fun h => b h s) := by split <;> rfl
theorem st_run {σ ε α : Type} (x : StateT σ (Except ε) α) (s : σ) : x.run s = x s := rfl
theorem st_forIn_one {σ ε α β : Type} (xs : Array α) (b : β)
    (f : α → β → StateT σ (Except ε) (ForInStep β)) (s : σ) (h : xs.size = 1) :
    (forIn xs b f) s =
      (f (xs[0]'(by omega)) b >>= fun r => match r with | .done b => pure b | .yield b => pure b) s := by
  obtain ⟨l⟩ := xs
  match l, h with
  | [a], _ =>
    rw [← Array.forIn_toList]
    simp only [List.forIn_cons, List.forIn_nil]
    rfl
theorem st_forIn_two {σ ε α β : Type} (xs : Array α) (b : β)
    (f : α → β → StateT σ (Except ε) (ForInStep β)) (s : σ) (h : xs.size = 2) :
    (forIn xs b f) s = (f (xs[0]'(by omega)) b >>= fun r => match r with
      | .done b => pure b
      | .yield b => f (xs[1]'(by omega)) b >>= fun r => match r with | .done b => pure b | .yield b => pure b) s := by
  obtain ⟨l⟩ := xs
  match l, h with
  | [a, c], _ =>
    rw [← Array.forIn_toList]
    simp only [List.forIn_cons, List.forIn_nil]
    rfl
theorem st_discard {σ ε α : Type} (x : StateT σ (Except ε) α) (s : σ) :
    (Functor.discard x) s = (x s >>= fun p => .ok (PUnit.unit, p.2)) := by
  show (Functor.mapConst PUnit.unit x) s = _
  simp only [Functor.mapConst, Function.comp_apply]
  show (x >>= fun a => pure (Function.const _ PUnit.unit a)) s = _
  show (x s >>= fun p => _) = _
  rfl
theorem st_forIn_empty {σ ε α β : Type} (xs : Array α) (b : β) (f : α → β → StateT σ (Except ε) (ForInStep β)) (s : σ)
    (h : xs.size = 0) : (forIn xs b f) s = .ok (b, s) := by
  rw [Array.eq_empty_of_size_eq_zero h]; rfl
theorem st_forIn_range1 {σ ε β : Type} (r : Std.Legacy.Range) (b : β)
    (f : Nat → β → StateT σ (Except ε) (ForInStep β)) (s : σ) (h1 : r.start = 0) (h2 : r.stop = 1) (h3 : r.step = 1) :
    (forIn r b f) s = (f 0 b >>= fun x => pure (match x with | .done b => b | .yield b => b)) s := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  have : List.range' r.start r.size r.step = [0] := by
    simp only [Std.Legacy.Range.size, h1, h2, h3]; rfl
  rw [this]
  have e : (forIn [0] b f : StateT σ (Except ε) β) =
      (f 0 b >>= fun x => pure (match x with | .done b => b | .yield b => b)) := by
    rw [List.forIn_cons]; congr 1; funext x; cases x <;> rfl
  rw [e]
theorem mapM_nil {ε α β : Type} (f : α → Except ε β) : (#[] : Array α).mapM f = pure #[] := by
  simp only [Array.mapM_eq_mapM_toList, List.mapM_nil]; rfl
theorem mapM_one {ε α β : Type} (f : α → Except ε β) (a : α) :
    #[a].mapM f = (do let x ← f a; pure #[x]) := by
  simp only [Array.mapM_eq_mapM_toList, List.mapM_cons, List.mapM_nil]; cases f a <;> rfl
theorem mapM_pair {ε α β : Type} (f : α → Except ε β) (a b : α) :
    #[a, b].mapM f = (do let x ← f a; let y ← f b; pure #[x, y]) := by
  simp only [Array.mapM_eq_mapM_toList, List.mapM_cons, List.mapM_nil]; cases f a <;> cases f b <;> rfl
theorem mapM_three {ε α β : Type} (f : α → Except ε β) (a b c : α) :
    #[a, b, c].mapM f = (do let x ← f a; let y ← f b; let z ← f c; pure #[x, y, z]) := by
  simp only [Array.mapM_eq_mapM_toList, List.mapM_cons, List.mapM_nil]
  cases f a <;> cases f b <;> cases f c <;> rfl
theorem mapM_empty {ε α β : Type} (f : α → Except ε β) (xs : Array α) (h : xs.size = 0) : xs.mapM f = pure #[] := by
  rw [Array.eq_empty_of_size_eq_zero h]; simp only [Array.mapM_eq_mapM_toList, List.mapM_nil]; rfl
theorem st_mapM_empty {σ ε α β : Type} (f : α → StateT σ (Except ε) β) (xs : Array α) (s : σ) (h : xs.size = 0) :
    (xs.mapM f) s = .ok (#[], s) := by
  rw [Array.eq_empty_of_size_eq_zero h]; simp only [Array.mapM_eq_mapM_toList, List.mapM_nil]; rfl
theorem ex_forIn_range_empty {ε β : Type} (r : Std.Legacy.Range) (b : β) (f : Nat → β → Except ε (ForInStep β))
    (h : r.size = 0) : forIn r b f = .ok b := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range', h]; rfl
theorem ex_forIn_empty {ε α β : Type} (xs : Array α) (b : β) (f : α → β → Except ε (ForInStep β))
    (h : xs.size = 0) : forIn xs b f = .ok b := by
  rw [Array.eq_empty_of_size_eq_zero h]; rfl
theorem ex_forIn_one {ε α β : Type} [Inhabited α] (xs : Array α) (b : β) (f : α → β → Except ε (ForInStep β))
    (h : xs.size = 1) :
    forIn xs b f = (f xs[0]! b >>= fun r => pure (match r with | .done b => b | .yield b => b)) := by
  obtain ⟨l⟩ := xs
  match l, h with
  | [a], _ =>
    rw [← Array.forIn_toList]
    simp only [List.forIn_cons, List.forIn_nil]
    show (f a b >>= _) = (f a b >>= _)
    congr 1; funext r; cases r <;> rfl
theorem ex_forIn_range1 {ε β : Type} (r : Std.Legacy.Range) (b : β) (f : Nat → β → Except ε (ForInStep β))
    (h1 : r.start = 0) (h2 : r.stop = 1) (h3 : r.step = 1) :
    forIn r b f = (f 0 b >>= fun x => pure (match x with | .done b => b | .yield b => b)) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  have : List.range' r.start r.size r.step = [0] := by
    simp only [Std.Legacy.Range.size, h1, h2, h3]; rfl
  rw [this, List.forIn_cons]; congr 1; funext x; cases x <;> rfl
theorem prod_fst {α β : Type} (a : α) (b : β) : (a, b).1 = a := rfl
theorem prod_snd {α β : Type} (a : α) (b : β) : (a, b).2 = b := rfl
theorem ds_defs (d i b w) : Flock.Program.DecState.defs (Flock.Program.DecState.mk d i b w) = d := rfl
theorem ds_index (d i b w) : Flock.Program.DecState.index (Flock.Program.DecState.mk d i b w) = i := rfl
theorem ds_busy (d i b w) : Flock.Program.DecState.busy (Flock.Program.DecState.mk d i b w) = b := rfl
theorem ds_walked (d i b w) : Flock.Program.DecState.walked (Flock.Program.DecState.mk d i b w) = w := rfl

/-! ## The program's types and names -/

theorem ofJson_V : Ty.ofJson (ja [js "Value", jn 16]) = .ok (.value 16) := by
  rw [Ty.ofJson.eq_def]; rfl
theorem ofJson_A : Ty.ofJson (ja [js "Array", jn 2, ja [js "Value", jn 16]]) = .ok (.array 2 (.value 16)) := by
  rw [Ty.ofJson.eq_def]
  show (do let x ← Ty.ofJson (ja [js "Value", jn 16]); pure (Ty.array 2 x)) = _
  rw [ofJson_V]; rfl
theorem leaves_V (w : Nat) : (Ty.value w).leaves = 1 := by rw [Ty.leaves]
theorem leaves_A : (Ty.array 2 (.value 16)).leaves = 2 := by rw [Ty.leaves, leaves_V]
theorem beq_A : (Ty.array 2 (.value 16) == Ty.array 2 (.value 16)) = true := by
  show Ty.beq _ _ = true
  rw [Ty.beq, Ty.beq]; rfl
theorem beq_V : (Ty.value 16 == Ty.value 16) = true := by
  show Ty.beq _ _ = true
  rw [Ty.beq]; rfl
theorem beq_of_A (a b : Ty) (ha : a = .array 2 (.value 16)) (hb : b = .array 2 (.value 16)) : (a == b) = true := by
  subst ha hb; exact beq_A
theorem beq_of_V (a b : Ty) (ha : a = .value 16) (hb : b = .value 16) : (a == b) = true := by
  subst ha hb; exact beq_V
theorem leaves_of_A (t : Ty) (h : t = .array 2 (.value 16)) : t.leaves = 2 := by subst h; exact leaves_A
theorem leaves_of_V (t : Ty) (h : t = .value 16) : t.leaves = 1 := by subst h; exact leaves_V 16

theorem ofJson_V1 : Ty.ofJson (ja [js "Value", jn 1]) = .ok (.value 1) := by
  rw [Ty.ofJson.eq_def]; rfl
theorem ofJson_B : Ty.ofJson (ja [js "Array", jn 16, ja [js "Value", jn 1]]) = .ok (.array 16 (.value 1)) := by
  rw [Ty.ofJson.eq_def]
  show (do let x ← Ty.ofJson (ja [js "Value", jn 1]); pure (Ty.array 16 x)) = _
  rw [ofJson_V1]; rfl
theorem leaves_B : (Ty.array 16 (.value 1)).leaves = 16 := by rw [Ty.leaves, leaves_V]
theorem beq_B : (Ty.array 16 (.value 1) == Ty.array 16 (.value 1)) = true := by
  show Ty.beq _ _ = true
  rw [Ty.beq, Ty.beq]; rfl
theorem beq_V1 : (Ty.value 1 == Ty.value 1) = true := by
  show Ty.beq _ _ = true
  rw [Ty.beq]; rfl
theorem beq_of_B (a b : Ty) (ha : a = .array 16 (.value 1)) (hb : b = .array 16 (.value 1)) : (a == b) = true := by
  subst ha hb; exact beq_B
theorem beq_of_V1 (a b : Ty) (ha : a = .value 1) (hb : b = .value 1) : (a == b) = true := by
  subst ha hb; exact beq_V1
theorem leaves_of_B (t : Ty) (h : t = .array 16 (.value 1)) : t.leaves = 16 := by subst h; exact leaves_B
theorem leaves_of_V1 (t : Ty) (h : t = .value 1) : t.leaves = 1 := by subst h; exact leaves_V 1

theorem splitOn_root : "Root[Two_v1]_v1".splitOn "_v" = ["Root[Two", "1]", "1"] := by
  simp (config := { decide := true }) [String.splitOn, String.splitOnAux]
theorem splitOn_standIn : "StandIn[T]_v1".splitOn "_v" = ["StandIn[T]", "1"] := by
  simp (config := { decide := true }) [String.splitOn, String.splitOnAux]
theorem all_one : "1".all Char.isDigit = true := by rw [String.all_bool_eq]; decide
theorem toNat_one : "1".toNat! = 1 := by
  simp only [String.toNat!, String.Slice.toNat!, String.Slice.isNat, String.Slice.forIn_eq_forIn_toList,
    String.Slice.foldl_eq_foldl_toList]; decide
theorem splitOn_of (s : String) (h : s = "Root[Two_v1]_v1") : s.splitOn "_v" = ["Root[Two", "1]", "1"] := by
  subst h; exact splitOn_root
theorem splitOn_of_standIn (s : String) (h : s = "StandIn[T]_v1") : s.splitOn "_v" = ["StandIn[T]", "1"] := by
  subst h; exact splitOn_standIn
theorem all_of (s : String) (h : s = "1") : s.all Char.isDigit = true := by subst h; exact all_one
theorem toNat_of (s : String) (h : s = "1") : s.toNat! = 1 := by subst h; exact toNat_one

theorem recordFunctionIds_plain (fid : String) (h : fid.toList.contains '{' = false) : recordFunctionIds fid = #[] := by
  have : fid.contains '{' = false := by
    rw [String.contains_char_eq]; simpa using h
  simp [recordFunctionIds, recordPairs, this]

/-! ## The root's widths -/

theorem rangeWidths_V (lo hi : Nat) :
    Widths.rangeWidths (.value 16) lo hi = if lo == 0 && 0 < hi then [16] else [] := by
  rw [Widths.rangeWidths]

theorem homo_A : Widths.homo (.array 2 (.value 16)) = some 16 := by
  unfold Widths.homo
  rw [leaves_A, Widths.rangeWidths]
  simp only [leaves_V, rangeWidths_V]
  decide

theorem slice_A : Widths.sliceWidth (.array 2 (.value 16)) 0 2 = some 16 := by
  unfold Widths.sliceWidth
  rw [Widths.rangeWidths]
  simp only [leaves_V, rangeWidths_V]
  decide

/-- The plain width check holds no input widths, so `recutOk` leaves every part to the rest of `partOk`. -/
theorem recutOk_plain (params nodes : Array Ty) (hp hn : Array (Option ℕ)) (below : ℕ) (t : Ty) (ht : Option ℕ) (a : ℕ)
    (q : Part) : Widths.recutOk { params, nodes, hp, hn, below, inp := #[], first := #[] } t ht a q = pure none := by
  cases q <;> simp [Widths.recutOk, Widths.Src.inW?]

/-- A body of one node without arguments, of type `Array 2 (Value 16)` as its return is, returning that node's leaves
`0 … 1` as one part, reads no leaf at another width. -/
theorem bodyBad_one (defs : Array Defn) (key : String) (B : Body) (w : Nat)
    (hn : B.nodes.size = 1) (ha : (B.nodes[0]!).args.size = 0)
    (ho : (B.nodes[0]!).out = .array 2 (.value 16)) (ht : B.retType = .array 2 (.value 16))
    (hr : B.ret.parts = #[.affine false 0 0 2]) (hs : B.ret.starts = #[0]) :
    (Widths.bodyBad defs key B).run w = .ok none w := by
  obtain ⟨params, retType, nodes, ret, offsets, recut⟩ := B
  obtain ⟨parts, starts, count⟩ := ret
  simp only at hn ha ho ht hr hs
  subst ht hr hs
  obtain ⟨l⟩ := nodes
  match l, hn, ha, ho with
  | [⟨form, fn, args, out, axes, n, nxs⟩], _, ha, ho =>
    change args.size = 0 at ha
    change out = _ at ho
    subst ho
    rw [Array.eq_empty_of_size_eq_zero ha]
    obtain ⟨s', hs'⟩ : ∃ s', Std.Stream.next? (Std.toStream (#[0] : Array ℕ)) = some (0, s') := ⟨_, rfl⟩
    simp only [hs', recutOk_plain, Widths.bodyBad, List.size_toArray, List.length_cons, List.length_nil, Nat.zero_add,
      List.getElem!_toArray, List.getElem!_eq_getElem?_getD, Widths.seqBad, Widths.partOk, Widths.Src.ty?,
      Array.getElem?_map, List.map_toArray, List.map_cons, List.map_nil, List.getElem?_toArray, Widths.Src.homo?,
      homo_A, Part.count, slice_A, Bool.and_eq_true, beq_iff_eq, Bool.or_eq_true, gt_iff_lt, Nat.div_pos_iff, List.cons_append,
      List.nil_append, Int.cast_ofNat_Int, Int.zero_mul, Int.add_zero, Int.natCast_ediv, Int.natCast_emod,
      List.foldl_cons, Nat.min_assoc, Nat.max_assoc, List.all_eq_true, bne_iff_ne, ne_eq, ite_not,
      Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, Nat.sub_zero, Nat.add_one_sub_one, Nat.div_one,
      Bool.not_eq_eq_eq_not, Bool.not_true, Nat.toString_eq_repr, bind_assoc, Nat.reduceAdd, Nat.lt_add_one,
      Nat.div_self, List.range'_one, List.forIn_cons, getElem?_pos, List.getElem_cons_zero, Option.getD_some,
      List.range'_zero, Nat.not_lt_zero, not_false_eq_true, getElem!_neg, ↓reduceIte, List.forIn_nil, pure_bind,
      Nat.lt_one_iff, Option.isSome_some, Bool.true_and, List.forIn_toArray, Bool.false_eq_true, Option.join_some,
      Bool.true_eq_false, EStateM.run_bind]
    rfl

theorem rangeWidths_V1 (lo hi : Nat) :
    Widths.rangeWidths (.value 1) lo hi = if lo == 0 && 0 < hi then [1] else [] := by
  rw [Widths.rangeWidths]

theorem homo_B : Widths.homo (.array 16 (.value 1)) = some 1 := by
  unfold Widths.homo
  rw [leaves_B, Widths.rangeWidths]
  simp only [leaves_V, rangeWidths_V1]
  decide

theorem slice_B : Widths.sliceWidth (.array 16 (.value 1)) 0 16 = some 1 := by
  unfold Widths.sliceWidth
  rw [Widths.rangeWidths]
  simp only [leaves_V, rangeWidths_V1]
  decide

/-- `bodyBad_one` for `StandIn[T]_v1`'s body: one node of type `Array 16 (Value 1)` as its return is, returned whole. -/
theorem bodyBad_bits (defs : Array Defn) (key : String) (B : Body) (w : Nat)
    (hn : B.nodes.size = 1) (ha : (B.nodes[0]!).args.size = 0)
    (ho : (B.nodes[0]!).out = .array 16 (.value 1)) (ht : B.retType = .array 16 (.value 1))
    (hr : B.ret.parts = #[.affine false 0 0 16]) (hs : B.ret.starts = #[0]) :
    (Widths.bodyBad defs key B).run w = .ok none w := by
  obtain ⟨params, retType, nodes, ret, offsets, recut⟩ := B
  obtain ⟨parts, starts, count⟩ := ret
  simp only at hn ha ho ht hr hs
  subst ht hr hs
  obtain ⟨l⟩ := nodes
  match l, hn, ha, ho with
  | [⟨form, fn, args, out, axes, n, nxs⟩], _, ha, ho =>
    change args.size = 0 at ha
    change out = _ at ho
    subst ho
    rw [Array.eq_empty_of_size_eq_zero ha]
    obtain ⟨s', hs'⟩ : ∃ s', Std.Stream.next? (Std.toStream (#[0] : Array ℕ)) = some (0, s') := ⟨_, rfl⟩
    simp only [hs', recutOk_plain, Widths.bodyBad, List.size_toArray, List.length_cons, List.length_nil, Nat.zero_add,
      List.getElem!_toArray, List.getElem!_eq_getElem?_getD, Widths.seqBad, Widths.partOk, Widths.Src.ty?,
      Array.getElem?_map, List.map_toArray, List.map_cons, List.map_nil, List.getElem?_toArray, Widths.Src.homo?,
      homo_B, Part.count, slice_B, Bool.and_eq_true, beq_iff_eq, Bool.or_eq_true, gt_iff_lt, Nat.div_pos_iff, List.cons_append,
      List.nil_append, Int.cast_ofNat_Int, Int.zero_mul, Int.add_zero, Int.natCast_ediv, Int.natCast_emod,
      List.foldl_cons, Nat.min_assoc, Nat.max_assoc, List.all_eq_true, bne_iff_ne, ne_eq, ite_not,
      Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, Nat.sub_zero, Nat.add_one_sub_one, Nat.div_one,
      Bool.not_eq_eq_eq_not, Bool.not_true, Nat.toString_eq_repr, bind_assoc, Nat.reduceAdd, Nat.lt_add_one,
      Nat.div_self, List.range'_one, List.forIn_cons, getElem?_pos, List.getElem_cons_zero, Option.getD_some,
      List.range'_zero, Nat.not_lt_zero, not_false_eq_true, getElem!_neg, ↓reduceIte, List.forIn_nil, pure_bind,
      Nat.lt_one_iff, Option.isSome_some, Bool.true_and, List.forIn_toArray, Bool.false_eq_true, Option.join_some,
      Bool.true_eq_false, EStateM.run_bind]
    rfl

/-! ## The canonical JSON domain, with fuel

`domainError` is well founded; `domainErrorFuel f` is it unfolded `f` times, `domainError` itself past that, so the two
agree at every fuel (`domainError_fuel`). -/

def domainErrorFuel : Nat → Json → String → Option String
  | 0, j, path => Program.domainError j path
  | _ + 1, .null, _ => none
  | _ + 1, .bool _, _ => none
  | _ + 1, .num n, path =>
    if n.exponent != 0 then some s!"{path}: a float is not a JSON integer, string, boolean, null, array or object"
    else if n.mantissa.natAbs > MAX_SAFE_INT then some s!"{path}: integer {n.mantissa} exceeds 2^53 - 1 in magnitude"
    else none
  | _ + 1, .str s, path =>
    if s.toList.all (fun c : Char => c.toNat < 0x7f) then none else some s!"{path}: string is not ASCII without DEL"
  | f + 1, .arr a, path => (Array.range a.size).attach.findSome? fun i =>
    domainErrorFuel f (a[i.1]'(Array.mem_range.1 i.2)) s!"{path}[{i.1}]"
  | f + 1, .obj kvs, path => kvs.toArray.attach.findSome? fun x =>
    if x.1.1.toList.all (fun c : Char => c.toNat < 0x7f) then domainErrorFuel f x.1.2 s!"{path}.{x.1.1}"
    else some s!"{path}.<key>: string is not ASCII without DEL"

theorem domainError_fuel :
    ∀ (f : Nat) (j : Json) (path : String), Program.domainError j path = domainErrorFuel f j path
  | 0, _, _ => rfl
  | f + 1, j, path => by
    cases j with
    | null => rw [Program.domainError]; rfl
    | bool b => rw [Program.domainError]; rfl
    | num n => rw [Program.domainError]; rfl
    | str s => rw [Program.domainError]; rfl
    | arr a => rw [Program.domainError]; simp only [domainErrorFuel, domainError_fuel f]
    | obj kvs => rw [Program.domainError]; simp only [domainErrorFuel, domainError_fuel f]

/-! ## The program's canonical text and the walk -/

/-- The program without `annotations`, as `Program.decode` writes it before hashing. -/
def bareJ : Json := match progJ with
  | .obj top => Json.mkObj (top.toArray.filter (·.1 != "annotations")).toList
  | _ => .null

/-- `bareJ`'s canonical text: a literal, as `progHex` is, so that what reads the program's SHA-512 reads no `canon`. -/
def bareText : String :=
  "{\"boolean\":{\"T_v1\":\"StandIn[T]_v1\"},\"callees\":[\"T_v1\",\"Const1[0x0]_v1\"],\"definitions\":{\"Const1[\
    0x0]_v1\":{\"form\":\"primitive\",\"id\":\"Const1[0x0]_v1\",\"params\":[],\"ret\":[\"Value\",1]},\"Root[Two_v1\
    ]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"fn\":0,\"form\":\"batch\",\"n\":2,\"ou\
    t\":1}],\"params\":[],\"ret\":{\"affine\":[\"n\",0,0,2]},\"ret_type\":[\"Array\",2,[\"Value\",16]]},\"form\":\
    \"composite\",\"id\":\"Root[Two_v1]_v1\"},\"StandIn[T]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[]\
    ,\"axes\":[],\"fn\":1,\"form\":\"batch\",\"n\":16,\"out\":2}],\"params\":[],\"ret\":{\"affine\":[\"n\",0,0,16]\
    },\"ret_type\":[\"Array\",16,[\"Value\",1]]},\"form\":\"composite\",\"id\":\"StandIn[T]_v1\"},\"T_v1\":{\"form\
    \":\"primitive\",\"id\":\"T_v1\",\"params\":[],\"ret\":[\"Value\",16]}},\"inputs\":[],\"program_fn\":\"Two_v1\
    \",\"root\":\"Root[Two_v1]_v1\",\"schema\":\"verity-ir/descriptor/v2\",\"types\":[[\"Value\",16],[\"Array\",2,\
    [\"Value\",16]],[\"Array\",16,[\"Value\",1]]]}"

theorem canon_bareJ : canon bareJ = .ok bareText := by
  rw [canon_fuel 12]; kernel_rfl'
theorem canon_bare (j : Json) (h : j = bareJ) : canon j = .ok bareText := by subst h; exact canon_bareJ

/-- `paramsOf`'s duplicate-name check on no parameters finds none: the kernel doesn't reduce a `Std.HashSet`'s size, so
the walk, whose two definitions have no parameters, takes this step as a rule. -/
theorem dup_empty {α β : Type} [BEq α] [Hashable α] (f : Std.HashSet α → β → Std.HashSet α) :
    ((Array.foldl f (∅ : Std.HashSet α) (#[] : Array β)).size != (#[] : Array β).size) = false := by
  simp

theorem gateKind_T : gateKind? "T_v1" = none := by simp (config := { decide := true }) [gateKind?, widthAfter?]
theorem wAnd_T : widthAfter? "And" "T_v1" = none := by simp (config := { decide := true }) [widthAfter?]
theorem wXor_T : widthAfter? "Xor" "T_v1" = none := by simp (config := { decide := true }) [widthAfter?]
theorem mkObj_T : ∃ kv, Lean.Json.mkObj [("T_v1", .str "StandIn[T]_v1")] = .obj kv ∧
    kv.toArray = #[("T_v1", .str "StandIn[T]_v1")] := ⟨_, rfl, rfl⟩
theorem inputId_T : inputId? "T_v1" = none := by simp (config := { decide := true }) [inputId?, widthAfter?]
theorem inputId_C : inputId? "Const1[0x0]_v1" = none := by simp (config := { decide := true }) [inputId?, widthAfter?]
theorem misspelt_T : misspelt "T_v1" = false := by
  simp (config := { decide := true }) [misspelt, isInputId, inputId_T, gateKind_T]
theorem misspelt_C : misspelt "Const1[0x0]_v1" = false := by
  simp (config := { decide := true }) [misspelt, isInputId, inputId_C, gateKind?]
theorem isBooleanPrim_T : isBooleanPrim "T_v1" = false := by
  simp (config := { decide := true }) [isBooleanPrim, gateKind_T]
theorem isBooleanPrim_C : isBooleanPrim "Const1[0x0]_v1" = true := by decide

theorem booleanKeys_prog (defs : Array Defn) (index : Std.HashMap String Nat) (prims : Array String) (bm : Lean.Json)
    (hd : defs.size = 3) (h0k : defs[0]!.key = "T_v1") (h0p : defs[0]!.prim = true)
    (h1p : defs[1]!.prim = false) (h2k : defs[2]!.key = "Const1[0x0]_v1") (h2p : defs[2]!.prim = true)
    (h2a : defs[2]!.params = #[]) (h2r : defs[2]!.ret = .value 1)
    (hp : prims = #["Const1[0x0]_v1", "T_v1"]) (hi0 : index["T_v1"]? = some 0)
    (hi2 : index["Const1[0x0]_v1"]? = some 2) (hb : bm = Lean.Json.mkObj [("T_v1", .str "StandIn[T]_v1")]) :
    booleanKeys defs index prims bm = .ok #[("T_v1", "StandIn[T]_v1")] := by
  obtain ⟨l⟩ := defs
  match l, hd with
  | [d0, d1, d2], _ =>
    simp only [List.getElem!_toArray, List.getElem!_cons_zero, List.getElem!_cons_succ] at h0k h0p h1p h2k h2p h2a h2r
    subst hp hb
    obtain ⟨kv, hkv, hkva⟩ := mkObj_T
    unfold booleanKeys
    rw [hkv]
    simp [hkva, wAnd_T, wXor_T, beq_V1, Std.HashSet.contains_insert, h0k, h0p, h1p, h2k, h2p, h2a, h2r, misspelt_T, misspelt_C, gateKind?, GateKind.signature,
      hi0, hi2, isWordPrim, isInputId, inputId_T, inputId_C, isBooleanPrim_T, isBooleanPrim_C]
    all_goals rfl

theorem bits_B : (Ty.array 16 (.value 1)).bits = true := by
  rw [Ty.bits, Ty.bits]; rfl
theorem bitType_V16 : (Ty.value 16).bitType = .array 16 (.value 1) := by
  rw [Ty.bitType]; rfl
theorem isInputId_C : isInputId "Const1[0x0]_v1" = false := by simp [isInputId, inputId_C]

theorem nodes_one {ns : Array Node} (h : ns.size = 1) : ∃ n, ns = #[n] := by
  rcases ns with ⟨_ | ⟨n, _ | _⟩⟩ <;> simp_all

theorem extendReach_prog (defs : Array Defn) (hd : defs.size = 4) (h0p : defs[0]!.prim = true)
    (h0k : defs[0]!.key = "T_v1") (h1p : defs[1]!.prim = false) (h1n : defs[1]!.body.nodes.size = 1)
    (h1f : defs[1]!.body.nodes[0]!.fn = 0) (h2p : defs[2]!.prim = true) (h2k : defs[2]!.key = "Const1[0x0]_v1")
    (h3p : defs[3]!.prim = false) (h3n : defs[3]!.body.nodes.size = 1) (h3f : defs[3]!.body.nodes[0]!.fn = 2) :
    extendReach defs #[] = #[false, false, true, true] := by
  obtain ⟨l⟩ := defs
  match l, hd with
  | [d0, d1, d2, d3], _ =>
    simp only [List.getElem!_toArray, List.getElem!_cons_zero, List.getElem!_cons_succ] at *
    obtain ⟨n1, hn1⟩ := nodes_one h1n
    obtain ⟨n3, hn3⟩ := nodes_one h3n
    rw [hn1] at h1f
    rw [hn3] at h3f
    simp only [List.getElem!_toArray, List.getElem!_cons_zero] at h1f h3f
    unfold extendReach
    simp [Id.run, hn1, hn3, h1f, h3f, h0p, h0k, h1p, h2p, h2k, h3p, isBooleanPrim_T, isBooleanPrim_C,
      List.range'_succ]
    rfl

theorem inputOutsideRoot_prog (defs : Array Defn) (rootKey : String) (lo : Nat) (hd : defs.size = 4) (hlo : lo = 3)
    (h2p : defs[2]!.prim = true) (h2k : defs[2]!.key = "Const1[0x0]_v1") (h3p : defs[3]!.prim = false)
    (h3r : (defs[3]!.key == rootKey) = false) (h3n : defs[3]!.body.nodes.size = 1)
    (h3f : defs[3]!.body.nodes[0]!.fn = 2) :
    inputOutsideRoot defs rootKey lo = none := by
  subst hlo
  obtain ⟨l⟩ := defs
  match l, hd with
  | [d0, d1, d2, d3], _ =>
    simp only [List.getElem!_toArray, List.getElem!_cons_zero, List.getElem!_cons_succ] at *
    obtain ⟨n3, hn3⟩ := nodes_one h3n
    rw [hn3] at h3f
    simp only [List.getElem!_toArray, List.getElem!_cons_zero] at h3f
    unfold inputOutsideRoot
    simp [Id.run, hn3, h3f, h2p, h2k, h3p, isInputId_C, List.forIn_cons, List.forIn_nil, List.range'_succ]
    rfl

theorem checkBody_prog (defs : Array Defn) (index : Std.HashMap String Nat) (rootKey : String) (lo : Nat)
    (reach : Array Bool) (w b : String) (bi : Nat) (hd : defs.size = 4) (hlo : lo = 3) (hbi : bi = 3)
    (h2p : defs[2]!.prim = true) (h2k : defs[2]!.key = "Const1[0x0]_v1") (h3p : defs[3]!.prim = false)
    (h3r : (defs[3]!.key == rootKey) = false) (h3n : defs[3]!.body.nodes.size = 1)
    (h3f : defs[3]!.body.nodes[0]!.fn = 2) (hiw : index[w]? = some 0) (h0a : defs[0]!.params = #[])
    (h0r : defs[0]!.ret = .value 16) (h3a : defs[3]!.params = #[]) (h3t : defs[3]!.ret = .array 16 (.value 1))
    (hr : reach[3]! = true) :
    checkBody defs index rootKey lo reach w b bi = .ok () := by
  rw [checkBody, inputOutsideRoot_prog defs rootKey lo hd hlo h2p h2k h3p h3r h3n h3f]
  subst hbi
  simp [hiw, h3p, h0a, h0r, h3a, h3t, hr, bits_B, bitType_V16, beq_B]
  rfl

theorem parseParts_affine16 (j : Json) (r : Bool) (defs : Array Defn) (nodes : Array Node)
    (leaves : Bool → Nat → Option Nat) (args : Nat → Nat → Option Seq)
    (hj : j = Json.mkObj [("affine", ja [js "n", jn 0, jn 0, jn 16])]) (hr : r = false)
    (hl : leaves false 0 = some (Ty.array 16 (.value 1)).leaves) :
    parseParts j (leavesFor r defs nodes leaves) args = .ok #[.affine false 0 0 16] := by
  subst hr
  rw [leavesFor_false]
  rw [leaves_B] at hl
  subst hj
  have e : leaves = fun p i => if p = false ∧ i = 0 then some 16 else leaves p i := by
    funext p i
    by_cases h : p = false ∧ i = 0
    · obtain ⟨rfl, rfl⟩ := h; simp [hl]
    · simp [h]
  rw [e]
  unfold parseParts
  rfl

set_option maxHeartbeats 12800000 in
fwd_def progX decode_prog : Program.decode progBytes by [fromUTF8?_toUTF8, parse_prog, domainError_fuel 12,
    hm_get_insert, hm_get_empty, hs_contains_insert, hs_contains_erase, hs_contains_empty, hm_get_foldl,
    hm_contains_foldl, mapM_nil, mapM_one, mapM_pair, ofJson_V, ofJson_A, beq_of_A, beq_of_V, leaves_of_A,
    leaves_of_V, bodyBad_one, recordFunctionIds_plain, splitOn_of, all_of, toNat_of, splitOn_of_standIn, st_bind, st_get, st_set,
    st_modify, st_pure, st_throw, st_ite, st_dite, st_forIn_empty, st_forIn_range1, prod_fst, prod_snd, ds_defs,
    ds_index, ds_busy, ds_walked, mapM_empty, st_mapM_empty, st_run, qsort_fuel, canon_bare, canon_fuel 12,
    dup_empty, mapM_three, ofJson_V1, ofJson_B, beq_of_B, beq_of_V1, leaves_of_B, leaves_of_V1, bodyBad_bits,
    st_forIn_one, st_forIn_two, hs_contains_foldl, st_discard, booleanKeys_prog, extendReach_prog,
    checkBody_prog, parseParts_affine16]
  block [Std.HashMap.insert, Std.HashSet.insert, Std.HashSet.erase, Sha512.hash, Flock.Program.parsePartsIn]
  shallow []

set_option maxHeartbeats 1600000 in
fwd_def descX descParse_X : TemplateQuery.Desc.parse progBytes by [fromUTF8?_toUTF8, parse_prog, canon_bare,
    canon_fuel 12]
  block [Sha512.hash]
  shallow []

theorem descX_hex : hexOf descX.sha512 = progHex := by
  have h := progHex_eq
  rw [descParse_X] at h
  exact Option.some.inj h

end SpotFlock.InstanceStmt.Decode
