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
16-bit output, the root `Root[Two_v1]_v1`, a batch of two of its members returning both, and `T_v1`'s Boolean body
`StandIn[T]_v1` over the constant bit `Const1[0x0]_v1`. The decoder's tables are
`Std.HashMap`s and `Std.HashSet`s keyed by `String.hash`, which the kernel does not evaluate, so each lookup is read off
the inserts that built its table (`hm_get_insert`, `hs_contains_insert`, `hm_get_foldl`, `hm_contains_foldl`), and its
state is a `StateT` stepped by its monad's equations (`st_bind` and its siblings). Each well-founded function it calls
is a fact at the statement's own values (`Ty.ofJson`, `Ty.leaves`, `Ty.beq`, `String.splitOn`, `Widths.bodyBad`) or
its fuelled twin (`domainError_fuel`, `canon_fuel`). The program digest is composed (`Program.programDigestWith`): each
Definition's SHA-512 over its canonical entry with its references' digests, memoised in a `Std.HashMap` read off its
inserts as the tables are, then the SHA-512 of the program's own record. `descParse_X` is the template queries' reading
of the same bytes (`TemplateQuery.Desc.parse`), `descX`, whose digest is `progHex` (`descX_hex`, `progHex_eq`).
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

open private Flock.Program.DecState.mk Flock.Program.DecState.defs Flock.Program.DecState.index
  Flock.Program.DecState.busy Flock.Program.DecState.walked Flock.Program.fnRefs from Circuits.Program

namespace SpotFlock.InstanceStmt.Decode

open Flock Flock.Program
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

theorem hm_get_foldl_n {β : Type} (arr : Array (String × β)) (n : Nat) (a : String) (h : n = arr.size) :
    (arr.foldl (fun m (k, v) => m.insert k v) ({} : Std.HashMap String β) 0 n)[a]? =
      (arr.toList.reverse.find? (·.1 == a)).map (·.2) := by
  subst h; exact hm_get_foldl arr a

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
theorem st_lift {σ ε α : Type} (x : Except ε α) (s : σ) :
    (monadLift x : StateT σ (Except ε) α) s = (x >>= fun a => .ok (a, s)) := rfl
theorem st_mapConst {σ ε α β : Type} (b : β) (x : StateT σ (Except ε) α) (s : σ) :
    (Functor.mapConst b x) s = (x s >>= fun p => .ok (b, p.2)) := rfl
theorem lift_self {ε α : Type} (x : Except ε α) : (monadLift x : Except ε α) = x := rfl
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
/-- `m`'s result at a loop's last step, `ForInStep`'s value read by a match, as `forIn` returns it. -/
abbrev stepVal {β : Type} (x : ForInStep β) : β := match x with | .done b => b | .yield b => b

theorem st_forIn_one {σ ε α β : Type} (xs : Array α) (b : β)
    (f : α → β → StateT σ (Except ε) (ForInStep β)) (s : σ) (h : xs.size = 1) :
    (forIn xs b f) s = (f (xs[0]'(by omega)) b >>= fun x => pure (stepVal x)) s := by
  obtain ⟨l⟩ := xs
  match l, h with
  | [a], _ =>
    have e : (forIn #[a] b f : StateT σ (Except ε) β) = (f a b >>= fun x => pure (stepVal x)) := by
      rw [← Array.forIn_toList]; show forIn [a] b f = _
      rw [List.forIn_cons]; congr 1; funext x; cases x <;> rfl
    rw [e]; rfl

theorem st_forIn_two {σ ε α β : Type} (xs : Array α) (b : β)
    (f : α → β → StateT σ (Except ε) (ForInStep β)) (s : σ) (h : xs.size = 2) :
    (forIn xs b f) s = (f (xs[0]'(by omega)) b >>= fun x => match x with
      | .done c => pure c
      | .yield c => f (xs[1]'(by omega)) c >>= fun y => pure (stepVal y)) s := by
  obtain ⟨l⟩ := xs
  match l, h with
  | [a, a1], _ =>
    have e : (forIn #[a, a1] b f : StateT σ (Except ε) β) = (f a b >>= fun x => match x with
        | .done c => pure c
        | .yield c => f a1 c >>= fun y => pure (stepVal y)) := by
      rw [← Array.forIn_toList]; show forIn [a, a1] b f = _
      rw [List.forIn_cons]; congr 1; funext x; cases x with
      | done c => rfl
      | yield c =>
        show forIn [a1] c f = _
        rw [List.forIn_cons]; congr 1; funext y; cases y <;> rfl
    rw [e]; rfl

theorem st_forIn_three {σ ε α β : Type} (xs : Array α) (b : β)
    (f : α → β → StateT σ (Except ε) (ForInStep β)) (s : σ) (h : xs.size = 3) :
    (forIn xs b f) s = (f (xs[0]'(by omega)) b >>= fun x => match x with
      | .done c => pure c
      | .yield c => f (xs[1]'(by omega)) c >>= fun x => match x with
        | .done c => pure c
        | .yield c => f (xs[2]'(by omega)) c >>= fun y => pure (stepVal y)) s := by
  obtain ⟨l⟩ := xs
  match l, h with
  | [a, a1, a2], _ =>
    have e : (forIn #[a, a1, a2] b f : StateT σ (Except ε) β) = (f a b >>= fun x => match x with
        | .done c => pure c
        | .yield c => f a1 c >>= fun x => match x with
          | .done c => pure c
          | .yield c => f a2 c >>= fun y => pure (stepVal y)) := by
      rw [← Array.forIn_toList]; show forIn [a, a1, a2] b f = _
      rw [List.forIn_cons]; congr 1; funext x; cases x with
      | done c => rfl
      | yield c =>
        show forIn [a1, a2] c f = _
        rw [List.forIn_cons]; congr 1; funext x; cases x with
        | done c => rfl
        | yield c =>
          show forIn [a2] c f = _
          rw [List.forIn_cons]; congr 1; funext y; cases y <;> rfl
    rw [e]; rfl

theorem st_forIn_four {σ ε α β : Type} (xs : Array α) (b : β)
    (f : α → β → StateT σ (Except ε) (ForInStep β)) (s : σ) (h : xs.size = 4) :
    (forIn xs b f) s = (f (xs[0]'(by omega)) b >>= fun x => match x with
      | .done c => pure c
      | .yield c => f (xs[1]'(by omega)) c >>= fun x => match x with
        | .done c => pure c
        | .yield c => f (xs[2]'(by omega)) c >>= fun x => match x with
          | .done c => pure c
          | .yield c => f (xs[3]'(by omega)) c >>= fun y => pure (stepVal y)) s := by
  obtain ⟨l⟩ := xs
  match l, h with
  | [a, a1, a2, a3], _ =>
    have e : (forIn #[a, a1, a2, a3] b f : StateT σ (Except ε) β) = (f a b >>= fun x => match x with
        | .done c => pure c
        | .yield c => f a1 c >>= fun x => match x with
          | .done c => pure c
          | .yield c => f a2 c >>= fun x => match x with
            | .done c => pure c
            | .yield c => f a3 c >>= fun y => pure (stepVal y)) := by
      rw [← Array.forIn_toList]; show forIn [a, a1, a2, a3] b f = _
      rw [List.forIn_cons]; congr 1; funext x; cases x with
      | done c => rfl
      | yield c =>
        show forIn [a1, a2, a3] c f = _
        rw [List.forIn_cons]; congr 1; funext x; cases x with
        | done c => rfl
        | yield c =>
          show forIn [a2, a3] c f = _
          rw [List.forIn_cons]; congr 1; funext x; cases x with
          | done c => rfl
          | yield c =>
            show forIn [a3] c f = _
            rw [List.forIn_cons]; congr 1; funext y; cases y <;> rfl
    rw [e]; rfl

theorem stm_forIn_empty {σ ε α β : Type} (xs : Array α) (b : β) (f : α → β → StateT σ (Except ε) (ForInStep β))
    (h : xs.size = 0) : (forIn xs b f : StateT σ (Except ε) β) = pure b := by
  rw [Array.eq_empty_of_size_eq_zero h]; rfl
theorem stm_forIn_one {σ ε α β : Type} (xs : Array α) (b : β)
    (f : α → β → StateT σ (Except ε) (ForInStep β)) (h : xs.size = 1) :
    (forIn xs b f : StateT σ (Except ε) β) = (f (xs[0]'(by omega)) b >>= fun x => pure (stepVal x)) :=
  funext fun s => st_forIn_one xs b f s h
theorem stm_forIn_two {σ ε α β : Type} (xs : Array α) (b : β)
    (f : α → β → StateT σ (Except ε) (ForInStep β)) (h : xs.size = 2) :
    (forIn xs b f : StateT σ (Except ε) β) = (f (xs[0]'(by omega)) b >>= fun x => match x with
      | .done c => pure c
      | .yield c => f (xs[1]'(by omega)) c >>= fun y => pure (stepVal y)) :=
  funext fun s => st_forIn_two xs b f s h
theorem stm_forIn_three {σ ε α β : Type} (xs : Array α) (b : β)
    (f : α → β → StateT σ (Except ε) (ForInStep β)) (h : xs.size = 3) :
    (forIn xs b f : StateT σ (Except ε) β) = (f (xs[0]'(by omega)) b >>= fun x => match x with
      | .done c => pure c
      | .yield c => f (xs[1]'(by omega)) c >>= fun x => match x with
        | .done c => pure c
        | .yield c => f (xs[2]'(by omega)) c >>= fun y => pure (stepVal y)) :=
  funext fun s => st_forIn_three xs b f s h
theorem stm_forIn_four {σ ε α β : Type} (xs : Array α) (b : β)
    (f : α → β → StateT σ (Except ε) (ForInStep β)) (h : xs.size = 4) :
    (forIn xs b f : StateT σ (Except ε) β) = (f (xs[0]'(by omega)) b >>= fun x => match x with
      | .done c => pure c
      | .yield c => f (xs[1]'(by omega)) c >>= fun x => match x with
        | .done c => pure c
        | .yield c => f (xs[2]'(by omega)) c >>= fun x => match x with
          | .done c => pure c
          | .yield c => f (xs[3]'(by omega)) c >>= fun y => pure (stepVal y)) :=
  funext fun s => st_forIn_four xs b f s h

theorem ex_forIn_n1 {ε α β : Type} (xs : Array α) (b : β) (f : α → β → Except ε (ForInStep β)) (h : xs.size = 1) :
    forIn xs b f = (f (xs[0]'(by omega)) b >>= fun x => pure (stepVal x)) := by
  obtain ⟨l⟩ := xs
  match l, h with
  | [a], _ =>
    rw [← Array.forIn_toList]; show forIn [a] b f = _
    rw [List.forIn_cons]; congr 1; funext x; cases x <;> rfl
theorem ex_forIn_n2 {ε α β : Type} (xs : Array α) (b : β) (f : α → β → Except ε (ForInStep β)) (h : xs.size = 2) :
    forIn xs b f = (f (xs[0]'(by omega)) b >>= fun x => match x with
      | .done c => pure c
      | .yield c => f (xs[1]'(by omega)) c >>= fun y => pure (stepVal y)) := by
  obtain ⟨l⟩ := xs
  match l, h with
  | [a, a1], _ =>
    rw [← Array.forIn_toList]; show forIn [a, a1] b f = _
    rw [List.forIn_cons]; congr 1; funext x; cases x with
    | done c => rfl
    | yield c =>
      show forIn [a1] c f = _
      rw [List.forIn_cons]; congr 1; funext y; cases y <;> rfl
theorem ex_forIn_n3 {ε α β : Type} (xs : Array α) (b : β) (f : α → β → Except ε (ForInStep β)) (h : xs.size = 3) :
    forIn xs b f = (f (xs[0]'(by omega)) b >>= fun x => match x with
      | .done c => pure c
      | .yield c => f (xs[1]'(by omega)) c >>= fun x => match x with
        | .done c => pure c
        | .yield c => f (xs[2]'(by omega)) c >>= fun y => pure (stepVal y)) := by
  obtain ⟨l⟩ := xs
  match l, h with
  | [a, a1, a2], _ =>
    rw [← Array.forIn_toList]; show forIn [a, a1, a2] b f = _
    rw [List.forIn_cons]; congr 1; funext x; cases x with
    | done c => rfl
    | yield c =>
      show forIn [a1, a2] c f = _
      rw [List.forIn_cons]; congr 1; funext x; cases x with
      | done c => rfl
      | yield c =>
        show forIn [a2] c f = _
        rw [List.forIn_cons]; congr 1; funext y; cases y <;> rfl
theorem ex_forIn_n4 {ε α β : Type} (xs : Array α) (b : β) (f : α → β → Except ε (ForInStep β)) (h : xs.size = 4) :
    forIn xs b f = (f (xs[0]'(by omega)) b >>= fun x => match x with
      | .done c => pure c
      | .yield c => f (xs[1]'(by omega)) c >>= fun x => match x with
        | .done c => pure c
        | .yield c => f (xs[2]'(by omega)) c >>= fun x => match x with
          | .done c => pure c
          | .yield c => f (xs[3]'(by omega)) c >>= fun y => pure (stepVal y)) := by
  obtain ⟨l⟩ := xs
  match l, h with
  | [a, a1, a2, a3], _ =>
    rw [← Array.forIn_toList]; show forIn [a, a1, a2, a3] b f = _
    rw [List.forIn_cons]; congr 1; funext x; cases x with
    | done c => rfl
    | yield c =>
      show forIn [a1, a2, a3] c f = _
      rw [List.forIn_cons]; congr 1; funext x; cases x with
      | done c => rfl
      | yield c =>
        show forIn [a2, a3] c f = _
        rw [List.forIn_cons]; congr 1; funext x; cases x with
        | done c => rfl
        | yield c =>
          show forIn [a3] c f = _
          rw [List.forIn_cons]; congr 1; funext y; cases y <;> rfl

theorem push1_get0 {α : Type} (a : α) (h : 0 < (#[].push a).size) : (#[].push a)[0]'h = a := rfl
theorem push2_get0 {α : Type} (a b : α) (h : 0 < ((#[].push a).push b).size) : ((#[].push a).push b)[0]'h = a := rfl
theorem push2_get1 {α : Type} (a b : α) (h : 1 < ((#[].push a).push b).size) : ((#[].push a).push b)[1]'h = b := rfl
theorem push3_get0 {α : Type} (a b c : α) (h : 0 < (((#[].push a).push b).push c).size) :
    (((#[].push a).push b).push c)[0]'h = a := rfl
theorem push3_get1 {α : Type} (a b c : α) (h : 1 < (((#[].push a).push b).push c).size) :
    (((#[].push a).push b).push c)[1]'h = b := rfl
theorem push3_get2 {α : Type} (a b c : α) (h : 2 < (((#[].push a).push b).push c).size) :
    (((#[].push a).push b).push c)[2]'h = c := rfl
theorem push4_get0 {α : Type} (a b c d : α) (h : 0 < ((((#[].push a).push b).push c).push d).size) :
    ((((#[].push a).push b).push c).push d)[0]'h = a := rfl
theorem push4_get1 {α : Type} (a b c d : α) (h : 1 < ((((#[].push a).push b).push c).push d).size) :
    ((((#[].push a).push b).push c).push d)[1]'h = b := rfl
theorem push4_get2 {α : Type} (a b c d : α) (h : 2 < ((((#[].push a).push b).push c).push d).size) :
    ((((#[].push a).push b).push c).push d)[2]'h = c := rfl
theorem push4_get3 {α : Type} (a b c d : α) (h : 3 < ((((#[].push a).push b).push c).push d).size) :
    ((((#[].push a).push b).push c).push d)[3]'h = d := rfl

theorem defn_key (k i p ps r b bd g pl ol) : (Flock.Program.Defn.mk k i p ps r b bd g pl ol).key = k := rfl
theorem defn_prim (k i p ps r b bd g pl ol) : (Flock.Program.Defn.mk k i p ps r b bd g pl ol).prim = p := rfl
theorem defn_params (k i p ps r b bd g pl ol) : (Flock.Program.Defn.mk k i p ps r b bd g pl ol).params = ps := rfl
theorem defn_ret (k i p ps r b bd g pl ol) : (Flock.Program.Defn.mk k i p ps r b bd g pl ol).ret = r := rfl

/-- `fnRefs` on a definition's `bindings`: none (a primitive's entry has no `bindings`) or the empty object. -/
theorem fnRefs_null (acc : Array String) (j : Json) (h : j = .null) : Flock.Program.fnRefs acc j = acc := by
  subst h; unfold Flock.Program.fnRefs; rfl
theorem fnRefs_empty (acc : Array String) (j : Json) (h : j = Json.mkObj []) : Flock.Program.fnRefs acc j = acc := by
  subst h; unfold Flock.Program.fnRefs; rfl

theorem mapM_triple {ε α β : Type} (f : α → Except ε β) (a b c : α) :
    #[a, b, c].mapM f = (do let x ← f a; let y ← f b; let z ← f c; pure #[x, y, z]) := by
  simp only [Array.mapM_eq_mapM_toList, List.mapM_cons, List.mapM_nil]; cases f a <;> cases f b <;> cases f c <;> rfl
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
theorem ofJson_B : Ty.ofJson (ja [js "Value", jn 1]) = .ok (.value 1) := by
  rw [Ty.ofJson.eq_def]; rfl
theorem ofJson_W : Ty.ofJson (ja [js "Array", jn 16, ja [js "Value", jn 1]]) = .ok (.array 16 (.value 1)) := by
  rw [Ty.ofJson.eq_def]
  show (do let x ← Ty.ofJson (ja [js "Value", jn 1]); pure (Ty.array 16 x)) = _
  rw [ofJson_B]; rfl
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

theorem splitOn_root : "Root[Two_v1]_v1".splitOn "_v" = ["Root[Two", "1]", "1"] := by
  simp (config := { decide := true }) [String.splitOn, String.splitOnAux]
theorem all_one : "1".all Char.isDigit = true := by rw [String.all_bool_eq]; decide
theorem toNat_one : "1".toNat! = 1 := by
  simp only [String.toNat!, String.Slice.toNat!, String.Slice.isNat, String.Slice.forIn_eq_forIn_toList,
    String.Slice.foldl_eq_foldl_toList]; decide
theorem splitOn_of (s : String) (h : s = "Root[Two_v1]_v1") : s.splitOn "_v" = ["Root[Two", "1]", "1"] := by
  subst h; exact splitOn_root
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

/-- `paramsOf`'s duplicate-name check on no parameters finds none: the kernel doesn't reduce a `Std.HashSet`'s size, so
the walk, whose two definitions have no parameters, takes this step as a rule. -/
theorem dup_empty {α β : Type} [BEq α] [Hashable α] (f : Std.HashSet α → β → Std.HashSet α) :
    ((Array.foldl f (∅ : Std.HashSet α) (#[] : Array β)).size != (#[] : Array β).size) = false := by
  simp

set_option maxHeartbeats 8000000 in
fwd_def progX decode_prog : Program.decode progBytes by [fromUTF8?_toUTF8, parse_prog, domainError_fuel 12,
    hm_get_insert, hm_get_empty, hs_contains_insert, hs_contains_erase, hs_contains_empty, hm_get_foldl,
    hm_contains_foldl, mapM_nil, mapM_one, mapM_pair, mapM_triple, ofJson_V, ofJson_A, ofJson_B, ofJson_W,
    beq_of_A, beq_of_V, leaves_of_A,
    leaves_of_V, bodyBad_one, recordFunctionIds_plain, splitOn_of, all_of, toNat_of, st_bind, st_get, st_set,
    st_modify, st_pure, st_throw, st_ite, st_dite, st_forIn_empty, st_forIn_range1, prod_fst, prod_snd, ds_defs,
    ds_index, ds_busy, ds_walked, mapM_empty, st_mapM_empty, st_run, qsort_fuel, canon_fuel 12,
    dup_empty, st_lift, st_mapConst, lift_self,
    st_forIn_one, st_forIn_two, st_forIn_four, st_forIn_three, stm_forIn_three, hm_get_foldl_n,
    push1_get0, push2_get0, push2_get1, push3_get0, push3_get1, push3_get2, push4_get0, push4_get1, push4_get2,
    push4_get3, defn_key, defn_prim, defn_params, defn_ret, ex_forIn_n1, ex_forIn_n2, ex_forIn_n3, ex_forIn_n4, stm_forIn_one, stm_forIn_two, stm_forIn_four,
    fnRefs_null, fnRefs_empty]
  block [Std.HashMap.insert, Std.HashSet.insert, Std.HashSet.erase, Sha512.hash]
  shallow []

/-- The Boolean map's entry for `a`: `progJ` maps `T_v1` alone, to its body. -/
theorem bmX_get (a : String) :
    (match progJ.getObjValD "boolean" with
      | .obj kvs => kvs.toArray.foldl (fun (m : Std.HashMap String String) (k, v) => match v with
          | Json.str s => m.insert k s | _ => m)
          ({} : Std.HashMap String String)
      | _ => {})[a]? = if "T_v1" == a then some "StandIn[T]_v1" else none := by
  have e : progJ.getObjValD "boolean" = Json.mkObj [("T_v1", js "StandIn[T]_v1")] := rfl
  rw [e]
  show (((∅ : Std.HashMap String String)).insert "T_v1" "StandIn[T]_v1")[a]? = _
  rw [Std.HashMap.getElem?_insert, Std.HashMap.getElem?_empty]

set_option maxHeartbeats 1600000 in
fwd_def descX descParse_X : TemplateQuery.Desc.parse progBytes by [fromUTF8?_toUTF8, parse_prog, canon_fuel 12,
    hm_get_insert, hm_get_empty, hm_get_foldl, hm_contains_foldl, recordFunctionIds_plain, st_bind, st_get, st_set,
    st_modify, st_pure, st_throw, st_ite, st_dite, st_run, st_lift, st_mapConst, lift_self, prod_fst, prod_snd,
    qsort_fuel, mapM_nil, mapM_one, mapM_pair, mapM_triple,
    st_forIn_one, st_forIn_two, st_forIn_four, st_forIn_three, stm_forIn_three, hm_get_foldl_n,
    push1_get0, push2_get0, push2_get1, push3_get0, push3_get1, push3_get2, push4_get0, push4_get1, push4_get2,
    push4_get3, defn_key, defn_prim, defn_params, defn_ret, ex_forIn_n1, ex_forIn_n2, ex_forIn_n3, ex_forIn_n4, stm_forIn_one, stm_forIn_two, stm_forIn_four,
    fnRefs_null, fnRefs_empty, st_forIn_empty, Array.size_push, Array.size_append, Option.toArray_some,
    Option.toArray_none, List.size_toArray, Array.size_empty, bmX_get, stm_forIn_empty]
  block [Sha512.hash, Std.HashMap.insert, Std.HashSet.insert]
  shallow []

theorem descX_hex : hexOf descX.sha512 = progHex := by
  unfold descX
  simp only [canon_fuel 12]
  decide +kernel

/-- **The program's digest is `progHex`.** -/
theorem progHex_eq : ((fun d => hexOf d.sha512) <$> TemplateQuery.Desc.parse progBytes).toOption = some progHex := by
  rw [descParse_X]; exact congrArg some descX_hex

end SpotFlock.InstanceStmt.Decode
