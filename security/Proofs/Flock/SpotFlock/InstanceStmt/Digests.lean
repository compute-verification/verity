module

public import Proofs.Flock.SpotFlock.InstanceStmt.Header
public import Proofs.Flock.SpotFlock.InstanceStmt.Plain
import Proofs.Flock.Claims.Sampling.Select.SecondWitness
import all Circuits.Descriptor
import Flock.HmRow
import Flock.TemplateQuery
import Flock.Canon
import Flock.JsonParse
import all Init.Data.Array.QSort.Basic
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.String.Defs
import all Init.Data.String.Basic
import all Init.Data.Ord.String
import all Init.Data.Ord.Basic
import all Init.Data.Range.Basic
import all Init.Data.List.Control
import all Init.Data.Repr
import all Init.Data.Int.Repr
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic
import all Std.Data.TreeMap.Raw.Basic
import all Std.Data.DTreeMap.Raw.Basic
import all Std.Data.DTreeMap.Internal.Operations
import all Std.Data.DTreeMap.Internal.Queries
import all Std.Data.DTreeMap.Internal.Balancing
import all Std.Data.DTreeMap.Internal.Def
public import Proofs.Flock.SessionLoop.Concrete.Fwd
import Batteries.Tactic.OpenPrivate
import Flock.Program
import all Init.Data.String.TakeDrop
import all Init.Data.String.Slice
import all Init.Data.String.Pattern.Basic
import all Init.Data.String.Pattern.String
import all Init.Data.String.Pattern.Char
import all Init.Data.String.Pattern.Pred
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
import all Init.Data.Array.Attach
import all Init.Data.Array.Subarray
import all Init.Data.Stream
import all Init.Data.Nat.Fold
import all Init.WF

/-! # The program's and the partition's digests

`progHex` and `partHex` are literals, so that the walks over META read them; here they are what Flock computes: the
program's SHA-512 as `TemplateQuery.Desc.parse` reads the program (`progHex_eq`), and the partition's as
`HmRow.partitionDigest` writes it (`partHex_eq`). `canon` recurses by well-founded recursion, which the kernel does not
unfold, so each proof evaluates its fuelled twin (`canon_fuel`). The program digest's tables are `Std.HashMap`s keyed by
`String.hash`, which the kernel does not evaluate, so `descParse_X` walks `Desc.parse` forward and reads each lookup off
the inserts that built its table (`hm_get_insert`, `hm_get_foldl`), its state stepped by its monad's equations
(`st_bind` and its siblings), and the well-founded walks over a definition's empty or missing binding record by their
values there (`fnRefs_of_null`, `fnRefs_of_empty`, `subFns_of_empty`).
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

open private Flock.Program.fnRefs from Circuits.Descriptor

namespace SpotFlock.InstanceStmt

open Flock Flock.ZkServer
open FlockSoundness.Discharge.Recursion.ZkServerJson (parse_printJson)
open FlockSoundness.Refine
open Properties.Spot.SecurityProofs (canon_fuel)
open Lean (Json)

theorem plain_progJ : plain progJ = true := by
  simp only [progJ]
  plain_json
theorem fjnorm_progJ : FlockSoundness.Discharge.Recursion.ZkServerJson.norm progJ = progJ := by kernel_rfl

theorem parse_prog : JsonParse.parse progText = .ok progJ := by
  rw [progText, parse_printJson plain_progJ, fjnorm_progJ]

namespace Decode

open Flock.Program
open Properties.Spot.SecurityProofs (canonFuel qsort_fuel)

theorem hm_get_insert {β : Type} (m : Std.HashMap String β) (k a : String) (v : β) :
    (m.insert k v)[a]? = if k == a then some v else m[a]? := by
  rw [Std.HashMap.getElem?_insert]
theorem hm_get_empty {β : Type} (a : String) : (({} : Std.HashMap String β))[a]? = none :=
  Std.HashMap.getElem?_empty

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


theorem st_bind {σ ε α β : Type} (x : StateT σ (Except ε) α) (f : α → StateT σ (Except ε) β) (s : σ) :
    (x >>= f) s = (x s >>= fun p => f p.1 p.2) := rfl
theorem st_get {σ ε : Type} (s : σ) : (get : StateT σ (Except ε) σ) s = .ok (s, s) := rfl

theorem st_modify {σ ε : Type} (f : σ → σ) (s : σ) : (modify f : StateT σ (Except ε) PUnit) s = .ok (⟨⟩, f s) := rfl
theorem st_pure {σ ε α : Type} (a : α) (s : σ) : (pure a : StateT σ (Except ε) α) s = .ok (a, s) := rfl
theorem st_throw {σ ε α : Type} (e : ε) (s : σ) : (throw e : StateT σ (Except ε) α) s = .error e := rfl
theorem st_ite {σ ε α : Type} (c : Prop) [Decidable c] (a b : StateT σ (Except ε) α) (s : σ) :
    (ite c a b) s = ite c (a s) (b s) := by split <;> rfl
theorem st_dite {σ ε α : Type} (c : Prop) [Decidable c] (a : c → StateT σ (Except ε) α) (b : ¬c → StateT σ (Except ε) α)
    (s : σ) : (dite c a b) s = dite c (fun h => a h s) (fun h => b h s) := by split <;> rfl
theorem st_run {σ ε α : Type} (x : StateT σ (Except ε) α) (s : σ) : x.run s = x s := rfl

theorem mapM_nil {ε α β : Type} (f : α → Except ε β) : (#[] : Array α).mapM f = pure #[] := by
  simp only [Array.mapM_eq_mapM_toList, List.mapM_nil]; rfl
theorem mapM_one {ε α β : Type} (f : α → Except ε β) (a : α) :
    #[a].mapM f = (do let x ← f a; pure #[x]) := by
  simp only [Array.mapM_eq_mapM_toList, List.mapM_cons, List.mapM_nil]; cases f a <;> rfl
theorem mapM_pair {ε α β : Type} (f : α → Except ε β) (a b : α) :
    #[a, b].mapM f = (do let x ← f a; let y ← f b; pure #[x, y]) := by
  simp only [Array.mapM_eq_mapM_toList, List.mapM_cons, List.mapM_nil]; cases f a <;> cases f b <;> rfl

theorem prod_fst {α β : Type} (a : α) (b : β) : (a, b).1 = a := rfl
theorem prod_snd {α β : Type} (a : α) (b : β) : (a, b).2 = b := rfl

theorem recordFunctionIds_plain (fid : String) (h : fid.toList.contains '{' = false) : recordFunctionIds fid = #[] := by
  have : fid.contains '{' = false := by
    rw [String.contains_char_eq]; simpa using h
  simp [recordFunctionIds, recordPairs, this]

theorem fnRefs_of_null (acc : Array String) (j : Json) (h : j = .null) : Flock.Program.fnRefs acc j = acc := by
  subst h; rw [Flock.Program.fnRefs.eq_def]
theorem fnRefs_of_empty (acc : Array String) (j : Json) (h : j = Json.mkObj []) : Flock.Program.fnRefs acc j = acc := by
  subst h; rw [Flock.Program.fnRefs.eq_def]; first | rfl | decide | simp
theorem subFns_of_empty (d : String → Option String) (j : Json) (h : j = Json.mkObj []) :
    subFns d j = pure (Json.mkObj []) := by
  subst h; rw [subFns.eq_def]
  first | rfl | (simp only [Array.mapM_eq_mapM_toList, List.mapM_nil]; rfl) | simp [Array.mapM_eq_mapM_toList]
theorem st_discard {σ ε α : Type} (x : StateT σ (Except ε) α) (s : σ) :
    (discard x) s = (x s >>= fun p => .ok (⟨⟩, p.2)) := rfl
theorem st_lift {σ ε α : Type} (v : Except ε α) (s : σ) :
    (monadLift v : StateT σ (Except ε) α) s = (v >>= fun a => .ok (a, s)) := rfl
theorem st_forIn_empty {σ ε α β : Type} (xs : Array α) (b : β) (f : α → β → StateT σ (Except ε) (ForInStep β)) (s : σ)
    (h : xs.size = 0) : (forIn xs b f) s = .ok (b, s) := by
  rw [Array.eq_empty_of_size_eq_zero h]; rfl
theorem st_forIn_one {σ ε α β : Type} [Inhabited α] (xs : Array α) (b : β)
    (f : α → β → StateT σ (Except ε) (ForInStep β)) (s : σ) (h : xs.size = 1) :
    (forIn xs b f) s = (f xs[0]! b >>= fun r => pure (match r with | .done b => b | .yield b => b)) s := by
  obtain ⟨l⟩ := xs
  match l, h with
  | [a], _ =>
    rw [← Array.forIn_toList]
    simp only [List.forIn_cons, List.forIn_nil]
    show (f a b >>= _) s = (f a b >>= _) s
    rw [st_bind, st_bind]
    rcases f a b s with _ | ⟨_ | _, _⟩ <;> rfl
theorem st_forIn_two {σ ε α β : Type} [Inhabited α] (xs : Array α) (b : β)
    (f : α → β → StateT σ (Except ε) (ForInStep β)) (s : σ) (h : xs.size = 2) :
    (forIn xs b f) s = (f xs[0]! b >>= fun r => match r with
      | .done b => pure b
      | .yield b => f xs[1]! b >>= fun r => pure (match r with | .done b => b | .yield b => b)) s := by
  obtain ⟨l⟩ := xs
  match l, h with
  | [a0, a1], _ =>
    rw [← Array.forIn_toList]
    simp only [List.forIn_cons, List.forIn_nil]
    have h0 : ({ toList := [a0, a1] } : Array α)[0]! = a0 := rfl
    have h1 : ({ toList := [a0, a1] } : Array α)[1]! = a1 := rfl
    simp only [h0, h1]
    rw [st_bind, st_bind]
    rcases f a0 b s with _ | ⟨_ | b1, s1⟩
    · rfl
    · rfl
    · rw [Fwd.bind_ok, Fwd.bind_ok]
      dsimp only
      rw [st_bind, st_bind]
      rcases f a1 b1 s1 with _ | ⟨_ | _, _⟩ <;> rfl

set_option maxHeartbeats 1600000 in
fwd_def descX descParse_X : TemplateQuery.Desc.parse progBytes by [fromUTF8?_toUTF8, parse_prog, canon_fuel 12,
    qsort_fuel, hm_get_insert, hm_get_empty, hm_get_foldl, st_bind, st_get, st_modify, st_pure, st_throw, st_ite,
    st_dite, st_run, st_discard, st_lift, st_forIn_empty, st_forIn_one, st_forIn_two, mapM_nil, mapM_one, mapM_pair,
    prod_fst, prod_snd, recordFunctionIds_plain, fnRefs_of_null, fnRefs_of_empty, subFns_of_empty]
  block [Std.HashMap.insert, Sha512.hash]
  shallow []

theorem descX_hex : hexOf descX.sha512 = progHex := by decide +kernel

end Decode

/-- **The program's digest is `progHex`.** -/
theorem progHex_eq : ((fun d => hexOf d.sha512) <$> TemplateQuery.Desc.parse progBytes).toOption = some progHex := by
  rw [Decode.descParse_X]
  exact congrArg some Decode.descX_hex

/-- **The partition's digest is `partHex`.** -/
theorem partHex_eq : (hexOf <$> HmRow.partitionDigest partObj).toOption = some partHex := by
  unfold HmRow.partitionDigest
  simp only [canon_fuel 8]
  decide +kernel

/-- `isSha512Hex` through `toList`: the kernel does not run `String.all`, so no walk decides it on a digest by itself
(META's two, `prog_hex` and `part_hex`, and the partition's program, `program_hex`). -/
theorem sha512Hex_of (s : String)
    (h : (s.toList.length == 128 && s.toList.all fun c => c.isDigit || ('a' ≤ c && c ≤ 'f')) = true) :
    HmRow.isSha512Hex s = true := by
  rw [HmRow.isSha512Hex, String.all_bool_eq, ← String.length_toList]
  exact h

end SpotFlock.InstanceStmt
