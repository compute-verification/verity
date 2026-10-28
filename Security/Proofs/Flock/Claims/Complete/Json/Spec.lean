import Flock.Canon
import Std.Data.TreeMap.Raw.Lemmas
import Proofs.Flock.Claims.Complete.Json.Value
import Definitions.Claims
import Proofs.Flock.Claims.JsonEq
import Proofs.Flock.Claims.Complete.Prover.Print
import Proofs.Flock.Claims.Complete.Json.RoundTrip

/-!
# What the record's round trip needs from the JSON library

Nothing the session record's round trip reads is assumed (the prover's `printJ`, a total printer of
`Json.compress`'s bytes, then `Record.decodeWith`'s, `liveOf`'s and `recordDraw`'s `parse`): the verifier's parser
`Flock.JsonParse.parse` and its `canon` are total, the parser reads back what `printJ` and `canon` print
(`RoundTrip.lean`: `parse_print`, `parse_canon`), builds well-formed trees (`Parse.lean`), and `canon` is the same on
alike values (`Canon.lean`). The verifier reads no `Json` `==` (a `partial` definition no proof unfolds): it tests
flags by `matches`. Alike (`Same`), printable (`Good`) and well-formed (`Wf`) values are `Value.lean`'s. Here are the
lemmas on alike values the proofs read: alike
leaves are equal, and alike objects have alike values at each key (`Same.getObjVal`) and the same missing keys
(`Same.getObjVal_none`), and alike values have one form (`Same.jsonEquiv`), so the verifier's draw check (`drawEq`)
passes on them. The record's numbers are all `Lean.toJson` of naturals, so printable.
-/

namespace FlockComplete.JsonRT

open Lean (Json JsonNumber)
open Flock

export Claims.Json (Same Good Ascii Wf)

/-! ## Alike values -/

theorem _root_.Claims.Json.Same.symm : ∀ {a b : Json}, Same a b → Same b a
  | _, _, .null => .null
  | _, _, .bool b => .bool b
  | _, _, .num n => .num n
  | _, _, .str s => .str s
  | _, _, .arr hs he => .arr hs.symm fun i h₁ h₂ => (he i h₂ h₁).symm
  | _, _, .obj hm hn hk hv => .obj hn hm (fun k => (hk k).symm) fun k a b ha hb => (hv k b a hb ha).symm

/-- **Alike values are equivalent** (`FlockDeploy.JsonEquiv`): one form, whatever their objects' key order. -/
theorem _root_.Claims.Json.Same.jsonEquiv : ∀ {a b : Json}, Same a b → FlockDeploy.JsonEquiv a b
  | _, _, .null => rfl
  | _, _, .bool _ => rfl
  | _, _, .num _ => rfl
  | _, _, .str _ => rfl
  | _, _, .arr hs he => (FlockDeploy.jsonEquiv_arr_iff _ _).2 ⟨hs, fun i h₁ h₂ => (he i h₁ h₂).jsonEquiv⟩
  | _, _, .obj hm hn hk hv =>
    (FlockDeploy.jsonEquiv_obj_iff hm hn).2 ⟨hk, fun k a b ha hb => (hv k a b ha hb).jsonEquiv⟩

theorem same_str {s : String} {j : Json} (h : Same (.str s) j) : j = .str s := by cases h; rfl

theorem same_null {j : Json} (h : Same .null j) : j = .null := by cases h; rfl

theorem same_bool {b : Bool} {j : Json} (h : Same (.bool b) j) : j = .bool b := by cases h; rfl

theorem same_num {n : JsonNumber} {j : Json} (h : Same (.num n) j) : j = .num n := by cases h; rfl

theorem same_arr {a : Array Json} {j : Json} (h : Same (.arr a) j) :
    ∃ b : Array Json, j = .arr b ∧ a.size = b.size ∧ ∀ (i : Nat) (h₁ : i < a.size) (h₂ : i < b.size), Same a[i] b[i] := by
  cases h with
  | arr hs he => exact ⟨_, rfl, hs, he⟩

theorem same_obj {m : Std.TreeMap.Raw String Json} {j : Json} (h : Same (.obj m) j) :
    ∃ n : Std.TreeMap.Raw String Json, j = .obj n ∧ ∀ k, (m.get? k).isSome = (n.get? k).isSome ∧
      ∀ a b, m.get? k = some a → n.get? k = some b → Same a b := by
  cases h with
  | obj _ _ hk hv => exact ⟨_, rfl, fun k => ⟨hk k, hv k⟩⟩

/-- **A key's value in alike objects**: the value at a key of one is alike the other's at that key. -/
theorem _root_.Claims.Json.Same.getObjVal {j j' : Json} (h : Same j j') {k : String} {v : Json}
    (hv : j.getObjVal? k = .ok v) : ∃ v', j'.getObjVal? k = .ok v' ∧ Same v v' := by
  cases j with
  | obj m =>
    obtain ⟨n, rfl, hn⟩ := same_obj h
    simp only [Json.getObjVal?] at hv ⊢
    cases hm : m.get? k with
    | none => rw [hm] at hv; cases hv
    | some a =>
      rw [hm] at hv
      cases hv
      have hs := (hn k).1
      rw [hm] at hs
      cases hn' : n.get? k with
      | none => rw [hn'] at hs; cases hs
      | some b => exact ⟨b, rfl, (hn k).2 _ _ hm hn'⟩
  | _ => cases hv

/-- **A missing key of alike objects** is missing from both. -/
theorem _root_.Claims.Json.Same.getObjVal_none {j j' : Json} (h : Same j j') {k : String}
    {m : Std.TreeMap.Raw String Json} (hj : j = .obj m) (hk : m.get? k = none) : ∃ e, j'.getObjVal? k = .error e := by
  subst hj
  obtain ⟨n, rfl, hn⟩ := same_obj h
  have hs := (hn k).1
  rw [hk] at hs
  simp only [Json.getObjVal?]
  cases hn' : n.get? k with
  | none => exact ⟨_, rfl⟩
  | some b => rw [hn'] at hs; cases hs

/-! ## The masked proofs' names

The masked completeness proofs (`Zk/`, `ZkTables/`) name the oracle's 64-byte outputs by this; a statement the lock lists
reads `Claims.Out64`. -/

/-- `Claims.Out64`: the oracle's outputs are 64 bytes. -/
abbrev Out64 (H : Claims.Msg → Claims.Msg) : Prop := Claims.Out64 H

end FlockComplete.JsonRT
