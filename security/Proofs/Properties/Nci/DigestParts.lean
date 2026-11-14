module

public import Proofs.Properties.Nci.DigestDecode
public import Proofs.Properties.Nci.DigestEntries
public import Proofs.Properties.Nci.DigestRefs
import Batteries.Tactic.OpenPrivate
import all Circuits.Program
import all Lean.Data.Json.Basic

/-!
# Equivalent canonical entries hold equivalent flat parts

The canonical entry writes each reference as its flat parts (`written`): the one part, or `{"concat": parts}`. A flat
part the decoder parses holds no concatenation (`PartsRel`), so a lone part is never equivalent to a written
concatenation, and two equivalent written sequences hold the same number of parts, equivalent one by one
(`written_jsonEquiv`).

A part with neither an alias nor a concatenation parses to the same parts from equivalent well-formed values, at any
fuel and with any earlier arguments (`part_jsonEquiv`): the decoder reads the same fields, its index and leaf reads
agree on equivalent values, and its space reads agree even where the parts drop them.
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Lean (Json)
open Claims.Json (Wf)
open FlockDeploy (JsonEquiv jsonEquiv_arr_iff)
open NciCheck.DefinitionDigest (written)

theorem concat_mkObj1 (v : Json) : ((Json.mkObj [("concat", v)]).getObjVal? "concat").toOption = some v := by
  simp only [Json.mkObj, Json.getObjVal?]
  rw [Std.TreeMap.Raw.get?_eq_getElem?,
    Std.TreeMap.Raw.getElem?_ofList_of_mem Std.ReflCmp.compare_self (by simp) List.mem_cons_self]
  rfl

theorem written_one {ps : Array Json} (h : ps.size = 1) : written ps = ps[0] := by
  rcases ps with ⟨_ | ⟨p, _ | ⟨q, l⟩⟩⟩ <;> simp at h
  rfl

theorem written_many {ps : Array Json} (h : ¬ ps.size = 1) : written ps = Json.mkObj [("concat", .arr ps)] := by
  rcases ps with ⟨_ | ⟨p, _ | ⟨q, l⟩⟩⟩
  · rfl
  · simp at h
  · rfl

/-- Equivalent flat sequences, as the canonical entry writes them, of parts with no concatenation: the same parts up
to equivalence, one by one. -/
theorem written_jsonEquiv {ps ps' : Array Json} (hw : ∀ p ∈ ps, Wf p) (hw' : ∀ p ∈ ps', Wf p)
    (hc : ∀ p ∈ ps, (p.getObjVal? "concat").toOption = none) (hc' : ∀ p ∈ ps', (p.getObjVal? "concat").toOption = none)
    (h : JsonEquiv (written ps) (written ps')) :
    ps.size = ps'.size ∧ ∀ i (h1 : i < ps.size) (h2 : i < ps'.size), JsonEquiv ps[i] ps'[i] := by
  have one : ∀ (p : Json) (qs : Array Json), Wf p → (∀ q ∈ qs, Wf q) → (p.getObjVal? "concat").toOption = none →
      ¬ JsonEquiv p (Json.mkObj [("concat", .arr qs)]) := by
    intro p qs hp hq hpc he
    have := field_jsonEquiv hp (wf_mkObj1 (.arr hq)) he "concat"
    rw [hpc, concat_mkObj1] at this
    exact this
  by_cases h1 : ps.size = 1 <;> by_cases h1' : ps'.size = 1
  · rw [written_one h1, written_one h1'] at h
    refine ⟨by omega, fun i hi hi' => ?_⟩
    obtain rfl : i = 0 := by omega
    exact h
  · rw [written_one h1, written_many h1'] at h
    exact absurd h (one _ _ (hw _ (Array.getElem_mem _)) hw' (hc _ (Array.getElem_mem _)))
  · rw [written_many h1, written_one h1'] at h
    exact absurd h.symm (one _ _ (hw' _ (Array.getElem_mem _)) hw (hc' _ (Array.getElem_mem _)))
  · rw [written_many h1, written_many h1'] at h
    have := field_jsonEquiv (wf_mkObj1 (.arr hw)) (wf_mkObj1 (.arr hw')) h "concat"
    rw [concat_mkObj1, concat_mkObj1] at this
    exact (jsonEquiv_arr_iff ps ps').1 this

end NciCheck.Properties.Record.SecurityProofs

end

namespace NciCheck.Properties.Record.SecurityProofs

open Lean (Json)
open Flock Flock.Program
open Claims.Json (Wf)
open FlockDeploy (JsonEquiv jsonEquiv_arr_iff jsonForm)
open private natJ? intJ? from Circuits.Program

/-- The decoder's reference space: `p`, a parameter; `n`, a node's output. -/
private def spaceOf (s : Json) : Option Bool := match s with
  | .str "p" => some true
  | .str "n" => some false
  | _ => none

/-- The decoder's leaf range check: collection `i` of space `s` holds leaves `lo` to `hi`. -/
private def withinOf (leaves : Bool → Nat → Option Nat) (s i : Json) (lo hi : Int) : Option (Bool × Nat) :=
  match spaceOf s, natJ? i with
  | some p, some i => match leaves p i with
    | some n => if 0 ≤ lo && hi < n then some (p, i) else none
    | none => none
  | _, _ => none

private theorem parse_affine {n : Nat} {j a : Json} {leaves : Bool → Nat → Option Nat} {args : Nat → Nat → Option Seq}
    {qs : Array Part} (h : parsePartsIn (n + 1) j leaves args = .ok qs) (h1 : (j.getObjVal? "alias").toOption = none)
    (h2 : (j.getObjVal? "concat").toOption = none) (h3 : j.getObjVal? "affine" = .ok a) :
    ∃ s i b c b0 c0, a = .arr #[s, i, b, c] ∧ intJ? b = some b0 ∧ natJ? c = some c0 ∧
      ((c0 = 0 ∧ qs = #[.affine (spaceOf s == some true) 0 0 0]) ∨
        ∃ p i0, c0 ≠ 0 ∧ withinOf leaves s i b0 (b0 + c0 - 1) = some (p, i0) ∧ qs = #[.affine p i0 b0.toNat c0]) := by
  cases h1' : j.getObjVal? "alias" with
  | ok _ => rw [h1'] at h1; cases h1
  | error e1 =>
  cases h2' : j.getObjVal? "concat" with
  | ok _ => rw [h2'] at h2; cases h2
  | error e2 =>
  delta parsePartsIn parsePartsIn._f at h
  cases j with
  | obj kvs =>
    dsimp only at h
    rw [h1', h2', h3] at h
    dsimp only at h
    rcases a with _ | _ | _ | _ | ⟨⟨l⟩⟩ | _
    all_goals try (injection h; done)
    rcases l with _ | ⟨s, _ | ⟨i, _ | ⟨b, _ | ⟨c, _ | ⟨y, l⟩⟩⟩⟩⟩
    all_goals try (injection h; done)
    conv at h => lhs; whnf
    dsimp only [Array.getLit, List.getElem_toArray, List.getElem_cons_zero, List.getElem_cons_succ] at h
    split at h
    · rename_i b0 c0 hbc
      simp only [Prod.mk.injEq] at hbc
      split at h
      · rename_i hc0
        injection h with h
        exact ⟨s, i, b, c, b0, c0, rfl, hbc.1, hbc.2, .inl ⟨by simpa using hc0, h.symm⟩⟩
      · rename_i hc0
        split at h
        · rename_i p i0 hw
          injection h with h
          exact ⟨s, i, b, c, b0, c0, rfl, hbc.1, hbc.2, .inr ⟨p, i0, by simpa using hc0, hw, h.symm⟩⟩
        · exact absurd h (by intro h; injection h)
    · exact absurd h (by intro h; injection h)
  | _ => cases h

private theorem ite_t {α : Sort _} {c : Prop} [Decidable c] {a b : α} (h : c) : (if c then a else b) = a := by
  simp [h]

private theorem ite_f {α : Sort _} {c : Prop} [Decidable c] {a b : α} (h : ¬c) : (if c then a else b) = b := by
  simp [h]

private theorem affine_fwd {n : Nat} {j s i b c : Json} {leaves : Bool → Nat → Option Nat} {args : Nat → Nat → Option Seq}
    {qs : Array Part} {e1 e2 : String} {b0 : Int} {c0 : Nat}
    (h1 : j.getObjVal? "alias" = .error e1) (h2 : j.getObjVal? "concat" = .error e2)
    (h3 : j.getObjVal? "affine" = .ok (.arr #[s, i, b, c])) (hb : intJ? b = some b0) (hc : natJ? c = some c0)
    (hq : (c0 = 0 ∧ qs = #[.affine (spaceOf s == some true) 0 0 0]) ∨
        ∃ p i0, c0 ≠ 0 ∧ withinOf leaves s i b0 (b0 + c0 - 1) = some (p, i0) ∧ qs = #[.affine p i0 b0.toNat c0]) :
    parsePartsIn (n + 1) j leaves args = .ok qs := by
  delta parsePartsIn parsePartsIn._f
  cases j with
  | obj kvs =>
    dsimp only
    rw [h1, h2, h3]
    dsimp only
    conv => lhs; whnf
    dsimp only [Array.getLit, List.getElem_toArray, List.getElem_cons_zero, List.getElem_cons_succ]
    rw [hb, hc]
    dsimp only
    rcases hq with ⟨hc0, rfl⟩ | ⟨p, i0, hc0, hw, rfl⟩
    · rw [ite_t (by simp [hc0])]; rfl
    · rw [ite_f (by simpa using hc0)]
      split
      · rename_i p' i' heq
        change withinOf leaves s i b0 (b0 + c0 - 1) = some (p', i') at heq
        rw [hw] at heq
        cases heq
        rfl
      · rename_i heq
        exact absurd hw (heq p i0)
  | _ => cases h3

/-- The leaves at the corners of a strided reference, whose least and greatest the decoder range-checks. -/
private def stridedVals (b : Int) (inner : Nat) (os is : Int) (c : Nat) : List Int :=
  let q := (c - 1) / inner
  let last := (c - 1) % inner
  let corners : List (Nat × Nat) :=
    [(0, 0), (0, min inner c - 1), (q, 0), (q, last)] ++ (if q > 0 then [(q - 1, inner - 1)] else [])
  corners.map fun (x, y) => b + (x : Int) * os + (y : Int) * is

private theorem parse_strided {n : Nat} {j a : Json} {leaves : Bool → Nat → Option Nat} {args : Nat → Nat → Option Seq}
    {qs : Array Part} {e1 e2 e3 : String} (h : parsePartsIn (n + 1) j leaves args = .ok qs)
    (h1 : j.getObjVal? "alias" = .error e1) (h2 : j.getObjVal? "concat" = .error e2)
    (h3 : j.getObjVal? "affine" = .error e3) (h4 : j.getObjVal? "strided" = .ok a) :
    ∃ s i b inner os is c b0 inner0 os0 is0 c0, a = .arr #[s, i, b, inner, os, is, c] ∧ intJ? b = some b0 ∧
      natJ? inner = some inner0 ∧ intJ? os = some os0 ∧ intJ? is = some is0 ∧ natJ? c = some c0 ∧ inner0 ≠ 0 ∧
      ((c0 = 0 ∧ qs = #[.strided (spaceOf s == some true) 0 0 inner0 os0 is0 0]) ∨
        ∃ p i0, c0 ≠ 0 ∧ withinOf leaves s i ((stridedVals b0 inner0 os0 is0 c0).foldl min
            (stridedVals b0 inner0 os0 is0 c0).head!) ((stridedVals b0 inner0 os0 is0 c0).foldl max
            (stridedVals b0 inner0 os0 is0 c0).head!) = some (p, i0) ∧
          qs = #[.strided p i0 b0 inner0 os0 is0 c0]) := by
  delta parsePartsIn parsePartsIn._f at h
  cases j with
  | obj kvs =>
    dsimp only at h
    rw [h1, h2, h3, h4] at h
    dsimp only at h
    rcases a with _ | _ | _ | _ | ⟨⟨l⟩⟩ | _
    all_goals try (injection h; done)
    rcases l with _ | ⟨s, _ | ⟨i, _ | ⟨b, _ | ⟨inner, _ | ⟨os, _ | ⟨is, _ | ⟨c, _ | ⟨y, l⟩⟩⟩⟩⟩⟩⟩⟩
    all_goals try (injection h; done)
    conv at h => lhs; whnf
    dsimp only [Array.getLit, List.getElem_toArray, List.getElem_cons_zero, List.getElem_cons_succ] at h
    split at h
    · rename_i b0 inner0 os0 is0 c0 hbc
      simp only [Prod.mk.injEq] at hbc
      obtain ⟨hb, hi, ho, his, hc⟩ := hbc
      split at h
      · exact absurd h (by intro h; injection h)
      · rename_i hi0
        split at h
        · rename_i hc0
          injection h with h
          exact ⟨s, i, b, inner, os, is, c, b0, inner0, os0, is0, c0, rfl, hb, hi, ho, his, hc,
            by simpa using hi0, .inl ⟨by simpa using hc0, h.symm⟩⟩
        · rename_i hc0
          split at h
          · rename_i p i0 hw
            injection h with h
            exact ⟨s, i, b, inner, os, is, c, b0, inner0, os0, is0, c0, rfl, hb, hi, ho, his, hc,
              by simpa using hi0, .inr ⟨p, i0, by simpa using hc0, hw, h.symm⟩⟩
          · exact absurd h (by intro h; injection h)
    · exact absurd h (by intro h; injection h)
  | _ => cases h

private theorem strided_fwd {n : Nat} {j s i b inner os is c : Json} {leaves : Bool → Nat → Option Nat}
    {args : Nat → Nat → Option Seq} {qs : Array Part} {e1 e2 e3 : String} {b0 os0 is0 : Int} {inner0 c0 : Nat}
    (h1 : j.getObjVal? "alias" = .error e1) (h2 : j.getObjVal? "concat" = .error e2)
    (h3 : j.getObjVal? "affine" = .error e3) (h4 : j.getObjVal? "strided" = .ok (.arr #[s, i, b, inner, os, is, c]))
    (hb : intJ? b = some b0) (hi : natJ? inner = some inner0) (ho : intJ? os = some os0)
    (his : intJ? is = some is0) (hc : natJ? c = some c0) (hi0 : inner0 ≠ 0)
    (hq : (c0 = 0 ∧ qs = #[.strided (spaceOf s == some true) 0 0 inner0 os0 is0 0]) ∨
        ∃ p i0, c0 ≠ 0 ∧ withinOf leaves s i ((stridedVals b0 inner0 os0 is0 c0).foldl min
            (stridedVals b0 inner0 os0 is0 c0).head!) ((stridedVals b0 inner0 os0 is0 c0).foldl max
            (stridedVals b0 inner0 os0 is0 c0).head!) = some (p, i0) ∧
          qs = #[.strided p i0 b0 inner0 os0 is0 c0]) :
    parsePartsIn (n + 1) j leaves args = .ok qs := by
  delta parsePartsIn parsePartsIn._f
  cases j with
  | obj kvs =>
    dsimp only
    rw [h1, h2, h3, h4]
    dsimp only
    conv => lhs; whnf
    dsimp only [Array.getLit, List.getElem_toArray, List.getElem_cons_zero, List.getElem_cons_succ]
    rw [hb, hi, ho, his, hc]
    dsimp only
    rw [ite_f (by simpa using hi0)]
    rcases hq with ⟨hc0, rfl⟩ | ⟨p, i0, hc0, hw, rfl⟩
    · rw [ite_t (by simp [hc0])]; rfl
    · rw [ite_f (by simpa using hc0)]
      split
      · rename_i p' i' heq
        change withinOf leaves s i ((stridedVals b0 inner0 os0 is0 c0).foldl min
            (stridedVals b0 inner0 os0 is0 c0).head!) ((stridedVals b0 inner0 os0 is0 c0).foldl max
            (stridedVals b0 inner0 os0 is0 c0).head!) = some (p', i') at heq
        rw [hw] at heq
        cases heq
        rfl
      · rename_i heq
        exact absurd hw (heq p i0)
  | _ => cases h4

/-- The decoder's reading of one explicit reference `[space, index, leaf]`; `msg` is its refusal. -/
private def explicitItem (leaves : Bool → Nat → Option Nat) (msg : String) (r : Json) : V Ref := match r with
  | .arr #[s, i, l] => match intJ? l with
    | some l => match withinOf leaves s i l l with
      | some (p, i) => pure ⟨p, i, l.toNat⟩
      | none => throw msg
    | none => throw msg
  | _ => throw msg

private theorem parse_explicit {n : Nat} {j a : Json} {leaves : Bool → Nat → Option Nat} {args : Nat → Nat → Option Seq}
    {qs : Array Part} {e1 e2 e3 e4 : String} (h : parsePartsIn (n + 1) j leaves args = .ok qs)
    (h1 : j.getObjVal? "alias" = .error e1) (h2 : j.getObjVal? "concat" = .error e2)
    (h3 : j.getObjVal? "affine" = .error e3) (h4 : j.getObjVal? "strided" = .error e4)
    (h5 : j.getObjVal? "explicit" = .ok a) :
    ∃ items msg rs, a = .arr items ∧ items.mapM (explicitItem leaves msg) = .ok rs ∧ qs = #[.explicit rs] := by
  delta parsePartsIn parsePartsIn._f at h
  cases j with
  | obj kvs =>
    dsimp only at h
    rw [h1, h2, h3, h4, h5] at h
    dsimp only at h
    rcases a with _ | _ | _ | _ | items | _
    all_goals try (injection h; done)
    conv at h => lhs; whnf
    split at h
    · exact absurd h (by intro h; injection h)
    · rename_i rs heq
      injection h with h
      exact ⟨items, _, rs, rfl, heq, h.symm⟩
  | _ => cases h

private theorem explicit_fwd {n : Nat} {j : Json} {items : Array Json} {leaves : Bool → Nat → Option Nat}
    {args : Nat → Nat → Option Seq} {rs : Array Ref} {e1 e2 e3 e4 : String}
    (h1 : j.getObjVal? "alias" = .error e1) (h2 : j.getObjVal? "concat" = .error e2)
    (h3 : j.getObjVal? "affine" = .error e3) (h4 : j.getObjVal? "strided" = .error e4)
    (h5 : j.getObjVal? "explicit" = .ok (.arr items)) (H : ∀ msg, items.mapM (explicitItem leaves msg) = .ok rs) :
    parsePartsIn (n + 1) j leaves args = .ok #[.explicit rs] := by
  delta parsePartsIn parsePartsIn._f
  cases j with
  | obj kvs =>
    dsimp only
    rw [h1, h2, h3, h4, h5]
    dsimp only
    conv => lhs; whnf
    have hm := H s!"references {(Json.obj kvs).compress} name no leaf of a parameter or a node"
    split
    · rename_i err heq
      exact absurd (heq.symm.trans hm) (by intro h; injection h)
    · rename_i v heq
      cases heq.symm.trans hm
      rfl
  | _ => cases h5

private theorem spaceOf_jsonEquiv {a b : Json} (h : JsonEquiv a b) : spaceOf a = spaceOf b := by
  cases a <;> cases b <;> simp_all [JsonEquiv, jsonForm, spaceOf]

private theorem withinOf_jsonEquiv {leaves : Bool → Nat → Option Nat} {s s' i i' : Json} (hs : JsonEquiv s s')
    (hi : JsonEquiv i i') (lo hi' : Int) : withinOf leaves s i lo hi' = withinOf leaves s' i' lo hi' := by
  unfold withinOf
  rw [spaceOf_jsonEquiv hs, natJ_jsonEquiv hi]

private theorem arr_of_jsonEquiv {a : Array Json} {b : Json} (h : JsonEquiv (.arr a) b) :
    ∃ ys, b = .arr ys ∧ a.size = ys.size ∧ ∀ k (h₁ : k < a.size) (h₂ : k < ys.size), JsonEquiv a[k] ys[k] := by
  cases b with
  | arr ys => exact ⟨ys, rfl, (jsonEquiv_arr_iff a ys).1 h⟩
  | _ => simp [JsonEquiv, jsonForm] at h

private theorem explicitItem_jsonEquiv {leaves : Bool → Nat → Option Nat} {m m' : String} {r r' : Json} {x : Ref}
    (h : explicitItem leaves m r = .ok x) (he : JsonEquiv r r') : explicitItem leaves m' r' = .ok x := by
  rcases r with _ | _ | _ | _ | ⟨⟨l⟩⟩ | _
  all_goals try (delta explicitItem at h; injection h; done)
  obtain ⟨⟨l'⟩, rfl, hsz, hel⟩ := arr_of_jsonEquiv he
  rcases l with _ | ⟨s, _ | ⟨i, _ | ⟨k, _ | ⟨y, l⟩⟩⟩⟩
  all_goals try (delta explicitItem at h; injection h; done)
  rcases l' with _ | ⟨s', _ | ⟨i', _ | ⟨k', _ | ⟨y', l'⟩⟩⟩⟩
  all_goals try simp at hsz
  have hs : JsonEquiv s s' := hel 0 (by simp) (by simp)
  have hi : JsonEquiv i i' := hel 1 (by simp) (by simp)
  have hk : JsonEquiv k k' := hel 2 (by simp) (by simp)
  delta explicitItem at h ⊢
  conv at h => lhs; whnf
  conv => lhs; whnf
  dsimp only [Array.getLit, List.getElem_toArray, List.getElem_cons_zero, List.getElem_cons_succ] at h ⊢
  rw [← intJ_jsonEquiv hk]
  split at h
  · rw [← withinOf_jsonEquiv hs hi]
    split at h
    · exact h
    · exact absurd h (by intro h; injection h)
  · exact absurd h (by intro h; injection h)

private theorem list_mapM_transport {α β ε : Type} {f g : α → Except ε β} {R : α → α → Prop}
    (H : ∀ a a' x, R a a' → f a = .ok x → g a' = .ok x) :
    ∀ (l l' : List α) (rs : List β), l.length = l'.length → (∀ k (h₁ : k < l.length) (h₂ : k < l'.length), R l[k] l'[k]) →
      l.mapM f = .ok rs → l'.mapM g = .ok rs
  | [], [], rs, _, _, h => h
  | a :: l, a' :: l', rs, hl, hR, h => by
    simp only [List.mapM_cons] at h ⊢
    cases hfa : f a with
    | error e => rw [hfa] at h; cases h
    | ok x =>
      rw [hfa] at h
      cases hfl : l.mapM f with
      | error e => simp only [hfl] at h; cases h
      | ok xs =>
        simp only [hfl] at h
        rw [H a a' x (hR 0 (by simp) (by simp)) hfa,
          list_mapM_transport H l l' xs (by simpa using hl) (fun k h₁ h₂ => hR (k + 1) (by simpa using h₁)
            (by simpa using h₂)) hfl]
        exact h
  | [], _ :: _, _, hl, _, _ => by simp at hl
  | _ :: _, [], _, hl, _, _ => by simp at hl

private theorem field_cases {j j' : Json} (hj : Wf j) (hj' : Wf j') (he : JsonEquiv j j') (k : String) :
    (∃ e e', j.getObjVal? k = .error e ∧ j'.getObjVal? k = .error e') ∨
      ∃ a a', j.getObjVal? k = .ok a ∧ j'.getObjVal? k = .ok a' ∧ JsonEquiv a a' := by
  have F := field_jsonEquiv hj hj' he k
  cases h : j.getObjVal? k <;> cases h' : j'.getObjVal? k <;> rw [h, h'] at F <;>
    simp_all [OptEquiv, Except.toOption]

private theorem parse_none {n : Nat} {j : Json} {leaves : Bool → Nat → Option Nat} {args : Nat → Nat → Option Seq}
    {qs : Array Part} {e1 e2 e3 e4 e5 : String} (h1 : j.getObjVal? "alias" = .error e1)
    (h2 : j.getObjVal? "concat" = .error e2) (h3 : j.getObjVal? "affine" = .error e3)
    (h4 : j.getObjVal? "strided" = .error e4) (h5 : j.getObjVal? "explicit" = .error e5) :
    parsePartsIn (n + 1) j leaves args ≠ .ok qs := by
  intro h
  delta parsePartsIn parsePartsIn._f at h
  cases j with
  | obj kvs =>
    dsimp only at h
    rw [h1, h2, h3, h4, h5] at h
    injection h
  | _ => cases h

end NciCheck.Properties.Record.SecurityProofs

public section

namespace NciCheck.Properties.Record.SecurityProofs

open Lean (Json)
open Flock Flock.Program
open Claims.Json (Wf)
open FlockDeploy (JsonEquiv)

/-- A part with no alias and no concatenation parses alike from equivalent well-formed values, at any fuel and with
any earlier arguments. -/
theorem part_jsonEquiv {n m : Nat} {j j' : Json} {leaves : Bool → Nat → Option Nat}
    {args args' : Nat → Nat → Option Seq} {qs : Array Part} (hj : Wf j) (hj' : Wf j') (he : JsonEquiv j j')
    (ha : (j.getObjVal? "alias").toOption = none) (hc : (j.getObjVal? "concat").toOption = none)
    (h : parsePartsIn (n + 1) j leaves args = .ok qs) : parsePartsIn (m + 1) j' leaves args' = .ok qs := by
  have C := field_cases hj hj' he
  obtain ⟨e1, e1', h1, h1'⟩ | ⟨_, _, h1, -⟩ := C "alias"
  case inr => rw [h1] at ha; cases ha
  obtain ⟨e2, e2', h2, h2'⟩ | ⟨_, _, h2, -⟩ := C "concat"
  case inr => rw [h2] at hc; cases hc
  obtain ⟨e3, e3', h3, h3'⟩ | ⟨a, a', h3, h3', hae⟩ := C "affine"
  case inr =>
    obtain ⟨s, i, b, c, b0, c0, rfl, hb, hc0, hq⟩ := parse_affine h (by rw [h1]; rfl) (by rw [h2]; rfl) h3
    obtain ⟨⟨l'⟩, rfl, hsz, hel⟩ := arr_of_jsonEquiv hae
    rcases l' with _ | ⟨s', _ | ⟨i', _ | ⟨b', _ | ⟨c', _ | ⟨y', l'⟩⟩⟩⟩⟩
    all_goals try simp at hsz
    have hs : JsonEquiv s s' := hel 0 (by simp) (by simp)
    have hi : JsonEquiv i i' := hel 1 (by simp) (by simp)
    refine affine_fwd (b0 := b0) (c0 := c0) h1' h2' h3' ((intJ_jsonEquiv (hel 2 (by simp) (by simp))) ▸ hb)
      ((natJ_jsonEquiv (hel 3 (by simp) (by simp))) ▸ hc0) ?_
    rw [← spaceOf_jsonEquiv hs, ← withinOf_jsonEquiv hs hi]
    exact hq
  obtain ⟨e4, e4', h4, h4'⟩ | ⟨a, a', h4, h4', hae⟩ := C "strided"
  case inr =>
    obtain ⟨s, i, b, inner, os, is, c, b0, inner0, os0, is0, c0, rfl, hb, hin, ho, his, hc0, hi0, hq⟩ :=
      parse_strided h h1 h2 h3 h4
    obtain ⟨⟨l'⟩, rfl, hsz, hel⟩ := arr_of_jsonEquiv hae
    rcases l' with _ | ⟨s', _ | ⟨i', _ | ⟨b', _ | ⟨inner', _ | ⟨os', _ | ⟨is', _ | ⟨c', _ | ⟨y', l'⟩⟩⟩⟩⟩⟩⟩⟩
    all_goals try simp at hsz
    have hs : JsonEquiv s s' := hel 0 (by simp) (by simp)
    have hi : JsonEquiv i i' := hel 1 (by simp) (by simp)
    refine strided_fwd (b0 := b0) (inner0 := inner0) (os0 := os0) (is0 := is0) (c0 := c0) h1' h2' h3' h4' ((intJ_jsonEquiv (hel 2 (by simp) (by simp))) ▸ hb)
      ((natJ_jsonEquiv (hel 3 (by simp) (by simp))) ▸ hin) ((intJ_jsonEquiv (hel 4 (by simp) (by simp))) ▸ ho)
      ((intJ_jsonEquiv (hel 5 (by simp) (by simp))) ▸ his) ((natJ_jsonEquiv (hel 6 (by simp) (by simp))) ▸ hc0)
      hi0 ?_
    rw [← spaceOf_jsonEquiv hs, ← withinOf_jsonEquiv hs hi]
    exact hq
  obtain ⟨e5, e5', h5, h5'⟩ | ⟨a, a', h5, h5', hae⟩ := C "explicit"
  case inl => exact absurd h (parse_none h1 h2 h3 h4 h5)
  obtain ⟨items, msg, rs, rfl, hrs, rfl⟩ := parse_explicit h h1 h2 h3 h4 h5
  obtain ⟨items', rfl, hsz, hel⟩ := arr_of_jsonEquiv hae
  refine explicit_fwd h1' h2' h3' h4' h5' fun msg' => ?_
  rcases items with ⟨l⟩
  rcases items' with ⟨l'⟩
  simp only [Array.mapM_eq_mapM_toList] at hrs ⊢
  cases hl : l.mapM (explicitItem leaves msg) with
  | error e => rw [hl] at hrs; cases hrs
  | ok xs =>
    rw [hl] at hrs
    rw [list_mapM_transport (fun r r' x hr h => explicitItem_jsonEquiv h hr) l l' xs hsz
      (fun k h₁ h₂ => hel k h₁ h₂) hl]
    exact hrs

end NciCheck.Properties.Record.SecurityProofs

end
