import Flock.RecAlgebra
import Proofs.Flock.VBridge.OfVStar

/-!
# E's `GfResiduals` read as V*'s

E's stages conclude about `Algebra.residuals S w v`, over `Array`s (`Algebra.GfS`, `termValue`, `resValue`). V*'s unit
computes `resVal (gfS S) (termVal w v)` over V*'s S (`VStar.GfS`, lists) and the ports' elements as functions
(`OfVStar.laid_innerUnit`). `GfS.toV` reads E's S as V*'s, term for term.

* `bit_eq_bitOf`: `F128.bit`, which `termValue` reads a `wt` term's bit with, is `bitOf` below bit 128. Above it
  `F128.bit` wraps its shift, so the terms here have their `wt` bits below 128 (`BitsOk`, which `VStar.checkS` checks).
* `termValue_toV`, `opValue_toV`, `resValue_toV`: the values agree at `el w`, `el v` (an element past the port's end
  is 0, as `w[i]!` reads it).
* `residuals_toV`: when `Algebra.residuals S w v = .ok rs`, the ports have S's sizes and `rs[r]` is V*'s residual `r`.
-/

namespace Flock.RecAlgebra

open FlockVBridge

open Flock

/-- E's term read as V*'s. -/
def Term.toV : Term → VStar.Term
  | .w i => .w i
  | .v j => .v j
  | .wt i u => .wt i u

def Res.toV (r : Res) : VStar.Res := ⟨r.mul.toList, r.add.toList.map Term.toV⟩

/-- E's S read as V*'s. -/
def GfS.toV (S : GfS) : VStar.GfS :=
  ⟨S.nw, S.nv, S.ops.toList.map (·.toList.map Term.toV), S.res.toList.map Res.toV⟩

/-- A `wt` term's bit is below 128 (`VStar.checkTerm`). -/
def Term.BitOk : Term → Prop
  | .wt _ u => u < 128
  | _ => True

/-- Every term of S has its `wt` bit below 128. -/
def GfS.BitsOk (S : GfS) : Prop := (∀ o ∈ S.ops, ∀ t ∈ o, t.BitOk) ∧ ∀ r ∈ S.res, ∀ t ∈ r.add, t.BitOk

/-- A port's element `i`, 0 past its end. -/
def el (w : Array F128) (i : ℕ) : F128 := w.getD i 0

theorem el_ge {w : Array F128} {i : ℕ} (h : w.size ≤ i) : el w i = 0 := by
  simp [el, Array.getD, Nat.not_lt.2 h]

theorem getElem!_el (w : Array F128) (i : ℕ) : w[i]! = el w i := by
  rw [Array.getElem!_eq_getD]; rfl

theorem add_zero_f128 (x : F128) : x + 0 = x := by
  cases x
  show F128.add _ ⟨0, 0⟩ = _
  simp [F128.add]

theorem shift_bit (x : UInt64) {i : ℕ} (hi : i < 64) : ((x >>> i.toUInt64) &&& 1 == 1) = x.toNat.testBit i := by
  have h : ((x >>> i.toUInt64) &&& 1).toNat = x.toNat >>> i % 2 := by
    rw [UInt64.toNat_and, UInt64.toNat_shiftRight, UInt64.toNat_one, Nat.and_one_is_mod]
    simp [Nat.mod_eq_of_lt hi, Nat.toUInt64]
  rw [Bool.beq_eq_decide_eq]
  simp only [← UInt64.toNat_inj, h, UInt64.toNat_one, Nat.testBit]
  rcases Nat.mod_two_eq_zero_or_one (x.toNat >>> i) with h2 | h2 <;> simp [h2]

/-- **`F128.bit` is `bitOf` below bit 128.** -/
theorem bit_eq_bitOf (a : F128) {u : ℕ} (hu : u < 128) : a.bit u = bitOf a u := by
  unfold F128.bit bitOf FlockLevel3.w128
  rw [Nat.testBit_two_pow_mul_add _ a.lo.toNat_lt]
  by_cases h : u < 64
  · simp only [h, ite_true]; exact shift_bit a.lo h
  · simp only [h, ite_false]; exact shift_bit a.hi (by omega)

theorem foldl_mem_congr {α β : Type} {f g : α → β → α} :
    ∀ {l : List β} (a : α), (∀ x ∈ l, ∀ a, f a x = g a x) → l.foldl f a = l.foldl g a
  | [], _, _ => rfl
  | x :: l, a, h => by
    simp only [List.foldl_cons]
    rw [h x (by simp) a]
    exact foldl_mem_congr _ fun y hy => h y (by simp [hy])

theorem termValue_toV (w v : Array F128) (t : Term) (ht : t.BitOk) :
    termValue w v t = termVal (el w) (el v) (OfVStar.term t.toV) := by
  cases t with
  | w i => simp only [termValue, Term.toV, OfVStar.term, termVal, getElem!_el]
  | v j => simp only [termValue, Term.toV, OfVStar.term, termVal, getElem!_el]
  | wt i u =>
    simp only [Term.BitOk] at ht
    simp only [termValue, Term.toV, OfVStar.term, termVal, ofBits]
    rw [← Array.foldl_toList, Array.toList_range, List.foldl_map]
    congr 1
    funext acc j
    have hb : wbit (el w) (128 * (i + j) + u) = w[i + j]!.bit u := by
      rw [wbit_elem _ _ ht, bit_eq_bitOf _ ht, getElem!_el]
    simp only [hb]
    split <;> simp [add_zero_f128]

theorem opValue_toV (S : GfS) (w v : Array F128) (hS : ∀ o ∈ S.ops, ∀ t ∈ o, t.BitOk) (k : ℕ) :
    opValue S w v k = opVal (OfVStar.gfS S.toV) (termVal (el w) (el v)) k := by
  unfold opValue opVal OfVStar.gfS GfS.toV
  simp only [List.map_map]
  by_cases hk : k < S.ops.size
  · rw [getElem!_pos S.ops k hk, List.getD_eq_getElem?_getD, List.getElem?_map, Array.getElem?_toList,
      Array.getElem?_eq_getElem hk, Option.map_some, Option.getD_some, ← Array.foldl_toList]
    simp only [Function.comp_apply, List.map_map, List.foldl_map]
    exact foldl_mem_congr _ fun t ht a => by
      rw [termValue_toV w v t (hS _ (Array.getElem_mem hk) t (Array.mem_toList_iff.1 ht))]
  · rw [getElem!_neg S.ops k hk, List.getD_eq_getElem?_getD, List.getElem?_map, Array.getElem?_toList,
      Array.getElem?_eq_none (Nat.not_lt.1 hk)]
    rfl

theorem resValue_toV (S : GfS) (w v : Array F128) (hS : ∀ o ∈ S.ops, ∀ t ∈ o, t.BitOk) (r : Res)
    (hr : ∀ t ∈ r.add, t.BitOk) :
    resValue S w v r = resVal (OfVStar.gfS S.toV) (termVal (el w) (el v)) (OfVStar.res r.toV) := by
  unfold resValue resVal OfVStar.res Res.toV
  rw [← Array.foldl_toList, ← Array.foldl_toList]
  simp only [List.map_map, List.foldl_map]
  have hadd : r.add.toList.foldl (fun x t => x + termValue w v t) 0 =
      r.add.toList.foldl (fun x t => x + (termVal (el w) (el v) ∘ OfVStar.term ∘ Term.toV) t) 0 :=
    foldl_mem_congr _ fun t ht a => by
      rw [termValue_toV w v t (hr t (Array.mem_toList_iff.1 ht))]; rfl
  rw [hadd]
  congr 1
  funext x ab
  rw [opValue_toV S w v hS, opValue_toV S w v hS]

/-- **E's residuals are V*'s**: when `Algebra.residuals S w v = .ok rs`, the ports have S's sizes and each `rs[r]` is
V*'s residual `r` (`resVal`) at `el w`, `el v`. -/
theorem residuals_toV {S : GfS} {w v rs : Array F128} (hS : S.BitsOk) (h : residuals S w v = .ok rs) :
    w.size = S.nw ∧ v.size = S.nv ∧ ∃ hn : rs.size = S.toV.res.length, ∀ r (hr : r < rs.size),
      rs[r] = resVal (OfVStar.gfS S.toV) (termVal (el w) (el v)) (OfVStar.res (S.toV.res[r]'(hn ▸ hr))) := by
  unfold residuals at h
  split at h
  · cases h
  · rename_i hsz
    simp only [not_or, Decidable.not_not] at hsz
    cases h
    refine ⟨hsz.1, hsz.2, by simp [GfS.toV], fun r hr => ?_⟩
    have hr' : r < S.res.size := by simpa using hr
    simp only [Array.getElem_map, GfS.toV, List.getElem_map, Array.getElem_toList]
    exact resValue_toV S w v hS.1 _ (hS.2 _ (Array.getElem_mem hr'))

end Flock.RecAlgebra
