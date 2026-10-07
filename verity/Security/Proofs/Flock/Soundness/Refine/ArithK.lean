import Proofs.Flock.Soundness.Refine.Opening

/-!
# Element seams over `K = GF(2^256)` (PROTOCOL.md §2.4, §13)

The model's Ligerito computes in `K` through `execArith.embed` and `execArith.u`; the executable computes on `F256`
words. `gOf` (`GF256.of`) maps a word to its field element and is a ring homomorphism (`gOf_add`, `gOf_mul`). On top of
level 3's `ofLimbs`:

* `F256.ofF` is `embed`, and `a.smul s` is `a · embed s` (`gOf_smul`);
* the executable's round polynomials (`Poly`) are the model's `Quad` (`quadOf`: `ofMsg`, `eval`, `addScaled`);
* both equality tables are the model's `eqList`: `eqTable` over `F` (`eqTable_eqList`), `eqTableK` over `K`
  (`eqTableK_eqList`).
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-- A word of `K` as a model field element. -/
abbrev gOf (a : F256) : 𝕂 := GF256.of a

theorem gOf_add (a b : F256) : gOf (a + b) = gOf a + gOf b := (GF256.add_eq a b).symm
theorem gOf_mul (a b : F256) : gOf (a * b) = gOf a * gOf b := (GF256.mul_eq a b).symm

theorem gOf_limbs (a : F256) : gOf a = embed (fOf a.c0) + embed (fOf a.c1) * u := ofLimbs a.c0 a.c1

theorem gOf_ofF (a : F128) : gOf (F256.ofF a) = embed (fOf a) := rfl
theorem gOf_u : gOf F256.u = u := rfl

theorem gOf_zero : gOf 0 = 0 := by
  rw [gOf_limbs]; show embed (fOf 0) + embed (fOf 0) * u = 0; rw [fOf_zero, map_zero, zero_mul, add_zero]

theorem gOf_one : gOf 1 = 1 := by
  rw [gOf_limbs]; show embed (fOf 1) + embed (fOf 0) * u = 1; rw [fOf_one, fOf_zero, map_one, map_zero, zero_mul, add_zero]

/-- `K · F` coordinate-wise is multiplication by the embedded `F` element. -/
theorem gOf_smul (a : F256) (s : F128) : gOf (a.smul s) = gOf a * embed (fOf s) := by
  rw [gOf_limbs (a.smul s), gOf_limbs a]
  show embed (fOf (a.c0 * s)) + embed (fOf (a.c1 * s)) * u = _
  rw [fOf_mul, fOf_mul, map_mul, map_mul]
  ring

/-! ## Round polynomials -/

/-- An executable round polynomial as the model's. -/
def quadOf (p : Poly) : Model.Quad 𝕂 := (gOf p.a0, gOf p.a1, gOf p.a2)

theorem quadOf_ofMsg (msg : F256 × F256) (s : F256) :
    quadOf (Poly.ofMsg msg s) = Model.fromMsg (gOf msg.1, gOf msg.2) (gOf s) := by
  simp only [quadOf, Poly.ofMsg, Model.fromMsg, gOf_add]

theorem gOf_eval (p : Poly) (r : F256) : gOf (p.eval r) = (quadOf p).eval (gOf r) := by
  simp only [Poly.eval, Model.Quad.eval, quadOf, gOf_add, gOf_mul]
  ring

theorem quadOf_addScaled (p q : Poly) (b : F128) :
    quadOf (p.addScaled q b) = (quadOf p).addScaled (quadOf q) (embed (fOf b)) := by
  simp only [Poly.addScaled, Model.Quad.addScaled, quadOf, gOf_add, gOf_smul]
  refine Prod.ext (by ring) (Prod.ext (by simp only; ring) (by simp only; ring))

/-! ## Equality tables -/

/-- `eqList` is the product over the coordinates. -/
theorem eqList_prod {R : Type} [CommRing R] (j : ℕ) : ∀ (l : List R) (k : ℕ),
    ((l.zipIdx k).map fun (a, i) => if j.testBit i then a else 1 + a).prod =
      ∏ i ∈ Finset.range l.length, if j.testBit (k + i) then l.getD i 0 else 1 + l.getD i 0
  | [], _ => by simp
  | a :: l, k => by
      rw [List.zipIdx_cons, List.map_cons, List.prod_cons, eqList_prod j l (k + 1), List.length_cons,
        Finset.prod_range_succ', mul_comm]
      simp only [Nat.add_zero, List.getD_cons_zero, List.getD_cons_succ]
      congr 1
      refine Finset.prod_congr rfl fun i _ => ?_
      rw [show k + 1 + i = k + (i + 1) by omega]

theorem eqList_eq {R : Type} [CommRing R] (l : List R) (j : ℕ) :
    Model.eqList l j = ∏ i ∈ Finset.range l.length, if j.testBit i then l.getD i 0 else 1 + l.getD i 0 := by
  unfold Model.eqList
  rw [eqList_prod j l 0]
  simp

/-- The executable's `eqTable` over `F` is the model's `eqList` of its coordinates. -/
theorem eqTable_eqList (r : Array F128) (j : ℕ) (hj : j < 2 ^ r.size) :
    fOf (eqTable r)[j]! = Model.eqList (r.toList.map fOf) j := by
  show FlockLevel3.GF128.of (eqTable r)[j]! = _
  rw [eqTable_eqAt r j hj, eqList_eq]
  unfold FlockLevel3.eqAt
  rw [Fin.prod_univ_eq_prod_range (fun i => if j.testBit i then GF128.of r[i]! else 1 + GF128.of r[i]!) r.size]
  simp only [List.length_map, Array.length_toList]
  refine Finset.prod_congr rfl fun i hi => ?_
  rw [List.getD_eq_getElem _ _ (by simpa using hi), List.getElem_map, Array.getElem_toList,
    getElem!_pos r i (by simpa using hi)]

/-- One coordinate of `eqTableK`. -/
def eqStepK (t : Array F256) (rj : F256) : Array F256 := t.map (· * (rj + 1)) ++ t.map (· * rj)

theorem eqTableK_foldl (r : Array F256) : eqTableK r = r.foldl eqStepK #[1] := by
  unfold eqTableK
  simp only [Id.run, Array.forIn_pure_yield_eq_foldl]
  rfl

theorem eqStepK_size (t : Array F256) (x : F256) : (eqStepK t x).size = 2 * t.size := by
  simp [eqStepK]; omega

theorem foldl_eqStepK_size (l : List F256) : ∀ t : Array F256, (l.foldl eqStepK t).size = t.size * 2 ^ l.length := by
  induction l with
  | nil => intro t; simp
  | cons x l ih => intro t; rw [List.foldl_cons, ih, eqStepK_size, List.length_cons, pow_succ]; ring

theorem eqStepK_get (t : Array F256) (x : F256) (i : ℕ) (hi : i < (eqStepK t x).size) (ht : 0 < t.size) :
    gOf ((eqStepK t x)[i]) =
      gOf (t[i % t.size]'(Nat.mod_lt _ ht)) * (if (i / t.size).testBit 0 then gOf x else 1 + gOf x) := by
  have hi' : i < 2 * t.size := by rw [eqStepK_size] at hi; exact hi
  have h00 : Nat.testBit 0 0 = false := by decide
  have h10 : Nat.testBit 1 0 = true := by decide
  simp only [eqStepK, Array.getElem_append, Array.size_map, Array.getElem_map]
  by_cases h : i < t.size
  · have hm : i % t.size = i := Nat.mod_eq_of_lt h
    have hd : i / t.size = 0 := Nat.div_eq_of_lt h
    simp only [h, dite_true, hd, h00, Bool.false_eq_true, ite_false]
    rw [gOf_mul, gOf_add, gOf_one]
    simp only [hm]
    ring
  · have hm : i % t.size = i - t.size := by
      rw [Nat.mod_eq_sub_mod (by omega), Nat.mod_eq_of_lt (by omega)]
    have hd : i / t.size = 1 := Nat.div_eq_of_lt_le (by omega) (by omega)
    simp only [h, dite_false, hd, h10, ite_true]
    rw [gOf_mul]
    simp only [hm]

theorem foldl_eqStepK_get (l : List F256) : ∀ (t : Array F256) (ht : 0 < t.size) (i : ℕ)
    (hi : i < (l.foldl eqStepK t).size),
    gOf (l.foldl eqStepK t)[i] = gOf (t[i % t.size]'(Nat.mod_lt _ ht)) *
      ∏ j ∈ Finset.range l.length, (if (i / t.size).testBit j then gOf (l.getD j 0) else 1 + gOf (l.getD j 0)) := by
  induction l with
  | nil =>
      intro t ht i hi
      simp only [List.foldl_nil] at hi ⊢
      simp [Nat.mod_eq_of_lt hi]
  | cons x l ih =>
      intro t ht i hi
      have ht2 : 0 < (eqStepK t x).size := by rw [eqStepK_size]; omega
      simp only [List.foldl_cons] at hi ⊢
      rw [ih (eqStepK t x) ht2 i hi]
      have him : i % (eqStepK t x).size < (eqStepK t x).size := Nat.mod_lt _ ht2
      rw [eqStepK_get t x _ him ht, List.length_cons, Finset.prod_range_succ']
      have e1 : i % (eqStepK t x).size % t.size = i % t.size := by rw [eqStepK_size, Nat.mod_mul_left_mod]
      have e2 : (i % (eqStepK t x).size / t.size).testBit 0 = (i / t.size).testBit 0 := by
        rw [Nat.testBit_zero, Nat.testBit_zero, eqStepK_size, mul_comm, Nat.mod_mul_right_div_self, Nat.mod_mod]
      have e3 : ∀ j : ℕ, (i / (eqStepK t x).size).testBit j = (i / t.size).testBit (j + 1) := by
        intro j; rw [eqStepK_size, mul_comm, ← Nat.div_div_eq_div_mul, Nat.testBit_div_two]
      simp only [e1, e2, e3, List.getD_cons_zero, List.getD_cons_succ]
      ring

/-- **The executable's `eqTableK` over `K` is the model's `eqList` of its coordinates.** -/
theorem eqTableK_eqList (r : Array F256) (j : ℕ) (hj : j < 2 ^ r.size) :
    gOf (eqTableK r)[j]! = Model.eqList (r.toList.map gOf) j := by
  have hs : (eqTableK r).size = 2 ^ r.size := by
    rw [eqTableK_foldl, ← Array.foldl_toList, foldl_eqStepK_size]; simp
  rw [getElem!_pos _ j (by rw [hs]; exact hj), eqList_eq]
  have h := foldl_eqStepK_get r.toList #[1] (by simp) j
    (by rw [Array.foldl_toList, ← eqTableK_foldl, hs]; exact hj)
  simp only [Array.foldl_toList] at h
  simp only [eqTableK_foldl]
  rw [h]
  simp only [List.size_toArray, List.length_cons, List.length_nil, Nat.zero_add, Nat.mod_one, Nat.div_one,
    List.length_map, Array.length_toList]
  rw [show gOf (#[(1 : F256)][0]'(by simp)) = 1 from gOf_one, one_mul]
  refine Finset.prod_congr rfl fun i hi => ?_
  rw [List.getD_eq_getElem _ _ (by simpa using hi), List.getD_eq_getElem _ _ (by simpa using hi), List.getElem_map]

end FlockSoundness.Refine
