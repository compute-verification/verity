import Definitions.Core.Guarantees.Formalism
import Definitions.Core.Guarantees.Expand
import Proofs.Core.Expand
import Mathlib.Data.ZMod.Basic
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.Ring

/-!
# Proofs about the AND count

* `Circuit.andSize_le_size`: AND gates are operation gates, so `andSize S ≤ size S`, and credit is at most the lowered
  circuit's size (`CreditSizeLe`);
* `Circuit.not_affine_of_and`: a gate that computes `p ∧ q` on the inputs that are `p` at `x`, `q` at `y` and `0`
  elsewhere is not affine: an affine function's values on those four inputs sum to `0`, AND's to `1`;
* `ToCircuitAnd`: C-Flock's XOR, NOT and input gates are affine, and its `and a b` (`a ≠ b`) and its rows whose XORs
  differ are not.
-/

namespace Verity

open Finset

theorem Circuit.andSize_le_computed (C : Circuit) (S : Finset (Fin C.N)) :
    C.andSize S ≤ C.size (S ∩ C.computed) := by
  classical
  unfold Circuit.andSize Circuit.size
  refine card_le_card fun g hg => ?_
  rw [mem_filter] at hg
  exact mem_inter.2 ⟨hg.1, by simp [Circuit.computed, hg.2.1]⟩

theorem Circuit.andSize_le_size (C : Circuit) (S : Finset (Fin C.N)) : C.andSize S ≤ C.size S :=
  (C.andSize_le_computed S).trans (card_le_card inter_subset_left)

/-- The input that is `p` at `x`, `q` at `y` and `false` elsewhere. -/
def Circuit.andProbe {N : ℕ} (x y : Fin N) (p q : Bool) (z : Fin N) : Bool :=
  decide ((z = x ∧ p = true) ∨ (z = y ∧ q = true))

theorem Circuit.andProbe_x {N : ℕ} {x y : Fin N} (hxy : x ≠ y) (p q : Bool) : andProbe x y p q x = p := by
  cases p <;> cases q <;> simp [andProbe, hxy]

theorem Circuit.andProbe_y {N : ℕ} {x y : Fin N} (hxy : x ≠ y) (p q : Bool) : andProbe x y p q y = q := by
  cases p <;> cases q <;> simp [andProbe, Ne.symm hxy]

theorem Circuit.sum_andProbe {N : ℕ} (S : Finset (Fin N)) {x y : Fin N} (hxy : x ≠ y) (p q : Bool) :
    ∑ z ∈ S, (if andProbe x y p q z then (1 : ZMod 2) else 0) =
      (if x ∈ S ∧ p = true then 1 else 0) + (if y ∈ S ∧ q = true then 1 else 0) := by
  have : ∀ z, (if andProbe x y p q z then (1 : ZMod 2) else 0) =
      (if z = x ∧ p = true then 1 else 0) + (if z = y ∧ q = true then 1 else 0) := by
    intro z
    by_cases hx : z = x
    · subst hx
      cases p <;> cases q <;> simp [andProbe, hxy]
    · cases p <;> cases q <;> simp [andProbe, hx]
  simp only [this, sum_add_distrib]
  cases p <;> cases q <;> simp [sum_ite_eq']

/-- **A gate that is an AND on two inputs is not affine.** If gate `g` computes `p ∧ q` on inputs `v p q` whose bits are
`p` at `x`, `q` at `y` and `0` elsewhere, it is not affine. -/
theorem Circuit.not_affine_of_and (C : Circuit) (g : Fin C.N) {x y : Fin C.N} (hxy : x ≠ y)
    (v : Bool → Bool → (a : Fin C.N) → BitVec (C.w a))
    (hv : ∀ p q a, bitZ (v p q a) = if andProbe x y p q a then 1 else 0)
    (h : ∀ p q, bitZ (C.op g (v p q)) = if (p && q) then 1 else 0) : ¬ C.Affine g := by
  rintro ⟨S, c, -, hf⟩
  have e : ∀ p q, (if (p && q) then (1 : ZMod 2) else 0) =
      c + ((if x ∈ S ∧ p = true then 1 else 0) + (if y ∈ S ∧ q = true then 1 else 0)) := by
    intro p q
    rw [← h p q, hf, ← sum_andProbe S hxy p q]
    simp only [hv]
  have e00 := e false false
  have e10 := e true false
  have e01 := e false true
  have e11 := e true true
  simp only [Bool.and_self, Bool.and_false, Bool.false_and, Bool.false_eq_true, and_false, and_true, ite_false,
    ite_true, add_zero] at e00 e10 e01 e11
  have : (1 : ZMod 2) = 0 := by linear_combination e11 - e10 - e01 + e00
  exact absurd this (by decide)

namespace SecurityProofs

theorem AndSizeLe : Guarantees.AndSizeLe := fun C S => ⟨C.andSize_le_computed S, C.andSize_le_size S⟩

theorem CreditSizeLe : Guarantees.CreditSizeLe := fun lower d =>
  ((lower d).andSize_le_computed univ).trans (by rw [univ_inter])

theorem ExpandAndSize : Guarantees.ExpandAndSize := by
  intro d B P
  obtain ⟨h, hu⟩ := ExpandSize d B P
  exact ⟨h ▸ d.D.expand.andSize_le_size _, fun u => (hu u).1 ▸ d.D.expand.andSize_le_size _⟩

end SecurityProofs

end Verity

namespace FlockSoundness.Audit.Circuit

open Verity Verity.Circuit Finset

/-- A gate of `toCircuit`'s bit is its operation on the bits it reads. -/
theorem bitZ_toCircuit_op (B : Circuit) (g : Fin B.N) (v : (a : Fin B.toCircuit.N) → BitVec (B.toCircuit.w a)) :
    Verity.Circuit.bitZ (B.toCircuit.op g v) = if (B.op g).apply (fun a => (v a).getLsbD 0) then 1 else 0 := by
  show (if (BitVec.ofBool ((B.op g).apply fun a => (v a).getLsbD 0)).getLsbD 0 then (1 : ZMod 2) else 0) = _
  rw [BitVec.getLsbD_ofBool]
  simp

theorem getLsbD_toCircuit (B : Circuit) (z : Fin B.toCircuit.N) (b : Bool) :
    (BitVec.ofBool b : BitVec (B.toCircuit.w z)).getLsbD 0 = b := by
  cases b <;> rfl

theorem bitZ_toCircuit (B : Circuit) (z : Fin B.toCircuit.N) (b : Bool) :
    Verity.Circuit.bitZ (BitVec.ofBool b : BitVec (B.toCircuit.w z)) = if b then 1 else 0 := by
  cases b <;> rfl

/-- **An `and`-shaped gate of `toCircuit` is not affine**: if its operation computes `p ∧ q` on `andProbe x y p q`. -/
theorem not_affine (B : Circuit) (g : Fin B.N) {x y : Fin B.toCircuit.N} (hxy : x ≠ y)
    (h : ∀ p q, (B.op g).apply (andProbe x y p q) = (p && q)) : ¬ B.toCircuit.Affine g := by
  refine B.toCircuit.not_affine_of_and g hxy (fun p q a => BitVec.ofBool (andProbe x y p q a))
    (fun p q a => bitZ_toCircuit B a _) fun p q => (bitZ_toCircuit_op B g _).trans ?_
  show (if (B.op g).apply (fun a => (BitVec.ofBool (andProbe x y p q a) : BitVec (B.toCircuit.w a)).getLsbD 0) then
    (1 : ZMod 2) else 0) = _
  refine (congrArg (fun f => if (B.op g).apply f then (1 : ZMod 2) else 0)
    (funext fun a => getLsbD_toCircuit B a (andProbe x y p q a))).trans ?_
  exact congrArg (fun b => if b then (1 : ZMod 2) else 0) (h p q)

private theorem bit_xor (p q : Bool) :
    (if Bool.xor p q then (1 : ZMod 2) else 0) = (if p then 1 else 0) + (if q then 1 else 0) := by
  cases p <;> cases q <;> decide

private theorem bit_not (p : Bool) : (if !p then (1 : ZMod 2) else 0) = 1 + (if p then 1 else 0) := by
  cases p <;> decide

private theorem xorSum_cons {ι : Type} (z : ι) (l : List ι) (v : ι → Bool) :
    xorSum (z :: l) v = Bool.xor (v z) (xorSum l v) := rfl

private theorem bit_xorSum {N : ℕ} (l : List (Fin N)) {x y : Fin N} (hxy : x ≠ y) (p q : Bool) :
    (if xorSum l (andProbe x y p q) then (1 : ZMod 2) else 0) =
      (l.count x : ZMod 2) * (if p then 1 else 0) + (l.count y : ZMod 2) * (if q then 1 else 0) := by
  induction l with
  | nil => simp [xorSum]
  | cons z l ih =>
    rw [xorSum_cons, bit_xor, ih, List.count_cons, List.count_cons]
    by_cases hx : z = x
    · have hy : z ≠ y := hx ▸ hxy
      cases p <;> cases q <;> simp [andProbe, hx, hxy] <;> ring
    · by_cases hy : z = y
      · cases p <;> cases q <;> simp [andProbe, hy, Ne.symm hxy] <;> ring
      · cases p <;> cases q <;> simp [andProbe, hx, hy]

private theorem xorSum_andProbe {N : ℕ} (l : List (Fin N)) {x y : Fin N} (hxy : x ≠ y) (p q : Bool) (r s : Bool)
    (hr : (l.count x : ZMod 2) = if r then 1 else 0) (hs : (l.count y : ZMod 2) = if s then 1 else 0) :
    xorSum l (andProbe x y p q) = Bool.xor (r && p) (s && q) := by
  have h := bit_xorSum l hxy p q
  rw [hr, hs] at h
  have key : ∀ b c : Bool, ((if b then (1 : ZMod 2) else 0) = if c then 1 else 0) → b = c := by decide
  refine key _ _ (h.trans ?_)
  cases r <;> cases s <;> cases p <;> cases q <;> decide

private theorem count_odd {N : ℕ} (l : List (Fin N)) (x : Fin N) (h : Odd (l.count x)) :
    ((l.count x : ℕ) : ZMod 2) = if true then 1 else 0 := by
  simpa using (ZMod.natCast_eq_one_iff_odd).2 h

private theorem count_even {N : ℕ} (l : List (Fin N)) (x : Fin N) (h : Even (l.count x)) :
    ((l.count x : ℕ) : ZMod 2) = if false then 1 else 0 := by
  simpa using (ZMod.natCast_eq_zero_iff_even).2 h

/-- An `and` of two different gates computes `p ∧ q` on `andProbe`. -/
theorem apply_and_andProbe (B : Circuit) {a b : Fin B.N} (hab : a ≠ b) (p q : Bool) :
    (Op.and a b).apply (andProbe a b p q) = (p && q) := by
  show (andProbe a b p q a && andProbe a b p q b) = (p && q)
  rw [andProbe_x hab, andProbe_y hab]

/-- A row whose XORs differ at `x` and `y` computes `p ∧ q` on `andProbe`. -/
theorem apply_row_andProbe {N : ℕ} {a b : List (Fin N)} {x y : Fin N} (hxy : x ≠ y) (hxa : Odd (a.count x))
    (hxb : Even (b.count x)) (hya : Even (a.count y)) (hyb : Odd (b.count y)) (p q : Bool) :
    (Op.row a b).apply (andProbe x y p q) = (p && q) := by
  show (xorSum a (andProbe x y p q) && xorSum b (andProbe x y p q)) = (p && q)
  rw [xorSum_andProbe a hxy p q true false (count_odd a x hxa) (count_even a y hya),
    xorSum_andProbe b hxy p q false true (count_even b x hxb) (count_odd b y hyb)]
  cases p <;> cases q <;> rfl

end FlockSoundness.Audit.Circuit

namespace Verity.SecurityProofs

open FlockSoundness.Audit FlockSoundness.Audit.Circuit Finset

theorem ToCircuitAnd : Guarantees.ToCircuitAnd := by
  intro B g
  have hin : ∀ o, B.op g = o → o.isInput = false → B.toCircuit.isInput g = false := fun o h ho => by
    show (B.op g).isInput = false
    rw [h]; exact ho
  refine ⟨fun ⟨hi, hna⟩ => ?_, fun a b h hab => ⟨hin _ h rfl, ?_⟩,
    fun a b x y h hxa hxb hya hyb => ⟨hin _ h rfl, ?_⟩⟩
  · cases ho : B.op g with
    | input =>
      have : (B.op g).isInput = false := hi
      rw [ho] at this
      exact absurd this (by simp [Op.isInput])
    | and a b => exact Or.inl ⟨a, b, rfl⟩
    | row a b => exact Or.inr ⟨a, b, rfl⟩
    | xor a b =>
      exfalso; apply hna
      by_cases hab : a = b
      · subst hab
        refine ⟨∅, 0, empty_subset _, fun v => ?_⟩
        rw [bitZ_toCircuit_op, ho]
        simp [Op.apply]
      · refine ⟨{a, b}, 0, ?_, fun v => ?_⟩
        · show ({a, b} : Finset (Fin B.N)) ⊆ (B.op g).args.toFinset
          rw [ho]
          intro z hz
          simp only [mem_insert, mem_singleton] at hz
          rcases hz with rfl | rfl <;> simp [Op.args]
        · rw [bitZ_toCircuit_op, ho, zero_add]
          exact (bit_xor _ _).trans (sum_pair (f := fun z => Verity.Circuit.bitZ (v z)) hab).symm
    | not a =>
      exfalso; apply hna
      refine ⟨{a}, 1, ?_, fun v => ?_⟩
      · show ({a} : Finset (Fin B.N)) ⊆ (B.op g).args.toFinset
        rw [ho]
        simp [Op.args]
      · rw [bitZ_toCircuit_op, ho]
        exact (bit_not _).trans (congrArg (1 + ·) (sum_singleton (fun z => Verity.Circuit.bitZ (v z)) a).symm)
  · exact not_affine B g hab fun p q => by rw [h]; exact apply_and_andProbe B hab p q
  · have hxy : x ≠ y := by
      rintro rfl
      exact (Nat.not_even_iff_odd.2 hxa) hya
    exact not_affine B g hxy fun p q => by rw [h]; exact apply_row_andProbe hxy hxa hxb hya hyb p q

end Verity.SecurityProofs
