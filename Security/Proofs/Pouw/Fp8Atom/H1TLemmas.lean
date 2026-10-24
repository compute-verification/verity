import Definitions.Pouw.Fp8Atom.H1TLemmas
import Proofs.Pouw.Fp8Atom.H1TVec

/-! The proof-only part of `Definitions.Pouw.Fp8Atom.H1TLemmas`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.Fp8Atom.H1T

open Pouw.Fp8Atom

/-- Truncation toward zero to a multiple of `g`. -/
def truncTo (g y : ℚ) : ℚ := (if y < 0 then -1 else 1) * (⌊|y| / g⌋ : ℚ) * g

/-- Truncation toward zero to 14 significant bits: `|y|` rounded down to a multiple of `2^(binade |y| − 13)`. -/
def trunc14 (y : ℚ) : ℚ := if y = 0 then 0 else truncTo (2 ^ (binade |y| - 13)) y

/-- A slice's alignment exponent: the largest exponent of a nonzero product (`maxExp`, floor −139). -/
def gridExp (a b : Fin 32 → Code) : ℤ := maxExp (-139) (List.ofFn fun l => product (a l) (b l))

/-- The exact sum after the first truncation: each product truncated toward zero to the grid `2^(E − 13)`. -/
def preSum (a b : Fin 32 → Code) : ℚ := ∑ l, truncTo (2 ^ (gridExp a b - 13)) (val (a l) * val (b l))

/-- **The atom is two truncations**: on finite codes, `hopperStep` from `+0` writes a finite word whose value is
`trunc14 (preSum a b)`. -/
def H1TAtomValue : Prop :=
  ∀ a b : Fin 32 → Code, (∀ l, (a l).isNaN = false) → (∀ l, (b l).isNaN = false) →
    wordQ (hopperStep 0 a b) = some (trunc14 (preSum a b))

/-- **L7, `normalize` is monotone**: truncation toward zero to 14 significant bits keeps order. -/
def Trunc14Mono : Prop := Monotone trunc14

/-- **L3, the cell**: `trunc14` moves a nonzero `y` by less than `2^(binade |y| − 13)`, and a value with the same
truncation as `y` lies that close to `y`. -/
def Trunc14Cell : Prop :=
  ∀ y y' : ℚ, y ≠ 0 → |trunc14 y - y| < 2 ^ (binade |y| - 13) ∧
    (trunc14 y = trunc14 y' → |y - y'| < 2 ^ (binade |y| - 13))

/-- Tag lane `29 + i`'s sign bit flipped. -/
def negFlip (i : Fin 3) (σ : SliceSalt) : SliceSalt := { σ with neg := Function.update σ.neg i (!σ.neg i) }

/-- **L1 to L3 on one slice.** A slice's real codes `xs` and a column's real codes `b`, all finite; the slice's
`F = floorF (sliceMax xs)`, salt fields `σ`, and the column's lanes `colLanes b j τ`:
* (L3) block 1's sum is at most 6,142,976 and block 2's at most 738,304 in magnitude, and both are below `2^21·F`;
* (L1, L2) flipping tag lane `i`'s sign bit keeps both blocks' alignment exponents, and moves block 1's sum by exactly
  `−2·v·t` and block 2's by exactly `+2·v·t`, with `v` the lane's tag value and `t` the column's tag code. -/
def H1TSlice : Prop :=
  ∀ (xs : Fin 29 → Code) (σ : SliceSalt) (b : ℕ → Code) (j τ : ℕ), (∀ l, (xs l).isNaN = false) →
    (∀ r, (b r).isNaN = false) →
    (|preSum (formSlice xs σ).1 (colLanes b j τ)| ≤ 6142976 ∧
      |preSum (formSlice xs σ).1 (colLanes b j τ)| < 2 ^ 21 * floorF (sliceMax xs)) ∧
    (|preSum (formSlice xs σ).2 (colLanes b j τ)| ≤ 738304 ∧
      |preSum (formSlice xs σ).2 (colLanes b j τ)| < 2 ^ 21 * floorF (sliceMax xs)) ∧
    ∀ i : Fin 3,
      gridExp (formSlice xs (negFlip i σ)).1 (colLanes b j τ) = gridExp (formSlice xs σ).1 (colLanes b j τ) ∧
      gridExp (formSlice xs (negFlip i σ)).2 (colLanes b j τ) = gridExp (formSlice xs σ).2 (colLanes b j τ) ∧
      preSum (formSlice xs (negFlip i σ)).1 (colLanes b j τ) = preSum (formSlice xs σ).1 (colLanes b j τ) -
        2 * tagVal (floorF (sliceMax xs)) (σ.neg i) (σ.man i) * val (tuple j i) ∧
      preSum (formSlice xs (negFlip i σ)).2 (colLanes b j τ) = preSum (formSlice xs σ).2 (colLanes b j τ) +
        2 * tagVal (floorF (sliceMax xs)) (σ.neg i) (σ.man i) * val (tuple j i)

/-- **L7 for the chain**: running words are monotone in the fresh-step words up to their position. -/
def H1TChainMono : Prop :=
  ∀ (T : ℕ) (x b : ℕ → Code) (j : ℕ) (σ σ' : ℕ → SliceSalt) (p : ℕ),
    (∀ t ≤ p, freshVal T x b j σ t ≤ freshVal T x b j σ' t) → runVal T x b j σ p ≤ runVal T x b j σ' p

/-- **L4, the range**: on a legal unit every fresh-step word is finite and at most 6,142,976 (block 1) or 738,304
(block 2) in magnitude, and every running word is below `2^33`. -/
def H1TRange : Prop :=
  ∀ U : Unit, U.Legal → ∀ (i : Fin U.m) (j : Fin U.n) (g : U.Salt) (p : ℕ), p < 2 * U.T →
    (wordQ (freshWord U.T (U.row i) (U.col j) j.val (U.rowSalt g i) p)).isSome = true ∧
    |freshVal U.T (U.row i) (U.col j) j.val (U.rowSalt g i) p| ≤ (if p < U.T then 6142976 else 738304) ∧
    |runVal U.T (U.row i) (U.col j) j.val (U.rowSalt g i) p| < 2 ^ 33

/-- **The forming table covers every slice** (X-M2-4 (a)): a slice of finite codes has `F = fOf fi` for one of the
indices `fi ≤ 4` of `H1TFormTable`'s keys, and each of its codes is below `2^(fi + 5)`. -/
def H1TFormCover : Prop :=
  ∀ xs : Fin 29 → Code, (∀ l, (xs l).isNaN = false) →
    ∃ fi : ℕ, fi ≤ 4 ∧ floorF (sliceMax xs) = fOf fi ∧ ∀ l, |val (xs l)| < 2 ^ (fi + 5)

/-- **The one-flip clauses** (M4a, M4b, M4c): on every legal unit and row `i`, `HalfApartH1T`'s two clauses for the
pairs that are not `Hard` and for the words that are not `Stuck`. Two distinct credited words of row `i` that are not
`Hard` agree on at most half of row `i`'s salts, and a credited word that is not `Stuck` equals a given value on at
most half of them. -/
def H1TOneFlip : Prop :=
  ∀ U : Unit, U.Legal → ∀ i : Fin U.m,
    (∀ a a' : Fin U.n × Pos U.T, a ≠ a' → ¬ Hard a.1.val a'.1.val a.2 a'.2 →
      2 * (Finset.univ.filter fun s : Fin U.T → RawSlice =>
        U.word (i, a) (fun _ => s) = U.word (i, a') (fun _ => s)).card ≤ Fintype.card (Fin U.T → RawSlice)) ∧
    ∀ (a : Fin U.n × Pos U.T), ¬ Stuck a.1.val a.2 → ∀ q : ℚ,
      2 * (Finset.univ.filter fun s : Fin U.T → RawSlice => U.word (i, a) (fun _ => s) = q).card ≤
        Fintype.card (Fin U.T → RawSlice)

end Pouw.Fp8Atom.H1T
