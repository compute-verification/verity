import Definitions.Pouw.Fp8Atom.H1TLemmas

/-!
# H-1T: the cross-column late pairs in the exact-512 regime (statements)

The definitions the five `H1TCrossLateStaging` theorems read, in a module that imports no proofs.
`CrossLateResidue512` is the finite two-slice lemma over the last two quiet slices' tag fields, `Exact512` the regime
(real codes 0 on the last two slices, tag values multiples of 32, `R_(T−3)` exact on the 512 grid at the zero salt),
`H1TCrossLate512` the unit statement, and `absorbUnit` the unit of `H1TGapStaging`'s last section. The regime is empty
for `k ≤ 19,430`.
-/

namespace Pouw.Fp8Atom.H1T.CrossLate

open Pouw.Fp8Atom Pouw.Fp8Atom.H1T

/-- One slice's tag fields: the three tag lanes' sign bits and mantissas. -/
abbrev TagSalt := (Fin 3 → Bool) × (Fin 3 → Fin 8)

/-- A tag lane's value over 32F at `F = 1/4`: `±(8 + m)`, negative when the sign bit is set. -/
def tagMul (n : Bool) (m : Fin 8) : ℤ := (if n then -1 else 1) * (8 + (m.val : ℤ))

/-- A quiet slice's block-1 word over 32, for tag values `32·c` and tag fields `σ`. -/
def q32 (c : Fin 3 → ℤ) (σ : TagSalt) : ℤ :=
  tagMul (σ.1 0) (σ.2 0) * c 0 + tagMul (σ.1 1) (σ.2 1) * c 1 + tagMul (σ.1 2) (σ.2 2) * c 2

/-- The 512-steps one rounding adds to `512·a` for an increment `32·u` (`p`: `a` odd): `u/16` to nearest, ties to
even `a + ·`. -/
def rnd512 (p : Bool) (u : ℤ) : ℤ :=
  if u % 16 < 8 then u / 16 else if 8 < u % 16 then u / 16 + 1
  else if (u / 16 + if p then 1 else 0) % 2 = 0 then u / 16 else u / 16 + 1

/-- The 512-steps two roundings add to `512·a` for increments `32·u1`, `32·u2`. -/
def two512 (p : Bool) (u1 u2 : ℤ) : ℤ := rnd512 p u1 + rnd512 (p != (rnd512 p u1 % 2 == 1)) u2

/-- The difference of two columns' 512-steps over the two slices with tag fields `s`. -/
def cross512 (p p' : Bool) (c c' : Fin 3 → ℤ) (s : TagSalt × TagSalt) : ℤ :=
  two512 p (q32 c s.1) (q32 c s.2) - two512 p' (q32 c' s.1) (q32 c' s.2)

/-- **The finite two-slice lemma**: for tag values `32·c` (`|c_k| ≤ 14`) and `32·c'` with `c ≠ c'`, and either
parities `p`, `p'` of the two chains' `R_(T−3)/512`, the two columns' 512-steps over the last two quiet slices do not
differ by a constant as functions of those slices' tag fields. -/
def CrossLateResidue512 : Prop :=
  ∀ c c' : Fin 3 → ℤ, (∀ k, |c k| ≤ 14) → c ≠ c' → ∀ p p' : Bool,
    ∃ s s' : TagSalt × TagSalt, cross512 p p' c c' s ≠ cross512 p p' c c' s'

/-- **The exact-512 regime** of row `i` and column `j`: at least three slices; row and column real codes 0 on the last
two slices (`29(T − 2)` on); the column's tag values multiples of 32; and at the zero salt, `R_(T−3) = 512·a` with
`2^23 + 2^9 ≤ a ≤ 2^24 − 2^9`, and every block-2 word of slices `0..T−3` a multiple of 512. -/
def Exact512 (U : Unit) (i : Fin U.m) (j : Fin U.n) : Prop :=
  3 ≤ U.T ∧ (∀ r, 29 * (U.T - 2) ≤ r → val (U.row i r) = 0 ∧ val (U.col j r) = 0) ∧
    (∀ k, ∃ c : ℤ, val (tuple j.val k) = 32 * c) ∧
    (∃ a : ℤ, runVal U.T (U.row i) (U.col j) j.val (fun _ => extract rawZero) (U.T - 3) = 512 * a ∧
      2 ^ 23 + 2 ^ 9 ≤ a ∧ a + 2 ^ 9 ≤ 2 ^ 24) ∧
    ∀ q, U.T ≤ q → q + 3 ≤ 2 * U.T →
      ∃ n : ℤ, freshVal U.T (U.row i) (U.col j) j.val (fun _ => extract rawZero) q = 512 * n

/-- **The cross-column late pairs in the exact-512 regime**: on a legal unit, two running words of different columns
at positions from `T − 1` on, both columns in the regime, are different functions of the salt. -/
def H1TCrossLate512 : Prop :=
  ∀ U : Unit, U.Legal → ∀ (i : Fin U.m) (a a' : Fin U.n × Pos U.T), a.1 ≠ a'.1 → a.2.isRight = true →
    a'.2.isRight = true → U.T ≤ posIdx a.2 + 1 → U.T ≤ posIdx a'.2 + 1 → Exact512 U i a.1 →
    Exact512 U i a'.1 → U.word (i, a) ≠ U.word (i, a')

/-- The unit of `H1TGapStaging`'s last section: `k = 32,768` (`T = 1,130`), `x = b = 448` on slices `0..1127` and 0 on
slices 1128 and 1129, 23 columns. -/
def absorbUnit : Unit where
  m := 1
  k := 32768
  n := 23
  xq := fun _ r => if r.val < 32712 then ⟨0x7E, by norm_num⟩ else ⟨0, by norm_num⟩
  b := fun r _ => if r.val < 32712 then ⟨0x7E, by norm_num⟩ else ⟨0, by norm_num⟩

end Pouw.Fp8Atom.H1T.CrossLate
