module

public import Definitions.Pouw.Fp8Atom.H1TLemmas

/-! The definitions the five `H1TCrossLateStaging` theorems read; the exact-512 regime is empty for `k ≤ 19,430`. -/

@[expose] public section

namespace Pouw.Fp8Atom.H1T.CrossLate

open Pouw.Fp8Atom Pouw.Fp8Atom.H1T

abbrev TagSalt := (Fin 3 → Bool) × (Fin 3 → Fin 8)

/-- A tag lane's value over 32F at `F = 1/4`: `±(8 + m)`, negative when the sign bit is set. -/
def tagMul (n : Bool) (m : Fin 8) : ℤ := (if n then -1 else 1) * (8 + (m.val : ℤ))

def q32 (c : Fin 3 → ℤ) (σ : TagSalt) : ℤ :=
  tagMul (σ.1 0) (σ.2 0) * c 0 + tagMul (σ.1 1) (σ.2 1) * c 1 + tagMul (σ.1 2) (σ.2 2) * c 2

/-- The 512-steps one rounding adds to `512·a` for an increment `32·u` (`p`: `a` odd): `u/16` to nearest, ties to
even `a + ·`. -/
def rnd512 (p : Bool) (u : ℤ) : ℤ :=
  if u % 16 < 8 then u / 16 else if 8 < u % 16 then u / 16 + 1
  else if (u / 16 + if p then 1 else 0) % 2 = 0 then u / 16 else u / 16 + 1

def two512 (p : Bool) (u1 u2 : ℤ) : ℤ := rnd512 p u1 + rnd512 (p != (rnd512 p u1 % 2 == 1)) u2

def cross512 (p p' : Bool) (c c' : Fin 3 → ℤ) (s : TagSalt × TagSalt) : ℤ :=
  two512 p (q32 c s.1) (q32 c s.2) - two512 p' (q32 c' s.1) (q32 c' s.2)

def CrossLateResidue512 : Prop :=
  ∀ c c' : Fin 3 → ℤ, (∀ k, |c k| ≤ 14) → c ≠ c' → ∀ p p' : Bool,
    ∃ s s' : TagSalt × TagSalt, cross512 p p' c c' s ≠ cross512 p p' c c' s'

def Exact512 (U : Unit) (i : Fin U.m) (j : Fin U.n) : Prop :=
  3 ≤ U.T ∧ (∀ r, 29 * (U.T - 2) ≤ r → val (U.row i r) = 0 ∧ val (U.col j r) = 0) ∧
    (∀ k, ∃ c : ℤ, val (tuple j.val k) = 32 * c) ∧
    (∃ a : ℤ, runVal U.T (U.row i) (U.col j) j.val (fun _ => extract rawZero) (U.T - 3) = 512 * a ∧
      2 ^ 23 + 2 ^ 9 ≤ a ∧ a + 2 ^ 9 ≤ 2 ^ 24) ∧
    ∀ q, U.T ≤ q → q + 3 ≤ 2 * U.T →
      ∃ n : ℤ, freshVal U.T (U.row i) (U.col j) j.val (fun _ => extract rawZero) q = 512 * n

def H1TCrossLate512 : Prop :=
  ∀ U : Unit, U.Legal → ∀ (i : Fin U.m) (a a' : Fin U.n × Pos U.T), a.1 ≠ a'.1 → a.2.isRight = true →
    a'.2.isRight = true → U.T ≤ posIdx a.2 + 1 → U.T ≤ posIdx a'.2 + 1 → Exact512 U i a.1 →
    Exact512 U i a'.1 → U.word (i, a) ≠ U.word (i, a')

def absorbUnit : Unit where
  m := 1
  k := 32768
  n := 23
  xq := fun _ r => if r.val < 32712 then ⟨0x7E, by norm_num⟩ else ⟨0, by norm_num⟩
  b := fun r _ => if r.val < 32712 then ⟨0x7E, by norm_num⟩ else ⟨0, by norm_num⟩

end Pouw.Fp8Atom.H1T.CrossLate
