import Definitions.Pouw.Basic.Prob
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# R_copy: a restricted model with a provable, noise-dependent bound (trusted; tt-attacks §6)

Unit `u < N` has `n u` checked positions, and the value at position `p` is `X ω u p`, a function of the noise `ω`
(uniform over a finite type). A copy adversary fixes, before the salt, a class for every position. It computes one
value per class, at 16 units (one k16 output slot at `d = 16`), and copies it to the class's other positions, so it is
correct on `u` iff the values within each of `u`'s classes are equal.

R_copy captures tt-attacks §2's floor and its route 1 (copies) and nothing else: not adaptivity (the classes are
fixed before the salt), not the FP32 route (8 units per word), not shared tables. It is a witness that a TT-shaped
bound can hold because of the noise, not evidence for TT.
-/

namespace Pouw.Copy

open Finset

/-- A copy adversary: position `p` of unit `u` is in class `cls u p`. -/
structure Adversary where
  cls : ℕ → ℕ → ℕ

/-- Unit `u`'s classes. -/
def classes (n : ℕ → ℕ) (A : Adversary) (u : ℕ) : Finset ℕ := (range (n u)).image (A.cls u)

/-- The adversary's cost over units `u < N`: 16 per class. -/
def cost (n : ℕ → ℕ) (N : ℕ) (A : Adversary) : ℕ := 16 * ∑ u ∈ range N, (classes n A u).card

variable {Ω V : Type}

/-- Correct on `u`: the values within each of `u`'s classes are equal. -/
def Correct (n : ℕ → ℕ) (X : Ω → ℕ → ℕ → V) (A : Adversary) (ω : Ω) (u : ℕ) : Prop :=
  ∀ p < n u, ∀ p' < n u, A.cls u p = A.cls u p' → X ω u p = X ω u p'

open Classical in
/-- The units `u < N` it gets right. -/
noncomputable def correctSet (n : ℕ → ℕ) (X : Ω → ℕ → ℕ → V) (N : ℕ) (A : Adversary) (ω : Ω) : Finset ℕ :=
  (range N).filter (Correct n X A ω)

/-- **Nz** (tt-attacks §6): given the values at its unit's other positions, the values at any set `P` of a unit's
positions each take any fixed value with probability at most `1/2`, independently. Formally: for every `P` and every
target `g p` that reads only the unit's positions outside `P`, all of `P` hit their targets with probability at most
`2^−|P|`. -/
def Nz [Fintype Ω] (n : ℕ → ℕ) (X : Ω → ℕ → ℕ → V) : Prop :=
  ∀ (u : ℕ) (P : Finset ℕ), P ⊆ range (n u) → ∀ g : ℕ → (ℕ → V) → V,
    (∀ p ∈ P, ∀ x y : ℕ → V, (∀ q, q ∉ P → x q = y q) → g p x = g p y) →
    pr (fun ω => ∀ p ∈ P, X ω u p = g p (X ω u)) ≤ (1 / 2 : ℝ) ^ P.card

/-- Values uniform and independent on `{0, 1}`: position `p < M` reads bit `p` of `ω`. -/
def bits {M : ℕ} (ω : Fin M → Bool) (p : ℕ) : ℕ := if h : p < M then (ω ⟨p, h⟩).toNat else 0

end Pouw.Copy
