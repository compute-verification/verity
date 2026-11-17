module

public import Definitions.Pouw.Basic.Prob
public import Mathlib.Algebra.BigOperators.Group.Finset.Defs
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-! R_copy, a restricted model whose bound holds because of the noise: a witness that a TT-shaped bound can hold, not evidence for TT. -/

@[expose] public section

namespace Pouw.Copy

open Finset

structure Adversary where
  cls : ℕ → ℕ → ℕ

def classes (n : ℕ → ℕ) (A : Adversary) (u : ℕ) : Finset ℕ := (range (n u)).image (A.cls u)

def cost (n : ℕ → ℕ) (N : ℕ) (A : Adversary) : ℕ := 16 * ∑ u ∈ range N, (classes n A u).card

variable {Ω V : Type}

def Correct (n : ℕ → ℕ) (X : Ω → ℕ → ℕ → V) (A : Adversary) (ω : Ω) (u : ℕ) : Prop :=
  ∀ p < n u, ∀ p' < n u, A.cls u p = A.cls u p' → X ω u p = X ω u p'

open Classical in
noncomputable def correctSet (n : ℕ → ℕ) (X : Ω → ℕ → ℕ → V) (N : ℕ) (A : Adversary) (ω : Ω) : Finset ℕ :=
  (range N).filter (Correct n X A ω)

def Nz [Fintype Ω] (n : ℕ → ℕ) (X : Ω → ℕ → ℕ → V) : Prop :=
  ∀ (u : ℕ) (P : Finset ℕ), P ⊆ range (n u) → ∀ g : ℕ → (ℕ → V) → V,
    (∀ p ∈ P, ∀ x y : ℕ → V, (∀ q, q ∉ P → x q = y q) → g p x = g p y) →
    pr (fun ω => ∀ p ∈ P, X ω u p = g p (X ω u)) ≤ (1 / 2 : ℝ) ^ P.card

def bits {M : ℕ} (ω : Fin M → Bool) (p : ℕ) : ℕ := if h : p < M then (ω ⟨p, h⟩).toNat else 0

end Pouw.Copy
