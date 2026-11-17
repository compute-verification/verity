module

public import Definitions.Pouw.NCP.Relation
public import Mathlib.LinearAlgebra.Dimension.Finrank
public import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas

/-! Theorem D, the dimension bound in AlgM_16, over straight-line noise-oblivious programs whose free instructions are exact ℚ-linear combinations. -/

@[expose] public section

namespace Pouw.Dimension

open Finset Pouw.NCP Pouw.NCP.Relation

/-- Its output is one 32-bit register slot. -/
inductive Instr (G : Type) where
  /-- `c + Σ_{s < len} w s · slot s`, free; reads register slots only, never a predicate. -/
  | free (c : ℚ) (w : ℕ → ℚ)
  | priced (price : ℕ) (f : G → ℚ)

variable {G : Type}

def Instr.out : Instr G → List (G → ℚ) → (G → ℚ)
  | .free c w, L => fun g => c + ∑ s ∈ range L.length, w s * (L.getD s 0) g
  | .priced _ f, _ => f

def Instr.price : Instr G → ℕ
  | .free _ _ => 0
  | .priced p _ => p

def Instr.isPriced : Instr G → Bool
  | .free _ _ => false
  | .priced _ _ => true

def slots (P : List (Instr G)) (L : List (G → ℚ)) : List (G → ℚ) :=
  P.foldl (fun S ins => S ++ [ins.out S]) L

def nPriced (P : List (Instr G)) : ℕ := P.countP Instr.isPriced

def cost (P : List (Instr G)) : ℕ := (P.map Instr.price).sum

def H16 (P : List (Instr G)) : Prop := ∀ ins ∈ P, ins.isPriced = true → 16 ≤ ins.price

def span1 (L : List (G → ℚ)) : Submodule ℚ (G → ℚ) := Submodule.span ℚ (insert 1 {f | f ∈ L})

/-- Each coordinate is `v − 31`. -/
def coords (N m n k : ℕ) : List (Noise N m n k → ℚ) :=
  (List.finRange N).flatMap (fun u => (List.finRange m).flatMap fun i =>
      (List.finRange k).map fun l => fun ω => ((ω.1 u i l : ℕ) : ℚ) - 31) ++
    (List.finRange k).flatMap fun l => (List.finRange n).map fun j => fun ω => ((ω.2 l j : ℕ) : ℚ) - 31

def fnF (N m n k : ℕ) : Submodule ℚ (Noise N m n k → ℚ) :=
  Submodule.span ℚ {φ | ∀ ω ω' : Noise N m n k, ω.2 = ω'.2 → φ ω = φ ω'}

def V0 (N m n k : ℕ) : Submodule ℚ (Noise N m n k → ℚ) := span1 (coords N m n k) ⊔ fnF N m n k

def nonFinal (k m n : ℕ) : ℕ := (3 * k / 16 - 1) * (m * n)

def word (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (u p : ℕ) :
    Noise N m n k → ℚ :=
  fun ω => (checked k m n N R (A ω.2) B ω u p : ℚ)

def Outputs (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (P : List (Instr (Noise N m n k))) (L : List (Noise N m n k → ℚ)) : Prop :=
  ∀ u ∈ U, ∀ p < nonFinal k m n, word k m n N R A B u p ∈ slots P L

def MinRank2 (k : ℕ) (R : Mat) : Prop :=
  ∀ q x y : ℕ → ℚ,
    (∀ a < k, ∀ b < k, ∑ t ∈ range (3 * k / 16 - 1), q t * (ckMat k R (16 * (t + 1)) a b : ℚ) = x a * y b) →
    (∀ a < k, x a = 0) ∨ (∀ b < k, y b = 0)

def DimLePriced : Prop :=
  ∀ (G : Type) [Fintype G] (P : List (Instr G)) (L : List (G → ℚ)),
    Module.finrank ℚ (span1 (slots P L)) ≤ Module.finrank ℚ (span1 L) + nPriced P

def SecondDifference : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (c : ℕ → ℕ → ℚ), U ⊆ range N →
    (∑ u ∈ U, ∑ p ∈ range (nonFinal k m n), c u p • word k m n N R A B u p) ∈ V0 N m n k →
    ∀ u ∈ U, ∀ e < m * n, ∀ a < k, ∀ b < k,
      ∑ t ∈ range (3 * k / 16 - 1), c u (t * (m * n) + e) * (ckMat k R (16 * (t + 1)) a b : ℚ) = 0

def PricedCount : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))),
    CheckpointIndependent k 16 R → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) → Outputs k m n N R A B U P L →
    U.card * (m * n) * (3 * k / 16 - 1) ≤ nPriced P

def AlgBound : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))),
    CheckpointIndependent k 16 R → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) → H16 P → Outputs k m n N R A B U P L →
    16 * (U.card * (m * n) * (3 * k / 16 - 1)) ≤ cost P

/-- The event is impossible, not only unlikely: correctness is on the whole grid. -/
def TTNCPAlg : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))) (g₀ : ℝ),
    16 ∣ k → 0 < k → CheckpointIndependent k 16 R → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) → H16 P →
    Outputs k m n N R A B U P L → 16 / (3 * k : ℝ) ≤ g₀ → g₀ < 1 →
    ¬ ((cost P : ℝ) / (1 - g₀) < ∑ _u ∈ U, 3 * ((m * k * n : ℕ) : ℝ))

/-- The witness holds for every `k ≥ 1024` with `16 ∣ k`, at the shift by 24 (`independentWitness`), with one priced
instruction at 16 per word. -/
def AlgWitness : Prop :=
  ∀ k, 1024 ≤ k → 16 ∣ k → ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ),
    U ⊆ range N →
    ∃ (R : Mat) (P : List (Instr (Noise N m n k))), Admissible k 16 R ∧ CheckpointIndependent k 16 R ∧
      (∀ f ∈ coords N m n k, f ∈ V0 N m n k) ∧ H16 P ∧ Outputs k m n N R A B U P (coords N m n k) ∧
      cost P = 16 * (U.card * (m * n) * (3 * k / 16 - 1))

def AlgSharp : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ) (g₀ : ℝ),
    16 ∣ k → 0 < k → U ⊆ range N → U.Nonempty → 0 < m * n → g₀ < 16 / (3 * k : ℝ) →
    ∃ P : List (Instr (Noise N m n k)), H16 P ∧ Outputs k m n N R A B U P (coords N m n k) ∧
      (cost P : ℝ) / (1 - g₀) < ∑ _u ∈ U, 3 * ((m * k * n : ℕ) : ℝ)

end Pouw.Dimension
