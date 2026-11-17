module

public import Definitions.Pouw.Basic.Prob
public import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-! R_word-NCP, straight-line word programs over a unit's checked positions; `FinalOnly` is a named hypothesis, not proved, and the model shows a TT_NCP-shaped bound can hold, not evidence for TT_NCP. -/

@[expose] public section

namespace Pouw.NCP.Word

open Finset

def sgn : Bool → ℤ
  | true => 1
  | false => -1

inductive Rule
  | fresh
  | copy (s : Bool) (a : ℕ)
  | sum (s : Bool) (a : ℕ) (t : Bool) (b : ℕ)
  | const (c : ℤ)
  deriving DecidableEq

def Rule.cost : Rule → ℕ
  | .fresh => 16
  | .copy _ _ => 0
  | .sum _ _ _ _ => 8
  | .const _ => 0

def Rule.target (x : ℕ → ℤ) : Rule → ℤ
  | .fresh => 0
  | .copy s a => sgn s * x a
  | .sum s a t b => sgn s * x a + sgn t * x b
  | .const c => c

def Rule.Reads (p : ℕ) : Rule → Prop
  | .fresh => True
  | .copy _ a => a < p
  | .sum _ a _ b => a < p ∧ b < p
  | .const _ => True

structure Adversary where
  rule : ℕ → ℕ → Rule

def unitCost (n : ℕ → ℕ) (A : Adversary) (u : ℕ) : ℕ := ∑ p ∈ range (n u), (A.rule u p).cost

def cost (n : ℕ → ℕ) (N : ℕ) (A : Adversary) : ℕ := ∑ u ∈ range N, unitCost n A u

def Valid (n : ℕ → ℕ) (A : Adversary) : Prop := ∀ u, ∀ p < n u, (A.rule u p).Reads p

variable {Ω : Type}

def Holds (r : Rule) (x : ℕ → ℤ) (p : ℕ) : Prop := r = .fresh ∨ x p = r.target x

def Correct (n : ℕ → ℕ) (X : Ω → ℕ → ℕ → ℤ) (A : Adversary) (ω : Ω) (u : ℕ) : Prop :=
  ∀ p < n u, Holds (A.rule u p) (X ω u) p

open Classical in
noncomputable def correctSet (n : ℕ → ℕ) (X : Ω → ℕ → ℕ → ℤ) (N : ℕ) (A : Adversary) (ω : Ω) : Finset ℕ :=
  (range N).filter (Correct n X A ω)

def Identity (X : Ω → ℕ → ℕ → ℤ) (A : Adversary) (u p : ℕ) : Prop :=
  A.rule u p ≠ .fresh ∧ ∀ ω, X ω u p = (A.rule u p).target (X ω u)

open Classical in
noncomputable def identCount (n : ℕ → ℕ) (X : Ω → ℕ → ℕ → ℤ) (A : Adversary) (u : ℕ) : ℕ :=
  ((range (n u)).filter (Identity X A u)).card

def NzNCP [Fintype Ω] (n : ℕ → ℕ) (X : Ω → ℕ → ℕ → ℤ) : Prop :=
  ∀ (u : ℕ) (P : Finset ℕ), P ⊆ range (n u) → ∀ r : ℕ → Rule,
    (∀ p ∈ P, r p ≠ .fresh ∧ (r p).Reads p ∧ ¬ ∀ ω, X ω u p = (r p).target (X ω u)) →
    pr (fun ω => ∀ p ∈ P, X ω u p = (r p).target (X ω u)) ≤ (1 / 2 : ℝ) ^ P.card

/-- A named hypothesis, not proved for NCP. -/
def FinalOnly (n : ℕ → ℕ) (X : Ω → ℕ → ℕ → ℤ) (A : Adversary) (f : ℕ → ℕ) : Prop :=
  ∀ u, identCount n X A u ≤ f u

def bits {M : ℕ} (ω : Fin M → Bool) (p : ℕ) : ℤ := if h : p < M then ((ω ⟨p, h⟩).toNat : ℤ) else 0

def constAdv (x : ℕ → ℕ → ℤ) : Adversary := ⟨fun u p => .const (x u p)⟩

def WordBound : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (n : ℕ → ℕ) (X : Ω → ℕ → ℕ → ℤ) (N : ℕ) (A : Adversary) (g₀ : ℝ),
    0 ≤ g₀ → g₀ < 1 → Valid n A → NzNCP n X →
    pr (fun ω => (cost n N A : ℝ) / (1 - g₀) < 16 * ∑ u ∈ correctSet n X N A ω, (n u : ℝ)) ≤
      ∑ u ∈ range N, (2 : ℝ) ^ (-(g₀ * n u - identCount n X A u))

def WordFloor : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (n : ℕ → ℕ) (X : Ω → ℕ → ℕ → ℤ) (N : ℕ) (A : Adversary) (g₀ : ℝ) (w c : ℕ),
    0 ≤ g₀ → g₀ < 1 → Valid n A → NzNCP n X → (∀ u, n u = w * c) → FinalOnly n X A (fun _ => c) →
    pr (fun ω => (cost n N A : ℝ) / (1 - g₀) < 16 * ∑ u ∈ correctSet n X N A ω, (n u : ℝ)) ≤
      N * (2 : ℝ) ^ (-(c * (g₀ * w - 1)))

/-- The witness: independent uniform bits, `M` positions per unit. -/
def WordNzWitness : Prop :=
  ∀ M : ℕ, NzNCP (fun _ => M) (fun (ω : Fin M → Bool) (_ : ℕ) => bits ω)

/-- The witness: on independent uniform bits no straight-line rule is an identity. -/
def WordFinalWitness : Prop :=
  ∀ (M : ℕ) (A : Adversary), Valid (fun _ => M) A →
    FinalOnly (fun _ => M) (fun (ω : Fin M → Bool) (_ : ℕ) => bits ω) A (fun _ => 0)

def WordZeroNoise : Prop :=
  ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (n : ℕ → ℕ) (x : ℕ → ℕ → ℤ) (N : ℕ) (g₀ : ℝ),
    g₀ < 1 → (∃ u < N, 0 < n u) →
    NzNCP n (fun (_ : Ω) => x) ∧ cost n N (constAdv x) = 0 ∧
      (∀ u, identCount n (fun (_ : Ω) => x) (constAdv x) u = n u) ∧
      pr (fun ω : Ω => (cost n N (constAdv x) : ℝ) / (1 - g₀) <
        16 * ∑ u ∈ correctSet n (fun (_ : Ω) => x) N (constAdv x) ω, (n u : ℝ)) = 1

end Pouw.NCP.Word
