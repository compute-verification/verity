module

public import Definitions.Pouw.Assumptions.Dimension.Mixing

/-! The closed scalar pipe's definitions, on which `Pouw.Dimension.PipeAssumptions` states its named assumptions. -/

@[expose] public section

namespace Pouw.Dimension

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension.Assumptions

variable {N m n k : ℕ}

def PipeOpnd (Rs : List (Noise N m n k → ℚ)) (x : Noise N m n k → ℚ) : Prop :=
  x ∈ V0 N m n k ∨ ∃ r ∈ Rs, ∃ β : ℚ, x = β • r

def PipeStep (Rs : List (Noise N m n k → ℚ)) (f : Noise N m n k → ℚ) : Prop :=
  ∃ a b c, PipeOpnd Rs a ∧ PipeOpnd Rs b ∧ PipeOpnd Rs c ∧ f = a * b + c

def QuadStep (Rs : List (Noise N m n k → ℚ)) (f : Noise N m n k → ℚ) : Prop :=
  ∃ a b c, PipeOpnd Rs c ∧ f = a * b + c ∧
    ((a ∈ V0 N m n k ∧ b ∈ V0 N m n k) ∨ ((∃ α : ℚ, a = fun _ => α) ∧ PipeOpnd Rs b))

/-- A closed FP32 pipe, its outputs in program order. -/
def IsPipe (Rs : List (Noise N m n k → ℚ)) : Prop :=
  ∀ i (h : i < Rs.length), PipeStep (Rs.take i) (Rs[i]'h)

def IsQuadPipe (Rs : List (Noise N m n k → ℚ)) : Prop :=
  ∀ i (h : i < Rs.length), QuadStep (Rs.take i) (Rs[i]'h)

def ClosedPipe (cs : ℕ) (P : List (Instr (Noise N m n k))) : Prop :=
  (∀ c w, Instr.free c w ∉ P) ∧ (∀ p f, Instr.priced p f ∈ P → 16 ≤ p ∨ cs ≤ p) ∧ IsPipe (prodOuts P)

def QuadPipe (cs : ℕ) (P : List (Instr (Noise N m n k))) : Prop :=
  ClosedPipe cs P ∧ IsQuadPipe (prodOuts P)

def A1CostOn (C : ∀ {N m n k : ℕ}, List (Noise N m n k → ℚ) → Prop) (c γ : ℝ) (k : ℕ) (R : Mat) : Prop :=
  ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ), U ⊆ range N →
    ∀ Rs : List (Noise N m n k → ℚ), C Rs →
    ∀ S ⊆ U ×ˢ range (nonFinal k m n), (∀ x ∈ S, word k m n N R A B x.1 x.2 ∈ Rs) →
      c * S.card - c * γ * ((U.card * (m * n) * (3 * k / 16 - 1) : ℕ) : ℝ) ≤ Rs.length

end Pouw.Dimension
