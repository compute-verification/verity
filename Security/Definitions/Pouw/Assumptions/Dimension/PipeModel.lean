import Definitions.Pouw.Assumptions.Dimension.Mixing
import Definitions.Pouw.Dimension.Lift

/-!
# The closed scalar pipe (trusted definitions)

The model of `Pouw.Dimension.Pipe`'s docstring (AlgM_4090 with a closed scalar pipe): FP32 operands and steps, closed
and quadratic pipes, the programs `ClosedPipe` and `QuadPipe`, and `A1CostOn`, the conclusion `A1_cost` states for one
class of pipes. The named assumptions over this model are in `Pouw.Dimension.PipeAssumptions`.
-/

namespace Pouw.Dimension

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension.Assumptions

variable {N m n k : ℕ}

/-- An FP32 operand, given the earlier FP32 outputs `Rs`: a free value, or a multiple of an earlier FP32 output. -/
def PipeOpnd (Rs : List (Noise N m n k → ℚ)) (x : Noise N m n k → ℚ) : Prop :=
  x ∈ V0 N m n k ∨ ∃ r ∈ Rs, ∃ β : ℚ, x = β • r

/-- **An FP32 step** of the closed pipe: `a·b + c`, with every operand a `PipeOpnd`. -/
def PipeStep (Rs : List (Noise N m n k → ℚ)) (f : Noise N m n k → ℚ) : Prop :=
  ∃ a b c, PipeOpnd Rs a ∧ PipeOpnd Rs b ∧ PipeOpnd Rs c ∧ f = a * b + c

/-- **A quadratic FP32 step**: `a·b + c` with both factors free values, or with `a` a constant. -/
def QuadStep (Rs : List (Noise N m n k → ℚ)) (f : Noise N m n k → ℚ) : Prop :=
  ∃ a b c, PipeOpnd Rs c ∧ f = a * b + c ∧
    ((a ∈ V0 N m n k ∧ b ∈ V0 N m n k) ∨ ((∃ α : ℚ, a = fun _ => α) ∧ PipeOpnd Rs b))

/-- A closed FP32 pipe, listed in program order: every output is a step over the earlier ones. -/
def IsPipe (Rs : List (Noise N m n k → ℚ)) : Prop :=
  ∀ i (h : i < Rs.length), PipeStep (Rs.take i) (Rs[i]'h)

/-- A quadratic closed FP32 pipe. -/
def IsQuadPipe (Rs : List (Noise N m n k → ℚ)) : Prop :=
  ∀ i (h : i < Rs.length), QuadStep (Rs.take i) (Rs[i]'h)

/-- **AlgM_4090 with a closed scalar pipe** at FP32 price `cs`: no free instruction, every priced instruction costs at
least 16 or at least `cs`, and the instructions below 16 (`prodOuts`, in program order) form a closed FP32 pipe. -/
def ClosedPipe (cs : ℕ) (P : List (Instr (Noise N m n k))) : Prop :=
  (∀ c w, Instr.free c w ∉ P) ∧ (∀ p f, Instr.priced p f ∈ P → 16 ≤ p ∨ cs ≤ p) ∧ IsPipe (prodOuts P)

/-- The closed-pipe programs whose FP32 pipe is quadratic. -/
def QuadPipe (cs : ℕ) (P : List (Instr (Noise N m n k))) : Prop :=
  ClosedPipe cs P ∧ IsQuadPipe (prodOuts P)

/-- The conclusion of `A1_cost` for one class of pipes `C`: every pipe in `C` whose outputs include the non-final
checked words of `S ⊆ U × [0, nonFinal)` has at least `c·|S| − c·γ·D` steps. -/
def A1CostOn (C : ∀ {N m n k : ℕ}, List (Noise N m n k → ℚ) → Prop) (c γ : ℝ) (k : ℕ) (R : Mat) : Prop :=
  ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ), U ⊆ range N →
    ∀ Rs : List (Noise N m n k → ℚ), C Rs →
    ∀ S ⊆ U ×ˢ range (nonFinal k m n), (∀ x ∈ S, word k m n N R A B x.1 x.2 ∈ Rs) →
      c * S.card - c * γ * ((U.card * (m * n) * (3 * k / 16 - 1) : ℕ) : ℝ) ≤ Rs.length

end Pouw.Dimension
