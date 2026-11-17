module

public import Definitions.Pouw.Assumptions.Dimension.Forest

/-! The open quadratic pipe, in which an FP32 step may read any earlier priced value, and Theorem D in it with no A1. -/

@[expose] public section

namespace Pouw.Dimension.OpenPipe

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension Pouw.Dimension.Assumptions Pouw.Dimension.Forest

def OpenOK {N m n k : ℕ} (pb cs : ℕ) : List (Noise N m n k → ℚ) → List (Instr (Noise N m n k)) → Prop
  | _, [] => True
  | _, .free _ _ :: _ => False
  | Vs, .priced p f :: P => (pb ≤ p ∨ (cs ≤ p ∧ QuadStep Vs f)) ∧ OpenOK pb cs (Vs ++ [f]) P

/-- `pb` is the big instructions' least price and `cs` the FP32 steps'. -/
def OpenQuadPipe {N m n k : ℕ} (pb cs : ℕ) (P : List (Instr (Noise N m n k))) : Prop := OpenOK pb cs [] P

def QuadToOpen : Prop :=
  ∀ (N m n k cs : ℕ) (P : List (Instr (Noise N m n k))), QuadPipe cs P → OpenQuadPipe 16 cs P

def OpenCount {N m n k : ℕ} (w : ℕ → ℕ → Noise N m n k → ℚ) (U : Finset ℕ) : Prop :=
  ∀ (pb cs : ℕ) (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))), (∀ f ∈ L, f ∈ V0 N m n k) →
    OpenQuadPipe pb cs P → ∀ S ⊆ U ×ˢ range (nonFinal k m n), (∀ x ∈ S, w x.1 x.2 ∈ slots P L) →
    min pb (2 * cs) * S.card ≤ cost P

def Theorem1Open : Prop :=
  ∀ (N m n k : ℕ) (w : ℕ → ℕ → Noise N m n k → ℚ) (U : Finset ℕ),
    TwoAtomFree (familyQ w U) (atoms N m n k) → FamilyIndepQ w U → OpenCount w U

/-- **Theorem D in the open quadratic pipe**, with no A1: `16·(1 − γ)·D` at `γ = 1 − min(pb, 2·cs)/16`. -/
def AlgBoundOpenQuad : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))) (pb cs : ℕ),
    CheckpointIndependent k 16 R → MinRankGe k R 5 → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) →
    OpenQuadPipe pb cs P → Outputs k m n N R A B U P L →
    min pb (2 * cs) * (U.card * (m * n) * (3 * k / 16 - 1)) ≤ cost P

/-- At W1's prices, with no A1. -/
def AlgBoundOpenQuad8 : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))),
    CheckpointIndependent k 16 R → MinRankGe k R 5 → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) →
    OpenQuadPipe 16 8 P → Outputs k m n N R A B U P L →
    16 * (U.card * (m * n) * (3 * k / 16 - 1)) ≤ cost P

/-- **An open-pipe program that no closed pipe is**: one FP32 step's addend is a tensor-core word, so the open pipe
widens the class the bound covers. -/
def OpenPipeWitness : Prop :=
  ∀ k, 1024 ≤ k → 16 ∣ k → ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (u₀ : ℕ), U ⊆ range N → u₀ ∈ U → 0 < m → 0 < n →
    ∃ (R : Mat) (P : List (Instr (Noise N m n k))) (a b : Noise N m n k → ℚ), Admissible k 16 R ∧
      CheckpointIndependent k 16 R ∧ MinRankGe k R 16 ∧ OpenQuadPipe 16 8 P ∧ ¬ ClosedPipe 8 P ∧
      Outputs k m n N R A B U P (coords N m n k) ∧ a ∈ V0 N m n k ∧ b ∈ V0 N m n k ∧
      Instr.priced 16 (word k m n N R A B u₀ 0) ∈ P ∧ Instr.priced 8 (a * b + word k m n N R A B u₀ 0) ∈ P ∧
      cost P = 16 * (U.card * (m * n) * (3 * k / 16 - 1)) + 8

end Pouw.Dimension.OpenPipe
