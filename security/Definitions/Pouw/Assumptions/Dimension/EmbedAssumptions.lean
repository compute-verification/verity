module

public import Definitions.Pouw.Assumptions.Dimension.OpenPipe
public import Definitions.Pouw.Dimension.Lift

/-! The FP32 pipe over a parameter `fpSem`, for which no IEEE semantics is defined, and the named assumptions `A1_fp` and `CanonicalNaN` on it (draft semantics, pending review). -/

@[expose] public section

namespace Pouw.Dimension.Embed

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension Pouw.Dimension.Assumptions Pouw.Dimension.OpenPipe

variable {N m n k : ℕ}

def FpOpnd (Vs : List (Noise N m n k → ℚ)) (x : Noise N m n k → ℚ) : Prop := x ∈ V0 N m n k ∨ x ∈ Vs

/-- **One FP32 operation** `κ` (FFMA, FADD or FMUL with operand modifiers) of three operands, lane by lane. -/
def FpStep (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (Vs : List (Noise N m n k → ℚ)) (f : Noise N m n k → ℚ) : Prop :=
  ∃ κ x y z, FpOpnd Vs x ∧ FpOpnd Vs y ∧ FpOpnd Vs z ∧ f = fun ω => fpSem κ (x ω) (y ω) (z ω)

def FpOK (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) : List (Noise N m n k → ℚ) → List (Instr (Noise N m n k)) → Prop
  | _, [] => True
  | _, .free _ _ :: _ => False
  | Vs, .priced p f :: P => (16 ≤ p ∨ (8 ≤ p ∧ FpStep fpSem Vs f)) ∧ FpOK fpSem (Vs ++ [f]) P

def FpPipe (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (P : List (Instr (Noise N m n k))) : Prop := FpOK fpSem [] P

/-- The FADD-only semantics, `x + y`: every step is quadratic. -/
def faddSem (_κ : ℕ) (x y _z : ℚ) : ℚ := x + y

def A1fpOn (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (γ : ℝ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat)
    (B : ℕ → Mat) (U : Finset ℕ) : Prop :=
  ∀ (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))), (∀ f ∈ L, f ∈ V0 N m n k) → FpPipe fpSem P →
    ∀ S ⊆ U ×ˢ range (nonFinal k m n), (∀ x ∈ S, word k m n N R A B x.1 x.2 ∈ slots P L) →
      (1 - γ) * (16 * S.card : ℝ) ≤ cost P

/-- **Assumption `A1_fp(fpSem, γ)`** (named; draft semantics, pending review). -/
def A1fp (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (γ : ℝ) : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ),
    CheckpointIndependent k 16 R → MinRankGe k R 5 → U ⊆ range N → A1fpOn fpSem γ k m n N R A B U

/-- The int32 readings of the NaN patterns other than `0x7FFFFFFF` and `0xFFFFFFFF` (`−1`). Excluding `−1` is
conservative: no instruction priced 8 writes it. -/
def NaNGap (v : ℚ) : Prop := (-8388607 ≤ v ∧ v ≤ -2) ∨ (2139095041 ≤ v ∧ v ≤ 2147483646)

/-- **Assumption `CanonicalNaN fpSem`** (named): the RTX 4090's FFMA, FADD and FMUL output no NaN pattern but
`0x7FFFFFFF`. -/
def CanonicalNaN (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) : Prop := ∀ κ a b c, ¬ NaNGap (fpSem κ a b c)

end Pouw.Dimension.Embed
