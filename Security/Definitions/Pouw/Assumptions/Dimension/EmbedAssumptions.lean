import Definitions.Pouw.Assumptions.Dimension.OpenPipe

/-!
# The FP32 pipe, the residual assumption `A1_fp`, and `CanonicalNaN` (draft semantics)

**Draft semantics, pending review**.

`EmbedM4090` (`Pouw.Dimension.Embed`) maps every `M4090` program into the open quadratic pipe at no greater cost,
except its non-quadratic FP32 instructions. What is left is a statement about the FP32 class alone: the **FP32 pipe**
`FpPipe fpSem` has no free instruction, and every priced instruction is **big** (price at least 16, any function of
the noise) or an **FP32 step** (price at least 8): one FP32 operation `fpSem κ x y z` of free or earlier values. It
has no tensor cores, loads, selects, sub-word operations, moves or pre-salt data.

* **The FP32 class is exactly FFMA, FADD and FMUL**, with their operand modifiers. W1 prices by measured throughput
  per register written, and every other opcode measured on the 4090 costs at least 16 per register. That includes FSET,
  which is `int`: FSET.BF runs on the ALU at 16.4 per register, 16.6 with a data-dependent predicate, and
  `set.*.u32.f32` is FSETP + SEL (§11, Round 6).
* **`A1_fp fpSem γ`** (named assumption): under checkpoint independence and MinRank ≥ 5, every `FpPipe fpSem`
  program on inputs in `V₀` whose slots include the non-final checked words at `S` costs at least `(1 − γ)·16·|S|`
  (`A1fpOn` for one instance). What the embedding removes is every instruction class but FP32, not logical
  strength. Its satisfiability witnesses are `A1fpFadd` (`γ = 0`, FADD only) and `A1fpHalf` (`γ = 1/2`, every
  `fpSem`).
* **`CanonicalNaN fpSem`** (named assumption, the 4090's behaviour, measured in Round 5): no FFMA, FADD or FMUL
  outputs a value in the range gap `NaNGap`, the NaN patterns other than `0x7FFFFFFF` and `0xFFFFFFFF` read as
  int32. Its witness is `CanonicalNaNSat`.
* `fpSem` is a parameter. The intended reading is IEEE binary32 on the int32 reading, round to nearest,
  flush-to-zero off, with the canonical NaN; no IEEE semantics is defined here.
-/

namespace Pouw.Dimension.Embed

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension Pouw.Dimension.Assumptions Pouw.Dimension.OpenPipe

variable {N m n k : ℕ}

/-- An FP32 operand: a free value or an earlier priced value. -/
def FpOpnd (Vs : List (Noise N m n k → ℚ)) (x : Noise N m n k → ℚ) : Prop := x ∈ V0 N m n k ∨ x ∈ Vs

/-- **One FP32 operation** `κ` (FFMA, FADD or FMUL with operand modifiers) of three operands, lane by lane. -/
def FpStep (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (Vs : List (Noise N m n k → ℚ)) (f : Noise N m n k → ℚ) : Prop :=
  ∃ κ x y z, FpOpnd Vs x ∧ FpOpnd Vs y ∧ FpOpnd Vs z ∧ f = fun ω => fpSem κ (x ω) (y ω) (z ω)

/-- The FP32 pipe's rule, given the earlier priced values `Vs`: no free instruction; each priced instruction is big
(price at least 16) or an FP32 step over `Vs` (price at least 8). -/
def FpOK (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) : List (Noise N m n k → ℚ) → List (Instr (Noise N m n k)) → Prop
  | _, [] => True
  | _, .free _ _ :: _ => False
  | Vs, .priced p f :: P => (16 ≤ p ∨ (8 ≤ p ∧ FpStep fpSem Vs f)) ∧ FpOK fpSem (Vs ++ [f]) P

/-- **The FP32 pipe** for the FP32 semantics `fpSem`. -/
def FpPipe (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (P : List (Instr (Noise N m n k))) : Prop := FpOK fpSem [] P

/-- The FADD-only semantics, `x + y`: every step is quadratic. -/
def faddSem (_κ : ℕ) (x y _z : ℚ) : ℚ := x + y

/-- `A1_fp`'s conclusion for one instance: every FP32-pipe program on inputs in `V₀` whose slots include the non-final
checked words at `S ⊆ U × [0, nonFinal)` costs at least `(1 − γ)·16·|S|`. -/
def A1fpOn (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (γ : ℝ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat)
    (B : ℕ → Mat) (U : Finset ℕ) : Prop :=
  ∀ (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))), (∀ f ∈ L, f ∈ V0 N m n k) → FpPipe fpSem P →
    ∀ S ⊆ U ×ˢ range (nonFinal k m n), (∀ x ∈ S, word k m n N R A B x.1 x.2 ∈ slots P L) →
      (1 - γ) * (16 * S.card : ℝ) ≤ cost P

/-- **Assumption `A1_fp(fpSem, γ)`** (named; draft semantics, pending review): under checkpoint
independence and MinRank ≥ 5, `A1fpOn` holds: every FP32-pipe program that outputs `|S|` non-final checked words costs
at least `(1 − γ)·16·|S|`. -/
def A1fp (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (γ : ℝ) : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ),
    CheckpointIndependent k 16 R → MinRankGe k R 5 → U ⊆ range N → A1fpOn fpSem γ k m n N R A B U

/-- **The range gap**: the int32 readings of the NaN patterns other than the canonical `0x7FFFFFFF` and `0xFFFFFFFF`
(`−1`): `0xFF800001 … 0xFFFFFFFE` read as `−8388607 … −2`, and `0x7F800001 … 0x7FFFFFFE` read as
`2139095041 … 2147483646`. Excluding `−1` is conservative: no instruction priced 8 writes it (§10, §11), so it could
be re-admitted. -/
def NaNGap (v : ℚ) : Prop := (-8388607 ≤ v ∧ v ≤ -2) ∨ (2139095041 ≤ v ∧ v ≤ 2147483646)

/-- **Assumption `CanonicalNaN fpSem`** (named; the 4090's behaviour): no FFMA, FADD or FMUL output lies in the range
gap on the int32 reading. They output no NaN pattern but `0x7FFFFFFF`: measured in all 72 forms (rounding modes,
`.ftz`, `.sat`, neg and abs operands, immediates), 43.6·10⁹ NaN outputs, none other. -/
def CanonicalNaN (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) : Prop := ∀ κ a b c, ¬ NaNGap (fpSem κ a b c)

end Pouw.Dimension.Embed
