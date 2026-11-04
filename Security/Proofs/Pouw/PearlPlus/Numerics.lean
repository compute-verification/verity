import Evaluate.Fp32
import Evaluate.TC.Dots
import Definitions.Pouw.PearlC.RowPrims

/-!
# Pearl+'s bridge, the numerics: the check's FP32 add, square root and BF16 step are the semantics'

`PouwCheck.PearlPlus.Words` runs core's evaluator: `Evaluate.Fp32.f32AddV2` in the chain's promotion, the row statistics'
butterfly and sum and the peel's `B` factor, `Evaluate.Fp32.sqrtRn` for the row's `ρ`, and `Evaluate.TC.hopperBf16Dot16`
for the peel's four K16 steps. The semantics reads `TileCircuit.f32Add` (with `PeelExact.addWord?`, its rounding on finite
words), `RowCircuit.sqrtWord` and `PeelExact.k16Step`. Each statement says the two agree wherever the check reads the
result, which it refuses when it is not finite.
-/

namespace Pouw.PearlPlus.Bridge

open Pouw.Fp8Atom Pouw.PearlC

/-- **The check's FP32 add is the semantics'**: where `Evaluate.Fp32.f32AddV2 a b` is a finite word, `a` and `b` are
finite and it is `TileCircuit.f32Add a b`, so `PeelExact.addWord? a b` is `some` of it. -/
def Fp32AddIsTheSpec : Prop :=
  ∀ a b : ℕ, (wordQ (Evaluate.Fp32.f32AddV2 a b)).isSome →
    (wordQ a).isSome ∧ (wordQ b).isSome ∧ Evaluate.Fp32.f32AddV2 a b = TileCircuit.f32Add a b

/-- **The check's square root is the semantics'**: where `Evaluate.Fp32.sqrtRn a` is a finite word, it is
`RowCircuit.sqrtWord a` (both `sqrt.rn.f32`; they differ only in the NaN they write). -/
def SqrtIsTheSpec : Prop :=
  ∀ a : ℕ, (wordQ (Evaluate.Fp32.sqrtRn a)).isSome → Evaluate.Fp32.sqrtRn a = RowCircuit.sqrtWord a

/-- **The check's BF16 step is `k16Step`**: `Evaluate.TC.hopperBf16Dot16` on a finite FP32 accumulator and 16 pairs of
finite BF16 words is `PeelExact.k16Step` on the same words. -/
def Bf16DotIsTheStep : Prop :=
  ∀ (acc : ℕ) (a b : List ℕ), (wordQ acc).isSome → a.length = 16 → b.length = 16 →
    (∀ x ∈ a, x < 2 ^ 16 ∧ PeelExact.bf16Exp x ≠ 255) → (∀ y ∈ b, y < 2 ^ 16 ∧ PeelExact.bf16Exp y ≠ 255) →
    Evaluate.TC.hopperBf16Dot16 acc a b = PeelExact.k16Step acc (fun l => a.getD l.val 0) (fun l => b.getD l.val 0)

end Pouw.PearlPlus.Bridge
