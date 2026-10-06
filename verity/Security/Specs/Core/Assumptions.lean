import Definitions.Core.TC.Spec
import Definitions.Core.PrivatePartition

/-!
# Named assumptions

The hardware semantics Verity's theorems take without proof, each a `Prop` taken as a hypothesis (no `axiom`). The
moniker in each docstring is the entry of the overnight assumptions table that the red team rates.
-/

namespace Verity.Assumptions

open Verity.TC

/-- The domain of `tc_dot_total` on a pipeline: an FP32 accumulator word and `k` BF16 words per operand. -/
def StepInputs (pipe : Pipeline) (acc : Nat) (a b : List Nat) : Prop :=
  acc < 2 ^ 32 ∧ a.length = pipe.k ∧ b.length = pipe.k ∧ (∀ x ∈ a, x < 2 ^ 16) ∧ (∀ x ∈ b, x < 2 ^ 16)

/-- **`gemm-hopper-step(arch, bf16)`**: the device's BF16 k16 accumulator step is `tc_dot_total` on the Hopper pipeline
(`HOPPER_BF16_WGMMA_K16`: one group of sixteen products, a 26-bit truncating adder, exponent floor `-133`).

`step acc a b` is the FP32 word the device writes for one accumulator coordinate, from the FP32 word `acc` and sixteen
BF16 words of each operand: `wgmma.mma_async...m64nNk16.f32.bf16.bf16` on the H100 (sm_90), and `mma.sync.m16n8k16`
with BF16 operands on the H100 and the RTX 5090 (sm_120). Which device it is, is the moniker's `arch`; the evidence is
`models.HOPPER_BF16_WGMMA_K16.validated` and `models.HOPPER_BF16_M16N8K16.validated`. -/
def GemmHopperStep (step : Nat → List Nat → List Nat → Nat) : Prop :=
  ∀ acc a b, StepInputs HOPPER_BF16_WGMMA_K16 acc a b → step acc a b = Spec.tcDotTotal HOPPER_BF16_WGMMA_K16 acc a b

/-- **`gemm-ampere-step(arch, bf16)`**: the device's BF16 k16 accumulator step is `tc_dot_total` on the Ampere pipeline
(`AMPERE_BF16_M16N8K16`: two groups of eight products, a 25-bit truncating adder, exponent floor `-132`).

`step acc a b` is the FP32 word the device writes for one accumulator coordinate of `mma.sync.m16n8k16` with BF16
operands, on the A100 (sm_80) and the RTX 4090 (sm_89, whose `ADA_BF16_M16N8K16` has the same parameters). Which device it
is, is the moniker's `arch`; the evidence is `models.AMPERE_BF16_M16N8K16.validated`, `models.ADA_BF16_M16N8K16.validated`
and the special-value campaign in `total.py`. -/
def GemmAmpereStep (step : Nat → List Nat → List Nat → Nat) : Prop :=
  ∀ acc a b, StepInputs AMPERE_BF16_M16N8K16 acc a b → step acc a b = Spec.tcDotTotal AMPERE_BF16_M16N8K16 acc a b

/-- **`cr/sha-512`, as the advice root's binding**: a private partition's advice commits to at most one partition. The
root is SHA-512's (the one-hash rule), so two partitions with one root would be a collision. -/
def AdviceBinding {R : Type} (C : Circuit) (advice : (n : ℕ) → Partition C n → R) : Prop :=
  Function.Injective fun x : (n : ℕ) × Partition C n => advice x.1 x.2

end Verity.Assumptions
