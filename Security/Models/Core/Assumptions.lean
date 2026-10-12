import Definitions.Core.TC.Spec

/-!
# Named assumptions

The hardware semantics and the deployment Verity's theorems take without proof, each a `Prop` taken as a hypothesis (no
`axiom`). The moniker in each docstring is the entry of the overnight assumptions table that the red team rates; a
deployment assumption's moniker is its `verity.claims` id. An area's `Models/<Area>/Assumptions` cites these rather than
restating them.
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

/-- **`trusted-nebius-host`**: the auditor-trusted `device` (a network certifier, the memory challenger) runs its stated
`code` on the Nebius host it is deployed to. `run device x` is everything the device does on input `x`: what it emits,
and how it uses its clock and its keys. The host changes none of it. Nothing attests this, since no attestation hardware
stands behind the device: it is trust in Nebius. It doesn't say the keys stay secret: a protocol that relies on
the device's signatures proves its claim or finds, from the run, a forgery under the device's key. -/
def TrustedNebiusHost {Device In Out : Type} (device : Device) (run : Device → In → Out) (code : In → Out) : Prop :=
  ∀ x, run device x = code x

end Verity.Assumptions
