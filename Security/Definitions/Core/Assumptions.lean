import Definitions.Core.TC.Spec
import Definitions.Core.Accel.Card

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

/-- **`trusted-host`**: the auditor-trusted `device` (a network certifier, the active warden, the memory challenger)
is the auditor's hardware on a machine its host provider runs and the prover doesn't control; it runs its stated
`code` on its host, and its host never restores an earlier copy of the state the device keeps there. `run device x`
is everything the device does on input `x`: what it emits, when it emits it, and how it uses its clock and its keys.
The host changes none of it. Two parts of that are easy to miss, and both are part of the assumption: the host
isolates the device's CPU from every other tenant, so no other workload's load changes when the device runs or makes
it miss a deadline its code meets; and it gives the device a monotonic clock, which never steps backwards. Nothing
attests this, since no attestation hardware stands behind the device: it is trust in the auditor's device and its
host provider, never in a host the prover controls. It doesn't say the keys stay secret: a protocol that relies on
the device's signatures proves its claim or finds, from the run, a forgery under the device's key.

This definition states only the part about one run. The state the device keeps on its host outlasts a run, which a
statement of one run can't name; a property that relies on it names it in its docstring and takes no premise for it. -/
def TrustedHost {Device In Out : Type} (device : Device) (run : Device → In → Out) (code : In → Out) : Prop :=
  ∀ x, run device x = code x

/-- **`accel-product-form(card)`**, per product term: in every tensor-core MMA of the card whose semantics no capture
pins (`x.dot`, the accelerator model's open dots), each multiplier forms one product from one operand pair (an element
of a row of `A` and the same element of a column of `B`), and its output feeds exactly one adder tree, that row's and
column's output, as `Accel.mulOut` (the operands' classes, the product's sign, the significands' product and the
exponent fields' sum) and nothing else of the pair. The tree reads only its multipliers' outputs, its accumulator and
its block scales. That is P2 at the hardware's level: the routing of pairs to one tree is the MMA's semantics
(`Accel.mmaEval`, `P2Local`, `P2Distinct`), and this Prop states the rest for the open dots (`Accel.ProductForm`).

Checked per card by a pair-swap capture: swap operand pairs whose `mulOut` agree, within one output and across
outputs, and compare every output bit for bit. The captured dots have the form by their captures (H200:
`HOPPER_SM90_S8_K32`, `HOPPER_BF16_M16N8K16`, the E4M3 and E5M2 K32 captures at a zero accumulator; RTX PRO 6000:
the `BLACKWELL_SM120_*_M16N8K32` integer captures, BF16, and the E4M3 and E5M2 K32 captures at every accumulator;
`Accel.CapturedProductForm`, to prove). The open dots it covers, each still to capture: on the H200 FP8 at a nonzero
accumulator, mixed E4M3 × E5M2 `wgmma`, f16, tf32 and b1; on the RTX PRO 6000 the E2M1 K32 atom (its value captures
agree with `Sm120.f8Pipeline` on finite accumulators and left NaN and infinite ones record-only), the block-scaled
NVF4 and MXF4 (pinned in Python only), mixed f8f6f4 pairings, f16, tf32 and b1. The 2:4 sparse MMAs'
dots (`x.dotSp`, no capture on either card) have the form over their `k/2` kept pairs: the metadata only routes each
kept element of `A` to the element of `B` it names (`Accel.mmaSpEval`), and each such pair meets one multiplier. -/
def AccelProductForm (x : Accel.Open) : Prop :=
  (∀ fa fb sat k, Accel.ProductForm fa fb k (x.dot fa fb sat k)) ∧
    ∀ fa fb sat k, Accel.ProductForm fa fb (k / 2) (x.dotSp fa fb sat k)

/-- **`card-within-accel(card)`**: the card can do no more than its accelerator model, the refinement direction (a
theorem about `c.machine` reaches the card only through it). `can f t` says the card, running any code on its units
other than the exclusions below, takes every content `M` of its stores to `f M` within `t` ticks of the instance's
clock. Then the card's true open parameters are some `x` meeting θ's floors (`c.floors`) with the product form
(`AccelProductForm x`), and some program of `c.machine x`, under some encoding `e` of the card's register words
(`Accel.Card.decode`), takes `decode e M` to `decode e (f M)` on every content `M` within `t` ticks, meeting G1 and
G2 (`Machine.Fits`). The encoding reads every word outside the registers as it is. So a theorem that holds over the
instance at every `x` meeting the floors and the product form holds on the card for those words: the open
parameters are not a premise of their own, θ's floors and the product form are the whole of what is assumed about
them. With every register word `raw`, `decode` is the identity and this is a bit-for-bit refinement.

**Exclusions**, each a unit `can` doesn't use because it can't compute a committed word or a hash more cheaply than
the model prices (the reasons are rows of the coverage table, `Accel/README.md`): the video and JPEG engines
(fixed-function codecs on pixel formats); sm_120's RT cores (FP32 ray-box and ray-triangle tests); the texture path's
filtering (FP interpolation at 8-bit fractional weights); the copy engines (moves only, at device memory's and the
host link's bandwidth, which the model prices); and compute data compression (U7: the committed words are noised
and incompressible; a method bound by device memory or L2 on compressible data of its own is outside the claim).

**Gaps**, units the card has that the instances don't have yet, so the claim is open for code that uses them until each
is added with its rate: `f16` accumulators, the 64-bit and `inc`/`dec` reductions, sm_120's block-scaled sparse MMA, and
NVSwitch `multimem` reductions, and the H200's `mma.sync` FP8. **Over-approximated units:** DMMA, the FP64 lanes,
`HFMA2`, the float reductions and `redux.sync` are in the model as exact integer versions at the unit's rate and in
its register layout, computing what the unit does on the integers it computes exactly (`Accel.Isa`'s docstring).
**Encoding** (RT-LEAN-2 L30; coordinator, 26 Oct 04:30Z): the model's versions read two's-complement integers where
the card holds float codes, so the refinement is up to `decode`, which reads a register's float code of an integer
within its format's exact range as that integer. Code that uses these units in range on registers is inside the
claim. **Scope limit:** the card's results on non-integers and its roundings past the exact range are outside it (the
model reads an out-of-range operand as zero; coordinator's ruling, 26 Oct 04:00Z), and so are float reductions whose
accumulator is a nonzero float code in shared or device memory, which `decode` reads as it is. The node's host CPUs and its host and peer
links are other devices of the period (RT-PROTO P9, RT-UNITS U9), not units of the card. -/
def CardWithinAccel {ι : Type} (c : Accel.Card) (cap : ι → ℕ) (oracle : ℕ) (can : (c.Mem → c.Mem) → ℕ → Prop) :
    Prop :=
  ∃ x : Accel.Open, c.floors x ∧ AccelProductForm x ∧
    ∀ f t, can f t → ∃ (e : ℕ → Accel.Enc) (P : (c.machine x cap oracle).Prog ℕ) (sched : List (List ℕ)),
      sched.length ≤ t ∧ ∀ (M : c.Mem) (σ : (c.machine x cap oracle).State ℕ), (∀ θ, σ.pc θ = 0) →
        σ.mem = c.decode e M →
          Accel.Machine.Fits P σ sched ∧ (Accel.Machine.run P σ sched).1.mem = c.decode e (f M)

end Verity.Assumptions
