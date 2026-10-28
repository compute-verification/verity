import Definitions.Core.Accel.Card

/-!
# The accelerator model's named assumptions

The two premises that carry a theorem over the accelerator model (`Definitions/Core/Accel`) to a card: the open dots'
product form (`AccelProductForm`) and the refinement direction (`CardWithinAccel`). They sit apart from
`Assumptions` so that importing the model leaves the other named assumptions' statements, and the guarantees that read
them, as they elaborate without it.
-/

namespace Verity.Assumptions

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
`Accel.CapturedProductForm`, proved: `Accel.H200.capturedProductForm`, `Accel.Sm120.capturedProductForm`). The open
dots it covers, each still to capture: on the H200 FP8 at a nonzero accumulator, mixed E4M3 × E5M2 `wgmma`, f16, tf32
and b1; on the RTX PRO 6000 the E2M1 K32 atom (its value captures agree with `Sm120.f8Pipeline` on finite
accumulators and left NaN and infinite ones record-only), the block-scaled NVF4 and MXF4 (pinned in Python only),
mixed f8f6f4 pairings, f16, tf32 and b1. The 2:4 sparse MMAs' dots (`x.dotSp`, no capture on either card) have the
form over their `k/2` kept pairs: the metadata only routes each kept element of `A` to the element of `B` it names
(`Accel.mmaSpEval`), and each such pair meets one multiplier. -/
def AccelProductForm (x : Accel.Open) : Prop :=
  (∀ fa fb sat k, Accel.ProductForm fa fb k (x.dot fa fb sat k)) ∧
    ∀ fa fb sat k, Accel.ProductForm fa fb (k / 2) (x.dotSp fa fb sat k)

/-- **`card-within-accel(card)`**: the card can do no more than its accelerator model, the refinement direction (a
theorem about `c.machine` reaches the card only through it). `can f t D` says the card, running one code on its units
other than the exclusions below, takes every content `M` of its stores with `D M` to `f M` within `t` ticks of the
instance's clock, inside the scope limit below on each such `M`. Then the card's true open parameters are some `x`
meeting θ's floors (`c.floors`) with the product form (`AccelProductForm x`), and some program of `c.machine x`, under
some encoding `e` of the card's register words (`Accel.Card.decode`), takes `decode e M` to `decode e (f M)` on every
content `M` with `D M` within `t` ticks, meeting G1 and G2 (`Machine.Fits`). The program, its schedule and `e` are
chosen before the content, as the card's code is, so the model does the work on every content in `D`. The encoding reads every word outside the registers as it is. So a theorem that holds over the
instance at every `x` meeting the floors and the product form holds on the card for those words: the open
parameters are not a premise of their own, θ's floors and the product form are the whole of what is assumed about
them. With every register word `raw`, `decode` is the identity and this is a bit-for-bit refinement.

**Exclusions**, each a unit `can` doesn't use because it can't compute a committed word or a hash more cheaply than
the model prices (the reasons are rows of the coverage table, `Accel/README.md`): the video and JPEG engines
(fixed-function codecs on pixel formats); sm_120's RT cores (FP32 ray-box and ray-triangle tests); the texture path's
filtering (FP interpolation at 8-bit fractional weights); the copy engines (moves only, at device memory's and the
host link's bandwidth, which the model prices); and compute data compression (the committed words are noised
and incompressible; a method bound by device memory or L2 on compressible data of its own is outside the claim).

**Gaps**, units the card has that the instances don't have yet, so the claim is open for code that uses them until each
is added with its rate: `f16` accumulators, the 64-bit and `inc`/`dec` reductions, sm_120's block-scaled sparse MMA, and
NVSwitch `multimem` reductions, and the H200's `mma.sync` FP8. **Over-approximated units:** DMMA, the FP64 lanes,
`HFMA2`, the float reductions and `redux.sync` are in the model as exact integer versions at the unit's rate and in
its register layout, computing what the unit does on the integers it computes exactly (`Accel.Isa`'s docstring).
**Encoding**: the model's versions read two's-complement integers where
the card holds float codes, so the refinement is up to `decode`, which reads a register's float code of an integer
within its format's exact range as that integer. **Scope limit, per input**: `D` holds only
contents on which the code's run gives these units integer operands and results within their exact ranges, and starts
no float reduction into a nonzero float code in shared or device memory (which `decode` reads as it is). The card's
results on non-integers and its roundings past the exact range are outside the claim (the model reads an
out-of-range operand as zero), but only on the contents where they occur: a
code that leaves the range on some contents is inside the claim on the rest.

**Other devices of the period**: the node's host CPUs, with their AMX units, and its host
and peer links are not units of the card, and this Prop says nothing of them. They count in the period's capacity
`C` instead, at θ's measured rows: the host CPUs' SHA-512 at 4.90% (H200 node) and 5.07% (RTX PRO 6000 node) of the
card's measured hashing, and AMX's exact s8 MACs at 3.15% and 9.51% of its dense INT8; that this inventory is
complete is a premise of its own (A4, `PeriodInventory`). -/
def CardWithinAccel {ι : Type} (c : Accel.Card) (cap : ι → ℕ) (oracle : ℕ)
    (can : (c.Mem → c.Mem) → ℕ → (c.Mem → Prop) → Prop) : Prop :=
  ∃ x : Accel.Open, c.floors x ∧ AccelProductForm x ∧
    ∀ f t D, can f t D → ∃ (e : ℕ → Accel.Enc) (P : (c.machine x cap oracle).Prog ℕ) (sched : List (List ℕ)),
      sched.length ≤ t ∧ ∀ (M : c.Mem) (σ : (c.machine x cap oracle).State ℕ), D M → (∀ θ, σ.pc θ = 0) →
        σ.mem = c.decode e M →
          Accel.Machine.Fits P σ sched ∧ (Accel.Machine.run P σ sched).1.mem = c.decode e (f M)

end Verity.Assumptions
