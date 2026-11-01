import Definitions.Core.Accel.Machine

/-!
# The accelerator model's named assumptions

The premises that carry a theorem over the machine 𝓜 (`Definitions/Core/Accel`) to a card: its MMAs' product form (P2,
`AccelProductForm`) and the card premise (`CardWithinAccel`). They sit apart from `Assumptions` so that importing
the model leaves the other named assumptions' statements as they elaborate without it.
-/

open scoped NNRat

namespace Verity.Assumptions

/-- **`accel-product-form(card)`** (P2): at the semantics `x`, every MMA opcode's dot, dense and 2:4 sparse, is a
function of its accumulator and its pairs' products (`Accel.Open.ProductForm`): each multiplier forms one product from
one operand pair (an element of a row of `A` and the same element of a column of `B`) and its output feeds that row's
and column's output only, as `Accel.mulOut` and nothing else of the pair. Checked per card by a pair-swap capture
(swap operand pairs whose `mulOut` agree and compare every output bit for bit). The exact `s8` dot has it
(`Proofs/Core/Accel/Model.lean`). -/
def AccelProductForm (x : Accel.Open) : Prop := x.ProductForm

/-- **`card-within-accel(card)`, the card premise**: the card does no more than the machine at its numbers.
`can f T D` says the card, running one code on its SMs (not the exclusions below) at a clock within the table's
ceiling, takes every content `M` of its stores with `D M` to `f M` within `T` seconds. Then for some semantics `x`
meeting P1 and P2 (the card's own: what its instructions compute), some program of the machine at `c` and `x`, with a
schedule chosen before the content as the card's code is, takes every such `M` to `f M` within `T` seconds
(`Machine.Fits`). It holds because every instruction the card executes is dominated by an ACCEL op (`Accel.Op`)
reading and writing at least its words, at no more units of each resource (the coverage table,
`benchmarks/pouw/machine_model/data/cover/rows.toml`), and the model's coarsenings (pooling, no latency, no per-clock
limit, free resources) only let a program do more. A table is a SKU's at its firmware's maximum (`Accel.H200`,
`Accel.Sm120`), so the premise at `c` says too that the cards are `c`'s SKU and firmware; nothing the protocol checks
attests that (open).

**Exclusions**, units no code here uses because they can't compute a committed word or a hash more cheaply than the
model prices: the video and JPEG engines, sm_120's RT cores, the texture path's filtering, the copy engines (moves at
device memory's and the host link's bandwidth) and compute data compression (the committed words are noised). The
node's host CPUs and links are other devices of the period (`Pouw.Dither.Others`), not units of the card.

Open: an exclusion is a second adversary class until it has a rate row or a proof that the model dominates it.
sm_120's RT cores (at most 58 integer products a SM-clock, 6% of its `s8` rate) and the texture path (at most 64
products a SM-clock, 6% on sm_120 and 1.7% on the H200) exceed the headline's served margins,
3.9% on sm_120 and 2.6% on the H200. -/
def CardWithinAccel (c : Accel.Card) (can : (c.Mem → c.Mem) → ℚ≥0 → (c.Mem → Prop) → Prop) : Prop :=
  ∃ x : Accel.Open, c.Exact x ∧ AccelProductForm x ∧
    ∀ f T D, can f T D → ∃ (P : (Accel.Machine.mk c x).Prog ℕ) (sched : List ℕ),
      ∀ (σ : (Accel.Machine.mk c x).State ℕ), D σ.mem → (∀ θ, σ.pc θ = 0) →
        Accel.Machine.Fits P σ sched T ∧ (Accel.Machine.run P σ sched).1.mem = f σ.mem

end Verity.Assumptions
