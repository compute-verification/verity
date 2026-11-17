module

public import Definitions.Poub.Remote.Held

/-! What PoUB's held memory line assumes, each a named `Prop` on a trace: `Identity`, the token-bucket rate ceilings,
`ChipCeiling`, `Routed`, `Causal`, `SketchBound` and `LiveCoins`. -/

@[expose] public section

namespace ResourceAccounting.Remote.Held.Assumptions

open Finset ResourceAccounting.Remote.Held

variable {Ω : Type}

/-- **Identity** (statement C's links clause): on every outcome, no block comes into the node over its boundary
links. -/
def Identity (T : Trace Ω) : Prop := ∀ ω t, T.link ω t = 0

/-- **The links' ceiling**: on every outcome, over every interval `[u, v)`, the node's boundary links bring in at most
`β · (v − u) + B` blocks: `β` a tick and a burst of `B`. -/
def Links (T : Trace Ω) (β B : ℝ) : Prop := ∀ ω u v, u ≤ v → (over T.link ω u v : ℝ) ≤ β * (v - u : ℕ) + B

/-- **The squaring ceiling**: on every outcome, over every interval `[u, v)`, the node's compute re-encodes at most
`s / E · (v − u) + B` blocks: at most `s` squarings a tick at the codec's width, at least `E` a block, and a burst of
`B`, one maximal rebuild batch. -/
def SquaringCeiling (T : Trace Ω) (s E B : ℝ) : Prop :=
  ∀ ω u v, u ≤ v → (over T.rebuilt ω u v : ℝ) ≤ s / E * (v - u : ℕ) + B

/-- **The disks' ceiling**: on every outcome, over every interval `[u, v)`, the node's disks bring back at most
`β · (v − u) + B` blocks: `β` a tick and a burst of `B`, one maximal disk read. -/
def Disks (T : Trace Ω) (β B : ℝ) : Prop := ∀ ω u v, u ≤ v → (over T.disk ω u v : ℝ) ≤ β * (v - u : ℕ) + B

/-- **The chips' ceiling** (the Memory row): on every outcome, at most `S` registered blocks are on the node's chips at
any tick. -/
def ChipCeiling (T : Trace Ω) (S : ℕ) : Prop := ∀ ω t, (T.chip ω t).card ≤ S

/-- **Routed**: on every outcome, the blocks that come into node memory or onto a chip from neither at a tick are at
most those the tick's rebuilds, disk reads and links bring in. -/
def Routed (T : Trace Ω) : Prop := ∀ ω t, T.entered ω t ≤ T.rebuilt ω t + T.disk ω t + T.link ω t

/-- **Causal**: on every accepting outcome, the credited passes are in order, the first starting no sooner than `s₀`
(its seed chains from the first window's live coin, released at `s₀`) and the last root leaving by `e` (it arrived by
then). -/
def Causal (T : Trace Ω) (s₀ e : ℕ) : Prop :=
  ∀ ω, T.accepts ω → Monotone (T.tick ω) ∧ s₀ ≤ T.tick ω 0 ∧ T.tick ω T.passes ≤ e

/-- **Paced**: on every accepting outcome, the groups `g ≤ C` are each at least one pass, in order, within the
credited passes, their coins released in order with `e_g ≥ s_(g+1)`, and each pass of group `g` inside `[s_g, e_g]`
(its seed chains from group `g`'s live coin, and its root arrived by `e_g`). -/
def Paced (T : Trace Ω) (K : Pace Ω) : Prop :=
  ∀ ω, T.accepts ω →
    Monotone (T.tick ω) ∧ K.first ω 0 = 0 ∧ K.first ω (K.credited + 1) ≤ T.passes ∧
    (∀ g ≤ K.credited, K.first ω g < K.first ω (g + 1)) ∧ Monotone (K.released ω) ∧
    (∀ g ≤ K.credited, K.released ω (g + 1) ≤ K.ended ω g) ∧
    ∀ g ≤ K.credited, ∀ p, K.first ω g ≤ p → p < K.first ω (g + 1) →
      K.released ω g ≤ T.tick ω p ∧ T.tick ω (p + 1) ≤ K.ended ω g

/-- **Covers**: on every accepting outcome, each credited pass's plan holds every registered block (the paced
verifier refuses a pass that leaves one out). -/
def Covers (T : Trace Ω) : Prop := ∀ ω, T.accepts ω → ∀ p < T.passes, T.plan ω p = univ

variable [Fintype Ω]

/-- **The receipts' sketch bound**: except with probability `δ` when the audit accepts, every correct receipt of a
registered block in a credited pass was computed while the block was in node memory or on a chip during the pass. -/
def SketchBound (T : Trace Ω) (δ : ℝ) : Prop :=
  T.Holds δ fun ω => ∀ p < T.passes, ∀ j ∈ T.correct ω p,
    ∃ t, T.tick ω p ≤ t ∧ t < T.tick ω (p + 1) ∧ j ∈ T.present ω t

/-- **Live coins**: except with probability `δ` when the audit accepts, the credited passes' wrong receipts of
registered blocks are at most `W`, the openings' pooled bound. -/
def LiveCoins (T : Trace Ω) (W δ : ℝ) : Prop := T.Holds δ fun ω => (T.incorrect ω : ℝ) ≤ W

end ResourceAccounting.Remote.Held.Assumptions
