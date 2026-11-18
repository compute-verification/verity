module

public import Definitions.Core.Draw

/-!
# A scheme: what a resource accounting protocol supplies, and what it gets back

Every resource accounting protocol states its guarantee through these, and defines none of its own:

* `Scheme`: what a protocol supplies to the proof layers. Its blocks, the units the audit samples; each block's
  cost, the resource credited for it; its other charges, the resource it can't tie to a block; the runs the verifier
  accepts, with the resource each used and the resource each was credited; the blocks of a run that are incorrect; the
  event that breaks one of its own assumptions, with that event's probability bound; and its charge inequality: in an
  accepted run that breaks nothing, the uncredited resource is at most the cost of the incorrect blocks plus the other
  charges.
* `Scheme.exposure`: the largest cost of a block set a sample misses with probability above a cutoff.
* `Scheme.GetsBack`: what a scheme gets back from the audit, a named `Prop`. At cutoff `δ`, an accepted run's
  uncredited resource exceeds the exposure plus the other charges with probability at most `δ` plus the scheme's own
  term.

A scheme names no level, no commitment and no duration: the audit decides which values its blocks open, and the
probability `pr` is the audit game's, over any prover.
-/

@[expose] public section

namespace Verity

open scoped ENNReal

/-- **A scheme**: what a resource accounting protocol supplies to the proof layers. -/
structure Scheme where
  /-- The blocks: the units the audit samples. -/
  Block : Type
  [fintype : Fintype Block]
  /-- The resource credited for a block. -/
  cost : Block → ℕ
  /-- The other charges: resource the scheme can't tie to a block. -/
  other : ℕ
  /-- A run of the audit, as far as the scheme reads it. -/
  Run : Type
  /-- The verifier accepts the run. -/
  accepts : Run → Prop
  /-- The resource the run used. -/
  used : Run → ℕ
  /-- The resource the run was credited. -/
  credited : Run → ℕ
  /-- The run's incorrect blocks. -/
  incorrect : Run → Finset Block
  /-- The run breaks one of the scheme's own assumptions (a key guessed, a hash collision). -/
  breaks : Run → Prop
  /-- The scheme's own term: the bound on the probability of `breaks`. -/
  δscheme : ℝ≥0∞
  /-- The charge inequality: in an accepted run that breaks nothing, the uncredited resource is at most the cost of
  the incorrect blocks plus the other charges. -/
  charge : ∀ r, accepts r → ¬ breaks r → used r ≤ credited r + ∑ b ∈ incorrect r, cost b + other

attribute [instance] Scheme.fintype

namespace Scheme

open Classical in
/-- The exposure at cutoff `δ`: the largest cost of a block set the sample misses with probability above `δ`. -/
noncomputable def exposure (S : Scheme) (L : SamplingDesign S.Block) (δ : ℝ≥0∞) : ℕ :=
  (Finset.univ.filter fun T : Finset S.Block => δ < L.missProb T).sup fun T => ∑ b ∈ T, S.cost b

/-- **What a scheme gets back**: under the sampling design `L` and the audit game's probability `pr`, at cutoff `δ`
an accepted run's uncredited resource exceeds the exposure plus the other charges with probability at most
`δ + δscheme`. -/
def GetsBack (S : Scheme) (L : SamplingDesign S.Block) (pr : (S.Run → Prop) → ℝ≥0∞) (δ : ℝ≥0∞) : Prop :=
  pr (fun r => S.accepts r ∧ S.credited r + S.exposure L δ + S.other < S.used r) ≤ δ + S.δscheme

end Scheme

end Verity
