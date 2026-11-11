module

public import Definitions.Nci.Capability

/-!
# Local bounds: caps from dependency

What a block finishes, and what that costs it. Every consumer (`Nci.Capability.Consumers`: a use of counted products, an
output accumulating many of them, and so on) a block finishes is either **imported** (it spends a fresh bit of the
consumer's region) or **recomputed** (the block contains the consumer's required gates):

* `ImportOrRecompute`: served weight ≤ fresh bits + recomputed weight;
* `RecomputeByGates`: recomputed weight ≤ gates / `g`, `g` the fewest required gates of a consumer;
* `RecomputeByState`: no recomputation when it needs more updated state than the budget.
-/

@[expose] public section

namespace Nci.Guarantees

open Verity Finset Nci.Capability

/-- **Import or recompute.** Under the upstream certificate and overlap weights `ω`, the weight of the consumers a proof
block finishes is at most its unregistered input bits plus the weight of the consumers whose required gates it contains:
each finished consumer either spends an imported gate of its region, a gate whose weight over all regions is at most 1, or
computes all of its required gates. -/
def ImportOrRecompute : Prop :=
  ∀ (R : Committed) {κ : Type} [Fintype κ] (K : Consumers R.d κ) (ω : κ → ℝ),
    K.Upstream → K.OverlapWeights ω →
    ∀ u, ∑ c ∈ K.servedIn (R.d.𝓘.gates u), ω c ≤
      (R.d.unregistered (R.d.𝓘.gates u)).card +
        ∑ c ∈ (K.servedIn (R.d.𝓘.gates u)).filter (fun c => K.G c ⊆ R.d.𝓘.gates u), ω c

/-- **Recomputing costs gates.** With overlap weights `ω` and at least `g` required gates per consumer, the weight of the
consumers whose required gates lie within any gates `S` is at most `|S| / g`: in a `(c, k)`-partition, `c / g` a block. -/
def RecomputeByGates : Prop :=
  ∀ (d : RegisteredCircuit) {κ : Type} [Fintype κ] (K : Consumers d κ) (ω : κ → ℝ) (g : ℝ), K.OverlapWeights ω →
    (∀ c, g ≤ (K.G c).card) → ∀ (J : Finset κ) (S : Finset (Fin d.D.N)),
      g * ∑ c ∈ J.filter (fun c => K.G c ⊆ S), ω c ≤ S.card

/-- **Recomputing costs updated state.** If every proof block holds `W`-state priced at most `h` and recomputing any
consumer needs more, no proof block contains a consumer's required gates: every finished consumer is imported. -/
def RecomputeByState : Prop :=
  ∀ (R : Committed) {ι : Type} (W : Items R.d.D ι) {κ : Type} (K : Consumers R.d κ) (h : ℝ),
    HeldPriceAtMost R W h → K.RequiresState W h → ∀ (u : Fin R.d.n) (c : κ), ¬ K.G c ⊆ R.d.𝓘.gates u

end Nci.Guarantees
