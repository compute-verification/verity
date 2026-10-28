import Definitions.Nci.Capability.Products

/-!
# Local bounds: consumers and their dependencies

A **consumer** is anything a block finishes by computing some set of gates whole, and that depends on an upstream
computation of its own: a use of the counted products (one token's input to one layer), an output that accumulates many
products (a weight's gradient summed over tokens), and so on. Each has an upstream **region** that holds no registered
gate, and **required gates** in it.

The **upstream certificate** (`Upstream`), a property of the wiring checkable on a named circuit, says that every
required gate reaches, inside the region, a gate the consumer's finishing set reads. So a block that finishes the
consumer either is handed a gate of its region (an **import**, spending fresh bits) or contains all of its required
gates (a **recomputation**, spending gates and, possibly, updated state). **Overlap weights** let regions share gates.
The same three statements bound every kind of consumer (`Nci.Guarantees.ImportOrRecompute`, `RecomputeByGates`,
`RecomputeByState`); a new kind of dependency is a new `Consumers` value, not a new argument.
-/

namespace Nci.Capability

open Verity Finset

/-- **Consumers** of the declaration `d`, indexed by `κ`. Consumer `c` is finished by computing one of the gate sets
`fins c` whole; its upstream region `A c` holds no registered gate, and `G c ⊆ A c` are the gates a block must compute to
finish `c` without importing from `A c`. -/
structure Consumers (d : Registered) (κ : Type) where
  /-- The ways to finish each consumer: a block finishes `c` when it computes one of these sets whole. -/
  fins : κ → Finset (Finset (Fin d.D.N))
  /-- Each consumer's upstream region. -/
  A : κ → Finset (Fin d.D.N)
  /-- The gates each consumer requires. -/
  G : κ → Finset (Fin d.D.N)
  G_sub : ∀ c, G c ⊆ A c
  A_θ : ∀ c, Disjoint (A c) d.θ

namespace Consumers

variable {d : Registered} {κ : Type}

open Classical in
/-- **The consumers the gates `S` finish.** -/
noncomputable def servedIn [Fintype κ] (K : Consumers d κ) (S : Finset (Fin d.D.N)) : Finset κ :=
  univ.filter fun c => ∃ F ∈ K.fins c, F ⊆ S

/-- **The uses of counted products as consumers**: use `j` is finished by computing one of its counted products whole,
with the upstream region `A j` and required gates `G j` (one token's previous layer, say, and its products). -/
def ofUses (B : Nci.Covering.MulBlocks d) (A G : Fin B.nt → Finset (Fin d.D.N)) (hG : ∀ j, G j ⊆ A j)
    (hA : ∀ j, Disjoint (A j) d.θ) : Consumers d (Fin B.nt) where
  fins j := (B.P.filter fun q => q.2 = j).image B.m
  A := A
  G := G
  G_sub := hG
  A_θ := hA

/-- **Outputs as consumers**: output `o` is finished by computing its gate `out o`, with the upstream region `A o` and
required gates `G o`: for an output that accumulates a sum (a weight's gradient over many tokens), its addends' products
and the partial sums between them. -/
def ofOutputs (out : κ → Fin d.D.N) (A G : κ → Finset (Fin d.D.N)) (hG : ∀ c, G c ⊆ A c)
    (hA : ∀ c, Disjoint (A c) d.θ) : Consumers d κ where
  fins c := {{out c}}
  A := A
  G := G
  G_sub := hG
  A_θ := hA

/-- **The upstream certificate**, a property of the wiring checkable on a named circuit: for every way to finish a
consumer, every required gate reaches, inside the consumer's region, a gate that finishing set reads. It is the analysed
implementation's; an algebraic rewrite needs its own. -/
def Upstream (K : Consumers d κ) : Prop :=
  ∀ c, ∀ F ∈ K.fins c, ∀ g ∈ K.G c, ∃ a ∈ d.D.into F ∩ K.A c, Nci.Covering.FeedsWithin d.D (K.A c) g a

/-- **Overlap weights** on the consumers: nonnegative, summing to at most 1 over the regions that hold any one gate.
Disjoint regions take weight 1 each; regions sharing gates take less. -/
def OverlapWeights [Fintype κ] (K : Consumers d κ) (ω : κ → ℝ) : Prop :=
  (∀ c, 0 ≤ ω c) ∧ ∀ x : Fin d.D.N, ∑ c ∈ univ.filter (fun c => x ∈ K.A c), ω c ≤ 1

/-- **Recomputing any consumer needs state priced above `h`**: each consumer's required gates hold, among themselves and
what they read, a selection of `W` priced above `h`. -/
def RequiresState (K : Consumers d κ) {ι : Type} (W : Items d.D ι) (h : ℝ) : Prop :=
  ∀ c, ∃ (U : Finset ι) (σ : ι → Finset (Fin d.D.N)), Selects W U σ ∧ (∀ i ∈ U, σ i ⊆ holds d.D (K.G c)) ∧
    h < ∑ i ∈ U, W.price i

end Consumers

end Nci.Capability
