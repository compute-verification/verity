module

public import Mathlib.Basic.ENNReal.Inv
public import Mathlib.Data.Fintype.Powerset
public import Definitions.Slowdown.Formalism.Circuits

/-! Section 2 of the training formalism v2, inference: the policy, inference blocks, the slowdown `ρ`, kinds of work and their bounds, and Fit's kinds of block. -/

@[expose] public section

namespace Slowdown.Formalism

open Finset
open Verity (RegisteredCircuit Partition)
open scoped NNReal ENNReal

/-- **An inference block** of `d`: a set of gates `I` with `|I| ≤ c` and `|in(I) \ θ| ≤ k`. -/
def InferenceBlock (d : RegisteredCircuit) (c k : ℕ) (I : Finset (Fin d.D.N)) : Prop :=
  GateSet d.D I ∧ I.card ≤ c ∧ (d.unregistered I).card ≤ k

instance instDecidablePredFinsetFinNDInferenceBlock (d : RegisteredCircuit) (c k : ℕ) :
    DecidablePred (InferenceBlock d c k) := fun I =>
  inferInstanceAs (Decidable (I ⊆ d.D.computed ∧ I.card ≤ c ∧ (d.unregistered I).card ≤ k))

/-- `|𝓘| c / |C|`: the gates of a partition into `B` blocks of at most `c` gates, priced at `c` a block, against `|C|`. -/
noncomputable def partitionSlowdown (d : RegisteredCircuit) (B c : ℕ) : ℝ≥0∞ :=
  ((B * c : ℕ) : ℝ≥0∞) / (d.D.computed.card : ℝ≥0∞)

/-- **The slowdown** `ρ = min_𝓘 |𝓘| c / |C|` of `d` over its partitions `𝓘` into inference blocks, and `⊤` when there
is none. -/
noncomputable def slowdown (d : RegisteredCircuit) (c k : ℕ) : ℝ≥0∞ :=
  ⨅ (B : ℕ) (P : Partition d.D B) (_ : P.IsCK d c k), partitionSlowdown d B c

/-- **The caps of kinds of work** `K`: each kind's work `B_K` is a set of gates of `C`, and every inference block holds
at most `cap_K` of its gates. -/
def Caps (d : RegisteredCircuit) (c k : ℕ) {κ : Type} (B : κ → Finset (Fin d.D.N)) (cap : κ → ℝ≥0) : Prop :=
  ∀ K, GateSet d.D (B K) ∧ ∀ I, InferenceBlock d c k I → ((I ∩ B K).card : ℝ≥0) ≤ cap K

/-- **A kind's bound** `(|S| / |C|) / (cap / c)`: the blocks that hold at most `cap` gates of `S` each need for `S`,
priced at `c` a block, against `|C|`. -/
noncomputable def kindBound (d : RegisteredCircuit) (c : ℕ) (S : Finset (Fin d.D.N)) (cap : ℝ≥0) : ℝ≥0∞ :=
  ((S.card : ℝ≥0∞) / (d.D.computed.card : ℝ≥0∞)) / ((cap : ℝ≥0∞) / (c : ℝ≥0∞))

/-- **A partition-level bound** `Λ`: every partition `𝓘` of `d` into inference blocks has `|𝓘| c / |C| ≥ Λ`, the form
the bound of a kind whose work is amortized over the partition takes. -/
def PartitionBound (d : RegisteredCircuit) (c k : ℕ) (Λ : ℝ≥0∞) : Prop :=
  ∀ (m : ℕ) (P : Partition d.D m), P.IsCK d c k → Λ ≤ partitionSlowdown d m c

/-- `in(I) ∩ θ`: the registered values `I` reads, which are free. -/
def registeredReads (d : RegisteredCircuit) (I : Finset (Fin d.D.N)) : Finset (Fin d.D.N) := d.D.into I ∩ d.θ

/-- `in(I) ∩ (in(C) \ θ)`: the other inputs `I` reads, its data. -/
def dataReads (d : RegisteredCircuit) (I : Finset (Fin d.D.N)) : Finset (Fin d.D.N) := d.D.into I ∩ (inputs d.D \ d.θ)

/-- `in(I) \ in(C)`: the computed values `I` reads, which come from other blocks. -/
def computedReads (d : RegisteredCircuit) (I : Finset (Fin d.D.N)) : Finset (Fin d.D.N) := d.D.into I \ inputs d.D

/-- **A compute block**: beyond `θ`, `I` reads only a description of the update (`desc`, at most `Δ` values) and the
data of at most `s` tokens (`data t`, at most `τ` values each). -/
def ComputeBlock (d : RegisteredCircuit) (I : Finset (Fin d.D.N)) (Δ s τ : ℕ) : Prop :=
  ∃ (desc : Finset (Fin d.D.N)) (data : Fin s → Finset (Fin d.D.N)),
    desc.card ≤ Δ ∧ (∀ t, (data t).card ≤ τ) ∧ d.unregistered I ⊆ desc ∪ univ.biUnion data

/-- **A reduction block**: beyond `θ`, `I` reads only the accumulated gradient and optimizer state of at most two
children in a reduction tree (`state i`, at most `Δgrad` values each). -/
def ReductionBlock (d : RegisteredCircuit) (I : Finset (Fin d.D.N)) (Δgrad : ℕ) : Prop :=
  ∃ state : Fin 2 → Finset (Fin d.D.N), (∀ i, (state i).card ≤ Δgrad) ∧ d.unregistered I ⊆ univ.biUnion state

/-- **An update application block**: beyond `θ`, `I` reads only the weights in the form the update rule reads and
writes (`weights`, at most `Δt` values) and the reduced gradient and optimizer state (`state`, at most `Δgrad`
values). -/
def UpdateBlock (d : RegisteredCircuit) (I : Finset (Fin d.D.N)) (Δt Δgrad : ℕ) : Prop :=
  ∃ weights state : Finset (Fin d.D.N), weights.card ≤ Δt ∧ state.card ≤ Δgrad ∧ d.unregistered I ⊆ weights ∪ state

/-- **Session blocks**, beside a decoder of `dg` gates: every block of `P` holds between `c - dg` and `c` gates and is a
compute block, reading beyond `θ` a description of at most `Δ` values and the data of at most `s` tokens. -/
def SessionBlocks (d : RegisteredCircuit) {B : ℕ} (P : Partition d.D B) (c dg Δ s τ : ℕ) : Prop :=
  ∀ u, c - dg ≤ (P.gates u).card ∧ (P.gates u).card ≤ c ∧ ComputeBlock d (P.gates u) Δ s τ

/-- **Training blocks**, beside a decoder of `dg` gates: every block of `P` holds between `c - dg` and `c` gates and is a
compute block, a reduction block or an update application block. -/
def TrainingBlocks (d : RegisteredCircuit) {B : ℕ} (P : Partition d.D B) (c dg Δ Δt Δgrad s τ : ℕ) : Prop :=
  ∀ u, c - dg ≤ (P.gates u).card ∧ (P.gates u).card ≤ c ∧
    (ComputeBlock d (P.gates u) Δ s τ ∨ ReductionBlock d (P.gates u) Δgrad ∨ UpdateBlock d (P.gates u) Δt Δgrad)

/-- **A compute block fits**: `Δ + s τ ≤ k`. -/
def ServingFits (Δ s τ k : ℕ) : Prop := Δ + s * τ ≤ k

/-- **Each kind of training block fits**: a compute block, `Δ + s τ ≤ k`; a reduction block, `2 Δ_grad ≤ k`; an update
application block, `Δ_t + Δ_grad ≤ k`. -/
def TrainingFits (Δ Δt Δgrad s τ k : ℕ) : Prop := Δ + s * τ ≤ k ∧ 2 * Δgrad ≤ k ∧ Δt + Δgrad ≤ k

end Slowdown.Formalism
