import Mathlib.Basic.ENNReal.Inv
import Mathlib.Data.Fintype.Powerset
import Definitions.Nci.Formalism.Circuits

/-!
# The training formalism v2, §2: inference

* The **policy** is `θ ⊆ in(C)`, a `Verity.Declaration`'s registered inputs, and three numbers: `c`, the most gates an
  inference block holds; `k`, the most values it reads outside `θ`; `n`, the inference blocks allowed per period. A
  statement takes them as `c k n : ℕ`.
* An **inference block** (`InferenceBlock d c k I`) is a set of gates `I` with `|I| ≤ c` and `|in(I) \ θ| ≤ k`
  (`Verity.Declaration.unregistered`). A partition `𝓘` of `C` into inference blocks is core's `(c, k)`-partition
  (`Verity.Partition.IsCK`), its blocks the gates of its units. A partition into `B` units may leave some empty, which only
  raises `|𝓘| = B`. `Verity.Partition.IsCK` also reads that every value is a bit, which every circuit of §1 has.
* The **slowdown** `ρ = min_𝓘 |𝓘| c / |C|` (`slowdown d c k`) is an infimum in `ℝ≥0∞` over those partitions of
  `|𝓘| c / |C|` (`partitionSlowdown`), so it is `⊤` when none exists. A lower bound `X ≤ ρ` is the same as
  `X ≤ |𝓘| c / |C|` for each of them (`Nci.Formalism.le_slowdown_iff`), and the theorems state it so.
* `max_I |I ∩ S|` over the inference blocks `I` is `maxInBlock d c k S`.
* A block's inputs split three ways: registered values `in(I) ∩ θ` (`registeredReads`), which are free; other inputs
  `in(I) ∩ (in(C) \ θ)` (`dataReads`), which are data; computed values `in(I) \ in(C)` (`computedReads`), which come from
  other blocks. `k` bounds the last two together, `in(I) \ θ` (`Nci.Formalism.reads_split`).
* Fit's blocks: a **compute block** (`ComputeBlock`) reads beyond `θ` a description of the update and the data of `s`
  tokens; a **reduction block** (`ReductionBlock`) the gradient and optimizer state of two children in a reduction tree;
  an **update application block** (`UpdateBlock`) a description of the update and the reduced state. A partition into
  **session blocks** (`SessionBlocks`) has only compute blocks, one into **training blocks** (`TrainingBlocks`) any of
  the three, and each block holds between `c - d` and `c` gates of the computation, `d` the decoder's. `ServingFits` and
  `TrainingFits` are Fit's conditions on what they read.
-/

namespace Nci.Formalism

open Finset
open Verity (Declaration Partition)
open scoped ENNReal

/-- **An inference block** of `d`: a set of gates `I` with `|I| ≤ c` and `|in(I) \ θ| ≤ k`. -/
def InferenceBlock (d : Declaration) (c k : ℕ) (I : Finset (Fin d.D.N)) : Prop :=
  GateSet d.D I ∧ I.card ≤ c ∧ (d.unregistered I).card ≤ k

instance instDecidablePredFinsetFinNDInferenceBlock (d : Declaration) (c k : ℕ) :
    DecidablePred (InferenceBlock d c k) := fun I =>
  inferInstanceAs (Decidable (I ⊆ d.D.computed ∧ I.card ≤ c ∧ (d.unregistered I).card ≤ k))

/-- `|𝓘| c / |C|`: the gates of a partition into `B` blocks of at most `c` gates, priced at `c` a block, against `|C|`. -/
noncomputable def partitionSlowdown (d : Declaration) (B c : ℕ) : ℝ≥0∞ :=
  ((B * c : ℕ) : ℝ≥0∞) / (d.D.computed.card : ℝ≥0∞)

/-- **The slowdown** `ρ = min_𝓘 |𝓘| c / |C|` of `d` over its partitions `𝓘` into inference blocks, and `⊤` when there
is none. -/
noncomputable def slowdown (d : Declaration) (c k : ℕ) : ℝ≥0∞ :=
  ⨅ (B : ℕ) (P : Partition d.D B) (_ : P.IsCK d c k), partitionSlowdown d B c

/-- `max_I |I ∩ S|`: the most gates of `S` an inference block holds. -/
def maxInBlock (d : Declaration) (c k : ℕ) (S : Finset (Fin d.D.N)) : ℕ :=
  (univ.filter (InferenceBlock d c k)).sup fun I => (I ∩ S).card

/-- `in(I) ∩ θ`: the registered values `I` reads, which are free. -/
def registeredReads (d : Declaration) (I : Finset (Fin d.D.N)) : Finset (Fin d.D.N) := d.D.into I ∩ d.θ

/-- `in(I) ∩ (in(C) \ θ)`: the other inputs `I` reads, its data. -/
def dataReads (d : Declaration) (I : Finset (Fin d.D.N)) : Finset (Fin d.D.N) := d.D.into I ∩ (inputs d.D \ d.θ)

/-- `in(I) \ in(C)`: the computed values `I` reads, which come from other blocks. -/
def computedReads (d : Declaration) (I : Finset (Fin d.D.N)) : Finset (Fin d.D.N) := d.D.into I \ inputs d.D

/-- **A compute block**: beyond `θ`, `I` reads only a description of the update (`desc`, at most `Δ` values) and the
data of at most `s` tokens (`data t`, at most `τ` values each). -/
def ComputeBlock (d : Declaration) (I : Finset (Fin d.D.N)) (Δ s τ : ℕ) : Prop :=
  ∃ (desc : Finset (Fin d.D.N)) (data : Fin s → Finset (Fin d.D.N)),
    desc.card ≤ Δ ∧ (∀ t, (data t).card ≤ τ) ∧ d.unregistered I ⊆ desc ∪ univ.biUnion data

/-- **A reduction block**: beyond `θ`, `I` reads only the accumulated gradient and optimizer state of at most two
children in a reduction tree (`state i`, at most `Δgrad` values each). -/
def ReductionBlock (d : Declaration) (I : Finset (Fin d.D.N)) (Δgrad : ℕ) : Prop :=
  ∃ state : Fin 2 → Finset (Fin d.D.N), (∀ i, (state i).card ≤ Δgrad) ∧ d.unregistered I ⊆ univ.biUnion state

/-- **An update application block**: beyond `θ`, `I` reads only a description of the update (`desc`, at most `Δ`
values) and the reduced gradient and optimizer state (`state`, at most `Δgrad` values). -/
def UpdateBlock (d : Declaration) (I : Finset (Fin d.D.N)) (Δ Δgrad : ℕ) : Prop :=
  ∃ desc state : Finset (Fin d.D.N), desc.card ≤ Δ ∧ state.card ≤ Δgrad ∧ d.unregistered I ⊆ desc ∪ state

/-- **Session blocks**, beside a decoder of `dg` gates: every block of `P` holds between `c - dg` and `c` gates and is a
compute block, reading beyond `θ` a description of at most `Δ` values and the data of at most `s` tokens. -/
def SessionBlocks (d : Declaration) {B : ℕ} (P : Partition d.D B) (c dg Δ s τ : ℕ) : Prop :=
  ∀ u, c - dg ≤ (P.gates u).card ∧ (P.gates u).card ≤ c ∧ ComputeBlock d (P.gates u) Δ s τ

/-- **Training blocks**, beside a decoder of `dg` gates: every block of `P` holds between `c - dg` and `c` gates and is a
compute block, a reduction block or an update application block. -/
def TrainingBlocks (d : Declaration) {B : ℕ} (P : Partition d.D B) (c dg Δ Δgrad s τ : ℕ) : Prop :=
  ∀ u, c - dg ≤ (P.gates u).card ∧ (P.gates u).card ≤ c ∧
    (ComputeBlock d (P.gates u) Δ s τ ∨ ReductionBlock d (P.gates u) Δgrad ∨ UpdateBlock d (P.gates u) Δ Δgrad)

/-- **A compute block fits**: `Δ + s τ ≤ k`. -/
def ServingFits (Δ s τ k : ℕ) : Prop := Δ + s * τ ≤ k

/-- **Each kind of training block fits**: a compute block, `Δ + s τ ≤ k`; a reduction block, `2 Δ_grad ≤ k`; an update
application block, `Δ + Δ_grad ≤ k`. -/
def TrainingFits (Δ Δgrad s τ k : ℕ) : Prop := Δ + s * τ ≤ k ∧ 2 * Δgrad ≤ k ∧ Δ + Δgrad ≤ k

end Nci.Formalism
