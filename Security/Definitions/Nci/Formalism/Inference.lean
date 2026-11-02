import Mathlib.Basic.ENNReal.Inv
import Definitions.Nci.Formalism.Circuits

/-!
# The training formalism v1, §2: inference

* The **policy** is three numbers: `c`, the most gates a block holds; `k`, the most bits a block reads beyond `θ`; `n`,
  the blocks allowed per period. A statement takes them as `c k n : ℕ`.
* A **`(c, k)`-partition** (`IsCKPartition`) of a declaration `d` is a `Verity.Partition` of its gates into blocks `I`
  with `|I| ≤ c` and `|in(I) \ θ| ≤ k` (`Verity.Declaration.unregistered`). A partition into `B` units may leave some
  empty, which only raises `B`.
* The **slowdown** `ρ(C) = min_𝓘 |𝓘| c / |C|` (`slowdown d c k`) is an infimum in `ℝ≥0∞` over the `(c, k)`-partitions,
  so it is `⊤` when none exists.
* A block's inputs split three ways: registered values `in(I) ∩ θ` (`registeredReads`), which are free; other inputs
  `in(I) ∩ (in(C) \ θ)` (`dataReads`), which are data; computed values `in(I) \ in(C)` (`computedReads`), which come from
  other blocks. `k` bounds the last two together, `in(I) \ θ` (`Nci.Formalism.reads_split`).
* Fit's blocks: a **session block** (`SessionBlocks`) holds `c` gates and reads, beyond `θ`, a description of the
  update and the data of `s` tokens; a **tree block** (`TreeBlocks`) also reads the accumulated state of two children in
  a reduction tree. `ServingFits` and `TreeFits` are Fit's conditions on what they read.
-/

namespace Nci.Formalism

open Finset
open Verity (Declaration Partition)
open scoped ENNReal

/-- **A `(c, k)`-partition** of `d`: every block has at most `c` gates and reads at most `k` values outside `θ`. -/
def IsCKPartition (d : Declaration) {B : ℕ} (P : Partition d.D B) (c k : ℕ) : Prop :=
  (∀ u, (P.gates u).card ≤ c) ∧ ∀ u, (d.unregistered (P.gates u)).card ≤ k

/-- **The slowdown** `ρ = min_𝓘 |𝓘| c / |C|` of `d` over its `(c, k)`-partitions `𝓘`, and `⊤` when there is none. -/
noncomputable def slowdown (d : Declaration) (c k : ℕ) : ℝ≥0∞ :=
  ⨅ (B : ℕ) (P : Partition d.D B) (_ : IsCKPartition d P c k), ((B * c : ℕ) : ℝ≥0∞) / (d.D.computed.card : ℝ≥0∞)

/-- `in(C)`: the inputs some gate of `d` reads. -/
def inputs (d : Declaration) : Finset (Fin d.D.N) := d.D.into d.D.computed

/-- `in(I) ∩ θ`: the registered values `I` reads, which are free. -/
def registeredReads (d : Declaration) (I : Finset (Fin d.D.N)) : Finset (Fin d.D.N) := d.D.into I ∩ d.θ

/-- `in(I) ∩ (in(C) \ θ)`: the other inputs `I` reads, its data. -/
def dataReads (d : Declaration) (I : Finset (Fin d.D.N)) : Finset (Fin d.D.N) := d.D.into I ∩ (inputs d \ d.θ)

/-- `in(I) \ in(C)`: the computed values `I` reads, which come from other blocks. -/
def computedReads (d : Declaration) (I : Finset (Fin d.D.N)) : Finset (Fin d.D.N) := d.D.into I \ inputs d

/-- **Session blocks**: every block of `P` holds `c` gates, and reads beyond `θ` only a description of the update
(`desc`, at most `Δ` values) and the data of at most `s` tokens (`data t`, at most `τ` values each). -/
def SessionBlocks (d : Declaration) {B : ℕ} (P : Partition d.D B) (c Δ s τ : ℕ) : Prop :=
  ∀ u, (P.gates u).card = c ∧ ∃ (desc : Finset (Fin d.D.N)) (data : Fin s → Finset (Fin d.D.N)),
    desc.card ≤ Δ ∧ (∀ t, (data t).card ≤ τ) ∧ d.unregistered (P.gates u) ⊆ desc ∪ univ.biUnion data

/-- **Tree blocks**, a training step reduced by a tree: every block of `P` holds `c` gates, and reads beyond `θ` only a
description of the update (at most `Δ` values), the accumulated gradient and optimizer state of at most two children
(`state i`, at most `Δgrad` values each) and the data of at most `s` tokens (at most `τ` values each). A session block is
a tree block whose children's states are empty. -/
def TreeBlocks (d : Declaration) {B : ℕ} (P : Partition d.D B) (c Δ Δgrad s τ : ℕ) : Prop :=
  ∀ u, (P.gates u).card = c ∧ ∃ (desc : Finset (Fin d.D.N)) (state : Fin 2 → Finset (Fin d.D.N))
    (data : Fin s → Finset (Fin d.D.N)), desc.card ≤ Δ ∧ (∀ i, (state i).card ≤ Δgrad) ∧ (∀ t, (data t).card ≤ τ) ∧
    d.unregistered (P.gates u) ⊆ desc ∪ univ.biUnion state ∪ univ.biUnion data

/-- **A session block fits**: `Δ + s τ ≤ k`. -/
def ServingFits (Δ s τ k : ℕ) : Prop := Δ + s * τ ≤ k

/-- **A tree block fits**: `Δ + 2 Δ_grad + s τ ≤ k`. -/
def TreeFits (Δ Δgrad s τ k : ℕ) : Prop := Δ + 2 * Δgrad + s * τ ≤ k

end Nci.Formalism
