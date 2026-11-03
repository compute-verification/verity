import Mathlib.Data.Fintype.Prod
import Definitions.Nci.Formalism.Inference

/-!
# Products of two fresh operands, a kind of training work

A matrix product `C = A · B` by the standard algorithm inside a Boolean declaration `d`, `A` of `ni × nl` entries and
`B` of `nl × nj`. Its products are the triples `(i, l, j)`; **MAC** `(i, l, j)` (`Standard.mac`) is the set of gates
that multiplies `A_il` by `B_lj` and adds the product into `C_ij`'s partial sum. In a training step's forward pass after
the first, both operands are fresh, the activations and the updated weights `θ'`, and each is passed at its width: `b_A`
bits an entry of `A`, `b_B` of `B`, `b_C` of `C`'s partial sums.

What the guarantees (`Nci.Products.Guarantees`) read of the product (`Standard`):

* `mac i l j`, the MAC's gates; `entry i j`, the gates of `C_ij`'s MACs; `out i j`, the gates of `C_ij`'s final value;
* `opA i l` and `opB l j`, the gates of the operands' bits;
* `coneA i l` and `coneB l j`, the operands' **cones**: the values whose reads pay for an operand, its own bits and
  those of its computation that no other operand's computation reads (the increment an updated weight adds, the entry of
  an earlier product an activation applies its function to).

Two premises, facts of the given circuit:

* **The standard algorithm** (`Standard.Algorithm`): the MACs are nonempty, pairwise disjoint and connected sets of
  computed gates (a set holding some of a MAC's gates but not all has a read across it); each MAC reads every bit of
  both its operands; the cones are pairwise disjoint and disjoint from the MACs; and each entry's final value lies in
  its MACs. A product by a fast algorithm (Strassen's) has no such MACs and stays outside.
* **Widths as passed** (`Standard.Passed`), at the policy's `(c, k)`: an inference block that holds or reads every bit
  of `A_il` reads at least `b_A` bits of `A_il`'s cone beyond `θ`, and of `B_lj`'s at least `b_B`; and an inference
  block that holds a whole MAC of `C_ij` hands at least `b_C` bits of `C_ij`'s gates to its gates outside the block, or
  holds that many of `C_ij`'s final bits. The first charges an operand through reads of its cone, the same whether the
  block reads the operand or computes it itself. An operand or a partial sum passed narrower stays outside, as does an
  operand some inference block gets for fewer reads of its cone than its width: one in `θ`, which costs nothing to read,
  or one computed from registered values.

`whole I` is the set of products whose MAC lies in `I`, `outBits` the bits of `C`'s final values `Σ_ij |out_ij|`, and
`handed C G I` the gates of `G` in `I` that a gate of `G` outside `I` reads. `capAt k b_A b_B b_C μ` is the cap a block
at the weight `μ > 0`, `max (k² / (16 μ b_A b_B b_C)) ((μ + 1) k)` products, and `bound` the products kind's bound on
`ρ`, `((ni nl nj - μ Σ_ij |out_ij|) / |C|) / (capAt / c)`.
-/

namespace Nci.Products

open Finset
open Verity (Circuit Declaration)
open Nci.Formalism
open scoped ENNReal

/-- `handed C G I`: the gates of `G` in `I` that a gate of `G` outside `I` reads, what `I` hands to the rest of `G`. -/
def handed (C : Circuit) (G I : Finset (Fin C.N)) : Finset (Fin C.N) :=
  (I ∩ G).filter fun h => ∃ g ∈ G, g ∉ I ∧ h ∈ C.reads g

/-- **A product `C = A · B` by the standard algorithm** in `d`, `A` of `ni × nl` entries and `B` of `nl × nj`: its MACs,
its operands, their cones and its entries' final values, as sets of `d`'s gates. -/
structure Standard (d : Declaration) (ni nl nj : ℕ) where
  /-- MAC `(i, l, j)`: the gates multiplying `A_il` by `B_lj` and adding the product into `C_ij`'s partial sum. -/
  mac : Fin ni → Fin nl → Fin nj → Finset (Fin d.D.N)
  /-- The gates of `A_il`'s bits. -/
  opA : Fin ni → Fin nl → Finset (Fin d.D.N)
  /-- The gates of `B_lj`'s bits. -/
  opB : Fin nl → Fin nj → Finset (Fin d.D.N)
  /-- `A_il`'s cone: the values whose reads pay for it. -/
  coneA : Fin ni → Fin nl → Finset (Fin d.D.N)
  /-- `B_lj`'s cone. -/
  coneB : Fin nl → Fin nj → Finset (Fin d.D.N)
  /-- The gates of `C_ij`'s final value. -/
  out : Fin ni → Fin nj → Finset (Fin d.D.N)

namespace Standard

variable {d : Declaration} {ni nl nj : ℕ} (S : Standard d ni nl nj)

/-- The gates of `C_ij`'s MACs. -/
def entry (i : Fin ni) (j : Fin nj) : Finset (Fin d.D.N) := univ.biUnion fun l => S.mac i l j

/-- The gates of all the MACs. -/
def macs : Finset (Fin d.D.N) := univ.biUnion fun p : Fin ni × Fin nl × Fin nj => S.mac p.1 p.2.1 p.2.2

/-- The gates of all the cones. -/
def cones : Finset (Fin d.D.N) :=
  (univ.biUnion fun p : Fin ni × Fin nl => S.coneA p.1 p.2) ∪ univ.biUnion fun p : Fin nl × Fin nj => S.coneB p.1 p.2

/-- The products whose MAC lies in `I`. -/
def whole (I : Finset (Fin d.D.N)) : Finset (Fin ni × Fin nl × Fin nj) :=
  univ.filter fun p => S.mac p.1 p.2.1 p.2.2 ⊆ I

/-- The bits of `C`'s final values, `Σ_ij |out_ij|`. -/
def outBits : ℕ := ∑ p : Fin ni × Fin nj, (S.out p.1 p.2).card

/-- **The standard algorithm**, a fact of the given circuit: the MACs are nonempty, pairwise disjoint and connected sets
of computed gates; each reads every bit of both its operands; the cones are pairwise disjoint and disjoint from the
MACs; and each entry's final value lies in its MACs. -/
structure Algorithm : Prop where
  mac_computed : ∀ i l j, S.mac i l j ⊆ d.D.computed
  mac_nonempty : ∀ i l j, (S.mac i l j).Nonempty
  mac_disjoint : ∀ p q : Fin ni × Fin nl × Fin nj, p ≠ q →
    Disjoint (S.mac p.1 p.2.1 p.2.2) (S.mac q.1 q.2.1 q.2.2)
  /-- A set holding some of a MAC's gates but not all has a read across it. -/
  mac_connected : ∀ i l j (X : Finset (Fin d.D.N)), X ⊆ S.mac i l j → X.Nonempty → X ≠ S.mac i l j →
    ∃ h ∈ S.mac i l j, ∃ g ∈ S.mac i l j, h ∈ d.D.reads g ∧ (h ∈ X ↔ g ∉ X)
  readsA : ∀ i l j, ∀ a ∈ S.opA i l, ∃ g ∈ S.mac i l j, a ∈ d.D.reads g
  readsB : ∀ i l j, ∀ b ∈ S.opB l j, ∃ g ∈ S.mac i l j, b ∈ d.D.reads g
  coneA_disjoint : ∀ p q : Fin ni × Fin nl, p ≠ q → Disjoint (S.coneA p.1 p.2) (S.coneA q.1 q.2)
  coneB_disjoint : ∀ p q : Fin nl × Fin nj, p ≠ q → Disjoint (S.coneB p.1 p.2) (S.coneB q.1 q.2)
  cones_disjoint : ∀ i l l' j, Disjoint (S.coneA i l) (S.coneB l' j)
  cones_macs : Disjoint S.cones S.macs
  out_entry : ∀ i j, S.out i j ⊆ S.entry i j

/-- **Widths as passed** at the policy's `(c, k)`, a fact of the given circuit: an inference block that holds or reads
every bit of `A_il` reads at least `b_A` bits of its cone beyond `θ`, and of `B_lj`'s at least `b_B`; and one that holds
a whole MAC of `C_ij` hands at least `b_C` bits of `C_ij`'s gates to its gates outside the block, or holds that many of
`C_ij`'s final bits. -/
structure Passed (c k bA bB bC : ℕ) : Prop where
  operandA : ∀ I, InferenceBlock d c k I → ∀ i l, S.opA i l ⊆ I ∪ d.D.into I →
    bA ≤ (d.unregistered I ∩ S.coneA i l).card
  operandB : ∀ I, InferenceBlock d c k I → ∀ l j, S.opB l j ⊆ I ∪ d.D.into I →
    bB ≤ (d.unregistered I ∩ S.coneB l j).card
  partialSum : ∀ I, InferenceBlock d c k I → ∀ i l j, S.mac i l j ⊆ I →
    bC ≤ (handed d.D (S.entry i j) I).card + (I ∩ S.out i j).card

end Standard

/-- **The cap a block** at the weight `μ > 0`: `max (k² / (16 μ b_A b_B b_C)) ((μ + 1) k)` products, amortized over a
partition into inference blocks with each split MAC charged one bit. -/
noncomputable def capAt (k bA bB bC : ℕ) (μ : ℝ) : ℝ :=
  max ((k : ℝ) ^ 2 / (16 * μ * bA * bB * bC)) ((μ + 1) * k)

/-- **The products kind's bound** at the weight `μ > 0`: `((ni nl nj - μ Σ_ij |out_ij|) / |C|) / (capAt / c)`, a
kind's bound `(|B_K| / |C|) / (cap_K / c)` (`Nci.Formalism.kindBound`) with the cap amortized, `μ Σ_ij |out_ij|` the
products it leaves uncharged. -/
noncomputable def Standard.bound {d : Declaration} {ni nl nj : ℕ} (S : Standard d ni nl nj) (c k bA bB bC : ℕ)
    (μ : ℝ) : ℝ≥0∞ :=
  ENNReal.ofReal ((ni * nl * nj : ℕ) - μ * S.outBits) / (d.D.computed.card : ℝ≥0∞) /
    (ENNReal.ofReal (capAt k bA bB bC μ) / (c : ℝ≥0∞))

end Nci.Products
