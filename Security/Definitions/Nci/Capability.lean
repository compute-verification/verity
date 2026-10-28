import Definitions.Core.Formalism
import Definitions.Nci.Covering

/-!
# The capability argument: the vocabulary

The objects the local assumption sets and coverage (`Definitions.Nci.Assumptions.Capability`) and their consequences
(`Definitions.Nci.Guarantees.Capability`) are stated over, on core's circuit, registration and local description length
(`Definitions.Core.Formalism`) and the covering's multiplication blocks (`Nci.Covering.MulBlocks`). A counted product
`(i, j)` multiplies a version of the updated parameter slot `i` by the operand of the use `j`, one activation
occurrence (one coordinate of one token's input to one layer at one training step), computed inside the use's upstream
computation `T j`. Indexing products by the pair allows one product per slot and use.

* `UpdatedState`: the slots' eligible versions and their prices;
* `Operands`: the gates holding each use's operand;
* `HeldStateAtMost`, `HeldUsesAtMost`: what every proof block of a registration holds, the form the local description
  bounds give;
* `UpstreamCert`: Set 1's graph certificate, checkable on a named circuit;
* `completed`: the counted products a proof block computes whole.
-/

namespace Nci.Capability

open Verity Finset

/-- **Updated state** of the multiplication blocks `B`: for each updated parameter slot `i` the gates `ver i` holding its
eligible versions (a slot has one version per optimizer step that writes it), and its price `p i`, the bits a joint
description of it costs, at least 0. Each counted product reads a version of its slot, among its own gates or what it
reads. -/
structure UpdatedState {d : Declaration} (B : Nci.Covering.MulBlocks d) where
  /-- The gates holding slot `i`'s eligible versions. -/
  ver : Fin B.nw → Finset (Fin d.D.N)
  /-- Slot `i`'s price, in bits. -/
  p : Fin B.nw → ℝ
  p_nonneg : ∀ i, 0 ≤ p i
  /-- Each counted product reads a version of its slot. -/
  reads : ∀ q ∈ B.P, ∃ g ∈ ver q.1, g ∈ B.m q ∪ d.D.into (B.m q)

/-- **The operands** of the multiplication blocks `B`: for each use `j` the gates `act j` holding its operand, which
every counted product of `j` reads, among its own gates or what it reads. -/
structure Operands {d : Declaration} (B : Nci.Covering.MulBlocks d) where
  /-- The gates holding use `j`'s operand. -/
  act : Fin B.nt → Finset (Fin d.D.N)
  /-- Each counted product reads its use's operand. -/
  reads : ∀ q ∈ B.P, act q.2 ⊆ B.m q ∪ d.D.into (B.m q)

/-- **The counted products proof block `u` computes whole.** -/
def completed (R : Registration) (B : Nci.Covering.MulBlocks R.d) (u : Fin R.B) : Finset (Fin B.nw × Fin B.nt) :=
  B.P.filter fun q => B.m q ⊆ R.P.gates u

/-- **Each proof block holds updated state priced at most `kb`**: for every proof block `u` of the registration, every set
`U` of slots and every choice `σ` of eligible versions among the block's gates and what it reads, `Σ_{i ∈ U} p_i ≤ kb`. -/
def HeldStateAtMost (R : Registration) {B : Nci.Covering.MulBlocks R.d} (W : UpdatedState B) (kb : ℝ) : Prop :=
  ∀ (u : Fin R.B) (U : Finset (Fin B.nw)) (σ : Fin B.nw → Fin R.d.D.N),
    (∀ i ∈ U, σ i ∈ W.ver i ∧ σ i ∈ R.P.gates u ∪ R.d.D.into (R.P.gates u)) → ∑ i ∈ U, W.p i ≤ kb

/-- **Each proof block holds at most `n` operands**: for every proof block `u` and every set `V` of uses whose operands
lie among the block's gates and what it reads, `|V| ≤ n`. -/
def HeldUsesAtMost (R : Registration) {B : Nci.Covering.MulBlocks R.d} (O : Operands B) (n : ℝ) : Prop :=
  ∀ (u : Fin R.B) (V : Finset (Fin B.nt)),
    (∀ j ∈ V, O.act j ⊆ R.P.gates u ∪ R.d.D.into (R.P.gates u)) → (V.card : ℝ) ≤ n

/-- **Set 1's upstream certificate** at demand `dm`, a property of the wiring: the uses' upstream computations `T j`
are disjoint and hold no registered gate (`Nci.Covering.MulBlocks.T_disjoint`, `T_θ`), and each use `j` of a counted
product has upstream products `Up`, all of whose gates lie in `T j`, over at least `dm` distinct slots, every gate of
which reaches, inside `T j`, a gate that each counted product of `j` reads. A block that computes a counted product of
`j` without being handed a gate of `T j` therefore contains all of `Up`. It is the covering's form of
`Nci.Cover.MulBlocks.Upstream`. -/
def UpstreamCert {d : Declaration} (B : Nci.Covering.MulBlocks d) (dm : ℕ) : Prop :=
  ∀ q ∈ B.P, ∃ Up ⊆ B.P, (∀ q' ∈ Up, B.m q' ⊆ B.T q.2) ∧ dm ≤ (Up.image Prod.fst).card ∧
    ∀ q' ∈ Up, ∀ g ∈ B.m q', ∀ p ∈ B.P, p.2 = q.2 →
      ∃ a ∈ d.D.into (B.m p) ∩ B.T q.2, Nci.Covering.FeedsWithin d.D (B.T q.2) g a

end Nci.Capability
