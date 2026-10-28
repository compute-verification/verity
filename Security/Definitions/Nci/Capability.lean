import Definitions.Core.Formalism
import Definitions.Nci.Covering

/-!
# The capability argument: the vocabulary

The objects the three scientific hypotheses (`Definitions.Nci.Assumptions.Capability`) and their consequences
(`Definitions.Nci.Guarantees.Capability`) are stated over, on core's circuit, registration and local description length
(`Definitions.Core.Formalism`) and the covering's multiplication blocks (`Nci.Covering.MulBlocks`):

* `UpdatedState`: the updated parameter slots' eligible versions and their prices;
* `HeldStateAtMost`: every proof block of a registration holds updated state priced at most a budget, Lemma 1's
  consequence under local update hardness (`Nci.Guarantees.UpdatedStatePerBlock`).
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

/-- **Each proof block holds updated state priced at most `kb`**: for every proof block `u` of the registration, every set
`U` of slots and every choice `σ` of eligible versions among the block's gates and what it reads, `Σ_{i ∈ U} p_i ≤ kb`. -/
def HeldStateAtMost (R : Registration) {B : Nci.Covering.MulBlocks R.d} (W : UpdatedState B) (kb : ℝ) : Prop :=
  ∀ (u : Fin R.B) (U : Finset (Fin B.nw)) (σ : Fin B.nw → Fin R.d.D.N),
    (∀ i ∈ U, σ i ∈ W.ver i ∧ σ i ∈ R.P.gates u ∪ R.d.D.into (R.P.gates u)) → ∑ i ∈ U, W.p i ≤ kb

end Nci.Capability
