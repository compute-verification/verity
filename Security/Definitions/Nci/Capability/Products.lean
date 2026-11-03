import Definitions.Nci.Capability.Items
import Definitions.Nci.Covering

/-!
# Local bounds: counted products and their incidence

The counted computation is the covering's multiplication blocks (`Nci.Covering.MulBlocks`): product `(i, j)` multiplies
a version of the updated parameter slot `i` by an operand of the use `j` (one token's input to one layer at one training
step). A block **completes** a product when it computes all of its gates, and **serves** the uses of the products it
completes. The **incidence** says which slot and which operand occurrence each product reads, at the granularity a block
can select.
-/

namespace Nci.Capability

open Verity Finset

/-- **The counted products the gates `S` compute whole.** -/
def completedIn {d : Declaration} (B : Nci.Covering.MulBlocks d) (S : Finset (Fin d.D.N)) :
    Finset (Fin B.nw × Fin B.nt) :=
  B.P.filter fun q => B.m q ⊆ S

/-- **The uses the gates `S` serve**: those of a counted product they compute whole. -/
def served {d : Declaration} (B : Nci.Covering.MulBlocks d) (S : Finset (Fin d.D.N)) : Finset (Fin B.nt) :=
  (completedIn B S).image Prod.snd

/-- **Each counted product reads a cell of its slot**, among its gates and what they read. -/
def SlotsRead {d : Declaration} (B : Nci.Covering.MulBlocks d) (W : Items d.D (Fin B.nw)) : Prop :=
  ∀ q ∈ B.P, ∃ c ∈ W.cells q.1, c ⊆ holds d.D (B.m q)

/-- **The incidence** of the counted products with the slot items `W` and the operand items `O`: each product `q` reads
a cell of its slot and a cell of its operand occurrence `occ q`, one scalar of one token's activation, say: a block may
hold some of a token's activation without the rest. -/
structure Incidence {d : Declaration} (B : Nci.Covering.MulBlocks d) {κ : Type} (W : Items d.D (Fin B.nw))
    (O : Items d.D κ) where
  /-- The operand occurrence each product reads. -/
  occ : Fin B.nw × Fin B.nt → κ
  slots : SlotsRead B W
  operands : ∀ q ∈ B.P, ∃ c ∈ O.cells (occ q), c ⊆ holds d.D (B.m q)

end Nci.Capability
