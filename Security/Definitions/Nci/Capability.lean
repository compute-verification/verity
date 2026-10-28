import Definitions.Core.Formalism
import Definitions.Nci.Covering

/-!
# Local bounds: the vocabulary

The objects every local bound is stated over, on core's circuit, registration and local description length
(`Definitions.Core.Formalism`) and the covering's multiplication blocks (`Nci.Covering.MulBlocks`, whose counted product
`(i, j)` pairs an updated parameter slot `i` with a use `j`, one token's input to one layer at one training step).

A block can **import** information (spending its `k` unregistered input bits) or **reconstruct** it (spending its `c`
gates). The vocabulary names what it holds, what its values cost, and the wiring that decides which route a use needs:

* information: `Items` (things a block may hold, each with eligible cells and a price), `Items.sum` (two families
  priced jointly), `Selects`, `holds`, `HeldPriceAtMost` (the budget form every information bound takes);
* products: `completedIn`, `served`, `SlotsRead`, `Incidence` (which slot and which operand each product reads);
* dependency: `UseRegions` (each use's upstream region and the gates it requires), `Upstream` (the certificate, checkable
  on a named circuit), `OverlapWeights` (how much regions may overlap), `RequiresState` (what recomputing a use costs in
  updated state).
-/

namespace Nci.Capability

open Verity Finset

/-- **Items** of the circuit `C`, indexed by `ι`: things whose values a block may hold. Item `i` has eligible **cells**
`cells i`, each the gates holding one version of its value (a parameter slot has one version per optimizer step that
writes it, each a multi-bit value; an activation occurrence has one), and a **price** `price i ≥ 0`, the bits its value
costs in a joint description. -/
structure Items (C : Circuit) (ι : Type) where
  /-- The eligible cells of each item. -/
  cells : ι → Finset (Finset (Fin C.N))
  /-- Each item's price, in bits. -/
  price : ι → ℝ
  price_nonneg : ∀ i, 0 ≤ price i

/-- **The joint family** of `X` and `Y`: their items side by side, each with its own cells and price. Pricing a selection
from both in one budget is what a joint premise asserts; it is not implied by two separate ones. -/
def Items.sum {C : Circuit} {ι κ : Type} (X : Items C ι) (Y : Items C κ) : Items C (ι ⊕ κ) where
  cells := Sum.elim X.cells Y.cells
  price := Sum.elim X.price Y.price
  price_nonneg := by
    intro i
    cases i with
    | inl i => exact X.price_nonneg i
    | inr i => exact Y.price_nonneg i

/-- **What the gates `S` hold**: their own gates and the gates they read, `S ∪ in(S)`. -/
def holds (C : Circuit) (S : Finset (Fin C.N)) : Finset (Fin C.N) := S ∪ C.into S

/-- **A selection** of the items `U` of `X`, with an eligible cell `σ i` for each. -/
def Selects {C : Circuit} {ι : Type} (X : Items C ι) (U : Finset ι) (σ : ι → Finset (Fin C.N)) : Prop :=
  ∀ i ∈ U, σ i ∈ X.cells i

/-- **Each proof block holds `X`-state priced at most `h`**: every selection whose cells lie among what a proof block of
the registration holds costs at most `h`. Every information bound reaches the counting in this form. -/
def HeldPriceAtMost (R : Committed) {ι : Type} (X : Items R.d.D ι) (h : ℝ) : Prop :=
  ∀ (u : Fin R.B) (U : Finset ι) (σ : ι → Finset (Fin R.d.D.N)), Selects X U σ →
    (∀ i ∈ U, σ i ⊆ holds R.d.D (R.P.gates u)) → ∑ i ∈ U, X.price i ≤ h

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
a cell of its slot and a cell of its operand occurrence `occ q`, at the granularity a block can select (one scalar of
one token's activation, say: a block may hold some of a token's activation without the rest). -/
structure Incidence {d : Declaration} (B : Nci.Covering.MulBlocks d) {κ : Type} (W : Items d.D (Fin B.nw))
    (O : Items d.D κ) where
  /-- The operand occurrence each product reads. -/
  occ : Fin B.nw × Fin B.nt → κ
  slots : SlotsRead B W
  operands : ∀ q ∈ B.P, ∃ c ∈ O.cells (occ q), c ⊆ holds d.D (B.m q)

/-- **Use regions**: for each use `j`, its upstream region `A j`, which holds no registered gate, and the gates `G j ⊆ A j`
a block must compute to serve `j` without importing from `A j`. -/
structure UseRegions (d : Declaration) (nt : ℕ) where
  /-- Each use's upstream region. -/
  A : Fin nt → Finset (Fin d.D.N)
  /-- The gates each use requires. -/
  G : Fin nt → Finset (Fin d.D.N)
  G_sub : ∀ j, G j ⊆ A j
  A_θ : ∀ j, Disjoint (A j) d.θ

/-- **The upstream certificate**, a property of the wiring, checkable on a named circuit: every required gate of a counted
product's use reaches, inside that use's region, a gate the product reads. So a block that computes the product whole and
is handed no gate of the region contains every required gate (`Nci.Guarantees.ImportOrRecompute`). It is the analysed
implementation's; an algebraic rewrite needs its own. -/
def Upstream {d : Declaration} (B : Nci.Covering.MulBlocks d) (Rg : UseRegions d B.nt) : Prop :=
  ∀ p ∈ B.P, ∀ g ∈ Rg.G p.2, ∃ a ∈ d.D.into (B.m p) ∩ Rg.A p.2, Nci.Covering.FeedsWithin d.D (Rg.A p.2) g a

/-- **Overlap weights** on the uses: nonnegative, summing to at most 1 over the regions that hold any one gate. Disjoint
regions take weight 1 each; regions sharing gates take less. -/
def OverlapWeights {d : Declaration} {nt : ℕ} (Rg : UseRegions d nt) (ω : Fin nt → ℝ) : Prop :=
  (∀ j, 0 ≤ ω j) ∧ ∀ x : Fin d.D.N, ∑ j ∈ univ.filter (fun j => x ∈ Rg.A j), ω j ≤ 1

/-- **Recomputing any use needs updated state priced above `h`**: each use's required gates hold, among themselves and
what they read, a selection of `W` priced above `h`. -/
def RequiresState {d : Declaration} {nt : ℕ} (Rg : UseRegions d nt) {ι : Type} (W : Items d.D ι) (h : ℝ) : Prop :=
  ∀ j, ∃ (U : Finset ι) (σ : ι → Finset (Fin d.D.N)), Selects W U σ ∧ (∀ i ∈ U, σ i ⊆ holds d.D (Rg.G j)) ∧
    h < ∑ i ∈ U, W.price i

end Nci.Capability
