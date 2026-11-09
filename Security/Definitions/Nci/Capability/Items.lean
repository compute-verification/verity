module

public import Definitions.Core.Formalism

/-!
# Local bounds: items, what a block holds, and budgets

Information is about **items**: things a block may hold. An item has eligible **cells** (the gates holding one version
of its value) and a **price** in bits. A **selection** picks some items and one cell each. A block **holds** a selection
when its cells lie among the block's gates and what they read. Every information premise reaches the counting as a
**budget**: what each block holds costs at most `h` (`HeldPriceAtMost`).
-/

@[expose] public section

namespace Nci.Capability

open Verity Finset

/-- **Items** of the circuit `C`, indexed by `ι`: things whose values a block may hold. Item `i` has eligible **cells**
`cells i`, each the gates holding one version of its value (a parameter slot has one version per optimizer step that
writes it, each a multi-bit value; an activation occurrence has one), and a **price** `price i ≥ 0`, the bits its value
costs. -/
structure Items (C : Circuit) (ι : Type) where
  /-- The eligible cells of each item. -/
  cells : ι → Finset (Finset (Fin C.N))
  /-- Each item's price, in bits. -/
  price : ι → ℝ
  price_nonneg : ∀ i, 0 ≤ price i

/-- **What the gates `S` hold**: their own gates and the gates they read, `S ∪ in(S)`. -/
def holds (C : Circuit) (S : Finset (Fin C.N)) : Finset (Fin C.N) := S ∪ C.into S

/-- **A selection** of the items `U` of `X`, with an eligible cell `σ i` for each. -/
def Selects {C : Circuit} {ι : Type} (X : Items C ι) (U : Finset ι) (σ : ι → Finset (Fin C.N)) : Prop :=
  ∀ i ∈ U, σ i ∈ X.cells i

/-- **The values of a selection** in the run `T`: its cells' gates, in gate order, as bits. -/
def selectionBits {C : Circuit} {ι : Type} (T : C.Trace) (U : Finset ι) (σ : ι → Finset (Fin C.N)) :
    List Bool :=
  C.bits T ((U.biUnion σ).sort (· ≤ ·))

/-- **Each proof block holds `X`-state priced at most `h`**: every selection whose cells lie among what a proof block of
the committed declaration `R` holds costs at most `h`. Every information bound reaches the counting in this form. -/
def HeldPriceAtMost (R : Committed) {ι : Type} (X : Items R.d.D ι) (h : ℝ) : Prop :=
  ∀ (u : Fin R.d.n) (U : Finset ι) (σ : ι → Finset (Fin R.d.D.N)), Selects X U σ →
    (∀ i ∈ U, σ i ⊆ holds R.d.D (R.d.𝓘.gates u)) → ∑ i ∈ U, X.price i ≤ h

end Nci.Capability
