import Models.Accounting.Assumptions.Isa
import Models.Accounting.Assumptions.Sheet

/-!
# A6 and A7 from an instruction table (trusted)

Statements over the instruction table's premises (`Models.Accounting.Assumptions.Isa`,
note:20261019T0520Z-draft-mm-isa), each `def G : Prop`, proved as `Accounting.SecurityProofs.Isa.G` in
`Proofs.Accounting.Isa`:

* `IsaSheet`: on any device, a table whose rows pass the discipline (`Accounting.Isa.Row.ok`) and a model that reads
  it give A6 (`SheetWords`) and A7 (`SheetSemantics`) from the class list's completeness, the labelling and the rows.
* `Sm120Sheet`: the same on sm_120's table (`Accounting.Isa.Sm120.row`), whose discipline is decided and whose
  absent families (wgmma, tcgen05, tensor memory) need no premise.
* `Sm90Sheet`, `Sm100Sheet`: the same on sm_90a's table (no tcgen05 or tensor memory) and sm_100a's (no wgmma).
-/

namespace Accounting.Guarantees.Isa

open Accounting.TS Accounting.Isa Accounting.Assumptions.TS Accounting.Assumptions.Isa

/-- **A6 and A7 from an instruction table.** On a labelled run the labelling splits, a table `Tb` whose every row
passes the discipline and a model `Z` that reads it (side unit `u` being class `σ u`'s) give `SheetWords` and
`SheetSemantics` (NoSum and SideUnits), from the class list's completeness, the labelling's reading of the table and
each family's rows. -/
def IsaSheet : Prop :=
  ∀ {K R T L U F X : Type} [Fintype K] [DecidableEq K] [Fintype T] [Fintype L] [Fintype U]
    (Tb : K → Row) (σ : U → K) (Z : Model K R T L U F X) (A : Anatomy K T L U) (H : Heads K T L U),
    (∀ k, (Tb k).ok = true) → ReadsTable Tb σ Z → A.Split → IsaComplete A H → IsaLabelled Tb σ Z A H →
    IsaRowsMma Tb A H → IsaRowsWgmma Tb A H → IsaRowsTcgen05 Tb A H → IsaRowsTmem Tb A H → IsaRowsLane Tb A H →
    IsaRowsUniform Tb A H → IsaRowsMemory Tb A H → IsaRowsBulk Tb A H → IsaRowsTexture Tb A H →
    IsaRowsWarp Tb A H → SheetWords Z A ∧ SheetSemantics Z A

/-- **A6 and A7 on sm_120.** As `IsaSheet` for sm_120's table: a model over sm_120's classes that reads it gives
`SheetWords` and `SheetSemantics` from the class list's completeness, the labelling and the seven families sm_120
has. -/
def Sm120Sheet : Prop :=
  ∀ {R T L U F X : Type} [Fintype T] [Fintype L] [Fintype U]
    (σ : U → Sm120) (Z : Model Sm120 R T L U F X) (A : Anatomy Sm120 T L U) (H : Heads Sm120 T L U),
    ReadsTable Sm120.row σ Z → A.Split → IsaComplete A H → IsaLabelled Sm120.row σ Z A H →
    IsaRowsMma Sm120.row A H → IsaRowsLane Sm120.row A H → IsaRowsUniform Sm120.row A H →
    IsaRowsMemory Sm120.row A H → IsaRowsBulk Sm120.row A H → IsaRowsTexture Sm120.row A H →
    IsaRowsWarp Sm120.row A H → SheetWords Z A ∧ SheetSemantics Z A

/-- **A6 and A7 on sm_90a.** As `Sm120Sheet` for sm_90a's table (`Accounting.Isa.Sm90.row`), with its wgmma rows and
no tcgen05 or tensor-memory premise. -/
def Sm90Sheet : Prop :=
  ∀ {R T L U F X : Type} [Fintype T] [Fintype L] [Fintype U]
    (σ : U → Sm90) (Z : Model Sm90 R T L U F X) (A : Anatomy Sm90 T L U) (H : Heads Sm90 T L U),
    ReadsTable Sm90.row σ Z → A.Split → IsaComplete A H → IsaLabelled Sm90.row σ Z A H →
    IsaRowsMma Sm90.row A H → IsaRowsWgmma Sm90.row A H → IsaRowsLane Sm90.row A H →
    IsaRowsUniform Sm90.row A H → IsaRowsMemory Sm90.row A H → IsaRowsBulk Sm90.row A H →
    IsaRowsTexture Sm90.row A H → IsaRowsWarp Sm90.row A H → SheetWords Z A ∧ SheetSemantics Z A

/-- **A6 and A7 on sm_100a.** As `Sm120Sheet` for sm_100a's table (`Accounting.Isa.Sm100.row`), with its tcgen05.mma
and tensor-memory rows and no wgmma premise. -/
def Sm100Sheet : Prop :=
  ∀ {R T L U F X : Type} [Fintype T] [Fintype L] [Fintype U]
    (σ : U → Sm100) (Z : Model Sm100 R T L U F X) (A : Anatomy Sm100 T L U) (H : Heads Sm100 T L U),
    ReadsTable Sm100.row σ Z → A.Split → IsaComplete A H → IsaLabelled Sm100.row σ Z A H →
    IsaRowsMma Sm100.row A H → IsaRowsTcgen05 Sm100.row A H → IsaRowsTmem Sm100.row A H →
    IsaRowsLane Sm100.row A H → IsaRowsUniform Sm100.row A H → IsaRowsMemory Sm100.row A H →
    IsaRowsBulk Sm100.row A H → IsaRowsTexture Sm100.row A H → IsaRowsWarp Sm100.row A H →
    SheetWords Z A ∧ SheetSemantics Z A

end Accounting.Guarantees.Isa
