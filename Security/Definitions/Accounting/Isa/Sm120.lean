import Definitions.Accounting.Isa.Classes

/-!
# sm_120's instruction table (trusted)

The RTX PRO 6000's (sm_120) instruction classes (note:20261019T0520Z-draft-mm-isa): the mma.sync forms and the classes
outside the tensor cores (`Isa/Classes.lean`), 126 in all: DS-120's 122 census classes, `BMMA168256`, `IMMAS4`, `REDUX`
and `VSIMD`. sm_120 has no wgmma, tcgen05 or tensor memory.
-/

namespace Accounting.Isa

/-- **sm_120's instruction classes**: an mma.sync form, or a class outside the tensor cores. -/
inductive Sm120
  | sync (s : Sync)
  | base (c : Base)
  deriving DecidableEq, Fintype, Repr

/-- **sm_120's rows**: the shared rows (`Sync.row`, `Base.row`). -/
def Sm120.row : Sm120 → Row
  | .sync s => s.row
  | .base c => c.row

end Accounting.Isa
