module

public import Definitions.Pouw.Assumptions.Dimension.H100Floor
import Mathlib.Data.Sym.Sym2.Init
import Mathlib.Tactic.NormNum.GCD
import Mathlib.Tactic.Positivity.Finset
import Proofs.Pouw.Dimension.H100LoopFreeProofs
import Proofs.Pouw.Dimension.H100WProofs

/-! `MH100`'s floor table satisfies `H32`. -/

@[expose] public section

namespace Pouw.Dimension.H100

theorem h32Floor : H32 h100Floor :=
  ⟨fun s => by cases s <;> decide, by decide⟩

theorem distinctWritesLoopFree : DistinctWritesLoopFree :=
  fun G F hF S fpSem L P T => distinctWritesMH100On h100LoopFree h32LoopFree G F hF S fpSem L P T

theorem distinctWritesFloor : DistinctWritesFloor :=
  fun G F hF S fpSem L P T => distinctWritesMH100On h100Floor h32Floor G F hF S fpSem L P T

end Pouw.Dimension.H100
