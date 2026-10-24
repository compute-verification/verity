import Definitions.Accounting.Remote.Run

/-! The proof-only part of `Definitions.Accounting.Remote.Run`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Accounting.Remote

namespace Run

variable {τ : Type} [Fintype τ] {N : Node τ} {a : Audit} {Ω : Type} (r : Run N a Ω)

variable [Fintype Ω]

/-- **Certified** (PoUW's per-run guarantee at π_r): PoUW's account did at least `W_r` at the rebuild pricing. -/
def Certified (Wr δ : ℝ) : Prop := r.Holds δ fun ω => Wr ≤ r.pouwWork ω

end Run

end Accounting.Remote
