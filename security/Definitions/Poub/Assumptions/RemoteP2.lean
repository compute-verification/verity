module

public import Definitions.Poub.Assumptions.RateAudit
public import Definitions.Poub.Assumptions.RemoteHeld
import Mathlib.Basic.ENNReal.Real

/-! PoUB over PoUS's P2 store: its codec's work and the squaring ceiling at P2, drafted and not decided, which no
guarantee takes. -/

@[expose] public section

namespace ResourceAccounting.Assumptions

/-- `WorkHardness` at P2's form (drafted, not decided): completing `x` of a P2 block's 16,448 bits costs at least
`x/16,448` of its encode, two square roots each priced `c_root`. PoUS's guarantee prices its responder in sequential
depth, not total work, so it does not give this form. -/
def WorkHardnessP2 (cRoot : ℕ) (cost : ℕ → ℕ) : Prop := WorkHardness 16448 2 cRoot cost

end ResourceAccounting.Assumptions

namespace ResourceAccounting.Remote.Held.Assumptions

variable {Ω : Type}

/-- The squaring ceiling at P2's encode (drafted, not decided): over every interval `[u, v)` the node re-encodes at
most `s / (4 · 32,890) · (v − u) + B` units of a P2 store. No measurement gives `s` at P2's width. -/
def SquaringCeilingP2 (T : Trace Ω) (s B : ℝ) : Prop := SquaringCeiling T s (4 * 32890) B

end ResourceAccounting.Remote.Held.Assumptions
