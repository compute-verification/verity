module

public import Definitions.Flock.Game
public import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Algebra.Order.BigOperators.Expect
import Mathlib.Algebra.Order.Field.Power
import Mathlib.Analysis.Complex.Order
import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Data.EReal.Inv
import Mathlib.Tactic.Positivity.Finset

/-! Extraction by filling: the extractor's table of the first opening its `K` reruns make at each position
(`firstTable`). -/

@[expose] public section

namespace FlockSoundness

namespace Extract

open Game Rewinding
variable {β P V : Type} [Fintype P] [DecidableEq V]

/-- **The extractor's table** from runs `bs`: at each position, the first opening a run made there;
`z₀` where none did. -/
def firstTable (op : β → P → Option V) (z₀ : P → V) (bs : List β) (p : P) : V :=
  (bs.findSome? fun b => op b p).getD (z₀ p)

end Extract

end FlockSoundness
