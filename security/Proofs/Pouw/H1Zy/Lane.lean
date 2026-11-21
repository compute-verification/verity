module

public import Definitions.Pouw.H1Zy.Lane
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Rat.Cast.Order
import Mathlib.Tactic.Linarith.Frontend
import Mathlib.Tactic.NormNum.Abs
import Mathlib.Tactic.NormNum.DivMod
import Mathlib.Tactic.NormNum.OfScientific
import Mathlib.Tactic.Ring.RingNF

/-!
# What a lane's `ok = 1` gives, on words

`Pouw.H1Zy.laneOk`, `PouwH1LaneZy_v1{PouwH1Tile_v1}`'s body on words at a 32 × 32 leaf, is `1` only where (`laneOk_spec`):

* the index's 11 high bits are `0`, so the lane `x.l` its low 5 bits name is its index;
* each of its 32 cells `(laneRow l s, laneCol l t)` of the record has z the exact dot of its A row's and B row's codes,
  as an s32 word, and y the dequant of that z at the rows' scales: the `PouwMuxBit` tree picks lane `l`'s cell
  (`pick_ofFn`), and the EXACT dot's `BlackwellS8ImmaDot32` steps never clamp at a `K` its bind takes (`gemmAccS8_eq`:
  `32 ∣ K` and `K < 2^17`, every partial sum within `K · 2^14 < 2^31`);
* its partials row is its chain's words masked by the admit bits' OR.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Pouw.H1Zy

/-! ### Trees and picks -/

/-! ### The cell check -/

/-! ### The EXACT dot -/

/-- Product `k` of the rows' codes, `0` past their end. -/
private def prodAt {K : ℕ} (x w : Fin K → BitVec 8) (k : ℕ) : ℤ := (codeAt x k).toInt * (codeAt w k).toInt

/-! ### The lane -/

end Pouw.H1Zy
