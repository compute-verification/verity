module

public import Definitions.Pouw.H1Zy.LaneBits
public import Definitions.Pouw.H1Zy.Stream
public import Proofs.Pouw.H1Zy.LaneChunk
import Mathlib.Data.Rat.Cast.Order
import Mathlib.Tactic.NormNum.Abs
import Mathlib.Tactic.NormNum.DivMod
import Mathlib.Tactic.NormNum.OfScientific
import Mathlib.Tactic.NormNum.Pow

/-!
# H1's drawn-leaf statements on words: the proofs

`LaneChunk`: a lane's chunks in sequence are the whole lane (`Pouw.H1Zy.run_join`, `Pouw.H1Zy.zRun_join`).
-/

@[expose] public section
