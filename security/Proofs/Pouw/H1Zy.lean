module

public import Proofs.Pouw.H1Zy.ChainVectors
public import Proofs.Pouw.H1Zy.Combine
public import Proofs.Pouw.H1Zy.CombineBits
public import Proofs.Pouw.H1Zy.DequantVectors
public import Proofs.Pouw.H1Zy.Lane
public import Proofs.Pouw.H1Zy.LaneBits
public import Definitions.Pouw.H1Zy.Stream
public import Proofs.Pouw.H1Zy.Tile
public import Proofs.Pouw.H1Zy.LaneChunk
import Mathlib.Data.Rat.Cast.Order
import Mathlib.Tactic.NormNum.Abs
import Mathlib.Tactic.NormNum.DivMod
import Mathlib.Tactic.NormNum.OfScientific
import Mathlib.Tactic.NormNum.Pow

/-! H1's drawn-leaf statements: their Definitions on words mean what `LeafChecks` reads; no theorem relates
`Stream`'s `ok` over `PouwH1Tile_v3` to its `ok` over `PouwH1Tile_v1`. -/

@[expose] public section
