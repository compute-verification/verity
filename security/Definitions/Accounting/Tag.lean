module

public import Definitions.Core.Prob
public import Definitions.Accounting.Tag.PolyHash
public import Definitions.Accounting.Tag.Sketch

/-! `Definitions.Accounting.Tag`: block tags (trusted): the polynomial hash of the bandwidth audit's uncertified fast
check (`gf31-poly/v1`) and exact ±1 sketches. Neither certifies bandwidth. -/

@[expose] public section
