module

public import Difftest
public import Spot.Period
public import Spot.Pieces
public import Spot.SecondWires

@[expose] public section

def main : IO Unit :=
  Difftest.serve (Difftest.table (Spot.Pieces.table ++
    [("second_wires", Spot.SecondWires.piece), ("period_windows", Spot.Period.windowsPiece)]))
