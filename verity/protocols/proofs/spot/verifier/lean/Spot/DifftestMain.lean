module

public import Difftest
public import Spot.Pieces
public import Spot.SecondWires
public import Spot.CertifiedRows

@[expose] public section

def main : IO Unit :=
  Difftest.serve (Difftest.table (Spot.Pieces.table ++
    [("second_wires", Spot.SecondWires.piece), ("certified_rows", Spot.CertifiedRows.piece)]))
