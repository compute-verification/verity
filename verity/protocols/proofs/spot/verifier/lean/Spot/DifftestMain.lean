module

public import Difftest
public import Spot.Pieces
public import Spot.Cut
public import Spot.SecondWires
public import Spot.CertifiedRows
public import Spot.BoundaryWritten
public import Spot.SecondWindow

@[expose] public section

def main : IO Unit :=
  Difftest.serve (Difftest.table (Spot.Pieces.table ++ [("second_wires", Spot.SecondWires.piece),
    ("cut", Spot.Cut.piece), ("cut_why", Spot.Cut.pieceWhy), ("boundary_written", Spot.BoundaryWritten.piece),
    ("boundary_written_unrolled", Spot.BoundaryWritten.pieceRef),
    ("certified_rows", Spot.CertifiedRows.piece), ("second_window", Spot.SecondWindow.piece)]))
