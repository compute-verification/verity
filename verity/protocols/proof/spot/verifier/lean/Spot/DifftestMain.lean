module

public import Difftest
public import Spot.Pieces
public import Spot.SecondWires
public import Spot.BoundaryWritten
public import Spot.SecondWindow
public import Spot.BlockFront
public import Spot.BlockFile
public import Spot.Restrict
public import Spot.Supplied
public import Spot.HeaderHeldAt

@[expose] public section

def main : IO Unit :=
  Difftest.serve (Difftest.table (Spot.Pieces.table ++ [("second_wires", Spot.SecondWires.piece),
    ("boundary_written", Spot.BoundaryWritten.piece), ("boundary_written_unrolled", Spot.BoundaryWritten.pieceRef),
    ("second_window", Spot.SecondWindow.piece),
    ("block_front", Spot.BlockFront.piece), ("block_file", Spot.BlockFile.piece), ("restrict", Spot.Restrict.piece),
    ("supplied_marks", Spot.Supplied.piece), ("header_held_at", Spot.HeaderHeldAt.piece)]))
