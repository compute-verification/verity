module

public import Difftest
public import Spot.Pieces
public import Spot.SecondWires
public import Spot.CertifiedRows
public import Spot.BoundaryWritten
public import Spot.SecondWindow
public import Spot.BlockFront
public import Spot.BlockFile
public import Spot.Restrict

@[expose] public section

def main : IO Unit :=
  Difftest.serve (Difftest.table (Spot.Pieces.table ++ [("second_wires", Spot.SecondWires.piece),
    ("boundary_written", Spot.BoundaryWritten.piece), ("boundary_written_unrolled", Spot.BoundaryWritten.pieceRef),
    ("certified_rows", Spot.CertifiedRows.piece), ("second_window", Spot.SecondWindow.piece),
    ("block_front", Spot.BlockFront.piece), ("block_program", Spot.BlockFront.programPiece),
    ("definition", Spot.BlockFront.definitionPiece), ("block_file", Spot.BlockFile.piece),
    ("restrict", Spot.Restrict.piece)]))
