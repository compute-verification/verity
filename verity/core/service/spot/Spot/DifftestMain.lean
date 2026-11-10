module

public import Difftest
public import Spot.Pieces

@[expose] public section

def main : IO Unit := Difftest.serve (Difftest.table Spot.Pieces.table)
