module

public import Difftest
public import SampledProofs.Pieces

@[expose] public section

def main : IO Unit := Difftest.serve (Difftest.table SampledProofs.Pieces.table)
