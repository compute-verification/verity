module

public import Difftest
public import Spot.SiteDifftest

@[expose] public section

def main : IO Unit := Difftest.serve (Difftest.table Spot.SiteDifftest.table)
