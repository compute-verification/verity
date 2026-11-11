module

public import CoreProtocol.PeriodDifftest

@[expose] public section

def main : IO Unit := Difftest.serve Core.PeriodDifftest.table
