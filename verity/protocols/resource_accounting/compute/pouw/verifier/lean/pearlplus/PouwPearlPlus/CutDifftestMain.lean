module

public import PouwPearlPlus.CutDifftest

@[expose] public section

def main : IO Unit := Difftest.serve PouwVerifier.PearlPlus.CutDifftest.table
