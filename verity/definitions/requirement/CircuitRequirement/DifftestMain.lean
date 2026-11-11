module

public import CircuitRequirement.Difftest

@[expose] public section

def main : IO Unit := Difftest.serve CircuitRequirement.Difftest.table
