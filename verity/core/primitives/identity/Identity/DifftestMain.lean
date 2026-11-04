module

public import Identity.Difftest

@[expose] public section

def main : IO Unit := Difftest.serve Identity.Difftest.table
