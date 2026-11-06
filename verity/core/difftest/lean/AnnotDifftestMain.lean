module

public import AnnotDifftest

@[expose] public section

def main : IO Unit := Difftest.serve AnnotDifftest.table
