module

public import MemoryChallenger.Difftest

@[expose] public section

def main : IO Unit := Difftest.serve MemoryChallenger.Difftest.table
