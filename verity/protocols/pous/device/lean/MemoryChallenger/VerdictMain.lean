module

public import Difftest
public import MemoryChallenger.Judge

@[expose] public section

def main : IO Unit := Difftest.serve (Difftest.table [("verdict", MemoryChallenger.Judge.verdictFn)])
