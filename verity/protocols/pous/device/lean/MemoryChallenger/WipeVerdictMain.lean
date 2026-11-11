module

public import Difftest
public import MemoryChallenger.WipeJudge

@[expose] public section

def main : IO Unit := Difftest.serve (Difftest.table [("verdict", MemoryChallenger.StreamedWipe.verdictFn)])
