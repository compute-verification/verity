import Difftest
import MemoryChallenger.WipeJudge

def main : IO Unit := Difftest.serve (Difftest.table [("verdict", MemoryChallenger.StreamedWipe.verdictFn)])
