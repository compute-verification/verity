import Difftest
import MemoryChallenger.Judge

def main : IO Unit := Difftest.serve (Difftest.table [("verdict", MemoryChallenger.Judge.verdictFn)])
