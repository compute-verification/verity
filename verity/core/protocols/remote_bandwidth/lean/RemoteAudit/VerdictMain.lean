import Difftest
import RemoteAudit.PostSeed

def main : IO Unit := Difftest.serve (Difftest.table RemoteAudit.PostSeed.fns)
