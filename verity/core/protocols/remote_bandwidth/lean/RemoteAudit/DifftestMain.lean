module

public import RemoteAudit.Difftest

@[expose] public section

def main : IO Unit := Difftest.serve RemoteAudit.Difftest.table
