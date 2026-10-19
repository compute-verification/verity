import RemoteAudit.Difftest

def main : IO Unit := Difftest.serve RemoteAudit.Difftest.table
